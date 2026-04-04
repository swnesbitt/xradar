/// Decode the 3;21;203 ODIM Polar Volume sweep loop from Section 4 data.
///
/// Structure per sweep:
///   - Start/end timestamps (2 × year/month/day/hour/min/sec)
///   - Product type (8-bit code table)
///   - Elevation angle (15-bit, scale=2, ref=-9000)
///   - n_bins (12-bit)
///   - bin_size_m (14-bit, meters)
///   - bin_offset_m (14-bit, scale=-1 → ×10)
///   - n_azimuths (11-bit)
///   - start_azimuth (16-bit, scale=2)
///   - rotation_direction (2-bit)
///   - Per-parameter delayed replication:
///       product_type (8-bit) + 3;21;206 compressed data

use std::collections::HashMap;

use super::bit_reader::BitReader;
use super::compression::decompress_to_f64;
use super::descriptors::*;
use super::errors::BufrError;

#[derive(Debug, Clone)]
pub struct SweepGeometry {
    pub elevation_deg: f64,
    pub n_bins: usize,
    pub bin_size_m: f64,
    pub bin_offset_m: f64, // first bin center distance
    pub n_azimuths: usize,
    pub start_azimuth_deg: f64,
    pub rotation_clockwise: bool,
    pub start_time: String, // ISO8601
    pub end_time: String,
}

#[derive(Debug)]
pub struct SweepData {
    pub geometry: SweepGeometry,
    /// Moment name → 2D data (n_azimuths × n_bins), physical units
    pub moments: HashMap<String, Vec<f64>>,
}

/// Decode all sweeps from Section 4 bytes.
pub fn decode_sweeps(section4: &[u8]) -> Result<Vec<SweepData>, BufrError> {
    let mut reader = BitReader::new(section4, 0);

    // 3;21;204 — Station identifiers
    // 1;02;000 (delayed rep of 2 descriptors)
    let n_stations = reader.read_bits(8)? as usize; // 0;31;001
    let mut _station_type = String::new();
    let mut _station_id = String::new();
    for _ in 0..n_stations {
        _station_type = reader.read_ccitt(24)?;  // 0;01;192 — 24-bit CCITT
        _station_id = reader.read_ccitt(128)?;   // 0;01;193 — 128-bit CCITT
    }

    // 3;01;031 — Location/Time header
    reader.read_bits(7)?;   // 0;01;001 WMO block (7-bit)
    reader.read_bits(10)?;  // 0;01;002 WMO station (10-bit)
    reader.read_bits(2)?;   // 0;02;001 station type (2-bit)
    // Date: year(12) month(4) day(6) hour(5) minute(6)
    reader.read_bits(12)?;  // year
    reader.read_bits(4)?;   // month
    reader.read_bits(6)?;   // day
    reader.read_bits(5)?;   // hour
    reader.read_bits(6)?;   // minute
    // Lat(25) Lon(26) Height(15)
    reader.read_bits(25)?;  // lat
    reader.read_bits(26)?;  // lon
    reader.read_bits(15)?;  // height

    // 3;21;203 — ODIM Polar Volume
    // 1;12;000 (delayed replication of 12 descriptors)
    let n_sweeps = reader.read_bits(8)? as usize; // 0;31;001 — sweep count

    let mut sweeps = Vec::with_capacity(n_sweeps);

    for _ in 0..n_sweeps {
        // 3;21;205 — Two timestamps (start + end)
        // = 1;02;002 × (3;01;011 + 3;01;013)
        // = 2 × (year/month/day/hour/min/sec)
        let start_time = read_timestamp(&mut reader)?;
        let end_time = read_timestamp(&mut reader)?;

        // 0;30;196 — Top-level product type (sweep-level, often same as first param)
        reader.read_bits(8)?;

        // 0;02;135 — Elevation angle (15-bit, scale=2, ref=-9000)
        let elev_raw = reader.read_bits(15)?;
        let elevation_deg = decode_elevation_angle(elev_raw);

        // 0;30;194 — Number of bins (12-bit)
        let n_bins = reader.read_bits(12)? as usize;

        // 0;21;201 — Range bin size (14-bit, meters)
        let bin_size_raw = reader.read_bits(14)?;
        let bin_size_m = decode_bin_size(bin_size_raw);

        // 0;21;203 — Range bin offset (14-bit, scale=-1 → ×10 meters)
        let bin_offset_raw = reader.read_bits(14)?;
        let bin_offset_m = decode_bin_offset(bin_offset_raw);

        // 0;30;195 — Number of azimuths (11-bit)
        let n_azimuths = reader.read_bits(11)? as usize;

        // 0;02;134 — Start azimuth (16-bit, scale=2)
        let az_raw = reader.read_bits(16)?;
        let start_azimuth_deg = decode_azimuth(az_raw);

        // NOTE: 0;02;193 (rotation, 2-bit) is NOT in the 3;21;203 expansion per lrose BufrTables.cc
        let rotation_clockwise = true;

        let geometry = SweepGeometry {
            elevation_deg,
            n_bins,
            bin_size_m,
            bin_offset_m,
            n_azimuths,
            start_azimuth_deg,
            rotation_clockwise,
            start_time,
            end_time,
        };

        // 1;02;000 (delayed rep of 2 descriptors) — parameters
        let n_params = reader.read_bits(8)? as usize; // 0;31;001

        let mut moments: HashMap<String, Vec<f64>> = HashMap::new();

        for _ in 0..n_params {
            // 0;30;196 — product type code for this parameter
            let product_code = reader.read_bits(8)? as u8;
            let moment_name = product_code_to_cfradial(product_code)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("UNKNOWN_{}", product_code));

            // 3;21;206 — compressed data array
            // 0;30;197 — compression method (8-bit, should be 0 = zlib)
            let _compression_method = reader.read_bits(8)?;

            // Accumulate bytes from nested delayed replication
            // 1;03;000 (delayed rep of 3 descriptors)
            let n_outer_chunks = reader.read_bits(16)? as usize; // 0;31;002

            let mut compressed_bytes: Vec<u8> = Vec::new();
            for _ in 0..n_outer_chunks {
                // 1;01;000 (delayed rep of 1 descriptor)
                let n_inner_bytes = reader.read_bits(16)? as usize; // 0;31;002
                for _ in 0..n_inner_bytes {
                    let byte = reader.read_bits(8)? as u8; // 0;30;198
                    compressed_bytes.push(byte);
                }
            }

            let expected = n_azimuths * n_bins;
            match decompress_to_f64(&compressed_bytes, expected) {
                Ok(data) => { moments.insert(moment_name, data); }
                Err(e) => {
                    // Log and skip bad moment rather than failing whole sweep
                    eprintln!("Warning: failed to decompress moment {}: {}", moment_name, e);
                }
            }
        }

        sweeps.push(SweepData { geometry, moments });
    }

    Ok(sweeps)
}

/// Read a timestamp: year(12) month(4) day(6) hour(5) min(6) sec(6) = 39 bits
fn read_timestamp(reader: &mut BitReader) -> Result<String, BufrError> {
    let year = reader.read_bits(12)?;
    let month = reader.read_bits(4)?;
    let day = reader.read_bits(6)?;
    let hour = reader.read_bits(5)?;
    let minute = reader.read_bits(6)?;
    let second = reader.read_bits(6)?;
    Ok(format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, second
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_elevation_known() {
        // 0.51 degrees → raw = 0.51 * 100 + 9000 = 9051
        let angle = decode_elevation_angle(9051);
        assert!((angle - 0.51).abs() < 1e-6);
    }

    #[test]
    fn test_decode_azimuth_known() {
        // 13.00 degrees → raw = 1300
        let az = decode_azimuth(1300);
        assert!((az - 13.0).abs() < 1e-6);
    }

    #[test]
    fn test_decode_bin_offset_known() {
        // 2100 meters → raw = 210 (scale -1 means ×10)
        let offset = decode_bin_offset(210);
        assert!((offset - 2100.0).abs() < 1e-9);
    }
}
