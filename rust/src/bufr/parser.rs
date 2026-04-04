/// Top-level Argentine BUFR file parser.
///
/// Handles gzip decompression of .BUFR.gz files, section parsing,
/// and sweep data extraction.

use super::compression::decompress_gzip;
use super::errors::BufrError;
use super::sections::{find_section4, parse_section0, parse_section1, Section1};
use super::sweep::{decode_sweeps, SweepData};

pub struct BufrFile {
    pub section1: Section1,
    pub sweeps: Vec<SweepData>,
    /// Station identifier (e.g. "RMA1")
    pub station_id: String,
    /// Station type (e.g. "RMA")
    pub station_type: String,
    pub latitude: f64,
    pub longitude: f64,
    pub height_m: f64,
}

impl BufrFile {
    /// Parse a BUFR file from raw bytes (may be gzip-compressed).
    pub fn parse(raw: &[u8]) -> Result<Self, BufrError> {
        // Detect gzip by magic bytes 0x1f 0x8b
        let data = if raw.len() >= 2 && raw[0] == 0x1f && raw[1] == 0x8b {
            decompress_gzip(raw)?
        } else {
            raw.to_vec()
        };

        parse_section0(&data)?;
        let s1 = parse_section1(&data)?;
        let offsets = find_section4(&data)?;

        let section4 = &data[offsets.section4_offset
            ..offsets.section4_offset + offsets.section4_length];

        // Parse station ID and location from section 4, then sweeps
        let (station_type, station_id, latitude, longitude, height_m, sweeps) =
            parse_section4_full(section4)?;

        Ok(BufrFile {
            section1: s1,
            sweeps,
            station_id,
            station_type,
            latitude,
            longitude,
            height_m,
        })
    }
}

/// Parse Section 4 to extract station info and all sweeps.
fn parse_section4_full(
    section4: &[u8],
) -> Result<(String, String, f64, f64, f64, Vec<SweepData>), BufrError> {
    use super::bit_reader::BitReader;
    use super::descriptors::*;

    let mut reader = BitReader::new(section4, 0);

    // 3;21;204 — Station identifiers
    let n_stations = reader.read_bits(8)? as usize;
    let mut station_type = String::new();
    let mut station_id = String::new();
    for _ in 0..n_stations {
        station_type = reader.read_ccitt(24)?;   // 0;01;192 — 24-bit (3 chars)
        station_id = reader.read_ccitt(128)?;    // 0;01;193 — 128-bit (16 chars)
    }

    // 3;01;031 — Location/Time header
    reader.read_bits(7)?;   // 0;01;001 WMO block
    reader.read_bits(10)?;  // 0;01;002 WMO station
    reader.read_bits(2)?;   // 0;02;001 station type
    reader.read_bits(12)?;  // year
    reader.read_bits(4)?;   // month
    reader.read_bits(6)?;   // day
    reader.read_bits(5)?;   // hour
    reader.read_bits(6)?;   // minute
    let lat_raw = reader.read_bits(25)?;
    let lon_raw = reader.read_bits(26)?;
    let hgt_raw = reader.read_bits(15)?;

    let latitude = decode_latitude(lat_raw);
    let longitude = decode_longitude(lon_raw);
    let height_m = decode_height(hgt_raw);

    // Now read the sweep data starting from current position
    let byte_pos = reader.byte_offset;
    let bit_pos = reader.bit_offset;

    // We need to pass the remaining bytes + current bit position to decode_sweeps.
    // Since decode_sweeps expects to start from byte 0 of its slice at a specific
    // sub-byte offset, we create a sub-slice and adjust.
    // The easiest approach: pass the full section4 and let decode_sweeps use a
    // separate reader starting at the current position.
    let sweeps = decode_sweeps_from(section4, byte_pos, bit_pos)?;

    Ok((station_type, station_id, latitude, longitude, height_m, sweeps))
}

/// Decode sweeps starting at a specific byte/bit offset in the section4 data.
fn decode_sweeps_from(
    section4: &[u8],
    byte_offset: usize,
    bit_offset: u8,
) -> Result<Vec<SweepData>, BufrError> {
    use super::bit_reader::BitReader;
    use super::compression::decompress_to_f64;
    use super::descriptors::*;
    use std::collections::HashMap;

    let mut reader = BitReader::new(section4, byte_offset);
    reader.bit_offset = bit_offset;

    // 3;21;203 — ODIM Polar Volume
    let n_sweeps = reader.read_bits(8)? as usize;
    let mut sweeps = Vec::with_capacity(n_sweeps);

    for _ in 0..n_sweeps {
        // 3;21;205 — Two timestamps
        let start_time = read_timestamp_from(&mut reader)?;
        let end_time = read_timestamp_from(&mut reader)?;

        // 0;30;196 — sweep-level product type
        reader.read_bits(8)?;

        // Elevation (15-bit, scale=2, ref=-9000)
        let elev_raw = reader.read_bits(15)?;
        let elevation_deg = decode_elevation_angle(elev_raw);

        // n_bins (12-bit)
        let n_bins = reader.read_bits(12)? as usize;

        // bin_size_m (14-bit)
        let bin_size_raw = reader.read_bits(14)?;
        let bin_size_m = decode_bin_size(bin_size_raw);

        // bin_offset_m (14-bit, scale=-1)
        let bin_offset_raw = reader.read_bits(14)?;
        let bin_offset_m = decode_bin_offset(bin_offset_raw);

        // n_azimuths (11-bit)
        let n_azimuths = reader.read_bits(11)? as usize;

        // start_azimuth (16-bit, scale=2)
        let az_raw = reader.read_bits(16)?;
        let start_azimuth_deg = decode_azimuth(az_raw);

        // NOTE: 0;02;193 (rotation, 2-bit) is NOT in the 3;21;203 expansion per lrose BufrTables.cc
        let rotation_clockwise = true;

        let geometry = super::sweep::SweepGeometry {
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

        // Parameters
        let n_params = reader.read_bits(8)? as usize;
        let mut moments: HashMap<String, Vec<f64>> = HashMap::new();

        for _ in 0..n_params {
            let product_code = reader.read_bits(8)? as u8;
            let moment_name = product_code_to_cfradial(product_code)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("UNKNOWN_{}", product_code));

            // Compression method
            reader.read_bits(8)?;

            // Accumulate compressed bytes
            let n_outer = reader.read_bits(16)? as usize;
            let mut compressed: Vec<u8> = Vec::new();
            for _ in 0..n_outer {
                let n_inner = reader.read_bits(16)? as usize;
                for _ in 0..n_inner {
                    compressed.push(reader.read_bits(8)? as u8);
                }
            }

            let expected = n_azimuths * n_bins;
            match decompress_to_f64(&compressed, expected) {
                Ok(data) => { moments.insert(moment_name, data); }
                Err(e) => {
                    eprintln!("Warning: failed to decompress moment {}: {}", moment_name, e);
                }
            }
        }

        sweeps.push(SweepData { geometry, moments });
    }

    Ok(sweeps)
}

fn read_timestamp_from(
    reader: &mut super::bit_reader::BitReader,
) -> Result<String, BufrError> {
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
