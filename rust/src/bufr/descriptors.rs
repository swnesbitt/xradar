/// Argentine BUFR local table B/D expansion (WMO centre 41, local version 2).
///
/// This module hardcodes the descriptor sequences needed to decode Argentine RMA
/// radar BUFR files. No runtime table lookup is required.
///
/// Descriptor notation: F;X;Y where F=0 (element), 1 (replication), 2 (operator), 3 (sequence)

/// Product type code → CFRadial moment name
pub fn product_code_to_cfradial(code: u8) -> Option<&'static str> {
    match code {
        0 => Some("DBZH"),
        40 => Some("VRADH"),
        60 | 92 => Some("WRADH"),
        80 => Some("ZDR"),
        242 => Some("TDR"),
        91 | 230 => Some("DBTH"),
        231 => Some("DBTV"),
        232 => Some("DBZV"),
        233 => Some("DBZV"),
        239 => Some("PHIDP"),
        240 => Some("KDP"),
        241 => Some("RHOHV"),
        243 => Some("CM"),
        _ => None,
    }
}

/// Descriptor 0;02;135 — Antenna elevation angle
/// 15-bit, scale=2, ref=-9000 → physical = (raw - 9000) / 100.0 degrees
pub fn decode_elevation_angle(raw: u64) -> f64 {
    (raw as i64 - 9000) as f64 / 100.0
}

/// Descriptor 0;02;134 — Antenna beam azimuth
/// 16-bit, scale=2, ref=0 → physical = raw / 100.0 degrees
pub fn decode_azimuth(raw: u64) -> f64 {
    raw as f64 / 100.0
}

/// Descriptor 0;05;001 — Latitude (high accuracy)
/// 25-bit, scale=5, ref=-9000000 → physical = (raw - 9000000) / 100000.0 degrees
pub fn decode_latitude(raw: u64) -> f64 {
    (raw as i64 - 9_000_000) as f64 / 100_000.0
}

/// Descriptor 0;06;001 — Longitude (high accuracy)
/// 26-bit, scale=5, ref=-18000000 → physical = (raw - 18000000) / 100000.0 degrees
pub fn decode_longitude(raw: u64) -> f64 {
    (raw as i64 - 18_000_000) as f64 / 100_000.0
}

/// Descriptor 0;07;001 — Height of station above MSL
/// 15-bit, scale=0, ref=-400 → physical = raw - 400 meters
pub fn decode_height(raw: u64) -> f64 {
    (raw as i64 - 400) as f64
}

/// Descriptor 0;21;203 — Range bin offset
/// 14-bit, scale=-1, ref=0 → physical = raw * 10 meters
pub fn decode_bin_offset(raw: u64) -> f64 {
    raw as f64 * 10.0
}

/// Descriptor 0;21;201 — Range bin size
/// 14-bit, scale=0, ref=0 → physical = raw meters
pub fn decode_bin_size(raw: u64) -> f64 {
    raw as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_code_mapping() {
        assert_eq!(product_code_to_cfradial(0), Some("DBZH"));
        assert_eq!(product_code_to_cfradial(40), Some("VRADH"));
        assert_eq!(product_code_to_cfradial(239), Some("PHIDP"));
        assert_eq!(product_code_to_cfradial(241), Some("RHOHV"));
        assert_eq!(product_code_to_cfradial(243), Some("CM"));
        assert_eq!(product_code_to_cfradial(99), None);
    }

    #[test]
    fn test_elevation_angle() {
        // raw=9051 → (9051 - 9000) / 100 = 0.51 degrees
        let angle = decode_elevation_angle(9051);
        assert!((angle - 0.51).abs() < 1e-9, "got {}", angle);
    }

    #[test]
    fn test_azimuth() {
        // raw=1300 → 1300 / 100 = 13.00 degrees
        let az = decode_azimuth(1300);
        assert!((az - 13.0).abs() < 1e-9, "got {}", az);
    }

    #[test]
    fn test_latitude() {
        // RMA1 lat = -31.4413°
        // raw = -31.4413 * 100000 + 9000000 = 5855870
        let raw: u64 = 5_855_870;
        let lat = decode_latitude(raw);
        assert!((lat - (-31.4413)).abs() < 1e-4, "got {}", lat);
    }

    #[test]
    fn test_longitude() {
        // RMA1 lon = -64.1919°
        // raw = -64.1919 * 100000 + 18000000 = 11580810
        // Approximate
        let raw: u64 = 11_580_810;
        let lon = decode_longitude(raw);
        assert!((lon - (-64.1919)).abs() < 1e-3, "got {}", lon);
    }

    #[test]
    fn test_height() {
        // raw=884 → 884 - 400 = 484 meters
        let h = decode_height(884);
        assert!((h - 484.0).abs() < 1e-9, "got {}", h);
    }

    #[test]
    fn test_bin_offset() {
        // raw=210 → 210 * 10 = 2100 meters
        let offset = decode_bin_offset(210);
        assert!((offset - 2100.0).abs() < 1e-9, "got {}", offset);
    }

    #[test]
    fn test_bin_size() {
        // raw=120 → 120 meters
        let size = decode_bin_size(120);
        assert!((size - 120.0).abs() < 1e-9, "got {}", size);
    }
}
