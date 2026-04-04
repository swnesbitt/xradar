/// Zlib decompression for Argentine BUFR data arrays.
///
/// The 3;21;206 descriptor accumulates bytes from a nested delayed-replication
/// structure and decompresses them as a single zlib stream into n_azimuths × n_bins
/// IEEE 754 float64 little-endian values in physical units.

use flate2::read::ZlibDecoder;
use std::io::Read;

use super::errors::BufrError;

/// Decompress a zlib byte stream and interpret as little-endian f64 values.
pub fn decompress_to_f64(compressed_bytes: &[u8], expected_count: usize) -> Result<Vec<f64>, BufrError> {
    let mut decoder = ZlibDecoder::new(compressed_bytes);
    let mut decompressed = Vec::with_capacity(expected_count * 8);
    decoder.read_to_end(&mut decompressed).map_err(|e| {
        BufrError::DecompressionError(format!("zlib decompression failed: {}", e))
    })?;

    if decompressed.len() < expected_count * 8 {
        return Err(BufrError::DecompressionError(format!(
            "Decompressed {} bytes, expected at least {}",
            decompressed.len(),
            expected_count * 8
        )));
    }

    let values: Vec<f64> = decompressed[..expected_count * 8]
        .chunks_exact(8)
        .map(|chunk| {
            let arr: [u8; 8] = chunk.try_into().unwrap();
            f64::from_le_bytes(arr)
        })
        .collect();

    Ok(values)
}

/// Decompress gzip-compressed file bytes (the outer .BUFR.gz wrapper).
pub fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>, BufrError> {
    use flate2::read::GzDecoder;
    let mut decoder = GzDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).map_err(|e| {
        BufrError::DecompressionError(format!("gzip decompression failed: {}", e))
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn make_zlib(values: &[f64]) -> Vec<u8> {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        for v in values {
            encoder.write_all(&v.to_le_bytes()).unwrap();
        }
        encoder.finish().unwrap()
    }

    #[test]
    fn test_decompress_f64_round_trip() {
        let original = vec![1.5f64, -31.4413, 0.51, 120.0, 956.0];
        let compressed = make_zlib(&original);
        let result = decompress_to_f64(&compressed, original.len()).unwrap();
        assert_eq!(result.len(), original.len());
        for (a, b) in result.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-12, "{} != {}", a, b);
        }
    }

    #[test]
    fn test_decompress_too_short() {
        let compressed = make_zlib(&[1.0f64]); // only 1 value
        let result = decompress_to_f64(&compressed, 10); // expect 10
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_bad_data() {
        let result = decompress_to_f64(b"not zlib data", 1);
        assert!(result.is_err());
    }
}
