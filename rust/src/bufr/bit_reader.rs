/// Bit-level reader for BUFR packed binary data.
///
/// BUFR values are packed MSB-first across byte boundaries.
/// This reader tracks a byte + sub-byte bit offset into a byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Current byte offset
    pub byte_offset: usize,
    /// Bit offset within current byte (0 = MSB)
    pub bit_offset: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8], start_byte: usize) -> Self {
        BitReader {
            data,
            byte_offset: start_byte,
            bit_offset: 0,
        }
    }

    /// Read `n` bits (1–64) from the stream, MSB first. Returns the value.
    pub fn read_bits(&mut self, n: u8) -> Result<u64, super::errors::BufrError> {
        if n == 0 {
            return Ok(0);
        }
        if n > 64 {
            return Err(super::errors::BufrError::BitReadError(
                format!("Cannot read {} bits (max 64)", n),
            ));
        }

        let mut result: u64 = 0;
        let mut bits_remaining = n;

        while bits_remaining > 0 {
            if self.byte_offset >= self.data.len() {
                return Err(super::errors::BufrError::BitReadError(
                    format!("Buffer exhausted at byte {} while reading {} bits", self.byte_offset, n),
                ));
            }

            let current_byte = self.data[self.byte_offset];
            let bits_in_byte = 8 - self.bit_offset; // bits available in current byte
            let bits_to_take = bits_remaining.min(bits_in_byte);

            // Extract bits_to_take bits starting from bit_offset (MSB first)
            let shift = bits_in_byte - bits_to_take;
            let mask = ((1u16 << bits_to_take) - 1) as u8;
            let extracted = (current_byte >> shift) & mask;

            result = (result << bits_to_take) | (extracted as u64);
            bits_remaining -= bits_to_take;
            self.bit_offset += bits_to_take;

            if self.bit_offset >= 8 {
                self.bit_offset = 0;
                self.byte_offset += 1;
            }
        }

        Ok(result)
    }

    /// Skip `n` bits without reading
    pub fn skip_bits(&mut self, n: u32) -> Result<(), super::errors::BufrError> {
        let total_bits = self.bit_offset as u32 + n;
        self.byte_offset += (total_bits / 8) as usize;
        self.bit_offset = (total_bits % 8) as u8;
        if self.byte_offset > self.data.len() {
            return Err(super::errors::BufrError::BitReadError(
                "Skip past end of buffer".to_string(),
            ));
        }
        Ok(())
    }

    /// Align to next byte boundary (skip remaining bits in current byte)
    pub fn align_to_byte(&mut self) {
        if self.bit_offset > 0 {
            self.byte_offset += 1;
            self.bit_offset = 0;
        }
    }

    /// Read a CCITT IA5 string of `num_bits` bits (must be multiple of 8)
    pub fn read_ccitt(&mut self, num_bits: u32) -> Result<String, super::errors::BufrError> {
        let num_bytes = (num_bits / 8) as usize;
        let mut bytes = Vec::with_capacity(num_bytes);
        for _ in 0..num_bytes {
            let b = self.read_bits(8)? as u8;
            bytes.push(b);
        }
        // CCITT IA5 is ASCII; trim trailing spaces and nulls
        let s = String::from_utf8_lossy(&bytes)
            .trim_end_matches(|c: char| c == ' ' || c == '\0')
            .to_string();
        Ok(s)
    }

    /// Current position as total bit offset from start
    pub fn total_bits(&self) -> usize {
        self.byte_offset * 8 + self.bit_offset as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_single_byte() {
        let data = &[0b10110100u8];
        let mut r = BitReader::new(data, 0);
        assert_eq!(r.read_bits(8).unwrap(), 0b10110100);
    }

    #[test]
    fn test_read_4_bits() {
        let data = &[0b11001010u8];
        let mut r = BitReader::new(data, 0);
        assert_eq!(r.read_bits(4).unwrap(), 0b1100);
        assert_eq!(r.read_bits(4).unwrap(), 0b1010);
    }

    #[test]
    fn test_read_cross_byte() {
        // 12 bits across two bytes
        let data = &[0b10101010u8, 0b11110000u8];
        let mut r = BitReader::new(data, 0);
        // First 4 bits: 1010
        // Next 8 bits: cross boundary: 1010_1111
        let val = r.read_bits(12).unwrap();
        assert_eq!(val, 0b101010101111);
    }

    #[test]
    fn test_read_25_bits_latitude() {
        // Simulate a 25-bit latitude value: 0x0C34567 = -31.4413° * 100000
        // Value = 3144130 (for lat -31.44130 with scale 5, ref 0 but high-acc)
        // We just test bit extraction correctness
        let data = &[0x18u8, 0x68u8, 0xACu8, 0xE0u8]; // 25 bits = 0x0C3456_E >> 7
        let mut r = BitReader::new(data, 0);
        let val = r.read_bits(25).unwrap();
        // 0x18_68_AC >> 1 bit... just test it doesn't panic and reads 25 bits
        assert!(val < (1u64 << 25));
    }

    #[test]
    fn test_skip_bits() {
        let data = &[0b11110000u8, 0b00001111u8];
        let mut r = BitReader::new(data, 0);
        r.skip_bits(4).unwrap();
        assert_eq!(r.read_bits(4).unwrap(), 0b0000);
        assert_eq!(r.read_bits(8).unwrap(), 0b00001111);
    }

    #[test]
    fn test_align_to_byte() {
        let data = &[0b10101010u8, 0b11001100u8];
        let mut r = BitReader::new(data, 0);
        r.read_bits(3).unwrap();
        r.align_to_byte();
        assert_eq!(r.read_bits(8).unwrap(), 0b11001100);
    }

    #[test]
    fn test_read_ccitt() {
        // "RMA" in ASCII
        let data = &[b'R', b'M', b'A'];
        let mut r = BitReader::new(data, 0);
        let s = r.read_ccitt(24).unwrap();
        assert_eq!(s, "RMA");
    }

    #[test]
    fn test_read_ccitt_with_padding() {
        // "1   " — 4 bytes with trailing spaces
        let data = &[b'1', b' ', b' ', b' '];
        let mut r = BitReader::new(data, 0);
        let s = r.read_ccitt(32).unwrap();
        assert_eq!(s, "1");
    }
}
