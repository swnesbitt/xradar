/// BUFR Section 0 and Section 1 parsing.
///
/// BUFR Edition 4 layout:
/// Section 0: 8 bytes  — magic "BUFR" + 3-byte length + edition
/// Section 1: 22 bytes — metadata
/// Section 2: absent (optional section flag = 0)
/// Section 3: 14 bytes — descriptor list
/// Section 4: variable — data
/// Section 5: 4 bytes  — "7777"

use super::errors::BufrError;

#[derive(Debug)]
pub struct Section0 {
    pub total_length: u32, // bytes 4-6, 3-byte big-endian
    pub edition: u8,
}

#[derive(Debug)]
pub struct Section1 {
    pub master_table: u8,
    pub originating_centre: u16,
    pub originating_subcentre: u16,
    pub update_sequence: u8,
    pub optional_section: u8,
    pub data_category: u8,       // 6 = radar
    pub intl_sub_category: u8,
    pub local_sub_category: u8,
    pub master_table_version: u8,
    pub local_table_version: u8,
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
}

/// Offsets into the BUFR message
#[derive(Debug)]
pub struct SectionOffsets {
    pub section4_offset: usize, // byte offset of Section 4 data
    pub section4_length: usize, // length of Section 4 (excluding 4-byte length prefix)
}

pub fn parse_section0(data: &[u8]) -> Result<Section0, BufrError> {
    if data.len() < 8 {
        return Err(BufrError::InvalidSection("Too short for Section 0".to_string()));
    }
    if &data[0..4] != b"BUFR" {
        return Err(BufrError::InvalidMagic(data[0..4].to_vec()));
    }
    let total_length = ((data[4] as u32) << 16) | ((data[5] as u32) << 8) | (data[6] as u32);
    let edition = data[7];
    if edition != 4 {
        return Err(BufrError::UnsupportedEdition(edition));
    }
    Ok(Section0 { total_length, edition })
}

pub fn parse_section1(data: &[u8]) -> Result<Section1, BufrError> {
    // Section 1 starts at byte 8
    let s1 = &data[8..];
    if s1.len() < 22 {
        return Err(BufrError::InvalidSection("Too short for Section 1".to_string()));
    }
    let section_length = ((s1[0] as u32) << 16) | ((s1[1] as u32) << 8) | (s1[2] as u32);
    if section_length < 22 {
        return Err(BufrError::InvalidSection(format!("Section 1 length {} too short", section_length)));
    }
    Ok(Section1 {
        master_table: s1[3],
        originating_centre: ((s1[4] as u16) << 8) | s1[5] as u16,
        originating_subcentre: ((s1[6] as u16) << 8) | s1[7] as u16,
        update_sequence: s1[8],
        optional_section: s1[9],
        data_category: s1[10],
        intl_sub_category: s1[11],
        local_sub_category: s1[12],
        master_table_version: s1[13],
        local_table_version: s1[14],
        year: ((s1[15] as u16) << 8) | s1[16] as u16,
        month: s1[17],
        day: s1[18],
        hour: s1[19],
        minute: s1[20],
        second: s1[21],
    })
}

/// Find Section 4 data offset and length.
/// Assumes no optional Section 2 (optional_section flag = 0).
pub fn find_section4(data: &[u8]) -> Result<SectionOffsets, BufrError> {
    // Section 0: 8 bytes
    // Section 1: variable (read length)
    if data.len() < 11 {
        return Err(BufrError::InvalidSection("Buffer too short".to_string()));
    }
    let s1_len = ((data[8] as usize) << 16) | ((data[9] as usize) << 8) | data[10] as usize;
    let s1_end = 8 + s1_len;

    // Skip optional Section 2 (should be absent, but handle it)
    let s1_optional = data[8 + 9]; // byte 9 of section 1 = optional section flag
    let s2_end = if s1_optional != 0 {
        if s1_end + 3 > data.len() {
            return Err(BufrError::InvalidSection("Buffer too short for Section 2".to_string()));
        }
        let s2_len = ((data[s1_end] as usize) << 16) | ((data[s1_end + 1] as usize) << 8) | data[s1_end + 2] as usize;
        s1_end + s2_len
    } else {
        s1_end
    };

    // Section 3
    if s2_end + 3 > data.len() {
        return Err(BufrError::InvalidSection("Buffer too short for Section 3".to_string()));
    }
    let s3_len = ((data[s2_end] as usize) << 16) | ((data[s2_end + 1] as usize) << 8) | data[s2_end + 2] as usize;
    let s3_end = s2_end + s3_len;

    // Section 4 starts here
    if s3_end + 4 > data.len() {
        return Err(BufrError::InvalidSection("Buffer too short for Section 4 header".to_string()));
    }
    let s4_len = ((data[s3_end] as usize) << 16) | ((data[s3_end + 1] as usize) << 8) | data[s3_end + 2] as usize;
    let section4_offset = s3_end + 4; // skip 4-byte length prefix
    let section4_length = s4_len.saturating_sub(4);

    Ok(SectionOffsets { section4_offset, section4_length })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_section0() -> Vec<u8> {
        let mut v = vec![b'B', b'U', b'F', b'R'];
        // total length = 1000
        v.push(0x00); v.push(0x03); v.push(0xE8);
        v.push(4); // edition 4
        v
    }

    #[test]
    fn test_parse_section0_valid() {
        let data = make_section0();
        // Extend with dummy section 1
        let mut full = data;
        full.extend(vec![0u8; 100]);
        let s0 = parse_section0(&full).unwrap();
        assert_eq!(s0.edition, 4);
        assert_eq!(s0.total_length, 1000);
    }

    #[test]
    fn test_parse_section0_bad_magic() {
        let data = b"XXXX\x00\x03\xE8\x04";
        assert!(parse_section0(data).is_err());
    }

    #[test]
    fn test_parse_section0_wrong_edition() {
        let data = b"BUFR\x00\x03\xE8\x03";
        assert!(parse_section0(data).is_err());
    }

    #[test]
    fn test_parse_section1() {
        // Build minimal Section 1 at offset 8
        let mut data = vec![0u8; 8]; // Section 0 placeholder
        // Section 1: 22 bytes
        data.push(0x00); data.push(0x00); data.push(22); // length = 22
        data.push(0); // master_table
        data.push(0); data.push(41); // centre = 41
        data.push(0); data.push(0);  // subcentre
        data.push(0); // update
        data.push(0); // optional
        data.push(6); // data_category = 6 (radar)
        data.push(2); // intl sub
        data.push(0); // local sub
        data.push(16); // master table version
        data.push(2); // local table version
        data.push(0x07); data.push(0xE2); // year = 2018
        data.push(11); // month
        data.push(11); // day
        data.push(0); // hour
        data.push(4); // minute
        data.push(57); // second

        let s1 = parse_section1(&data).unwrap();
        assert_eq!(s1.originating_centre, 41);
        assert_eq!(s1.data_category, 6);
        assert_eq!(s1.year, 2018);
        assert_eq!(s1.month, 11);
        assert_eq!(s1.local_table_version, 2);
    }
}
