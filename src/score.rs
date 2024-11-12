#[derive(Debug, Clone, Copy)]
pub(crate) struct AddressScore {
    pub(crate) score: u32,
}

impl AddressScore {
    pub(crate) fn calculate(address: &[u8]) -> Self {
        let leading_zeros = Self::get_leading_nibble_count(address, 0, 0);
        let leading_fours = Self::get_leading_nibble_count(address, leading_zeros as usize, 4);
        let total_fours = Self::count_fours(address);
        let has_double_44 = address[18] == 0x44 && address[19] == 0x44;

        let score =
            Self::compute_total_score(leading_zeros, leading_fours, total_fours, has_double_44);

        Self { score }
    }

    #[inline]
    fn compute_total_score(
        leading_zeros: u8,
        leading_fours: u8,
        total_fours: u32,
        has_double_44: bool,
    ) -> u32 {
        if leading_fours == 0 {
            return 0;
        }

        let mut score = (leading_zeros * 10) as u32;
        score += match leading_fours {
            4 => 60,
            5..=u8::MAX => 40,
            _ => 0,
        };
        score += total_fours;
        score += if has_double_44 { 20 } else { 0 };

        score
    }

    fn get_nibble(addr: &[u8], nibble_index: usize) -> u8 {
        let curr_byte = addr[nibble_index / 2];
        if nibble_index % 2 == 0 {
            curr_byte >> 4
        } else {
            curr_byte & 0x0F
        }
    }

    fn get_leading_nibble_count(addr: &[u8], start_index: usize, comparison: u8) -> u8 {
        let mut count = 0;
        for i in start_index..addr.len() * 2 {
            let current_nibble = Self::get_nibble(addr, i);
            if current_nibble != comparison {
                return count;
            }
            count += 1;
        }
        count
    }

    fn count_fours(address: &[u8]) -> u32 {
        let mut count = 0;
        for i in 0..address.len() * 2 {
            if Self::get_nibble(address, i) == 4 {
                count += 1;
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_calculation() {
        let test_address = [
            0x40, 0x44, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x44, 0x44,
        ];

        let score = AddressScore::calculate(&test_address);

        assert!(score.score > 0);
    }
}
