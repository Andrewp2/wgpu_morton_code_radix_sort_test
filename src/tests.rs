use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

use crate::{constants, utilities::calculate_number_of_workgroups_u32};

fn select_digit(pass_number: u32, code_1: u32, code_2: u32) -> u32 {
    let val = match pass_number {
        0..=4 => (code_1 >> (pass_number * 6)) & 63,
        5 => ((code_1 >> 30) & 3) | ((code_2 & 15) << 2),
        6..=10 => (code_2 >> ((pass_number - 6) * 6 + 4)) & 63,
        _ => panic!(),
    };
    return val;
}

fn select_digit_8(pass_number: u32, code_1: u32, code_2: u32) -> u32 {
    let val = match pass_number {
        0..=3 => (code_1 >> (pass_number * 8)) & 255,
        4..=8 => (code_2 >> ((pass_number - 4) * 8)) & 255,
        _ => panic!(),
    };
    return val;
}

fn is_sorted<T: Ord>(vec: &Vec<T>) -> bool {
    vec.windows(2).all(|w| w[0] <= w[1])
}

#[test]
fn workgroup_local_index_test() {
    for seed in 0..1000 {
        let mut rng_gen = ChaCha8Rng::seed_from_u64(seed);
        let mut offset_bmp: [[u32; 8]; 256] = [[0xFFFFFFFF; 8]; 256];
        let mut local_offset: [u32; 256] = [0; 256];
        let mut ballot_mem: [u32; 8] = [0; 8];
        // Test with a specific pattern of digits (bucket IDs)
        let mut digits = [0; 256];
        for i in 0..256 {
            digits[i] = rng_gen.gen_range(0..256);
        }
        for k in 0..8 {
            ballot_mem = [0; 8];
            for wid in 0..256 {
                let bucket_id = digits[wid];
                let index = wid / 32;
                let bit_index = wid % 32;
                ballot_mem[index] |= ((bucket_id >> k) & 0x01) << bit_index;
            }
            for wid in 0..256 {
                if ((wid >> k) & 0x01) != 0 {
                    for j in 0..8 {
                        offset_bmp[wid][j] &= ballot_mem[j];
                    }
                } else {
                    for j in 0..8 {
                        offset_bmp[wid][j] &= !ballot_mem[j];
                    }
                }
            }
        }
        for wid in 0..256 {
            let mut count = 0;
            let digit = digits[wid] as usize;
            let full_u32s = (wid + 1) / 32;
            let remaining_bits = (wid + 1) % 32;

            // Count set bits in full u32 values
            for i in 0..full_u32s {
                count += offset_bmp[digit][i].count_ones();
            }

            // Count set bits in the remaining partial u32 value
            if remaining_bits > 0 {
                let mask = (1 << remaining_bits) - 1;
                count += (offset_bmp[digit][full_u32s] & mask).count_ones();
            }
            local_offset[wid] = count - 1;
        }
        let mut true_local_offset = [0; 256];
        let mut digit_count = [0; 256];
        for (i, digit) in digits.iter().enumerate() {
            true_local_offset[i] = digit_count[*digit as usize];
            digit_count[*digit as usize] += 1;
        }
        let mut histogram = [0; 256];
        digits.iter().for_each(|x| histogram[*x as usize] += 1);
        let p_sum: Vec<u32> = histogram
            .iter()
            .scan(0u32, |state, &x| {
                *state += x;
                Some(*state - x)
            })
            .collect();
        let mut sorted_output = [0; 256];
        digits.iter().enumerate().for_each(|(index, val)| {
            let f = p_sum[*val as usize] + local_offset[index];
            sorted_output[f as usize] = digits[index];
        });
        assert!(is_sorted(&sorted_output.into_iter().collect::<Vec<u32>>()));
    }
}

#[test]
fn workgroup_calculation() {
    assert_eq!(calculate_number_of_workgroups_u32(0, 256), 0, "i: {}", 0);
    for i in 1..256 {
        assert_eq!(calculate_number_of_workgroups_u32(i, 256), 1, "i: {}", i);
    }
    for i in 257..512 {
        assert_eq!(calculate_number_of_workgroups_u32(i, 256), 2, "i: {}", i);
    }
}

#[test]
fn digit_test() {
    let digit: u64 = 63;

    for i in 0..11 {
        let y: u64 = digit.wrapping_shl(i * 6);
        let hi: u32 = (y >> 32) as u32;
        let lo: u32 = (y & (u32::MAX as u64)) as u32;
        let selected = select_digit(i as u32, lo, hi);
        if i < 10 {
            assert_eq!(63, selected);
        } else {
            assert_eq!(15, selected);
        }
    }
}

#[test]
fn digit_test_8() {
    let digit: u64 = 255;

    for i in 0..8 {
        let y: u64 = digit.wrapping_shl(i * 8);
        let hi: u32 = (y >> 32) as u32;
        let lo: u32 = (y & (u32::MAX as u64)) as u32;
        let selected = select_digit_8(i as u32, lo, hi);
        assert_eq!(255, selected);
    }
}
