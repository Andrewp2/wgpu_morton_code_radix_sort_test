use crate::utilities::calculate_number_of_workgroups_u32;

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
