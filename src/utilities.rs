use std::ops::Sub;

use std::ops::Add;

use std::cmp::PartialEq;

use std::ops::Rem;

/// Divides l by r, and takes the ceiling instead of the floor.
pub fn div_ceil_u32(l: u32, r: u32) -> u32 {
    return (l + r - 1) / r;
}

/// Divids l by r and takes the ceiling instead of the floor.
pub fn div_ceil_u64(l: u64, r: u64) -> u64 {
    return (l + r - 1) / r;
}

/// Calculates the number of workgroups needed to handle all the elements.
pub fn calculate_number_of_workgroups_u32(element_count: u32, workgroup_size: u32) -> u32 {
    return div_ceil_u32(element_count, workgroup_size);
}

// Calculates the number of workgroups needed to handle all the elements.
pub fn calculate_number_of_workgroups_u64(element_count: u64, workgroup_size: u64) -> u64 {
    return div_ceil_u64(element_count, workgroup_size);
}

/// Calculates the next multiple of a u32 number, such that l % r == 0.
pub fn round_up_u32<
    T: Rem<Output = T> + PartialEq<u32> + Add<Output = T> + Sub<Output = T> + Copy,
>(
    l: T,
    r: T,
) -> T {
    let remainder = l % r;
    if remainder == 0u32 {
        return l;
    }
    return l + r - remainder;
}

/// Calculates the next multiple of a u64 number, such that l % r == 0.
pub fn round_up_u64<
    T: Rem<Output = T> + PartialEq<u64> + Add<Output = T> + Sub<Output = T> + Copy,
>(
    l: T,
    r: T,
) -> T {
    let remainder = l % r;
    if remainder == 0u64 {
        return l;
    }
    return l + r - remainder;
}
