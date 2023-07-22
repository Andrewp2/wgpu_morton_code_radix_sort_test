const NUM_BANKS = 32u;
const LOG_NUM_BANKS = 5u;
const BITS_PER_KEY = 64u;
const BITS_PER_DIGIT = 5u;

// guranteed to be smaller or equal to 1024
// guaranteed to be padded out to multiple of 256
@group(0) @binding(0)
var<storage, read_write> block_sums: array<u32>;

var<workgroup> scratch: array<u32, 256>;

fn prefix_sum_swap(wid: u32, lo: u32, hi: u32) {
    let before = scratch[wid + lo];
    let after = scratch[wid + hi];
    scratch[wid + lo] = after;
    scratch[wid + hi] += before;
}

fn prefix_sum_block_exclusive(wid: u32, offset: u32) {
    let v = block_sums[wid + offset];
    scratch[wid] = v;
    for (var i: u32 = 1u; i < 256u; i = i << 1u) {
        workgroupBarrier();
        if wid % (2u * i) == 0u {
            scratch[wid + (2u * i) - 1u] += scratch[wid + i - 1u];
        }
    }
    workgroupBarrier();
    // special case for first iteration
    if wid % 256u == 0u {
        let before = scratch[127u];
        scratch[127u] = 0u;
        scratch[255u] = before;
    }
    // 128 64 32 16 8 4 2
    for (var i: u32 = 128u; i > 1u; i = i >> 1u) {
        workgroupBarrier();
        if wid % i == 0u {
            prefix_sum_swap(wid, (i / 2u) - 1u, i - 1u);
        }
    }
    workgroupBarrier();
}

@compute @workgroup_size(256, 1, 1)
fn radix_sort_block_sum_small(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let id = invocation_id.x;
    let wid = workgroup_id.x;

    prefix_sum_block_exclusive(wid, 0u);
    let m_1 = block_sums[255u] + scratch[255u];
    //let m_1 = block_sums[255u] + workgroupUniformLoad(&scratch[255u]);
    block_sums[wid] = scratch[wid];
    if arrayLength(&block_sums) > 256u {
        prefix_sum_block_exclusive(wid, 256u);
        let m_2 = block_sums[511u] + scratch[255u] + m_1;
        block_sums[wid + 256u] = scratch[wid] + m_1;
        if arrayLength(&block_sums) > 512u {
            prefix_sum_block_exclusive(wid, 512u);
            let m_3 = block_sums[767u] + scratch[255u] + m_2;
            block_sums[wid + 512u] = scratch[wid] + m_2;
            if arrayLength(&block_sums) > 768u {
                prefix_sum_block_exclusive(wid, 768u);
                block_sums[wid + 768u] = scratch[wid] + m_3;
            }
        }
    }
}