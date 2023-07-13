const NUM_BANKS = 32u;
const LOG_NUM_BANKS = 5u;
const BITS_PER_KEY = 64u;
const BITS_PER_DIGIT = 5u;

// guaranteed to be padded out to multiple of 256
@group(0) @binding(0)
var<storage, read_write> vals: array<u32>;

@group(0) @binding(1)
var<storage, read_write> block_sums: array<u32>;

var<workgroup, read_write> scratch: array<u32, 256>;

fn prefix_sum_swap(wid: u32, lo: u32, hi: u32) {
    let before = scratch[wid + lo];
    let after = scratch[wid + hi];
    scratch[wid + lo] = after;
    scratch[wid + hi] += before;
}

fn prefix_sum_block_exclusive(wid: u32, id: u32) {
    let v = vals[id];
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
fn radix_sort_block_sum_large(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let id = invocation_id.x;
    let wid = workgroup_id.x;

    prefix_sum_block_exclusive(wid, id);
    if wid == 255u {
        block_sums[id / 256u] = scratch[wid] + vals[id];
    }
    vals[id] += scratch[wid];
}

@compute @workgroup_size(256, 1, 1)
fn radix_sort_block_sum_large_after(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let id = invocation_id.x;
    if id > 256u {
        vals[id] += block_sums[(id / 256u) - 1u];
    }
}