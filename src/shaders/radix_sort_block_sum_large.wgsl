// guaranteed to be padded out to multiple of 256
@group(0) @binding(0)
var<storage, read_write> vals: array<u32>;

@group(0) @binding(1)
var<storage, read_write> block_sums: array<u32>;

var<workgroup> scratch: array<u32, 256>;

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
        if i >= 32u {
            workgroupBarrier();
        }
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
        if i >= 32u {
            workgroupBarrier();
        }
        if wid % i == 0u {
            prefix_sum_swap(wid, (i / 2u) - 1u, i - 1u);
        }
    }
    workgroupBarrier();
}

fn prefix_sum_block_exclusive_unrolled(wid: u32, id: u32) {
    let v = vals[id];
    scratch[wid] = v;
    if wid % 2u == 0u {
        scratch[wid + 1u] += scratch[wid];
    }
    if wid % 4u == 0u {
        scratch[wid + 3u] += scratch[wid + 1u];
    }
    if wid % 8u == 0u {
        scratch[wid + 7u] += scratch[wid + 3u];
    }
    if wid % 16u == 0u {
        scratch[wid + 15u] += scratch[wid + 7u];
    }
    if wid % 32u == 0u {
        scratch[wid + 31u] += scratch[wid + 15u];
    }
    workgroupBarrier();
    if wid % 64u == 0u {
        scratch[wid + 63u] += scratch[wid + 31u];
    }
    workgroupBarrier();
    if wid % 128u == 0u {
        scratch[wid + 127u] += scratch[wid + 63u];
    }
    workgroupBarrier();
    if wid % 256u == 0u {
        scratch[wid + 255u] += scratch[wid + 127u];
    }
    workgroupBarrier();
    // special case for first iteration
    if wid % 256u == 0u {
        let before = scratch[127u];
        scratch[127u] = 0u;
        scratch[255u] = before;
    }
    var i = 128u;
    workgroupBarrier();
    if wid % 128u == 0u {
        let lo = 63u;
        let hi = 127u;
        let before = scratch[wid + lo];
        let after = scratch[wid + hi];
        scratch[wid + lo] = after;
        scratch[wid + hi] += before;
    }
    workgroupBarrier();
    if wid % 64u == 0u {
        let lo = 31u;
        let hi = 63u;
        let before = scratch[wid + lo];
        let after = scratch[wid + hi];
        scratch[wid + lo] = after;
        scratch[wid + hi] += before;
    }
    workgroupBarrier();
    if wid % 32u == 0u {
        let lo = 15u;
        let hi = 31u;
        let before = scratch[wid + lo];
        let after = scratch[wid + hi];
        scratch[wid + lo] = after;
        scratch[wid + hi] += before;
    }
    if wid % 16u == 0u {
        let lo = 7u;
        let hi = 15u;
        let before = scratch[wid + lo];
        let after = scratch[wid + hi];
        scratch[wid + lo] = after;
        scratch[wid + hi] += before;
    }
    if wid % 8u == 0u {
        let lo = 3u;
        let hi = 7u;
        let before = scratch[wid + lo];
        let after = scratch[wid + hi];
        scratch[wid + lo] = after;
        scratch[wid + hi] += before;
    }
    if wid % 4u == 0u {
        let lo = 1u;
        let hi = 3u;
        let before = scratch[wid + lo];
        let after = scratch[wid + hi];
        scratch[wid + lo] = after;
        scratch[wid + hi] += before;
    }
    if wid % 2u == 0u {
        let hi = 1u;
        let before = scratch[wid];
        let after = scratch[wid + hi];
        scratch[wid] = after;
        scratch[wid + hi] += before;
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

    prefix_sum_block_exclusive_unrolled(wid, id);
    if wid == 255u {
        block_sums[id / 256u] = scratch[wid] + vals[id];
    }
    vals[id] = scratch[wid];
}

@compute @workgroup_size(256, 1, 1)
fn radix_sort_block_sum_large_after(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let id = invocation_id.x;
    if id < arrayLength(&vals) {
        vals[id] += block_sums[(id / 256u)];
    }
}