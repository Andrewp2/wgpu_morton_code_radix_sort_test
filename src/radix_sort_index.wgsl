struct PrefixUniforms {
    pass_number: u32,
};

@group(0) @binding(0)
var<uniform> uniforms: PrefixUniforms;

@group(0) @binding(1)
var<storage, read> codes: array<u32>;

@group(0) @binding(2)
var<storage, read> storage_histograms: array<u32>;

@group(0) @binding(3)
var<storage, read_write> final_locations: array<u32>;

var<workgroup> histogram: array<atomic<u32>, 256>;
var<workgroup> histogram_na: array<u32, 256>;
var<workgroup> sorting_scratch: array<u32, 256>;
var<workgroup> local_ids: array<u32, 256>;

fn select_digit(id: u32) -> u32 {
    var val = 256u;
    if uniforms.pass_number < 4u {
        val = (codes[2u * id] >> (uniforms.pass_number * 8u)) & 255u;
    } else {
        val = (codes[2u * id + 1u] >> ((uniforms.pass_number - 4u) * 8u)) & 255u;
    }
    return val;
}

// --------------------------------------------------------

fn prefix_sum_swap(wid: u32, lo: u32, hi: u32) {
    let before = histogram_na[wid + lo];
    let after = histogram_na[wid + hi];
    histogram_na[wid + lo] = after;
    histogram_na[wid + hi] += before;
}

fn prefix_sum_block_exclusive(wid: u32, prefix_sum_size: u32) {
    for (var i: u32 = 1u; i < prefix_sum_size; i = i << 1u) {
        workgroupBarrier();
        if wid % (2u * i) == 0u {
            histogram_na[wid + (2u * i) - 1u] += histogram_na[wid + i - 1u];
        }
    }
    workgroupBarrier();
    // special case for first iteration
    if wid % prefix_sum_size == 0u {
        // 256 / 2 - 1 = 127
        let before = histogram_na[(prefix_sum_size / 2u) - 1u];

        histogram_na[(prefix_sum_size / 2u) - 1u] = 0u;
        histogram_na[prefix_sum_size - 1u] = before;
    }
    
    // 128 64 32 16 8 4 2
    for (var i: u32 = prefix_sum_size / 2u; i > 1u; i = i >> 1u) {
        workgroupBarrier();
        if wid % i == 0u {
            prefix_sum_swap(wid, (i / 2u) - 1u, i - 1u);
        }
    }
    workgroupBarrier();
}

// ----------------------------------------------------------

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

fn cmp_and_swap(wid: u32, lo: u32, hi: u32) {
    if wid < 128u {
        let val_at_lo = sorting_scratch[lo];
        let val_at_hi = sorting_scratch[hi];
        if val_at_lo > val_at_hi {
            sorting_scratch[hi] = val_at_lo;
            sorting_scratch[lo] = val_at_hi;
            local_ids[val_at_lo & 255u] = hi;
            local_ids[val_at_hi & 255u] = lo;
        }
    }
}

fn do_flip(t: u32, h: u32) {
    let half_h = h / 2u;
    let q = ((t * 2u) / h) * h;
    let zero = q + (t % half_h);
    let one = q + h - 1u - (t % half_h);
    cmp_and_swap(t, zero, one);
}

fn do_disperse(t: u32, h: u32) {
    let half_h = h / 2u;
    let q = ((t * 2u) / h) * h;
    let zero = q + (t % half_h);
    let one = q + (t % half_h) + half_h;
    cmp_and_swap(t, zero, one);
}

// https://poniesandlight.co.uk/reflect/bitonic_merge_sort/
fn local_sort(wid: u32) {
    for (var h: u32 = 2u; h <= 256u; h *= 2u) {
        do_flip(wid, h);
        workgroupBarrier();
        for (var hh: u32 = h / 2u; hh > 1u; hh /= 2u) {
            do_disperse(wid, hh);
            workgroupBarrier();
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn radix_sort_index(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_workgroup_invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let id = invocation_id.x;
    let wid = local_workgroup_invocation_id.x;
    local_ids[wid] = wid;
    let num_elements = arrayLength(&codes) / 2u;

    var digit = 16000u;

    if id < num_elements {
        digit = select_digit(id);
    }
    let v = (digit << 16u) | wid;
    sorting_scratch[wid] = v;
    workgroupBarrier();
    local_sort(wid);
    workgroupBarrier();
    if id < num_elements {
        atomicAdd(&histogram[digit], 1u);
    }
    workgroupBarrier();
    let b = atomicLoad(&histogram[wid]);
    histogram_na[wid] = b;
    prefix_sum_block_exclusive(wid, 256u);
    // local_id is now filled with sorted index
    // doing a prefix sum and calculating:
    // local_id - histogram_na[digit] 
    // calculates the number of elements with the same value before this in the block
    // adding on the storage histogram tells us how many lower value/lower-index_same-value digits there are

    if id < num_elements {
        let num_blocks_before_this: u32 = id / 256u;
        let histogram_column = num_blocks_before_this;
        let histogram_row = digit;
        let histogram_index = (num_workgroups.x * histogram_row) + histogram_column;
        let l = local_ids[wid];
        let final_location = (l + storage_histograms[histogram_index]) - histogram_na[digit];
        final_locations[id] = final_location;
    }
}
