struct TriangleIndices {
    tri_indices: vec3<u32>,
    // TODO: Think about this (we can fit one u32 for free, but not 3)
    node_index: u32,
    material_id: u32,
    flags: u32,
};

struct PrefixUniforms {
    // 0-12
    pass_number: u32,
};

const NUM_BANKS = 32u;
const LOG_NUM_BANKS = 5u;
const BITS_PER_KEY = 64u;
const BITS_PER_DIGIT = 5u;

@group(0) @binding(0)
var<uniform> uniforms: PrefixUniforms;

@group(0) @binding(1)
var<storage, read_write> triangles: array<TriangleIndices>;

@group(0) @binding(2)
var<storage, read_write> codes: array<u32>;

@group(0) @binding(3)
var<storage, read_write> triangles_2: array<TriangleIndices>;

@group(0) @binding(4)
var<storage, read_write> codes_2: array<u32>;

@group(0) @binding(5)
var<storage, read> storage_histograms: array<u32>;

var<workgroup> histogram: array<atomic<u32>, 64>;
var<workgroup> histogram_na: array<u32, 64>;
var<workgroup> sorting_scratch: array<u32, 256>;
var<workgroup> local_ids: array<u32, 256>;

fn select_digit(id: u32) -> u32 {
    var val = 64u;
    // determines if we're using the ping-pong buffer or not
    let use_original_array = uniforms.pass_number % 2u == 0u;
    if uniforms.pass_number < 5u {
        if use_original_array {
            val = (codes[2u * id + 0u] >> (uniforms.pass_number * 6u)) & 63u;
        } else {
            val = (codes_2[2u * id + 0u] >> (uniforms.pass_number * 6u)) & 63u;
        }
    } else if uniforms.pass_number > 5u {
        if use_original_array {
            val = (codes[2u * id + 1u] >> ((uniforms.pass_number - 6u) * 6u + 4u)) & 63u;
        } else {
            val = (codes_2[2u * id + 1u] >> ((uniforms.pass_number - 6u) * 6u + 4u)) & 63u;
        }
    } else {
        val = ((codes_2[2u * id + 0u] >> 30u) & 3u) | (((codes_2[2u * id + 1u]) & 15u) << 2u);
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
        if wid < prefix_sum_size {
            if wid % (2u * i) == 0u {
                histogram_na[wid + (2u * i) - 1u] += histogram_na[wid + i - 1u];
            }
        }
    }
    workgroupBarrier();
    // special case for first iteration
    if wid < prefix_sum_size {
        if wid % prefix_sum_size == 0u {
        // 64 / 2 - 1 = 31
            let before = histogram_na[(prefix_sum_size / 2u) - 1u];

            histogram_na[(prefix_sum_size / 2u) - 1u] = 0u;
            histogram_na[prefix_sum_size - 1u] = before;
        }
    }
    // 32 16 8 4 2
    for (var i: u32 = prefix_sum_size / 2u; i > 1u; i = i >> 1u) {
        workgroupBarrier();
        if wid < prefix_sum_size {
            if wid % i == 0u {
                prefix_sum_swap(wid, (i / 2u) - 1u, i - 1u);
            }
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
fn radix_sort_scatter(
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
    //--------------------------------------------------------------------------
    var b = 1000u;
    if wid < 64u {
        b = atomicLoad(&histogram[wid]);
    }
    workgroupBarrier();
    if wid < 64u {
        histogram_na[wid] = b;
    }
    workgroupBarrier();
    prefix_sum_block_exclusive(wid, 64u);
    //--------------------------------------------------------------------------
    workgroupBarrier();
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
        // codes_2[id * 2u] = histogram_na[digit];

        // codes_2[id * 2u] = (l << 16u) | storage_histograms[histogram_index];
        // codes_2[id * 2u + 1u] = (histogram_na[digit] << 16u) | digit;

        // if wid < 64u {
        //     codes_2[id * 2u] = histogram_na[digit];
        // }
        // if digit > 63u {
        //     codes_2[id * 2u] = 4000u;
        // }
        //codes_2[id * 2u + 1u] = (histogram_na[digit] << 16u);
        //codes_2[id * 2u + 1u] = (atomicLoad(&histogram[digit]) << 16u) | digit;

        if uniforms.pass_number % 2u == 0u {
            codes_2[final_location * 2u] = codes[id * 2u];
            codes_2[final_location * 2u + 1u] = codes[id * 2u + 1u];
            triangles_2[final_location] = triangles[id];
        } else {
            codes[final_location * 2u] = codes_2[id * 2u];
            codes[final_location * 2u + 1u] = codes_2[id * 2u + 1u];
            triangles[final_location] = triangles_2[id];
        }
    }
}
