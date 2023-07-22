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
var<storage, read_write> storage_histograms: array<u32>;

var<workgroup> histogram: array<atomic<u32>, 64>;
var<workgroup> histogram_na: array<u32, 64>;
var<workgroup> sorting_scratch: array<u32, 256>;

var<private> local_id: u32;

fn select_digit(id: u32) -> u32 {
    var val = 64u;
    // 0      1      2        3        4        5
    // [0, 4] [5, 9] [10, 14] [15, 19] [20, 24] [25, 29]
    // 6 
    // [30, 34]
    // 7        8        9        10       11       12 
    // [35, 39] [40, 44] [45, 49] [50, 54] [55, 59] [60, 63]
    if uniforms.pass_number < 6u {
        if uniforms.pass_number % 2u == 0u {
            val = (codes[2u * id + 1u] >> (uniforms.pass_number * 5u)) & 63u;
        } else {
            val = (codes_2[2u * id + 1u] >> (uniforms.pass_number * 5u)) & 63u;
        }
    } else if uniforms.pass_number > 6u {
        if uniforms.pass_number % 2u == 0u {
            val = (codes[2u * id] >> (uniforms.pass_number * 5u + 2u)) & 63u;
        } else {
            val = (codes_2[2u * id] >> (uniforms.pass_number * 5u + 2u)) & 63u;
        }
    } else {
        val = ((codes[2u * id + 1u] >> 30u) & 3u) | ((codes[2u * id]) & 7u);
    }
    return val;
}

fn prefix_sum_swap(wid: u32, lo: u32, hi: u32) {
    let before = histogram_na[wid + lo];
    let after = histogram_na[wid + hi];
    histogram_na[wid + lo] = after;
    histogram_na[wid + hi] += before;
}

fn prefix_sum_block_exclusive(wid: u32, id: u32, v: u32, prefix_sum_size: u32) {
    histogram_na[wid] = v;
    for (var i: u32 = 1u; i < prefix_sum_size; i = i << 1u) {
        workgroupBarrier();
        if wid % (2u * i) == 0u {
            histogram_na[wid + (2u * i) - 1u] += histogram_na[wid + i - 1u];
        }
    }
    workgroupBarrier();
   // special case for first iteration
    if wid % prefix_sum_size == 0u {
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

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

fn cmp_and_swap(wid: u32, lo: u32, hi: u32) {
    if sorting_scratch[lo] > sorting_scratch[hi] {
        if wid < 128u {
            let x = sorting_scratch[hi];
            sorting_scratch[hi] = sorting_scratch[lo];
            sorting_scratch[lo] = x;
        }
        if hi == local_id {
            local_id = lo;
        } else {
            local_id = hi;
        }
    }
}

fn do_flip(t: u32, h: u32) {
    let q = ((t * 2u) / h) * h;
    let zero = q + t % (h / 2u);
    let one = q + h - 1u - (t % (h / 2u));
    cmp_and_swap(t, zero, one);
}

fn do_disperse(t: u32, h: u32) {
    let q = ((t * 2u) / h) * h;
    let zero = q + (t % (h / 2u));
    let one = q + (t % (h / 2u)) + (h / 2u);
    cmp_and_swap(t, zero, one);
}

// https://poniesandlight.co.uk/reflect/bitonic_merge_sort/
fn local_sort(wid: u32) {
    for (var h = 2u; h <= 256u; h *= 2u) {
        do_flip(wid, h);
        workgroupBarrier();
        for (var hh: u32 = h / 2u; hh > 1u; h /= 2u) {
            do_disperse(wid, hh);
            workgroupBarrier();
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn radix_sort_scatter(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let id = invocation_id.x;
    let wid = workgroup_id.x;
    local_id = wid;
    let num_elements = arrayLength(&codes) / 2u;

    var digit = 1000u;

    if id < num_elements {
        digit = select_digit(id);
    }
    let v = (digit << 8u) | wid;
    sorting_scratch[wid] = v;
    workgroupBarrier();
    local_sort(wid);
    if id < num_elements {
        atomicAdd(&histogram[digit], 1u);
    }
    workgroupBarrier();
    if wid < 64u {
        prefix_sum_block_exclusive(wid, id, atomicLoad(&histogram[wid]), 64u);
    }
    // local_id is now filled with sorted index
    // doing a prefix sum and calculating:
    // local_id - histogram_na[digit] 
    // calculates the number of elements with the same value before this in the block
    // adding on the storage histogram tells us how many lower value/lower-index_same-value digits there are
    if id < num_elements {
        let num_blocks_before_this: u32 = id / 256u;
        let final_location = (local_id - histogram_na[digit]) + storage_histograms[((div_ceil(num_elements, 256u) * 4u) * wid) + num_blocks_before_this];
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
