
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
var<storage, read> triangles: array<TriangleIndices>;

@group(0) @binding(2)
var<storage, read> codes: array<u32>;

@group(0) @binding(3)
var<storage, read> triangles_2: array<TriangleIndices>;

@group(0) @binding(4)
var<storage, read> codes_2: array<u32>;

@group(0) @binding(5)
var<storage, read_write> storage_histograms: array<u32>;

var<workgroup, read_write> histogram: array<atomic<u32>, 64>;
//var<workgroup, read_write> histogram_na: array<u32, 64>;

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

//fn prefix_sum_swap(wid: u32, lo: u32, hi: u32) {
//    let before = histogram_na[wid + lo];
//    let after = histogram_na[wid + hi];
//    histogram_na[wid + lo] = after;
//    histogram_na[wid + hi] += before;
//}
//
//fn prefix_sum_block_exclusive(wid: u32, id: u32, v: u32, prefix_sum_size: u32) {
//    histogram_na[wid] = v;
//    for (var i: u32 = 1u; i < prefix_sum_size; i = i << 1u) {
//        workgroupBarrier();
//        if wid % (2u * i) == 0u {
//            histogram_na[wid + (2u * i) - 1u] += histogram_na[wid + i - 1u];
//        }
//    }
//    workgroupBarrier();
//    // special case for first iteration
//    if wid % prefix_sum_size == 0u {
//        block_sums[id / 256u] = histogram_na[prefix_sum_size - 1u];
//        let before = histogram_na[(prefix_sum_size / 2u) - 1u];
//
//        histogram_na[(prefix_sum_size / 2u) - 1u] = 0u;
//        histogram_na[prefix_sum_size - 1u] = before;
//    }
//    // 128 64 32 16 8 4 2
//    for (var i: u32 = prefix_sum_size / 2u; i > 1u; i = i >> 1u) {
//        workgroupBarrier();
//        if wid % i == 0u {
//            prefix_sum_swap(wid, (i / 2u) - 1u, i - 1u);
//        }
//    }
//    workgroupBarrier();
//}

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

@compute @workgroup_size(256, 1, 1)
fn radix_sort_compute_histogram(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let id = invocation_id.x;
    let wid = workgroup_id.x;

    let num_elements = arrayLength(&codes) / 2u;
    let num_blocks_before_this = id / 256u;

    // compute histogram and prefix sum

    // id may go past the end of the array, so we must account for that by not incrementing the histogram.
    if id < num_elements {
        let digit = select_digit(id);
        atomicAdd(&histogram[digit], 1u);
    }

    workgroupBarrier();
    if wid < 64u {
        //prefix_sum_block_exclusive(wid, id, atomicLoad(&histogram[wid]), 64u);
        // 0-255 256-511
        // 0-63 256-319 512-576
        // bottom is top - (192 * num blocks)
        // 0-63 64-127 128-191
        // dividing by 4 because 256 / 64 = 4
        // if there are 10,000,000 elements then - 
        // 10,000,000 / 256 = 39,062.5
        // 10,000,000 * 64 = 2,500,000
        // BUT 2,500,000 isn't divisible by 256
        // 2,500,032 is though, its 256 * 192

        // div_ceil(10,000,000, 256) = 39,063
        // 39,063 * 64 = 2,500,032
        // this is row major order, putting all 64 bits next to each other
        //storage_histograms[id - num_blocks_before_this * 192u] = histogram_na[wid];
        let storage_histogram_index = (div_ceil(num_elements, 256u) * 4u) * wid + num_blocks_before_this;
        storage_histograms[storage_histogram_index] = atomicLoad(&histogram[wid]);
    }
}