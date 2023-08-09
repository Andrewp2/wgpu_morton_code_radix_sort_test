
struct TriangleIndices {
    tri_indices: vec3<u32>,
    // TODO: Think about this (we can fit one u32 for free, but not 3)
    // node_index: u32,
    // material_id: u32,
    // flags: u32,
};


struct PrefixUniforms {
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
var<storage, read_write> storage_histograms: array<u32>;

var<workgroup> histogram: array<atomic<u32>, 64>;

// 0: 1 -> 2
// 1: 2 -> 1
// 2: 1 -> 2
// 3: 2 -> 1
// 4: 1 -> 2
// 5: 2 -> 1
// 6: 1 -> 2
// 7: 2 -> 1
// 8: 1 -> 2
// 9: 2 -> 1
// 10: 1 -> 2


fn select_digit(id: u32) -> u32 {
    var val = 64u;
    if uniforms.pass_number < 5u {
        val = (codes[2u * id + 0u] >> (uniforms.pass_number * 6u)) & 63u;
    } else if uniforms.pass_number > 5u {
        val = (codes[2u * id + 1u] >> ((uniforms.pass_number - 6u) * 6u + 4u)) & 63u;
    } else {
        val = ((codes[2u * id + 0u] >> 30u) & 3u) | (((codes[2u * id + 1u]) & 15u) << 2u);
    }
    return val;
}

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

        let storage_histogram_index = num_workgroups.x * wid + num_blocks_before_this;
        storage_histograms[storage_histogram_index] = atomicLoad(&histogram[wid]);
    }
}