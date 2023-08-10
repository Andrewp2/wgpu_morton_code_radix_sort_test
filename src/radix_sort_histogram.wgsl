
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

@group(0) @binding(0)
var<uniform> uniforms: PrefixUniforms;

@group(0) @binding(1)
var<storage, read> triangles: array<TriangleIndices>;

@group(0) @binding(2)
var<storage, read> codes: array<u32>;

@group(0) @binding(3)
var<storage, read_write> storage_histograms: array<u32>;

var<workgroup> histogram: array<atomic<u32>, 256>;

fn select_digit(id: u32) -> u32 {
    var val = 256u;
    if uniforms.pass_number < 4u {
        val = (codes[2u * id] >> (uniforms.pass_number * 8u)) & 255u;
    } else {
        val = (codes[2u * id + 1u] >> ((uniforms.pass_number - 4u) * 8u)) & 255u;
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
    let storage_histogram_index = num_workgroups.x * wid + num_blocks_before_this;
    storage_histograms[storage_histogram_index] = atomicLoad(&histogram[wid]);
}