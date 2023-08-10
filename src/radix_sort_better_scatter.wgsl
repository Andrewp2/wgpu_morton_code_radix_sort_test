struct TriangleIndices {
    tri_indices: vec3<u32>,
    // TODO: Think about this (we can fit one u32 for free, but not 3)
    // node_index: u32,
    // material_id: u32,
    // flags: u32,
};

struct ScatterUniforms {
    // 0-x
    scatter_step: u32,
    // constant, greater than 0
    number_sections: u32,
};

@group(0) @binding(0)
var<uniform> uniforms: ScatterUniforms;

@group(0) @binding(1)
var<storage, read> triangles: array<TriangleIndices>;

@group(0) @binding(2)
var<storage, read> codes: array<u32>;

@group(0) @binding(3)
var<storage, read_write> triangles_2: array<TriangleIndices>;

@group(0) @binding(4)
var<storage, read_write> codes_2: array<u32>;

@group(0) @binding(5)
var<storage, read> final_locations: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn radix_sort_scatter(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_workgroup_invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {

    let id = invocation_id.x;
    let wid = local_workgroup_invocation_id.x;
    let num_elements = arrayLength(&codes) / 2u;

    if id < num_elements {
        let section_size = num_elements / uniforms.number_sections;
        let start_index = section_size * uniforms.scatter_step;
        let end_index = start_index + section_size;
        let val = final_locations[id];
        if val >= start_index && val < end_index {
            codes_2[2u * val] = codes[2u * id];
            codes_2[2u * val + 1u] = codes[2u * id + 1u];
            triangles_2[val] = triangles[id];
        }
    }
}