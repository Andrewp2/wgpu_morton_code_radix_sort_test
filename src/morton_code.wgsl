const NUMBER_OF_BITS_FOR_SIZE: u32 = 6u;
// 2 times number of bits
const SIZE_LUT_NUMBER_OF_BITS: u32 = 12u;
// 2^12 = 4096
const SIZE_LUT_NUMBER_OF_POSSIBLE_VALUES: u32 = 4096u;
const SIZE_LUT_NUMBER_OF_POSSIBLE_VALUES_U32: u32 = 8192u;

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uvs: vec2<f32>,
    // this is after skinning, therefore unnecessary
    //bone_ids: vec4<u32>,
    //bone_weights: vec4<f32>,
    // albedo: vec3<f32>,
};

struct TriangleIndices {
    tri_indices: vec3<u32>,
    // TODO: Think about this (we can fit one u32 for free, but not 3)
    node_index: u32,
    material_id: u32,
    flags: u32,
};

struct MortonCodeUniforms {
    lut: array<u32, 4608>,
    size_lut: array<u32, SIZE_LUT_NUMBER_OF_POSSIBLE_VALUES_U32>,
    //shifts: array<array<u32, 64>, 3>,
    //axes: vec2<u64>,
    morton_index_scale: f32,
    //bits_per_axis: vec3<u32>,
    offset: vec3<f32>,
    size_multiplier: f32,
    multiplier: vec3<f32>,
};


@group(0) @binding(0)
var<storage, read> uniforms: MortonCodeUniforms;

@group(0) @binding(1)
var<storage, read> vertices: array<Vertex>;

@group(0) @binding(2)
var<storage, read> indices: array<TriangleIndices>;

// TODO: Check if need to change to 'read_write'
@group(0) @binding(3)
var<storage, write> morton_codes: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn morton_code(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let vert_1 = vertices[indices[invocation_id.x].tri_indices.x];
    let vert_2 = vertices[indices[invocation_id.x].tri_indices.y];
    let vert_3 = vertices[indices[invocation_id.x].tri_indices.z];

}