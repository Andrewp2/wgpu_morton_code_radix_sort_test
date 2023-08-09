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
    // bone_ids: vec4<u32>,
    // bone_weights: vec4<f32>,
    // albedo: vec3<f32>,
};

struct TriangleIndices {
    tri_indices: vec3<u32>,
    // TODO: Think about this (we can fit one u32 for free, but not 3)
    // node_index: u32,
    // material_id: u32,
    // flags: u32,
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


// note - this is intended to be storage, easy workaround some issues with creating type as a uniform https://github.com/teoxoy/encase/issues/43
@group(0) @binding(0)
var<storage, read> uniforms: MortonCodeUniforms;

@group(0) @binding(1)
var<storage, read> vertices: array<Vertex>;

@group(0) @binding(2)
var<storage, read> indices: array<TriangleIndices>;

@group(0) @binding(3)
var<storage, read_write> morton_codes: array<u32>;

fn translate_coords_lut(i: i32, byte: i32, first_or_last: i32) -> i32 {
    let starting_point = i * 512;
    let add = byte * 2;
    return starting_point + add + first_or_last;
}

@compute @workgroup_size(64, 1, 1)
fn morton_code(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    if invocation_id.x > arrayLength(&indices) {
        return;
    }
    let vert_1: vec3<f32> = vertices[indices[invocation_id.x].tri_indices.x].position;
    let vert_2: vec3<f32> = vertices[indices[invocation_id.x].tri_indices.y].position;
    let vert_3: vec3<f32> = vertices[indices[invocation_id.x].tri_indices.z].position;
    let min = min(min(vert_1, vert_2), vert_3);
    let max = max(max(vert_1, vert_2), vert_3);
    let p: vec3<u32> = vec3<u32>(((min + max) * 0.5 - uniforms.offset) * uniforms.multiplier - 0.1);
    let j = u32(length(max - min) * uniforms.size_multiplier);
    let b: u32 = 0xFFu;

    let morton_lo = uniforms.lut[translate_coords_lut(0, i32(p.x & b), 0)] | uniforms.lut[translate_coords_lut(1, i32(p.y & b), 0)] | uniforms.lut[translate_coords_lut(2, i32(p.z & b), 0)] | uniforms.lut[translate_coords_lut(3, i32((p.x >> 8u) & b), 0)] | uniforms.lut[translate_coords_lut(4, i32((p.y >> 8u) & b), 0)] | uniforms.lut[translate_coords_lut(5, i32((p.z >> 8u) & b), 0)] | uniforms.lut[translate_coords_lut(6, i32((p.x >> 16u) & b), 0)] | uniforms.lut[translate_coords_lut(7, i32((p.y >> 16u) & b), 0)] | uniforms.lut[translate_coords_lut(8, i32((p.z >> 16u) & b), 0)] | uniforms.size_lut[i32(j * 2u)];

    let morton_hi = uniforms.lut[translate_coords_lut(0, i32(p.x & b), 1)] | uniforms.lut[translate_coords_lut(1, i32(p.y & b), 1)] | uniforms.lut[translate_coords_lut(2, i32(p.z & b), 1)] | uniforms.lut[translate_coords_lut(3, i32((p.x >> 8u) & b), 1)] | uniforms.lut[translate_coords_lut(4, i32((p.y >> 8u) & b), 1)] | uniforms.lut[translate_coords_lut(5, i32((p.z >> 8u) & b), 1)] | uniforms.lut[translate_coords_lut(6, i32((p.x >> 16u) & b), 1)] | uniforms.lut[translate_coords_lut(7, i32((p.y >> 16u) & b), 1)] | uniforms.lut[translate_coords_lut(8, i32((p.z >> 16u) & b), 1)] | uniforms.size_lut[i32(j * 2u) + 1];

    morton_codes[(invocation_id.x * 2u) + 0u] = morton_lo;
    morton_codes[(invocation_id.x * 2u) + 1u] = morton_hi;
}
