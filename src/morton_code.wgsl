const NUMBER_OF_BITS_FOR_SIZE: u32 = 6u;
// 2 times number of bits
const SIZE_LUT_NUMBER_OF_BITS: u32 = 12u;
// 2^12 = 4096
const SIZE_LUT_NUMBER_OF_POSSIBLE_VALUES: u32 = 4096u;
const SIZE_LUT_NUMBER_OF_POSSIBLE_VALUES_U32: u32 = 8192u;
const SIZE_LUT_NUMBER_OF_POSSIBLE_VALUES_UVEC: u32 = 2048u;

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
    //lut: array<vec4<u32> 1152>, 
    //size_lut: array<vec4<u32>, SIZE_LUT_NUMBER_OF_POSSIBLE_VALUES_UVEC>,

    morton_index_scale: f32,
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
    let add = (byte & 0xFF) * 2;
    return starting_point + add + first_or_last;
}

@compute @workgroup_size(32, 1, 1)
fn morton_code(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
) {
    //let id = invocation_id.x * 8u + invocation_id.y
    let id = invocation_id.y * (arrayLength(&indices) / 8u) + invocation_id.x;
    if id > arrayLength(&indices) {
        return;
    }
    let vert_1: vec3<f32> = vertices[indices[id].tri_indices.x].position;
    let vert_2: vec3<f32> = vertices[indices[id].tri_indices.y].position;
    let vert_3: vec3<f32> = vertices[indices[id].tri_indices.z].position;
    let min = min(min(vert_1, vert_2), vert_3);
    let max = max(max(vert_1, vert_2), vert_3);
    let p: vec3<u32> = vec3<u32>(((min + max) * 0.5 - uniforms.offset) * uniforms.multiplier - 0.001);
    let j = u32(length(max - min) * uniforms.size_multiplier);
    let b: u32 = 0xFFu;

    var morton_lo = 0u;

    for (var i: i32 = 0; i < 3; i += 1) {
        morton_lo |= uniforms.lut[translate_coords_lut(0 + (3 * i), i32((p.x >> (u32(i) * 3u))), 0)];
        morton_lo |= uniforms.lut[translate_coords_lut(1 + (3 * i), i32((p.y >> (u32(i) * 3u))), 0)];
        morton_lo |= uniforms.lut[translate_coords_lut(2 + (3 * i), i32((p.z >> (u32(i) * 3u))), 0)];
    }

    morton_lo |= uniforms.size_lut[i32(j * 2u)];

    var morton_hi = 0u;
    for (var i: i32 = 0; i < 3; i += 1) {
        morton_hi |= uniforms.lut[translate_coords_lut(0 + (3 * i), i32((p.x >> (u32(i) * 3u))), 1)];
        morton_hi |= uniforms.lut[translate_coords_lut(1 + (3 * i), i32((p.y >> (u32(i) * 3u))), 1)];
        morton_hi |= uniforms.lut[translate_coords_lut(2 + (3 * i), i32((p.z >> (u32(i) * 3u))), 1)];
    }

    morton_hi |= uniforms.size_lut[i32(j * 2u) + 1];
    morton_codes[(id * 2u) + 0u] = morton_lo;
    morton_codes[(id * 2u) + 1u] = morton_hi;
}
