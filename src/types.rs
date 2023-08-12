use encase::ShaderType;

use glam::UVec3;

use glam::Vec2;

use glam::Vec3;
use wgpu::Buffer;
use wgpu::ShaderModule;

/// A vertex represented by its position, normal, and uv.
#[derive(ShaderType, Copy, Clone, Default)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

/// A list of vertices
#[derive(ShaderType, Clone)]
pub struct Vertices {
    #[size(runtime)]
    pub vertices: Vec<Vertex>,
}

/// A single triangle represented by indices into the vertex list,
/// as well as a node_index for the BVH
/// a material_id for choosing the material,
/// and a flag bitset.
#[derive(ShaderType, Clone, Copy)]
pub struct TriangleIndices {
    pub indices: UVec3,
    // node_index: u32,
    // material_id: u32,
    // flags: u32,
}

/// A list of triangles.
#[derive(ShaderType, Clone)]
pub struct Triangles {
    #[size(runtime)]
    pub triangles: Vec<TriangleIndices>,
}

/// Uniforms for calculating morton code in morton code shader.
#[derive(ShaderType)]
pub struct MortonUniforms {
    pub lut: [u32; 4608],
    pub size_lut: [u32; 1 << (extended_morton_coder::SIZE_LUT_NUMBER_OF_BITS + 1)],
    pub morton_index_scale: f32,
    pub offset: Vec3,
    pub size_multiplier: f32,
    pub multiplier: Vec3,
}

/// Uniforms needed for digit selection in radix sort.
#[derive(ShaderType)]
pub struct RadixUniforms {
    pub pass_number: u32,
}

#[derive(ShaderType)]
pub struct ScatterUniforms {
    pub scatter_step: u32,
    pub number_sections: u32,
}

pub struct Buffers {
    pub vertices: Buffer,
    pub indices: Buffer,
    pub indices_2: Buffer,
    pub morton_uniforms: Buffer,
    pub histogram: Buffer,
    pub morton_codes: Buffer,
    pub morton_codes_2: Buffer,
    pub prefix_sums: Vec<Buffer>,
    pub final_locations: Buffer,
}

pub struct ShaderModules {
    pub morton_code: ShaderModule,
    pub radix_sort_histogram: ShaderModule,
    pub radix_sort_prefix_large: ShaderModule,
    pub radix_sort_prefix_small: ShaderModule,
    pub radix_sort_index: ShaderModule,
    pub radix_sort_scatter: ShaderModule,
}
