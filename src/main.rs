use std::{
    borrow::Cow,
    cmp::PartialEq,
    ops::{Add, Rem, Sub},
};

use encase::{private::WriteInto, ShaderType, StorageBuffer, UniformBuffer};
use extended_morton_coder::MortonCodeGenerator;
use glam::{UVec3, Vec2, Vec3};
use rand::{distributions::Uniform, prelude::Distribution, Rng};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroupEntry, BindingType, BufferBindingType, BufferDescriptor,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, Device, Instance,
    InstanceDescriptor, PipelineLayoutDescriptor, PowerPreference, RequestAdapterOptions,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, *,
};

const NUM_VERTICES: u32 = 500000u32;
const NUM_TRIANGLES: u32 = 1500000u32;
const BITS_PER_PASS: u32 = 6;
const HISTOGRAM_SIZE: u32 = 1 << BITS_PER_PASS;
const WORKGROUP_SIZE: u32 = 256u32;
const ITEMS_PROCESSED_PER_LANE_HISTOGRAM_PASS: u32 = 1u32;
const ITEMS_PER_HISTOGRAM_PASS: u32 = WORKGROUP_SIZE * ITEMS_PROCESSED_PER_LANE_HISTOGRAM_PASS;
const ITEMS_PER_LARGE_PREFIX: u32 = 256u32;
const MAX_ITEMS_IN_SMALL_PREFIX: u32 = 1024u32;

const RNG_SEED: u64 = 7;

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
    indices: UVec3,
    node_index: u32,
    material_id: u32,
    flags: u32,
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
    lut: [u32; 4608],
    size_lut: [u32; 1 << (extended_morton_coder::SIZE_LUT_NUMBER_OF_BITS + 1)],
    morton_index_scale: f32,
    offset: Vec3,
    size_multiplier: f32,
    multiplier: Vec3,
}

/// Uniforms needed for digit selection in radix sort.
#[derive(ShaderType)]
pub struct RadixUniforms {
    pass_number: u32,
}

/// Creates a GPU buffer from type T.
fn create_buffer<T: ShaderType + WriteInto>(
    t: T,
    device: &Device,
    label_str: &str,
    usage: BufferUsages,
) -> Buffer {
    let buf: Vec<u8> = Vec::new();
    // This looks strange, but is actually the way Bevy internally calculates its buffers.
    let mut x: StorageBuffer<Vec<u8>> = StorageBuffer::new(buf);
    x.write(&t).unwrap();
    let final_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label_str),
        contents: x.as_ref(),
        usage,
    });
    final_buffer
}

/// Divides l by r, and takes the ceiling instead of the floor.
fn div_ceil_u32(l: u32, r: u32) -> u32 {
    return (l + r - 1) / r;
}

/// Divids l by r and takes the ceiling instead of the floor.
fn div_ceil_u64(l: u64, r: u64) -> u64 {
    return (l + r - 1) / r;
}

/// Calculates the number of workgroups needed to handle all the elements.
fn calculate_number_of_workgroups_u32(element_count: u32, workgroup_size: u32) -> u32 {
    return div_ceil_u32(element_count, workgroup_size);
}

// Calculates the number of workgroups needed to handle all the elements.
fn calculate_number_of_workgroups_u64(element_count: u64, workgroup_size: u64) -> u64 {
    return div_ceil_u64(element_count, workgroup_size);
}

/// Calculates the next multiple of a u32 number, such that l % r == 0.
fn round_up_u32<T: Rem<Output = T> + PartialEq<u32> + Add<Output = T> + Sub<Output = T> + Copy>(
    l: T,
    r: T,
) -> T {
    let remainder = l % r;
    if remainder == 0u32 {
        return l;
    }
    return l + r - remainder;
}

/// Calculates the next multiple of a u64 number, such that l % r == 0.
fn round_up_u64<T: Rem<Output = T> + PartialEq<u64> + Add<Output = T> + Sub<Output = T> + Copy>(
    l: T,
    r: T,
) -> T {
    let remainder = l % r;
    if remainder == 0u64 {
        return l;
    }
    return l + r - remainder;
}

fn copy_buffer_to_buffer(encoder: &mut CommandEncoder, source: &Buffer, destination: &Buffer) {
    encoder.copy_buffer_to_buffer(source, 0, destination, 0, source.size());
}

/// Creates a "scene" of vertices and indices (each triplet of indices is a triangle) into those triangles,
/// as well as the morton code generator needed to morton code those triangles.
/// Produces 2 copies of vertices and triangles in order to test CPU implementation vs GPU implementation.
fn create_scene() -> (
    Vertices,
    Triangles,
    MortonCodeGenerator,
    Vertices,
    Triangles,
) {
    let mut vertices: Vertices = Vertices {
        vertices: Vec::with_capacity(NUM_VERTICES as usize),
    };
    let mut rng_gen = ChaCha8Rng::seed_from_u64(RNG_SEED);
    let mut scene_min = Vec3::splat(f32::MAX);
    let mut scene_max = Vec3::splat(f32::MIN);
    for _ in 0..NUM_VERTICES {
        let position = Vec3::new(rng_gen.gen(), rng_gen.gen(), rng_gen.gen());
        // don't min or max in the case that this vertex is never used in scene
        //scene_min = scene_min.min(position);
        //scene_max = scene_max.max(position);
        vertices.vertices.push(Vertex {
            position,
            normal: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::new(0.0, 0.5),
        });
    }
    let mut triangles: Triangles = Triangles {
        triangles: Vec::with_capacity(NUM_TRIANGLES as usize),
    };
    let rng_distribution: Uniform<u32> = Uniform::new(0, NUM_VERTICES as u32);
    for _ in 0..NUM_TRIANGLES {
        let indices = UVec3 {
            x: rng_distribution.sample(&mut rng_gen),
            y: rng_distribution.sample(&mut rng_gen),
            z: rng_distribution.sample(&mut rng_gen),
        };
        scene_min = scene_min.min(vertices.vertices[indices.x as usize].position);
        scene_min = scene_min.min(vertices.vertices[indices.y as usize].position);
        scene_min = scene_min.min(vertices.vertices[indices.z as usize].position);

        scene_max = scene_max.max(vertices.vertices[indices.x as usize].position);
        scene_max = scene_max.max(vertices.vertices[indices.y as usize].position);
        scene_max = scene_max.max(vertices.vertices[indices.z as usize].position);
        triangles.triangles.push(TriangleIndices {
            indices,
            node_index: 0,
            material_id: rng_gen.gen_range(0..5),
            flags: 0,
        })
    }

    let triangles_copy = triangles.clone();
    let vertices_copy = vertices.clone();
    let morton_code_generator = MortonCodeGenerator::new(scene_min, scene_max);
    (
        vertices,
        triangles,
        morton_code_generator,
        vertices_copy,
        triangles_copy,
    )
}

/// Creates the morton uniforms needed for the shader from the MortonCodeGenerator produced on the CPU.
/// Need this function in order to flatten the 2d arrays into 1d arrays, and to convert from u64 to 2x the u32s.
fn create_morton_uniforms(morton_code_generator: &MortonCodeGenerator) -> MortonUniforms {
    let lut: Vec<u32> = morton_code_generator
        .lut
        .iter()
        .flat_map(|x| x.iter().flat_map(|y| [(*y >> 0) as u32, (*y >> 32) as u32]))
        .collect();
    assert!(lut.len() == 4608, "lut is the wrong length");
    let size_lut: Vec<u32> = morton_code_generator
        .size_lut
        .iter()
        .flat_map(|x| [(*x >> 0) as u32, (*x >> 32) as u32])
        .collect();
    assert!(size_lut.len() == 8192);
    MortonUniforms {
        lut: lut.try_into().unwrap(),
        size_lut: size_lut.try_into().unwrap(),
        morton_index_scale: morton_code_generator.morton_index_scale,
        offset: morton_code_generator.offset,
        size_multiplier: morton_code_generator.size_multiplier,
        multiplier: morton_code_generator.multiplier,
    }
}

/// Calculates the size and number of large prefix buffers needed to radix sort that number of elements.
fn calculate_num_items_prefix_buffers(num_elements: u64) -> Vec<u64> {
    // Multiply by number of values per radix pass because on the first pass we produce a histogram
    // of size NUM_VALS_IN_RADIX_PASS.
    // That means that technically we're dividing by a factor of 4.
    let mut num_items_in_buffer: u64 =
        calculate_number_of_workgroups_u64(num_elements, ITEMS_PER_HISTOGRAM_PASS as u64)
            * HISTOGRAM_SIZE as u64;
    // we need to round up to the nearest 256 as the small prefix pass assumes that a buffer size is always a multiple of 256
    num_items_in_buffer = round_up_u64(num_items_in_buffer, ITEMS_PER_LARGE_PREFIX as u64);
    if num_items_in_buffer <= MAX_ITEMS_IN_SMALL_PREFIX as u64 {
        // In this case, we don't need to create any prefix buffers as we can directly perform a small prefix sum on
        // the histogram buffer.
        // println!(
        //     "no prefix buffers necessary, as number of workgroups is {}, and total size is {}",
        //     calculate_number_of_workgroups_u64(num_elements, ITEMS_PER_HISTOGRAM_PASS as u64),
        //     num_items_in_buffer
        // );
        return vec![];
    }
    num_items_in_buffer /= 256;
    num_items_in_buffer = round_up_u64(num_items_in_buffer, ITEMS_PER_LARGE_PREFIX as u64);
    let mut num_items_in_buffers = vec![];
    // While the number of elements we're processing is larger than the maximum we can handle in a small prefix pass
    while num_items_in_buffer > MAX_ITEMS_IN_SMALL_PREFIX as u64 {
        //println!("number of elements in buffer {}", num_items_in_buffer);
        // Do another large prefix pass
        num_items_in_buffers.push(num_items_in_buffer);
        // Divide the number of elements by the number of elements reduced in a single large prefix workgroup
        // This is guaranteed to produce an integer, as we already rounded up to the next multiple of ITEMS_PER_LARGE_PREFIX.
        num_items_in_buffer /= ITEMS_PER_LARGE_PREFIX as u64;
        // Round up the number of elements to the next 256 to not break anything
        num_items_in_buffer = round_up_u64(num_items_in_buffer, ITEMS_PER_LARGE_PREFIX as u64);
    }
    //println!("num_items_in_large_buffers: {:?}", num_items_in_buffers);
    // Now that we have all the large prefix buffers, let's create the small one. We know there will always be at least one
    // if we got this point.
    // We might have 0 large prefix buffers and 1 small, 1 large 1 small, N large 1 small
    num_items_in_buffers.push(num_items_in_buffer);
    //println!("all prefix buffers: {:?}", num_items_in_buffers);
    return num_items_in_buffers;
}

/// Creates buffers for prefix sum in radix sort.
fn create_prefix_buffers(device: &Device) -> Vec<Buffer> {
    let mut prefix_sum_buffers: Vec<Buffer> = vec![];
    // This is the number of elements in LARGE buffers. It doesn't include the last number of elements
    let buffers_num_items = calculate_num_items_prefix_buffers(NUM_TRIANGLES as u64);
    // size of a u32 is 4 bytes
    let bytes_per_element = 4u64;
    // Create all the buffers
    for number_of_items_in_buffer in &buffers_num_items {
        prefix_sum_buffers.push(device.create_buffer(&BufferDescriptor {
            label: Some(format!("prefix sum buffer {number_of_items_in_buffer}").as_str()),
            size: number_of_items_in_buffer * bytes_per_element,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    prefix_sum_buffers
}

// region: create bind group layouts
fn create_morton_code_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("bind group layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_radix_bgl_histogram(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Radix bind group layout original histograms"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_radix_bgl_prefix_large(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Radix bind group layout large prefix sum"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_radix_bgl_prefix_small(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Radix bind group layout small prefix sum"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn create_radix_bgl_scatter(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Radix bind group layout scatter"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

// endregion

/// Creates a bind group given a certain layout, a list of buffers, and a name.
fn create_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    buffers: Vec<&Buffer>,
    label: &str,
) -> BindGroup {
    let entries: Vec<BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|x| BindGroupEntry {
            binding: x.0 as u32,
            resource: x.1.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&BindGroupDescriptor {
        label: Some(label),
        layout: layout,
        entries: &entries,
    })
}

async fn run() {
    env_logger::init();
    #[cfg(histogram_readback)]
    log::info!("histogram_readback");
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        dx12_shader_compiler: Default::default(),
    });
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .unwrap();
    let info = adapter.get_info();
    log::info!("backend: {:?}", info.backend);
    device.on_uncaptured_error(Box::new(move |error| {
        log::error!("{}", &error);
        panic!(
            "wgpu error (handling all wgpu errors as fatal):\n{:?}\n{:?}",
            &error, &info,
        );
    }));
    let number_of_workgroups =
        calculate_number_of_workgroups_u32(NUM_TRIANGLES, ITEMS_PER_HISTOGRAM_PASS);
    // region: create triangles and morton code generator
    #[cfg(morton_code_readback)]
    let (vertices, triangles, morton_code_generator, vertices_copy, triangles_copy) =
        create_scene();
    #[cfg(not(morton_code_readback))]
    let (vertices, triangles, morton_code_generator, _vertices_copy, _triangles_copy) =
        create_scene();
    assert_eq!(triangles.triangles.len(), NUM_TRIANGLES as usize);
    // endregion
    // region: creating buffers
    let morton_uniforms = create_morton_uniforms(&morton_code_generator);
    let morton_uniforms_b = create_buffer(
        morton_uniforms,
        &device,
        "morton uniform buffer",
        BufferUsages::STORAGE,
    );

    let histogram_buffer_number_elements = round_up_u32(number_of_workgroups * HISTOGRAM_SIZE, 256);
    //println!(
    //     "histogram buffer num elements {}",
    //     histogram_buffer_number_elements
    // );

    let histogram_b = device.create_buffer(&BufferDescriptor {
        label: Some("original histogram buffer"),
        size: (histogram_buffer_number_elements * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let prefix_sum_bs = create_prefix_buffers(&device);
    let vertices_b = create_buffer(vertices, &device, "vertices buffer", BufferUsages::STORAGE);
    let indices_b = create_buffer(triangles, &device, "indices buffer", BufferUsages::STORAGE);
    let indices_2_b = device.create_buffer(&BufferDescriptor {
        label: Some("second indices buffer"),
        size: indices_b.size(),
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let morton_code_size: u64 = (std::mem::size_of::<u64>()).try_into().unwrap();
    // 4000 u32s
    let morton_code_b_size = morton_code_size * NUM_TRIANGLES as u64;
    let morton_code_b = device.create_buffer(&BufferDescriptor {
        label: Some("morton code buffer"),
        size: morton_code_b_size,
        usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let morton_code_2_b = device.create_buffer(&BufferDescriptor {
        label: Some("morton code buffer 2"),
        size: morton_code_b_size,
        usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    #[cfg(any(morton_code_readback, histogram_readback))]
    let morton_code_readback_b = device.create_buffer(&BufferDescriptor {
        label: Some("morton code readback buffer"),
        size: morton_code_b_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    #[cfg(radix_sort_readback)]
    let radix_sort_morton_codes_readback_b = device.create_buffer(&BufferDescriptor {
        label: Some("radix_sort_morton_codes_readback_b"),
        size: morton_code_b_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    #[cfg(radix_sort_readback)]
    let radix_sort_morton_codes_2_readback_b = device.create_buffer(&BufferDescriptor {
        label: Some("radix_sort_morton_codes_2_readback_b"),
        size: morton_code_b_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    #[cfg(any(histogram_readback, prefix_sum_readback))]
    let histogram_readback_b = device.create_buffer(&BufferDescriptor {
        label: Some("histogram_readback_b"),
        size: histogram_b.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    #[cfg(prefix_sum_readback)]
    let prefix_sum_readback_b = device.create_buffer(&BufferDescriptor {
        label: Some("prefix_sum_readback_b"),
        size: histogram_b.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    device.start_capture();

    // endregion
    // region: shader modules
    let morton_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("morton_module"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("morton_code.wgsl"))),
    });

    let radix_sort_histogram_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("radix_sort_histogram_module"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("radix_sort.wgsl"))),
    });

    let radix_sort_prefix_large_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("radix_sort_prefix_large_module"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "radix_sort_block_sum_large.wgsl"
        ))),
    });

    let radix_sort_prefix_small_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("radix_sort_prefix_small_module"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "radix_sort_block_sum_small.wgsl"
        ))),
    });

    let radix_sort_scatter_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("radix_sort_scatter_module"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("radix_sort_scatter.wgsl"))),
    });

    // endregion
    // region: bind group layouts
    let morton_code_l = create_morton_code_bind_group_layout(&device);
    let radix_histogram_l = create_radix_bgl_histogram(&device);
    let radix_prefix_large_l = create_radix_bgl_prefix_large(&device);
    let radix_prefix_small_l = create_radix_bgl_prefix_small(&device);
    let radix_scatter_l = create_radix_bgl_scatter(&device);
    // endregion
    // region: compute pipelines
    let morton_code_p = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("morton code compute pipeline"),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&morton_code_l],
            push_constant_ranges: &[],
        })),
        module: &morton_module,
        entry_point: "morton_code",
    });

    let radix_histogram_p = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("radix sort histogram compute pipeline"),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&radix_histogram_l],
            push_constant_ranges: &[],
        })),
        module: &radix_sort_histogram_module,
        entry_point: "radix_sort_compute_histogram",
    });

    let radix_prefix_large_p = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("radix prefix large pipeline"),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&radix_prefix_large_l],
            push_constant_ranges: &[],
        })),
        module: &radix_sort_prefix_large_module,
        entry_point: "radix_sort_block_sum_large",
    });

    let radix_prefix_large_p_2 = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("radix prefix large pipeline 2"),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&radix_prefix_large_l],
            push_constant_ranges: &[],
        })),
        module: &radix_sort_prefix_large_module,
        entry_point: "radix_sort_block_sum_large_after",
    });

    let radix_prefix_small_p = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("radix sort prefix sum small compute pipeline"),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&radix_prefix_small_l],
            push_constant_ranges: &[],
        })),
        module: &radix_sort_prefix_small_module,
        entry_point: "radix_sort_block_sum_small",
    });

    let radix_scatter_p = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("morton code compute pipeline"),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&radix_scatter_l],
            push_constant_ranges: &[],
        })),
        module: &radix_sort_scatter_module,
        entry_point: "radix_sort_scatter",
    });

    // endregion
    // region: encoder
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("command encoder"),
    });

    let morton_code_bg: BindGroup = create_bind_group(
        &device,
        &morton_code_l,
        vec![&morton_uniforms_b, &vertices_b, &indices_b, &morton_code_b],
        "morton code bind group",
    );

    {
        let compute_pass_desc = ComputePassDescriptor {
            label: Some("Compute Pass"),
        };
        let mut compute_pass = encoder.begin_compute_pass(&compute_pass_desc);
        compute_pass.set_pipeline(&morton_code_p);
        compute_pass.set_bind_group(0, &morton_code_bg, &[]);
        compute_pass.insert_debug_marker("compute morton code");
        // morton code dispatch is only 64 wide, not a full 256.
        compute_pass.dispatch_workgroups(div_ceil_u32(NUM_TRIANGLES, 64), 1, 1);
    }

    for i in 0..11 {
        let radix_uniforms_b = create_radix_uniforms_buffer(&device, i);
        // region: bind groups
        let radix_histogram_bg = create_bind_group(
            &device,
            &radix_histogram_l,
            vec![
                &radix_uniforms_b,
                &indices_b,
                &morton_code_b,
                &indices_2_b,
                &morton_code_2_b,
                &histogram_b,
            ],
            "radix sort histogram bind group",
        );

        let mut radix_prefix_large_bind_groups = vec![];

        if !prefix_sum_bs.is_empty() {
            log::info!("prefix_sum_bs was not empty");
            radix_prefix_large_bind_groups.push(create_bind_group(
                &device,
                &radix_prefix_large_l,
                vec![&histogram_b, &prefix_sum_bs[0]],
                "radix sort prefix sum large bind group",
            ));
            for i in 0..prefix_sum_bs.len() - 1 {
                radix_prefix_large_bind_groups.push(create_bind_group(
                    &device,
                    &create_radix_bgl_prefix_large(&device),
                    vec![&prefix_sum_bs[i], &prefix_sum_bs[i + 1]],
                    "radix sort prefix sum large bind group",
                ));
            }
        } else {
            log::info!("We don't have any prefix sum buffers so we don't need any bindgroups to them, but we still need to create a small bind group on the histogram");
        }

        let radix_prefix_small_bg = match prefix_sum_bs.is_empty() {
            true => {
                log::info!("Create a small bind group directly on the histogram buffer");
                create_bind_group(
                    &device,
                    &radix_prefix_small_l,
                    vec![&histogram_b],
                    "radix sort prefix sum small bind group",
                )
            }
            false => {
                //println!("doing the bind group on a small prefix buffer instead of histogram");
                create_bind_group(
                    &device,
                    &radix_prefix_small_l,
                    vec![&prefix_sum_bs.last().unwrap()],
                    "radix sort prefix sum small bind group",
                )
            }
        };

        let radix_scatter_bg = create_bind_group(
            &device,
            &radix_scatter_l,
            vec![
                &radix_uniforms_b,
                &indices_b,
                &morton_code_b,
                &indices_2_b,
                &morton_code_2_b,
                &histogram_b,
            ],
            "radix sort scatter bind group",
        );
        // endregion
        {
            let compute_pass_desc = ComputePassDescriptor {
                label: Some("Compute Pass 2"),
            };
            let mut compute_pass = encoder.begin_compute_pass(&compute_pass_desc);
            compute_pass.set_pipeline(&radix_histogram_p);
            compute_pass.set_bind_group(0, &radix_histogram_bg, &[]);
            compute_pass.insert_debug_marker("histogram pass");
            compute_pass.dispatch_workgroups(number_of_workgroups, 1, 1);
        }

        #[cfg(prefix_sum_readback)]
        copy_buffer_to_buffer(&mut encoder, &histogram_b, &histogram_readback_b);

        {
            let compute_pass_desc = ComputePassDescriptor {
                label: Some("Compute Pass 3"),
            };
            let mut compute_pass = encoder.begin_compute_pass(&compute_pass_desc);
            let buffer_num_items = calculate_num_items_prefix_buffers(NUM_TRIANGLES as u64);
            compute_pass.set_pipeline(&radix_prefix_large_p);
            assert!(buffer_num_items.len() == radix_prefix_large_bind_groups.len());

            for (i, large_bg) in radix_prefix_large_bind_groups.iter().enumerate() {
                compute_pass.insert_debug_marker("large prefix");
                compute_pass.set_bind_group(0, &large_bg, &[]);
                // let mut prefix_workgroup = 0;

                let prefix_workgroup = match i {
                    0 => div_ceil_u64(
                        histogram_buffer_number_elements as u64,
                        ITEMS_PER_LARGE_PREFIX as u64,
                    ),
                    _ => div_ceil_u64(buffer_num_items[i - 1], ITEMS_PER_LARGE_PREFIX as u64),
                };
                //println!(
                //     "id: {}, number of workgroups dispatched: {}",
                //     i, prefix_workgroup
                // );
                compute_pass.dispatch_workgroups(prefix_workgroup as u32, 1, 1);
            }

            log::info!("run the small prefix bind group");
            compute_pass.set_pipeline(&radix_prefix_small_p);
            compute_pass.set_bind_group(0, &radix_prefix_small_bg, &[]);
            compute_pass.insert_debug_marker("small prefix");
            compute_pass.dispatch_workgroups(1, 1, 1);

            compute_pass.set_pipeline(&radix_prefix_large_p_2);

            for i in (0..radix_prefix_large_bind_groups.len()).rev() {
                let index = i;
                let bind_group = &radix_prefix_large_bind_groups[index];
                compute_pass.insert_debug_marker("large prefix 2");
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let prefix_workgroup = match i {
                    0 => div_ceil_u64(
                        histogram_buffer_number_elements as u64,
                        ITEMS_PER_LARGE_PREFIX as u64,
                    ),
                    _ => div_ceil_u64(buffer_num_items[i - 1], ITEMS_PER_LARGE_PREFIX as u64),
                };
                //println!(
                //     "id: {}, run large prefix bind group cleanup pass: {}",
                //     index, prefix_workgroup
                // );
                compute_pass.dispatch_workgroups(prefix_workgroup as u32, 1, 1);
            }

            log::info!("scatter final results");
            compute_pass.set_pipeline(&radix_scatter_p);
            compute_pass.set_bind_group(0, &radix_scatter_bg, &[]);
            compute_pass.insert_debug_marker("scatter");
            compute_pass.dispatch_workgroups(number_of_workgroups, 1, 1);
        }
    }
    #[cfg(morton_code_readback)]
    copy_buffer_to_buffer(&mut encoder, &morton_code_b, &morton_code_readback_b);
    #[cfg(histogram_readback)]
    {
        copy_buffer_to_buffer(&mut encoder, &morton_code_b, &morton_code_readback_b);
        copy_buffer_to_buffer(&mut encoder, &histogram_b, &histogram_readback_b);
    }
    #[cfg(radix_sort_readback)]
    copy_buffer_to_buffer(
        &mut encoder,
        &morton_code_b,
        &radix_sort_morton_codes_readback_b,
    );
    #[cfg(radix_sort_readback)]
    copy_buffer_to_buffer(
        &mut encoder,
        &morton_code_2_b,
        &radix_sort_morton_codes_2_readback_b,
    );

    #[cfg(prefix_sum_readback)]
    copy_buffer_to_buffer(&mut encoder, &histogram_b, &prefix_sum_readback_b);

    // endregion
    log::info!("submit");
    queue.submit(Some(encoder.finish()));

    device.stop_capture();
    // We need to scope the mapping variables so that we can
    // unmap the buffer
    #[cfg(morton_code_readback)]
    {
        let morton_code_readback_slice = morton_code_readback_b.slice(..);
        // NOTE: We have to create the mapping THEN device.poll() before await
        // the future. Otherwise the application will freeze.
        let (morton_code_tx, morton_code_rx) =
            futures_intrusive::channel::shared::oneshot_channel();
        morton_code_readback_slice.map_async(MapMode::Read, move |result| {
            morton_code_tx.send(result).unwrap();
        });
        device.poll(Maintain::Wait);
        if let Some(Ok(())) = morton_code_rx.receive().await {
            let morton_code_data = morton_code_readback_slice.get_mapped_range();
            let morton_codes: Vec<u64> = bytemuck::cast_slice(&morton_code_data).to_vec();
            drop(morton_code_data);
            morton_code_readback_b.unmap();
            for (i, compute_code) in morton_codes.iter().enumerate() {
                let indices = triangles_copy.triangles[i];
                let min = vertices_copy.vertices[indices.indices.x as usize]
                    .position
                    .min(
                        vertices_copy.vertices[indices.indices.y as usize]
                            .position
                            .min(vertices_copy.vertices[indices.indices.z as usize].position),
                    );
                let max = vertices_copy.vertices[indices.indices.x as usize]
                    .position
                    .max(
                        vertices_copy.vertices[indices.indices.y as usize]
                            .position
                            .max(vertices_copy.vertices[indices.indices.z as usize].position),
                    );
                let expected = morton_code_generator.code(min, max);
                if *compute_code != expected {
                    if compute_code ^ expected > 1000u64 {
                        //println!(
                        //     "compute code and real code are not close {}\n{:#066b}\n{:#066b}\n{:#066b}\n{}",
                        //     i,
                        //     compute_code,
                        //     expected,
                        //     compute_code ^ expected,
                        //     compute_code ^ expected,
                        // );
                    }
                }
            }
        } else {
            panic!("failed to run compute on GPU!!!!!");
        }
    }
    #[cfg(histogram_readback)]
    {
        let histogram_readback_slice = histogram_readback_b.slice(..);
        let morton_code_readback_slice = morton_code_readback_b.slice(..);
        // NOTE: We have to create the mapping THEN device.poll() before await
        // the future. Otherwise the application will freeze.
        let (histogram_tx, histogram_rx) = futures_intrusive::channel::shared::oneshot_channel();
        let (morton_code_tx, morton_code_rx) =
            futures_intrusive::channel::shared::oneshot_channel();
        histogram_readback_slice.map_async(MapMode::Read, move |result| {
            histogram_tx.send(result).unwrap();
        });
        morton_code_readback_slice.map_async(MapMode::Read, move |result| {
            morton_code_tx.send(result).unwrap();
        });
        device.poll(Maintain::Wait);
        if let (Some(Ok(())), Some(Ok(()))) =
            (histogram_rx.receive().await, morton_code_rx.receive().await)
        {
            let morton_code_data = morton_code_readback_slice.get_mapped_range();
            let morton_codes: Vec<u64> = bytemuck::cast_slice(&morton_code_data).to_vec();
            drop(morton_code_data);
            morton_code_readback_b.unmap();
            let histogram_data = histogram_readback_slice.get_mapped_range();
            let histogram_vals: Vec<u32> = bytemuck::cast_slice(&histogram_data).to_vec();
            drop(histogram_data);
            histogram_readback_b.unmap();

            let mut full_manual_histogram: Vec<u32> =
                vec![0; (number_of_workgroups * HISTOGRAM_SIZE) as usize];
            assert_eq!(
                (number_of_workgroups * HISTOGRAM_SIZE) as usize,
                histogram_vals.len()
            );

            for (chunk_id, morton_codes_workgroup) in morton_codes.chunks(256).enumerate() {
                let mut manual_histogram: Vec<u32> = vec![0; 64];
                for (i, mc) in morton_codes_workgroup.iter().enumerate() {
                    let digit = mc & 63;
                    //println!("digit: {:#08b}, chunk_id: {}, wid: {}", digit, chunk_id, i);
                    manual_histogram[digit as usize] += 1;
                }
                for (i, h) in manual_histogram.iter().enumerate() {
                    full_manual_histogram[i * (number_of_workgroups as usize) + chunk_id] = *h;
                }
            }
            for (i, histogram_val) in full_manual_histogram.iter().enumerate() {
                let compute_value = histogram_vals[i];
                assert_eq!(*histogram_val, compute_value);
                // print!("{}, {}\t", histogram_val, compute_value);
                // if (i as u32) % number_of_workgroups == 7 {
                //println!("");
                // }
            }
        } else {
            panic!("compute didn't run properly");
        }
    }
    #[cfg(prefix_sum_readback)]
    {
        let histogram_readback_slice = histogram_readback_b.slice(..);
        let prefix_sum_readback_slice = prefix_sum_readback_b.slice(..);
        // NOTE: We have to create the mapping THEN device.poll() before await
        // the future. Otherwise the application will freeze.
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        prefix_sum_readback_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let (histogram_tx, histogram_rx) = futures_intrusive::channel::shared::oneshot_channel();
        histogram_readback_slice.map_async(MapMode::Read, move |result| {
            histogram_tx.send(result).unwrap();
        });
        device.poll(Maintain::Wait);
        if let (Some(Ok(())), Some(Ok(()))) = (rx.receive().await, histogram_rx.receive().await) {
            let prefix_sum_readback_data = prefix_sum_readback_slice.get_mapped_range();
            let gpu_prefix_summed_histogram: Vec<u32> =
                bytemuck::cast_slice(&prefix_sum_readback_data).to_vec();
            drop(prefix_sum_readback_data);
            prefix_sum_readback_b.unmap();
            let histogram_data = histogram_readback_slice.get_mapped_range();
            let gpu_histogram_vals: Vec<u32> = bytemuck::cast_slice(&histogram_data).to_vec();
            drop(histogram_data);
            histogram_readback_b.unmap();

            let cpu_prefix_gpu_histogram_vals: Vec<u64> = gpu_histogram_vals
                .iter()
                .map(|x| *x as u64)
                .scan(0, |sum, i| {
                    *sum += i;
                    Some(*sum - i)
                })
                .collect::<Vec<u64>>();
            assert_eq!(
                cpu_prefix_gpu_histogram_vals.len(),
                gpu_prefix_summed_histogram.len()
            );
            for (i, gpu_prefix_sum_histogram_val) in gpu_prefix_summed_histogram.iter().enumerate()
            {
                print!(
                    "gpu: {:5}, cpu: {:5}\n",
                    gpu_prefix_sum_histogram_val, cpu_prefix_gpu_histogram_vals[i]
                );
                if *gpu_prefix_sum_histogram_val != (cpu_prefix_gpu_histogram_vals[i] as u32) {
                    //println!("not correct!!!!");
                }
                if (i as u32) % number_of_workgroups == 7 {
                    //println!("");
                }
                if (i as u32) % 256 == 255 {
                    //println!("-------");
                }
            }
        } else {
            panic!("failed to run compute on GPU!!!!!");
        }
    }
    //need both buffers for morton codes, as well as histogram
    #[cfg(radix_sort_readback)]
    {
        let radix_sort_morton_codes_2_readback_slice =
            radix_sort_morton_codes_2_readback_b.slice(..);
        // NOTE: We have to create the mapping THEN device.poll() before await
        // the future. Otherwise the application will freeze.
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        radix_sort_morton_codes_2_readback_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let radix_sort_morton_codes_readback_slice = radix_sort_morton_codes_readback_b.slice(..);
        // NOTE: We have to create the mapping THEN device.poll() before await
        // the future. Otherwise the application will freeze.
        let (tx2, rx2) = futures_intrusive::channel::shared::oneshot_channel();
        radix_sort_morton_codes_readback_slice.map_async(MapMode::Read, move |result| {
            tx2.send(result).unwrap();
        });

        device.poll(Maintain::Wait);
        if let (Some(Ok(())), Some(Ok(()))) = (rx.receive().await, rx2.receive().await) {
            let radix_sort_morton_codes_2_readback_data =
                radix_sort_morton_codes_2_readback_slice.get_mapped_range();
            let radix_sort_morton_codes_2: Vec<u64> =
                bytemuck::cast_slice(&radix_sort_morton_codes_2_readback_data).to_vec();
            drop(radix_sort_morton_codes_2_readback_data);
            radix_sort_morton_codes_2_readback_b.unmap();

            let radix_sort_morton_codes_readback_data =
                radix_sort_morton_codes_readback_slice.get_mapped_range();
            let radix_sort_morton_codes: Vec<u64> =
                bytemuck::cast_slice(&radix_sort_morton_codes_readback_data).to_vec();
            drop(radix_sort_morton_codes_readback_data);
            radix_sort_morton_codes_readback_b.unmap();

            let mut full_manual_histogram: Vec<u32> =
                vec![0; (number_of_workgroups * HISTOGRAM_SIZE) as usize];

            for (chunk_id, morton_codes_workgroup) in
                radix_sort_morton_codes.chunks(256).enumerate()
            {
                let mut manual_histogram: Vec<u32> = vec![0; 64];
                for (i, mc) in morton_codes_workgroup.iter().enumerate() {
                    let digit = mc & 63;
                    //println!("digit: {:#08b}, chunk_id: {}, wid: {}", digit, chunk_id, i);
                    manual_histogram[digit as usize] += 1;
                }
                for (i, h) in manual_histogram.iter().enumerate() {
                    full_manual_histogram[i * (number_of_workgroups as usize) + chunk_id] = *h;
                }
            }

            let prefix_summed_histogram_vals = full_manual_histogram
                .iter()
                .scan(0, |sum, i| {
                    *sum += i;
                    Some(*sum - i)
                })
                .collect::<Vec<u32>>();

            // log::info!("results length: {}", radix_sort_morton_codes.len());
            // for x in radix_sort_morton_codes.chunks(256) {
            //     for (i, c) in x.iter().enumerate() {}
            // }

            let gpu_codes: Vec<&[u64]> = radix_sort_morton_codes_2.chunks(256).collect();

            let mut sorted_codes = radix_sort_morton_codes.clone();
            sorted_codes.sort_by_key(|x| x & 63);

            let mut cpu_sort: Vec<u64> = vec![0; NUM_TRIANGLES as usize];

            // for (chunk_id, chunk) in radix_sort_morton_codes.chunks(256).enumerate() {
            //println!("-------------------------------------------------------");
            //println!("chunk_id: {}", chunk_id);
            //     let digits: Vec<u32> = chunk.iter().map(|x| (x & (63 as u64)) as u32).collect();
            //     let mut indexed_digits: Vec<(usize, &u32)> = digits.iter().enumerate().collect();
            //     indexed_digits.sort_by_key(|&(_, num)| num);
            //     let mut map: Vec<usize> = vec![0; digits.len()];
            //     for (new_idx, &(original_idx, _)) in indexed_digits.iter().enumerate() {
            //         map[original_idx] = new_idx;
            //     }
            //     let sorted_data: Vec<u32> = indexed_digits.iter().map(|&(_, num)| *num).collect();
            //     let original_indices: Vec<usize> =
            //         indexed_digits.iter().map(|&(idx, _)| idx).collect();
            //     let mut local_histogram: Vec<u32> = vec![0; 64];
            //     digits
            //         .iter()
            //         .for_each(|x| local_histogram[*x as usize] += 1);
            //     let local_prefix_sum: Vec<u32> = local_histogram
            //         .iter()
            //         .scan(0u32, |state, elem| {
            //             *state += *elem;
            //             Some(*state - *elem)
            //         })
            //         .collect();

            //     let mut secondary_local_prefix_sum = vec![0u32; 64];

            //     for thread_lane in 0..64 {
            //         secondary_local_prefix_sum[thread_lane] = local_histogram[thread_lane];
            //     }
            //     let mut i = 1u32;
            //     while i < 64 {
            //         for wid in 0..64 {
            //             if wid % (2 * i) == 0 {
            //                 let first = (wid + (2 * i) - 1) as usize;
            //                 let second = (wid + i - 1) as usize;
            //                 secondary_local_prefix_sum[first] += secondary_local_prefix_sum[second];
            //             }
            //         }
            //         //println!("");
            //         i <<= 1;
            //     }

            //println!("{:?}", secondary_local_prefix_sum);

            //     for wid in 0..64 {
            //         if wid % 64 == 0 {
            //             let before = secondary_local_prefix_sum[32 - 1];
            //             secondary_local_prefix_sum[(32 - 1) as usize] = 0;
            //             secondary_local_prefix_sum[(64 - 1) as usize] = before;
            //         }
            //     }

            //     let mut i = 32;
            //     while i > 1 {
            //         for wid in 0..64 {
            //             if wid % i == 0 {
            //                 let lo = (i / 2) - 1;
            //                 let hi = i - 1;

            //                 let before = secondary_local_prefix_sum[wid + lo];
            //                 let after = secondary_local_prefix_sum[wid + hi];
            //                 secondary_local_prefix_sum[wid + lo] = after;
            //                 secondary_local_prefix_sum[wid + hi] += before;
            //             }
            //         }
            //         i >>= 1;
            //     }

            // for (i, v) in secondary_local_prefix_sum.iter().enumerate() {
            //     //println!("{}, {}, {}", *v, local_histogram[i], local_prefix_sum[i]);
            // }

            // for (i, val) in secondary_local_prefix_sum.iter().enumerate() {
            //     let manual = *val;
            //     let correct = local_prefix_sum[i];

            //     //println!("manual: {}, correct: {}", manual, correct);
            //     if manual != correct {
            //         //println!("incorrect!");
            //     }
            // }

            //     for (wid, code) in chunk.iter().enumerate() {
            //         if wid % 64 == 0 {
            //             //println!("-------------------");
            //         }
            //         if wid % 128 == 0 {
            //             //println!("_-_-_-_-_-_-");
            //         }
            //         if wid % 192 == 0 {
            //             //println!("______________");
            //         }
            //         let digit = digits[wid];
            //         let id = chunk_id * 256 + wid;
            //         let histogram_column = id / 256;
            //         let histogram_row = digit;
            //         let histogram_index: u32 = (8 * histogram_row) + histogram_column as u32;
            //         let local_id_cpu = map[wid] as u32;
            //         let storage_histogram_cpu =
            //             prefix_summed_histogram_vals[histogram_index as usize];

            //         let histogram_na_cpu = local_prefix_sum[digit as usize] as u32;

            //         let final_location: u64 = ((local_id_cpu + storage_histogram_cpu)
            //             .wrapping_sub(histogram_na_cpu))
            //             as u64;
            //         let final_location_gpu = gpu_codes[chunk_id][wid];

            //         let code_gpu = gpu_codes[chunk_id][wid];
            //         let local_id_gpu = (code_gpu >> 16) & 65535;
            //         let storage_histogram_gpu = code_gpu & 65535;
            //         let histogram_prefix_gpu = (code_gpu >> 48) & 65535;
            //         let digit_gpu = (code_gpu >> 32) & 65535;

            //         cpu_sort[final_location as usize] = *code;
            //         //println!(
            //         //     "local_id: {}, storage_histogram: {}, local_prefix_sum: {}, digit_cpu: {}, histogram: {}",
            //         //     local_id_cpu, storage_histogram_cpu, histogram_na_cpu, digit, local_histogram[digit as usize]
            //         // );
            //         //println!(
            //         //     "local_id_gpu: {}, storage_histogram_gpu: {}, local_prefix_sum_gpu: {}, digit_gpu: {}\n",
            //         //     local_id_gpu, storage_histogram_gpu, histogram_prefix_gpu, digit_gpu
            //         // );
            //         //println!(
            //         //     "local_prefix_sum: {}, digit_cpu: {}, histogram: {}",
            //         //     histogram_na_cpu, digit, local_histogram[digit as usize]
            //         // );
            //         //println!(
            //         //     "local_prefix_sum_gpu: {}, digit_gpu: {}\n",
            //         //     histogram_prefix_gpu, digit_gpu
            //         // );

            //         //println!("gpu: {}, cpu: {}", storage_histogram_gpu, histogram_na_cpu);

            //         //println!("cpu: {}, gpu: {}", final_location, final_location_gpu);
            //     }
            // }
            let mut sorted = true;
            let mut all_zero = true;
            let mut times_unsorted = 0;
            // for i in cpu_sort.windows(2) {
            //     let before = i[0] & 63;
            //     let after = i[1] & 63;
            //println!("{}", before);
            //     if before > after {
            //         times_unsorted += 1;
            //         sorted = false;
            //     }
            //     if before != 0u64 {
            //         all_zero = false;
            //     }
            //     if after != 0u64 {
            //         all_zero = false;
            //     }
            // }
            // if !sorted {
            //     log::error!("Not sorted!!! {}", times_unsorted);
            // } else {
            //println!("sorted!");
            // }
            // if all_zero {
            //     log::error!("all zero!!!");
            // }

            let mut sorted = true;
            let mut all_zero = true;
            for (x, i) in radix_sort_morton_codes_2.windows(2).enumerate() {
                let before = i[0];
                let after = i[1];
                //println!("{:6}, {:#018x}", x, before);
                if before > after {
                    //println!("not sorted!");
                    //println!("bef: {:018x}", before);
                    //println!("aft: {:018x}", after);
                    //println!("xor: {:018x}", before ^ after);
                    //println!("leading_zeros: {}", (before ^ after).leading_zeros());
                    sorted = false;
                }
                if before != 0u64 {
                    all_zero = false;
                }
                if after != 0u64 {
                    all_zero = false;
                }
            }
            if !sorted {
                log::error!("Not sorted!!!");
            } else {
                //println!("Sorted !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            }
            if all_zero {
                log::error!("all zero!!!");
            }

            // let mut set: HashSet<u64> = HashSet::new();
            // for v in radix_sort_morton_codes {
            //println!("{}", v);
            //     let x = set.insert(v);
            //     if !x {
            //         log::error!("value already found! {}", v);
            //     }
            // }

            // for (i, c) in radix_sort_morton_codes_2.chunks(256).enumerate() {
            //     let mut set: HashSet<u32> = HashSet::new();
            //     let mut bar = false;
            //     for (j, code) in c.iter().enumerate() {
            //         if i > 0 {
            //             let local_id = (code >> 16) & 65535;
            //             let storage_histogram = code & 65535;
            //             let histogram_prefix = (code >> 48) & 65535;
            //             let digit = (code >> 32) & 65535;
            //             //println!(
            //                 "local_id: {}, histogram_na: {}, histogram_prefix: {}, digit: {}",
            //                 local_id, storage_histogram, histogram_prefix, digit
            //             );
            //             if local_id < histogram_prefix {
            //                 //println!("smaller");
            //             }
            //             if (local_id + storage_histogram)
            //                 .overflowing_sub(histogram_prefix)
            //                 .0
            //                 > 2000
            //             {
            //                 //println!(
            //                     "bad id {}",
            //                     (local_id + storage_histogram)
            //                         .overflowing_sub(histogram_prefix)
            //                         .0
            //                 );
            //             }

            //             let inserted = set.insert(local_id as u32);
            //             if !inserted {
            //                 //println!("");
            //             }
            //         }
            //         if i == 0 {
            //             let histogram = code & 65535;
            //             //println!("histogram {}", histogram);
            //             //println!("--------------------");
            //             bar = true;
            //         }
            //     }
            // }

            // for v in radix_sort_morton_codes {
            //     let local_id = (v >> 16) & 65535;
            //     let storage_histogram = v & 65535;
            //     let histogram_prefix = (v >> 32) & (u32::MAX as u64);
            //println!(
            //         "local_id: {}, histogram_na: {}, histogram_prefix: {}",
            //         local_id, storage_histogram, histogram_prefix
            //     );
            //     if local_id < histogram_prefix {
            //         //println!("smaller");
            //     }
            //     if local_id + storage_histogram - histogram_prefix > 2000 {
            //         //println!("bad id {}", local_id + storage_histogram - histogram_prefix);
            //     }
            // }

            // for v in radix_sort_morton_codes.chunks(256) {
            //     let mut set: HashSet<u64> = HashSet::new();
            //     for code in v {
            //         if set.contains(code) {
            //             log::error!("value already found");
            //         }
            //         set.insert(*code);
            //     }
            // }
        } else {
            panic!("failed to run compute on GPU!!!!!");
        }
    }
}

fn create_radix_uniforms_buffer(device: &Device, pass_number: u32) -> Buffer {
    let radix_uniforms = RadixUniforms { pass_number };

    let radix_uniforms_temp_buf: Vec<u8> = Vec::new();
    let mut radix_uniforms_temp_buf_ub: UniformBuffer<Vec<u8>> =
        UniformBuffer::new(radix_uniforms_temp_buf);
    radix_uniforms_temp_buf_ub.write(&radix_uniforms).unwrap();
    let radix_uniforms_b = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("radix uniforms buffer"),
        contents: radix_uniforms_temp_buf_ub.as_ref(),
        usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::UNIFORM,
    });
    radix_uniforms_b
}

fn select_digit(pass_number: u32, code_1: u32, code_2: u32) -> u32 {
    let mut val = 64;

    if pass_number < 5 {
        val = (code_1 >> (pass_number * 6)) & 63;
    } else if pass_number > 5 {
        val = (code_2 >> ((pass_number - 6) * 6 + 4)) & 63;
    } else {
        val = ((code_1 >> 30) & 3) | ((code_2 & 15) << 2);
    }
    return val;
}

fn main() {
    pollster::block_on(run());
}

#[cfg(test)]
mod tests {
    use crate::{calculate_number_of_workgroups_u32, select_digit};

    #[test]
    fn workgroup_calculation() {
        assert_eq!(calculate_number_of_workgroups_u32(0, 256), 0, "i: {}", 0);
        for i in 1..256 {
            assert_eq!(calculate_number_of_workgroups_u32(i, 256), 1, "i: {}", i);
        }
        for i in 257..512 {
            assert_eq!(calculate_number_of_workgroups_u32(i, 256), 2, "i: {}", i);
        }
    }

    #[test]
    fn digit_test() {
        let digit: u64 = 63;

        for i in 0..11 {
            let y: u64 = digit.wrapping_shl(i * 6);
            //println!("{:064b}", y);

            let hi: u32 = (y >> 32) as u32;
            let lo: u32 = (y & (u32::MAX as u64)) as u32;
            //println!("{:032b} {:032b}", hi, lo);
            let selected = select_digit(i as u32, lo, hi);
            //println!("selected {}, digit {}", selected, digit);
        }
    }
}
