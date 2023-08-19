use std::collections::HashSet;
#[cfg(morton_code_readback)]
use std::{fs::File, io::Write};

use constants::*;
use encase::{private::WriteInto, ShaderType, StorageBuffer, UniformBuffer};
use extended_morton_coder::MortonCodeGenerator;
use glam::{UVec3, Vec2, Vec3};
use rand::{distributions::Uniform, prelude::Distribution, Rng};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use types::*;
use utilities::*;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroupEntry, BufferDescriptor, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, Device, Instance, InstanceDescriptor, PipelineLayoutDescriptor,
    PowerPreference, RequestAdapterOptions, *,
};

use bind_group_layouts::{create_all_bind_group_layouts, create_radix_bgl_prefix_large};
use shader_modules::create_all_shader_modules;

mod bind_group_layouts;
mod constants;
mod shader_modules;
mod state;
mod types;
mod utilities;

fn create_all_buffers(
    device: &Device,
    vertices: Vertices,
    triangles: Triangles,
    morton_code_generator: &MortonCodeGenerator,
    histogram_buffer_number_elements: u32,
) -> Buffers {
    let morton_uniforms = create_morton_uniforms(&morton_code_generator);
    let morton_uniforms_b = create_buffer(
        morton_uniforms,
        &device,
        "morton uniform buffer",
        BufferUsages::STORAGE,
    );
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
    let morton_code_b_size = morton_code_size * constants::NUM_TRIANGLES as u64;
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
    Buffers {
        vertices: vertices_b,
        indices: indices_b,
        indices_2: indices_2_b,
        morton_uniforms: morton_uniforms_b,
        histogram: histogram_b,
        morton_codes: morton_code_b,
        morton_codes_2: morton_code_2_b,
        prefix_sums: prefix_sum_bs,
    }
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

#[allow(dead_code)]
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
    let mut rng_gen = ChaCha8Rng::seed_from_u64(constants::RNG_SEED);
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
        triangles: Vec::with_capacity(constants::NUM_TRIANGLES as usize),
    };
    let rng_distribution: Uniform<u32> = Uniform::new(0, NUM_VERTICES as u32);
    for _ in 0..constants::NUM_TRIANGLES {
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
            // node_index: 0,
            // material_id: rng_gen.gen_range(0..5),
            // flags: 0,
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
        return vec![];
    }
    num_items_in_buffer /= 256;
    num_items_in_buffer = round_up_u64(num_items_in_buffer, ITEMS_PER_LARGE_PREFIX as u64);
    let mut num_items_in_buffers = vec![];
    // While the number of elements we're processing is larger than the maximum we can handle in a small prefix pass
    while num_items_in_buffer > MAX_ITEMS_IN_SMALL_PREFIX as u64 {
        // Do another large prefix pass
        num_items_in_buffers.push(num_items_in_buffer);
        // Divide the number of elements by the number of elements reduced in a single large prefix workgroup
        // This is guaranteed to produce an integer, as we already rounded up to the next multiple of ITEMS_PER_LARGE_PREFIX.
        num_items_in_buffer /= ITEMS_PER_LARGE_PREFIX as u64;
        // Round up the number of elements to the next 256 to not break anything
        num_items_in_buffer = round_up_u64(num_items_in_buffer, ITEMS_PER_LARGE_PREFIX as u64);
    }
    // Now that we have all the large prefix buffers, let's create the small one. We know there will always be at least one
    // if we got this point.
    // We might have 0 large prefix buffers and 1 small, 1 large 1 small, N large 1 small
    num_items_in_buffers.push(num_items_in_buffer);
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
        //println!("size of buffer {}", number_of_items_in_buffer);
        prefix_sum_buffers.push(device.create_buffer(&BufferDescriptor {
            label: Some(format!("prefix sum buffer {number_of_items_in_buffer}").as_str()),
            size: number_of_items_in_buffer * bytes_per_element,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    prefix_sum_buffers
}

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
        layout,
        entries: &entries,
    })
}

fn create_all_pipelines(
    device: &Device,
    morton_code_l: &BindGroupLayout,
    morton_module: ShaderModule,
    radix_histogram_l: &BindGroupLayout,
    radix_sort_histogram_module: ShaderModule,
    radix_prefix_large_l: &BindGroupLayout,
    radix_sort_prefix_large_module: ShaderModule,
    radix_prefix_small_l: &BindGroupLayout,
    radix_sort_prefix_small_module: ShaderModule,
    radix_index_l: &BindGroupLayout,
    radix_sort_index_module: ShaderModule,
) -> (
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
) {
    (
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("morton code compute pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[morton_code_l],
                push_constant_ranges: &[],
            })),
            module: &morton_module,
            entry_point: "morton_code_unrolled",
        }),
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("radix sort histogram compute pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[radix_histogram_l],
                push_constant_ranges: &[],
            })),
            module: &radix_sort_histogram_module,
            entry_point: "radix_sort_compute_histogram",
        }),
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("radix prefix large pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[radix_prefix_large_l],
                push_constant_ranges: &[],
            })),
            module: &radix_sort_prefix_large_module,
            entry_point: "radix_sort_block_sum_large",
        }),
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("radix prefix large pipeline 2"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[radix_prefix_large_l],
                push_constant_ranges: &[],
            })),
            module: &radix_sort_prefix_large_module,
            entry_point: "radix_sort_block_sum_large_after",
        }),
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("radix sort prefix sum small compute pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[radix_prefix_small_l],
                push_constant_ranges: &[],
            })),
            module: &radix_sort_prefix_small_module,
            entry_point: "radix_sort_block_sum_small",
        }),
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("radix index pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[radix_index_l],
                push_constant_ranges: &[],
            })),
            module: &radix_sort_index_module,
            entry_point: "radix_sort_index",
        }),
    )
}

fn create_index_uniforms(device: &Device, pass_number: u32) -> Buffer {
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

pub fn run_compute_shaders(device: &Device, mut encoder: &mut CommandEncoder) -> Vec<Buffer> {
    let number_of_workgroups =
        calculate_number_of_workgroups_u32(NUM_TRIANGLES, ITEMS_PER_HISTOGRAM_PASS);
    let (vertices, triangles, morton_code_generator, _vertices_copy, _triangles_copy) =
        create_scene();
    let histogram_buffer_number_elements = round_up_u32(number_of_workgroups * HISTOGRAM_SIZE, 256);
    let buffers = create_all_buffers(
        device,
        vertices,
        triangles,
        &morton_code_generator,
        histogram_buffer_number_elements,
    );

    #[cfg(radix_sort_readback)]
    let radix_sort_morton_codes_readback_b = device.create_buffer(&BufferDescriptor {
        label: Some("radix_sort_morton_codes_readback_b"),
        size: buffers.morton_codes.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    #[cfg(radix_sort_readback)]
    let radix_sort_morton_codes_2_readback_b = device.create_buffer(&BufferDescriptor {
        label: Some("radix_sort_morton_codes_2_readback_b"),
        size: buffers.morton_codes.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    #[cfg(morton_code_readback)]
    let morton_code_readback_b = device.create_buffer(&BufferDescriptor {
        label: Some("morton code readback buffer"),
        size: buffers.morton_codes.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // region: shader modules
    let shader_modules = create_all_shader_modules(device);
    let (
        morton_module,
        radix_sort_histogram_module,
        radix_sort_prefix_large_module,
        radix_sort_prefix_small_module,
        radix_sort_index_module,
    ) = (
        shader_modules.morton_code,
        shader_modules.radix_sort_histogram,
        shader_modules.radix_sort_prefix_large,
        shader_modules.radix_sort_prefix_small,
        shader_modules.radix_sort_index,
    );
    // endregion
    // region: bind group layouts
    let (
        morton_code_l,
        radix_histogram_l,
        radix_prefix_large_l,
        radix_prefix_small_l,
        radix_index_l,
    ) = create_all_bind_group_layouts(device);
    // endregion
    // region: compute pipelines
    let (
        morton_code_p,
        radix_histogram_p,
        radix_prefix_large_p,
        radix_prefix_large_p_2,
        radix_prefix_small_p,
        radix_index_p,
    ) = create_all_pipelines(
        device,
        &morton_code_l,
        morton_module,
        &radix_histogram_l,
        radix_sort_histogram_module,
        &radix_prefix_large_l,
        radix_sort_prefix_large_module,
        &radix_prefix_small_l,
        radix_sort_prefix_small_module,
        &radix_index_l,
        radix_sort_index_module,
    );

    // endregion
    // let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
    //     label: Some("command encoder"),
    // });

    let morton_code_bg: BindGroup = create_bind_group(
        device,
        &morton_code_l,
        vec![
            &buffers.morton_uniforms,
            &buffers.vertices,
            &buffers.indices,
            &buffers.morton_codes,
        ],
        "morton code bind group",
    );
    let mut radix_prefix_large_bind_groups = vec![];

    if !buffers.prefix_sums.is_empty() {
        log::info!("prefix_sum_bs was not empty");
        radix_prefix_large_bind_groups.push(create_bind_group(
            device,
            &radix_prefix_large_l,
            vec![&buffers.histogram, &buffers.prefix_sums[0]],
            "radix sort prefix sum large bind group",
        ));
        for i in 0..buffers.prefix_sums.len() - 1 {
            radix_prefix_large_bind_groups.push(create_bind_group(
                device,
                &create_radix_bgl_prefix_large(device),
                vec![&buffers.prefix_sums[i], &buffers.prefix_sums[i + 1]],
                "radix sort prefix sum large bind group",
            ));
        }
    } else {
        log::info!("We don't have any prefix sum buffers so we don't need any bindgroups to them, but we still need to create a small bind group on the histogram");
    }

    let radix_prefix_small_bg = match buffers.prefix_sums.is_empty() {
        true => {
            log::info!("Create a small bind group directly on the histogram buffer");
            create_bind_group(
                device,
                &radix_prefix_small_l,
                vec![&buffers.histogram],
                "radix sort prefix sum small bind group",
            )
        }
        false => create_bind_group(
            device,
            &radix_prefix_small_l,
            vec![buffers.prefix_sums.last().unwrap()],
            "radix sort prefix sum small bind group",
        ),
    };

    let mut radix_histogram_bgs = vec![];
    let mut radix_index_bgs = vec![];

    for i in 0..NUM_PASSES {
        let radix_uniforms_b = create_index_uniforms(device, i);
        let ping_pong = i % 2 == 0;
        let radix_histogram_bg = match ping_pong {
            true => create_bind_group(
                device,
                &radix_histogram_l,
                vec![
                    &radix_uniforms_b,
                    &buffers.indices,
                    &buffers.morton_codes,
                    &buffers.histogram,
                ],
                "radix sort histogram bind group",
            ),
            false => create_bind_group(
                device,
                &radix_histogram_l,
                vec![
                    &radix_uniforms_b,
                    &buffers.indices_2,
                    &buffers.morton_codes_2,
                    &buffers.histogram,
                ],
                "radix sort histogram bind group",
            ),
        };
        radix_histogram_bgs.push(radix_histogram_bg);

        let radix_index_bg = match ping_pong {
            true => create_bind_group(
                device,
                &radix_index_l,
                vec![
                    &radix_uniforms_b,
                    &buffers.indices,
                    &buffers.morton_codes,
                    &buffers.indices_2,
                    &buffers.morton_codes_2,
                    &buffers.histogram,
                ],
                "radix sort index bind group",
            ),
            false => create_bind_group(
                device,
                &radix_index_l,
                vec![
                    &radix_uniforms_b,
                    &buffers.indices_2,
                    &buffers.morton_codes_2,
                    &buffers.indices,
                    &buffers.morton_codes,
                    &buffers.histogram,
                ],
                "radix sort index bind group",
            ),
        };
        radix_index_bgs.push(radix_index_bg);
    }

    let buffer_num_items = calculate_num_items_prefix_buffers(NUM_TRIANGLES as u64);
    assert!(buffer_num_items.len() == radix_prefix_large_bind_groups.len());

    {
        let compute_pass_desc = ComputePassDescriptor {
            label: Some("Morton Coding"),
        };
        let mut compute_pass = encoder.begin_compute_pass(&compute_pass_desc);
        compute_pass.set_pipeline(&morton_code_p);
        compute_pass.set_bind_group(0, &morton_code_bg, &[]);
        compute_pass.insert_debug_marker("compute morton code");
        let num_workgroups_x = div_ceil_u32(NUM_TRIANGLES, 2048);
        //println!("num_workgroups_x {}", num_workgroups_x);
        compute_pass.dispatch_workgroups(num_workgroups_x, 8, 1);
    }

    for i in 0..NUM_PASSES {
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Histogram calculation"),
            });
            compute_pass.set_pipeline(&radix_histogram_p);
            compute_pass.set_bind_group(0, &radix_histogram_bgs[i as usize], &[]);
            compute_pass.insert_debug_marker("histogram pass");
            compute_pass.dispatch_workgroups(number_of_workgroups, 1, 1);
        }
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Large Prefix Sum part 1"),
            });
            compute_pass.set_pipeline(&radix_prefix_large_p);

            for (i, large_bg) in radix_prefix_large_bind_groups.iter().enumerate() {
                compute_pass.insert_debug_marker("large prefix");
                compute_pass.set_bind_group(0, &large_bg, &[]);
                let prefix_workgroup = match i {
                    0 => div_ceil_u64(
                        histogram_buffer_number_elements as u64,
                        ITEMS_PER_LARGE_PREFIX as u64,
                    ),
                    _ => div_ceil_u64(buffer_num_items[i - 1], ITEMS_PER_LARGE_PREFIX as u64),
                };
                compute_pass.dispatch_workgroups(prefix_workgroup as u32, 1, 1);
            }
        }
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Small Prefix Sum"),
            });
            log::info!("run the small prefix bind group");
            compute_pass.set_pipeline(&radix_prefix_small_p);
            compute_pass.set_bind_group(0, &radix_prefix_small_bg, &[]);
            compute_pass.insert_debug_marker("small prefix");
            compute_pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Large Prefix Sum part 2"),
            });
            compute_pass.set_pipeline(&radix_prefix_large_p_2);

            let buffer_num_items = calculate_num_items_prefix_buffers(NUM_TRIANGLES as u64);

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
                compute_pass.dispatch_workgroups(prefix_workgroup as u32, 1, 1);
            }
        }
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Indexing"),
            });
            compute_pass.set_pipeline(&radix_index_p);
            compute_pass.set_bind_group(0, &radix_index_bgs[i as usize], &[]);
            compute_pass.insert_debug_marker("index");
            compute_pass.dispatch_workgroups(number_of_workgroups, 1, 1);
        }
    }
    #[cfg(radix_sort_readback)]
    {
        copy_buffer_to_buffer(
            &mut encoder,
            &buffers.morton_codes,
            &radix_sort_morton_codes_readback_b,
        );
        copy_buffer_to_buffer(
            &mut encoder,
            &buffers.morton_codes_2,
            &radix_sort_morton_codes_2_readback_b,
        );
    }
    #[cfg(morton_code_readback)]
    copy_buffer_to_buffer(&mut encoder, &buffers.morton_codes, &morton_code_readback_b);
    #[cfg(radix_sort_readback)]
    return vec![
        radix_sort_morton_codes_readback_b,
        radix_sort_morton_codes_2_readback_b,
    ];
    #[cfg(not(any(radix_sort_readback, morton_code_readback)))]
    return vec![];
}

async fn run_headless() {
    env_logger::init();
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
    device.start_capture();

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("command encoder"),
    });

    #[allow(unused_variables)]
    let readback_buffers = run_compute_shaders(&device, &mut encoder);

    queue.submit(Some(encoder.finish()));

    device.stop_capture();
    #[cfg(radix_sort_readback)]
    let radix_sort_morton_codes_readback_b = &readback_buffers[0];
    #[cfg(radix_sort_readback)]
    let radix_sort_morton_codes_2_readback_b = &readback_buffers[1];

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
            let mut file = File::create("morton_codes.txt").unwrap();
            for &value in &morton_codes {
                let line = format!("{}\n", value);
                file.write_all(&line.as_bytes()).unwrap();
            }

            let mut file_2 = File::create("indices.txt").unwrap();

            for triangle in _triangles_copy.triangles {
                let line = format!(
                    "({}, {}, {})\n",
                    triangle.indices.x, triangle.indices.y, triangle.indices.z
                );
                // Write bytes to file
                file_2.write_all(&line.as_bytes()).unwrap();
            }
        } else {
            panic!("failed to run compute on GPU!!!!!");
        }
    }

    #[cfg(radix_sort_readback)]
    {
        let radix_sort_morton_codes_2_readback_slice =
            radix_sort_morton_codes_2_readback_b.slice(..);
        // NOTE: We have to create the mapping THEN device.poll() before awaiting
        // the future. Otherwise the application will freeze.
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        radix_sort_morton_codes_2_readback_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let radix_sort_morton_codes_readback_slice = radix_sort_morton_codes_readback_b.slice(..);
        let (tx2, rx2) = futures_intrusive::channel::shared::oneshot_channel();
        radix_sort_morton_codes_readback_slice.map_async(MapMode::Read, move |result| {
            tx2.send(result).unwrap();
        });

        device.poll(Maintain::Wait);
        if let (Some(Ok(())), Some(Ok(()))) = (rx.receive().await, rx2.receive().await) {
            let radix_sort_morton_codes_readback_data =
                radix_sort_morton_codes_readback_slice.get_mapped_range();
            let radix_sort_morton_codes: Vec<u64> =
                bytemuck::cast_slice(&radix_sort_morton_codes_readback_data).to_vec();
            drop(radix_sort_morton_codes_readback_data);
            radix_sort_morton_codes_readback_b.unmap();

            let radix_sort_morton_codes_2_readback_data =
                radix_sort_morton_codes_2_readback_slice.get_mapped_range();
            let radix_sort_morton_codes_2: Vec<u64> =
                bytemuck::cast_slice(&radix_sort_morton_codes_2_readback_data).to_vec();
            drop(radix_sort_morton_codes_2_readback_data);
            radix_sort_morton_codes_2_readback_b.unmap();

            // let mut set = HashSet::new();
            // for (i, c) in radix_sort_morton_codes_2.iter().enumerate() {
            //     let new = set.insert(*c);
            //     if !new {
            //         println!("already in set");
            //     }
            //     if *c > 1500 {
            //         println!("i: {}, c: {}", i, *c);
            //     } else {
            //         println!("{}", *c);
            //     }
            // }
            // println!("set size {}", set.len());

            let output_vec = if NUM_PASSES % 2 == 0 {
                &radix_sort_morton_codes
            } else {
                &radix_sort_morton_codes_2
            };

            let last_pass_vec = if NUM_PASSES % 2 == 0 {
                &radix_sort_morton_codes_2
            } else {
                &radix_sort_morton_codes
            };

            let mut sorted = true;
            let mut all_zero = true;
            let mut times_unsorted = 0u32;
            for (x, i) in output_vec.windows(2).enumerate() {
                if x == 0 {
                    println!("-----------------");
                }
                let before = select_bits(i[0], NUM_PASSES);
                let after = select_bits(i[1], NUM_PASSES);
                println!("{:6}, {:#018x}", x, before);
                if before > after {
                    times_unsorted += 1;
                    println!("not sorted!");
                    println!("bef: {:#018x}", before);
                    println!("aft: {:#018x}", after);
                    println!("xor: {:#018x}", before ^ after);
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
                log::error!("Not sorted!!! {}", times_unsorted);
            } else {
                println!("Sorted !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            }
            if all_zero {
                log::error!("all zero!!!");
            }

            // for (i, chunk) in radix_sort_morton_codes.chunks(256).enumerate() {
            //     let mut digits = [0; 256];
            //     chunk.iter().enumerate().for_each(|(i, x)| {
            //         digits[i] = x & 255;
            //     });
            //     let wrong: &[u64] = radix_sort_morton_codes_2.chunks(256).nth(i).unwrap();
            //     let mut true_local_offset = [0; 256];
            //     let mut digit_count = [0; 256];
            //     for (i, digit) in digits.iter().enumerate() {
            //         true_local_offset[i] = digit_count[*digit as usize];
            //         digit_count[*digit as usize] += 1;
            //     }

            //     let mut error_count = 0;
            //     for (j, val) in wrong.iter().enumerate() {
            //         if *val != true_local_offset[j] {
            //             println!("error {} {}", *val, true_local_offset[j]);
            //             error_count += 1;
            //         } else {
            //             println!("{} {}", *val, true_local_offset[j]);
            //         }
            //     }
            //     println!("num errors {}", error_count);
            //     let mut histogram = [0; 256];
            //     for (i, digit) in digits.iter().enumerate() {
            //         if i < chunk.len() {
            //             histogram[*digit as usize] += 1;
            //         }
            //     }
            //     let p_sum: Vec<u32> = histogram
            //         .iter()
            //         .scan(0u32, |state, &x| {
            //             *state += x;
            //             Some(*state - x)
            //         })
            //         .collect();
            //     let mut sorted_output = [0; 256];
            //     digits.iter().enumerate().for_each(|(index, val)| {
            //         let f = p_sum[*val as usize] + true_local_offset[index] as u32;
            //         sorted_output[f as usize] = digits[index];
            //     });
            //     println!("sorted: {:?}", sorted_output);
            // }

            // Testing the radix sort with 1500 elements.
            // let num_codes: u32 = 1500;
            // let mut radix_sort_codes: Vec<u64> = vec![];
            // // Creating a distribution of different values.
            // for i in 0..15 {
            //     radix_sort_codes.extend(vec![i; 100]);
            // }
            // assert_eq!(radix_sort_codes.len() as u32, num_codes);

            // This is the device-wide histogram. It has 6 arrays of 256, where 256 = 2^8 where we are considering
            // 8 bits in the radix sort. We have 6 groups of 256. 6 * 256 = 1,536, which is greater than 1500.
            // Because not all chunks will be filled to their limit, we have to make sure to limit our prefix sum
            // to the chunk size. Call this check "Check A".
            if false {
                let mut storage_histogram_cpu = [[0u32; 256]; 6];
                assert_eq!(calculate_number_of_workgroups_u32(NUM_TRIANGLES, 256), 6);
                for (i, c) in radix_sort_morton_codes.chunks(256).enumerate() {
                    for val in c {
                        // incrementing the storage histogram at the current digit value by 1.
                        storage_histogram_cpu[i][((*val) & 255) as usize] += 1;
                    }
                }
                // performing a prefix sum over our large device-wide histogram.
                let mut sum = 0;
                // The order of the iteration is first over the digit values, then over the blocks.
                // This is best illustrated with an example - suppose we have the digit 3 somewhere in block 4.
                // Then, the final location will be equal to the sum of digit values [0,2] in all blocks [0, 5], as well
                // as the sum of digit value 3 in blocks [0, 4).

                // Also note this an exclusive prefix sum, not inclusive.
                for k in 0..256 {
                    for j in 0..6 {
                        let g = storage_histogram_cpu[j][k];
                        storage_histogram_cpu[j][k] = sum;
                        sum += g;
                    }
                }
                //println!("{:?}", storage_histogram_cpu);
                // now we are going to calculate a prefix sum per block histogram.
                let mut block_prefix_sums = vec![];
                for (index, chunk) in radix_sort_morton_codes.chunks(256).enumerate() {
                    // // first, we find the digits in our current chunk.
                    // let mut digits = [0; 256];
                    // chunk.iter().enumerate().for_each(|(i, x)| {
                    //     digits[i] = x & 255;
                    // });

                    let mut digits = [0u32; 256];
                    let mut output = [0u32; 256];
                    for (i, &val) in chunk.iter().enumerate() {
                        output[i] += digits[(val & 255) as usize];
                        digits[(val & 255) as usize] += 1;
                    }
                    let x: Vec<u32> = output.iter().map(|x| *x as u32).collect();
                    block_prefix_sums.push(x);
                    // then, we compute a histogram over those digit values.
                    // let mut histogram = [0; 256];
                    // for (i, digit) in digits.iter().enumerate() {
                    //     // This is the "Check A" that is necessary. Digits is always of length 256, but for the final
                    //     // block it's only going to be of length 220 = (1500 - (5 * 256)).
                    //     if i < chunk.len() {
                    //         histogram[*digit as usize] += 1;
                    //     }
                    // }

                    // // Now we compute the block prefix sum over the histogram. Note that this is also an exclusive sum.
                    // let block_prefix_sum: Vec<u32> = histogram
                    //     .iter()
                    //     .scan(0u32, |state, &x| {
                    //         *state += x;
                    //         Some(*state - x)
                    //     })
                    //     .collect();
                    // block_prefix_sums.push(histogram);
                }
                // We iterate over our values one more time to check whether we've calculated them correctly.
                let mut sorted_cpu = vec![1000; NUM_TRIANGLES as usize];
                let mut set = HashSet::new();
                for (i, c) in radix_sort_morton_codes.chunks(256).enumerate() {
                    // One chunk at a time
                    for (j, code) in c.iter().enumerate() {
                        let gpu_vals = radix_sort_morton_codes_2.chunks(256).nth(i).unwrap()[j];
                        let local_offset = gpu_vals >> 16;
                        let storage_histogram = gpu_vals & 65535;
                        let digit = *code & 255;
                        // We know that the block prefix sum + device-wide prefix sum should be less than the size of the
                        // overall array we're sorting.
                        let cpu_final_loc =
                            block_prefix_sums[i][j] + storage_histogram_cpu[i][digit as usize];
                        // if storage_histogram_cpu[i][digit as usize] != storage_histogram as u32 {
                        //     println!(" ISSUE )))))))))))))))))))))))))))))))))")
                        // }
                        if cpu_final_loc != (local_offset + storage_histogram) as u32 {
                            println!(
                                "!!!!!!!!!!!!! cpu: {} + {}, gpu: {} + {}",
                                block_prefix_sums[i][j],
                                storage_histogram_cpu[i][digit as usize],
                                local_offset,
                                storage_histogram
                            );
                        }
                        let new = set.insert(cpu_final_loc);
                        if cpu_final_loc >= NUM_TRIANGLES {
                            // We are reaching this line at the end of the program with the values 120, 1400, 5, 14.
                            println!(
                            " CPU CALCULATION WRONG block_prefix_sum: {} storage_histogram_cpu: {} i: {}, digit: {}",
                            block_prefix_sums[i][j],
                            storage_histogram_cpu[i][digit as usize],
                            i,
                            digit
                        );
                        }
                        sorted_cpu[cpu_final_loc as usize] = digit;
                    }
                    println!("-----");
                }
                println!("set size: {}", set.len());
                println!("{:?}", sorted_cpu);
            }
        } else {
            panic!("failed to run compute on GPU!!!!!");
        }
    }
}

fn main() {
    #[cfg(radix_sort_readback)]
    pollster::block_on(run_headless());
    #[cfg(not(any(radix_sort_readback, morton_code_readback)))]
    pollster::block_on(state::run());
}

fn select_bits(value: u64, i: u32) -> u64 {
    if i < 1 || i > 8 {
        panic!("Input must be between 1 and 8");
    }
    if i == 8 {
        value
    } else {
        value & ((1 << (i * 8)) - 1)
    }
}

#[cfg(test)]
mod tests;
