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
    let final_locations = device.create_buffer(&BufferDescriptor {
        label: Some("final locations"),
        size: (constants::NUM_TRIANGLES * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
        final_locations,
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
    radix_scatter_l: &BindGroupLayout,
    radix_sort_scatter_module: ShaderModule,
) -> (
    ComputePipeline,
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
            entry_point: "morton_code",
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
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("scatter compute p"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[radix_scatter_l],
                push_constant_ranges: &[],
            })),
            module: &radix_sort_scatter_module,
            entry_point: "radix_sort_scatter",
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

fn create_radix_scatter_uniforms(device: &Device, scatter_step: u32) -> Buffer {
    let radix_scatter_uniforms = ScatterUniforms {
        scatter_step,
        number_sections: NUMBER_SECTIONS_SCATTER,
    };

    let radix_scatter_uniforms_temp_buf: Vec<u8> = Vec::new();
    let mut radix_scatter_uniforms_temp_buf_ub: UniformBuffer<Vec<u8>> =
        UniformBuffer::new(radix_scatter_uniforms_temp_buf);
    radix_scatter_uniforms_temp_buf_ub
        .write(&radix_scatter_uniforms)
        .unwrap();
    let radix_scatter_uniforms_b = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("radix scatter uniforms buffer"),
        contents: radix_scatter_uniforms_temp_buf_ub.as_ref(),
        usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::UNIFORM,
    });
    radix_scatter_uniforms_b
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
    let readback_buffers = run_compute_shaders(&device, &queue, &mut encoder);

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

            let mut sorted = true;
            let mut all_zero = true;
            for (x, i) in radix_sort_morton_codes.windows(2).enumerate() {
                let before = i[0];
                let after = i[1];
                //println!("{:6}, {:#018x}", x, before);
                if before > after {
                    // println!("not sorted!");
                    // println!("bef: {:018x}", before);
                    // println!("aft: {:018x}", after);
                    // println!("xor: {:018x}", before ^ after);
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
                println!("Sorted !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            }
            if all_zero {
                log::error!("all zero!!!");
            }
        } else {
            panic!("failed to run compute on GPU!!!!!");
        }
    }

    #[cfg(final_locations_readback)]
    {
        let slice = final_locations_readback_b.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(Maintain::Wait);
        if let Some(Ok(())) = rx.receive().await {
            let readback_data = slice.get_mapped_range();
            let final_locations: Vec<u32> = bytemuck::cast_slice(&readback_data).to_vec();
            drop(readback_data);
            final_locations_readback_b.unmap();

            for (i, loc) in final_locations.iter().enumerate() {
                println!("i: {}, loc: {}", i, loc);
            }
        }
    }
}

pub fn run_compute_shaders(
    device: &Device,
    queue: &Queue,
    mut encoder: &mut CommandEncoder,
) -> Vec<Buffer> {
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
    #[cfg(final_locations_readback)]
    let final_locations_readback_b = device.create_buffer(&BufferDescriptor {
        label: Some("final locations readback buffer"),
        size: buffers.final_locations.size(),
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
        radix_sort_scatter_module,
    ) = (
        shader_modules.morton_code,
        shader_modules.radix_sort_histogram,
        shader_modules.radix_sort_prefix_large,
        shader_modules.radix_sort_prefix_small,
        shader_modules.radix_sort_index,
        shader_modules.radix_sort_scatter,
    );
    // endregion
    // region: bind group layouts
    let (
        morton_code_l,
        radix_histogram_l,
        radix_prefix_large_l,
        radix_prefix_small_l,
        radix_index_l,
        radix_scatter_l,
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
        radix_scatter_p,
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
        &radix_scatter_l,
        radix_sort_scatter_module,
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
    let mut radix_scatter_bg_vecs = vec![];

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
                    &buffers.morton_codes,
                    &buffers.histogram,
                    &buffers.final_locations,
                ],
                "radix sort index bind group",
            ),
            false => create_bind_group(
                device,
                &radix_index_l,
                vec![
                    &radix_uniforms_b,
                    &buffers.morton_codes_2,
                    &buffers.histogram,
                    &buffers.final_locations,
                ],
                "radix sort index bind group",
            ),
        };
        radix_index_bgs.push(radix_index_bg);

        let mut radix_scatter_bg_vec = vec![];
        for j in 0..NUMBER_SECTIONS_SCATTER {
            let radix_scatter_uniforms = create_radix_scatter_uniforms(device, j);
            let radix_scatter_bg = match ping_pong {
                true => create_bind_group(
                    device,
                    &radix_scatter_l,
                    vec![
                        &radix_scatter_uniforms,
                        &buffers.indices,
                        &buffers.morton_codes,
                        &buffers.indices_2,
                        &buffers.morton_codes_2,
                        &buffers.final_locations,
                    ],
                    "radix sort scatter bind group",
                ),
                false => create_bind_group(
                    device,
                    &radix_scatter_l,
                    vec![
                        &radix_scatter_uniforms,
                        &buffers.indices_2,
                        &buffers.morton_codes_2,
                        &buffers.indices,
                        &buffers.morton_codes,
                        &buffers.final_locations,
                    ],
                    "radix sort scatter bind group",
                ),
            };
            radix_scatter_bg_vec.push(radix_scatter_bg);
        }
        radix_scatter_bg_vecs.push(radix_scatter_bg_vec);
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
        let num_workgroups_x = div_ceil_u32(div_ceil_u32(NUM_TRIANGLES, 256), 8);
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
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Scatter"),
            });
            compute_pass.set_pipeline(&radix_scatter_p);
            compute_pass.insert_debug_marker("scatter");
            for j in 0..NUMBER_SECTIONS_SCATTER {
                compute_pass.set_bind_group(0, &radix_scatter_bg_vecs[i as usize][j as usize], &[]);
                // multiplying by 4 because scatter only uses 64 lanes
                compute_pass.dispatch_workgroups(number_of_workgroups, 1, 1);
            }
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
    #[cfg(final_locations_readback)]
    copy_buffer_to_buffer(
        &mut encoder,
        &buffers.final_locations,
        &final_locations_readback_b,
    );

    #[cfg(radix_sort_readback)]
    return vec![
        radix_sort_morton_codes_readback_b,
        radix_sort_morton_codes_2_readback_b,
    ];
    #[cfg(not(any(radix_sort_readback, morton_code_readback, final_locations_readback)))]
    return vec![];
}

fn main() {
    #[cfg(radix_sort_readback)]
    pollster::block_on(run_headless());
    #[cfg(not(any(radix_sort_readback, morton_code_readback, final_locations_readback)))]
    pollster::block_on(state::run());
}

#[cfg(test)]
mod tests;
