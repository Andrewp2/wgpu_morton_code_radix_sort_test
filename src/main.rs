use std::borrow::Cow;

use encase::{private::WriteInto, ShaderType, StorageBuffer, UniformBuffer};
use extended_morton_coder::MortonCodeGenerator;
use glam::{UVec3, Vec2, Vec3};
use rand::{distributions::Uniform, prelude::Distribution, Rng};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    *,
};

#[derive(ShaderType, Copy, Clone, Default)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(ShaderType, Clone)]
pub struct Vertices {
    #[size(runtime)]
    pub vertices: Vec<Vertex>,
}

#[derive(ShaderType)]
pub struct TriangleIndices {
    indices: UVec3,
    node_index: u32,
    material_id: u32,
    flags: u32,
}

#[derive(ShaderType)]
pub struct Triangles {
    #[size(runtime)]
    pub triangles: Vec<TriangleIndices>,
}

#[derive(ShaderType)]
pub struct MortonUniforms {
    lut: [u32; 4608],
    size_lut: [u32; 1 << (extended_morton_coder::SIZE_LUT_NUMBER_OF_BITS + 1)],
    morton_index_scale: f32,
    offset: Vec3,
    size_multiplier: f32,
    multiplier: Vec3,
}

fn produce_buffer<T: ShaderType + WriteInto>(
    t: T,
    device: &wgpu::Device,
    label_str: &str,
    usage: BufferUsages,
) -> Buffer {
    let buf: Vec<u8> = Vec::new();
    let mut x: StorageBuffer<Vec<u8>> = StorageBuffer::new(buf);
    x.write(&t).unwrap();
    let final_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label_str),
        contents: x.as_ref(),
        usage,
    });
    final_buffer
}

async fn run() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        dx12_shader_compiler: Default::default(),
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .unwrap();
    // 3 input buffers: uniforms/vertices/indices
    // 1 output buffer: 1 for morton codes
    let num_vertices = 1000;
    let num_triangles = 2000;
    let mut vertices: Vertices = Vertices {
        vertices: Vec::with_capacity(num_vertices),
    };
    let mut rng_gen = rand::thread_rng();
    let mut scene_min = Vec3::splat(f32::MAX);
    let mut scene_max = Vec3::splat(f32::MIN);
    for _ in 0..num_vertices {
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
        triangles: Vec::with_capacity(num_triangles),
    };
    let rng_distribution: Uniform<u32> = Uniform::new(0, num_vertices as u32);
    for _ in 0..num_triangles {
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
    let morton_code_generator = MortonCodeGenerator::new(scene_min, scene_max);
    // region: creating buffers
    let lut: Vec<u32> = morton_code_generator
        .lut
        .iter()
        .flat_map(|x| x.iter().flat_map(|y| [(*y << 32) as u32, *y as u32]))
        .collect();
    if lut.len() != 4608 {
        panic!("lut wrong length");
    }

    let size_lut: Vec<u32> = morton_code_generator
        .size_lut
        .iter()
        .flat_map(|x| [(*x << 32) as u32, *x as u32])
        .collect();
    let morton_uniforms = MortonUniforms {
        lut: lut.try_into().unwrap(),
        size_lut: size_lut.try_into().unwrap(),
        morton_index_scale: morton_code_generator.morton_index_scale,
        offset: morton_code_generator.offset,
        size_multiplier: morton_code_generator.size_multiplier,
        multiplier: morton_code_generator.multiplier,
    };
    let morton_buffer = produce_buffer(
        morton_uniforms,
        &device,
        "morton code buffer",
        BufferUsages::STORAGE,
    );
    let vertices_buffer =
        produce_buffer(vertices, &device, "vertices buffer", BufferUsages::STORAGE);
    let index_buffer = produce_buffer(triangles, &device, "indices buffer", BufferUsages::STORAGE);
    let morton_code_size: u64 = (std::mem::size_of::<u64>()).try_into().unwrap();
    let morton_code_buffer_size = morton_code_size * num_triangles as u64;
    let morton_code_buffer_desc = wgpu::BufferDescriptor {
        label: None,
        size: morton_code_buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    };
    let morton_code_buffer = device.create_buffer(&morton_code_buffer_desc);
    // endregion
    let morton_coder_src = include_str!("morton_code.wgsl");
    let morton_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Morton coder"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(morton_coder_src)),
    });
    // region: bind group layouts
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("bind group layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    // endregion
    // region: compute pipeline
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &morton_module,
        entry_point: "morton_code",
    });
    // endregion
    // region: bind groups
    let bind_group: BindGroup = device.create_bind_group(&BindGroupDescriptor {
        label: Some("bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(morton_buffer.as_entire_buffer_binding()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(vertices_buffer.as_entire_buffer_binding()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(index_buffer.as_entire_buffer_binding()),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: BindingResource::Buffer(morton_code_buffer.as_entire_buffer_binding()),
            },
        ],
    });
    // endregion
    // region: encoder
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let compute_pass_desc = wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
        };
        let mut compute_pass = encoder.begin_compute_pass(&compute_pass_desc);
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(num_triangles.try_into().unwrap(), 0, 0);
    }
    // endregion
    queue.submit(Some(encoder.finish()));
    // We need to scope the mapping variables so that we can
    // unmap the buffer
    {
        let buffer_slice = morton_code_buffer.slice(..);
        // NOTE: We have to create the mapping THEN device.poll() before await
        // the future. Otherwise the application will freeze.
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.receive().await.unwrap().unwrap();
    }
    morton_code_buffer.unmap();
}

fn main() {
    pollster::block_on(run());
}
