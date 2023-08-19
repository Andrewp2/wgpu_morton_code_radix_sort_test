use std::borrow::Cow;

use wgpu::ShaderSource;

use wgpu::ShaderModuleDescriptor;

use wgpu::Device;

pub use crate::types::ShaderModules;

pub fn create_all_shader_modules(device: &Device) -> ShaderModules {
    ShaderModules {
        morton_code: device.create_shader_module(ShaderModuleDescriptor {
            label: Some("morton_module"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("./shaders/morton_code.wgsl"))),
        }),
        radix_sort_histogram: device.create_shader_module(ShaderModuleDescriptor {
            label: Some("radix_sort_histogram_module"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "./shaders/radix_sort_histogram.wgsl"
            ))),
        }),
        radix_sort_prefix_large: device.create_shader_module(ShaderModuleDescriptor {
            label: Some("radix_sort_prefix_large_module"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "./shaders/radix_sort_block_sum_large.wgsl"
            ))),
        }),
        radix_sort_prefix_small: device.create_shader_module(ShaderModuleDescriptor {
            label: Some("radix_sort_prefix_small_module"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "./shaders/radix_sort_block_sum_small.wgsl"
            ))),
        }),
        radix_sort_index: device.create_shader_module(ShaderModuleDescriptor {
            label: Some("radix_sort_index_module"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "./shaders/radix_sort_index.wgsl"
            ))),
        }),
    }
}
