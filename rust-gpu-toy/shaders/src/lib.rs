#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv)
)]

use rust_gpu_toy_shared::Config;
#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use spirv_std::{
    glam::{vec2, vec4, UVec3, Vec2, Vec4},
    image, Sampler,
};

#[spirv(compute(threads(16, 16)))]
pub fn main(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] config: &Config,
    #[spirv(descriptor_set = 0, binding = 1)] output_buffer: &image::StorageImage2d,
    #[spirv(global_invocation_id)] global_ix: UVec3,
) {
    let frag_coord = global_ix.truncate().as_f32()
        / vec2(config.width as f32, config.height as f32)
        - vec2(0.5, 0.5);

    // Shadertoy like code goes here
    let frag_color = vec4(
        frag_coord.x + 0.5,
        frag_coord.y + 0.5,
        config.time.sin(),
        1.0,
    );

    unsafe { output_buffer.write(global_ix.truncate(), frag_color) }
}

// A simple vert/frag shader to copy an image to the swapchain.

#[spirv(fragment)]
pub fn fs_main(
    #[spirv(descriptor_set = 0, binding = 0)] image: &image::Image2d,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    in_tex_coord: Vec2,
    output: &mut Vec4,
) {
    *output = image.sample(*sampler, in_tex_coord);
}

#[spirv(vertex)]
pub fn vs_main(
    #[spirv(vertex_index)] in_vertex_index: u32,
    #[spirv(instance_index)] in_instance_index: u32,
    #[spirv(position, invariant)] out_pos: &mut Vec4,
    out_tex_coord: &mut Vec2,
) {
    let x = ((in_vertex_index & 1) ^ in_instance_index) as f32;
    let y = ((in_vertex_index >> 1) ^ in_instance_index) as f32;
    *out_pos = vec4(x * 2. - 1., 1. - y * 2., 0., 1.);
    *out_tex_coord = vec2(x, y);
}
