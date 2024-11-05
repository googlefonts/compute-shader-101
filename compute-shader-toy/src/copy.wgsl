// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

// A simple vert/frag shader to copy an image to the swapchain.

struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @location(1) @interpolate(flat) dense_end: u32,
    @builtin(position) position: vec4<f32>,
};

struct Config {
    width: u32,
    height: u32,
    strip_height: u32,
}

struct Strip {
    // TODO: probably need path_id here, but we can run this in a mode
    // where we render each path separately.
    xy: u32, // this could be u16's on the Rust side
    col: u32,
    winding: i32,
}

@group(0) @binding(1)
var<uniform> config: Config;

@group(0) @binding(2)
var<storage> strips: array<Strip>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(in_vertex_index & 1u);
    let y = f32(in_vertex_index >> 1u);
    let strip = strips[in_instance_index];
    let next_strip = strips[in_instance_index + 1u];
    let x0 = strip.xy & 0xffffu;
    let y0 = strip.xy >> 16u;
    var width = next_strip.col - strip.col;
    out.dense_end = strip.col + width;
    if next_strip.winding != 0 && y0 == next_strip.xy >> 16u {
        width = (next_strip.xy & 0xffffu) - x0;
    }
    let pix_x = f32(x0) + f32(width) * x;
    let pix_y = f32(y0) + y * f32(config.strip_height);
    let gl_x = (pix_x + 0.5) * 2.0 / f32(config.width) - 1.0;
    let gl_y = 1.0 - (pix_y + 0.5) * 2.0 / f32(config.height);
    out.position = vec4<f32>(gl_x, gl_y, 0.0, 1.0);
    out.tex_coord = vec2<f32>(f32(strip.col) + x * f32(width), y * f32(config.strip_height));
    return out;
}

@group(0) @binding(0)
var<storage> alphas: array<u32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let x = u32(floor(in.tex_coord.x));
    var alpha = 1.0;
    if x < in.dense_end {
        let y = u32(floor(in.tex_coord.y));
        let a = alphas[x];
        alpha = f32((a >> (y * 8u)) & 0xffu) * (1.0 / 255.0);
    }
    return alpha * vec4(1.0, 1.0, 1.0, 1.0);
}
