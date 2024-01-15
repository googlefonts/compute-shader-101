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
    @builtin(position) position: vec4<f32>,
};

struct Config {
    width: u32,
    height: u32,
    strip_height: u32,
}

struct Strip {
    path_id: u32,
    y: u32,
    x0: u32,
    x1: u32,
}

@group(0) @binding(2)
var<uniform> config: Config;

@group(0) @binding(3)
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
    let pix_x = f32(strip.x0) + (f32(strip.x1) - f32(strip.x0)) * x;
    let pix_y = (f32(strip.y) + y) * f32(config.strip_height);
    let gl_x = (pix_x + 0.5) * 2.0 / f32(config.width) - 1.0;
    let gl_y = 1.0 - (pix_y + 0.5) * 2.0 / f32(config.height);
    out.position = vec4<f32>(gl_x, gl_y, 0.0, 1.0);
    out.tex_coord = vec2<f32>(x, y);
    return out;
}

@group(0) @binding(0)
var r_color: texture_2d<f32>;
@group(0) @binding(1)
var r_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(r_color, r_sampler, in.tex_coord);
}
