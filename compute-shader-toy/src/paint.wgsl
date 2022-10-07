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

struct Params {
    width: u32,
    height: u32,
    iTime: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_ix: vec3<u32>) {
    let fragCoord: vec2<f32> = vec2<f32>(global_ix.xy) / vec2<f32>(f32(params.width), f32(params.height))
        - vec2<f32>(0.5, 0.5);

    // Shadertoy-like code can go here.
    let fragColor: vec4<f32> = vec4<f32>(fragCoord.x + 0.5, fragCoord.y + 0.5, sin(params.iTime), 1.0);

    textureStore(outputTex, vec2<i32>(global_ix.xy), fragColor);
}
