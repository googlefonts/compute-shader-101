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

struct Config {
    num_keys: u32,
    num_blocks_per_wg: u32,
    num_wgs: u32,
    shift: u32,
}

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> src: array<u32>;

@group(0) @binding(2)
var<storage, read_write> counts: array<u32>;

const OFFSET = 42u;

const WG = 256u;
const BITS_PER_PASS = 4u;
const BIN_COUNT = 1u << BITS_PER_PASS;
const HISTOGRAM_SIZE = WG * BIN_COUNT;
const ELEMENTS_PER_THREAD = 1u; // 4 in source

var<workgroup> histogram: array<atomic<u32>, HISTOGRAM_SIZE>;

@compute
@workgroup_size(WG)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // TODO: a more interesting computation than this.
    for (var i = 0u; i < BIN_COUNT; i++) {
        histogram[i * WG + local_id.x] = 0u;
    }
    workgroupBarrier();
    var block_index = local_id.x; // TODO: add threadgroup block start
    let block_size = 1024u;
    let num_blocks = config.num_blocks_per_wg;
    let shift_bit = config.shift;
    for (var block_count = 0u; block_count < num_blocks; block_count++) {
        var data_index = block_index;
        for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
            if data_index < config.num_keys {
                let local_key = (src[data_index] >> shift_bit) & 0xfu;
                atomicAdd(&histogram[local_key * WG + local_id.x], 1u);
            }
            data_index += WG;
        }
        block_index += block_size;
    }
    if local_id.x < BIN_COUNT {
        var sum = 0u;
        for (var i = 0u; i < WG; i++) {
            sum += histogram[local_id.x * WG + i];
        }
        counts[local_id.x] = sum;
    }
}

