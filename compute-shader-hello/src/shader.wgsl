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
const NUM_REDUCE_WG_PER_BIN = 1u; // config?

var<workgroup> histogram: array<u32, HISTOGRAM_SIZE>;

@compute
@workgroup_size(WG)
fn count(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    for (var i = 0u; i < BIN_COUNT; i++) {
        histogram[i * WG + local_id.x] = 0u;
    }
    workgroupBarrier();
    let num_blocks = config.num_blocks_per_wg;
    let wg_block_start = HISTOGRAM_SIZE * num_blocks * wg_id.x;
    // TODO: handle additional blocks
    var block_index = wg_block_start + local_id.x;
    let block_size = 1024u;
    let shift_bit = config.shift;
    for (var block_count = 0u; block_count < num_blocks; block_count++) {
        var data_index = block_index;
        for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
            if data_index < config.num_keys {
                let local_key = (src[data_index] >> shift_bit) & 0xfu;
                histogram[local_key * WG + local_id.x] += 1u;
            }
            data_index += WG;
        }
        block_index += block_size;
    }
    workgroupBarrier();
    if local_id.x < BIN_COUNT {
        var sum = 0u;
        for (var i = 0u; i < WG; i++) {
            sum += histogram[local_id.x * WG + i];
        }
        counts[local_id.x] = sum;
    }
}

var<workgroup> sums: array<u32, WG>;

@compute
@workgroup_size(WG)
fn reduce(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let bin_id = wg_id.x / NUM_REDUCE_WG_PER_BIN;
    let bin_offset = bin_id * config.num_wgs;
    let base_index = (wg_id.x % NUM_REDUCE_WG_PER_BIN) * ELEMENTS_PER_THREAD * WG;
    var sum = 0u;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * WG + local_id.x;
        if data_index < config.num_wgs {
            sum += src[bin_offset + data_index];
        }
    }
    sums[local_id.x] = sum;
    for (var i = 0u; i < 8u; i++) {
        workgroupBarrier();
        if local_id.x < ((WG / 2u) >> i) {
            sum += sums[local_id.x + ((WG / 2u) >> i)];
            sums[local_id.x] = sum;
        }
    }
    if local_id.x == 0u {
        counts[wg_id.x] = sum;
    }
}

var<workgroup> lds: array<array<u32, WG>, ELEMENTS_PER_THREAD>;

@compute
@workgroup_size(WG)
fn scan_add (
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let bin_id = wg_id.x / NUM_REDUCE_WG_PER_BIN;
    let bin_offset = bin_id * config.num_wgs;
    let base_index = (wg_id.x % NUM_REDUCE_WG_PER_BIN) * ELEMENTS_PER_THREAD * WG;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        lds[row][col] = src[bin_offset + data_index]; // TODO: gate?
    }
    workgroupBarrier();
    var sum = 0u;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let tmp = lds[i][local_id.x];
        lds[i][local_id.x] = sum;
        sum += tmp;
    }
    // workgroup prefix sum
    sums[local_id.x] = sum;
    for (var i = 0u; i < 8u; i++) {
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            sum += sums[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        sums[local_id.x] = sum;
    }
    workgroupBarrier();
    // TODO: load partial sum here
    sum = 0u;
    if local_id.x > 0u {
        sum = sums[local_id.x - 1u];
    }
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        lds[i][local_id.x] += sum;
    }
    // lds now contains exclusive prefix sum
    // Note: storing inclusive might be slightly cheaper here
    workgroupBarrier();
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        counts[bin_offset + data_index] = lds[row][col]; // TODO: gate?
    }
}

var<workgroup> bin_offset_cache: array<u32, WG>;

var<workgroup> local_histogram: array<u32, BIN_COUNT>;

var<workgroup> lds_scratch: array<u32, WG>;

