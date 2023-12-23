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
    num_wgs_with_additional_blocks: u32,
    num_reduce_wg_per_bin: u32,
    num_scan_values: u32,
    shift: u32,
}

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> src: array<u32>;

@group(0) @binding(2)
var<storage, read_write> counts: array<u32>;

@group(0) @binding(3)
var<storage, read_write> reduced: array<u32>;

@group(0) @binding(4)
var<storage, read_write> out: array<u32>;

const OFFSET = 42u;

const WG = 256u;
const BITS_PER_PASS = 4u;
const BIN_COUNT = 1u << BITS_PER_PASS;
const HISTOGRAM_SIZE = WG * BIN_COUNT;
const ELEMENTS_PER_THREAD = 4u;
const BLOCK_SIZE = WG * ELEMENTS_PER_THREAD;

var<workgroup> histogram: array<u32, HISTOGRAM_SIZE>;

@compute
@workgroup_size(WG)
fn count(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    for (var i = 0u; i < BIN_COUNT; i++) {
        histogram[i * WG + local_id.x] = 0u;
    }
    workgroupBarrier();
    var num_blocks = config.num_blocks_per_wg;
    var wg_block_start = BLOCK_SIZE * num_blocks * group_id.x;
    let num_not_additional = config.num_wgs - config.num_wgs_with_additional_blocks;
    if group_id.x >= num_not_additional {
        wg_block_start += (group_id.x - num_not_additional) * BLOCK_SIZE;
        num_blocks += 1u;
    }
    var block_index = wg_block_start + local_id.x;
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
        block_index += BLOCK_SIZE;
    }
    workgroupBarrier();
    if local_id.x < BIN_COUNT {
        var sum = 0u;
        for (var i = 0u; i < WG; i++) {
            sum += histogram[local_id.x * WG + i];
        }
        counts[local_id.x * config.num_wgs + group_id.x] = sum;
    }
}

var<workgroup> sums: array<u32, WG>;

@compute
@workgroup_size(WG)
fn reduce(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let bin_id = group_id.x / config.num_reduce_wg_per_bin;
    let bin_offset = bin_id * config.num_wgs;
    let base_index = (group_id.x % config.num_reduce_wg_per_bin) * BLOCK_SIZE;
    var sum = 0u;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * WG + local_id.x;
        if data_index < config.num_wgs {
            sum += counts[bin_offset + data_index];
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
        reduced[group_id.x] = sum;
    }
}

var<workgroup> lds: array<array<u32, WG>, ELEMENTS_PER_THREAD>;

@compute
@workgroup_size(WG)
fn scan(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    // We only dispatch a single wg, so I think this is always 0
    let base_index = BLOCK_SIZE * group_id.x;
    let num_values_to_scan = config.num_scan_values;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        lds[row][col] = reduced[data_index];
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
    sum = 0u;
    if local_id.x > 0u {
        sum = sums[local_id.x - 1u];
    }
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        lds[i][local_id.x] += sum;
    }
    // lds now contains exclusive prefix sum
    workgroupBarrier();
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        if data_index < num_values_to_scan {
            reduced[data_index] = lds[row][col];
        }
    }
}

@compute
@workgroup_size(WG)
fn scan_add(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let bin_id = group_id.x / config.num_reduce_wg_per_bin;
    let bin_offset = bin_id * config.num_wgs;
    let base_index = (group_id.x % config.num_reduce_wg_per_bin) * ELEMENTS_PER_THREAD * WG;
    let num_values_to_scan = config.num_wgs;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        // This is not gated, we let robustness do it for us
        lds[row][col] = counts[bin_offset + data_index];
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
    sum = reduced[group_id.x];
    if local_id.x > 0u {
        sum += sums[local_id.x - 1u];
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
        if data_index < num_values_to_scan {
            counts[bin_offset + data_index] = lds[row][col];
        }
    }
}

var<workgroup> bin_offset_cache: array<u32, WG>;

var<workgroup> local_histogram: array<atomic<u32>, BIN_COUNT>;

@compute
@workgroup_size(WG)
fn scatter (
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    if local_id.x < BIN_COUNT {
        bin_offset_cache[local_id.x] = sums[local_id.x * config.num_wgs + group_id.x];
    }
    workgroupBarrier();
    let wg_block_start = BLOCK_SIZE * config.num_blocks_per_wg * group_id.x;
    let num_blocks = config.num_blocks_per_wg;
    // TODO: handle additional as above
    let block_index = wg_block_start + local_id.x;
    for (var block_count = 0u; block_count < num_blocks; block_count++) {
        var data_index = block_index;
        for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
            if local_id.x < BIN_COUNT {
                local_histogram[local_id.x] = 0u;
            }
            var local_key = ~0u;
            if data_index < config.num_keys {
                local_key = src[data_index + i];
            }
            for (var bit_shift = 0u; bit_shift < BITS_PER_PASS; bit_shift += 2u) {
                let key_index = (local_key >> config.shift) & 0xfu;
                let bit_key = (key_index >> bit_shift) & 3u;
                var packed_histogram = 1u << (bit_key * 8u);
                // workgroup prefix sum
                var sum = packed_histogram;
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
                packed_histogram = sums[WG - 1u];
                packed_histogram = (packed_histogram << 8u) + (packed_histogram << 16u) + (packed_histogram << 24u);
                var local_sum = packed_histogram;
                if local_id.x > 0u {
                    local_sum += sums[local_id.x - 1u];
                }
                let key_offset = (local_sum >> (bit_key * 8u)) & 0xffu;
                sums[key_offset] = local_key;
                workgroupBarrier();
                local_key = sums[local_id.x];
                // TODO: handle value here (if we had it)
                workgroupBarrier();
            }
            let key_index = (local_key >> config.shift) & 0xfu;
            atomicAdd(&local_histogram[key_index], 1u);
            workgroupBarrier();
            var histogram_local_sum = 0u;
            if local_id.x < BIN_COUNT {
                histogram_local_sum = local_histogram[local_id.x];
            }
            // workgroup prefix sum of histogram
            var histogram_prefix_sum = histogram_local_sum;
            if local_id.x < BIN_COUNT {
                sums[local_id.x] = histogram_prefix_sum;
            }
            for (var i = 0u; i < 4u; i++) {
                workgroupBarrier();
                if local_id.x >= (1u << i) && local_id.x < BIN_COUNT {
                    histogram_prefix_sum += sums[local_id.x - (1u << i)];
                }
                workgroupBarrier();
                if local_id.x < BIN_COUNT {
                    sums[local_id.x] = histogram_prefix_sum;
                }
            }
            let global_offset = bin_offset_cache[key_index];
            workgroupBarrier();
            var local_offset = local_id.x;
            if key_index > 0u {
                local_offset -= sums[key_index - 1u];
            }
            let total_offset = global_offset + local_offset;
            if total_offset < config.num_keys {
                out[total_offset] = local_key;
            }
            workgroupBarrier();
            if local_id.x < BIN_COUNT {
                bin_offset_cache[local_id.x] += local_histogram[local_id.x];
            }
            data_index += WG;
        }
    }
}
