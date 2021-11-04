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

[[block]]
struct DataBuf {
    data: [[stride(4)]] array<u32>;
};

[[block]]
struct StateBuf {
    state: [[stride(4)]] array<atomic<u32>>;
};

[[group(0), binding(0)]]
var<storage, read_write> main_buf: DataBuf;

[[group(0), binding(1)]]
var<storage, read_write> state_buf: StateBuf;

let FLAG_NOT_READY = 0u;
let FLAG_AGGREGATE_READY = 1u;
let FLAG_PREFIX_READY = 2u;

let workgroup_size: u32 = 512u;
let N_SEQ = 8u;

var<workgroup> part_id: u32;
var<workgroup> scratch: array<u32, workgroup_size>;
var<workgroup> shared_prefix: u32;
var<workgroup> shared_flag: u32;

[[stage(compute), workgroup_size(512)]]
fn main([[builtin(local_invocation_id)]] local_id: vec3<u32>) {
    if (local_id.x == 0u) {
        part_id = atomicAdd(&state_buf.state[0], 1u);
    }
    workgroupBarrier();
    let my_part_id = part_id;
    let mem_base = my_part_id * workgroup_size;
    var local: array<u32, N_SEQ>;
    var el = main_buf.data[(mem_base + local_id.x) * N_SEQ];
    local[0] = el;
    for (var i: u32 = 1u; i < N_SEQ; i = i + 1u) {
        el = el + main_buf.data[(mem_base + local_id.x) * N_SEQ + i];
        local[i] = el;
    }
    scratch[local_id.x] = el;
    // This must be lg2(workgroup_size)
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        workgroupBarrier();
        if (local_id.x >= (1u << i)) {
            el = el + scratch[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        scratch[local_id.x] = el;
    }
    var exclusive_prefix = 0u;

    var flag = FLAG_AGGREGATE_READY;
    if (local_id.x == workgroup_size - 1u) {
        atomicStore(&state_buf.state[my_part_id * 3u + 2u], el);
        if (my_part_id == 0u) {
            atomicStore(&state_buf.state[my_part_id * 3u + 3u], el);
            flag = FLAG_PREFIX_READY;
        }
    }
    // make sure these barriers are in uniform control flow
    storageBarrier();
    if (local_id.x == workgroup_size - 1u) {
        atomicStore(&state_buf.state[my_part_id * 3u + 1u], flag);
    }

    if (my_part_id != 0u) {
        // decoupled look-back
        var look_back_ix = my_part_id - 1u;
        loop {
            if (local_id.x == workgroup_size - 1u) {
                shared_flag = atomicOr(&state_buf.state[look_back_ix * 3u + 1u], 0u);
            }
            workgroupBarrier();
            flag = shared_flag;
            storageBarrier();
            if (flag == FLAG_PREFIX_READY) {
                if (local_id.x == workgroup_size - 1u) {
                    let their_prefix = atomicOr(&state_buf.state[look_back_ix * 3u + 3u], 0u);
                    exclusive_prefix = their_prefix + exclusive_prefix;
                }
                break;
            } elseif (flag == FLAG_AGGREGATE_READY) {
                if (local_id.x == workgroup_size - 1u) {
                    let their_agg = atomicOr(&state_buf.state[look_back_ix * 3u + 2u], 0u);
                    exclusive_prefix = their_agg + exclusive_prefix;
                }
                look_back_ix = look_back_ix - 1u;
            }
            // else spin
        }

        // compute inclusive prefix
        if (local_id.x == workgroup_size - 1u) {
            let inclusive_prefix = exclusive_prefix + el;
            shared_prefix = exclusive_prefix;
            atomicStore(&state_buf.state[my_part_id * 3u + 3u], inclusive_prefix);
        }
        storageBarrier();
        if (local_id.x == workgroup_size - 1u) {
            atomicStore(&state_buf.state[my_part_id * 3u + 1u], FLAG_PREFIX_READY);
        }
    }
    var prefix = 0u;
    workgroupBarrier();
    if (my_part_id != 0u) {
        prefix = shared_prefix;
    }

    // do final output
    for (var i: u32 = 0u; i < N_SEQ; i = i + 1u) {
        var old = 0u;
        if (local_id.x > 0u) {
            old = scratch[local_id.x - 1u];
        }
        main_buf.data[(mem_base + local_id.x) * N_SEQ + i] = prefix + old + local[i];
    }
}
