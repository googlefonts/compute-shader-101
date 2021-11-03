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

struct Element {
    data: atomic<u32>;
    flag: atomic<u32>;
};

[[block]]
struct DataBuf {
    data: [[stride(8)]] array<Element>;
};

[[block]]
struct ControlBuf {
    strategy: u32;
    failures: atomic<u32>;
};

[[group(0), binding(0)]]
var<storage, read_write> data_buf: DataBuf;

[[group(0), binding(1)]]
var<storage, read_write> control_buf: ControlBuf;

// Put the flag in quite a different place than the data, which
// should increase the number of failures, as they likely won't
// be on the same cache line.
fn permute_flag_ix(data_ix: u32) -> u32 {
    return (data_ix * 419u) & 0xffffu;
}

[[stage(compute), workgroup_size(256)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let ix = global_id.x;

    let wr_flag_ix = permute_flag_ix(ix);
    atomicStore(&data_buf.data[ix].data, 1u);
    storageBarrier(); // release semantics for writing flag
    atomicStore(&data_buf.data[wr_flag_ix].flag, 1u);

    // Read from a different workgroup
    let read_ix = (ix * 4099u) & 0xffffu;
    let read_flag_ix = permute_flag_ix(read_ix);

    let flag = atomicLoad(&data_buf.data[read_flag_ix].flag);
    storageBarrier(); // acquire semantics for reading flag
    let data = atomicLoad(&data_buf.data[read_ix].data);
    if (flag > data) {
        let unused = atomicAdd(&control_buf.failures, 1u);
    }
}
