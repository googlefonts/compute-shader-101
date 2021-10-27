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
    data: [[stride(4)]] array<atomic<u32>>;
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
    return (data_ix * 31u) & 0xffffu;
}

[[stage(compute), workgroup_size(256)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let ix = global_id.x;
    // Originally this was passed in, but is now hardcoded, as D3DCompiler
    // thinks control flow becomes nonuniform if it's read from input.
    let n_iter = 1024u;
    let strategy = control_buf.strategy;
    var failures = 0u;
    for (var i: u32 = 0u; i < n_iter; i = i + 1u) {
        let wr_flag_ix = permute_flag_ix(ix);
        atomicStore(&data_buf.data[ix * 2u], i + 1u);
        storageBarrier(); // release semantics for writing flag
        atomicStore(&data_buf.data[wr_flag_ix * 2u + 1u], i + 1u);

        // Read from a different workgroup
        let read_ix = ((ix & 0xffu) << 8u) | (ix >> 8u);
        let read_flag_ix = permute_flag_ix(read_ix);

        let flag = atomicLoad(&data_buf.data[read_flag_ix * 2u + 1u]);
        //let flag = atomicOr(&data_buf.data[read_flag_ix * 2u + 1u], 0u);
        storageBarrier(); // acquire semantics for reading flag
        var data = 0u;
        if (strategy == 0u) {
            data = atomicLoad(&data_buf.data[read_ix * 2u]);
        } else {
            data = atomicOr(&data_buf.data[read_ix * 2u], 0u);
        }
        if (flag > data) {
            failures = failures + 1u;
        }
    }
    let unused = atomicAdd(&control_buf.failures, failures);
}
