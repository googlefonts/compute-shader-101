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

struct Tile {
    loc: Loc,
    // TODO: actual tile representation, footprint is standin
    footprint: u32,
}

struct Loc {
    path_id: u32,
    xy: u32, // two u16's packed
}

struct Count {
    la: Loc,
    fa: u32,
    lb: Loc,
    fb: u32,
    cols: u32,
}

@group(0)
@binding(0)
var<storage> tiles: array<Tile>;

@group(0) @binding(1)
var<storage, read_write> counts: array<Count>;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile = tiles[global_id.x];
    counts[global_id.x] = Count(tile.loc, tile.footprint, Loc(), 2, global_id.x);
}
