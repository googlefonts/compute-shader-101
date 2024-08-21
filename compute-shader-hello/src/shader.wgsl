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
    // An observation: loc could be fetched from tile
    la: Loc,
    fa: u32,
    lb: Loc,
    fb: u32,
    cols: u32,
}

@group(0) @binding(0)
var<storage> tiles: array<Tile>;

@group(0) @binding(1)
var<storage, read_write> counts: array<Count>;

@group(0) @binding(2)
var<storage, read_write> footprints: array<u32>;

const WG = 256u;

var<workgroup> sh_histo: array<u32, WG>;
var<workgroup> sh_start: array<u32, WG>;
// We use atomics for simplicity, but could do monoid reduction
var<workgroup> sh_cols: atomic<u32>;

fn expand_footprint(fp: u32) -> u32 {
    return (fp * 0x204081) & 0x1010101;
}

// Note: histo must be nonzero. Fills in missing interior.
fn from_expanded(histo: u32) -> u32 {
    return (2u << (firstLeadingBit(histo) / 8u)) - (1u << (firstTrailingBit(histo) / 8u));
}

fn loc_eq(a: Loc, b: Loc) -> bool {
    return all(vec2(a.path_id, a.xy) == vec2(b.path_id, b.xy));
}

fn same_strip(a: Loc, b: Loc) -> bool {
    return a.path_id == b.path_id && ((b.xy - a.xy) / 2) == 0;
}

@compute
@workgroup_size(WG)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    atomicStore(&sh_cols, 0u);
    let tile = tiles[global_id.x];
    // inclusive prefix sum of histo
    var sum = expand_footprint(tile.footprint);
    var start = 0u;
    // Note: this matches slice_alloc, but another viable choice is to
    // use global_id; this would catch starts at beginning of partition
    if local_id.x > 0 && !loc_eq(tile.loc, tiles[global_id.x - 1].loc) {
        start = local_id.x;
    }
    for (var i = 0u; i < firstTrailingBit(WG); i++) {
        sh_histo[local_id.x] = sum;
        sh_start[local_id.x] = start;
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            sum += sh_histo[local_id.x - (1u << i)];
            start = max(start, sh_start[local_id.x - (1u << i)]);
        }
        workgroupBarrier();
    }
    sh_histo[local_id.x] = sum;
    sh_start[local_id.x] = start;
    workgroupBarrier();
    if local_id.x == WG - 1 {
        let first = sh_histo[max(start, 1u) - 1];
        var last_histo = sum - first;
        if start == 0 {
            last_histo |= first;
        }
        let fp = from_expanded(last_histo);
        counts[wg_id.x].fb = fp;
        if start == 0 {
            counts[wg_id.x].fa = fp;
        }
        counts[wg_id.x].lb = tile.loc;
    } else if !loc_eq(tile.loc, tiles[global_id.x + 1].loc) {
        // last tile in segment
        if start == 0 {
            counts[wg_id.x].fa = from_expanded(sum);
        } else {
            let histo = sum - sh_histo[start - 1];
            let fp = from_expanded(histo);
            footprints[wg_id.x * WG + start] = fp;
            atomicAdd(&sh_cols, countOneBits(fp));
        }
    }
    workgroupBarrier();
    if local_id.x == 0 {
        counts[wg_id.x].la = tile.loc;
        counts[wg_id.x].cols = atomicLoad(&sh_cols);
    }
}
