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
    delta: i32,
}

struct Loc {
    path_id: u32,
    xy: u32, // two u16's packed
}

struct Count {
    // An observation: loc could be fetched from tile
    fa: u32,
    fb: u32,
    cols: u32,
    strips: u32,
    delta: i32,
    la: Loc,
    lb: Loc,
}

// same struct as Count but missing loc
// could nest the structs
struct MiniCount {
    fa: u32,
    fb: u32,
    cols: u32,
    strips: u32,
    delta: i32,
}

struct Strip {
    xy: u32,
    col: u32,
    // maybe don't need, look at start of next strip
    width: u32,
    sparse_width: u32,
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
var<workgroup> sh_delta: array<i32, WG>;
// We use atomics for simplicity, but could do monoid reduction
var<workgroup> sh_cols: atomic<u32>;
var<workgroup> sh_strips: atomic<u32>;
var<workgroup> sh_row_start: atomic<u32>;

fn expand_footprint(fp: u32) -> u32 {
    return (fp * 0x204081) & 0x1010101;
}

// Note: histo must be nonzero. Fills in missing interior.
fn from_expanded(histo: u32) -> u32 {
    return (2u << (firstLeadingBit(histo) / 8u)) - (1u << (firstTrailingBit(histo) / 8u));
}

fn count_footprint(fp: u32) -> u32 {
    return 1 + firstLeadingBit(fp) - firstTrailingBit(fp);
}

fn loc_eq(a: Loc, b: Loc) -> bool {
    return a.path_id == b.path_id && a.xy == b.xy;
}

fn same_strip(a: Loc, b: Loc) -> bool {
    let diff = b.xy - a.xy;
    return a.path_id == b.path_id && (diff == 0 || diff == 1);
}

fn same_row(a: Loc, b: Loc) -> bool {
    return a.path_id == b.path_id && (a.xy >> 16) == (b.xy >> 16);
}

fn combine_minicount(a: MiniCount, b: MiniCount, ala: Loc, alb: Loc, bla: Loc, blb: Loc) -> MiniCount {
    let breaks = u32(!loc_eq(ala, alb))
        + u32(!loc_eq(alb, bla)) * 2
        + u32(!loc_eq(bla, blb)) * 4;
    var fa = a.fa;
    var afb = a.fb;
    var bfa = b.fa;
    var fb = b.fb;
    var strips = a.strips + b.strips;
    if !same_strip(alb, bla) {
        strips += 1u;
        // strip_start = true
    } else if (breaks & 2) != 0 {
        // strip_start = false
        // same strip but different segment; glue footprints together
        afb |= 8u;
        if (breaks & 1) == 0 {
            fa = afb;
        }
        bfa |= 1u;
        if (breaks & 4) == 0 {
            fb = bfa;
        }
    }
    // else strip_start = a.strip_start
    // if breaks&4, strip_start = b.strip_start
    var cols = a.cols + b.cols;
    if breaks == 3 || breaks == 7 {
        cols += count_footprint(afb);
    }
    if breaks == 6 || breaks == 7 {
        cols += count_footprint(bfa);
    }
    if breaks == 5 {
        cols += count_footprint(afb | bfa);
    }
    if breaks == 0 || breaks == 4 {
        fa |= bfa;
    }
    if breaks == 0 || breaks == 1 {
        fb |= afb;
    }
    var delta = b.delta;
    if same_row(alb, blb) {
        delta += a.delta;
    }
    return MiniCount(fa, fb, cols, strips, delta);
}

// spec for reduce:
// each partition generates count monoid
// only the partition is considered; boundaries at partition
// boundaries are expected to be resolved in scan (where logic
// is cheaper)
//
// currently, we also store footprints of segments where the
// boundaries are within the partition

@compute
@workgroup_size(WG)
fn count_reduce(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let tile = tiles[global_id.x];
    if local_id.x == 0 {
        atomicStore(&sh_cols, 0u);
        atomicStore(&sh_strips, 0u);
        atomicStore(&sh_row_start, 0u);
    }
    workgroupBarrier();
    // inclusive prefix sum of histo
    var start = 0u;
    // Note: this matches slice_alloc, but another viable choice is to
    // use global_id; this would catch starts at beginning of partition
    var fp = tile.footprint;
    if local_id.x > 0 {
        let prev_tile = tiles[global_id.x - 1];
        if !loc_eq(prev_tile.loc, tile.loc) {
            start = local_id.x;
            if !same_strip(prev_tile.loc, tile.loc) {
                atomicAdd(&sh_strips, 1u);
            } else {
                // same strip but different segment, extend fp toward lsb
                fp |= 1u;
            }
        }
        if !same_row(prev_tile.loc, tile.loc) {
            atomicMax(&sh_row_start, local_id.x);
        }
    }
    var is_end = false;
    if local_id.x < WG - 1 {
        let next_tile = tiles[global_id.x + 1];
        is_end = !loc_eq(tile.loc, next_tile.loc);
        if same_strip(tile.loc, next_tile.loc) {
            // same strip but different segment, extend fp toward msb
            fp |= 8u;
        }
    }
    var sum = expand_footprint(fp);
    var delta = tile.delta;
    for (var i = 0u; i < firstTrailingBit(WG); i++) {
        sh_histo[local_id.x] = sum;
        sh_start[local_id.x] = start;
        sh_delta[local_id.x] = delta;
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            sum += sh_histo[local_id.x - (1u << i)];
            start = max(start, sh_start[local_id.x - (1u << i)]);
            delta += sh_delta[local_id.x - (1u << i)];
        }
        workgroupBarrier();
    }
    sh_histo[local_id.x] = sum;
    sh_delta[local_id.x] = delta;
    // don't need to store sh_start
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
        let row_start = atomicLoad(&sh_row_start);
        if row_start != 0 {
            delta -= sh_delta[row_start - 1];
        }
        counts[wg_id.x].delta = delta;
    } else if is_end {
        if start == 0 {
            counts[wg_id.x].fa = from_expanded(sum);
        } else {
            let histo = sum - sh_histo[start - 1];
            let fp = from_expanded(histo);
            footprints[wg_id.x * WG + start] = fp;
            atomicAdd(&sh_cols, count_footprint(fp));
        }
    }
    workgroupBarrier();
    if local_id.x == 0 {
        counts[wg_id.x].la = tile.loc;
        counts[wg_id.x].cols = atomicLoad(&sh_cols);
        counts[wg_id.x].strips = atomicLoad(&sh_strips);
    }
}

var<workgroup> sh_count: array<MiniCount, WG>;

@compute @workgroup_size(WG)
fn count_scan(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // TODO: pull n_tiles from config, gate input and output
    let c = counts[global_id.x];
    var fa = c.fa;
    var count = MiniCount(c.fa, c.fb, c.cols, c.strips, c.delta);
    var bla = c.la;
    let blb = c.lb;
    for (var i = 0u; i < firstTrailingBit(WG); i++) {
        sh_count[local_id.x] = count;
        // sh_count[j] has (j + 1).saturating_sub(2^i)..j + 1
        workgroupBarrier();
        if local_id.x >= 1u << i {
            let offset = (2u << i) - 1;
            let ala = counts[max(global_id.x, offset) - offset].la;
            let alb = counts[global_id.x - (1u << i)].lb;
            let ix = local_id.x - (1u << i);
            count = combine_minicount(sh_count[ix], count, ala, alb, bla, blb);
            bla = ala;
        }
        workgroupBarrier();
    }
    sh_count[local_id.x] = count;
    workgroupBarrier();
    if local_id.x > 0 {
        fa |= sh_count[local_id.x - 1].fb;
    }
    var cols = countOneBits(fa);
    var delta = 0;
    if local_id.x > 0 {
        cols += sh_count[local_id.x - 1].cols;
        if same_row(counts[global_id.x - 1].lb, bla) {
            delta = sh_count[local_id.x - 1].delta;
        }
    }
    counts[global_id.x].cols = cols;
    counts[global_id.x].delta = delta;
    // TODO: store strip count
}

// spec for scan:
// cols is the start for the first segment boundary within the partition
// note: if there is no boundary, cols is irrelevant; merge won't store
// any columns, that will all be done within the reduction fixup

// strips = exclusive prefix sum, but need to figure out behavior when
// a strip boundary coincides with a partition boundary

var<workgroup> sh_strip_count: array<u32, WG>;

// This just outputs strips, but will be combined with actual rendering
@compute @workgroup_size(WG)
fn mk_strips(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let tile = tiles[global_id.x];
    var is_strip_start = false;
    var is_seg_start = false;
    if global_id.x > 0 {
        let prev_tile = tiles[global_id.x - 1];
        if !same_strip(prev_tile.loc, tile.loc) {
            is_strip_start = true;
        }
        is_seg_start = loc_eq(prev_tile.loc, tile.loc);
    }
    var strip_count = u32(is_strip_start);
    // it's possible we should store partial results retained
    // from reduction step, but here we redo the computation,
    // mostly for memory reasons
    var sum = expand_footprint(tile.footprint);
    for (var i = 0u; i < firstTrailingBit(WG); i++) {
        sh_strip_count[local_id.x] = strip_count;
        sh_histo[local_id.x] = sum;
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            strip_count += sh_strip_count[local_id.x - (1u << i)];
            sum += sh_histo[local_id.x - (1u << i)];
        }
        workgroupBarrier();
    }
}
