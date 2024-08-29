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
    strip_start: u32,
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
    strip_start: u32,
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
var<storage, read_write> strips: array<Strip>;

const WG = 256u;

var<workgroup> sh_histo: array<u32, WG>;
var<workgroup> sh_start: array<u32, WG>;
var<workgroup> sh_delta: array<i32, WG>;
// We use atomics for simplicity, but could do monoid reduction
var<workgroup> sh_col_count: atomic<u32>;
var<workgroup> sh_strips: atomic<u32>;
var<workgroup> sh_row_start: atomic<u32>;

var<workgroup> sh_cols: array<u32, WG>;

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
    return a.path_id == b.path_id && b.xy - a.xy <= 1;
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
    } else if (breaks & 2) != 0 {
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
    let strip_start = max(a.strip_start, b.strip_start);
    return MiniCount(fa, fb, cols, strips, delta, strip_start);
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
        atomicStore(&sh_col_count, 0u);
        atomicStore(&sh_strips, 0u);
        atomicStore(&sh_row_start, 0u);
    }
    workgroupBarrier();
    // Clever encoding: this the index of the start of the segment * 2,
    // plus 1 if the segment start is also a strip boundary
    var start_s = 0u;
    // Note: this matches slice_alloc, but another viable choice is to
    // use global_id; this would catch starts at beginning of partition
    var fp = tile.footprint;
    if local_id.x > 0 {
        let prev_tile = tiles[global_id.x - 1];
        if !loc_eq(prev_tile.loc, tile.loc) {
            start_s = local_id.x * 2;
            if !same_strip(prev_tile.loc, tile.loc) {
                start_s += 1u;
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
        if is_end && same_strip(tile.loc, next_tile.loc) {
            // same strip but different segment, extend fp toward msb
            fp |= 8u;
        }
    }
    // inclusive prefix sum of histo
    var sum = expand_footprint(fp);
    var delta = tile.delta;
    for (var i = 0u; i < firstTrailingBit(WG); i++) {
        sh_histo[local_id.x] = sum;
        sh_start[local_id.x] = start_s;
        sh_delta[local_id.x] = delta;
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            sum += sh_histo[local_id.x - (1u << i)];
            start_s = max(start_s, sh_start[local_id.x - (1u << i)]);
            delta += sh_delta[local_id.x - (1u << i)];
        }
        workgroupBarrier();
    }
    sh_histo[local_id.x] = sum;
    sh_delta[local_id.x] = delta;
    // don't need to store sh_start
    let start = start_s / 2u;
    workgroupBarrier();
    if local_id.x == WG - 1 {
        counts[wg_id.x].strip_start = start_s;
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
            atomicAdd(&sh_col_count, count_footprint(fp));
        }
    }
    workgroupBarrier();
    if local_id.x == 0 {
        counts[wg_id.x].la = tile.loc;
        counts[wg_id.x].cols = atomicLoad(&sh_col_count);
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
    var strip_start = c.strip_start;
    if strip_start != 0 {
        strip_start += global_id.x * WG;
    }
    var count = MiniCount(c.fa, c.fb, c.cols, c.strips, c.delta, strip_start);
    let this_la = c.la;
    var bla = this_la;
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
    // sh_count now has inclusive scan of count semigroup
    var cols = 0u;
    var delta = 0;
    var strip_count = 0u;
    var fp = 0u;
    if local_id.x > 0 {
        fp = sh_count[local_id.x - 1].fb;
        cols = count_footprint(sh_count[local_id.x - 1].fa) + sh_count[local_id.x - 1].cols;
        strip_count = sh_count[local_id.x - 1].strips;
        let prev_lb = counts[global_id.x - 1].lb;
        if same_row(prev_lb, this_la) {
            delta = sh_count[local_id.x - 1].delta;
        }
        let strip_boundary = !same_strip(prev_lb, this_la);
        let seg_boundary = !loc_eq(prev_lb, this_la);
        if !seg_boundary {
            fp |= fa;
        }
        let strip_start = sh_count[local_id.x - 1].strip_start;
        if (max(strip_start, 1u) & 1) != 0 && !loc_eq(prev_lb, blb) {
            strips[strip_count].xy = 4 * prev_lb.xy + firstTrailingBit(fp);
            strips[strip_count].col = cols;
            strips[strip_count].sparse_width = 16u;
        }
        if seg_boundary {
            if !strip_boundary {
                fp |= 8u;
            }
            cols += count_footprint(fp);
            fp = fa;
            if strip_boundary {
                strip_count += 1u;
                if !loc_eq(this_la, blb) {
                    strips[strip_count].xy = 4 * this_la.xy + firstTrailingBit(fp);
                    strips[strip_count].col = cols;
                    strips[strip_count].sparse_width = 17u;
                }
            } else {
                fp |= 1u;
            }
        }
    } else {
        fp = fa;
        if !loc_eq(this_la, blb) {
            strips[0].xy = 4 * this_la.xy + firstTrailingBit(fp);
            strips[0].col = 0u;
        }
    }
    cols += count_footprint(fp);
    counts[global_id.x].cols = cols;
    counts[global_id.x].delta = delta;
    // TODO: adjust strip count for boundaries
    counts[global_id.x].strips = strip_count;
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
    // Note: could be global_ix, which would see boundaries at partition start
    if local_id.x > 0 {
        let prev_tile = tiles[global_id.x - 1];
        is_strip_start = !same_strip(prev_tile.loc, tile.loc);
        is_seg_start = !loc_eq(prev_tile.loc, tile.loc);
    }
    var strip_count = u32(is_strip_start);
    var start_s = select(0u, local_id.x * 2 + strip_count, is_seg_start);
    var sum = expand_footprint(tile.footprint);
    for (var i = 0u; i < firstTrailingBit(WG); i++) {
        sh_strip_count[local_id.x] = strip_count;
        sh_histo[local_id.x] = sum;
        sh_start[local_id.x] = start_s;
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            strip_count += sh_strip_count[local_id.x - (1u << i)];
            sum += sh_histo[local_id.x - (1u << i)];
            start_s = max(start_s, sh_start[local_id.x - (1u << i)]);
        }
        workgroupBarrier();
    }
    // debug info; remove
    sh_histo[local_id.x] = sum;
    workgroupBarrier();
    var is_end = false;
    var cols_in = 0u;
    if local_id.x < WG - 1 {
        let next_tile = tiles[global_id.x + 1];
        is_end = !loc_eq(tile.loc, next_tile.loc);
        if is_end && start_s != 0 {
            let histo = sum - sh_histo[start_s / 2 - 1];
            var fp = from_expanded(histo);
            if (start_s & 1) == 0 {
                fp |= 1u;
            }
            if same_strip(tile.loc, next_tile.loc) {
                fp |= 8u;
            }
            cols_in = count_footprint(fp);
        }
    }
    // at the last tile of a complete segment, cols_in contains the column count
    var cols = cols_in;
    for (var i = 0u; i < firstTrailingBit(WG); i++) {
        sh_cols[local_id.x] = cols;
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            cols += sh_cols[local_id.x - (1u << i)];
        }
        workgroupBarrier();
    }
    // cols is inclusive prefix sum of column counts
    if local_id.x < WG - 1 {
        if is_end && (start_s & 1) != 0 {
            // end of first segment of strip, output strip start
            let strip_ix = counts[wg_id.x].strips + strip_count;
            let histo = sum - sh_histo[(start_s / 2) - 1];
            let xy = tile.loc.xy * 4 + firstTrailingBit(histo) / 8;
            strips[strip_ix].xy = xy;
            let col = counts[wg_id.x].cols + cols - cols_in;
            // store start column
            // note: it would also make sense to store at strip end
            strips[strip_ix].col = col;
            // this is for debug
            strips[strip_ix].width = wg_id.x;
        }
    }
}
