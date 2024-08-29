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

//! A simple application to run a compute shader.

use std::time::Instant;

use rand::{thread_rng, Rng};
use wgpu::{util::DeviceExt, PipelineCompilationOptions};

use bytemuck::{self, Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Footprint(u32);

#[repr(C)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Pod, Zeroable)]
struct Loc {
    path_id: u32,
    x: u16,
    y: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Tile {
    loc: Loc,
    // A real tile would have a line segment, and we'd derive a footprint from it,
    // but we're just interested in strip allocation.
    footprint: Footprint,
    delta: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Count {
    fa: Footprint,
    fb: Footprint,
    cols: u32,
    strips: u32,
    delta: i32,
    strip_start: u32,
    la: Loc,
    lb: Loc,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Strip {
    xy: u32, // this could be u16's on the Rust side
    col: u32,
    // maybe don't need, look at start of next strip
    width: u32,
    sparse_width: u32,
}

async fn run() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let features = adapter.features();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();
    // TODO: re-enable timestamp queries
    let query_set = if false && features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        Some(device.create_query_set(&wgpu::QuerySetDescriptor {
            count: 2,
            ty: wgpu::QueryType::Timestamp,
            label: None,
        }))
    } else {
        None
    };

    let start_instant = Instant::now();
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        //source: wgpu::ShaderSource::SpirV(bytes_to_u32(include_bytes!("alu.spv")).into()),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    println!("shader compilation {:?}", start_instant.elapsed());
    const N: usize = 256;
    //let input_v = (0..256).map(|i| i as f32).collect::<Vec<_>>();
    let tiles = gen_tiles(256);
    let strips = mk_strips(&tiles);
    // for i in 248..256 {
    //    println!("{i}: {:?}", tiles[i]);
    // }
    for strip in strips.iter().take(64) {
        println!("{strip:x?}");
    }
    let input: &[u8] = bytemuck::cast_slice(&tiles);
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_buf"),
        contents: input,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let output_byte_len = (N * std::mem::size_of::<Count>()) as u64;
    let strip_byte_len = 1 << 16;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_buf"),
        size: output_byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_staging"),
        size: output_byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let strip_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("strip_buf"),
        size: strip_byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let strip_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("strip_staging"),
        size: strip_byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let query_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });
    let query_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "count_reduce",
        cache: None,
        compilation_options: PipelineCompilationOptions::default(),
    });
    let scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "count_scan",
        cache: None,
        compilation_options: PipelineCompilationOptions::default(),
    });
    let mk_strip_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "mk_strips",
        cache: None,
        compilation_options: PipelineCompilationOptions::default(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: strip_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 0);
    }
    // TODO: launch multiple workgroups
    let n_wg = tiles.len() / 256;
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(n_wg as u32, 1, 1);
        cpass.set_pipeline(&scan_pipeline);
        cpass.dispatch_workgroups(1, 1, 1);
        cpass.set_pipeline(&mk_strip_pipeline);
        cpass.dispatch_workgroups(n_wg as u32, 1, 1);
    }
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &output_staging, 0, output_byte_len);
    encoder.copy_buffer_to_buffer(&strip_buf, 0, &strip_staging, 0, strip_byte_len);
    if let Some(query_set) = &query_set {
        encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
    }
    encoder.copy_buffer_to_buffer(&query_buf, 0, &query_staging_buf, 0, 16);
    queue.submit(Some(encoder.finish()));

    let count_slice = output_staging.slice(..);
    let buf_slice = strip_staging.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    count_slice.map_async(wgpu::MapMode::Read, |_| ());
    buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    let query_slice = query_staging_buf.slice(..);
    // Assume that both buffers become available at the same time. A more careful
    // approach would be to wait for both notifications to be sent.
    query_slice.map_async(wgpu::MapMode::Read, |_| ());
    device.poll(wgpu::Maintain::Wait);
    if let Some(Ok(())) = receiver.receive().await {
        let count_raw = count_slice.get_mapped_range();
        let count_data: &[Count] = bytemuck::cast_slice(&count_raw);
        for i in 0..n_wg {
            println!("{i}: {:?}", count_data[i]);
            // let count = count_reduce(&tiles[i * 256..][..256]);
            //println!("!: {:?}", count);
        }
        let data_raw = buf_slice.get_mapped_range();
        let data: &[Strip] = bytemuck::cast_slice(&data_raw);
        for i in 0..64 {
            println!("{i}: {:x?}", data[i]);
        }
    }
    if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        let ts_period = queue.get_timestamp_period();
        let ts_data_raw = &*query_slice.get_mapped_range();
        let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
        println!(
            "compute shader elapsed: {:?}ms",
            (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
        );
    }
}

/// Generate plausible test data.
fn gen_tiles(n_strips: usize) -> Vec<Tile> {
    let mut rng = thread_rng();
    let mut v = vec![];
    let mut x = 0;
    let mut y = 0;
    // We're not modifying this, but perhaps we should.
    let path_id = 0;
    for _ in 0..n_strips {
        let strip_width = (1.0 / rng.gen_range(1e-3..2f64)).ceil() as usize;
        //println!("{strip_width}");
        let start_x = rng.gen_range(0..4);
        for i in 0..strip_width {
            let loc = Loc { x, y, path_id };
            let mut fp = 15u8;
            if i == 0 {
                fp = (15 << start_x) & 15;
            }
            let lsb = fp & fp.wrapping_neg();
            // TODO: refine end also
            // TODO: n_reps distribution should probably have long tail
            let n_reps = rng.gen_range(1..4);
            for j in 0..n_reps {
                let this_fp = if n_reps == 1 || lsb == fp {
                    fp
                } else if j == n_reps - 1 {
                    fp & !lsb
                } else {
                    lsb
                };
                let footprint = Footprint(this_fp as u32);
                let delta = rng.gen_range(-1..=1);
                let tile = Tile { loc, footprint, delta };
                v.push(tile);
            }
            x += 1;
        }
        x += rng.gen_range(0..=1);
        if rng.gen::<f64>() < 1e-2 {
            y += 1;
            x = 0;
        }
    }
    v
}

impl Loc {
    fn same_strip(&self, other: &Self) -> bool {
        self.same_row(other) && (other.x - self.x) / 2 == 0
    }

    fn same_row(&self, other: &Self) -> bool {
        self.path_id == other.path_id && self.y == other.y
    }
}

/// Make strips from tiles.
///
/// This is the CPU implementation of tile allocation, useful mostly for
/// validating the GPU implementation. It can likely be adapted to a CPU
/// shader.
fn mk_strips(tiles: &[Tile]) -> Vec<Strip> {
    let mut strips = vec![];
    let mut strip_start = true;
    let mut cols = 0;
    let mut prev_tile = &tiles[0];
    let mut fp = prev_tile.footprint.0;
    for tile in &tiles[1..] {
        if prev_tile.loc != tile.loc {
            let same_strip = prev_tile.loc.same_strip(&tile.loc);
            if same_strip {
                fp |= 8;
            }
            if strip_start {
                let xy = (1 << 18) * prev_tile.loc.y as u32 + 4 * prev_tile.loc.x as u32 + fp.trailing_zeros();
                let strip = Strip { xy, col: cols, width: 0, sparse_width: 0 };
                strips.push(strip);
            }
            let col_count = 32 - (fp.leading_zeros() + fp.trailing_zeros());
            cols += col_count;
            fp = if same_strip { 1 } else { 0 };
            strip_start = !same_strip;
        }
        fp |= tile.footprint.0;
        //println!("cols = {cols}, fp = {fp}");
        prev_tile = tile;
    }
    strips
}

#[allow(unused)]
fn count_reduce(tiles: &[Tile]) -> Count {
    let mut start_s = 0;
    let mut prev_tile = &tiles[0];
    let mut fp = prev_tile.footprint.0;
    let mut fa = 0;
    let mut cols = 0;
    let mut strips = 0;
    for i in 1..tiles.len() {
        let tile = &tiles[i];
        if prev_tile.loc != tile.loc {
            let same_strip = prev_tile.loc.same_strip(&tile.loc);
            if same_strip {
                fp |= 8;
            }
            if start_s == 0 {
                fa = fp;
            } else {
                let col_count = 32 - (fp.leading_zeros() + fp.trailing_zeros());
                cols += col_count;
            }
            strips += (!same_strip) as u32;
            start_s = 2 * i as u32 + (!same_strip) as u32;
            fp = if same_strip { 1 } else { 0 };
        }
        fp |= tile.footprint.0;
        prev_tile = tile;
    }
    if start_s == 0 {
        fa = fp;
    }
    // TODO: if we're going to compare to GPU shader, fill interior bits
    Count {
        fa: Footprint(fa),
        fb: Footprint(fp),
        cols,
        strips,
        delta: 0,
        strip_start: start_s,
        la: tiles[0].loc,
        lb: tiles[tiles.len() - 1].loc,
    }
}

fn main() {
    pollster::block_on(run());
}
