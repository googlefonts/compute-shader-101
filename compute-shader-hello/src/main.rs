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
    la: Loc,
    fa: Footprint,
    lb: Loc,
    fb: Footprint,
    cols: u32,
    strips: u32,
    delta: i32,
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
    let input: &[u8] = bytemuck::cast_slice(&tiles);
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_buf"),
        contents: input,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let output_byte_len = (N * std::mem::size_of::<Count>()) as u64;
    let footprint_byte_len = (tiles.len() * std::mem::size_of::<u32>()) as u64;
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
    let footprint_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("footprint_buf"),
        size: footprint_byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let footprint_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("footprint_staging"),
        size: footprint_byte_len,
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
        entry_point: "main",
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
                resource: footprint_buf.as_entire_binding(),
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
    }
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &output_staging, 0, output_byte_len);
    encoder.copy_buffer_to_buffer(&footprint_buf, 0, &footprint_staging, 0, footprint_byte_len);
    if let Some(query_set) = &query_set {
        encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
    }
    encoder.copy_buffer_to_buffer(&query_buf, 0, &query_staging_buf, 0, 16);
    queue.submit(Some(encoder.finish()));

    let buf_slice = output_staging.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    let query_slice = query_staging_buf.slice(..);
    // Assume that both buffers become available at the same time. A more careful
    // approach would be to wait for both notifications to be sent.
    query_slice.map_async(wgpu::MapMode::Read, |_| ());
    println!("pre-poll {:?}", std::time::Instant::now());
    device.poll(wgpu::Maintain::Wait);
    println!("post-poll {:?}", std::time::Instant::now());
    if let Some(Ok(())) = receiver.receive().await {
        let data_raw = &*buf_slice.get_mapped_range();
        let data: &[Count] = bytemuck::cast_slice(data_raw);
        for i in 0..n_wg {
            println!("{i}: {:?}", data[i]);
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
        if rng.gen::<f64>() < 1e-2 {
            y += 1;
            x = 0;
        }
    }
    v
}

fn main() {
    pollster::block_on(run());
}
