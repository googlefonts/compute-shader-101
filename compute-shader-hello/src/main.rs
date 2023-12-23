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

use wgpu::util::DeviceExt;

use bytemuck::{self, bytes_of, Pod, Zeroable};

const WG: u32 = 256;
const ELEMENTS_PER_THREAD: u32 = 4;
const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;
const BIN_COUNT: u32 = 16;

#[derive(Clone, Copy, Pod, Zeroable, Debug)]
#[repr(C)]
struct Config {
    num_keys: u32,
    num_blocks_per_wg: u32,
    num_wgs: u32,
    num_wgs_with_additional_blocks: u32,
    num_reduce_wg_per_bin: u32,
    num_scan_values: u32,
    shift: u32,
}

async fn run() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let features = adapter.features();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: features & wgpu::Features::TIMESTAMP_QUERY,
                limits: Default::default(),
            },
            None,
        )
        .await
        .unwrap();
    let query_set = if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        Some(device.create_query_set(&wgpu::QuerySetDescriptor {
            count: 2,
            ty: wgpu::QueryType::Timestamp,
            label: None,
        }))
    } else {
        None
    };

    let n = 1 << 25;
    let input_f = (0..n).map(|_| fastrand::u32(..)).collect::<Vec<_>>();

    // compute buffer and dispatch sizes
    let num_blocks = (n + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
    // TODO: multiple blocks per wg for large problems
    let num_wgs = num_blocks;
    let num_blocks_per_wg = num_blocks / num_wgs;
    let num_wgs_with_additional_blocks = num_blocks % num_wgs;
    // I think the else always has the same value, but fix later.
    let num_reduce_wgs = BIN_COUNT * if BLOCK_SIZE > num_wgs {
        1
    } else {
        (num_wgs + BLOCK_SIZE - 1) / BLOCK_SIZE
    };

    let config = Config {
        num_keys: n,
        num_blocks_per_wg,
        num_wgs,
        num_wgs_with_additional_blocks,
        num_reduce_wg_per_bin: 1,
        num_scan_values: num_reduce_wgs,
        shift: 0,
    };
    println!("{config:?}");

    let start_instant = Instant::now();
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    println!("shader compilation {:?}", start_instant.elapsed());
    let input: &[u8] = bytemuck::cast_slice(&input_f);
    let config_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytes_of(&config),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: input,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let count_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (num_blocks * 64).into(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let reduced_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (BLOCK_SIZE * 4).into(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (num_blocks * 64).into(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            wgpu::BindGroupLayoutEntry {
                binding: 3,
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
    let count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "count",
    });
    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "reduce",
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: config_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: count_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: reduced_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 0);
    }
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&count_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(config.num_wgs, 1, 1);
        cpass.set_pipeline(&reduce_pipeline);
        cpass.dispatch_workgroups(num_reduce_wgs, 1, 1);
    }
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }
    encoder.copy_buffer_to_buffer(&reduced_buf, 0, &output_staging_buf, 0, 1024 * 4);
    if let Some(query_set) = &query_set {
        encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
    }
    encoder.copy_buffer_to_buffer(&query_buf, 0, &query_staging_buf, 0, 16);
    queue.submit(Some(encoder.finish()));

    let buf_slice = output_staging_buf.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    let query_slice = query_staging_buf.slice(..);
    // Assume that both buffers become available at the same time. A more careful
    // approach would be to wait for both notifications to be sent.
    let _query_future = query_slice.map_async(wgpu::MapMode::Read, |_| ());
    println!("pre-poll {:?}", std::time::Instant::now());
    device.poll(wgpu::Maintain::Wait);
    println!("post-poll {:?}", std::time::Instant::now());
    if let Some(Ok(())) = receiver.receive().await {
        let data_raw = &*buf_slice.get_mapped_range();
        let data: &[u32] = bytemuck::cast_slice(data_raw);
        println!("data: {:?}", &data[..16]);
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

fn main() {
    pollster::block_on(run());
}
