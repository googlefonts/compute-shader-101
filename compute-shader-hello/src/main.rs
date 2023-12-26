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

use bytemuck::{self, bytes_of, Pod, Zeroable, offset_of};

const WG: u32 = 256;
const ELEMENTS_PER_THREAD: u32 = 4;
const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;
const BIN_COUNT: u32 = 16;

#[derive(Clone, Copy, Default, Pod, Zeroable, Debug)]
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

    let n = 1 << 16;
    //let input = (0..n).map(|_| fastrand::u32(..)).collect::<Vec<_>>();
    let input = (0..n).collect::<Vec<_>>();

    let start_sort = std::time::Instant::now();
    let expected = sort_all(&input);
    println!("CPU sort elapsed: {:?}", start_sort.elapsed());

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
    let num_reduce_wg_per_bin = num_reduce_wgs / BIN_COUNT;

    let config = Config {
        num_keys: n,
        num_blocks_per_wg,
        num_wgs,
        num_wgs_with_additional_blocks,
        num_reduce_wg_per_bin,
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
    let config_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytes_of(&config),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&input),
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
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: input_buf.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_buf_2 = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: input_buf.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_buf.size(),
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
    let shifts = (0..8).map(|x| x * 4).collect::<Vec<u32>>();
    let shifts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&shifts),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
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
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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
        label: Some("count"),
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "count",
    });
    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("reduce"),
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "reduce",
    });
    let scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("scan"),
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "scan",
    });
    let scan_add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("scan_add"),
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "scan_add",
    });
    let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("scatter"),
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "scatter",
    });

    let bind_group_init = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });
    let bind_group_odd = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: config_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: count_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: reduced_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buf_2.as_entire_binding(),
            },
        ],
    });
    let bind_group_even = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: config_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf_2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: count_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: reduced_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    for iter in 0..10 {
        let mut encoder = device.create_command_encoder(&Default::default());
        if let Some(query_set) = &query_set {
            encoder.write_timestamp(query_set, 0);
        }
        for pass in 0..8 {
            // The most straightforward way to update the shift amount would be
            // queue.buffer_write, but that has performance problems, so we copy
            // from a pre-initialized buffer.
            let shift_offset = offset_of!(Config, shift);
            encoder.copy_buffer_to_buffer(&shifts_buf, 4 * pass, &config_buf, shift_offset as _, 4);
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            // TODO: set config based on pass
            cpass.set_pipeline(&count_pipeline);
            let bind_group = if pass == 0 {
                &bind_group_init
            } else if pass % 2 == 1 {
                &bind_group_odd
            } else {
                &bind_group_even
            };
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups(config.num_wgs, 1, 1);
            cpass.set_pipeline(&reduce_pipeline);
            cpass.dispatch_workgroups(num_reduce_wgs, 1, 1);
            cpass.set_pipeline(&scan_pipeline);
            cpass.dispatch_workgroups(1, 1, 1);
            cpass.set_pipeline(&scan_add_pipeline);
            cpass.dispatch_workgroups(num_reduce_wgs, 1, 1);
            cpass.set_pipeline(&scatter_pipeline);
            cpass.dispatch_workgroups(config.num_wgs, 1, 1);
        }
        if let Some(query_set) = &query_set {
            encoder.write_timestamp(query_set, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &output_staging_buf, 0, output_buf.size());
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
        let poll_start_time =  std::time::Instant::now();
        device.poll(wgpu::Maintain::Wait);
        println!("poll time {:?}", poll_start_time.elapsed());
        if let Some(Ok(())) = receiver.receive().await {
            let data_raw = &*buf_slice.get_mapped_range();
            let data: &[u32] = bytemuck::cast_slice(data_raw);
            if iter == 0 {
                println!("data size = {}", data.len());
                println!("data: {:x?}", &data[..32]);
                println!("expected: {:x?}", &expected[..32]);
                let first_diff = data.iter().zip(&expected).position(|(a, b)| a != b);
                if let Some(ix) = first_diff {
                    println!("discrepancy at {ix}, got {} expected {}", data[ix], expected[ix]);
                }
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
        output_staging_buf.unmap();
        query_staging_buf.unmap();
    }
}

// Useful for verifying the operation of a single digit
#[allow(unused)]
fn sort_digit(input: &[u32], digit: u32) -> Vec<u32> {
    let mut result = input.to_owned();
    let shift = digit * 4;
    result.sort_by(|a, b| ((a >> shift) & 0xf).cmp(&((b >> shift) & 0xf)));
    result
}

fn sort_all(input: &[u32]) -> Vec<u32> {
    let mut result = input.to_owned();
    result.sort();
    result
}

fn main() {
    pollster::block_on(run());
}
