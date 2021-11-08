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

use bytemuck;

const N_DATA: usize = 1 << 25;
const WG_SIZE: usize = 1 << 12;

// Verify that the data is OEIS A000217
fn verify(data: &[u32]) -> Option<usize> {
    data.iter().enumerate().position(|(i, val)| {
        let wrong = ((i * (i + 1)) / 2) as u32 != *val;
        if wrong {
            println!("diff @ {}: {} != {}", i, ((i * (i + 1)) / 2) as u32, *val);
        }
        wrong
    })
}

async fn run() {
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let features = adapter.features();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: features & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::CLEAR_COMMANDS),
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

    let start_instant = Instant::now();
    let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        //source: wgpu::ShaderSource::SpirV(bytes_to_u32(include_bytes!("alu.spv")).into()),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    println!("shader compilation {:?}", start_instant.elapsed());
    let input_f: Vec<u32> = (0..N_DATA as u32).collect();
    let input: &[u8] = bytemuck::cast_slice(&input_f);
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: input,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC,
    });
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: input.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    const N_WG: usize = N_DATA / WG_SIZE;
    const STATE_SIZE: usize = N_WG * 3 + 1;
    // TODO: round this up
    let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 * STATE_SIZE as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // This works if the buffer is initialized, otherwise reads all 0, for some reason.
    let query_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &[0; 16],
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
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
                resource: state_buf.as_entire_binding(),
            },
        ],
    });

    for i in 0..100 {
        let mut encoder = device.create_command_encoder(&Default::default());
        if let Some(query_set) = &query_set {
            encoder.write_timestamp(query_set, 0);
        }
        encoder.clear_buffer(&state_buf, 0, None);
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(N_WG as u32, 1, 1);
        }
        if let Some(query_set) = &query_set {
            encoder.write_timestamp(query_set, 1);
        }
        if i == 0 {
            encoder.copy_buffer_to_buffer(&input_buf, 0, &output_buf, 0, input.len() as u64);
        }
        if let Some(query_set) = &query_set {
            encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
        }
        queue.submit(Some(encoder.finish()));

        let buf_slice = output_buf.slice(..);
        let buf_future = buf_slice.map_async(wgpu::MapMode::Read);
        let query_slice = query_buf.slice(..);
        let query_future = query_slice.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);
        if buf_future.await.is_ok() {
            if i == 0 {
                let data_raw = &*buf_slice.get_mapped_range();
                let data: &[u32] = bytemuck::cast_slice(data_raw);
                println!("results correct: {:?}", verify(data));
            }
            output_buf.unmap();
        }
        if query_future.await.is_ok() {
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
        query_buf.unmap();
    }
}

fn main() {
    pollster::block_on(run());
}
