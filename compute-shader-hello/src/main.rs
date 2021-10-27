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

// A strategy of 0 is just atomic loads.
// A strategy of 1 replaces the flag load with an atomicOr.
const STRATEGY: u32 = 0;

const USE_SPIRV: bool = false;

async fn run() {
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let features = adapter.features();
    let mut feature_mask = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::CLEAR_COMMANDS;
    if USE_SPIRV {
        feature_mask |= wgpu::Features::SPIRV_SHADER_PASSTHROUGH;
    }
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: features & feature_mask,
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
    let cs_module = if USE_SPIRV {
        let shader_src: &[u32] = bytemuck::cast_slice(include_bytes!("shader.spv"));
        unsafe {
            device.create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                label: None,
                source: std::borrow::Cow::Owned(shader_src.into()),
           })
        }
    } else {
        device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        })
    };
    

    println!("shader compilation {:?}", start_instant.elapsed());
    let data_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 0x80000,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let config_buf =  device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&[STRATEGY, 0]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
    });
    // This works if the buffer is initialized, otherwise reads all 0, for some reason.
    let query_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &[0; 16],
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
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
        }],
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
                resource: data_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: config_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 0);
    }
    encoder.clear_buffer(&data_buf, 0, None);
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch(256, 1, 1);
    }
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }
    //encoder.copy_buffer_to_buffer(&input_buf, 0, &output_buf, 0, input.len() as u64);
    if let Some(query_set) = &query_set {
        encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
    }
    queue.submit(Some(encoder.finish()));

    let buf_slice = config_buf.slice(..);
    let buf_future = buf_slice.map_async(wgpu::MapMode::Read);
    let query_slice = query_buf.slice(..);
    let _query_future = query_slice.map_async(wgpu::MapMode::Read);
    device.poll(wgpu::Maintain::Wait);
    if buf_future.await.is_ok() {
        let data_raw = &*buf_slice.get_mapped_range();
        let data: &[u32] = bytemuck::cast_slice(data_raw);
        println!("failures with strategy {}: {}", data[0], data[1]);
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
    env_logger::init();
    pollster::block_on(run());
}
