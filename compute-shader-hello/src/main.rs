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

use bytemuck;

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

    let source = if USE_SPIRV {
        wgpu::util::make_spirv(include_bytes!("shader.spv"))
    } else {
        wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into())
    };
    let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source,
    });

    let data_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 0x80000,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 8,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::MAP_READ
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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
                resource: data_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: config_buf.as_entire_binding(),
            },
        ],
    });

    let mut failures = 0;
    for i in 0..1000 {
        let mut encoder = device.create_command_encoder(&Default::default());
        if let Some(query_set) = &query_set {
            encoder.write_timestamp(query_set, 0);
        }
        encoder.clear_buffer(&config_buf, 0, None);
        encoder.clear_buffer(&data_buf, 0, None);
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(256, 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        let buf_slice = config_buf.slice(..);
        let buf_future = buf_slice.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);
        if buf_future.await.is_ok() {
            let data_raw = buf_slice.get_mapped_range();
            let data: &[u32] = bytemuck::cast_slice(&*data_raw);
            if data[1] != 0 {
                if failures == 0 {
                    println!("first failing iteration {}, failures: {}", i, data[1]);
                }
                failures += data[1];
            }
            std::mem::drop(data_raw);
            config_buf.unmap();
        }
    }
    if failures != 0 {
        println!("{} total failures", failures);
    }
}

fn main() {
    env_logger::init();
    pollster::block_on(run());
}
