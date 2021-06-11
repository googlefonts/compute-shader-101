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

//! A simple compute shader example that draws into a window, based on wgpu.

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};

use rust_gpu_toy_shared::Config;
use wgpu::util::{make_spirv, DeviceExt};
use wgpu::{BufferUsage, Extent3d, ShaderModule};

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use spirv_builder::{CompileResult, SpirvBuilder};

const CONFIG_SIZE: u64 = std::mem::size_of::<Config>() as u64;

async fn run(
    event_loop: EventLoop<CompileResult>,
    window: Window,
    initial_compilation: CompileResult,
) {
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: Default::default(),
            compatible_surface: Some(&surface),
        })
        .await
        .expect("error finding adapter");

    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .expect("error creating device");
    let size = window.inner_size();
    let swapchain_format = adapter.get_swap_chain_preferred_format(&surface).unwrap();
    let sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };
    let swap_chain = device.create_swap_chain(&surface, &sc_desc);

    // We use a render pipeline just to copy the output buffer of the compute shader to the
    // swapchain. It would be nice if we could skip this, but swapchains with storage usage
    // are not fully portable.
    let copy_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        // Should filterable be false if we want nearest-neighbor?
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        filtering: false,
                        comparison: false,
                    },
                    count: None,
                },
            ],
        });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&copy_bind_group_layout],
        push_constant_ranges: &[],
    });

    let module = create_shader_module(&initial_compilation, &device);

    let mut render_pipeline =
        create_render_pipeline(&device, &pipeline_layout, swapchain_format, &module);
    let img = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::SAMPLED,
    });
    let img_view = img.create_view(&Default::default());

    let config_dev = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: CONFIG_SIZE,
        usage: BufferUsage::COPY_DST | BufferUsage::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(CONFIG_SIZE),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let config_resource = config_dev.as_entire_binding();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: config_resource,
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&img_view),
            },
        ],
    });
    let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let mut pipeline = create_compute_pipeline(&device, &module, &compute_layout);
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    let copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &copy_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&img_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    let start_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        // TODO: this may be excessive polling. It really should be synchronized with
        // swapchain presentation, but that's currently underbaked in wgpu.
        *control_flow = ControlFlow::Poll;
        match event {
            Event::RedrawRequested(_) => {
                let frame = swap_chain
                    .get_current_frame()
                    .expect("error getting frame from swap chain")
                    .output;

                let i_time: f32 = 0.5 + start_time.elapsed().as_micros() as f32 * 1e-6;
                let config = Config {
                    width: size.width,
                    height: size.height,
                    time: i_time,
                };
                // In theory, this should use push constants (maybe?)
                let config_host = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&config),
                    usage: BufferUsage::COPY_SRC,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(&config_host, 0, &config_dev, 0, CONFIG_SIZE);
                {
                    let mut cpass = encoder.begin_compute_pass(&Default::default());
                    cpass.set_pipeline(&pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch(size.width / 16, size.height / 16, 1);
                }
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[wgpu::RenderPassColorAttachment {
                            view: &frame.view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                store: true,
                            },
                        }],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &copy_bind_group, &[]);
                    rpass.draw(0..3, 0..2);
                }
                queue.submit(Some(encoder.finish()));
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::UserEvent(result) => {
                let module = create_shader_module(&result, &device);
                render_pipeline =
                    create_render_pipeline(&device, &pipeline_layout, swapchain_format, &module);
                pipeline = create_compute_pipeline(&device, &module, &compute_layout);
            }
            _ => (),
        }
    });
}

fn create_shader_module(compilation: &CompileResult, device: &wgpu::Device) -> ShaderModule {
    let shader_flags = wgpu::ShaderFlags::empty();
    let spirv_path = compilation.module.unwrap_single();
    let f = std::fs::read(spirv_path).expect("spirv should exist");
    let shader = make_spirv(&f);
    device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: shader,
        flags: shader_flags,
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &shader,
        entry_point: "main",
    })
}

fn create_render_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    swapchain_format: wgpu::TextureFormat,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[swapchain_format.into()],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
    });
    render_pipeline
}

fn main() {
    let (initial_tx, initial_rx) = std::sync::mpsc::sync_channel(0);
    let event_loop = EventLoop::with_user_event();
    let proxy = event_loop.create_proxy();
    let has_sent_first = AtomicBool::new(false);
    // Watch for changes on a background thread
    let thread = std::thread::spawn(|| {
        SpirvBuilder::new("./shaders", "spirv-unknown-vulkan1.2")
            .print_metadata(spirv_builder::MetadataPrintout::None)
            .capability(spirv_builder::Capability::StorageImageWriteWithoutFormat)
            .watch(move |result| {
                if let Ok(_) =
                    // TODO: This use of atomics is signic
                    has_sent_first.compare_exchange(
                        false,
                        true,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    )
                {
                    initial_tx.send(result).unwrap();
                } else {
                    proxy
                        .send_event(result)
                        .expect("Event loop should still be running");
                }
            })
            .expect("Correctly setup")
    });
    std::mem::forget(thread);
    let initial = initial_rx
        .recv()
        .expect("Watching should get an initial shader");
    let window = Window::new(&event_loop).unwrap();
    pollster::block_on(run(event_loop, window, initial));
}
