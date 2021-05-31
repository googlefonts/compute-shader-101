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

//! A simple compute shader example that draws in a window, based on
//! piet-gpu-hal.
//!
//! When changing the shader, use this command to run:
//!
//! ```shell
//! (cd shader && ninja) && cargo run
//! ```

use piet_gpu_hal::{BufferUsage, Error, Instance, ImageLayout, Session};
use piet_gpu_hal::include_shader;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const NUM_FRAMES: usize = 2;

unsafe fn toy() -> Result<(), Error> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize {
            width: 1024.0,
            height: 768.0,
        })
        .with_resizable(false) // currently not supported
        .build(&event_loop)?;
    let size = window.inner_size();
    println!("window size = {} x {}", size.width, size.height);
    let (instance, surface) = Instance::new(Some(&window))?;
    let surface = surface.ok_or("no surface")?;
    let device = instance.device(Some(&surface))?;
    let mut swapchain = instance.swapchain(size.width as _, size.height as _, &device, &surface)?;
    let session = Session::new(device);
    let config_size = 12;
    let mut config_host = session.create_buffer(config_size, BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE)?;
    let config_dev = session.create_buffer(config_size, BufferUsage::COPY_DST | BufferUsage::STORAGE)?;
    let staging_img = session.create_image2d(size.width, size.height)?;
    let start_time = std::time::Instant::now();

    let shader_code = include_shader!(&session, "../shader/gen/shader");
    let pipeline = session.pipeline_builder()
        .add_buffers(1)
        .add_images(1)
        .create_compute_pipeline(&session, shader_code)?;

    let descriptor_set = session.descriptor_set_builder()
        .add_buffers(&[&config_dev])
        .add_images(&[&staging_img])
        .build(&session, &pipeline)?;

    let mut current_frame = 0;
    let present_semaphores = (0..NUM_FRAMES)
        .map(|_| session.create_semaphore())
        .collect::<Result<Vec<_>, Error>>()?;

    event_loop.run(move |event, _, control_flow| {
        //println!("event {:?}", event);
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let frame_idx = current_frame % NUM_FRAMES;
                let (image_idx, acquisition_semaphore) = swapchain.next().unwrap();
                let swap_image = swapchain.image(image_idx);

                let i_time: f32 = 0.5 + start_time.elapsed().as_micros() as f32 * 1e-6;
                let config_data = [
                    size.width,
                    size.height,
                    i_time.to_bits(),
                ];
                config_host.write(&config_data).unwrap();

                let mut cmd_buf = session.cmd_buf().unwrap();
                cmd_buf.begin();
                cmd_buf.image_barrier(&swap_image, ImageLayout::Undefined, ImageLayout::BlitDst);
                cmd_buf.copy_buffer(&config_host, &config_dev);
                cmd_buf.memory_barrier();

                cmd_buf.image_barrier(&staging_img, ImageLayout::Undefined, ImageLayout::General);
                let wg_x = size.width / 16;
                let wg_y = size.height / 16;
                cmd_buf.dispatch(&pipeline, &descriptor_set, (wg_x, wg_y, 1), (16, 16, 1));
                cmd_buf.image_barrier(&staging_img, ImageLayout::General, ImageLayout::BlitSrc);
                cmd_buf.blit_image(&staging_img, &swap_image);
                cmd_buf.image_barrier(&swap_image, ImageLayout::BlitDst, ImageLayout::Present);
                cmd_buf.finish();
                let submitted = session
                    .run_cmd_buf(
                        cmd_buf,
                        &[&acquisition_semaphore],
                        &[&present_semaphores[frame_idx]],
                    )
                    .unwrap();
                swapchain
                    .present(image_idx, &[&present_semaphores[frame_idx]])
                    .unwrap();
                submitted.wait().unwrap();
                current_frame += 1;
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => (),
        }
    });
}

fn main() {
    unsafe {
        toy().unwrap();
    }
}
