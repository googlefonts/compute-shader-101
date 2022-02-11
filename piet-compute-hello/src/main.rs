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

//! A simple application to run a compute shader, based on
//! piet-gpu-hal.
//!
//! When changing the shader, use this command to run:
//!
//! ```shell
//! (cd shader && ninja) && cargo run
//! ```

use std::mem::size_of;
use piet_gpu_hal::{BufferUsage, Error, Instance, Session};
use piet_gpu_hal::include_shader;

unsafe fn compute() -> Result<(), Error> {
    let (instance, _) = Instance::new(None)?;
    let device = instance.device(None)?;
    let session = Session::new(device);

    let mut data: Vec<u32> = vec![0; 256];
    let data_size = (data.len() * size_of::<u32>()) as u64;
    println!("submitted: {:?}", data);

    let mut data_host = session.create_buffer(data_size, BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE)?;
    let data_device = session.create_buffer(data_size, BufferUsage::COPY_DST | BufferUsage::STORAGE)?;

    let shader_code = include_shader!(&session, "../shader/gen/shader");
    let pipeline = session.pipeline_builder()
        .add_buffers(1)
        .create_compute_pipeline(&session, shader_code)?;

    let descriptor_set = session.descriptor_set_builder()
        .add_buffers(&[&data_device])
        .build(&session, &pipeline)?;

    let query_pool = session.create_query_pool(4).unwrap();

    data_host.write(&data).unwrap();

    let mut cmd_buf = session.cmd_buf().unwrap();
    cmd_buf.begin();
    cmd_buf.reset_query_pool(&query_pool);
    cmd_buf.write_timestamp(&query_pool, 0);

    cmd_buf.copy_buffer(&data_host, &data_device);
    cmd_buf.memory_barrier();
    cmd_buf.write_timestamp(&query_pool, 1);

    cmd_buf.dispatch(&pipeline, &descriptor_set, (1, 1, 1), (256, 1, 1));
    cmd_buf.write_timestamp(&query_pool, 2);
    cmd_buf.memory_barrier();

    cmd_buf.copy_buffer(&data_device, &data_host);
    cmd_buf.host_barrier();
    cmd_buf.write_timestamp(&query_pool, 3);

    cmd_buf.finish_timestamps(&query_pool);
    cmd_buf.finish();
    let submitted = session
        .run_cmd_buf(
            cmd_buf,
            &[],
            &[],
        )
        .unwrap();
    submitted.wait()?;
    let ts = session.fetch_query_pool(&query_pool).unwrap();
    data_host.read(&mut data)?;
    println!("received: {:?}", data);
    // fetch_query_pool subtracts the first timestamp from all other timestamps
    println!("time to copy from host to device: {:.3} ms", ts[0] * 1e3);
    println!("dispatch time: {:.3} ms", (ts[1] - ts[0]) * 1e3);
    println!("time to copy from device to host: {:.3} ms", (ts[2] - ts[1]) * 1e3);
    Ok(())
}

fn main() {
    unsafe {
        compute().unwrap();
    }
}
