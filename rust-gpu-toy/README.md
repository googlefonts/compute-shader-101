# Compute shader toy based on wgpu

This is a starting point for running compute shaders and drawing the image output into a window, based on [wgpu] and [rust-gpu]. You should be able to change the [`shader`] and run using simply `cargo run --release`.

Using release mode is recommended because the compilation backend (`rustc_codegen_spirv`) runs very slowly when compiled without optimisations.

The shading language for this example is [rust-gpu], which produces SPIR-V shaders.

These shaders are reloaded automatically upon saving, without needing to restart the window.

[wgpu]: https://wgpu.rs/
[rust-gpu]: https://github.com/EmbarkStudios/rust-gpu
[naga]: https://github.com/gfx-rs/naga
[WGSL]: https://gpuweb.github.io/gpuweb/wgsl/
[naga#975]: https://github.com/gfx-rs/naga/issues/875
[`shader`]: ./shaders/src/lib.rs
