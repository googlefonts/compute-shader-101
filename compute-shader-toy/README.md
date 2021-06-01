# Compute shader toy based on wgpu

This is a starting point for running compute shaders and drawing the image output into a window, based on [wgpu]. You should be able to change the shader (paint.wgsl) and run using simply `cargo run`.

If there are syntax errors in the shader, the error message can be pretty cryptic. It might be useful to run [naga] from the commandline to validate the shader code.

The shading language for this example is [WGSL], translated by naga, but it is possible to run wgpu in native mode with SPIR-V shaders as well. At the time of this writing, compute shaders are blocked on [naga#875], but when the fix for that lands, the experience should be better, and that will also open up features like atomics (and possibly subgroup operations) that are not presently supported by the wgpu stack.

[naga]: https://github.com/gfx-rs/naga
[wgpu]: https://wgpu.rs/
[WGSL]: https://gpuweb.github.io/gpuweb/wgsl/
[naga#975]: https://github.com/gfx-rs/naga/issues/875
