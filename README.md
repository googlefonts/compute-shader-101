# Sample code for Compute Shader 101

This repo contains sample code to help you get started writing applications using compute shaders. It supplements a planned presentation including a video and slides (not ready yet).

Compute shaders are a promising approach to tap the power of GPU compute in a portable way, but it's not easy to get started. Most real-world usage of compute shaders today is in the context of games, where they do physics simulation, advanced image effects, and more. Games already have infrastructure to connect to the GPU, detect varying levels of capability at runtime, and abstract away details of graphics drivers and shader language translation.

To some extent compute shaders can be an alternative to CUDA, with a very different set of tradeoffs. The biggest advantage of compute shaders is portability, as they can run on most reasonably recent GPUs and are not tied to a specific hardware architecture. Proabbly the biggest shortcoming is lack of tool support.

## compute-shader-hello

The first subdirectory is a simple command-line application that runs a simple compute shader over a buffer of data. It's very much intended as a starter, as there are many directions it can be made more sophisticated.

Recommended activity: find some algorithm in the literature for which there are efficient GPU implementations (prefix sum, convolution, physics simulation, etc) and adapt it to run as a compute shader. Experiment with different implementation choices and observe the effect on performance.

## compute-shader-toy

The second subdirectory is a simple GUI application that runs a compute shader and draws its output in a window. It is intended to fulfill a similar function as [Shadertoy] by Inigo Quilez, but  potentially opens up the full power of compute shaders instead of being limited to WebGL.

Recommended activity: find an existing shadertoy that implements some interesting algorithm or visual effect, and port it to run in a compute shader. Is there some limitation of the original that could be improved by compute capabilities?

A great place to find shaders to adapt is [The Book of Shaders].

## A note on the choice of runtime

Your compute shader code cannot run on its own, but rather needs a *runtime* to connect to the GPU, set up resources such as buffers and compiled shader code, and manage the submission of that work to the GPU. There is, as of this writing, no standard runtime for such things, but I hope that will change in time.

The most promising candidate for standard runtime is an implementation of [WebGPU]. While WebGPU is being developed as a web standard and will allow deployment via web, it is also useful for native applications. This repo uses [wgpu-rs], which has a particularly nice Rust API and is also used by several other projects in the Rust GPU ecosystem. Another implementation of WebGPU is [Dawn], which is actively developed but primarily being used as the basis for WebGPU in Chromium.

WebGPU has many advantages. It is relatively easy to learn and use, implementations are portable against a wide range of hardware, and there is momentum behind it. The main disadvantage is that it is still work in progress and many parts are not done. There is currently a focus on implementing core functionality, leaving somewhat more advanced compute features to later.

Another alternative is to build your own runtime. This unlocks access to advanced features provided by Vulkan, D3D12, and Metal, but also requires more work. Because I'm actively exploring these more advanced compute features in [piet-gpu], I am also building my own runtime to match. It is possible that this repo will add samples using that runtime as well, depending on feedback from the experience using wgpu.

## A note on the choice of shader language

There is not yet a single obvious best choice for shader language. The most mature is GLSL, especially as it provides access to all functionality exposed to shader code by Vulkan, and Vulkan has many advanced features missing from other APIs.

The official shader language for WebGPU is [WGSL]. It is newer but does not yet have features such as subgroup operations or support for scalars of size other than 32 bits. It will be the required language for web deployments, but for native deployments, wgpu-rs also supports the SPIR-V intermediate representation, which can readily be derived from GLSL (or HLSL) source by compilers such as [glslangValidator].

Both of these choices are viable for getting started. More advanced work will require the use of GLSL. There are many tools for converting between different shader languages. In the wgpu-rs ecosystem, the new [naga] tool shows great promise, but it is still (as with many such things) unfinished.

Both GLSL and WGSL are fairly primitive and low-level, not far removed from their roots in [Cg]. There are a number of experimental languages intended to deliver a much higher level programming experience, of which the most interesting to me right now is [rust-gpu].

There are potentially many other paths to compute shader IR, including [IREE] which is mostly targeting machine learning applications.

## Additional resources

There are some more resources in the [docs/](./docs/) subdirectory, including a [glossary](./docs/glossary.md).

## Contributing

This resource is open source and we welcome contributions to improve it. See [CONTRIBUTING.md] for more details.

## License

Licensed under either of
  * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
    http://www.apache.org/licenses/LICENSE-2.0)
  * MIT license ([LICENSE-MIT](LICENSE-MIT) or
    http://opensource.org/licenses/MIT) at your option.

[Shadertoy]: https://www.shadertoy.com/
[WebGPU]: https://github.com/gpuweb/gpuweb
[WGSL]: https://gpuweb.github.io/gpuweb/wgsl/
[wgpu-rs]: https://github.com/gfx-rs/wgpu-rs
[Dawn]: https://dawn.googlesource.com/dawn
[glslangValidator]: https://github.com/KhronosGroup/glslang
[naga]: https://github.com/gfx-rs/naga
[Cg]: https://www.khronos.org/opengl/wiki/Cg
[rust-gpu]: https://github.com/EmbarkStudios/rust-gpu
[IREE]: https://google.github.io/iree/
[The Book of Shaders]: https://thebookofshaders.com/
