# Compute shader glossary

One barrier to learning and talking about GPU compute is the bewildering terminology. There are a large number of concepts that probably seem exotic and alien to people used to programming CPU's. To make things worse, there are approximately 5 API's that are relevant today, and in many cases terminology is wildly inconsistent between them. This post is an annotated glossary of some of these terms. It will be especially useful to people who know one of the API's and want to learn about others.

* [Vulkan](https://www.khronos.org/vulkan/) - [GLSL](https://www.khronos.org/opengl/wiki/Core_Language_(GLSL))
* [DX12](https://docs.microsoft.com/en-us/windows/win32/direct3d12/directx-12-programming-guide) - [HLSL](https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl)
* [Metal](https://developer.apple.com/metal/) - [Metal shading language](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (PDF)
* [WebGPU](https://github.com/gpuweb/gpuweb) - [WebGPU shading language](https://gpuweb.github.io/gpuweb/wgsl/)
* [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

My choice of 5 API's is somewhat opinionated, and future-leaning. I've left off OpenGL and OpenCL because they appear to be stagnating. Even though my main focus is writing compute shaders, because I'm interested in portability, I'm also including CUDA because it's by far the best developed API for writing GPU compute, and one that will no doubt be familiar to many readers.

I'm including WebGPU because I feel it has potential to become a common, widely implemented standard, on which it's possible to build a rich infrastructure of tools, libraries, and applications. The people working on WebGPU are doing a lot of deep thinking about achieving portability, performance, and security on GPU, and these discussions are openly available.

Another resource I've found invaluable for writing (somewhat) portable compute shader code is [gfx-hal]. This project exposes an API much like Vulkan (though not identical), with [backends](https://github.com/gfx-rs/gfx/tree/master/src/backend) for all the major API's. In many cases, the best way to answer the question "how to I do X in API Y" is to find the gfx-hal method for it and look up the code in the respective backend.

In general I favor Vulkan terminology, as it's an open standard, well documented, and in some areas (particularly memory model) is significantly ahead of its competition.

This post also especially welcomes open source contributions, as no doubt some of the entries are incomplete or confusing. File an issue or a PR at https://github.com/raphlinus/raphlinus.github.io.

## The thread grouping hierarchy

A defining characteristic of GPU compute is a fairly deep hierarchy of groups of threads. At each level, there is interaction with threads in the same group, very high bandwidth and low latency at the bottom of the hierarchy, access to more of the workload as you go further up. But the programming interfaces (and terminology) varies, reflecting the realities of GPU hardware.

We'll take this hierarchy from the bottom up.

### Thread; Invocation (Vulkan, formal); Lane (DX)

A thread in GPU compute is *like* a traditional CPU thread, but with some differences. While on CPU, the cost of spawning a thread is significant, on a GPU it is quite cheap, and often the "many small threads" pattern is the best way to achieve performance.

GPU threads are scheduled quite differently than on CPU. For one, forward progress guarantees are much weaker (though this depends on the GPU). The hardware model of GPU is often described as [Single Instruction Multiple Threads] (SIMT), which is similar to more traditional Single Instruction Multiple Data (SIMD), but with more support for creating the illusion that the threads are running independently, in particular execution masks for conditional control flow.

### Subgroup (Vulkan); Wave (DX); Warp (CUDA); SIMD-group (Metal)

A subgroup is a group of threads generally executed in parallel by an execution unit in the hardware.

Communication between threads using subgroup operations is often much faster than shared memory, often similar latency and bandwidth as pure ALU.

The size of a subgroup is a major concern for performance tuning. On Nvidia hardware, it can be assumed to be 32, but for portability, usually ranges from 8 to 64, with 128 as a possibility on some mobile hardware (Adreno, optionally, plus Imagination). On some GPUs, the subgroup size is fixed, but on many it is dynamic; on these GPUs it is difficult to reliably know or control the subgroup size unless the subgroup size extension is available.

A warning: subgroup operations can be a source of portability concerns. Not all GPUs support all subgroup operations, and dealing with the diversity of subgroup sizes is also a challenge. The discussion in [gpuweb#954] is illuminating, for more detail.

Even aside from using explicit subgroup operations, awareness of subgroup structure is relevant for performance, for a variety of reasons. For one, the performance cost of branch divergence generally respects subgroup granularities; if all threads in a subgroup have the same branch, the cost is much lower. In addition, uniform memory reads are generally amortized across the threads in a subgroup, though multiple reads by different subgroups of a workgroup-uniform location will generally hit in L1 cache.

The terminology space can get even more confusing: [ISPC] uses the word "gang" for this concept. But here we'll mostly stick to things that run on GPU.

Resources:

* [Vulkan Subgroup Tutorial] (Khronos)

### Workgroup (Vulkan, WebGPU); Threadgroup (Metal); Thread Block (CUDA)

One of the main purposes of organizing threads into workgroups is access to a shared memory buffer dedicated to the workgroup. Workgroups can also synchronize using barriers. This the highest level of hierarchy for which such synchronization is possible; there is *no* similar synchronization between workgroups in a dispatch.

While the driver and shader compiler (usually) choose the subgroup size, the workgroup size is entirely up to the application author, up to the supported limits of the device. A limit of 1024 threads total is typical for desktop GPUs, but on mobile smaller limits are common; Vulkan merely requires it be at least 128. It should generally be larger than the subgroup size to avoid performance problems due to unused threads.

### Dispatch

A dispatch is a unit of computation all sharing the same input and output buffers and code.

The size of a dispatch is specified as a 3d grid of workgroups.

Conceptually, all of the threads in a dispatch are executed in parallel, though in practice the hardware and driver will apply scheduling strategies, and some algorithms (such as decoupled look-back for [prefix sum]) take advantage of properties of real-world scheduling.

A dispatch gets access to its input and output buffers through a descriptor set.

### Compute shader vs kernel

The distinction between compute shader and kernel is blurry, and Metal especially splits the difference by using the term "kernel" for its compute shader capability. Generally a compute shader integrates seamlessly with other graphics rendering primitives, while OpenCL and CUDA kernels require their own separate runtime (possibly with interop features). Tools like [clvk] also blur the distinction, by providing a layer to run OpenCL-compatible kernels on compute shader infrastructure.

Further confusing the issue is the fact that SPIR-V exists in two flavors, one for OpenCL kernels and one for (compute) shaders; these are distinguished by the "Execution Model" field in the SPIR-V header. In the kernel execution model, pointers are enabled by default, while in compute shaders they are extensions.

### Command buffer (Vulkan, Metal, WebGPU); Command list (DX12)

A command buffer is a sequence of commands, of which the main substance is compute dispatches, but there are a bunch of other commands required to glue these together: pipeline barriers (see below), buffer copies, etc.

One performance benefit of explicit API's including Vulkan is that command buffers can be created in multiple threads.

In WebGPU, the aspect of actually adding commands to a command buffer is further split out into a "command encoder." It's reasonable to consider a WebGPU command encoder as a builder for the otherwise mostly opaque command buffer object. By contrast, in Vulkan, you call methods on the command buffer to add commands, then pass that same command buffer to a [queue submit](vkQueueSubmit) call.

Metal also has a concept of "command encoder", but it's best thought of as a builder for a single command (dispatch, render pass, etc) to be added to a command buffer.

### Queue (Vulkan, WebGPU); Command Queue (DX12, Metal); Stream (CUDA)

Command buffers are submitted to a queue. Conceptually, the commands in a queue are run in sequence, but overlap is possible (and in many cases desirable for performance), so great care is needed to add barriers and other synchronization primitives to avoid conflicts.

A single queue can be fairly effective in exploiting GPU throughput, but in some cases multiple queues can improve throughput further; work from one queue can be scheduled "in the gaps" left by another, such as waiting for CPU (host) synchronization.

In some cases, hardware may support specialized queues that only perform a subset of operations; it's not uncommon to have a transfer queue that can do copies but not compute dispatches.

The use of multiple queues is often called "async compute," especially in the context of AMD, which was an early adopter.

### Shared memory

Shared memory is a small chunk of high-performance memory (often limited to 32kB) available for communication between threads in a workgroup. Except in the case of embarassingly parallel workloads (where each thread is independent), shared memory is often the key to good performance of compute shaders.

In WebGPU, the term "shared memory" is used informally, but the formal term is a variable with workgroup storage class. In HLSL, the keyword is `groupshared`. In Metal, the storage qualifier is `threadgroup`. On AMD hardware, the term "Local Data Share" is used.

## Resource binding

Even though access to memory in GPUs is based on pointers under the hood, you generally don't call compute shaders with pointers and let them read or write this memory at their discretion. Rather, access to memory is mediated through *descriptors,* and a dispatch of a compute shader consists of binding a set of descriptors to a pipeline. Descriptors are created and managed CPU-side (and in some cases there is direct CPU access to buffers and, less likely, textures), but they exist primarily to give the GPU access to the memory.

In some cases, "bindless" approaches are becoming popular, as they can involve less overhead. This approach can "fake" pointers to some extent using descriptor arrays, which still involve indirection through the descriptors but are more flexible than traditional descriptor sets, which only address a small, fixed number of descriptor arrays. Two Vulkan extensions also reintroduce something resembling raw pointers: [VK_KHR_variable_pointers](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_variable_pointers.html) and [Buffer Device Address](https://community.arm.com/developer/tools-software/graphics/b/blog/posts/vulkan-buffer-device-address). Also note that OpenCL, Metal, and CUDA also allow pointers natively.

### Descriptor set (Vulkan); Bind group (WebGPU); Root signature (DX12); Pipeline descriptor (Metal)

Resources:

* [Resource Binding] (DX12)
* [Differences in memory management between Direct3D 12 and Vulkan](https://asawicki.info/articles/memory_management_vulkan_direct3d_12.php5) (blog)

## The synchronization zoo

GPUs are all about an extremely high degree of parallelism. Generally you want to be able to schedule as many things at the same time as possible. However, there are also dependencies in the workload, where, for example, one stage in a pipeline depends on previous results. With modern APIs, the programmer needs to add explicit synchronization between these stages.

It should be noted that older APIs (OpenGL) often go to great lengths to produce the illusion of sequential processing. That can generally be emulated by analyzing the workload; if a dispatch writes to a buffer, then another dispatch reads from the same buffer, it can infer that a pipeline barrier is needed. Or if the sets of buffers are completely disjoint, then the dispatches can be scheduled on the hardware with no such barrier. But this analysis increases the CPU burden, and in particular complicates efforts to build command queues on different threads.

Modern APIs provide a bewildering array of synchronization primitives. Some control execution within a compute shader. Others (pipeline barriers) prevent successive dispatches within a command buffer from interfering with each other. Still more 

Resources:

* [Understanding Vulkan Synchronization] (Khronos tutorial)
* [Yet another blog explaining Vulkan synchronization] (themaister)

### Barrier (shader internal); Sync + fence (CUDA)

A barrier (in a shader) is a synchronization point. In the vanilla [GLSL barrier], all threads synchronize at a barrier, and writes to shared memory before the barrier are available to threads after the barrier.

With the advent of the [Vulkan memory model], a richer set of options are available. The functionality of control barrier (when instructions are actually executed) is separated from memory barrier (when writes actually become properly visible to future reads). Both take a [scope](https://www.khronos.org/blog/comparing-the-vulkan-spir-v-memory-model-to-cs#_memory_and_execution_scopes), which is a unit of granularity of a group of threads; it can range from subgroup all the way to device. A scope is also given for atomic operations.

The use of "barrier" terminology is fairly consistent across api's, but a control barrier is called "sync" in DX and CUDA; the [barrier()](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/barrier.xhtml) GLSL function is roughly equivalent to [__syncthreads()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions) in CUDA. A memory-only barrier, without corresponding control barrier, is called a "memory fence"; a typical function is [__threadfence()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions).

In general, barriers require uniform control flow within the workgroup. In some cases, there is support for execution masks, but it's pretty specialized.

### Pipeline barrier (Vulkan); Fence (Metal); Resource Barrier (DX12)

In Metal, pipeline barriers are generally only needed if buffers and other resources are created in [untracked mode](https://developer.apple.com/documentation/metal/mtlhazardtrackingmode/mtlhazardtrackingmodeuntracked), in which case they are implemented using [MTLFence](https://developer.apple.com/documentation/metal/mtlfence). When hazard tracking is enabled, the driver does its own analysis and inserts fences as needed.

In Vulkan, a pipeline barrier can be split in two, using an [event](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkEvent.html). In theory, expressing dependencies in this finer-grained way gives the driver more flexibility to overlap scheduling of dispatches, but in practice it's not clear the extra complexity is worth it.

When dealing with image objects, Vulkan pipeline barriers also do [layout transitions](https://vulkan-tutorial.com/Texture_mapping/Images#page_Layout-transitions), which, among other things, can change the memory layout of image pixels within an image buffer to optimize for different usage cases.

WebGPU tracks resources and automatically inserts pipeline barriers, much like Metal in the default case.

Resources:

* [Breaking Down Barriers] (blog series)

### Fence (Vulkan)

Signaling the CPU when the GPU has completed its work is one of the most basic requirements of programming using compute shaders, yet is one of the most variable between the different APIs.

A Vulkan [fence](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkFence.html) is primarily a synchronization primitive to notify the host when a queue submission completes. The host can [wait](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkWaitForFences.html) on the fence, and then safely do any action that depended on the command buffers within that queue submission completing - read from buffers that were written, deallocate or reuse any resources, etc.

Metal is a bit different; you can [wait on the command buffer object](https://developer.apple.com/documentation/metal/mtlcommandbuffer/1443039-waituntilcompleted), and you can add a [completed handler](https://developer.apple.com/documentation/metal/mtlcommandbuffer/1442997-addcompletedhandler) to a command buffer before submitting ([commit](https://developer.apple.com/documentation/metal/mtlcommandbuffer/1443003-commit) in Metal-speak), and this handler will get called when the command buffer finishes.

In DX12, this is a bit of a multi-step process involving both a fence and an [event](https://docs.microsoft.com/en-us/windows/win32/sync/event-objects). First, add a command to the command queue to [signal](https://docs.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12commandqueue-signal) the fence. Then, [wire it up](https://docs.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12fence-seteventoncompletion) so that the fence triggers the event. It's then possible to wait on the event using any of the standard Windows [wait](https://docs.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject) methods.

WebGPU has a different approach; a queue gives you a promise for `onSubmittedWorkDone`, and that promise is resolved (by the underlying async runtime) when the queue completes. In many cases, you don't really care when the commands are done running, only when you can read the buffers that they've written into, and for that, the buffers provide a `mapAsync` method, also returning a promise.

### Semaphore (Vulkan); Fence (DX12)

A Vulkan [semaphore](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkSemaphore.html) is primarily a synchronization primitive to notify another queue when commands within a queue have completed. A very common special case is synchronizing swapchain presentation with rendering tasks; both signalling the swapchain that rendered content is available in an image, and also signalling the application when an image in the swapchain is no longer "front" and thus can be reused.

In DX12, synchronization across is generally done with a [Fence](https://docs.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization). Explicit semaphores are not needed for swapchain presentation, as that's generally expected to run in the same queue timeline as the rendering work.

## Memory model

The memory model is perhaps the sharpest contrast between the past and the future. A lot of work has gone into properly specifying the Vulkan memory model, but at the time of this writing, I'm unaware of anyone actually using it. Part of the issue is that it's not (yet) supported in DX12 or Vulkan.

Without a memory model, drivers and shader authors share a loose, intuitive, and ad hoc understanding of what is allowed. A driver can be either too conservative, leaving performance on the table, or too aggressive, causing incorrect results. With a memory model, there is a precise contract of exactly what is allowed, and the driver is free to optimize agressively as long as it meets these semantics.

Resources:

* [Vulkan memory model] (Khronos blog post)
* [Vulkan 1.2 Memory Model] (Vulkan spec)

### Scope (Vulkan only)

A scope is a unit of granularity of a group of threads. It can range anywhere from subgroup to device. All memory barriers, control barriers, and atomic operations specify a scope.

Note that pre-memory model GLSL often assumes that control barriers at a subgroup level are not necessary, because threads in a subgroup will be executed in lockstep. As of the [Nvidia Volta] (GTX 10xx) generation, that is no longer necessarily the case, as threads can be scheduled independently, so it is no longer a good idea to assume that.

The scope should be chosen as the smallest granularity that encompasses all the threads that will be reading and writing the memory. For example, to publish aggregates from one workgroup to another in an implementation of the decoupled look-back algorithm for [prefix sum], the most appropriate scope is "queue family," as that's the smallest scope that includes multiple workgroups from a dispatch.

Resources:

* [GL_KHR_memory_scope_semantics.txt] (GLSL spec)

## Unified memory / staging buffers

There are three basic models for managing buffer sharing between CPU and GPU: traditional discrete, integrated, and unified memory.

In the traditional discrete model, the GPU has "device memory" onboard which is separate from the CPU address space. In most cases, getting data uploaded to the GPU and read back involves "staging buffers", so there is an explict copy from a CPU buffer to a GPU buffer, and back. It's also possible for the GPU to access buffers in CPU memory (host visible), but this is generally done with PCI bus snooping and is quite slow.

In the integrated model, common on lower end graphics cards and also mobile, there is one memory space shared by both CPU and GPU. Staging buffers are not necessary, and the extra copy only wastes memory and slows things down. It's best to query at runtime whether staging buffers are necessary, and only create them when needed.

In a unified memory model, which is becoming more popular, buffers appear to be created in a single unified address space, but the physical location of any given page of memory may bounce between CPU memory, GPU memory, or copies in both places, depending on usage patterns. This feature has existed in CUDA for a while, but is only recently coming to compute shaders, where it goes by the marketing name "smart access memory" or the more technical name "Resizable BAR."

Resources:

* [Vulkan Memory Types on PC and How to Use Them] (blog post) 
* [System and driver support for Resizable BAR] (Microsoft docs, highly technical)

[WebGPU subgroup discussion]: https://github.com/gpuweb/gpuweb/pull/1459
[Vulkan Subgroup Tutorial]: https://www.khronos.org/blog/vulkan-subgroup-tutorial
[Resource Binding]: https://docs.microsoft.com/en-us/windows/win32/direct3d12/resource-binding
[GLSL barrier]: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/barrier.xhtml
[Vulkan memory model]: https://www.khronos.org/blog/comparing-the-vulkan-spir-v-memory-model-to-cs
[Understanding Vulkan Synchronization]: https://www.khronos.org/blog/understanding-vulkan-synchronization
[Yet another blog explaining Vulkan synchronization]: https://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/
[Nvidia Volta]: https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
[gpuweb#954]: https://github.com/gpuweb/gpuweb/pull/954
[prefix sum]: https://raphlinus.github.io/gpu/2020/04/30/prefix-sum.html
[GL_KHR_memory_scope_semantics.txt]: https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_memory_scope_semantics.txt
[Vulkan 1.2 memory model]: https://www.khronos.org/registry/vulkan/specs/1.2-khr-extensions/html/chap43.html#memory-model
[Breaking Down Barriers]: https://therealmjp.github.io/posts/breaking-down-barriers-part-1-whats-a-barrier/
[gfx-hal]: https://github.com/gfx-rs/gfx
[Single Instruction Multiple Threads]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads
[ISPC]: https://ispc.github.io/
[Vulkan Memory Types on PC and How to Use Them]: https://asawicki.info/news_1740_vulkan_memory_types_on_pc_and_how_to_use_them
[System and driver support for Resizable BAR]: https://docs.microsoft.com/en-us/windows-hardware/drivers/display/resizable-bar-support
[clvk]: https://github.com/kpet/clvk
