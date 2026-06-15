// icl ts is ai generated code so that i can learn webgpu and wgsl

// const a = new Float32Array([5]);
// const b = new Float32Array([1]); // not strictly needed, but keeping your idea

// async function init() {
//     if (!navigator.gpu) {
//         throw Error("WebGPU not supported.");
//     }

//     const adapter = await navigator.gpu.requestAdapter();
//     const device = await adapter.requestDevice();

//     // --------

//     // 1. Create buffer for input (a = 5)
//     const inputBuffer = device.createBuffer({
//         size: a.byteLength,
//         usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
//     });

//     // 2. Create buffer for output
//     const outputBuffer = device.createBuffer({
//         size: a.byteLength,
//         usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
//     });

//     // 3. Send data into GPU
//     device.queue.writeBuffer(inputBuffer, 0, a);

//     // 4. WGSL shader (GPU code)
//     const shaderModule = device.createShaderModule({ // this code will be ran on the gpu
//         code: `
//         @group(0) @binding(0)
//         var<storage, read> input: array<f32>;

//         @group(0) @binding(1)
//         var<storage, read_write> output: array<f32>;

//         @compute
//         @workgroup_size(1)
//         fn main(@builtin(global_invocation_id) id: vec3<u32>) {
//             output[id.x] = input[id.x] + 1.0;
//         }
//         `
//     });

//     // 5. Pipeline
//     const pipeline = device.createComputePipeline({ // a precompiled gpu program configuration that tells the gpu exactly how to run your shader
//         layout: "auto",
//         compute: {
//             module: shaderModule,
//             entryPoint: "main"
//         }
//     });

//     // 6. Bind buffers to WGSL
//     const bindGroup = device.createBindGroup({
//         layout: pipeline.getBindGroupLayout(0),
//         entries: [
//             { binding: 0, resource: { buffer: inputBuffer } },
//             { binding: 1, resource: { buffer: outputBuffer } }
//         ]
//     });

//     // 7. Run GPU
//     const encoder = device.createCommandEncoder();

//     const pass = encoder.beginComputePass();
//     pass.setPipeline(pipeline);
//     pass.setBindGroup(0, bindGroup);

//     pass.dispatchWorkgroups(1);
//     pass.end();

//     device.queue.submit([encoder.finish()]);
// }