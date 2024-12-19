/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: screen_shader.wgsl
 *
 *  Description: 
 *  This shader renders the image from ray tracing pass on a quad.
 * -----------------------------------------------------------------------------
 */
@group(0) @binding(0) var screen_sampler : sampler;
@group(0) @binding(1) var color_buffer: texture_2d<f32>;

struct VertexOutput{
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>
}

const positions = array<vec2<f32>,6>(
    vec2<f32>(1.0,1.0),
    vec2<f32>(1.0,-1.0),
    vec2<f32>(-1.0,-1.0),
    vec2<f32>(1.0,1.0),
    vec2<f32>(-1.0,-1.0),
    vec2<f32>(-1.0,1.0),
);

const UVs = array<vec2<f32>,6>(
    vec2<f32>(1.0,0.0),
    vec2<f32>(1.0,1.0),
    vec2<f32>(0.0,1.0),
    vec2<f32>(1.0,0.0),
    vec2<f32>(0.0,1.0),
    vec2<f32>(0.0,0.0),
);

@vertex
fn vs_main(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput{
    var output: VertexOutput;
    output.pos = vec4<f32>(positions[VertexIndex],0.0,1.0);
    output.uv = vec2<f32>(UVs[VertexIndex]);
    return output;
}

@fragment 
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32>{
    return textureSample(color_buffer,screen_sampler,uv);
}