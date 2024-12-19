/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: debug_screen_shader.wgsl
 *
 *  Description: 
 *  This shader renders an image of debug view on a quad.
 *  
 * -----------------------------------------------------------------------------
 */
@group(0) @binding(0) var color_buffer: texture_2d<f32>; //current sample




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
    let texture_size = textureDimensions(color_buffer);
    var texel_size = 1.0 / vec2f(f32(texture_size.x),f32(texture_size.y));	
	
	var screen_pos = vec2(0);
    screen_pos.x = i32(uv.x * f32(texture_size.x));
    screen_pos.y = i32(uv.y * f32(texture_size.y));
    
    let acc = textureLoad(color_buffer,screen_pos,0).xyz;
    let gamma = 2.2f;
    return vec4f(pow(acc, vec3(1.0/gamma)),1.f);
}