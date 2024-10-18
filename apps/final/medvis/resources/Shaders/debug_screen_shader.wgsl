@group(0) @binding(0) var color_buffer: texture_2d<f32>; //current sample




struct VertexOutput{
    @builtin(position) Position: vec4<f32>,
    @location(0) TexCoord: vec2<f32>
}

@vertex
fn vert_main(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput{
    let positions = array<vec2<f32>,6>(
        vec2<f32>(1.0,1.0),
        vec2<f32>(1.0,-1.0),
        vec2<f32>(-1.0,-1.0),
        vec2<f32>(1.0,1.0),
        vec2<f32>(-1.0,-1.0),
        vec2<f32>(-1.0,1.0),
    );

    let texCoords = array<vec2<f32>,6>(
        vec2<f32>(1.0,0.0),
        vec2<f32>(1.0,1.0),
        vec2<f32>(0.0,1.0),
        vec2<f32>(1.0,0.0),
        vec2<f32>(0.0,1.0),
        vec2<f32>(0.0,0.0),
    );

    var output: VertexOutput;
    output.Position = vec4<f32>(positions[VertexIndex],0.0,1.0);
    output.TexCoord = vec2<f32>(texCoords[VertexIndex]);
    return output;
}


@fragment 
fn frag_main(@location(0) TexCoord: vec2<f32>) -> @location(0) vec4<f32>{
    let texture_size = textureDimensions(color_buffer);
    var texel_size = 1.0 / vec2f(f32(texture_size.x),f32(texture_size.y));	
	
	var screen_pos = vec2(0);
    screen_pos.x = i32(TexCoord.x * f32(texture_size.x));
    screen_pos.y = i32(TexCoord.y * f32(texture_size.y));
    
    let acc = textureLoad(color_buffer,screen_pos,0).xyz;
    let gamma = 2.2f;
    return vec4f(pow(acc, vec3(1.0/gamma)),1.f);
}