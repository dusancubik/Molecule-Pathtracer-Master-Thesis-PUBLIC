struct VertexInput {
	@location(0) position: vec3f,
    //                        ^ This was a 2
	@location(1) color: vec3f,
};

struct VertexOutput {
	@builtin(position) position: vec4f,
	@location(0) tex_coord: vec2f,
};

/**
 * A structure holding the value of our uniforms
 */
struct MyUniforms {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,
    color: vec4f,
    time: f32,
};

// Instead of the simple uTime variable, our uniform variable is a struct

@group(0) @binding(0) var<uniform> uCamera: Camera;
const tex_coords = array<vec2f,3>(
	vec2f(0.0, 0.0),
	vec2f(2.0, 0.0),
	vec2f(0.0, 2.0)
);
@group(0) @binding(1) var<storage,read> spheres: array<Sphere>;
struct Camera {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
	position: vec4f,
	inversePV: mat4x4f
};
struct Sphere{
	origin: vec3f,
	radius: f32,
	color: vec4f
}



@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
	//return vs_main_optionA(in);
	var texCoord = tex_coords[in_vertex_index];
	var out : VertexOutput;
	out.tex_coord = texCoord;
	out.position = vec4f(texCoord*2.0 - 1.0,0.0,1.0);
	return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
	let color = vec3f(1.,0.f,0.5f);
	// Gamma-correction
	let corrected_color = pow(color, vec3f(2.2));
	return vec4f(color, 1.0);
}