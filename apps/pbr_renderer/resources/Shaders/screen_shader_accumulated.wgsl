/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: screen_shader_accumulated.wgsl
 *
 *  Description: 
 *  This shader accumulates samples and allows denoising using Bilateral filter.
 *  The rendered image is drawn on a quad.
 *  The Bilateral filter code is taken from https://github.com/tranvansang/bilateral-filter/blob/master/fshader.frag and is under MIT license.
 * -----------------------------------------------------------------------------
 */
@group(0) @binding(0) var screen_sampler : sampler;
@group(0) @binding(1) var<uniform> config: Config;
@group(0) @binding(2) var<uniform> bilateralFilterConfig: BilateralFilterConfig;
@group(1) @binding(0) var color_buffer: texture_2d<f32>; //current sample
@group(2) @binding(0) var accumulation_write: texture_storage_2d<rgba8unorm,write>;
@group(2) @binding(1) var accumulation_read: texture_2d<f32>;

//@group(3) @binding(0) var final_output_write: texture_storage_2d<rgba8unorm,write>;
//@group(3) @binding(1) var final_output_read: texture_2d<f32>;


struct Config{
	currentIteration: i32,
	maxIterations : i32,
	currentSample : i32,
	maxSamples : i32,
	time:f32,
	uniformRandom:f32,
	debugMode: i32,
	debugCollectingMode: i32,
	debugRayIndex: i32
}

struct AccumulationConfig{
	accumulationFinished: i32,
    filterUpdated: i32
}

struct BilateralFilterConfig{
    accumulationFinished: i32,
	on : i32,
    sigmaS : f32,
    sigmaL : f32,
}

struct VertexOutput{
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>
}

fn random(st:vec2f)->f32{
	return fract(sin(dot(st.xy,vec2(12.9898,78.233)))*43758.5453123);
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
    //posisions of a quad
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

    let gamma = 2.2f;
    if(bilateralFilterConfig.accumulationFinished == 0){//if accumulation 
        if(config.currentSample == 0){
            //let new_color = vec3(1.0,0.0,0.0);//textureLoad(color_buffer,screen_pos,0).xyz;
            let new_color = textureLoad(color_buffer,screen_pos,0).xyz;

            textureStore(accumulation_write,screen_pos,vec4f(new_color,0.f));
            return vec4f(pow(new_color, vec3(1.0/gamma)),1.f);
        }

        if(config.currentIteration == config.maxIterations-1){//je to posledni iterace
            var new_color = textureLoad(color_buffer,screen_pos,0).xyz;
            let countIt = textureLoad(color_buffer,screen_pos,0).w;
        // new_color = new_color / countIt;
            
            let acc = textureLoad(accumulation_read,screen_pos,0).xyz;
            let currentSample_f32 = f32(config.currentSample);
            let new_accumulation = (acc*f32(config.currentSample)+new_color)/f32(config.currentSample+1);
            //let new_accumulation = (acc+new_color/64.f);
            textureStore(accumulation_write,screen_pos,vec4f(new_accumulation,0.f));
          
            return vec4f(pow(new_accumulation, vec3(1.0/gamma)),1.f);
        }

        let acc = textureLoad(accumulation_read,screen_pos,0).xyz;

        
        return vec4f(pow(acc, vec3(1.0/gamma)),1.f);
    }else{//already accumulated
        //else if(config.currentSample == config.maxSamples-2 ){
                //SAVE LAST ACCUMULATION TO TEXTURE

        if(bilateralFilterConfig.on == 1){
            //var new_color = textureLoad(color_buffer,screen_pos,0).xyz;
            var new_color = textureLoad(accumulation_read,screen_pos,0).xyz;
            let sigmaS = bilateralFilterConfig.sigmaS;//bilateralFilterConfig.sigmaS;
            let sigmaL = bilateralFilterConfig.sigmaL;//bilateralFilterConfig.sigmaL;
            let EPS = 0.0001;
            var sigS = max(sigmaS, EPS);
            var sigL = max(sigmaL, EPS);

            var facS = -1./(2.*sigS*sigS);
            var facL = -1./(2.*sigL*sigL);

            var sumW = EPS;
            var sumC = vec4f(0.f);
            var halfSize = sigS * 2;
            //ivec2 textureSize2 = textureSize(texture, 0);

            var l = length(new_color.xyz);

            for (var i = i32(-halfSize); i <= i32(halfSize); i ++){
                for (var j = i32(-halfSize); j <= i32(halfSize); j ++){
                var pos = vec2i(i, j);

                var offsetColor = textureLoad(accumulation_read, screen_pos + pos,0);

                //var offsetColor = textureLoad(color_buffer, screen_pos + pos / vec2i(texture_size),0);

                var distS = length(vec2f(pos));
                var distL = length(offsetColor.xyz)-l;

                var wS = exp(facS*f32(distS*distS));
                var wL = exp(facL*f32(distL*distL));
                var w = wS*wL;

                sumW += w;
                sumC += offsetColor * w;
                }
            }
            new_color = sumC.xyz/sumW;
            //textureStore(accumulation_write,screen_pos,vec4f(new_color,0.f));

            return vec4f(pow(new_color, vec3(1.0/gamma)),1.f);
        }else{
            let acc = textureLoad(accumulation_read,screen_pos,0).xyz;

            return vec4f(pow(acc, vec3(1.0/gamma)),1.f);
        }
    }
    //return vec4f(acc,1.f);
    
    //return textureSample(color_buffer,screen_sampler,TexCoord);
}