/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: raytracing_kernel_bvh_accumulated.wgsl
 *
 *  Description: 
 *  This shader manages ray evaluation through BVH stack-based traversal and performs shading using the Lambertian and Cook-Torrance BRDF.
 *  One ray of a path is evaluated.
 * -----------------------------------------------------------------------------
 */

const PI = 3.1415926535f;


@group(0) @binding(0) var<storage,read> bvh: array<BVHNode>;
@group(0) @binding(1) var<storage,read> spheres: array<Sphere>;
@group(0) @binding(2) var<uniform> uCamera: Camera;
@group(0) @binding(3) var<uniform> config: Config;
@group(0) @binding(4) var cubemapTexture: texture_cube<f32>;
@group(0) @binding(5) var cubemapSampler : sampler;

@group(1) @binding(0) var origin_buffer: texture_storage_2d<rgba32float,write>;
@group(1) @binding(1) var direction_buffer: texture_storage_2d<rgba32float,write>;
@group(1) @binding(2) var origin_buffer_read: texture_2d<f32>;
@group(1) @binding(3) var direction_buffer_read: texture_2d<f32>;
@group(1) @binding(4) var color_buffer: texture_storage_2d<rgba8unorm,write>;
@group(1) @binding(5) var color_buffer_read: texture_2d<f32>;
struct Camera {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
	position: vec4f,
	inversePV: mat4x4f,
};
struct Config{
	currentIteration: i32,
	maxIterations : i32,
	currentSample : i32,
	maxSamples : i32,
	time:f32,
	uniformRandom:f32
}
struct Light{
	position:vec3f,
	direction:vec3f,
	diffuse:vec3f
}

struct BVHNode {
	minAABB: vec3f,
	leftChild: i32,
	maxAABB: vec3f,
	numberOfSpheres: i32,
};

struct Sphere{
	//origin: vec4f
	origin: vec3f,
	radius: f32,
	color: vec4f
};

struct Ray {
    origin:vec3f,     
    rayID: i32,
	direction:vec3f,  
	iteration: i32
};


struct Hit {
    intersection:vec3f, 
	t:f32,				  
    normal:vec3f,             
	material:vec4f			  
};

struct RayDataTex{
	origin:vec3f,     
    rayID: i32,
	direction:vec3f,  
	iteration: i32,
	attenuation:f32,
	prob:f32
}
//const emptySphere = Sphere(vec3f(0.f),-1.f,vec4f(0.f)) ; 
const miss = Hit(vec3f(0.0f), 1e20, vec3f(0.0f), vec4f(1.f,0.f,0.f,0.f));//vec4f(0.f));
const missT = 1e20;
const numberOfPros = 8*4;

const roughness_global = 0.01; 

const primaryId = 1;
const reflectId = 1;
const refractId = 1;

@compute @workgroup_size(8,4,1)

fn main(@builtin(workgroup_id) WorkgroupID:vec3<u32>,@builtin(local_invocation_id) LocalID:vec3<u32>){
	//let maxIteration = 30;
	let firstFrame = true;
	let epsilon = 0.001f;
	//var color = vec3f(0.f);
	var screen_pos : vec2<i32> = vec2<i32>(i32(8*WorkgroupID.x + LocalID.x),i32(4*WorkgroupID.y + LocalID.y));
	var ray : Ray;
	var iteration  = i32(uCamera.position.w);

	let screen_size: vec2<u32> = textureDimensions(color_buffer);
	let uv_x: f32 = (f32(screen_pos.x) - f32(screen_size.x)/2) / f32(screen_size.x);
	let uv_y: f32 = (f32(screen_size.y) / 2 - f32(screen_pos.y)) / f32(screen_size.x);

	var uv = vec2f(uv_x,uv_y);
	
	
	//textureStore(color_buffer,screen_pos,vec4f(random(uv),0.,0.,1.));
	//return;
	
	var attenuation = 0.f;
	var stop_mark = 1.f;
	var prev_cos_theta = 1.f;
	//iteration=0;
	if(iteration == 0){ //first iteration
		//clear
		//textureStore(color_buffer,screen_pos,vec4f(0.,0.,0.,1.));
		ray = generatePrimaryRay(screen_pos);
	}else{
		
		
		let rayData = getRayDataFromTexture(screen_pos);
		ray.origin = rayData.origin;
		ray.direction = rayData.direction;
		stop_mark = rayData.attenuation;
		prev_cos_theta = rayData.prob;
	}
	
	if(iteration<config.maxIterations /*&& config.currentSample<config.maxSamples*/){
			textureStore(color_buffer,screen_pos,vec4f(1.,0.,0.,1.));
			textureStore(origin_buffer,screen_pos,vec4f(0.,0.,0.,stop_mark));
			textureStore(direction_buffer,screen_pos,vec4f(0.,0.,0.,1.));
	}else if(iteration==config.maxIterations-1 && config.currentSample==config.maxSamples-1){
		let prev_color = textureLoad(color_buffer_read,screen_pos,0).xyz;
		textureStore(color_buffer,screen_pos,vec4f(prev_color,1.));
		return;
	}else if(iteration>config.maxIterations+2){
		//return;
	}

	/*if(iteration<=1){
			textureStore(color_buffer,screen_pos,vec4f(0.,0.,0.,1.));
			textureStore(origin_buffer,screen_pos,vec4f(0.,0.,0.,1.));
			textureStore(direction_buffer,screen_pos,vec4f(0.,0.,0.,1.));
	}else{
		return;
	}*///config.currentIteration == 0
	//var color = select(textureLoad(color_buffer_read,screen_pos,0).xyz,vec3f(0.f),config.currentIteration == 0);
	var color = select(textureLoad(color_buffer_read,screen_pos,0).xyz,vec3f(1.f,1.f,1.f),config.currentIteration == 0);
	//var color = textureLoad(color_buffer_read,screen_pos,0).xyz;
	//if(config.currentIteration==0){stop_mark=1.f; prev_cos_theta=1.f;}
	var stop = false;
	if(stop_mark != 0.f){ //not blank ray
		let hit = Trace(ray,attenuation,LocalID);
		if(hit.t != missT){
			
			if(hit.material.w == 0.f){//is light
				color = hit.material.xyz*color;
				stop = true;
				textureStore(origin_buffer,screen_pos,vec4f(0.f));
			}else{
				

				//diffuse
				//let r_x = random((vec2(config.uniformRandom*f32(config.currentSample*config.currentIteration))*uv));
				//let r_y = random((vec2(config.uniformRandom*2.*f32(config.currentSample*config.currentIteration))*uv));
				let r_x = random((vec2f(config.uniformRandom)*uv));
				let r_y = random(vec2f(r_x)*uv);
				if(r_y<0.){
						textureStore(color_buffer,screen_pos,vec4f(0.,1.,0.,0.));
						return;
					}
					if(r_y>1.){
						textureStore(color_buffer,screen_pos,vec4f(0.,0.,1.,0.));
						return;
					}
				//if( (r_x)<roughness_global){
				if(false){
					

					var L = cosineWSampleHemisphere(r_x,r_y);//uniformSampleHemisphere(r_x,r_y);

					let N = hit.normal;
					var T = vec3f(0.f);
					
					if (N.x<N.y && N.x<N.z) {
						T = cross(N, vec3f(1., 0., 0.));
					} else if (N.y<N.z) {
						T = cross(N, vec3f(0., 1., 0.));
					}else{
						T = cross(N, vec3f(0., 0., 1.));
					}

					T = normalize(T);
					let B = normalize(cross(N, T));
					L = T * L.x + B * L.y+N * L.z;
				
					let cos_theta = max(0.0f,dot(hit.normal,L));

					let newRay = Ray(hit.intersection +  epsilon * hit.normal, reflectId,L,iteration);
					textureStore(origin_buffer,screen_pos,vec4f(newRay.origin,1.f)); //hit.material.w atten
					textureStore(direction_buffer,screen_pos,vec4f(newRay.direction,(cos_theta)));
					
					var pbr = getDiffuseColor(hit.material.xyz);
					let pdf = cos_theta/PI;
					
					//let H = normalize(L+(-ray.direction));
					//let F = FresnelSchlick(hit.material.xyz, -ray.direction,H);

					if(config.currentIteration == config.maxIterations-1){
						color = vec3f(0.f);// vec3f(pbr)*PI;//vec3f(pbr)*PI;//vec3f(1.,0.,0.);
					}else{
						color *= /*(1.-F)*/ vec3f(pbr)*PI;//cos_theta* (1/pdf);
					}
				}
				else if(false){
					let N = hit.normal;
					var refDir = normalize(refract(ray.direction,N,1.00029f/1.5f));
					var refRay = Ray(hit.intersection +  epsilon * refDir,1, refDir,0);
					var refHit = Evaluate(refRay,LocalID);

					let L = normalize(refDir);
					let NdotL = dot(-N,L);

					color *= refHit.material.xyz*NdotL;

					//ref
					refDir = normalize(refract(hit.intersection,refHit.normal,1.5f/1.00029f));
					refRay = Ray(refHit.intersection +  epsilon * refDir,1, refDir,0);
					
					
					textureStore(origin_buffer,screen_pos,vec4f(refRay.origin,1.f)); //hit.material.w atten
					textureStore(direction_buffer,screen_pos,vec4f(refRay.direction,(0.)));

					//color += refHit.material.xyz * max(NdotV, 0.0); 
				
				}else{
					//let r_x = random((vec2(config.uniformRandom)*uv));
					//let r_y = random((vec2(config.uniformRandom*2.)*uv));

					//var reflected = importantSampleHemisphereIBL(r_x,r_y,roughness_global);//uniformSampleHemisphere(r_x,r_y);
					var L = importantSampleHemisphereBeckmann(r_x,r_y,roughness_global);//cosineWSampleHemisphere(r_x,r_y);
					var V = normalize(-ray.direction);
					var N = hit.normal;
					var T = vec3f(0.f);
					/*if (abs(N.y) < 0.999f) {
						T = cross(N, vec3f(0., 1., 0.));
					} else {
						T = cross(N, vec3f(1., 0., 0.));
					}*/
					if (N.x<N.y && N.x<N.z) {
						T = cross(N, vec3f(1., 0., 0.));
					} else if (N.y<N.z) {
						T = cross(N, vec3f(0., 1., 0.));
					}else{
						T = cross(N, vec3f(0., 0., 1.));
					}

					T = normalize(T);
					let B = normalize(cross(N, T));
					L = T * L.x + B * L.y+N * L.z;
					/*reflected.x = T.x * reflected.x + B.x * reflected.y+ N.x * reflected.z;
					reflected.y = T.y * reflected.x + B.y * reflected.y+ N.y * reflected.z;
					reflected.z = T.z * reflected.x + B.z * reflected.y+ N.z * reflected.z;*/
					L = normalize(L);
					//reflected = normalize(reflect(ray.direction,hit.normal));
					//reflected = reflected + N;
					var H = L;
					L = reflect(-V,H);
					//L = normalize(L);
					let cos_theta_L = max(0.f,dot(hit.normal,L));

					
					
					//var pbr = getSpecularColor(hit.material.xyz, hit.normal,ray.direction,reflected,prev_cos_theta);
					// ------------------------------ SPECULAR ------------------------------
					//calculate diffuse + specular
					//var L = normalize(light.position -  hit.intersection);
					
					
					//H = normalize((V)+L);
					let cos_theta_H = max(dot(N, H), 0.001);
					//var L = reflect(-V,H);
					//let H = normalize(reflected + V);
					L = normalize(L);
					N = normalize(N);
					V = normalize(V);
					H = normalize(H);

					let NdotL = max(dot(N, L), 0.001);
					var NdotV = max(dot(N,V), 0.001);
					let NdotH = clamp(dot(N, H), 0.001, 1.0);
					let VdotH = clamp(dot(V, H), 0.001, 1.0);

					let Geom = GeometricAttenuation_smith(N, V, L, H);// * GeometricAttenuation(N, reflected, L, H); //todo: použít i reflected?
					let Dist = BeckmannDistribution(N,H,roughness_global);//BeckmannDistribution(N, H, roughness_global);
					let F = FresnelSchlick(hit.material.xyz, V,H);
					let nominator = 4*NdotL*NdotV;//clamp((4*NdotL*NdotV), 0.001, 1.0);; //todo: dot product s reflected?
					var f_ct = (F*Geom*Dist)/nominator;
					//let ks = F;
					//let kd = (1.0-ks);
					var specular = ((f_ct));/// (4*abs(dot(H,N)));
					let pbr = specular;// vec3f(F*GGX_Distribution(N,H,roughness_global));//specular;//*hit.material.xyz;
					var pdf = (Dist * NdotH) / (4*VdotH);//clamp((Dist * NdotH) / (4*VdotH), 0.001, 1.0);

					let sin_theta_L = sqrt(1. - cos_theta_L * cos_theta_L);
					let sin_theta_H = sqrt(1. - cos_theta_H * cos_theta_H);
					//pdf*=sin_theta_L;
					let newRay = Ray(hit.intersection +  epsilon * hit.normal, reflectId,normalize(L),iteration);
					textureStore(origin_buffer,screen_pos,vec4f(newRay.origin,1.f)); //hit.material.w atten
					textureStore(direction_buffer,screen_pos,vec4f(newRay.direction,(NdotL)));
					
					// ----------------------------------------------------------------------
					if(config.currentIteration == config.maxIterations-1){
						color = vec3f(0.,0.,0.);
					}else{
						color *= pbr*NdotL* (1./pdf);//pbr*NdotL* (1./pdf);// * sin_theta_L*NdotL*(1./pdf);
					}
					
				}
			}
		}else{
			if(iteration==0){
				color = textureSampleLevel(cubemapTexture, cubemapSampler, ray.direction,0).rgb;
				textureStore(origin_buffer,screen_pos,vec4f(0.f));
			}else{
				color *= ((textureSampleLevel(cubemapTexture, cubemapSampler, ray.direction,0).rgb));// * prev_cos_theta;
				//color *= vec3f(0.f);
				textureStore(origin_buffer,screen_pos,vec4f(0.f));
			}
			stop = true;
		}

	}
	//textureStore(color_buffer,screen_pos,vec4f(color,1.));
	var count = 1.;
	/*if(stop || config.maxIterations-1 == config.currentIteration){
		count = f32(config.currentIteration+1);
	}*/
	textureStore(color_buffer,screen_pos,vec4f(color,count));// /f32(config.maxSamples),1.));
	
	//let ibl_sample = textureSampleLevel(cubemapTexture, cubemapSampler, ray.direction,0).rgb;
	//textureStore(color_buffer,screen_pos,vec4f(ibl_sample,1.));
	
}

fn getRayDataFromTexture(screen_pos : vec2<i32>) -> RayDataTex{
	var rayData:RayDataTex;
	let origin = textureLoad(origin_buffer_read,screen_pos,0);
	let direction = textureLoad(direction_buffer_read,screen_pos,0);
	rayData.direction = direction.xyz;
	rayData.origin = origin.xyz;
	rayData.attenuation = origin.w;
	rayData.prob = direction.w;
	return rayData;

}

fn getRayFromTexture(screen_pos : vec2<i32>) -> Ray{
	var ray:Ray;
	let origin = textureLoad(origin_buffer_read,screen_pos,0).xyz;
	let direction = textureLoad(direction_buffer_read,screen_pos,0).xyz;
	ray.direction = direction;
	ray.origin = origin;
	return ray;

}

fn generatePrimaryRay(screen_pos : vec2<i32>) -> Ray{
	let screen_size: vec2<u32> = textureDimensions(color_buffer);
	let uv_x: f32 = (f32(screen_pos.x) - f32(screen_size.x)/2) / f32(screen_size.x);
	let uv_y: f32 = (f32(screen_size.y) / 2 - f32(screen_pos.y)) / f32(screen_size.x);

	var uv = vec2f(uv_x,uv_y);
	let aspect_ratio = 1280.0/720.0;

	let P = (uCamera.inversePV * vec4f(uv, -1.f, 1.0)).xyz;
	var ray:Ray;
	ray.direction = normalize(P - uCamera.position.xyz);
	ray.origin = uCamera.position.xyz;

	return ray;

}
fn RaySphereIntersection(ray : Ray, sphereIndex : i32) -> Hit{//sphere : Sphere) -> Hit{
	let sphere = spheres[sphereIndex];
	//return miss;
	// Optimalized version.
	let oc = ray.origin - sphere.origin.xyz;
	let b = dot(ray.direction, oc);
	let c = dot(oc, oc) - (sphere.radius*sphere.radius);

	let det = b*b - c;
	if (det < 0.0) {
		return miss;
	}

	var t = -b - sqrt(det);
	if (t < 0.0){ 
		t = -b + sqrt(det);
	}
	if(t>=0){
		let intersection = oc + t * ray.direction;
		var n = normalize(intersection);
		if(dot(ray.direction, n) > 0) {
			n = -n; 
		}
		return Hit(intersection+sphere.origin.xyz, t, n,sphere.color);
	}else{
		return miss;
	}

	
}

fn RayCustomSphereIntersection(ray : Ray, sphere : Sphere) -> Hit{//sphere : Sphere) -> Hit{
	//let sphere = spheres[sphereIndex];
	//return miss;
	// Optimalized version.
	let oc = ray.origin - sphere.origin.xyz;
	let b = dot(ray.direction, oc);
	let c = dot(oc, oc) - (sphere.radius*sphere.radius);

	let det = b*b - c;
	if (det < 0.0) {
		return miss;
	}

	var t = -b - sqrt(det);
	if (t < 0.0){ 
		t = -b + sqrt(det);
	}
	if(t>=0){
		let intersection = oc + t * ray.direction;
		var n = normalize(intersection);
		if(dot(ray.direction, n) > 0) {
			n = -n; 
		}
		return Hit(intersection+sphere.origin.xyz, t, n,sphere.color);
	}else{
		return miss;
	}

}


fn Trace(ray : Ray,prev_F:f32,LocalID:vec3<u32>) -> Hit{

	var color = vec3f(0.0,0.0,0.0);
	let epsilon = 0.001f;

	var hit = Evaluate(ray,LocalID);
	
    return hit;
}

fn Evaluate(ray : Ray,LocalID:vec3<u32>) -> Hit{

	var closest_hit = miss;//RayPlaneIntersection(ray, vec3f(-1.0f, 0.f, 0.f), vec3f(7.0f,0.0f,0.0f),vec4f(0.5f,0.5f,0.5f,0.f));
	
	var bottom_plane_hit = miss;//RayPlaneIntersection(ray, vec3f(0.0f, 1.f, 0.f), vec3f(0.0f,2.0f,0.0f),vec4f(0.5f,0.5f,0.5f,1.f));
	var emissive_sphere_hit = RayCustomSphereIntersection(ray,Sphere(vec3f(0.0f, 5.f, 0.f),2.f,vec4f(1.0f,1.f,1.f,0.f)));

	if(bottom_plane_hit.t<closest_hit.t){
		closest_hit = bottom_plane_hit;
	}
	if(emissive_sphere_hit.t<closest_hit.t){
		closest_hit = emissive_sphere_hit;
	}
	//closest_hit = RayCustomSphereIntersection(ray,Sphere(vec3f(0.0f, 2.3f, 0.f),2.f,vec4f(1.0f,1.f,1.f,0.f)));;
	/*var bottom_plane_hit = RayPlaneIntersection(ray, vec3f(0.0f, -1.f, 1.f), vec3f(0.0f,5.0f,0.0f),vec4f(0.2f,0.5f,1.0f,1.f));
	if(bottom_plane_hit.t<closest_hit.t){
		closest_hit = bottom_plane_hit;
	}*/
	var kdResultHit = traverseKdTree(ray,LocalID);
	if(kdResultHit.t <= closest_hit.t){
		return kdResultHit;
	}

    return closest_hit;
}
fn test_renderAllSpheres(ray : Ray) -> Hit{
	var closest_hit = miss;
	let n = arrayLength(&spheres);
	for(var j : u32 = 0; j < n; j++){
		let hit = RaySphereIntersection(ray,i32(j));
		if(hit.t < closest_hit.t){
			closest_hit = hit;
		}

	}
	return closest_hit;
}
fn traverseKdTree(ray :Ray,LocalID:vec3<u32>) -> Hit{
	//return miss;
	return test_renderAllSpheres(ray);
	//return traverseBVH(ray);
	//return traverseBVH_packet(ray,LocalID);
	//return miss;

}


fn RayBoxIntersection(ray : Ray, minP : vec3f, maxP : vec3f) -> vec2f{ 

	let eps = 0.00001;
	
	let ray_min_tmp = (minP - ray.origin) /  (ray.direction);
	let ray_max_tmp = (maxP - ray.origin) / (ray.direction);

	let ray_min = min(ray_min_tmp,ray_max_tmp);
	let ray_max = max(ray_min_tmp,ray_max_tmp);

	let tmin = max(max(ray_min.x,ray_min.y),ray_min.z);
	let tmax = min(min(ray_max.x,ray_max.y),ray_max.z);

	if(tmin>tmax){ return vec2(missT,missT);}
	if(tmax<0){ return vec2(missT,missT);}

	return vec2(tmin,tmax);
}




fn getCloserHit(hit1: Hit, hit2: Hit) -> Hit{
	if(hit1.t<hit2.t){
		return hit1;
	}else{
		return hit2;
	}
}

fn isHitMiss(hit:Hit) -> bool{

	if(hit.t != miss.t){
		return false;
	}

	return true;
}





fn BeckmannDistribution(N:vec3f, H:vec3f, m:f32) -> f32
{
	let NdotH = max(0.0, dot(N, H));
	return ( max(0.0, exp((NdotH*NdotH - 1.0) / (m*m * NdotH*NdotH))) /max(0.0001, (m*m * NdotH*NdotH*NdotH*NdotH)) );
}

fn SmithGGX(N:vec3f, H:vec3f, m:f32) -> f32
{
	let NdotH = max(0.0, dot(N, H));
	let m2 = m*m;
	let dd = (PI*((m2-1.)*NdotH*NdotH + 1.));
	let dd2 = dd*dd;
	return (m2) / (PI*dd2);
}





fn GeometricAttenuation_smith(N:vec3f,V: vec3f,L:vec3f, H:vec3f) -> f32
{
	let NdotH = max(0.0, dot(N, H));
	let NdotV = max(0.0, dot(N, V));
	let VdotH = max(0.0, dot(V, H));
	let NdotL = max(0.0, dot(N, L));

	let alpha2 = roughness_global*roughness_global;

	let nom = 2. * NdotL*NdotV;
	let ii = NdotL *sqrt(alpha2+(1.-alpha2)*NdotV*NdotV);
	let rr = NdotV *sqrt(alpha2+(1.-alpha2)*NdotL*NdotL);
	return nom/(rr+ii);
}

fn random(st:vec2f)->f32{
	return fract(sin(dot(st.xy,vec2(12.9898,78.233)))*43758.5453123);
}


// --------------------------- BVH ---------------------------
fn traverseBVH(ray:Ray) -> Hit{
	var nodeId = 0;
	var node: BVHNode = bvh[nodeId];
    var stack: array<BVHNode, 32>;
    var stackLocation: u32 = 0;
	var closest_hit = miss;
	var j = 0;
	let main = RayBoxIntersection(ray,node.minAABB,node.maxAABB);

	 while (true) {
		j++;
        
		
        if (node.leftChild <= 0) {
			
				/*var h = miss;
				h.material = vec4(0.f,0.f,1.f,1.f);
				h.t = 10.f;
				return h;*/
			
            for (var i = 0; i < node.numberOfSpheres; i++) {
				let sphereId = i+(-1*node.leftChild);
                var hit = RaySphereIntersection(ray, sphereId);
				//hit.t = -10. * spheres[sphereId].radius;
                if(hit.t < closest_hit.t){
					closest_hit = hit;
				}
            }

			if (stackLocation == 0) {
				break;
				
            }
            else {
                stackLocation -= 1;
                node = stack[stackLocation];
            }
			continue;
        }
        var child1: BVHNode = bvh[node.leftChild];
		var child2: BVHNode = bvh[node.leftChild+1];

		var distance1 = RayBoxIntersection(ray,child1.minAABB,child1.maxAABB).x;//hit_aabb(ray, child1);
		var distance2 = RayBoxIntersection(ray,child2.minAABB,child2.maxAABB).x;
		/*if(distance1<miss.t && distance1 != -999.f){
			var h = miss;
			h.material = vec4(1.f,0.f,0.f,1.f);
			h.t = distance1;
			return h;
		}*/
		//if((distance1 > distance2 && distance2 != missT) || (distance1 == missT && distance2 != missT)){
		if(distance1 > distance2){
			//swap
			var c = child1;
			child1 = child2;
			child2 = c;

			var tmp = distance1;
			distance1 = distance2;
			distance2 = tmp;
		}
		if(distance1 == missT){//if miss
			if (stackLocation == 0) {

                break;
            }
            else {

                stackLocation -= 1;
                node = stack[stackLocation];
            }
		}else{
			node = child1;
			if(distance2 != missT){
				//stackLocation++;
				stack[stackLocation] = child2;
				stackLocation++;
			}
		}
		
		
		//j++;
    }
	/*closest_hit.material = vec4f(abs(node.minAABB.x/400.f),0.f,0.f,1.f);//vec4f(0.f,1.f,0.f,1.f);
	closest_hit.t = 10.f;*/

	return closest_hit;
}


fn uniformSampleHemisphere(r1:f32, r2:f32) -> vec3f
{

	let z = sqrt(r1);//(PI/2.0)*r1;
	let theta = acos(r1);
    let sinTheta = sin(theta);//sqrt(1 - r1 * r1);
    let phi = 2 * PI * r2;

    let x = sinTheta * cos(phi);
    let y = sinTheta * sin(phi);
    return vec3f(x, y, cos(theta));
}

fn cosineWSampleHemisphere(r1:f32, r2:f32) -> vec3f
{

	let z = sqrt(r1);//(PI/2.0)*r1;
	let theta = acos(z);
    let sinTheta = sin(theta);//sqrt(1 - r1 * r1);
    let phi = 2 * PI * r2;
    let x = z * cos(phi);
    let y = z * sin(phi);
    return vec3f(x, y, sqrt(1-r1));
}

fn importantSampleHemisphereBeckmann(r1:f32, r2:f32, m:f32) -> vec3f
{
	let theta = atan(sqrt(-1.*m*m*log(1-r1)));
	let phi = 2*PI*r2;
	let sinTheta = sin(theta);
	let x = sinTheta*cos(phi);
	let y = sinTheta*sin(phi);
	let z = cos(theta);
    return vec3f(x, y, z);
}



fn getDiffuseColor(albedo : vec3f) -> vec3f{
	return albedo/PI;
}


fn FresnelSchlick(color :vec3f, V:vec3f,H:vec3f) -> vec3f
{
	let VdotH = clamp(dot(V, H), 0.001, 1.0);

    var F0_vec3 = color;//vec3f(0.95,0.64,0.54);//(1.0 - metallic) * F0 + metallic * color.xyz;
	let res = F0_vec3 + (1.0f-F0_vec3) * (pow(clamp(1.0f - VdotH,0.f,1.f),5));
	return res;

	//return f0 + (1.0 - f0) * pow(1.0 - VdotH, 5.0);
}