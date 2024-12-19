/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
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


@group(0) @binding(0) var<uniform> uCamera: Camera;
@group(0) @binding(1) var<uniform> config: Config;
@group(0) @binding(2) var<storage,read> materials: array<Material>;
@group(0) @binding(3) var<storage, read_write> debugLineDataArray: array<DebugData>;
@group(0) @binding(4) var<uniform> debugConfig: DebugConfig;

@group(1) @binding(0) var origin_buffer: texture_storage_2d<rgba32float,write>;
@group(1) @binding(1) var direction_buffer: texture_storage_2d<rgba32float,write>;
@group(1) @binding(2) var origin_buffer_read: texture_2d<f32>;
@group(1) @binding(3) var direction_buffer_read: texture_2d<f32>;
@group(1) @binding(4) var color_buffer: texture_storage_2d<rgba8unorm,write>;
@group(1) @binding(5) var color_buffer_read: texture_2d<f32>;

@group(2) @binding(0) var cubemapTexture: texture_cube<f32>;
@group(2) @binding(1) var cubemapSampler : sampler;

//BVH+spheres
@group(3) @binding(0) var<storage,read> bvh: array<BVHNode>;
@group(3) @binding(1) var<storage,read> spheres: array<Sphere>;

struct Material{
	colour: vec4f
}

struct DebugData{
	data: vec4f,
	values: array<vec4f,10>
}
struct DebugConfig{
	coordinates: vec4f,
	visOptions: vec4f,
	cameraPosition: vec4f,
}
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
	uniformRandom:f32,
	debugMode: i32,
	debugCollectingMode: i32,
	debugRayIndex: i32
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
	materialIndex: f32
};
struct Ray {
    origin:vec3f,     
    rayID: i32,
	direction:vec3f,  
	iteration: i32
};

// The definition of an intersection.
struct Hit {
    intersection:vec3f,      
	t:f32,				  
    normal:vec3f,            
	material:vec4f			  
};

struct RayDataTex{
	origin:vec3f,     // The ray origin.
    rayID: i32,
	direction:vec3f,  // The ray direction.
	iteration: i32,
	stop:f32,
	prob:f32
}
//const emptySphere = Sphere(vec3f(0.f),-1.f,vec4f(0.f)) ; 
const miss = Hit(vec3f(0.0f), 1e20, vec3f(0.0f), vec4f(0.f,0.f,0.f,0.f));//vec4f(0.f));
const missT = 1e20;
const numberOfPros = 8*4;

const roughness_global = 0.051; 

const primaryId = 1;
const reflectId = 1;
const refractId = 1;

@compute @workgroup_size(8,4,1)
fn main(@builtin(workgroup_id) WorkgroupID:vec3<u32>,@builtin(local_invocation_id) LocalID:vec3<u32>){
	//let maxIteration = 30;
	let firstFrame = true;
	let epsilon = 0.01f;
	//var color = vec3f(0.f);
	var screen_pos : vec2<i32> = vec2<i32>(i32(8*WorkgroupID.x + LocalID.x),i32(4*WorkgroupID.y + LocalID.y));
	var ray : Ray;
	var iteration  = config.currentIteration;//i32(uCamera.position.w);

	let screen_size: vec2<u32> = textureDimensions(color_buffer);
	let uv_x: f32 = (f32(screen_pos.x) - f32(screen_size.x)/2) / f32(screen_size.x);
	let uv_y: f32 = (f32(screen_size.y) / 2 - f32(screen_pos.y)) / f32(screen_size.x);

	var uv = vec2f(uv_x,uv_y);
	
	
	


	
	var stop_mark = 1.f; //if path was terminated
	
	if(iteration == 0){ 
		//If first bounce -> generate primary ray
		ray = generatePrimaryRay(uv);
	}else{
		//Not first bounce -> load ray from texture
		
		let rayData = getRayDataFromTexture(screen_pos);
		ray.origin = rayData.origin;
		ray.direction = rayData.direction;
		stop_mark = rayData.stop;
	}
	if(iteration<config.maxIterations /*&& config.currentSample<config.maxSamples*/){
			textureStore(color_buffer,screen_pos,vec4f(1.,0.,0.,1.));
			textureStore(origin_buffer,screen_pos,vec4f(0.,0.,0.,stop_mark));
			textureStore(direction_buffer,screen_pos,vec4f(0.,0.,0.,1.));
	}else if(iteration==config.maxIterations-1 && config.currentSample==config.maxSamples-1){
		let prev_color = textureLoad(color_buffer_read,screen_pos,0).xyz;
		textureStore(color_buffer,screen_pos,vec4f(prev_color,1.));
		return;
	}

	//Load color/throughput from previous ray of the path
	var color = select(textureLoad(color_buffer_read,screen_pos,0).xyz,vec3f(1.f,1.f,1.f),config.currentIteration == 0);
	
	//If debug collecting is ON -> ray origins are stored in the debugLineDataArray buffer
	if(i32(debugConfig.coordinates.z) == 1){
		
		if(screen_pos.x == i32(debugConfig.coordinates.x) && screen_pos.y == i32(debugConfig.coordinates.y)){
			
			let index = config.currentSample;
			//if(config.currentSample == 5 || config.currentSample == 6){
				if(stop_mark == 0.f){
					if(debugLineDataArray[index].data.x == f32(config.currentIteration)){
						debugLineDataArray[index].values[config.currentIteration] = vec4f(ray.origin + 1000.f*ray.direction,0.);
					}
					
				}else{
					debugLineDataArray[index].values[config.currentIteration] = vec4f(ray.origin,0.);//vec4f(f32(config.currentIteration))*f32(config.currentSample);//vec4f(ray.origin,0.);
					debugLineDataArray[index].data.x = f32(config.currentIteration+1);
					
				}
			//}
		}
	}

	
	var stop = false;
	if(stop_mark != 0.f){ //has not been terminated
		let hit = Trace(ray); //Evaluate the ray
		if(hit.t != missT){//If not miss
			//Get material
			let materialIndex = (i32(hit.material.w)) % i32(arrayLength(&materials));
			let material = materials[materialIndex].colour;
			
			if(material.w == 0.f){//If the hit sphere is a light
				color = material.xyz*color;
				stop = true;
				textureStore(origin_buffer,screen_pos,vec4f(0.f));
			}else{
				
				//Generate random numbers for ray decisision, 
				let shifted_uv = (uv+1.f)/2.f;
				var r_dec = random((vec2f(config.time)*shifted_uv));
				var r_x = random((vec2f(config.uniformRandom)*shifted_uv));
				let r_y = random(vec2f(r_x)*shifted_uv);
				
				if(material.w==2.0f){ //If transparent
					let N = hit.normal;
					//1.00029f/1.5f = 0.66686
					var refDir = refract(ray.direction,N,0.66686);
					var refRay = Ray(hit.intersection +  epsilon * refDir,1, refDir,0);
					var refHit = RaySphereIntersection(refRay,i32(hit.material.z));

					let L = refDir;
					let NdotL = dot(N,-L);

					color *= material.xyz*NdotL;

					//ref
					//1.5f/1.00029f = 1.49956512611
					refDir = refract(refDir,refHit.normal,1.49956512611);
					refRay = Ray(refHit.intersection +  epsilon * refDir,1, refDir,0);
					
					
					textureStore(origin_buffer,screen_pos,vec4f(refRay.origin,1.f)); //hit.material.w atten
					textureStore(direction_buffer,screen_pos,vec4f(refRay.direction,(0.)));
				}
				else if( (r_dec)<material.w){ //If not transparent, decide if diffuse or specular ray
					// ------------------------------ DIFFUSE ------------------------------
					var L = cosineWSampleHemisphere(r_x,r_y);//uniformSampleHemisphere(r_x,r_y);

					let N = hit.normal;
					var T = vec3f(0.f);
					
					
					var U = vec3f(0.f);
					if (abs(N.z) < 0.999f) {
						U = cross(N, vec3f(0., 0., 1.));
					} else {
						U = cross(N, vec3f(0., 1., 0.));
					}
					T = normalize(cross(U,N));
					let B = normalize(cross(N, T));
					L = T * L.x + B * L.y+N * L.z;
				
					let cos_theta = max(0.0f,dot(hit.normal,L));

					let newRay = Ray(hit.intersection +  epsilon * hit.normal, reflectId,L,iteration);
					textureStore(origin_buffer,screen_pos,vec4f(newRay.origin,1.f)); 
					textureStore(direction_buffer,screen_pos,vec4f(newRay.direction,(cos_theta)));
					
					
					var pbr = getDiffuseColor(material.xyz);
					let pdf = cos_theta/PI;
					
					

					if(config.currentIteration == config.maxIterations-1){ //LAST BOUNCE IN SAMPLE
						color = vec3f(0.f);
						
					}else{
						color *= vec3f(pbr)*PI;
					}

					
				
				}else{
					// ------------------------------ SPECULAR ------------------------------
					var L = importantSampleHemisphereBeckmann(r_x,r_y,material.w);
					var V = normalize(-ray.direction);
					var N = hit.normal;
					var T = vec3f(0.f);
					
					var U = vec3f(0.f);
					if (abs(N.z) < 0.999f) {
						U = cross(N, vec3f(0., 0., 1.));
					} else {
						U = cross(N, vec3f(0., 1., 0.));
					}
					T = normalize(cross(U,N));
					let B = normalize(cross(N, T));
					L = T * L.x + B * L.y+N * L.z;
					
					L = normalize(L);
					
					var H = L;
					L = reflect(-V,H);
					//L = normalize(L);
					//let cos_theta_L = max(0.f,dot(hit.normal,L));

					
					
					//let cos_theta_H = max(dot(N, H), 0.001);
					//var L = reflect(-V,H);
					//let H = normalize(reflected + V);
					//L = normalize(L);
					//N = normalize(N);
					//V = normalize(V);
					//H = normalize(H);

					let NdotL = max(dot(N, L), 0.001);
					var NdotV = max(dot(N,V), 0.001);
					let NdotH = clamp(dot(N, H), 0.001, 1.0);
					let VdotH = clamp(dot(V, H), 0.001, 1.0);

					//COOK-TORRENCE PBR
					let Geom = GeometricAttenuation_smith(NdotH, NdotV, NdotL, VdotH);
					let Dist = BeckmannDistribution(N, H,material.w);
					
					let F = FresnelSchlick(material.xyz, V,H);
					let nominator = 4*NdotL*NdotV;
					var f_ct = (F*Geom*Dist)/nominator;
					
					var specular = ((f_ct));
					let pbr = specular;
					var pdf = (Dist * NdotH) / (4*VdotH);

					//let sin_theta_L = sqrt(1. - cos_theta_L * cos_theta_L);
					//let sin_theta_H = sqrt(1. - cos_theta_H * cos_theta_H);
					
					let newRay = Ray(hit.intersection +  epsilon * hit.normal, reflectId,normalize(L),iteration);
					textureStore(origin_buffer,screen_pos,vec4f(newRay.origin,1.f)); //hit.material.w atten
					textureStore(direction_buffer,screen_pos,vec4f(newRay.direction,(NdotL)));
					
					// ----------------------------------------------------------------------
					if(config.currentIteration == config.maxIterations-1){
						color = vec3f(0.f);
						
					}else{
						color *= pbr*NdotL* (1./pdf);
					}
					
				}
			}
		}else{
			//IF miss, sample cube map
			if(iteration==0){
				color = textureSampleLevel(cubemapTexture, cubemapSampler, ray.direction,0).rgb;
				textureStore(origin_buffer,screen_pos,vec4f(ray.origin,0.f));
				textureStore(direction_buffer,screen_pos,vec4f(ray.direction,0.f));
			}else{
				color *= ((textureSampleLevel(cubemapTexture, cubemapSampler, ray.direction,0).rgb));
				//color *= vec3f(0.f);
				textureStore(origin_buffer,screen_pos,vec4f(ray.origin,0.f));
				textureStore(direction_buffer,screen_pos,vec4f(ray.direction,0.f));
			}
			stop = true;
		}

	}


	textureStore(color_buffer,screen_pos,vec4f(color,1.));
	
}

fn getRayDataFromTexture(screen_pos : vec2<i32>) -> RayDataTex{
	var rayData:RayDataTex;
	let origin = textureLoad(origin_buffer_read,screen_pos,0);
	let direction = textureLoad(direction_buffer_read,screen_pos,0);
	rayData.direction = direction.xyz;
	rayData.origin = origin.xyz;
	rayData.stop = origin.w;
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

fn generatePrimaryRay(uv : vec2<f32>) -> Ray{
	let P = (uCamera.inversePV * vec4f(uv, -1.f, 1.0)).xyz;
	var ray:Ray;
	ray.direction = normalize(P - uCamera.position.xyz);
	ray.origin = uCamera.position.xyz;

	return ray;

}
fn RaySphereIntersection(ray : Ray, sphereIndex : i32) -> Hit{//sphere : Sphere) -> Hit{
	let sphere = spheres[sphereIndex];
	let oc = ray.origin - sphere.origin.xyz;
	let b = dot(ray.direction, oc);
	let c = dot(oc, oc) - (1.f*1.f);

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
		return Hit(intersection+sphere.origin.xyz, t, n,vec4f(0.f,0.f,f32(sphereIndex),sphere.materialIndex));
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
	let c = dot(oc, oc) - (1.f*1.f);

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
		return Hit(intersection+sphere.origin.xyz, t, n,vec4f(sphere.materialIndex));
	}else{
		return miss;
	}

}


fn Trace(ray : Ray) -> Hit{

	var color = vec3f(0.0,0.0,0.0);
	let epsilon = 0.001f;

	var hit = Evaluate(ray);
	
    return hit;
}

fn Evaluate(ray : Ray) -> Hit{

	var closest_hit = miss;
	var bvhHit = traverseAccStructure(ray);
	if(bvhHit.t <= closest_hit.t){
		return bvhHit;
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
fn traverseAccStructure(ray :Ray) -> Hit{

	//return test_renderAllSpheres(ray);
	return traverseBVH(ray);
	
	//return miss;

}


fn RayBoxIntersection(ray : Ray, minP : vec3f, maxP : vec3f) -> vec2f{ 

	let eps = 0.00001;
    
    
    let ray_min_tmp = (minP - ray.origin) / (ray.direction + vec3f(eps, eps, eps));
    let ray_max_tmp = (maxP - ray.origin) / (ray.direction + vec3f(eps, eps, eps));
    
    let ray_min = min(ray_min_tmp, ray_max_tmp);
    let ray_max = max(ray_min_tmp, ray_max_tmp);

    let tmin = max(max(ray_min.x, ray_min.y), ray_min.z);
    let tmax = min(min(ray_max.x, ray_max.y), ray_max.z);

    if (tmin > tmax) { return vec2f(missT, missT); } 
    if (tmax < 0) { return vec2f(missT, missT); }    

    return vec2f(tmin, tmax);
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
	return ( max(0.0, exp((NdotH*NdotH - 1.0) / (m*m * NdotH*NdotH))) /max(0.0001, (PI*m*m * NdotH*NdotH*NdotH*NdotH)) );
}



//fn GeometricAttenuation_smith(N:vec3f,V: vec3f,L:vec3f, H:vec3f) -> f32
fn GeometricAttenuation_smith(NdotH:f32,NdotV: f32,NdotL:f32, VdotH:f32) -> f32
{
	/*let NdotH = max(0.0, dot(N, H));
	let NdotV = max(0.0, dot(N, V));
	let VdotH = max(0.0, dot(V, H));
	let NdotL = max(0.0, dot(N, L));*/

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
//Taken from (https://github.com/jbikker/bvh_article/blob/main/cl/tools.cl) which is under "Unlicense" license.
fn traverseBVH(ray:Ray) -> Hit{
	var nodeId = 0;
	var node: BVHNode = bvh[nodeId];
    var stack: array<BVHNode, 32>;
    var stackLocation: u32 = 0;
	var closest_hit = miss;
	let main = RayBoxIntersection(ray,node.minAABB,node.maxAABB);

	 while (true) {

        
		
        if (node.leftChild <= 0) {
			
		
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
		
		
		
    }


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



fn RayPlaneIntersection(ray : Ray, normal:vec3f, point:vec3f, material:vec4f) -> Hit {
    let nd = dot(normal, ray.direction);
    
    if (abs(nd) < 1e-6) {
		var h = miss;
		h.material = vec4f(0.f,0.f,1.f,1.f);
		return h;
    }

    let sp = point - ray.origin;
    let t = dot(sp, normal) / nd;

    if (t < 0.0) { 
        var h = miss;
		h.material = vec4f(0.f,0.f,1.f,1.f);
		return h;
    }

    let intersection = ray.origin + t * ray.direction;

    // circle
    //if(length(intersection) > 8){
    if (intersection.x > 8.0 || intersection.x < -8.0 || intersection.y > 8.0 || intersection.y < -8.0 || intersection.z > 8.0 || intersection.z < -8.0) {
	    var h = miss;
		//h.material = vec4f(1.f,0.f,0.f,1.f);
		return h;
    }
	
    return Hit(intersection, t, normal, material);
}

fn getDiffuseColor(albedo : vec3f) -> vec3f{
	return albedo/PI;
}


fn FresnelSchlick(color :vec3f, V:vec3f,H:vec3f) -> vec3f
{
	let VdotH = clamp(dot(V, H), 0.001, 1.0);

    var F0_vec3 = color;
	let res = F0_vec3 + (1.0f-F0_vec3) * (pow(clamp(1.0f - VdotH,0.f,1.f),5));
	return res;

}

