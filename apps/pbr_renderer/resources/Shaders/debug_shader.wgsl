/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: debug_screen.wgsl
 *
 *  Description: 
 *  This shader handles rendering of Debug View. It takes ray origins from debugLineDataArray and renders cylinders representing rays.
 *  
 * -----------------------------------------------------------------------------
 */

const PI = 3.1415926535f;

@group(0) @binding(0) var<uniform> uCamera: Camera;
@group(0) @binding(1) var<storage, read_write> debugLineDataArray: array<DebugData>;
@group(0) @binding(2) var color_buffer: texture_storage_2d<rgba8unorm,write>;
@group(0) @binding(3) var<uniform> config: Config;
@group(0) @binding(4) var<uniform> debugConfig: DebugConfig;

@group(1) @binding(0) var<storage,read> bvh: array<BVHNode>;
@group(1) @binding(1) var<storage,read> spheres: array<Sphere>;
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
	origin: vec3f,
	materialIndex: f32
};

struct Ray {
    origin:vec3f,     // The ray origin.
	direction:vec3f,  // The ray direction.
};

// The definition of an intersection.
struct Hit {
    intersection:vec3f,      // The intersection point.
	t:f32,				  // The distance between the ray origin and the intersection points along the ray. 
    normal:vec3f,             // The surface normal at the interesection point.
	material:vec4f			  // The material of the object at the intersection point.
};


const ray_colors = array<vec3f,10>( 
    vec3f(1.0, 0.0, 0.0),    // Red for bounce 1
    vec3f(1.0, 0.5, 0.0),  // Orange for bounce 2
    vec3f(1.0, 1.0, 0.0),    // Yellow for bounce 3
    vec3f(0.0, 0.5, 0.0),  // Green for bounce 4
    vec3f(0.0, 1.0, 1.0),    // Cyan for bounce 5
    vec3f(0.0, 0.0, 1.0),    // Blue for bounce 6
    vec3f(0.5, 0.0, 0.5),// Purple for bounce 7
    vec3f(1.0, 0.0, 1.0),    // Magenta for bounce 8
    vec3f(1.0, 0.753, 0.796),// Pink for bounce 9
    vec3f(1.0, 1.0, 1.0)     // White for bounce 10
);


//const emptySphere = Sphere(vec3f(0.f),-1.f,vec4f(0.f)) ; 
const miss = Hit(vec3f(0.0f), 1e20, vec3f(0.0f), vec4f(0.f,0.f,0.f,0.f));//vec4f(0.f));
const missT = 1e20;
const numberOfPros = 8*4;

const roughness_global = 0.1; 

const primaryId = 1;
const reflectId = 1;
const refractId = 1;



@compute @workgroup_size(8,4,1)
fn main(@builtin(workgroup_id) WorkgroupID:vec3<u32>,@builtin(local_invocation_id) LocalInvocationID:vec3<u32>){
	//let maxIteration = 30;
	let firstFrame = true;
	let epsilon = 0.001f;
	//var color = vec3f(0.f);
	var screen_pos : vec2<i32> = vec2<i32>(i32(8*WorkgroupID.x + LocalInvocationID.x),i32(4*WorkgroupID.y + LocalInvocationID.y));
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
	
	ray = generatePrimaryRay(uv);
	let color = Trace(ray); 
	textureStore(color_buffer,screen_pos,vec4f(color,0.f));
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
		return Hit(intersection+sphere.origin.xyz, t, n,vec4f(0.926f, 0.721f, 0.504f,1.f));
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


fn Trace(ray : Ray) -> vec3f{
    var light : Light;
	light.position = vec3f(0.0f,120.5f,10.0f);
	light.diffuse = vec3f(0.8f);
	



	var color = vec3f(0.0,0.0,0.0);
    var attenuation = vec3f(1.0);

	let epsilon = 0.001f;

	

	var tmpRay = ray;

	
	for(var i = 0;i<1;i++){

		let hit = Evaluate(tmpRay);
		var L = normalize(light.position -  hit.intersection);

		if (!isHitMiss(hit)) {

			let N = hit.normal;
			var V = normalize(-tmpRay.direction);
			let H = normalize(L + V);
			let NdotL = max(dot(N, L), 0.0);

			let ambient = 0.1*hit.material.xyz;
			color += 10.*ambient;
		
			
			var NdotV = dot(H,V);
			

			
			let F = schlickFresnel(NdotV,hit.material.xyz);
			//dif
			color += NdotL * light.diffuse * hit.material.xyz * attenuation;
			//specular
			let Geom = GeometricAttenuation_smith(N, V, L, H);
			let Dist = BeckmannDistribution(N, H, 0.2);
			NdotV = dot(N,V);
			color += Dist * Geom * F / 4.0 / NdotV;


			
			
			

			let reflected = reflect(tmpRay.direction,hit.normal);

			let newOr = hit.intersection +  epsilon * N;
			let newRay = Ray(newOr, reflected);
			tmpRay = newRay;
		}

	}
	

    return color;
}

fn Evaluate(ray : Ray) -> Hit{

	var closest_hit = miss;
	if(i32(debugConfig.visOptions.y)==1){
		for(var i = 0;i<config.maxSamples;i++){
			let index = i;
			if(i32(debugConfig.visOptions.w)==1){
				let cylinderHit = rayCylinderIntersection(ray.origin, ray.direction, debugLineDataArray[index].values[i32(debugConfig.visOptions.z)-1].xyz,debugLineDataArray[index].values[i32(debugConfig.visOptions.z)].xyz,0.06f,ray_colors[i32(debugConfig.visOptions.z)]);
				if(cylinderHit.t < closest_hit.t){
					closest_hit = cylinderHit;
				}
			}else{
				//config.debugRayIndex;
				let count = i32(debugLineDataArray[index].data.x);
				for(var j = 0;j<count;j++){
					
					let cylinderHit = rayCylinderIntersection(ray.origin, ray.direction, debugLineDataArray[index].values[j].xyz,debugLineDataArray[index].values[j+1].xyz,0.06f,ray_colors[j]);
					if(cylinderHit.t < closest_hit.t && j < i32(debugConfig.visOptions.z)){
						closest_hit = cylinderHit;
					}
				}
			}
		}
	}else{
		let index = i32(debugConfig.visOptions.x);
		if(i32(debugConfig.visOptions.w)==1){
			
			let cylinderHit = rayCylinderIntersection(ray.origin, ray.direction, debugLineDataArray[index].values[i32(debugConfig.visOptions.z)-1].xyz,debugLineDataArray[index].values[i32(debugConfig.visOptions.z)].xyz,0.06f,ray_colors[i32(debugConfig.visOptions.z)]);
			if(cylinderHit.t < closest_hit.t){
				closest_hit = cylinderHit;
			}
		}else{
			
			let count = i32(debugLineDataArray[index].data.x);
			for(var j = 0;j<count;j++){
				let cylinderHit = rayCylinderIntersection(ray.origin, ray.direction, debugLineDataArray[index].values[j].xyz,debugLineDataArray[index].values[j+1].xyz,0.06f,ray_colors[j]);
				if(cylinderHit.t < closest_hit.t && j < i32(debugConfig.visOptions.z)){
					closest_hit = cylinderHit;
				}
			}
		}
	}
	
	var kdResultHit = traverseAccStructure(ray);
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
fn traverseAccStructure(ray :Ray) -> Hit{
	//return miss;
	//return test_renderAllSpheres(ray);
	return traverseBVH(ray);
	//return traverseBVH_packet(ray,LocalInvocationID);
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

fn schlickFresnel(vDotH : f32,color:vec3f) -> vec3f{
	var F0 = vec3f(0.04);


	let res = F0 + (1.0f-F0) * (pow(clamp(1.0f - vDotH,0.f,1.f),5));
	return res;
}



fn BeckmannDistribution(N:vec3f, H:vec3f, m:f32) -> f32
{
	let NdotH = max(0.0, dot(N, H));
	return ( max(0.0, exp((NdotH*NdotH - 1.0) / (m*m * NdotH*NdotH))) /max(0.0001, (m*m * NdotH*NdotH*NdotH*NdotH)) );
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
//Taken from (https://github.com/jbikker/bvh_article/blob/main/cl/tools.cl) which is under "Unlicense" license.
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










fn FresnelSchlick(color :vec3f, V:vec3f,H:vec3f) -> vec3f
{
	let VdotH = clamp(dot(V, H), 0.001, 1.0);

	let ior = 2.0;
	var F0 = abs((1.0 - ior) / (1.0 + ior));
    F0 = F0 * F0;
	let metallic = 1.f;
    var F0_vec3 = vec3f(0.95,0.64,0.54);//(1.0 - metallic) * F0 + metallic * color.xyz;
	let res = F0_vec3 + (1.0f-F0_vec3) * (pow(clamp(1.0f - VdotH,0.f,1.f),5));
	return res;

	//return f0 + (1.0 - f0) * pow(1.0 - VdotH, 5.0);
}

fn rayCylinderIntersection(rayOrigin: vec3<f32>, rayDir: vec3<f32>, 
                           cylStart: vec3<f32>, cylEnd: vec3<f32>, cylRadius: f32, color: vec3f) -> Hit {
    
    
	let h = cylEnd - cylStart;
	let h_normal = normalize(h);

	let w = rayOrigin - cylStart;

	let VdotH = dot(rayDir,h_normal);
	let WdotH = dot(w,h_normal);

	let a = dot(rayDir,rayDir) - (VdotH*VdotH);
	let b = 2.0 * ((dot(rayDir,w) - VdotH * WdotH));
	let c = dot(w,w) - WdotH * WdotH - cylRadius*cylRadius;

	let disc = b * b - 4.0 * a * c;
	if(disc<0.f){
		return miss;
	} 

	let sqrtDisc = sqrt(disc);
    let t_min = (-b - sqrtDisc) / (2.0 * a);
    let t_max = (-b + sqrtDisc) / (2.0 * a);

	var t = 0.f;
	if(t_min<t_max){
		t = t_min;
	}else{
		t = t_max;
	}
	let P1 = rayOrigin + t * rayDir;
    let P2 = rayOrigin + t_max * rayDir;

	let P1_proj = dot(P1 - cylStart, h_normal);
	if (0. <= P1_proj && P1_proj <= length(h) && t>0.f){
		let intersection = rayOrigin + t * rayDir;
		var n = normalize(intersection);
		if(dot(rayDir, n) > 0) {
			n = -n; 
		}
		return Hit(intersection, t, n,vec4f(color,1.));
	}

	return miss;
}