const vec2BlueNoiseInDisk = array<vec2f,64>(
    vec2f(0.478712,0.875764),
    vec2f(-0.337956,-0.793959),
    vec2f(-0.955259,-0.028164),
    vec2f(0.864527,0.325689),
    vec2f(0.209342,-0.395657),
    vec2f(-0.106779,0.672585),
    vec2f(0.156213,0.235113),
    vec2f(-0.413644,-0.082856),
    vec2f(-0.415667,0.323909),
    vec2f(0.141896,-0.939980),
    vec2f(0.954932,-0.182516),
    vec2f(-0.766184,0.410799),
    vec2f(-0.434912,-0.458845),
    vec2f(0.415242,-0.078724),
    vec2f(0.728335,-0.491777),
    vec2f(-0.058086,-0.066401),
    vec2f(0.202990,0.686837),
    vec2f(-0.808362,-0.556402),
    vec2f(0.507386,-0.640839),
    vec2f(-0.723494,-0.229240),
    vec2f(0.489740,0.317826),
    vec2f(-0.622663,0.765301),
    vec2f(-0.010640,0.929347),
    vec2f(0.663146,0.647618),
    vec2f(-0.096674,-0.413835),
    vec2f(0.525945,-0.321063),
    vec2f(-0.122533,0.366019),
    vec2f(0.195235,-0.687983),
    vec2f(-0.563203,0.098748),
    vec2f(0.418563,0.561335),
    vec2f(-0.378595,0.800367),
    vec2f(0.826922,0.001024),
    vec2f(-0.085372,-0.766651),
    vec2f(-0.921920,0.183673),
    vec2f(-0.590008,-0.721799),
    vec2f(0.167751,-0.164393),
    vec2f(0.032961,-0.562530),
    vec2f(0.632900,-0.107059),
    vec2f(-0.464080,0.569669),
    vec2f(-0.173676,-0.958758),
    vec2f(-0.242648,-0.234303),
    vec2f(-0.275362,0.157163),
    vec2f(0.382295,-0.795131),
    vec2f(0.562955,0.115562),
    vec2f(0.190586,0.470121),
    vec2f(0.770764,-0.297576),
    vec2f(0.237281,0.931050),
    vec2f(-0.666642,-0.455871),
    vec2f(-0.905649,-0.298379),
    vec2f(0.339520,0.157829),
    vec2f(0.701438,-0.704100),
    vec2f(-0.062758,0.160346),
    vec2f(-0.220674,0.957141),
    vec2f(0.642692,0.432706),
    vec2f(-0.773390,-0.015272),
    vec2f(-0.671467,0.246880),
    vec2f(0.158051,0.062859),
    vec2f(0.806009,0.527232),
    vec2f(-0.057620,-0.247071),
    vec2f(0.333436,-0.516710),
    vec2f(-0.550658,-0.315773),
    vec2f(-0.652078,0.589846),
    vec2f(0.008818,0.530556),
    vec2f(-0.210004,0.519896) 
);

const PI = 3.1415926535f;

//layout (local_size_x = 16,local_size_y = 16) in;

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
    origin:vec3f,     // The ray origin.
    rayID: i32,
	direction:vec3f,  // The ray direction.
	iteration: i32
};

// The definition of an intersection.
struct Hit {
    intersection:vec3f,      // The intersection point.
	t:f32,				  // The distance between the ray origin and the intersection points along the ray. 
    normal:vec3f,             // The surface normal at the interesection point.
	material:vec4f			  // The material of the object at the intersection point.
};

struct RayDataTex{
	origin:vec3f,     // The ray origin.
    rayID: i32,
	direction:vec3f,  // The ray direction.
	iteration: i32,
	attenuation:f32,
	prob:f32
}
//const emptySphere = Sphere(vec3f(0.f),-1.f,vec4f(0.f)) ; 
const miss = Hit(vec3f(0.0f), 1e20, vec3f(0.0f), vec4f(0.f));
const missT = 1e20;
const numberOfPros = 8*4;

const primaryId = 1;
const reflectId = 1;
const refractId = 1;

@compute @workgroup_size(8,4,1)
//fn main(@builtin(global_invocation_id) GlobalInvocationID:vec3<u32>){
fn main(@builtin(workgroup_id) WorkgroupID:vec3<u32>,@builtin(local_invocation_id) LocalInvocationID:vec3<u32>){
	let maxIteration = 10;
	let firstFrame = true;
	let epsilon = 0.001f;
	//var color = vec3f(0.f);
	var screen_pos : vec2<i32> = vec2<i32>(i32(8*WorkgroupID.x + LocalInvocationID.x),i32(4*WorkgroupID.y + LocalInvocationID.y));
	var ray : Ray;
	var iteration  = i32(uCamera.position.w);

	let screen_size: vec2<u32> = textureDimensions(color_buffer);
	let horizontal_coefficient: f32 = (f32(screen_pos.x) - f32(screen_size.x)/2) / f32(screen_size.x);
    let vertical_coefficient: f32 = (f32(screen_pos.y) - f32(screen_size.y)/2) / f32(screen_size.x);

    let forwards: vec3<f32> = vec3<f32>(0.0,0.0,1.0);
    let right: vec3<f32> = vec3<f32>(1.0,0.0,0.0);
    let up: vec3<f32> = vec3<f32>(0.0,-1.0,0.0);

	var uv = vec2f(horizontal_coefficient,vertical_coefficient);
	
	
	//textureStore(color_buffer,screen_pos,vec4f(random(uv),0.,0.,1.));
	//return;
	if(iteration<config.maxIterations /*&& config.currentSample<config.maxSamples*/){
			textureStore(color_buffer,screen_pos,vec4f(0.,0.,0.,1.));
			textureStore(origin_buffer,screen_pos,vec4f(0.,0.,0.,1.));
			textureStore(direction_buffer,screen_pos,vec4f(0.,0.,0.,1.));
	}else if(iteration==config.maxIterations-1 && config.currentSample==config.maxSamples-1){
		let prev_color = textureLoad(color_buffer_read,screen_pos,0).xyz;
		textureStore(color_buffer,screen_pos,vec4f(prev_color,1.));
		return;
	}else if(iteration>config.maxIterations+2){
		//return;
	}
	var attenuation = 0.f;
	var stop_mark = 1.f;
	var prev_cos_theta = 1.f;
	//iteration=0;
	if(iteration == 0){ //first iteration
		//clear
		//textureStore(color_buffer,screen_pos,vec4f(0.,0.,0.,1.));
		ray = generatePrimaryRay(screen_pos);
	}else{
		//pop from front
		
		let rayData = getRayDataFromTexture(screen_pos);
		ray.origin = rayData.origin;
		ray.direction = rayData.direction;
		stop_mark = rayData.attenuation;
		prev_cos_theta = rayData.prob;
	}
	
	/*if(iteration<=1){
			textureStore(color_buffer,screen_pos,vec4f(0.,0.,0.,1.));
			textureStore(origin_buffer,screen_pos,vec4f(0.,0.,0.,1.));
			textureStore(direction_buffer,screen_pos,vec4f(0.,0.,0.,1.));
	}else{
		return;
	}*///config.currentIteration == 0
	var color = select(textureLoad(color_buffer_read,screen_pos,0).xyz,vec3f(0.f,0.f,0.f),config.currentIteration == 0);
	//var color = textureLoad(color_buffer_read,screen_pos,0).xyz;
	//if(config.currentIteration==0){stop_mark=1.f; prev_cos_theta=1.f;}
	var stop = false;
	if(stop_mark != 0.f){ //not blank ray
		let hit = Trace(ray,attenuation,LocalInvocationID);
		if(hit.t != missT){
			
			if(hit.material.w == 0.f){//is light
				if(config.currentIteration==0){
					color = hit.material.xyz;
				}else{
					color *= (hit.material.xyz*prev_cos_theta);
				}
				stop = true;
				textureStore(origin_buffer,screen_pos,vec4f(0.f));
			}else{
				let NdotV = max(0.f,dot(ray.direction,hit.normal));
				//let ks = schlickFresnel(NdotV,hit.material.xyz);
				//decide if diffuse/specular/transmittion
				//if previous generated was diffuse/specular		
				//if(config.uniformRandom>0.5f){
				if(stop_mark == 2.f){//diffuse was generated
					var pbr = getDiffuseColor(hit.material.xyz);
					
					if(config.currentIteration>0){
						color *= ((pbr)*prev_cos_theta);
					}else if(config.currentIteration==0){
						color = hit.material.xyz;//((pbr)*prev_cos_theta);
					}
				}else if(stop_mark == 3.f){
					var pbr = getSpecularColor(hit.material.xyz, hit.normal,ray.direction,ray.direction,prev_cos_theta);
					if(config.currentIteration>0){
						color *= ((pbr)*prev_cos_theta);
					}else if(config.currentIteration==0){
						color = hit.material.xyz;//((pbr)*prev_cos_theta);
					}
				}else{
					color = hit.material.xyz;
				}
				//generating
				if(true){
					let r_x = random((vec2(config.uniformRandom)*uv));
					let r_y = random((vec2(config.uniformRandom*2.)*uv));
					//let r_x = random((1+config.uniformRandom)*uv);
					//let r_y = random((1+config.uniformRandom*2.)*uv);
					if(r_x<0.){
						textureStore(color_buffer,screen_pos,vec4f(0.,1.,0.,1.));
						return;
					}
					if(r_x>1.){
						textureStore(color_buffer,screen_pos,vec4f(0.,0.,1.,1.));
						return;
					}
					var reflected = uniformSampleHemisphere(r_x,r_y);//uniformSampleHemisphere(r_x,r_y);

					let N = hit.normal;
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
					reflected = T * reflected.x + B * reflected.y+N * reflected.z;
					/*reflected.x = T.x * reflected.x + B.x * reflected.y+ N.x * reflected.z;
					reflected.y = T.y * reflected.x + B.y * reflected.y+ N.y * reflected.z;
					reflected.z = T.z * reflected.x + B.z * reflected.y+ N.z * reflected.z;*/
					reflected = normalize(reflected);
					//reflected = normalize(reflect(ray.direction,hit.normal));

					//reflected = reflected + N;
					let cos_theta = max(0.f,dot(hit.normal,reflected));

					let newRay = Ray(hit.intersection +  epsilon * hit.normal, reflectId,reflected,iteration);
					textureStore(origin_buffer,screen_pos,vec4f(newRay.origin,2.f)); //hit.material.w 1.f diffuse/ 2.f specular
					textureStore(direction_buffer,screen_pos,vec4f(newRay.direction,(cos_theta)));

				}else{
					let r_x = random((vec2(config.uniformRandom)*uv));
					let r_y = random((vec2(config.uniformRandom*2.)*uv));

					var reflected = importantSampleHemisphereIBL(r_x,r_y,0.8f);//uniformSampleHemisphere(r_x,r_y);

					let N = hit.normal;
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
					reflected = T * reflected.x + B * reflected.y+N * reflected.z;
					/*reflected.x = T.x * reflected.x + B.x * reflected.y+ N.x * reflected.z;
					reflected.y = T.y * reflected.x + B.y * reflected.y+ N.y * reflected.z;
					reflected.z = T.z * reflected.x + B.z * reflected.y+ N.z * reflected.z;*/
					reflected = normalize(reflected);
					//reflected = normalize(reflect(ray.direction,hit.normal));
					//reflected = reflected + N;

					let cos_theta = max(0.f,dot(hit.normal,reflected));

					let newRay = Ray(hit.intersection +  epsilon * hit.normal, reflectId,reflected,iteration);
					textureStore(origin_buffer,screen_pos,vec4f(newRay.origin,3.f)); //hit.material.w atten
					textureStore(direction_buffer,screen_pos,vec4f(newRay.direction,(cos_theta)));

				}
			}
		}else{
			if(iteration==0){
				color = textureSampleLevel(cubemapTexture, cubemapSampler, ray.direction,0).rgb;
				textureStore(origin_buffer,screen_pos,vec4f(0.f));
			}else{
				color *= (textureSampleLevel(cubemapTexture, cubemapSampler, ray.direction,0).rgb * prev_cos_theta)/PI;
				textureStore(origin_buffer,screen_pos,vec4f(0.f));
			}
			stop = true;
		}

	}
	//textureStore(color_buffer,screen_pos,vec4f(color,1.));
	var count = 1.;
	if(stop || config.maxIterations-1 == config.currentIteration){
		count = f32(config.currentIteration+1);
	}
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
	let horizontal_coefficient: f32 = (f32(screen_pos.x) - f32(screen_size.x)/2) / f32(screen_size.x);
	let vertical_coefficient: f32 = (f32(screen_pos.y) - f32(screen_size.y)/2) / f32(screen_size.x);

	let forwards: vec3<f32> = vec3<f32>(0.0,0.0,1.0);
	let right: vec3<f32> = vec3<f32>(1.0,0.0,0.0);
	let up: vec3<f32> = vec3<f32>(0.0,-1.0,0.0);

	var uv = vec2f(horizontal_coefficient,vertical_coefficient);
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

	/*let co: vec3<f32> = ray.origin - sphere.origin;
    let a: f32 = dot(ray.direction, ray.direction);
    let b: f32 = 2.0 * dot(ray.direction, co);
    let c: f32 = dot(co, co) - sphere.radius * sphere.radius;
    let discriminant: f32 = b * b - 4.0 * a * c;

    if (discriminant > 0.0) {

        let t: f32 = (-b - sqrt(discriminant)) / (2 * a);

        //if (t > tMin && t < tMax) {
            let intersection = co + t * ray.direction;
			return Hit(intersection+sphere.origin, t, vec3(1.),sphere.color);
        //}
    }
    return miss;*/
}



fn Trace(ray : Ray,prev_F:f32,LocalInvocationID:vec3<u32>) -> Hit{

	var color = vec3f(0.0,0.0,0.0);
	let epsilon = 0.001f;

	var hit = Evaluate(ray,LocalInvocationID);
	
    return hit;
}

fn Evaluate(ray : Ray,LocalInvocationID:vec3<u32>) -> Hit{

	var closest_hit = RayPlaneIntersection(ray, vec3f(0.0f, -1.f, 1.f), vec3f(0.0f,5.0f,0.0f),vec4f(1.0f,0.0f,0.0f,0.f));
	/*var right_plane_hit = RayPlaneIntersection(ray, vec3f(1.0f, 0.f, 0.f), vec3f(-6.0f,0.0f,0.0f),vec4f(0.f,1.f,0.f,0.f));
	if(right_plane_hit.t<closest_hit.t){
		closest_hit = right_plane_hit;
	}
	var bottom_plane_hit = RayPlaneIntersection(ray, vec3f(0.0f, -1.f, 1.f), vec3f(0.0f,5.0f,0.0f),vec4f(0.2f,0.5f,1.0f,1.f));
	if(bottom_plane_hit.t<closest_hit.t){
		closest_hit = bottom_plane_hit;
	}*/
	var kdResultHit = traverseKdTree(ray,LocalInvocationID);
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
fn traverseKdTree(ray :Ray,LocalInvocationID:vec3<u32>) -> Hit{
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
	//TODO: if is metal

	/*if(metallic){
		F0 = color;
	}*/

	let res = F0 + (1.0f-F0) * (pow(clamp(1.0f - vDotH,0.f,1.f),5));
	return res;
}

fn schlickFresnel_refract(n1 : f32, n2 : f32, vDotH : f32) -> f32{

	var r0 = (n1-n2) / (n1+n2);
	r0 *= r0;
	var cosX = vDotH;
	if (n1 > n2)
	{
		let n = n1/n2;
		let sinT2 = n*n*(1.0-cosX*cosX);
		// Total internal reflection
		if (sinT2 > 1.0){ return 1.0;}
		cosX = sqrt(1.0-sinT2);
	}
	let x = 1.0-cosX;
	var ret = r0+(1.0-r0)*x*x*x*x*x;

	// adjust reflect multiplier for object reflectivity
	ret = (0.01 + (1.0- 0.01) * ret);
	return ret;
}

fn BeckmannDistribution(N:vec3f, H:vec3f, m:f32) -> f32
{
	let NdotH = max(0.0, dot(N, H));
	return ( max(0.0, exp((NdotH*NdotH - 1.0) / (m*m * NdotH*NdotH))) /max(0.0001, (m*m * NdotH*NdotH*NdotH*NdotH)) );
}

fn GeometricAttenuation(N:vec3f,V: vec3f,L:vec3f, H:vec3f) -> f32
{
	let NdotH = max(0.0, dot(N, H));
	let NdotV = max(0.0, dot(N, V));
	let VdotH = max(0.0, dot(V, H));
	let NdotL = max(0.0, dot(N, L));
	return min(1.0, min(2.0 * NdotH * NdotV / VdotH, 2.0 * NdotH * NdotL / VdotH));
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
			if(closest_hit.t<missT){
				return closest_hit;
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


var<workgroup> m : array<i32,32>;
fn P_SUM(value:i32,LocalInvocationID:vec3<u32>) -> i32{
	let pID = LocalInvocationID.x*8 + LocalInvocationID.y;
	m[pID] = value;
	for(var i = 0;i<5;i++){
		let pw = i32(pow(f32(i),f32(2)));
		let a1 = (pw*2) * i32(pID);
		let a2 = a1 + pw;
		if(a2 < 32){
			m[a1] = (m[a1]+m[a2]);
		}
	}
	return m[0];
}



var<workgroup> leftN : BVHNode;
var<workgroup> rightN : BVHNode;
var<workgroup> stack : array<BVHNode,8>;
var<workgroup> stackLocation : u32;

var<workgroup> condition : bool;
var<workgroup> foundLeaf : bool;
var<workgroup> isEmpty : bool;
var<workgroup> mainLoop : bool;
var<workgroup> atleastOneTraversing : bool;
var<workgroup> root : BVHNode;
var<workgroup> node : BVHNode;

fn traverseBVH_packet(ray:Ray,LocalInvocationID:vec3<u32>) -> Hit{
	let pID = LocalInvocationID.x*8 + LocalInvocationID.y;
	var nodeId = 0;


	if(pID==0){
		root = bvh[nodeId];
		node = root;
		isEmpty = true;
		mainLoop = true;
		foundLeaf = false;
		condition = false;
	}
	workgroupBarrier();
	//var node = bvh[nodeId];
	
    //var stackLocation: u32 = 0;
	var closest_hit = miss;
	var j = 0;
	//let main = RayBoxIntersection(ray,node.minAABB,node.maxAABB);
	
	while (workgroupUniformLoad(&mainLoop)) {

		//j++;
		if(pID == 0){
			if (node.leftChild <= 0) {
				foundLeaf = true;//break;
			}else{
				foundLeaf = false;
			}
		}
		//workgroupBarrier();
		if(workgroupUniformLoad(&foundLeaf)){
			//closest_hit.t = 10.f;
			//closest_hit.material = vec4f(0.f,1.f,0.f,1.f);
			/*var ff = RayBoxIntersection(ray,node.minAABB,node.maxAABB);
			if(ff.x < closest_hit.t){
				closest_hit.t = ff.x;
				closest_hit.material = vec4f(1.,0.,0.,1.f);
			}*/
			for (var i = 0; i < node.numberOfSpheres; i++) {
				let sphereId = i+(-1*node.leftChild);
                var hit = RaySphereIntersection(ray, sphereId);
				//hit.t = -10. * spheres[sphereId].radius;
                if(hit.t < closest_hit.t){
					closest_hit = hit;
				}
            }
			if(pID == 0){
				if (stackLocation == 0) {
					mainLoop=false;//break;
				}else{
					stackLocation -= 1;
					node = stack[stackLocation];
				}
				
			}
			foundLeaf = false;
			//workgroupBarrier();
		}else{
			if(pID<1){
				leftN = bvh[node.leftChild];
				rightN = bvh[node.leftChild+1];
			}

			//workgroupBarrier();
			//var leftN = bvh[node.leftChild];
			//var rightN = bvh[node.leftChild+1];
			//var leftChild = bvh[node.leftChild];
			//var rightChild = bvh[node.leftChild+1];

			var distance1 = RayBoxIntersection(ray,leftN.minAABB,leftN.maxAABB);//hit_aabb(ray, child1);
			var distance2 = RayBoxIntersection(ray,rightN.minAABB,rightN.maxAABB);

			let b1 = i32((distance1.x<distance1.y) && (distance1.x < closest_hit.t) &&  distance1.y >= 0.);
			let b2 = i32((distance2.x<distance2.y) && (distance2.x < closest_hit.t) &&  distance2.y >= 0.);

			/*if(distance1.x < closest_hit.t){
				closest_hit.t = distance1.x;
				closest_hit.material = vec4f(1.,0.,0.,1.f);
			}
			mainLoop=false;*/
			if(pID<4){
				m[pID] = 0;
			}
			m[2*b1+b2] = 1;
			//workgroupBarrier();
			condition = bool(m[3]) || (bool(m[1]) || bool(m[2]));
			if(workgroupUniformLoad(&condition)){
				m[pID] = 2*i32(bool(b2) && (distance2.x<distance1.x))-1;
				//workgroupBarrier();
				for(var i = 0;i<5;i++){
					let pw = i32(pow(f32(2),f32(i)));
					let a1 = (pw*2) * i32(pID);
					let a2 = a1 + pw;
					if(a2 < 32){
						m[a1] = (m[a1]+m[a2]);
					}		
					//workgroupBarrier();
				}


				//workgroupBarrier();
				var nearChild = leftN;
				var farChild = rightN;

				if(m[0]<0){
					var tmp = nearChild;
					nearChild = farChild;
					farChild = tmp;
				}

				

				if(pID == 0){
					//push
					stack[stackLocation] = farChild;
					stackLocation++;
				}
				//if(pID == 0){node = nearChild;}
				node = nearChild;
			}else if(m[1] == 1){
				//if(pID == 0){node = leftN;}
				node = leftN;
			}else if(m[2] == 1){
				//if(pID == 0){node = rightN;}
				node = rightN;
			}else{
				if(pID == 0){
					if (stackLocation == 0) {
						mainLoop=false;//break;
					}else{
						stackLocation -= 1;
						node = stack[stackLocation];		
					}
				}
			}
			//workgroupBarrier();
		}
		
    }
	//closest_hit.t = 10.f;
	//closest_hit.material = vec4f(1.f,0.f,0.f,1.f);
	return closest_hit;
}
/*
//var<workgroup> leftN : BVHNode;
//var<workgroup> rightN : BVHNode;
var<workgroup> stack : array<BVHNode,32>;
var<workgroup> stackLocation : u32;

var<workgroup> isEmpty : bool;
var<workgroup> a : bool;
var<workgroup> root : BVHNode;

fn traverseBVH_packet(ray:Ray,LocalInvocationID:vec3<u32>) -> Hit{
	let pID = LocalInvocationID.x*8 + LocalInvocationID.y;
	var nodeId = 0;


	if(pID==0){
		root = bvh[nodeId];
		//node = root;
		isEmpty = true;
	}
	workgroupBarrier();
	var node = bvh[nodeId];
	a=true;
    //var stackLocation: u32 = 0;
	var closest_hit = miss;
	var j = 0;
	//let main = RayBoxIntersection(ray,node.minAABB,node.maxAABB);
	
	while (workgroupUniformLoad(&a)) {

		j++;
		if(node.leftChild <= 0){
			for (var i = 0; i < node.numberOfSpheres; i++) {
				let sphereId = i+(-1*node.leftChild);
                var hit = RaySphereIntersection(ray, sphereId);
				//hit.t = -10. * spheres[sphereId].radius;
                if(hit.t < closest_hit.t){
					closest_hit = hit;
				}
            }

			if (stackLocation == 0) {
				a=false;//break;
            }else{
                if(pID == 0){stackLocation -= 1;
                	node = stack[stackLocation];
				}
			}
		}else{
			/*if(pID<2){
				leftN = bvh[node.leftChild];
				rightN = bvh[node.leftChild+1];
			}*/

			
			var leftN = bvh[node.leftChild];
			var rightN = bvh[node.leftChild+1];
			//var leftChild = bvh[node.leftChild];
			//var rightChild = bvh[node.leftChild+1];

			var distance1 = RayBoxIntersection(ray,leftN.minAABB,leftN.maxAABB);//hit_aabb(ray, child1);
			var distance2 = RayBoxIntersection(ray,rightN.minAABB,rightN.maxAABB);

			let b1 = i32((distance1.x<distance1.y) && (distance1.x < closest_hit.t) &&  distance1.y >= 0.);
			let b2 = i32((distance2.x<distance2.y) && (distance2.x < closest_hit.t) &&  distance2.y >= 0.);

			if(pID<4){
				m[pID] = 0;
			}
			m[2*b1+b2] = 1;
			if((bool(m[3]) || bool(m[1])) && bool(m[2])){
				m[pID] = 2*i32(bool(b2) && (distance2.x<distance1.x))-1;
				//workgroupBarrier();
				for(var i = 0;i<6;i++){
					let pw = i32(pow(f32(i),f32(2)));
					let a1 = (pw*2) * i32(pID);
					let a2 = a1 + pw;
					if(a2 < 64){
						m[a1] = (m[a1]+m[a2]);
					}
				}
				//workgroupBarrier();
				var nearChild = leftN;
				var farChild = rightN;

				if(m[0]>=0){
					var tmp = nearChild;
					nearChild = farChild;
					farChild = tmp;
				}

				

				if(pID == 0){
					//push
					stack[stackLocation] = farChild;
					stackLocation++;
				}
				if(pID == 0){node = nearChild;}
			}else if(m[1] == 1){
				if(pID == 0){node = leftN;}
			}else if(m[2] == 1){
				if(pID == 0){node = rightN;}
			}else{
				if (stackLocation == 0) {
					//a=false;//break;
				}else{
					if(pID == 0){stackLocation -= 1;
						node = stack[stackLocation];
					}
				}
			}
		}
		
    }
	return closest_hit;
}*/

fn uniformSampleHemisphere(r1:f32, r2:f32) -> vec3f
{
    // cos(theta) = r1 = y
    // cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
	/*let z = sqrt(r1);//(PI/2.0)*r1;
	let theta = acos(z);
    let sinTheta = sin(theta);//sqrt(1 - r1 * r1);
    let phi = 2 * PI * r2;*/
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
    // cos(theta) = r1 = y
    // cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
	let z = sqrt(r1);//(PI/2.0)*r1;
	let theta = acos(z);
    let sinTheta = sin(theta);//sqrt(1 - r1 * r1);
    let phi = 2 * PI * r2;
    let x = sinTheta * cos(phi);
    let y = sinTheta * sin(phi);
    return vec3f(x, y, cos(theta));
}
fn importantSampleHemisphere(r1:f32, r2:f32) -> vec3f
{
    // cos(theta) = r1 = y
    // cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
	let z = sqrt(r1);//(PI/2.0)*r1;
	let theta = acos(z);
    let sinTheta = sin(theta);//sqrt(1 - r1 * r1);
    let phi = 2 * PI * r2;
    let x = sinTheta * cos(phi);
    let y = sinTheta * sin(phi);
    return vec3f(x, y, cos(theta));
}
fn importantSampleHemisphereIBL(r1:f32, r2:f32, m:f32) -> vec3f
{
	//let m = 0.1f;
	let m2 = m*m;

    let phi = 2 * PI * r2;
	let cosTheta = sqrt((1.-r2)/(1.+(m2-1.)*r2));
	let sinTheta = sqrt(1.-cosTheta*cosTheta);
    let x = sinTheta * cos(phi);
    let y = sinTheta * sin(phi);
    return vec3f(x, y, cosTheta);
}

fn RayPlaneIntersection(ray : Ray, normal:vec3f, point:vec3f, material:vec4f) -> Hit {
    let nd = dot(normal, ray.direction);
    
    // Check if ray is nearly parallel to the plane
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
    if(length(intersection) > 8){
        var h = miss;
		//h.material = vec4f(1.f,0.f,0.f,1.f);
		return h;
    }
	
    return Hit(intersection, t, normal, material);
}

fn getDiffuseColor(albedo : vec3f) -> vec3f{
	return albedo/PI;
}
/*fn getSpecularColor(albedo : vec3f, N:vec3f,hitIntersection:vec3f,rayDir:vec3f,reflected:vec3f,cos_theta:f32) -> vec3f{
	var light : Light;
	light.position = vec3f(0.0f,5.5f,0.0f);
	light.diffuse = vec3f(0.8f);
	//calculate diffuse + specular
	//var L = normalize(light.position -  hit.intersection);
	
	var V = normalize(-rayDir);
	let H = normalize(reflected + V);
	let NdotL = max(dot(N, reflected), 0.0);
	var NdotV = dot(N,V);

	
	let Geom = GeometricAttenuation(N, V, reflected, H);// * GeometricAttenuation(N, reflected, L, H); //todo: použít i reflected?
	let Dist = BeckmannDistribution(N, H, 0.1f);
	let F = schlickFresnel(NdotV,albedo);
	let nominator = 1.f;//4*NdotV*cos_theta; //todo: dot product s reflected?
	var f_ct = (F*Dist*Geom)/nominator;
	let ks = F;
	let kd = (1.0-ks);
	var specular = ((f_ct));/// (4*abs(dot(H,N)));
	return specular;

}*/

fn getSpecularColor(albedo : vec3f, N:vec3f,rayDir:vec3f,H:vec3f,cos_theta:f32) -> vec3f{
	var light : Light;
	light.position = vec3f(0.0f,5.5f,0.0f);
	light.diffuse = vec3f(0.8f);
	//calculate diffuse + specular
	//var L = normalize(light.position -  hit.intersection);
	
	var V = normalize(-rayDir);
	var L = reflect(-V,H);
	//let H = normalize(reflected + V);
	let NdotL = max(dot(N, L), 0.001);
	var NdotV = max(dot(N,V), 0.001);
	let NdotH = clamp(dot(N, H), 0.001, 1.0);
	let VdotH = clamp(dot(V, H), 0.001, 1.0);

	let Geom = GeometricAttenuation(N, V, L, H);// * GeometricAttenuation(N, reflected, L, H); //todo: použít i reflected?
	let Dist = BeckmannDistribution(N, H, 0.8f);
	let F = FresnelSchlick(vec3f(0.04f), V,H);//schlickFresnel(NdotV,albedo);
	let nominator = 4*NdotL*NdotV; //todo: dot product s reflected?
	var f_ct = (F*Geom*Dist)/nominator;
	let ks = F;
	let kd = (1.0-ks);
	var specular = ((f_ct));/// (4*abs(dot(H,N)));
	return specular*albedo;

}

fn FresnelSchlick(f0 :vec3f, V:vec3f,H:vec3f) -> vec3f
{
	let VdotH = clamp(dot(V, H), 0.001, 1.0);
	return f0 + (1.0 - f0) * pow(1.0 - VdotH, 5.0);
}