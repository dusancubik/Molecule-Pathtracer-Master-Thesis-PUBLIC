

//layout (local_size_x = 16,local_size_y = 16) in;
@group(0) @binding(0) var color_buffer: texture_storage_2d<rgba8unorm,write>;
@group(0) @binding(1) var<storage,read> bvh: array<BVHNode>;
@group(0) @binding(2) var<storage,read> spheres: array<Sphere>;
@group(0) @binding(3) var<uniform> uCamera: Camera;

struct Camera {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
	position: vec4f,
	inversePV: mat4x4f
};

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
	origin: vec4f
	/*origin: vec3f,
	radius: f32,
	color: vec4f*/
};

struct Ray {
    origin:vec3f,     // The ray origin.
    direction:vec3f  // The ray direction.
};

// The definition of an intersection.
struct Hit {
    intersection:vec3f,      // The intersection point.
	t:f32,				  // The distance between the ray origin and the intersection points along the ray. 
    normal:vec3f,             // The surface normal at the interesection point.
	material:vec4f			  // The material of the object at the intersection point.
};
//const emptySphere = Sphere(vec3f(0.f),-1.f,vec4f(0.f)) ; 
const miss = Hit(vec3f(0.0f), 1e20, vec3f(0.0f), vec4f(0.f));
const missT = 1e20;
const numberOfPros = 8*4;



@compute @workgroup_size(8,4,1)
//fn main(@builtin(global_invocation_id) GlobalInvocationID:vec3<u32>){
fn main(@builtin(workgroup_id) WorkgroupID:vec3<u32>,@builtin(local_invocation_id) LocalInvocationID:vec3<u32>){

    let screen_size: vec2<u32> = textureDimensions(color_buffer);

    //let screen_pos : vec2<i32> = vec2<i32>(i32(GlobalInvocationID.x),i32(GlobalInvocationID.y));
	let screen_pos : vec2<i32> = vec2<i32>(i32(8*WorkgroupID.x + LocalInvocationID.x),i32(4*WorkgroupID.y + LocalInvocationID.y));

    let horizontal_coefficient: f32 = (f32(screen_pos.x) - f32(screen_size.x)/2) / f32(screen_size.x);
    let vertical_coefficient: f32 = (f32(screen_pos.y) - f32(screen_size.y)/2) / f32(screen_size.x);

    let forwards: vec3<f32> = vec3<f32>(0.0,0.0,1.0);
    let right: vec3<f32> = vec3<f32>(1.0,0.0,0.0);
    let up: vec3<f32> = vec3<f32>(0.0,-1.0,0.0);

	var uv = vec2f(horizontal_coefficient,vertical_coefficient);
	let aspect_ratio = 1280.0/720.0;
    var ray : Ray;
    //ray.direction = normalize(forwards + horizontal_coefficient * right + vertical_coefficient*up);
    //ray.origin = vec3<f32>(0.,0.5,-10.f);

	let P = (uCamera.inversePV * vec4f(uv, -1.f, 1.0)).xyz;

	ray.direction = normalize(P - uCamera.position.xyz);
    ray.origin = uCamera.position.xyz;

    var pixel_color: vec3<f32> = vec3<f32>(0.5,0.0,0.25);
	//workgroupBarrier();
    let color = Trace(ray,LocalInvocationID);
    //var closest_hit = miss;//RaySphereIntersection(myRay,mySphere);
    /*for(var i = 0; i < 4000; i++){
		var hit = RaySphereIntersection(ray, spheres[i]);
		if(hit.t < closest_hit.t){
			closest_hit = hit;
		}
	}*/
    //vec4f(color,1.)
    //closest_hit.material
    textureStore(color_buffer,screen_pos,vec4f(color,1.));
}


fn RaySphereIntersection(ray : Ray, sphereIndex : i32) -> Hit{//sphere : Sphere) -> Hit{
	let sphere = spheres[sphereIndex];
	//return miss;
	// Optimalized version.
	let oc = ray.origin - sphere.origin.xyz;
	let b = dot(ray.direction, oc);
	let c = dot(oc, oc) - 1.f;//(sphere.radius*sphere.radius);

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
		/*if (dot(ray.direction, n) > 0) {
			n = -n; 
		}*/
		return Hit(intersection+sphere.origin.xyz, t, n,vec4(0.f,1.f,0.f,1.f));//sphere.color);
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



fn Trace(ray : Ray,LocalInvocationID:vec3<u32>) -> vec3f{
    var light : Light;
	light.position = vec3f(0.0f,120.5f,10.0f);
	light.diffuse = vec3f(0.8f);
	// The accumulated color and attenuation used when tracing the rays throug the scene.



	var color = vec3f(0.0,0.0,0.0);
    var attenuation = vec3f(1.0);

	

	// Due to floating-point precision errors, when a ray intersects geometry at a surface, the point of intersection could possibly be just below the surface.
	// The subsequent reflection and shadow rays would then bounce off the *inside* wall of the surface. This is known as self-intersection.
	// We, therefore, use a small epsilon to offset the subsequent rays.
	let epsilon = 0.001f;

	

	var tmpRay = ray;

	
	for(var i = 0;i<1;i++){

		let hit = Evaluate(tmpRay,LocalInvocationID);
		var L = normalize(light.position -  hit.intersection);
		color += hit.material.xyz;//calculatePBR(ray,hit.intersection,N,hit.material.xyz);
		
			//continue;
		/*if (!isHitMiss(hit)) {

			let N = hit.normal;
			var V = normalize(-tmpRay.direction);
			let H = normalize(L + V);
			let NdotL = max(dot(N, L), 0.0);

			let ambient = 0.1*hit.material.xyz;
			color += 10.*ambient;
			let shadowOrigin = hit.intersection +  epsilon * N;

			let shadowRay  = (Ray(shadowOrigin, L));
			
			//color += hit.material.xyz;//calculatePBR(ray,hit.intersection,N,hit.material.xyz);
			//continue;
			let shadowHit = Evaluate(shadowRay,LocalInvocationID);

			
			
			var NdotV = dot(H,V);
			var F = schlickFresnel(NdotV,hit.material.xyz);
			//var F = schlickFresnel_refract(1.00029f,1.125f,NdotV);
			var F_ref = 1.0 - F;
			//REFRACTIVE_INDEX_OUTSIDE 1.00029
			//REFRACTIVE_INDEX_INSIDE  1.125

			//color += calculatePBR(ray,hit.intersection,N,hit.material.xyz);
			if(isHitMiss(shadowHit)){
				//color += calculatePBR(ray,hit.intersection,N,hit.material.xyz);
				//dif
				color += NdotL * light.diffuse * hit.material.xyz * attenuation;
				//specular
				let Geom = GeometricAttenuation(N, V, L, H);
				let Dist = BeckmannDistribution(N, H, 0.2);
				NdotV = dot(N,V);
				color += Dist * Geom * F / 4.0 / NdotV;
			}

			
			
			attenuation *= F;
			/*else{
				color +=ambient;
			}*/
			let reflected = reflect(tmpRay.direction,hit.normal);
			let newRay = Ray(hit.intersection +  epsilon * N, reflected);
			tmpRay = newRay;
		}*/

	}
	

    return color;
}

fn Evaluate(ray : Ray,LocalInvocationID:vec3<u32>) -> Hit{

	var closest_hit = miss;//RayPlaneIntersection(ray, vec3f(0.0f, 1.f, 0.f), vec3f(0.0f,0.0f,0.0f));
	
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