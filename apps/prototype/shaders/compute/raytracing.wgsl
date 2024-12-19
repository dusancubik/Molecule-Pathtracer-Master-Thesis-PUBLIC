/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: screen_shader.wgsl
 *
 *  Description: 
 *  This shader manages ray tracing using Kd-tree.
 *  
 * -----------------------------------------------------------------------------
 */
@group(0) @binding(0) var color_buffer: texture_storage_2d<rgba8unorm,write>;
@group(0) @binding(1) var<storage,read> kdTree: array<KdTreeNodeUBO>;
@group(0) @binding(2) var<storage,read> leaves: array<LeafUBO>;
@group(0) @binding(3) var<storage,read> spheres: array<Sphere>;

struct Light{
	position:vec3f,
	direction:vec3f,
	diffuse:vec3f
}

struct Sphere{
	origin: vec3f,
	radius: f32,
	color: vec4f
};

struct LeafUBO{
	minAABB: vec3f,
	firstIndex: i32,
	maxAABB: vec3f,
	numberOfSpheres: i32,
	ropes: array<i32,6>
};

struct KdTreeNodeUBO{
	splitPoint: f32,
	leafId:i32,
	leftChild:i32,
	rightChild:i32,
};

struct NodeId{
	node:KdTreeNodeUBO,
	id:i32
}
struct Ray {
    origin:vec3f,     
    direction:vec3f  
};


struct Hit {
    intersection:vec3f,    
	t:f32,				  
    normal:vec3f,             
	material:vec4f			 
};
const emptySphere = Sphere(vec3f(0.f),-1.f,vec4f(0.f)) ; 
const miss = Hit(vec3f(0.0f), 1e20, vec3f(0.0f), vec4f(0.f));
const blankNode = KdTreeNodeUBO(-1.f,-1,-1,-1); 

@compute @workgroup_size(8,8,1)
fn main(@builtin(workgroup_id) WorkgroupID:vec3<u32>,@builtin(local_invocation_id) LocalID:vec3<u32>){

    let screen_size: vec2<u32> = textureDimensions(color_buffer);

	let screen_pos : vec2<i32> = vec2<i32>(i32(8*WorkgroupID.x + LocalID.x),i32(8*WorkgroupID.y + LocalID.y));

    let uv_x: f32 = (f32(screen_pos.x) - f32(screen_size.x)/2) / f32(screen_size.x);
	let uv_y: f32 = (f32(screen_size.y) / 2 - f32(screen_pos.y)) / f32(screen_size.x);

	var uv = vec2f(uv_x,uv_y);

    var mySphere: Sphere;
    mySphere.origin = vec3<f32>(3.0,0.0,0.0);
    mySphere.radius = 1.f;
    mySphere.color = vec4f(0.f,1.f,0.f,1.f);

    var ray : Ray;
	let P = (uCamera.inversePV * vec4f(uv, -1.f, 1.0)).xyz;

	ray.direction = normalize(P - uCamera.position.xyz);
    ray.origin = vec3<f32>(0.,0.5,-10.f);


    var pixel_color: vec3<f32> = vec3<f32>(0.5,0.0,0.25);

    let color = Trace(ray,LocalID);

    textureStore(color_buffer,screen_pos,vec4f(color,1.));
}

fn RaySphereIntersection(ray : Ray, sphereIndex : i32) -> Hit{//sphere : Sphere) -> Hit{
	let sphere = spheres[sphereIndex];
	//return miss;
	// Optimalized version.
	let oc = ray.origin - sphere.origin;
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
		/*if (dot(ray.direction, n) > 0) {
			n = -n; 
		}*/
		return Hit(intersection+sphere.origin, t, n,sphere.color);
	}else{
		return miss;
	}

}

fn locateLeafWithId(nodeIndex : i32, point : vec3f) -> NodeId{
	// &kdTree[0];
	//a = &kdTree[1];
	var i = nodeIndex;
	
	var currentNode = kdTree[nodeIndex];
	let rootLeaf = leaves[0];
	//if(i == 0){
	
		let minAABB = rootLeaf.minAABB;
		let maxAABB = rootLeaf.maxAABB;

		if(!( minAABB.x <= point.x && point.x <= maxAABB.x && minAABB.y <= point.y && point.y <= maxAABB.y && minAABB.z <= point.z && point.z <= maxAABB.z)) { //TODO
			return NodeId(blankNode,-1);
		}
	//}
	

	var dimSplit = -(currentNode.leafId+1);//0;
	
	while(currentNode.leafId < 1){
		
		let left = point[dimSplit] < currentNode.splitPoint;
		i = select(currentNode.rightChild,currentNode.leftChild, left);
		//return blankNode;
		currentNode = kdTree[i]; 

		dimSplit = -(currentNode.leafId+1);//(dimSplit+1)%3;
	
		
		
	}
	
	return NodeId(currentNode,i);
	
}

fn Trace(ray : Ray,LocalID:vec3<u32>) -> vec3f{
    var light : Light;
	light.position = vec3f(0.0f,120.5f,10.0f);
	light.diffuse = vec3f(0.8f);
	



	var color = vec3f(0.0,0.0,0.0);
    var attenuation = vec3f(1.0);


	let epsilon = 0.001f;

	

	var tmpRay = ray;

	
	for(var i = 0;i<2;i++){

		let hit = Evaluate(tmpRay,LocalID);
		var L = normalize(light.position -  hit.intersection);

		if (!isHitMiss(hit)) {

			let N = hit.normal;
			var V = normalize(-tmpRay.direction);
			let H = normalize(L + V);
			let NdotL = max(dot(N, L), 0.0);

			let ambient = 0.1*hit.material.xyz;
			color += 10.*ambient;
			let shadowOrigin = hit.intersection +  epsilon * N;

			let shadowRay  = (Ray(shadowOrigin, L));
			

			let shadowHit = Evaluate(shadowRay,LocalID);

			
			
			var NdotV = dot(H,V);
			var F = schlickFresnel(NdotV,hit.material.xyz);
			
			var F_ref = 1.0 - F;

			if(isHitMiss(shadowHit)){

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
		}

	}
	

    return color;
}

fn Evaluate(ray : Ray,LocalID:vec3<u32>) -> Hit{

	var closest_hit = miss;//RayPlaneIntersection(ray, vec3f(0.0f, 1.f, 0.f), vec3f(0.0f,0.0f,0.0f));
	
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
	//return test_renderAllSpheres(ray);
	
	//return test_renderSmallCube(ray);
	//return test_renderAllLeaves(ray);
	return test_renderUsingKdTree(ray);
	
	
	//return miss;

}




fn test_renderUsingKdTree(ray : Ray) -> Hit{
	//let root = kdTree[0];
	
	let rootAABB = leaves[0];
	var ff = RayBoxIntersection(ray,(rootAABB).minAABB,(rootAABB).maxAABB); //ff.x entry distane, ff.y exit distance
	//var ff = RayBoxIntersection(ray,vec4f(0.f),vec4f(1.f));	
	let eps = 0.01f;
	if(ff.x == -999.f || ff.y < 0.f){
		return miss;
	}
	var point = ray.origin;
	if(!(ff.x < 0.f)){
		point += ray.direction *(ff.x + eps);
	}
	
	var currentNode = (locateLeaf(0,point));
	var closest_hit = miss;
	
	while(currentNode.leafId > 0){
		
		//j++;
		let leaf = leaves[currentNode.leafId];
		let firstIndex = leaf.firstIndex;
		let numberOfSpheres = leaf.numberOfSpheres;
		
		for(var i = 0; i < numberOfSpheres; i++){
			let sphereId = i+firstIndex;
			//Sphere(vec3f(1.f),1.f,vec4f(1.f,0.f,1.f,1.f));//spheres[i];
			let hit = RaySphereIntersection(ray,  sphereId);

			if(hit.t < closest_hit.t){
				closest_hit = hit;
			}	
			
		}
		
		
		if(closest_hit.t  < miss.t){
			return closest_hit;
		}		

		ff = RayBoxIntersection(ray,(leaf).minAABB,(leaf).maxAABB);
		point = ray.origin + ray.direction * (ff.y + eps);	
		currentNode = (locateLeaf(0,point));
		/*let rope = exitRope(ray,leaf); 

		if(leaf.ropes[rope] == -1){ return miss;}
		currentNode = (locateLeaf(leaf.ropes[rope],point));*/
		
		
	}
	
	return closest_hit;
}

fn RayBoxIntersection(ray : Ray, minP : vec3f, maxP : vec3f) -> vec2f{ 

	let eps = 0.00001;
	
	let ray_min_tmp = (minP - ray.origin) /  (ray.direction);
	let ray_max_tmp = (maxP - ray.origin) / (ray.direction);

	let ray_min = min(ray_min_tmp,ray_max_tmp);
	let ray_max = max(ray_min_tmp,ray_max_tmp);

	let tmin = max(max(ray_min.x,ray_min.y),ray_min.z);
	let tmax = min(min(ray_max.x,ray_max.y),ray_max.z);

	if(tmin>tmax){ return vec2(-999.f,-999.f);}
	if(tmax<0){ return vec2(-999.f,-999.f);}

	return vec2(tmin,tmax);
}



fn locateLeaf(nodeIndex : i32, point : vec3f) -> KdTreeNodeUBO{
	// &kdTree[0];
	//a = &kdTree[1];
	var i = nodeIndex;
	
	var currentNode = kdTree[nodeIndex];
	let rootLeaf = leaves[0];
	//if(i == 0){
	
		let minAABB = rootLeaf.minAABB;
		let maxAABB = rootLeaf.maxAABB;

		if(!( minAABB.x <= point.x && point.x <= maxAABB.x && minAABB.y <= point.y && point.y <= maxAABB.y && minAABB.z <= point.z && point.z <= maxAABB.z)) { //TODO
			return blankNode;
		}
	//}
	

	var dimSplit = -(currentNode.leafId+1);//0;
	
	while(currentNode.leafId < 1){
		
		let left = point[dimSplit] < currentNode.splitPoint;
		i = select(currentNode.rightChild,currentNode.leftChild, left);
		//return blankNode;
		currentNode = kdTree[i]; 

		dimSplit = -(currentNode.leafId+1);//(dimSplit+1)%3;
	
		
		
	}
	
	return currentNode;
	
}


fn exitRope(ray:Ray, leaf : LeafUBO) -> i32{
	let ray_min_tmp = (leaf.minAABB - ray.origin) /  (ray.direction);
	let ray_max_tmp = (leaf.maxAABB - ray.origin) / (ray.direction);

	let ray_min = min(ray_min_tmp,ray_max_tmp);
	let ray_max = max(ray_min_tmp,ray_max_tmp);

	
	var maxTmin = 9999999.f;
	var maxFace = -1;
	for(var i = 0;i<3;i++){
		if(ray_max[i] < maxTmin){
			maxTmin = ray_max[i];
			maxFace = i*2 + i32(ray.direction[i] > 0.0);
		}
	}

	return maxFace;

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

