/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: raytracing_kdtree_ropes.wgsl
 *
 *  Description: 
 *  This shader is part of the prototype. It performs ray tracing(direct lighting) using Kd-tree with ropes.
 *  
 * -----------------------------------------------------------------------------
 */
struct Sphere{
	origin: vec3f,
	radius: f32,
	color: vec4f
};

const missT = 1e20;
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

@group(0) @binding(1) var<storage,read> kdTree: array<KdTreeNodeUBO>;
@group(0) @binding(2) var<storage,read> leaves: array<LeafUBO>;
@group(0) @binding(3) var<storage,read> spheres: array<Sphere>;

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
const spPerLeaf = 3;



struct Light{
	position:vec3f,
	direction:vec3f,
	diffuse:vec3f
}

struct Camera {
	
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
	position: vec4f,
	inversePV: mat4x4f
};
@group(0) @binding(0) var<uniform> uCamera: Camera;
const tex_coords = array<vec2f,3>(
	vec2f(0.0, 0.0),
	vec2f(2.0, 0.0),
	vec2f(0.0, 2.0)
);



struct VertexOutput {
	@builtin(position) position: vec4f,
	@location(0) tex_coord: vec2f,
};


@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
	//return vs_main_optionA(in);
	var texCoord = tex_coords[in_vertex_index];
	var out : VertexOutput;
	out.tex_coord = texCoord;
	out.position = vec4f(texCoord*2.0 - 1.0,0.0,1.0);
	return out;
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

fn RayPlaneIntersection(ray : Ray, normal:vec3f, point:vec3f) -> Hit {
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
    if(length(intersection) > 8){
        var h = miss;
		//h.material = vec4f(1.f,0.f,0.f,1.f);
		return h;
    }
	
    return Hit(intersection, t, normal, vec4f(0.5f));
}

fn RayBoxIntersection(ray : Ray, minP : vec3f, maxP : vec3f) -> vec2f{ //TODO: předělat na hit
	
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


fn RaySphereIntersection(ray : Ray, sphereId : i32) -> Hit{//sphere : Sphere) -> Hit{

	let sphere = spheres[sphereId];
	// Optimalized version.
	let oc = ray.origin - sphere.origin;
	let b = dot(ray.direction, oc);
	let c = dot(oc, oc) - (sphere.radius*sphere.radius);

	var det = b*b - c;
	if (det < 0.0) { return miss;}

	det = sqrt(det);
	/*var t = -b - det;
	if (t < 0.0){ 
		t = -b + det;
	}*/

	var t = select(-b - det,-b + det,-b - det < 0);
	if(t<0.0){return miss;}

	let intersection = oc + t * ray.direction;
	var n = normalize(intersection);
	if (dot(ray.direction, n) > 0) {
		n = -n; 
	}
	return Hit(intersection+sphere.origin, t, n,sphere.color);

}

fn Evaluate(ray : Ray) -> Hit{
	
	var closest_hit = miss;
	var kdResultHit = traverseKdTree(ray);
	if(kdResultHit.t <= closest_hit.t){
		return kdResultHit;
	}

    return closest_hit;
}

fn locateLeaf(nodeIndex : i32, point : vec3f) -> KdTreeNodeUBO{
	// &kdTree[0];
	//a = &kdTree[1];
	var i = nodeIndex;
	var dess = 0;
	var currentNode = kdTree[nodeIndex];
	//var dimSplit = currentNode.metaData.x;
	//return blankNode;
	//if(i == 0){
	
		let minAABB = leaves[0].minAABB;
		let maxAABB = leaves[0].maxAABB;

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
	
		dess++;

		
		
	}
	
	return currentNode;
	
}


fn test_renderLeafRopes(ray : Ray, leaf:LeafUBO) -> Hit{ //just to confirm that ray hits the box correctly

	var test = miss;
	var ffMain = RayBoxIntersection(ray,leaf.minAABB,leaf.maxAABB);
	let eps = 0.01f;
	if(ffMain.x != -999.f){
		test.t = ffMain.x;
	test.intersection = ray.origin + ray.direction * test.t;
	test.material = vec4f(1.0f,0.f,1.0f,1.f);
	}


	for(var i = 0;i<6;i++){
		let rope = leaf.ropes[i];
		if(rope == -1){
			continue;
		}
		
		let point = ray.origin + ray.direction *ffMain.y+eps;
		let nodeRope = locateLeaf(rope,point);
		let leafRope = leaves[nodeRope.leafId];
		var ff = RayBoxIntersection(ray,leafRope.minAABB,leafRope.maxAABB);
		let eps = 0.01f;
		if(ff.x == -999.f){
			continue;
		}
		if(ff.x<test.t){
			test.t = ff.x;
			test.intersection = ray.origin + ray.direction * test.t;
			test.material = vec4f(0.0f,1.f,0.0f,1.f);
		}
	}

	return test;
}
fn test_renderAllLeaves(ray : Ray) -> Hit{
	var closest_hit = miss;
	let n = arrayLength(&leaves);
	for(var j : u32 = 0; j < n; j++){
		let leaf = leaves[j];
		var ff = RayBoxIntersection(ray,leaf.minAABB,leaf.maxAABB);
		if(ff.x == missT){
			continue;
		}
		if(leaf.firstIndex == -1){continue;}
		for(var i = 0; i < leaf.numberOfSpheres; i++){
			//let leaf = &leaves[1s];
			let sphereId = i+leaf.firstIndex;
			let hit = RaySphereIntersection(ray,  sphereId);
			
			if(hit.t < closest_hit.t){
				closest_hit = hit;
			}
		}
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

fn test_renderUsingKdTree(ray : Ray) -> Hit{

	//return test_renderLeafRopes(ray, leaves[1]);
	var closest_hit = miss;
	closest_hit.material = vec4f(1.f,0.f,1.f,1.f);
	let rootAABB = leaves[0];
	var ff = RayBoxIntersection(ray,(rootAABB).minAABB,(rootAABB).maxAABB); //ff.x entry distane, ff.y exit distance
	//var ff = RayBoxIntersection(ray,vec4f(0.f),vec4f(1.f));
	let eps = 0.001f;
	if(ff.x > ff.y || ff.y < 0.f){
		return miss;
	}
	
	var point = ray.origin;
	if(ff.x >= 0){
		point += ray.direction *(ff.x + eps);
	}
	
	var currentNode = (locateLeaf(0,point));
	

	//var j =0;
	while(currentNode.leafId > 0){
		//j++;	
		let leaf = leaves[currentNode.leafId];
		let firstIndex = leaf.firstIndex;
		let numberOfSpheres = leaf.numberOfSpheres;
		//return test_renderLeaf(ray,leaf.minAABB,leaf.maxAABB);

		for(var i = 0; i < numberOfSpheres; i++){
			let sphereId = i+firstIndex;//firstIndex;//i+firstIndex;
			
			let hit = RaySphereIntersection(ray,  sphereId);
			if(hit.t < closest_hit.t){
				closest_hit = hit;
				//closest_hit.material = vec4f(0.f,1.f,0.f,1.f);
				//return closest_hit;
			}
		}
		
	
		
		
		if(closest_hit.t < miss.t){
			//closest_hit.material = vec4f(1.f,1.f,0.f,1.f);
			return closest_hit;
		}
		
		
		ff = RayBoxIntersection(ray,(leaf).minAABB,(leaf).maxAABB);
		if(ff.y<0){
			point = ray.origin - ray.direction * (ff.y - eps);
		}else{
			point = ray.origin + ray.direction * (ff.y + eps);
		}
		//point = ray.origin + ray.direction * (ff.y + eps);
		let rope = exitRope(ray,currentNode.leafId); 
		
		if(leaf.ropes[rope] != -1){ 
			currentNode = (locateLeaf(leaf.ropes[rope],point));
		}else{
			currentNode = (locateLeaf(0,point));
		}
		//currentNode = (locateLeaf(leaf.ropes[rope],point));
		
		
	}
	
	return closest_hit;
}

fn exitRope(ray:Ray, leafId : i32) -> i32{
	let leaf = leaves[leafId];
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





fn traverseKdTree(ray :Ray) -> Hit{
	//return miss;
	//return test_renderAllSpheres(ray);
	//return test_renderSmallCube(ray);
	//return test_renderAllLeaves(ray);
	return test_renderUsingKdTree(ray);
	
	
	//return miss;

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
			let shadowOrigin = hit.intersection +  epsilon * N;

			let shadowRay  = (Ray(shadowOrigin, L));
			
			

			
			
			var NdotV = dot(H,V);
			var F = schlickFresnel(NdotV,hit.material.xyz);
			//var F = schlickFresnel_refract(1.00029f,1.125f,NdotV);
			var F_ref = 1.0 - F;
			
			//if(isHitMiss(shadowHit)){
				//color += calculatePBR(ray,hit.intersection,N,hit.material.xyz);
				//dif
				color += NdotL * light.diffuse * hit.material.xyz * attenuation;
				//specular
				let Geom = GeometricAttenuation(N, V, L, H);
				let Dist = BeckmannDistribution(N, H, 0.2);
				NdotV = dot(N,V);
				color += Dist * Geom * F / 4.0 / NdotV;
			//}
 
			
			let reflected = reflect(tmpRay.direction,hit.normal);
			let newRay = Ray(hit.intersection +  epsilon * N, reflected);
			tmpRay = newRay;
		}else{
			if(i==0){
				//color = hit.material.xyz;
				break;
			}
			//color = hit.material.xyz;
		}

	}


    return color;
}

fn getBBoxCenter(minP:vec4f,maxP:vec4f) -> vec4f{
	var center = vec4f(0.f);
	center.x = (maxP.x - minP.x)/2;
	center.y = (maxP.y - minP.y)/2;
	center.z = (maxP.z - minP.z)/2;
	center.w = 1.f;
	return center;
}

fn isHitMiss(hit:Hit) -> bool{
	/*if(hit.intersection.x != miss.intersection.x && hit.intersection.y != miss.intersection.y && hit.intersection.z != miss.intersection.z ){
		return false;
	}*/
	if(hit.t != miss.t){
		return false;
	}
	/*if(hit.normal.x != miss.normal.x && hit.normal.y != miss.normal.y && hit.normal.z != miss.normal.z ){
		return false;
	}
	if(hit.material.x != miss.material.x && hit.material.y != miss.material.y && hit.material.z != miss.material.z ){
		return false;
	}*/
	return true;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {

	//ProjectionView
	let aspect_ratio = 1280.0/720.0;
	let uv = (2.0*in.tex_coord-1.0)* vec2f(aspect_ratio, 1.0);// * vec2f(aspect_ratio, 1.0);
	let P = (uCamera.inversePV * vec4f(uv, -1.f, 1.0)).xyz;

	let direction = normalize(P - uCamera.position.xyz);
	
	let ray = Ray(uCamera.position.xyz, direction);
	let color = Trace(ray); 
	return vec4f(color, 1.0);
}


const PI = 3.14159265;
const roughnessConst = 0.5f;
const metallic = false;
fn schlickFresnel(vDotH : f32,color:vec3f) -> vec3f{
	var F0 = vec3f(0.04);


	let res = F0 + (1.0f-F0) * (pow(clamp(1.0f - vDotH,0.f,1.f),5));
	return res;
}



fn ggxDistribution(nDotH:f32) -> f32{
	let alpha2 = roughnessConst * roughnessConst * roughnessConst * roughnessConst;
	let d = nDotH * nDotH * (alpha2 - 1.f) + 1.f;
	let ggxDistr = alpha2 / (PI * d * d);
	return ggxDistr;
}

fn geomSmith(dp:f32) -> f32{
	let k = (roughnessConst + 1.f) * (roughnessConst + 1.f) / 8.0f;
	let denom = dp*(1.f-k) + k;
	return dp/denom;
}




fn getCloserHit(hit1: Hit, hit2: Hit) -> Hit{
	if(hit1.t<hit2.t){
		return hit1;
	}else{
		return hit2;
	}
}