/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: raytracing_bvh_frag.wgsl
 *
 *  Description: 
 *  This shader is part of the prototype. It performs ray tracing(direct lighting) using Kd-tree with sequential traversal.
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
};

struct KdTreeNodeUBO{
	//dimSplit: i32, //x=0, y=1, z=2
	//splitDistance: f32,
	/*splitPoint: vec4f,
	dimSplit:i32,
	depth:i32,
	leafId:i32,
	padd:i32
	//isLeaf: i32,
	//depth: i32,
	metaData: vec4f, //(dimSplit,depth,isLeaf,1.0)
	spheres:array<Sphere, 3>,
	minAABB: vec4f,
	maxAABB: vec4f*/

	//splitPoint: vec4f,
	//dimSplit:i32,
	splitPoint: f32,
	leafId:i32,
	leftChild:i32,
	rightChild:i32,
};
/*struct Spheres{
	sphere: array<Sphere>;	
};*/
//@group(0) @binding(1) var<storage> spheres1: array<Sphere>;
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
const miss = Hit(vec3f(0.0f), missT, vec3f(0.0f), vec4f(0.f));
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

/*fn RayBoxIntersection(ray: Ray, minP: vec3f, maxP: vec3f) -> vec2f {
    let eps = 0.00001;

    // Handle division by zero in ray direction
    let safe_dir = ray.direction + vec3f(eps);
    let ray_min_tmp = (minP - ray.origin) / safe_dir;
    let ray_max_tmp = (maxP - ray.origin) / safe_dir;

    // Correct for ray direction signs
    let ray_min = min(ray_min_tmp, ray_max_tmp);
    let ray_max = max(ray_min_tmp, ray_max_tmp);

    // Calculate entry and exit distances
    let tmin = max(max(ray_min.x, ray_min.y), ray_min.z);
    let tmax = min(min(ray_max.x, ray_max.y), ray_max.z);

    // Handle edge cases
    if (tmin > tmax + eps) { return vec2(missT, missT); } // Ray misses box
    if (tmax < 0) { return vec2(missT, missT); }          // Box is behind ray

    // Clamp tmin to 0 for rays starting inside the box
    let clamped_tmin = max(tmin, 0.0);

    return vec2(clamped_tmin, tmax);
}*/
fn RaySphereIntersection(ray : Ray, sphereId : i32) -> Hit{//sphere : Sphere) -> Hit{
	
	
	let sphere = spheres[sphereId];
	let rayOrigin = ray.origin - sphere.origin;
	let rayDirection = ray.direction;
	let a = dot(rayDirection,rayDirection);
	let b = 2.0f * dot(rayOrigin,rayDirection);
	let c = dot(rayOrigin,rayOrigin) - sphere.radius * sphere.radius;

	let disc = b * b - 4.0f * a * c;
	var t = (-b - sqrt(disc))/(2.0f * a);

	if(t<0.f){
		t = (-b + sqrt(disc))/(2.0f * a);
	}
		
	if(disc>=0.0f && t>=0){

		let hitPoint = rayOrigin + rayDirection * t;
		//var n = hitPoint-sphere.origin;
		var n = normalize(hitPoint);
		let r = normalize(ray.direction);
		
        if (dot(r, n) > 0) {
            n = -n; // Invert the normal
        }
		return Hit(hitPoint+sphere.origin, t, n/*select(n, -1*n, dot(r,n)>0)*/,sphere.color);
	}else{
		return miss;
	}




}

fn Evaluate(ray : Ray) -> Hit{
	
	var closest_hit = miss;
	//let kdTree = _kdTree;
	var kdResultHit = traverseKdTree(ray);
	if(kdResultHit.t <= closest_hit.t){
		return kdResultHit;
	}

    return closest_hit;
}

fn locateLeaf(nodeIndex : i32, point : vec3f,root : KdTreeNodeUBO,rootLeaf : LeafUBO) -> KdTreeNodeUBO{
	// &kdTree[0];
	//a = &kdTree[1];
	var i = nodeIndex;
	
	var currentNode = root;//kdTree[0];
	//var dimSplit = currentNode.metaData.x;
	//return blankNode;
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
	
		/*if(*dimSplit < 0){
			return blankNode;
		}*/

		
		/*var left = false;
		if(*dimSplit == 0){
			left = point[0] < currentNode.splitPoint.x;
		}else if(*dimSplit == 1){
			left = point.y < currentNode.splitPoint.y;
		}else if(*dimSplit == 2){
			left = point.z < currentNode.splitPoint.z;
		}*/

		/*switch(*dimSplit){
			case 0: {
				left = point.x < currentNode.splitPoint.x;
			}
			//break;
			case 1: {
				left = point.y < currentNode.splitPoint.y;
			}
			//break;
			case 2: {
				left = point.z < currentNode.splitPoint.z;
			}

			default:{//blank node
				//return blankNode;
			}
			//break;
		}*/
		//i = select(2 * i + 2,2 * i + 1, left);

		//currentNode = kdTree[i]; 
		
	}
	//return &KdTreeNodeUBO(currentNode.splitPoint,currentNode.dimSplit,currentNode.depth,currentNode.leafId,-1);
	return currentNode;
	//return KdTreeNodeUBO(currentNode.splitPoint,currentNode.metaData,currentNode.spheres,currentNode.minAABB,currentNode.maxAABB);//currentNode;
}
fn test_renderSmallCube(ray : Ray) -> Hit{ //just to confirm that ray hits the box correctly
	//let rootAABB = &leaves[0];
	//var ff = RayBoxIntersection(ray,(*rootAABB).minAABB,(*rootAABB).maxAABB); //ff.x entry distane, ff.y exit distance
	var ff = RayBoxIntersection(ray,vec3f(-372.f,0.f,-372.f),vec3f(372.f,42.f,372.f));
	let eps = 0.01f;
	if(ff.x == missT){
		return miss;
	}
	/*if(ff.y < 0.f){
		return miss;
	}*/
	var test = miss;
	test.t = ff.x;
	test.intersection = ray.origin + ray.direction * test.t;
	test.material = vec4f(1.0f,0.f,0.f,1.f);
	test.normal = vec3(1.f,0.0f,0.f);
	return test;
}

fn test_renderLeaf(ray : Ray, minP : vec3f, maxP : vec3f) -> Hit{ //just to confirm that ray hits the box correctly
	//let rootAABB = &leaves[0];
	//var ff = RayBoxIntersection(ray,(*rootAABB).minAABB,(*rootAABB).maxAABB); //ff.x entry distane, ff.y exit distance
	var ff = RayBoxIntersection(ray,minP,maxP);
	let eps = 0.01f;
	if(ff.x == missT){
		var h = miss;
		h.material = vec4f(0.f,1.0f,0.f,1.f);
		return h;
	}
	/*if(ff.y < 0.f){
		return miss;
	}*/
	var test = miss;
	test.t = ff.x;
	test.intersection = ray.origin + ray.direction * test.t;
	test.material = vec4f(1.0f,0.f,1.0f,1.f);
	test.normal = vec3(1.f,0.0f,0.f);
	return test;
}

/*fn test_renderAllLeaves(ray : Ray) -> Hit{
	var closest_hit = miss;
	let n = arrayLength(&leaves);
	for(var j : u32 = 0; j < n; j++){
		let leaf = &leaves[j];
		for(var i = 0; i < spPerLeaf; i++){
			//let leaf = &leaves[1s];
			let sphere = &((*leaf).spheres[i]);
			if((*sphere).radius < 0.f){
				break;
			}
			let hit = RaySphereIntersection(ray,  *sphere);
			
			if(hit.t < closest_hit.t){
				closest_hit = hit;
			}
		}
	}
	return closest_hit;
}*/

fn test_renderUsingKdTree(ray : Ray) -> Hit{
	let root = kdTree[0];
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
	if(ff.x >= 0){// || (ff.x > 0.f && ff.y < 0.f)){  ??
		//point does NOT lie inside the box
		point += ray.direction *(ff.x + eps);
		
		//closest_hit.material =  vec4f(0.f,0.f,1.f,1.f);

	}
	//if(ff.x<=0.f ){
		/*var h = miss;
		h.material = vec4f((ff.x+ eps)/740.f,0.5f,0.f,1.f);
		return h;*/
	//}

	
	//select(n, -1*n, dot(r,n)>0)
	/*if(!(ff.x < 0.f)){// || (ff.x > 0.f && ff.y < 0.f)){  ??
		//point does NOT lie inside the box
		var h = miss;
		h.material = vec4f(0.f,0.5f,1.f,1.f);
		return h;
	}*/
	var currentNode = (locateLeaf(0,point,root,rootAABB));
	/*if(currentNode.leafId == -1){
		var h = miss;
		h.material = vec4f(0.f,0.f,1.f,1.f);
		return h;
	}else{
		var h = miss;
		h.material = vec4f(1.f,0.f,0.f,1.f);
		return h;
	}*/
	//return miss;
	var j = 0;
	var prev_id = currentNode.leafId;


	while(currentNode.leafId > 0){
		
		j++;
		let leaf = leaves[currentNode.leafId];
		let firstIndex = leaf.firstIndex;
		let numberOfSpheres = leaf.numberOfSpheres;

		if(firstIndex == -1){
			closest_hit.material = vec4f(1.f,0.5f,0.2f,1.f);//numberOfSpheres

		}
		//return test_renderLeaf(ray,leaf.minAABB,leaf.maxAABB);
		/*var h = miss;
		h.t = 0.5f;
		h.material = vec4f(0.f,f32(firstIndex)/100.f,0.f,1.f);//numberOfSpheres
		return h;*/
		/*if(ff.x == -999.f || ff.y < 0.f){
			return miss;
		}*/
		for(var i = 0; i < numberOfSpheres; i++){
			let sphereId = i+firstIndex;//firstIndex;//i+firstIndex;
			/*if(sphereId < 0){
				break;
			}*/
			//let sphere = (spheres[sphereId]);
			//closest_hit.sphere = sphere;
			
			/*if((sphere).radius < 0.f){
				break;
			}*/
			
			let hit = RaySphereIntersection(ray,  sphereId);
			//closest_hit = getCloserHit(closest_hit,hit);
			if(hit.t < closest_hit.t){
				closest_hit = hit;
				//closest_hit.material = vec4f(0.f,1.f,0.f,1.f);
				//return closest_hit;
			}
			
			//continue;
		}
		
		/*if(!isHitMiss(closest_hit)){
			return closest_hit;
		}*/
		
		
		if(closest_hit.t < miss.t){
			//closest_hit.material = vec4f(1.f,1.f,0.f,1.f);
			return closest_hit;
		}
		
		/*if(ff.y<0){
			point = ray.origin + ray.direction * (ff.y - eps);
		}else{
			point = ray.origin + ray.direction * (ff.y + eps);
		}*/
		ff = RayBoxIntersection(ray,(leaf).minAABB,(leaf).maxAABB);
        //point = ray.origin + ray.direction * (ff.y + eps);

		//point = ray.origin + ray.direction * (ff.y + eps);
		if(ff.y<0){
			point = ray.origin - ray.direction * (ff.y - eps);
		}else{
			point = ray.origin + ray.direction * (ff.y + eps);
		}
		currentNode = (locateLeaf(0,point,root,rootAABB));
		if (prev_id == currentNode.leafId){ break;};
		/*if(prev_id == currentNode.leafId){
			closest_hit.material = vec4f(0.f,1.f,0.f,1.f);	
			break;
		}*/
		/*if(currentNode.leafId < 0 && firstIndex == -1){
			var h = miss;
			//h.t = 0.5f;
			h.material = vec4f(0.5f,0.5f,0.5f,1.f);//numberOfSpheres
			return h;
		}*/
		prev_id = currentNode.leafId;
		
		/*if(!( (leaf).minAABB.x <= point.x && point.x <= (leaf).maxAABB.x && (leaf).minAABB.y <= point.y && point.y <= (leaf).maxAABB.y && (leaf).minAABB.z <= point.z && point.z <= (leaf).maxAABB.z)) { //TODO
			return miss;
		}*/

		/*if(currentNode.leafId == -1){
			var h = miss;
			h.material = vec4f(0.5f,0.f,1.f,1.f);
			return h;
		}*/
		//return miss; //pro jednu iteraci
		
	}
	/*if(j==29){
		closest_hit.material = vec4f(0.f,0.f,1.f,1.f);
	}*/
	//closest_hit.material = vec4f(1.f,0.f,0.f,1.f);
	return closest_hit;
}



fn test_renderAllLeaves(ray : Ray) -> Hit{
	var closest_hit = miss;
	let n = arrayLength(&leaves);
	for(var j : u32 = 0; j < n; j++){
		let leaf = leaves[j];
		var ff = RayBoxIntersection(ray,leaf.minAABB,leaf.maxAABB);
		/*if(ff.x == missT){
			continue;
		}*/
		if (ff.x > ff.y || ff.y < 0.0) {
			continue;
		}
		//if(leaf.firstIndex == -1){continue;}
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

fn traverseKdTree(ray :Ray) -> Hit{
	//return miss;
	
	//return test_renderSmallCube(ray);
	//return test_renderAllLeaves(ray);
	return test_renderUsingKdTree(ray);
	//return kdTreePushDown(ray);
	
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

//------------------- PBR -------------------
const PI = 3.14159265;
const roughnessConst = 0.5f;
const metallic = false;
fn schlickFresnel(vDotH : f32,color:vec3f) -> vec3f{
	var F0 = vec3f(0.04);
	//TODO: if is metal

	if(metallic){
		F0 = color;
	}

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


fn calculatePBR(_cameraRay:Ray,localP:vec3f,normal:vec3f,color:vec3f) -> vec3f{
	var cameraRay = _cameraRay;
	//cameraRay.direction = localP - cameraRay.origin;
	var light : Light;
	light.position = vec3f(0.0f,4.1f,0.0f);
	light.diffuse = vec3f(1.f);

	var L = normalize(light.position -  localP);
	let N = normal;
	let V = normalize(-cameraRay.direction);
	let H = normalize(V + L);

	let nDotH = max(dot(N,H),0.0f);
	let vDotH = max(dot(V,H),0.0f);
	let nDotL = max(dot(N,L),0.0f);
	let nDotV = max(dot(N,V),0.0f);

	let F = schlickFresnel(vDotH,color);

	let kS = F;
	let kD = 1.0 - kS;

	let specBRDF_nom = ggxDistribution(nDotH) * F * geomSmith(nDotL) * geomSmith(nDotV);

	let specBRDF_denom = 4.0f * nDotV * nDotL + 0.0001;

	let specBRDF = specBRDF_nom /specBRDF_denom;

	var flambert = vec3(0.f);
	if(metallic){
		flambert = color;
	}

	let diffuseBRDF = kD * flambert / PI;

	let finalColor = (diffuseBRDF + specBRDF) * light.diffuse * nDotL;
	return finalColor;
}

fn getCloserHit(hit1: Hit, hit2: Hit) -> Hit{
	if(hit1.t<hit2.t){
		return hit1;
	}else{
		return hit2;
	}
}