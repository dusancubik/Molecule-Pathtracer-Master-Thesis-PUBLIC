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
// ----------------------------------------------------------------------------
// Ray Tracing Structures
// ----------------------------------------------------------------------------
// The definition of a ray.
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

/*struct VertexInput {
	@location(0) position: vec3f,
    //                        ^ This was a 2
	@location(1) color: vec3f,
};*/

struct VertexOutput {
	@builtin(position) position: vec4f,
	@location(0) tex_coord: vec2f,
};

/**
 * A structure holding the value of our uniforms
 */
/*struct MyUniforms {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,
    color: vec4f,
    time: f32,
};*/

// Instead of the simple uTime variable, our uniform variable is a struct
//@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;

//const pi = 3.14159265359;
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
	
    return Hit(intersection, t, normal, vec4f(0.5f));
}

fn RayBoxIntersection(ray : Ray, minP : vec3f, maxP : vec3f) -> vec2f{ //TODO: předělat na hit
	//ray.origin-= getBBoxCenter(minP,maxP).xyz;
	//var ray = ray1;
	//ray.origin-= getBBoxCenter(minP,maxP).xyz;
	//let invDir = 1 / ray.direction;
	let eps = 0.00001;
	/*var tmin = (minP.x - ray.origin.x)  / (ray.direction.x + eps); //todo: použít variantu s násobením
    var tmax = (maxP.x - ray.origin.x) / (ray.direction.x + eps); 
 
    if (tmin > tmax) {
		let tmp = tmin;
		tmin = tmax;
		tmax = tmp;
	}; 
 
    var tymin = (minP.y - ray.origin.y)   / (ray.direction.y + eps); 
    var tymax = (maxP.y - ray.origin.y)  / (ray.direction.y + eps); 
 
    if (tymin > tymax) {
		let tmp = tymin;
		tymin = tymax;
		tymax = tmp;
	}; 
 
    if ((tmin > tymax) || (tymin > tmax)){ 
        return vec2f(-999.f,-999.f); //prozatím vec2f(-111.f,-111.f) == miss, todo: předělat na Hit 
	}
 
    if (tymin > tmin) {
        tmin = tymin; 
	}
    if (tymax < tmax) {
        tmax = tymax; 
	}
 
    var tzmin = (minP.z - ray.origin.z)    / (ray.direction.z + eps); 
    var tzmax = (maxP.z - ray.origin.z)  / (ray.direction.z + eps); 
 
    if (tzmin > tzmax) {
		let tmp = tzmin;
		tzmin = tzmax;
		tzmax = tmp;
		
	}; 
 
    if ((tmin > tzmax) || (tzmin > tmax)){
        return vec2f(-999.f,-999.f); 
	}
 
    if (tzmin > tmin) {
        tmin = tzmin; 
	}
    if (tzmax < tmax) {
        tmax = tzmax; 
	}
 
    return vec2f(tmin,tmax); */
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


fn RaySphereIntersection(ray : Ray, sphereId : i32) -> Hit{//sphere : Sphere) -> Hit{
	/*let rayOrigin = ray.origin - sphere.origin;
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
		// Check if the ray is inside the sphere
        if (dot(r, n) > 0) {
            n = -n; // Invert the normal
        }
		return Hit(hitPoint+sphere.origin, t, n/*select(n, -1*n, dot(r,n)>0)*/,sphere.color);
	}else{
		return miss;
	}*/
	
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
	// Sets the closes hit either to miss
	var closest_hit = miss;//RayPlaneIntersection(ray, vec3f(0.0f, 1.f, 0.f), vec3f(0.0f,0.0f,0.0f));
	//closest_hit.material = vec4f(1.f,0.f,1.f,1.f);
	
	//let spheres = array<Sphere, 2>(sphere1,sphere2);
	/*for(var i = 0; i < 20; i++){
		var hit = RaySphereIntersection(ray, spheres[i]);
		if(hit.t < closest_hit.t){
			closest_hit = hit;
		}
	}

	for(var i = 0; i < 20; i++){
		var node = kdTree[i];
		let dimSplit = node.metaData.x;
		let depth = node.metaData.y;
		let isLeaf = node.metaData.z;
		if(isLeaf == 1.0f){
			for(var i = 0; i < 3; i++){
				let sphere = node.spheres[i];
				if(sphere.radius <= 0.001f){
					break;
				}
				var hit = RaySphereIntersection(ray,  sphere);
				if(hit.t < closest_hit.t){
					closest_hit = hit;
				}
			}
		}
	}*/
	//let kdTree = _kdTree;
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
/*fn test_renderSmallCube(ray : Ray) -> Hit{ //just to confirm that ray hits the box correctly
	//let rootAABB = &leaves[0];
	//var ff = RayBoxIntersection(ray,(*rootAABB).minAABB,(*rootAABB).maxAABB); //ff.x entry distane, ff.y exit distance
	var ff = RayBoxIntersection(ray,vec3f(-372.f,0.f,-372.f),vec3f(372.f,42.f,372.f));
	let eps = 0.01f;
	if(ff.x == -999.f){
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
	if(ff.x == -999.f){
		var h = miss;
		h.material = vec4f(0.f,1.0f,0.f,1.f);
		return h;
	}

	var test = miss;
	test.t = ff.x;
	test.intersection = ray.origin + ray.direction * test.t;
	test.material = vec4f(1.0f,0.f,1.0f,1.f);
	test.normal = vec3(1.f,0.0f,0.f);
	return test;
}*/

fn test_renderLeafRopes(ray : Ray, leaf:LeafUBO) -> Hit{ //just to confirm that ray hits the box correctly
	//let rootAABB = &leaves[0];
	//var ff = RayBoxIntersection(ray,(*rootAABB).minAABB,(*rootAABB).maxAABB); //ff.x entry distane, ff.y exit distance
	var test = miss;
	var ffMain = RayBoxIntersection(ray,leaf.minAABB,leaf.maxAABB);
	let eps = 0.01f;
	if(ffMain.x != -999.f){
		test.t = ffMain.x;
	test.intersection = ray.origin + ray.direction * test.t;
	test.material = vec4f(1.0f,0.f,1.0f,1.f);
	}
	/*if(ff.y < 0.f){
		return miss;
	}*/

	//test.normal = vec3(1.f,0.0f,0.f);

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
		if(ff.x == -999.f){
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
	let eps = 0.01f;
	if(ff.x == -999.f || ff.y < 0.f){
		return miss;
	}
	
	var point = ray.origin;
	if(!(ff.x < 0.f)){// || (ff.x > 0.f && ff.y < 0.f)){  ??
		//point does NOT lie inside the box
		point += ray.direction *(ff.x + eps);
	}
	
	var currentNode = (locateLeaf(0,point));
	

	var dess =0;
	while(currentNode.leafId > 0){
				
		let leaf = leaves[currentNode.leafId];
		let firstIndex = leaf.firstIndex;
		let numberOfSpheres = leaf.numberOfSpheres;
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
			
			let hit = RaySphereIntersection(ray,  sphereId);
			closest_hit = getCloserHit(closest_hit,hit);
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
		point = ray.origin + ray.direction * (ff.y + eps);
		let rope = exitRope(ray,currentNode.leafId); 
		if(leaf.ropes[rope] == -1){ return miss;}
		currentNode = (locateLeaf(leaf.ropes[rope],point));
		
		
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



/*fn test_renderAllLeaves(ray :Ray) -> Hit{
	let root = kdTree[0];
	var closest_hit = miss;
	let rootAABB = leaves[0];
	var ff = RayBoxIntersection(ray,(rootAABB).minAABB,(rootAABB).maxAABB);
	let sceneMax = ff.y;
	let sceneMin = ff.x;
	
	var tMax = sceneMin;
	var tMin = sceneMin;
	for(var i = 0; i < 2000; i++){
		let hit = RaySphereIntersection(ray,  i);
		
		closest_hit = getCloserHit(closest_hit,hit);
	
	}
	return closest_hit;
}*/

fn traverseKdTree(ray :Ray) -> Hit{
	//return miss;
	return test_renderAllSpheres(ray);
	//return test_renderSmallCube(ray);
	//return test_renderAllLeaves(ray);
	//return test_renderUsingKdTree(ray);
	//return kdTreePushDown(ray);
	
	//return miss;

}

fn Trace(ray : Ray) -> vec3f{
    var light : Light;
	light.position = vec3f(0.0f,120.5f,10.0f);
	light.diffuse = vec3f(0.8f);
	// The accumulated color and attenuation used when tracing the rays throug the scene.



	var color = vec3f(0.0,0.0,0.0);
    var attenuation = vec3f(1.0);

	

	let epsilon = 0.001f;

	

	var tmpRay = ray;

	
	for(var i = 0;i<2;i++){

		let hit = Evaluate(tmpRay);
		var L = normalize(light.position -  hit.intersection);

		if (!isHitMiss(hit)) {

			let N = hit.normal;
			var V = normalize(-tmpRay.direction);
			let H = normalize(L + V);
			let NdotL = max(dot(N, L), 0.0);

			let ambient = 0.1*hit.material.xyz;
			//color += ambient;
			let shadowOrigin = hit.intersection +  epsilon * N;

			let shadowRay  = (Ray(shadowOrigin, L));
			
			//color += hit.material.xyz;//calculatePBR(ray,hit.intersection,N,hit.material.xyz);
			//continue;
			let shadowHit = Evaluate(shadowRay);

			
			
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

			//Refraction, TODO: lze optimalizovat, abych znovu nehledal kouli?
			//transmission ray
			
			/*	var refDir = normalize(refract(tmpRay.direction,N,1.00029f/1.4f));
				var refRay = Ray(hit.intersection +  epsilon * refDir, refDir);
				var refHit = Evaluate(refRay);

				color += refHit.material.xyz *  0.1f;

				//ref
				refDir = normalize(refract(refHit.intersection,refHit.normal,1.4f/1.00029f));
				refRay = Ray(refHit.intersection +  epsilon * refDir, refDir);
				refHit = Evaluate(refRay);
			
			

				V = normalize(-refDir);
				NdotV = dot(normalize(V+L),V);
				//let ref_F = schlickFresnel_refract(1.125f,1.00029f,NdotV);//schlickFresnel(NdotV,hit.material.xyz);//schlickFresnel_refract(1.125f,1.00029f,NdotV);
				//let ref_F_ref = 1.0 - F;

				color += F_ref * refHit.material.xyz * max(dot(refHit.normal, normalize(light.position -  refHit.intersection)), 0.0) * attenuation; */
			
			attenuation *= F;
			/*else{
				color +=ambient;
			}*/
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
		//break;
		/*if(length(uCamera.position-vec4f(hit.intersection,1.0f))>5.f){
			break;
		}*/
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

fn kdTreePushDown(ray :Ray) -> Hit{
	let root = kdTree[0];
	var closest_hit = miss;
	let rootAABB = leaves[0];
	var ff = RayBoxIntersection(ray,(rootAABB).minAABB,(rootAABB).maxAABB);
	let sceneMax = ff.y;
	let sceneMin = ff.x;
	
	var tMax = sceneMin;
	var tMin = sceneMin;
	var pushedNode = root;

	var pushdown = true;

	while(tMax<sceneMax){
		
		var currentNode = pushedNode;
		tMin = tMax;
		tMax = sceneMax;
		pushdown = true;
		while(currentNode.leafId < 1){
			let a = -(currentNode.leafId+1);
			let t = (currentNode.splitPoint - ray.origin[a]) / (ray.direction[a]);

			//let order = ray.direction[a] >= 0;
			if(t >= tMax){
				//near
				
				if(ray.direction[a] >= 0 ){
					currentNode = kdTree[currentNode.leftChild];
				}else{
					currentNode = kdTree[currentNode.rightChild];
				}
				/*var h =miss;
				h.t = 1.f;
				h.material = vec4f(0.f,0.f,1.f,1.f);
				return h;*/
			}else if (t <= tMin){
				//far
				if(ray.direction[a] >= 0){
					currentNode = kdTree[currentNode.rightChild];
				}else{
					currentNode = kdTree[currentNode.leftChild];
				}

			}else{
				//near
				if(ray.direction[a] >= 0){
					currentNode = kdTree[currentNode.leftChild];
				}else{
					currentNode = kdTree[currentNode.rightChild];
				}
				tMax = t;
				pushdown = false;
				/*var h =miss;
				h.t = 1.f;
				h.material = vec4f(1.f,0.f,0.f,1.f);
				return h;*/
			}
			if(pushdown){
				pushedNode = currentNode;
			}
		}

		let leaf = leaves[(currentNode).leafId];
		let firstIndex = leaf.firstIndex;
		let numberOfSpheres = leaf.numberOfSpheres;

		//return test_renderLeaf(ray,leaf.minAABB,leaf.maxAABB);
		//return test_renderLeaf(ray,leaf.minAABB,leaf.maxAABB);
		/*if(ff.x == -999.f || ff.y < 0.f){
			return miss;
		}*/
		for(var i = 0; i < numberOfSpheres; i++){
			/*if(sphereId < 0){
				break;
			}*/
			//let sphere = (spheres[firstIndex+i]);
		
			let hit = RaySphereIntersection(ray,  firstIndex+i);
			
			closest_hit = getCloserHit(closest_hit,hit);
			/*if(hit.t < closest_hit.t){
				closest_hit = hit;
				//closest_hit.material = vec4f(1.f,0.f,0.f,1.f);	
				
			}*/
			
		}
		if(closest_hit.t < tMax){
			return closest_hit;
		}
	}
	return closest_hit;
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