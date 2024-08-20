

//layout (local_size_x = 16,local_size_y = 16) in;
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

@compute @workgroup_size(8,8,1)
//fn main(@builtin(global_invocation_id) GlobalInvocationID:vec3<u32>){
fn main(@builtin(workgroup_id) WorkgroupID:vec3<u32>,@builtin(local_invocation_id) LocalInvocationID:vec3<u32>){

    let screen_size: vec2<u32> = textureDimensions(color_buffer);

    //let screen_pos : vec2<i32> = vec2<i32>(i32(GlobalInvocationID.x),i32(GlobalInvocationID.y));
	let screen_pos : vec2<i32> = vec2<i32>(i32(8*WorkgroupID.x + LocalInvocationID.x),i32(8*WorkgroupID.y + LocalInvocationID.y));

    let horizontal_coefficient: f32 = (f32(screen_pos.x) - f32(screen_size.x)/2) / f32(screen_size.x);
    let vertical_coefficient: f32 = (f32(screen_pos.y) - f32(screen_size.y)/2) / f32(screen_size.x);

    let forwards: vec3<f32> = vec3<f32>(0.0,0.0,1.0);
    let right: vec3<f32> = vec3<f32>(1.0,0.0,0.0);
    let up: vec3<f32> = vec3<f32>(0.0,-1.0,0.0);

    var mySphere: Sphere;
    mySphere.origin = vec3<f32>(3.0,0.0,0.0);
    mySphere.radius = 1.f;
    mySphere.color = vec4f(0.f,1.f,0.f,1.f);

    var ray : Ray;
    ray.direction = normalize(forwards + horizontal_coefficient * right + vertical_coefficient*up);
    ray.origin = vec3<f32>(0.,0.5,-10.f);


    var pixel_color: vec3<f32> = vec3<f32>(0.5,0.0,0.25);

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

	
	for(var i = 0;i<2;i++){

		let hit = Evaluate(tmpRay,LocalInvocationID);
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
		}

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
	//return stackless_popov(ray,LocalInvocationID);
	//return test_renderSmallCube(ray);
	//return test_renderAllLeaves(ray);
	return test_renderUsingKdTree(ray);
	//return kdTreePushDown(ray);
	
	//return miss;

}

var<workgroup> N_traversed : KdTreeNodeUBO;

fn stackless_popov(ray:Ray,LocalInvocationID:vec3<u32>) -> Hit{
	let eps = 0.001;
	var closest_hit = miss;
	let rootAABB = leaves[0];
	var ff = RayBoxIntersection(ray,(rootAABB).minAABB,(rootAABB).maxAABB); //ff.x entry distane, ff.y exit distance
	if(ff.x == -999.f || ff.y < 0.f){return miss;}
	var isActive = false;
	var point = ray.origin;
	if(!(ff.x < 0.f)){// || (ff.x > 0.f && ff.y < 0.f)){  ??
		//point does NOT lie inside the box
		point += ray.direction *(ff.x + eps);
		
		//closest_hit.material =  vec4f(0.f,0.f,1.f,1.f);

	}

	var N_current = kdTree[0];
	var N_currentID = 0;
	var j = 0;
	while(0==0 && j<10){
		if(ff.x >= ff.y){
			N_current = blankNode;
		}
		j++;
		N_traversed = P_MAX(N_current,LocalInvocationID,N_currentID);
		//workgroupBarrier();
		if(N_traversed.leftChild == -1 && N_traversed.rightChild == -1 && N_traversed.leafId<0){
			break;
		}
		var k = 0;
		while(N_traversed.leafId<1 && k<10){
			k++;
			if(N_traversed.leftChild == N_current.leftChild && N_traversed.rightChild == N_current.rightChild && N_traversed.leafId<0){
				var dimSplit = -(N_traversed.leafId+1);
				let left = point[dimSplit] < N_traversed.splitPoint;
				let i = select(N_traversed.rightChild,N_traversed.leftChild, left);
				//return blankNode;
				N_current = kdTree[i]; 
				N_currentID = i;
				//isActive = true;
			}
			let axis = -(N_current.leafId+1);
			let b = point[axis] < N_current.splitPoint;
			let b1 = P_OR(b && ray.direction[axis] > 0.0);
			
			let b2 = !P_OR(!b);
			let b3 = P_SUM(select(1,-1,point[axis] < N_current.splitPoint),LocalInvocationID) < 0;
			if(b1 || b2 || b3){
				N_traversed = kdTree[N_current.leftChild];
			}else{
				N_traversed = kdTree[N_current.rightChild];
			}
		}

		let leaf = leaves[N_traversed.leafId];
		let firstIndex = leaf.firstIndex;
		let numberOfSpheres = leaf.numberOfSpheres;
		for(var i = 0; i < numberOfSpheres; i++){
			let sphereId = i+firstIndex;
		
			
			let hit = RaySphereIntersection(ray,  sphereId);
			closest_hit = getCloserHit(closest_hit,hit);
			
		}

		if(closest_hit.t < miss.t){
			//closest_hit.material = vec4f(1.f,1.f,0.f,1.f);
			return closest_hit;
		}
		
		if(N_traversed.leftChild == N_current.leftChild && N_traversed.rightChild == N_current.rightChild && N_traversed.leafId<0){//TODO: leaf case
			ff = RayBoxIntersection(ray,(leaf).minAABB,(leaf).maxAABB);
			point = ray.origin + ray.direction * (ff.y + eps);
			let rope = exitRope(ray,leaf); 

			if(leaf.ropes[rope] == -1){ return miss;}
			let tmp = locateLeafWithId(leaf.ropes[rope],point);
			N_current = tmp.node;
			N_currentID = tmp.id;
		}

		//return miss;
	}


	return closest_hit;
}

var<workgroup> sharedCond : bool;
fn P_OR(condition : bool) -> bool{
	sharedCond = false;
	if(condition){
		sharedCond = true;
	}
	//workgroupBarrier();
	return sharedCond;
}

var<workgroup> m_maxAddr : array<i32,256>;
var<workgroup> m_max : array<KdTreeNodeUBO,256>;
fn P_MAX(value:KdTreeNodeUBO,LocalInvocationID:vec3<u32>,addr : i32) -> KdTreeNodeUBO{
	let pID = LocalInvocationID.x*16 + LocalInvocationID.y;
	m_max[pID] = value;
	m_maxAddr[pID] = addr;
	for(var i = 0;i<8;i++){
		let pw = u32(pow(f32(i),f32(2)));
		let a1 = (pw*2) * pID;
		let a2 = a1 + pw;
		if(a2 < 256){
			if(m_maxAddr[a1] < m_maxAddr[a2]){
				m_max[a1] = m_max[a2];
				m_maxAddr[a1] = m_maxAddr[a2];
			}
			//let mm = max(a1,a2);
			//m_max[a1] = m_max[mm]; 
		}
	}
	return m_max[0];
}

var<workgroup> m_sum : array<i32,256>;
fn P_SUM(value:i32,LocalInvocationID:vec3<u32>) -> i32{
	let pID = LocalInvocationID.x*16 + LocalInvocationID.y;
	m_sum[pID] = value;
	for(var i = 0;i<8;i++){
		let pw = i32(pow(f32(i),f32(2)));
		let a1 = (pw*2) * i32(pID);
		let a2 = a1 + pw;
		if(a2 < 256){
			m_sum[a1] = (m_sum[a1]+m_sum[a2]);
		}
	}
	return m_sum[0];
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

	//let tmin = max(max(ray_min.x,ray_min.y),ray_min.z);
	//let tmax = min(min(ray_max.x,ray_max.y),ray_max.z);

	var maxTmin = 9999999.f;
	var maxFace = -1;
	for(var i = 0;i<3;i++){
		if(ray_max[i] < maxTmin){
			maxTmin = ray_max[i];
			maxFace = i*2 + i32(ray.direction[i] > 0.0);
		}
	}

	return maxFace;
	//select(currentNode.rightChild,currentNode.leftChild, left);
	

	
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