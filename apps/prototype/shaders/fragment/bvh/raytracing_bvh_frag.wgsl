
@group(0) @binding(0) var<uniform> uCamera: Camera;
const tex_coords = array<vec2f,3>(
	vec2f(0.0, 0.0),
	vec2f(2.0, 0.0),
	vec2f(0.0, 2.0)
);
//layout (local_size_x = 16,local_size_y = 16) in;

@group(0) @binding(1) var<storage,read> bvh: array<BVHNode>;
@group(0) @binding(2) var<storage,read> spheres: array<Sphere>;
//@group(0) @binding(3) var<storage,read> materials: array<Material>;


struct Light{
	position:vec3f,
	direction:vec3f,
	diffuse:vec3f
}

struct BVHNode {
	minAABB: vec3f,
	leftChild: i32,
	maxAABB: vec3f,
	numberOfSpheres: i32
};

struct Sphere{
	//origin: vec4f
	origin: vec3f,
	radius: f32,
	color: vec4f
};

struct Ray {
    origin:vec3f,     // The ray origin.
    direction:vec3f,  // The ray direction.
	octant: array<i32,3>,
	dir_inv:vec3f,
	neg_org_div_dir:vec3f
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
//const blankNode = KdTreeNodeUBO(-1.f,-1,-1,-1); 
struct Camera {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
	position: vec4f,
	inversePV: mat4x4f
};
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {

	//ProjectionView
	let aspect_ratio = 1280.0/720.0;
	let uv = (2.0*in.tex_coord-1.0)* vec2f(aspect_ratio, 1.0);// * vec2f(aspect_ratio, 1.0);
	let P = (uCamera.inversePV * vec4f(uv, -1.f, 1.0)).xyz;

	let direction = normalize(P - uCamera.position.xyz);
	
	let oct : array<i32,3> = array<i32,3>(select(0,3,direction[0]<0),select(0,3,direction[1]<0),select(0,3,direction[2]<0));
	let dir_inv = vec3f(1.0)/direction;
	let neg_inv = -uCamera.position.xyz *dir_inv;
	let ray = Ray(uCamera.position.xyz, direction,oct,dir_inv,neg_inv);
	let color = Trace(ray); 
	return vec4f(color, 1.0);
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



fn Trace(ray : Ray) -> vec3f{
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

			let oct : array<i32,3> = array<i32,3>(select(0,3,L[0]<0),select(0,3,L[1]<0),select(0,3,L[2]<0));
			let dir_inv = vec3f(1.0)/L;
			let neg_inv = -shadowOrigin * dir_inv;

			let shadowRay  = (Ray(shadowOrigin, L,oct,dir_inv,neg_inv));
			
			/*color += hit.material.xyz;//calculatePBR(ray,hit.intersection,N,hit.material.xyz);
			let reflected1 = reflect(tmpRay.direction,hit.normal);
			let newRay1 = Ray(hit.intersection +  epsilon * N, reflected1);
			tmpRay = newRay1;
			continue;*/
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

			
			
			attenuation *= F;
			/*else{
				color +=ambient;
			}*/
			let reflected = reflect(tmpRay.direction,hit.normal);
			let newOr = hit.intersection +  epsilon * N;
			let oct2 : array<i32,3> = array<i32,3>(select(0,3,reflected[0]<0),select(0,3,reflected[1]<0),select(0,3,reflected[2]<0));
			let dir_inv2 = vec3f(1.0)/reflected;
			let neg_inv2 = -newOr * dir_inv2;

			let newRay = Ray(newOr, reflected,oct2,dir_inv2,neg_inv2);
			tmpRay = newRay;
		}

	}
	

    return color;
}

fn Evaluate(ray : Ray) -> Hit{

	var closest_hit = miss;//RayPlaneIntersection(ray, vec3f(0.0f, 1.f, 0.f), vec3f(0.0f,0.0f,0.0f));
	
	var kdResultHit = traverseKdTree(ray);
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
fn traverseKdTree(ray :Ray) -> Hit{
	//return miss;
	//return test_renderAllSpheres(ray);
	return traverseBVH(ray);
	//return traverseBVH_trail(ray);
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

fn RayBoxIntersection_new(ray : Ray, minP : vec3f, maxP : vec3f) -> vec2f{ 

	let tmin0 = minP[0+ray.octant[0]] * ray.dir_inv[0] + ray.neg_org_div_dir[0];
	let tmin1 = minP[1+ray.octant[1]] * ray.dir_inv[1] + ray.neg_org_div_dir[1];
	let tmin2 = minP[2+ray.octant[2]] * ray.dir_inv[2] + ray.neg_org_div_dir[2];

	let tmax0 = maxP[0+ray.octant[0]] * ray.dir_inv[0] + ray.neg_org_div_dir[0];
	let tmax1 = maxP[1+ray.octant[1]] * ray.dir_inv[1] + ray.neg_org_div_dir[1];
	let tmax2 = maxP[2+ray.octant[2]] * ray.dir_inv[2] + ray.neg_org_div_dir[2];

	let tmin = max(max(tmin0,tmin1),tmin2);
	let tmax = min(min(tmax0,tmax1),tmax2);

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
	let main = RayBoxIntersection_new(ray,node.minAABB,node.maxAABB);

	 while (true) {
        var i = 0;
		while(node.leftChild > 0){
			i++;
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
		}
        
			
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
			
        
        
    }
	/*closest_hit.material = vec4f(abs(node.minAABB.x/400.f),0.f,0.f,1.f);//vec4f(0.f,1.f,0.f,1.f);
	closest_hit.t = 10.f;*/

	return closest_hit;
}

/*fn traverseBVH(ray:Ray) -> Hit{
	var nodeId = 0;
	var node: BVHNode = bvh[nodeId];
    var stack: array<BVHNode, 32>;
    var stackLocation: u32 = 0;
	var closest_hit = miss;
	var j = 0;
	let main = RayBoxIntersection_new(ray,node.minAABB,node.maxAABB);

	 while (true) {
		//j++;
        
		
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
}*/

fn traverseBVH_trail(ray:Ray) -> Hit{
	var nodeId = 0;
	var node: BVHNode = bvh[nodeId];
	
    var shortStack: array<BVHNode, 8>;
    var stackLocation: i32 = 0;
	var closest_hit = miss;
	var j = 0;
	
	var trail : u32 = 0;
	var level : u32 = 0;
	var popLevel : u32 = 9999999;

	while(true && j<20){
		j++;
		var i = 0;
		while(node.leftChild > 0 && i<20){
			i++;
			var nearChild: BVHNode = bvh[node.leftChild];
			var farChild: BVHNode = bvh[node.leftChild+1];

			var nearDistance = RayBoxIntersection(ray,nearChild.minAABB,nearChild.maxAABB).x;
			var farDistance = RayBoxIntersection(ray,farChild.minAABB,farChild.maxAABB).x;

			if(nearDistance != missT && farDistance != missT){
				if(nearDistance>farDistance){
					let tmp = nearChild;
					nearChild = farChild;
					farChild = tmp;
				}
				level++;
				if( firstLeadingBit(level) != 0){
					node = farChild;
				}else{
					node = nearChild;
					//push far
					shortStack[stackLocation] = node;
					stackLocation++;
				}
			}
			else if(nearDistance == missT && farDistance != missT){
				level++;
				if(level != popLevel){
					let shift : u32 =  u32(1<<(31-level));
					level = level | shift;
					node = farChild;
				}else{
					//pop
					trail = trail & ~(level);
					trail = trail + level;

					let temp = trail >> 1;
					level = (((temp-1)^temp)+1);

					if(level>>31 == 1){
						//terminate
						return miss;
					}
					popLevel = level;
					if(stackLocation == 8){ //exhaseusted
						node = bvh[0];
						level = 0;
						stackLocation = 0;
					}else{
						stackLocation--;
						if (stackLocation == -1) {
							return miss;
						}
						node = shortStack[stackLocation];
					}
				}
			}else if(nearDistance != missT && farDistance == missT){
				level++;
				if(level != popLevel){
					let shift : u32 =  u32(1<<(31-level));
					level = level | shift;
					node = nearChild;
				}else{
					//pop
					trail = trail & ~(level);
					trail = trail + level;

					let temp = trail >> 1;
					level = (((temp-1)^temp)+1);

					if(firstLeadingBit(level) == 0){
						//terminate
						return miss;
					}
					popLevel = level;
					if(stackLocation == 8){ //exhaseusted
						node = bvh[0];
						level = 0;
						stackLocation = 0;
					}else{
						stackLocation--;
						if (stackLocation == -1) {
							return miss;
						}
						node = shortStack[stackLocation];
					}
				}
			}else{
				//POP
				trail = trail & ~(level);
				trail = trail + level;

				let temp = trail >> 1;
				level = (((temp-1)^temp)+1);

				if(firstLeadingBit(level) == 0){
					//terminate
					return miss;
				}
				popLevel = level;
				if(stackLocation == 8){ //exhaseusted
					node = bvh[0];
					level = 0;
					stackLocation = 0;
				}else{
					stackLocation--;
					if (stackLocation == -1) {
						return miss;
					}
					node = shortStack[stackLocation];
				}
			}
			
		}

		for (var i = 0; i < node.numberOfSpheres; i++) {
			let sphereId = i+(-1*node.leftChild);
			var hit = RaySphereIntersection(ray, sphereId);
			//hit.t = -10. * spheres[sphereId].radius;
			if(hit.t < closest_hit.t){
				closest_hit = hit;
			}
		}
		if(closest_hit.t< missT){
			return closest_hit;
		}

		//pop

		trail = trail & ~(level);
		trail = trail + level;

		let temp = trail >> 1;
		level = (((temp-1)^temp)+1);

		if(firstLeadingBit(level) == 0){
			//terminate
			return miss;
		}
		popLevel = level;
		if(stackLocation == 8){ //exhaseusted
			node = bvh[0];
			level = 0;
			stackLocation = 0;
		}else{
			stackLocation--;
			if (stackLocation == -1) {
                return miss;
            }
			node = shortStack[stackLocation];
		}
		

	}
	return closest_hit;
}

/*fn pop( level : ptr<function,i32>,
		popLevel : ptr<function,i32>,
		trail : ptr<function,i32>,
		shortStack : ptr<function,array<BVHNode, 16>>,
		stackLocation : ptr<function,i32>,
		node : ptr<function,BVHNode>
		)-> bool{//false = terminate
	trail = trail & ~(level);
	trail = trail + level;

	let temp = trail >> 1;
	level = (((temp-1)^temp)+1);

	if(level>>31 == 1){
		//terminate
		return false;
	}
	popLevel = level;
	if(stackLocation == 31){ //exhaseusted
		node = bvh[0];
		level = 0;
	}else{
		stackLocation--;
		node = shortStack[stackLocation];
	}
	return true;
}

fn pushStack(node : ptr<function,BVHNode>,shortStack : ptr<function,array<BVHNode, 16>>,stackLocation : ptr<function,i32>){
		&shortStack[&stackLocation] = &node;
		stackLocation++;
}


fn popStack(shortStack : ptr<function,array<BVHNode, 16>>,stackLocation : ptr<function,i32>)->BVHNode{
		stackLocation--;
		return &shortStack[&stackLocation];
}*/