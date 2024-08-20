
@group(0) @binding(0) var<uniform> uCamera: Camera;
const tex_coords = array<vec2f,3>(
	vec2f(0.0, 0.0),
	vec2f(2.0, 0.0),
	vec2f(0.0, 2.0)
);
//layout (local_size_x = 16,local_size_y = 16) in;

@group(0) @binding(1) var<storage,read> bvh: array<BVH4Node>;
@group(0) @binding(2) var<storage,read> spheres: array<Sphere>;
//@group(0) @binding(3) var<storage,read> materials: array<Material>;


struct Light{
	position:vec3f,
	direction:vec3f,
	diffuse:vec3f
}

struct BVH4Node { 
	bbox: array<f16,8*3>, //96
	child: array<i32,4>,
	//axis : i32,
	numberOfSpheres : i32
};

struct Sphere{
	origin: vec4f
	/*origin: vec3f,
	radius: f32,
	color: vec4f*/
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
	
	let ray = Ray(uCamera.position.xyz, direction);
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

	
	for(var i = 0;i<2;i++){

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
			let newRay = Ray(hit.intersection +  epsilon * N, reflected);
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
	var node: BVH4Node = bvh[nodeId];
    var stack: array<BVH4Node, 4>;
    var stackLocation: u32 = 0;
	var closest_hit = miss;
	var j = 0;
	//let main = RayBoxIntersection(ray,node.minAABB,node.maxAABB);

	
	 while (true) {
		//j++;
        
		
        if (node.child[0] <= 0) {
			
				/*var h = miss;
				h.material = vec4(0.f,0.f,1.f,1.f);
				h.t = 10.f;
				return h;*/
			
            for (var i = 0; i < node.numberOfSpheres; i++) {
				let sphereId = i+(-1*node.child[0]);
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
        //var child1: BVH4Node = bvh[node.child[0]];
		//var child2: BVH4Node = bvh[node.child[0]+1];
		//var child3: BVH4Node = bvh[node.child[0]+2];
		//var child4: BVH4Node = bvh[node.child[0]+3];

		//var distance1 = RayBoxIntersection(ray,vec3f(node.bbox[0],node.bbox[1],node.bbox[2]),vec3f(node.bbox[3],node.bbox[4],node.bbox[5])).x;//hit_aabb(ray, child1);
		//var distance2 = RayBoxIntersection(ray,vec3f(node.bbox[6],node.bbox[7],node.bbox[8]),vec3f(node.bbox[9],node.bbox[10],node.bbox[11])).x;
		var distance3 = RayBoxIntersection(ray,vec3f(node.bbox[12],node.bbox[13],node.bbox[14]),vec3f(node.bbox[15],node.bbox[16],node.bbox[17])).x;
		var distance4 = RayBoxIntersection(ray,vec3f(node.bbox[18],node.bbox[19],node.bbox[20]),vec3f(node.bbox[21],node.bbox[22],node.bbox[23])).x;
		
		//SORT
		/*if(distance1 > distance2){
			var tmp = distance1;
			distance1 = distance2;
			distance2 = tmp;
		}*/

		if(distance3 > distance4){
			var tmp = distance3;
			distance3 = distance4;
			distance4 = tmp;
		}
		

		if(distance3 == missT){//if miss
			if (stackLocation == 0) {

                break;
            }
            else {

                stackLocation -= 1;
                node = stack[stackLocation];
            }
		}else{
			node = bvh[node.child[0]+2];
			if(distance4 != missT){
				//stackLocation++;
				stack[stackLocation] = bvh[node.child[0]+3];
				stackLocation++;
			}
			/*if(distance3 != missT){
				//stackLocation++;
				stack[stackLocation] = bvh[node.child[0]+2];
				stackLocation++;
			}
			if(distance4 != missT){
				//stackLocation++;
				stack[stackLocation] = bvh[node.child[0]+3];
				stackLocation++;
			}*/
		}
		
		var h = miss;
		h.material = vec4(1.,0.,0.,1.);
		h.t = 10.;
		return h;
		//j++;
    }
	/*closest_hit.material = vec4f(abs(node.minAABB.x/400.f),0.f,0.f,1.f);//vec4f(0.f,1.f,0.f,1.f);
	closest_hit.t = 10.f;*/

	return closest_hit;
}

