struct Sphere{
	origin: vec3f,
	radius: f32,
	color: vec4f
};

/*struct Spheres{
	sphere: array<Sphere>;	
};*/
@group(0) @binding(1) var<storage> spheres: array<Sphere>;
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
	material:vec4f,			  // The material of the object at the intersection point.
	sphere:Sphere //the actual sphere TODO: think if necessecary or use pointer
};
const miss = Hit(vec3f(0.0f), 1e20, vec3f(0.0f), vec4f(0.f),Sphere(vec3f(0.f),0.f,vec4f(0.f)));
const PI = 3.14159265;
const roughnessConst = 0.5f;
const metallic = true;
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


/*fn CastShadowRay(ray : Ray, normal:vec3f, point:vec3f) -> Hit {
    let nd = dot(normal, ray.direction);
    
    // Check if ray is nearly parallel to the plane
    if (abs(nd) < 1e-6) {
        return miss;
    }

    let sp = point - ray.origin;
    let t = dot(sp, normal) / nd;

    if (t < 0.0) { 
        return miss; 
    }

    let intersection = ray.origin + t * ray.direction;

    // circle
    if(length(intersection) > 8){
        return miss;
    }

    return Hit(intersection, t, normal, vec3f(0.5f));
}*/

fn RayPlaneIntersection(ray : Ray, normal:vec3f, point:vec3f) -> Hit {
    let nd = dot(normal, ray.direction);
    
    // Check if ray is nearly parallel to the plane
    if (abs(nd) < 1e-6) {
        return miss;
    }

    let sp = point - ray.origin;
    let t = dot(sp, normal) / nd;

    if (t < 0.0) { 
        return miss; 
    }

    let intersection = ray.origin + t * ray.direction;

    // circle
    if(length(intersection) > 8){
        return miss;
    }

    return Hit(intersection, t, normal, vec4f(0.5f),Sphere(vec3f(0.f),0.f,vec4f(0.f)));
}


fn RaySphereIntersection(ray : Ray, sphere : Sphere) -> Hit{
	let rayOrigin = ray.origin - sphere.origin;
	let rayDirection = ray.direction;
	let a = dot(rayDirection,rayDirection);
	let b = 2.0f * dot(rayOrigin,rayDirection);
	let c = dot(rayOrigin,rayOrigin) - sphere.radius * sphere.radius;

	let disc = b * b - 4.0f * a * c;
	let t = (-b - sqrt(disc))/(2.0f * a);

		
	if(disc>=0.0f && t>=0){

		let hitPoint = rayOrigin + rayDirection * t;
		//var n = hitPoint-sphere.origin;
		var n = normalize(hitPoint);
		let r = normalize(ray.direction);
		// Check if the ray is inside the sphere
        if (dot(r, n) > 0) {
            n = n; // Invert the normal
        }
		return Hit(hitPoint+sphere.origin, t, n/*select(n, -1*n, dot(r,n)>0)*/,sphere.color,sphere);
	}else{
		return miss;
	}
}

fn Evaluate(ray :Ray) -> Hit{
	// Sets the closes hit either to miss
	var closest_hit = RayPlaneIntersection(ray, vec3f(0.0f, 1.f, 0.f), vec3f(0.0f,0.0f,0.0f));

	
	for(var i = 0; i < 200; i++){
		var hit = RaySphereIntersection(ray, spheres[i]);
		if(hit.t < closest_hit.t){
			closest_hit = hit;
		}
	}

    return closest_hit;
}

fn schlickFresnel(vDotH : f32,color:vec3f) -> vec3f{
	var F0 = vec3f(0.04);
	//TODO: if is metal

	if(metallic){
		F0 = color;
	}

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


fn calculatePBR(_cameraRay:Ray,localP:vec3f,normal:vec3f,color:vec3f) -> vec3f{
	var cameraRay = _cameraRay;
	//cameraRay.direction = localP - cameraRay.origin;
	var light : Light;
	light.position = vec3f(0.0f,1.1f,0.0f);
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

fn Trace(ray : Ray) -> vec3f{
    var light : Light;
	light.position = vec3f(0.0f,20.5f,0.0f);
	light.diffuse = vec3f(1.f);
	// The accumulated color and attenuation used when tracing the rays throug the scene.



	var color = vec3f(0.0);
    var attenuation = vec3f(1.0);

	

	// Due to floating-point precision errors, when a ray intersects geometry at a surface, the point of intersection could possibly be just below the surface.
	// The subsequent reflection and shadow rays would then bounce off the *inside* wall of the surface. This is known as self-intersection.
	// We, therefore, use a small epsilon to offset the subsequent rays.
	let epsilon = 0.01f;

	// ----------------- TODO: Write your implementation here! ----------------- //
	// Add difuse lighting, shadows and skybox.
	// Hints: The skybox texture is stored in 'skybox_tex'.
	//        You can keep a bit of color in areas with shadows (e.g., 0.2*hit.material).
	// ------------------------------------------------------------------------- //
	let hit = Evaluate(ray);

	// The direction towards the light.
	var L = normalize(light.position -  hit.intersection);

	if (!isHitMiss(hit)) {
		let N = hit.normal;
		//return calculatePBR(ray,hit.intersection,N,hit.material);//ambient + 0.8*hit.material*NdotL *light.diffuse; 
		let NdotL = max(dot(N, L), 0.0);

		let ambient = 0.1*hit.material.xyz;
		//let shadowOrigin = hit.sphere.origin + (hit.sphere.radius+epsilon) * N ;//hit.intersection +  epsilon * N;
		let shadowOrigin = hit.intersection +  epsilon * N;

		let shadowRay = Ray(shadowOrigin, L);
		let shadowHit = Evaluate(shadowRay);
		//color =  ambient + 0.8*hit.material*NdotL * light.diffuse; 
		if(!isHitMiss(shadowHit) && dot(shadowRay.direction,N)>0.0f){
			color = ambient;//+vec3f(0.f,0.f,1.f);
			/*if(dot(shadowRay.direction,N)<0.0f){
				color = 1.5*ambient;
			}else{
				color =  ambient + 0.8*hit.material*NdotL *light.diffuse; 
			}*/
		}else{
			color = ambient+calculatePBR(ray,hit.intersection,N,hit.material.xyz);//ambient + 0.8*hit.material*NdotL *light.diffuse; 
		}
			
		//return normalize(shadowRay.direction);
	}else{
		color = vec3f(0.2f);
	}

    return color;
}

fn isHitMiss(hit:Hit) -> bool{
	if(hit.intersection.x != miss.intersection.x && hit.intersection.y != miss.intersection.y && hit.intersection.z != miss.intersection.z ){
		return false;
	}
	if(hit.t != miss.t){
		return false;
	}
	if(hit.normal.x != miss.normal.x && hit.normal.y != miss.normal.y && hit.normal.z != miss.normal.z ){
		return false;
	}
	if(hit.material.x != miss.material.x && hit.material.y != miss.material.y && hit.material.z != miss.material.z ){
		return false;
	}
	return true;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
	//ProjectionView
	let aspect_ratio = 1280.0/720.0;
	let uv = (2.0*in.tex_coord-1.0)* vec2f(aspect_ratio, 1.0);// * vec2f(aspect_ratio, 1.0);
	let P = (uCamera.inversePV * vec4f(uv, -1.0, 1.0)).xyz;//vec3f(0.0f,0.0f,2.0f);//(uCamera.inversePV * vec4f(uv, -1.0, 1.0)).xyz;
	let direction = normalize(P - uCamera.position.xyz);
	
	let ray = Ray(P, direction);
	let color = Trace(ray);
	return vec4f(color, 1.0);
}