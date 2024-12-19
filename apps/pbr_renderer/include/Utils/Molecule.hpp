#pragma once
#include "glm/glm.hpp"

struct SphereCPU {
	glm::vec3 origin;
	float radius;
	glm::vec4 color;
	int id;
};

struct SphereGPU {
	glm::vec4 origin; //xyz - position, w - material index
};