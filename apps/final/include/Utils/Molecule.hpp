#pragma once
#include "glm/glm.hpp"

struct SphereCPU {
	glm::vec3 origin;
	float radius;
	glm::vec4 color;
	int id;
};

struct SphereGPU {
	glm::vec3 origin; //xyz - position, w - type
	float radius;
	glm::vec4 color;
};