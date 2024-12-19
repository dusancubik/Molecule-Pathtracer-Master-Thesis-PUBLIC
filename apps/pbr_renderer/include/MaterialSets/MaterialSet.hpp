/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: MaterialSet.hpp
 *
 *  Description:
 *  The Material struct stores information about a material's base color in the xyz components 
 *  and its roughness in the w component of a vector.
 *  The MaterialSet class serves as a container for managing multiple Material structs.
 * -----------------------------------------------------------------------------
 */
#pragma once
#include"glm/glm.hpp"
#include"glm/ext.hpp"
#include <vector>
struct Material {
	glm::vec4 colour;
};

class MaterialSet {
	std::vector<Material> materials;

	public:
		std::vector<Material> getMaterials() { return materials; }
		void addMaterial(Material material);
};