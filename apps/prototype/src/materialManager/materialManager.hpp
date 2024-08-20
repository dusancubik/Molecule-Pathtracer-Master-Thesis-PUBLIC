#pragma once

struct Material {
	glm::vec3 albedo;
	float radius;
};

class MaterialManager {
	public:
		vector<Material> getMaterials() { return materials; }
		void createNewMaterial(glm::vec3 albedo, float radius) { materials.push_back(Material(albedo,radius)); }
	private:
		vector<Material> materials;
};