#pragma once
#include "bvh.hpp"
#include "../float16/half.h"
struct BVH4ConstructNode : BVHConstructNode{
	bool child1 = false;
	bool child2 = false;
	bool child3 = false;
	bool child4 = false;
};

struct alignas(128) BVH4NodeSSBO {
	uint16_t bbox[2 * 4 * 3]; //12B * 8 = 96B
	//glm::vec3 bbox[8];
	int child[4]; // + 4*4 = 16B ... 112B
	int axis = -1; //todo: ostatní axis? //+4B .. 116B
	int numberOfSpheres = 0; //+4B .. 120B
};

struct BVH4Division {
	std::vector<SphereCPU*> spheres1;
	std::vector<SphereCPU*> spheres2;
	std::vector<SphereCPU*> spheres3;
	std::vector<SphereCPU*> spheres4;
	bool leaf = false;
};


class BVH4 : public BVH {
	public:
		void construct(std::vector<SphereCPU*> spheres) override;

		std::vector<Sphere> getSpheres() { return sphereSSBOs; }
		std::vector<BVH4NodeSSBO> getBVHSSBOs() { return bvh4SSBOs; }
	private:
		BVH4Division subdivideSAH(BVH4ConstructNode* node);
		BVH4Division subdivide(BVH4ConstructNode* node);
		void constructBVH4(std::vector<SphereCPU*> spheres);

		std::vector<BVH4ConstructNode*> BVH4Construct;
		std::vector<BVH4NodeSSBO> bvh4SSBOs;
		void tranformToSSBO();
};