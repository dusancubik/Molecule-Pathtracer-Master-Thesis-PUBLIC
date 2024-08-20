#pragma once
#include "../Pipelines/renderPipelineInterface.hpp"
#include <stack>
struct sortSpheresBVH
{
	int dim;
	inline bool operator() (const SphereCPU* sphere1, const SphereCPU* sphere2)
	{
		switch (dim) {
		case 0:
			return (sphere1->origin.x < sphere2->origin.x);
		case 1:
			return (sphere1->origin.y < sphere2->origin.y);
		case 2:
			return (sphere1->origin.z < sphere2->origin.z);
		}
		//return (sphere1.origin.x < sphere2.origin.x);
	}
};

struct BVHDivision {
	std::vector<SphereCPU*> leftSpheres;
	std::vector<SphereCPU*> rightSpheres;
	bool leaf = false;
};

struct BVHConstructNode {
	int dimSplit = -1; //x=0, y=1, z=2
	float splitDistance;
	glm::vec3 point;
	glm::vec3 minAABB;
	glm::vec3 maxAABB;
	bool isLeaf = false;
	std::vector<SphereCPU*> spheres;
	int depth = -1;
	bool left = false;
	int parent = -1;
	int id = -1;
};

struct BVHNodeSSBO {
	glm::vec3 minAABB;
	int leftChild = -1;
	glm::vec3 maxAABB;
	int numberOfSpheres = 0;
};

class BVH {
	public:
		virtual void construct(std::vector<SphereCPU*> spheres);
		
		std::vector<Sphere> getSpheres() { return sphereSSBOs;	}
		std::vector<BVHNodeSSBO> getBVHSSBOs() { return bvhSSBOs; }
	protected:
		int totalDepth = 0;
		void constructBVH(std::vector<SphereCPU*> spheres);
		//void updateAABB(BVHConstructNode* node);
		void updateAABB(BVHConstructNode* node, std::vector<SphereCPU*> spheres);
		BVHDivision subdivide(BVHConstructNode* node);
		BVHDivision subdivideSAH(BVHConstructNode* node);
		void updateBounds(glm::vec3& minAABB, glm::vec3& maxAABB, glm::vec3 minCentroid, glm::vec3 maxCentroid);

		glm::vec3 offset(glm::vec3 minAABB, glm::vec3 maxAABB, glm::vec3 centroid);
		float getSurfaceArea(glm::vec3 minAABB, glm::vec3 maxAABB);

		void tranformToSSBO();

		std::vector<BVHNodeSSBO> bvhSSBOs;
		std::vector<Sphere> sphereSSBOs;
	
		std::vector<BVHConstructNode*> BVHConstruct;
};