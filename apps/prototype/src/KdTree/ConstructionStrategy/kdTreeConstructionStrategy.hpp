#pragma once
#include "Pipelines/renderPipelineInterface.hpp"
struct sortSpheres
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
struct KdTreeNode {
	int dimSplit = -1; //x=0, y=1, z=2
	float splitDistance;
	glm::vec3 point;
	glm::vec3 minAABB;
	glm::vec3 maxAABB;
	bool isLeaf = false;
	std::vector<Sphere> spheres;
	int depth = -1;
};

struct KdTreeNode_new {
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

struct KdTreeNodeRopes {
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
	//0: left, 1: right, 2: bottom,3: top, 4: front,5: back
	int ropes[6] = { -1,-1,-1,-1,-1,-1 };

	int leftChildId = -1;
	int rightChildId = -1;
};

struct SahDivision {
	std::vector<SphereCPU*> A;
	std::vector<SphereCPU*> B;
	int dimension = -1;
	float cost = -1.f;
	float splitPoint = 0.f;
};

template<typename T>
class KdTreeConstructionStrategy {
	public:
		virtual ~KdTreeConstructionStrategy() {}

		virtual std::vector<T*> constructTree(std::vector<SphereCPU*> spheres) = 0;
};