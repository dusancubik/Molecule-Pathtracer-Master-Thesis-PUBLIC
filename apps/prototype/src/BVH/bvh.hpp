/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: BVH.hpp
 *
 *  Description:
 *  The BVH class is responsible for constructing Bounding Volume Hierarchy structure.
 *
 *  Functions subdivideSAH(), updateBounds(), offset() and getSurfaceArea()
 *  are based on Section 4.3 Bounding Volume Hierarchies from the book Physically Based Rendering: From Theory to Implementation
 *  (https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies).
 * -----------------------------------------------------------------------------
 */
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

struct PROTO_BVHDivision {
	std::vector<SphereCPU*> leftSpheres;
	std::vector<SphereCPU*> rightSpheres;
	bool leaf = false;
};

struct PROTO_BVHConstructNode {
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

class PROTO_BVH {
	public:
		virtual void construct(std::vector<SphereCPU*> spheres);
		
		std::vector<Sphere> getSpheres() { return sphereSSBOs;	}
		std::vector<BVHNodeSSBO> getBVHSSBOs() { return bvhSSBOs; }
	protected:
		int totalDepth = 0;
		void constructBVH(std::vector<SphereCPU*> spheres);
		//void updateAABB(BVHConstructNode* node);
		void updateAABB(PROTO_BVHConstructNode* node, std::vector<SphereCPU*> spheres);
		PROTO_BVHDivision subdivide(PROTO_BVHConstructNode* node);
		PROTO_BVHDivision subdivideSAH(PROTO_BVHConstructNode* node);
		void updateBounds(glm::vec3& minAABB, glm::vec3& maxAABB, glm::vec3 minCentroid, glm::vec3 maxCentroid);

		glm::vec3 offset(glm::vec3 minAABB, glm::vec3 maxAABB, glm::vec3 centroid);
		float getSurfaceArea(glm::vec3 minAABB, glm::vec3 maxAABB);

		void tranformToSSBO();

		std::vector<BVHNodeSSBO> bvhSSBOs;
		std::vector<Sphere> sphereSSBOs;
	
		std::vector<PROTO_BVHConstructNode*> BVHConstruct;
};