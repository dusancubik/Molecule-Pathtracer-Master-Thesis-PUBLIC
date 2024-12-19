/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
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
#include "../Utils/Molecule.hpp"
#include <stack>
#include <vector>
#include "glm/glm.hpp"
#include <iostream>
#include <algorithm>

struct sortSpheresBVH
{
	int dim;
	inline bool operator() (const std::shared_ptr<SphereCPU> sphere1, const std::shared_ptr<SphereCPU> sphere2)
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
	std::vector<std::shared_ptr<SphereCPU>>leftSpheres;
	std::vector<std::shared_ptr<SphereCPU>> rightSpheres;
	bool leaf = false;
};

struct BVHConstructNode {
	int dimSplit = -1; //x=0, y=1, z=2
	float splitDistance;
	glm::vec3 minAABB;
	glm::vec3 maxAABB;
	bool isLeaf = false;
	std::vector<std::shared_ptr<SphereCPU>> spheres;
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
		~BVH();
		virtual void construct(std::vector<std::shared_ptr<SphereCPU>> spheres);

		std::vector<SphereGPU> getSpheresGPU() { return spheresGPU_SSBOs; }
		std::vector<BVHNodeSSBO> getBVHSSBOs() { return bvhSSBOs; }
	protected:
		int totalDepth = 0;
		void constructBVH(std::vector<std::shared_ptr<SphereCPU>> spheres);
		//void updateAABB(BVHConstructNode* node);
		void updateAABB(BVHConstructNode* node, std::vector<std::shared_ptr<SphereCPU>> spheres);
		BVHDivision subdivide(BVHConstructNode* node);
		BVHDivision subdivideSAH(BVHConstructNode* node);
		void updateBounds(glm::vec3& minAABB, glm::vec3& maxAABB, glm::vec3 minCentroid, glm::vec3 maxCentroid);

		glm::vec3 offset(glm::vec3 minAABB, glm::vec3 maxAABB, glm::vec3 centroid);
		float getSurfaceArea(glm::vec3 minAABB, glm::vec3 maxAABB);

		void tranformToSSBO();

		std::vector<BVHNodeSSBO> bvhSSBOs;
		std::vector<SphereGPU> spheresGPU_SSBOs;

		std::vector<BVHConstructNode*> BVHConstruct;
};