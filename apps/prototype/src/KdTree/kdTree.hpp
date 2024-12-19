/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: BVH.hpp
 *
 *  Description:
 *  The KdTree class is responsible for constructing Kd-tree.
 *
 *
 * -----------------------------------------------------------------------------
 */
#pragma once
#include "Pipelines/renderPipelineInterface.hpp"
#include "ConstructionStrategy/kdTreeConstructionStrategy.hpp"
#include "ConstructionStrategy/medianConstruction.hpp"
#include "ConstructionStrategy/sahConstruction.hpp"
#include <vector>
#include <algorithm>
#include <tuple>
#include <stack>
#include "kdTreeInterface.hpp"


struct LeafUBO {

	//int spheresIdx[4] = { -1,-1,-1,-1 };
	glm::vec3 minAABB = glm::vec3(0.f);
	int firstIndex;
	glm::vec3 maxAABB = glm::vec3(0.f);
	int numberOfSpheres;
};

struct KdTreeNodeUBO {
	//int dimSplit; //x=0, y=1, z=2
	//float splitDistance;
	glm::vec4 point = glm::vec4(glm::vec3(0.f), 1.f);
	int dimSplit = -1;
	int depth = -1;
	int leafId = -1; //0 == root, >0 leaves
	int padd = -1; //?? použít aliagn?
	
};



class KdTree : public KdTreeInterface<KdTreeNodeSSBO, LeafUBO, Sphere> {
	public:
		//KdTree();
		void construct(std::vector<SphereCPU*> spheres);
		std::vector<KdTreeNode> getTree() { return tree; }
		std::vector<KdTreeNodeUBO> getTreeUBOs() { return treeUBOs; }

		//std::vector<KdTreeNode> getTreeNew() { return tree_new; }
		std::vector<KdTreeNodeSSBO> getTreeSSBOs() { return treeSSBOs; }

		std::vector<LeafUBO> getLeavesUBOs() { return leavesUBOs; }

		std::vector<Sphere> getSphereSSBOs() { return sphereSSBOs; }
	private:
		void constructTree(std::vector<Sphere> spheres);
		void createTreeUbos(std::vector<KdTreeNode> tree);
		std::tuple<glm::vec3, glm::vec3> computeBoundingPoint(bool isLeft,int dimenstionSplit, glm::vec3 splitPoint, glm::vec3 minP, glm::vec3 maxP);

		void constructTree_new(std::vector<Sphere> spheres);
		void createTreeSsbos(std::vector<KdTreeNode_new*> tree);

		std::vector<KdTreeNode> tree;
		std::vector<KdTreeNodeUBO> treeUBOs;

		std::vector<KdTreeNode_new*> tree_new;
		std::vector<KdTreeNodeSSBO> treeSSBOs;

		std::vector<LeafUBO> leavesUBOs;

		std::vector<Sphere> sphereSSBOs;

		int spPerLeaf = 3;
		int maxDepth = -1;


		std::vector<Sphere> getExtendingSpheres(std::vector<Sphere> spheres, glm::vec3 point, int dimension);
};