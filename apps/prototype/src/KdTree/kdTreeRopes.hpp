#pragma once
#include "Pipelines/renderPipelineInterface.hpp"
#include "ConstructionStrategy/kdTreeConstructionStrategy.hpp"
#include "ConstructionStrategy/sahRopesConstruction.hpp"
#include "ConstructionStrategy/medianRopesConstruction.hpp"
#include <vector>
#include <algorithm>
#include <tuple>
#include <stack>
#include "kdTreeInterface.hpp"


struct alignas(64) LeafRopesUBO {
	glm::vec3 minAABB = glm::vec3(0.f);
	int firstIndex;
	glm::vec3 maxAABB = glm::vec3(0.f);
	int numberOfSpheres;
	//0: left, 1: right, 2: bottom,3: top, 4: front,5: back
	int ropes[6] = { -1,-1,-1,-1,-1,-1 };
};




class KdTreeRopes : public KdTreeInterface<KdTreeNodeSSBO, LeafRopesUBO, Sphere> {
public:
	//KdTree();
	void construct(std::vector<SphereCPU*> spheres);
	
	std::vector<KdTreeNodeSSBO> getTreeSSBOs() { return treeSSBOs; }

	std::vector<LeafRopesUBO> getLeavesUBOs() { return leavesUBOs; }

	std::vector<Sphere> getSphereSSBOs() { return sphereSSBOs; }
private:
	std::vector<KdTreeNodeRopes*> constructTree(std::vector<SphereCPU*> spheres);
	void createTreeSsbos(std::vector<KdTreeNodeRopes*> tree);
	void addRopes(std::vector<KdTreeNodeRopes*> tree);
	void optimizeRopes(std::vector<KdTreeNodeRopes*> tree);
	void optimizeNodeRopes(KdTreeNodeRopes* node);
	int getSplitAxis(int face);

	std::tuple<int, int> getFaceIndexes(int dimension);
	void setRopes(KdTreeNodeRopes* node, int ropes[6]);

	std::vector<KdTreeNodeRopes*> tree_new;
	std::vector<KdTreeNodeSSBO> treeSSBOs;

	std::vector<LeafRopesUBO> leavesUBOs;

	std::vector<Sphere> sphereSSBOs;

	int spPerLeaf = 3;
	int maxDepth = -1;
};