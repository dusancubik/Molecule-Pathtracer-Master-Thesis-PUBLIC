#pragma once
//#include "../Pipelines/renderPipelineInterface.hpp"
#include <vector>
#include "IKdTree.hpp"

struct KdTreeNodeSSBO {
	//glm::vec4 point = glm::vec4(glm::vec3(0.f), 1.f);
	//int dimSplit = -1;
	float splitPoint = 0.f;
	int leafId = -1; //0 == root, >0 leaves
	int leftChild = -1;
	int rightChild = -1; //?? použít aliagn?
};


template<typename N, typename L, typename S>

class KdTreeInterface :public IKdTree {
	public:
		virtual ~KdTreeInterface() {}
		//KdTree(std::vector<SphereCPU*> spheres);
		//virtual void construct(std::vector<SphereCPU*> spheres) = 0;

		virtual std::vector<N> getTreeSSBOs() { return {}; }

		virtual std::vector<L> getLeavesUBOs() { return {}; }

		virtual std::vector<S> getSphereSSBOs() { return {}; }

	/*protected:
		std::vector<N> treeSSBOs;

		std::vector<L> leavesUBOs;

		std::vector<S> sphereSSBOs;*/
};