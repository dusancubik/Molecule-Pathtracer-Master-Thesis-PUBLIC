#pragma once
#include "kdTreeConstructionStrategy.hpp"

class MedianConstruction : public KdTreeConstructionStrategy<KdTreeNode_new>{
	public:
		std::vector<KdTreeNode_new*> constructTree(std::vector<SphereCPU*> spheres) override;

	private:
		std::vector<SphereCPU*> getExtendingSpheres(std::vector<SphereCPU*> spheres, glm::vec3 point, int dimension);
		std::tuple<glm::vec3, glm::vec3> computeBoundingPoint(bool isLeft, int dimenstionSplit, glm::vec3 splitPoint, glm::vec3 minP, glm::vec3 maxP);
};