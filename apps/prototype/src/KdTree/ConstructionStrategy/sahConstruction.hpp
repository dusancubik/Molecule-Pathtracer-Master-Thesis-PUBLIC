#pragma once
#include "kdTreeConstructionStrategy.hpp"
#include <limits>


class SahConstruction : public KdTreeConstructionStrategy<KdTreeNode_new>{
public:
	std::vector<KdTreeNode_new*> constructTree(std::vector<SphereCPU*> spheres) override;

private:
	SahDivision performSAH(std::vector<SphereCPU*> spheres, KdTreeNode_new* node, int dimSplit); //todo: pointers
	SahDivision findBestHalfsSAH(std::vector<SphereCPU*> sortedSpheres, KdTreeNode_new* node);
	std::vector<SphereCPU*> getExtendingSpheres(std::vector<SphereCPU*> spheres, glm::vec3 point, int dimension);
	std::tuple<glm::vec3, glm::vec3> computeBoundingPoint(bool isLeft, int dimenstionSplit, glm::vec3 splitPoint, glm::vec3 minP, glm::vec3 maxP);
};