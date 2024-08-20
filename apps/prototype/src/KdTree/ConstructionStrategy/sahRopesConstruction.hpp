#pragma once
#include "kdTreeConstructionStrategy.hpp"
#include "../kdTreeRopes.hpp"
#include <limits>



class SahRopesConstruction : public KdTreeConstructionStrategy<KdTreeNodeRopes> {
public:
	std::vector<KdTreeNodeRopes*> constructTree(std::vector<SphereCPU*> spheres) override;

private:
	SahDivision performSAH(std::vector<SphereCPU*> spheres, KdTreeNodeRopes* node, int dimSplit); //todo: pointers
	SahDivision findBestHalfsSAH(std::vector<SphereCPU*> sortedSpheres, KdTreeNodeRopes* node);
	std::vector<SphereCPU*> getExtendingSpheres(std::vector<SphereCPU*> spheres, glm::vec3 point, int dimension);
	std::tuple<glm::vec3, glm::vec3> computeBoundingPoint(bool isLeft, int dimenstionSplit, glm::vec3 splitPoint, glm::vec3 minP, glm::vec3 maxP);

	std::tuple<int, int> getFaceIndexes(int dimension);
	void setRopes(KdTreeNodeRopes* node, int ropes[6]);
};