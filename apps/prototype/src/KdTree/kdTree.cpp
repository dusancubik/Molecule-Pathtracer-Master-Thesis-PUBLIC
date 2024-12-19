#include "kdTree.hpp"
#include "ConstructionStrategy/medianConstruction.cpp"
#include "ConstructionStrategy/sahConstruction.cpp"


void KdTree::construct(std::vector<SphereCPU*> spheres) {
	std::cout << "construct\n";
	MedianConstruction medianStrategy;
	SahConstruction sahStrategy;
	tree_new = sahStrategy.constructTree(spheres);
	createTreeSsbos(tree_new);
}

void KdTree::createTreeSsbos(std::vector<KdTreeNode_new*> tree) {
	std::cout << "constructing tree UBOs\n";

	//std::vector<KdTreeNodeUBO> tree

	int leavesIndex = 1;

	for (int i = 0;i < tree.size();i++) {
		KdTreeNode_new* node = tree[i];
		//if (node->depth < 0) break;
		KdTreeNodeSSBO nodeSSBO;

		if (i == 0) {
			LeafUBO leafUBO; //is not actually leaf, just root
			leafUBO.minAABB = node->minAABB;//glm::vec4(node->minAABB, 1.f); //necessary for bounding box
			leafUBO.maxAABB = node->maxAABB;//glm::vec4(node->maxAABB, 1.f);
			leavesUBOs.push_back(leafUBO);
			leafUBO.firstIndex = 0;
			leafUBO.numberOfSpheres = 0;
			nodeSSBO.leafId = 0;

		}

		if (node->id >= tree.size()) {
			std::cout << "xxx\n";
		}

		if (node->dimSplit != -1) {

			//nodeUBO.meta = glm::vec4(static_cast<float>(node->dimSplit), static_cast<float>(node->depth), node->isLeaf ? 1.f : 0.f, 1);
			//nodeSSBO.dimSplit = node->dimSplit;
			//nodeSSBO.depth = node->depth;
			if (i != 0) {
				if (node->left) {
					treeSSBOs[node->parent].leftChild = node->id;

				}
				else {
					treeSSBOs[node->parent].rightChild = node->id;
				}
				//treeSSBOs[node->parent].
			}

			//nodeUBO.minAABB = glm::vec4(node->minAABB, 1.f);
			//nodeUBO.maxAABB = glm::vec4(node->maxAABB, 1.f);

			nodeSSBO.splitPoint = node->point[node->dimSplit];//glm::vec4(node->point, 1.f);
			/*nodeUBO.depth = node->depth;
			nodeUBO.dimSplit = node->dimSplit;
			nodeUBO.isLeaf = node->isLeaf?1:0;*/
			if (node->isLeaf) {
				nodeSSBO.leafId = leavesIndex; //link leaf with node
				leavesIndex++;

				LeafUBO leafUBO;
				int firstId = -1;
				for (int j = 0;j < node->spheres.size();j++) {
					SphereCPU* sphereCPU = node->spheres[j];
					//leafUBO.spheres[j] = node->spheres[j];
					//if (sphereCPU->id < 0) {
						//is not in the sphere buffer yet

					sphereCPU->id = sphereSSBOs.size();
					if (j == 0) firstId = sphereCPU->id;
					Sphere sphere{ sphereCPU->origin,sphereCPU->radius,sphereCPU->color };
					sphereSSBOs.push_back(sphere);
					//leafUBO.spheresIdx[j] = sphereCPU->id;
					/* }
					else {
						leafUBO.spheresIdx[j] = sphereCPU->id;
					}*/
				}
				leafUBO.firstIndex = firstId;
				leafUBO.numberOfSpheres = node->spheres.size();
				leafUBO.minAABB = node->minAABB;//glm::vec4(node->minAABB, float(firstId));
				leafUBO.maxAABB = node->maxAABB;//glm::vec4(node->maxAABB, float(node->spheres.size()));

				leavesUBOs.push_back(leafUBO);

			}
			else {
				//x: -1, y: -2,z: -3
				nodeSSBO.leafId = (-1 * (node->dimSplit + 1));
			}

		}

		treeSSBOs.push_back(nodeSSBO);
	}
		std::cout << "tree Ssbos constructed\n";
}

/*void KdTree::createTreeUbos(std::vector<KdTreeNode> tree) {

}*/

