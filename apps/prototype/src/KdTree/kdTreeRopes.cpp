#include "kdTreeRopes.hpp"
#include "ConstructionStrategy/sahRopesConstruction.cpp"
#include "ConstructionStrategy/medianRopesConstruction.cpp"
/*KdTree::KdTree(std::vector<SphereCPU*> spheres) {
	//constructTree_new(spheres);
	MedianConstruction medianStrategy;
	SahConstruction sahStrategy;
	tree_new = sahStrategy.constructTree(spheres);
	createTreeSsbos(tree_new);
}*/

void KdTreeRopes::construct(std::vector<SphereCPU*> spheres) {
	std::cout << "KdTreeRopes : construct() \n";
	MedianRopesConstruction medianStrategy;
	SahRopesConstruction sahStrategy;
	tree_new = sahStrategy.constructTree(spheres);
	createTreeSsbos(tree_new);
}

void KdTreeRopes::optimizeRopes(std::vector<KdTreeNodeRopes*> tree) {
	std::cout << "optimizing ropes\n";
	for (int i = 0;i < tree.size();i++) {
		KdTreeNodeRopes* node = tree[i];
		std::cout << "i: "<< i<<"\n";
		for (int j = 0;j < 6;j++) {
			while (node->ropes[j] != -1) {
				if (tree[node->ropes[j]]->isLeaf) break;

				if (getSplitAxis(j) == tree_new[node->ropes[j]]->dimSplit && (j==0 || j==2 || j == 4)) {
					node->ropes[j] = tree_new[node->ropes[j]]->rightChildId;

					
					//if (tree[node->ropes[j]]->point[tree[node->ropes[j]]->dimSplit] > node->maxAABB[tree[node->ropes[j]]->dimSplit]) break;
				}
				else if (tree[node->ropes[j]]->point[tree[node->ropes[j]]->dimSplit] < node->minAABB[tree[node->ropes[j]]->dimSplit]) {
					node->ropes[j] = tree_new[node->ropes[j]]->rightChildId;
				}


				if (getSplitAxis(j) == tree_new[node->ropes[j]]->dimSplit && (j == 1 || j == 3 || j == 5)) {
					node->ropes[j] = tree_new[node->ropes[j]]->leftChildId;

					//if (tree[node->ropes[j]]->point[tree[node->ropes[j]]->dimSplit] > node->maxAABB[tree[node->ropes[j]]->dimSplit]) break;
				}
				else if (tree[node->ropes[j]]->point[tree[node->ropes[j]]->dimSplit] > node->maxAABB[tree[node->ropes[j]]->dimSplit]) {
					node->ropes[j] = tree_new[node->ropes[j]]->leftChildId;
				}

				
			}
		}
	}
}



void KdTreeRopes::optimizeNodeRopes(KdTreeNodeRopes* node) {
	for (int j = 0;j < 6;j++) {
		int rope = node->ropes[j];
		if (rope == -1) continue;
		KdTreeNodeRopes* ropeNode = tree_new[rope];
		int counter = 0;
		while (!ropeNode->isLeaf) {
			int splitAxis = ropeNode->dimSplit;
			float splitPlane = ropeNode->point[splitAxis];

			if (splitAxis == getSplitAxis(j)){
				std::cout << "axis: " << splitAxis << "\n";
				if ((j == 0 || j == 2 || j == 4)) { 
					rope = ropeNode->rightChildId;
				}else {
					rope = ropeNode->leftChildId;
				}
			}		
			else if (splitPlane < node->minAABB[splitAxis]) {
				rope = ropeNode->rightChildId;
			}
			else if (splitPlane > node->maxAABB[splitAxis]) {
				rope = ropeNode->leftChildId;
			}
			else {
				break;
			}

			counter++;
			if (rope == -1) break;
			ropeNode = tree_new[rope];
		}
		node->ropes[j] = rope;
		std::cout << "counter: " << counter << "\n";

	}
}

void KdTreeRopes::addRopes(std::vector<KdTreeNodeRopes*> tree) {
	std::cout << "adding ropes\n";

	for (int i = 0;i < tree.size();i++) {
		KdTreeNodeRopes* node = tree[i];
		if (!node->isLeaf) {

			optimizeNodeRopes(node);

			int leftChild = node->leftChildId;
			int rightChild = node->rightChildId;

			auto [S_L, S_R] = getFaceIndexes(node->dimSplit);

			if (leftChild != -1) {

				setRopes(tree[leftChild], node->ropes);
				tree[leftChild]->ropes[S_R] = tree[rightChild]->id;
			}

			if (rightChild != -1) {

				setRopes(tree[rightChild], node->ropes);
				tree[rightChild]->ropes[S_L] = tree[leftChild]->id;
			}

			std::cout << "adding ropes i: "<<i<<"\n";
			
		}
		

	}
	
}



int KdTreeRopes::getSplitAxis(int face) {
	switch (face) {
		case 0: case 1: return 0;
		case 2: case 3: return 1;
		case 4: case 5: return 2;
		default: return -1;
	}
}

void KdTreeRopes::createTreeSsbos(std::vector<KdTreeNodeRopes*> tree) {
	std::cout << "constructing tree SSBOs - ROPES\n";

	//std::vector<KdTreeNodeUBO> tree

	addRopes(tree);
	optimizeRopes(tree);
	int leavesIndex = 1;

	for (int i = 0;i < tree.size();i++) {
		KdTreeNodeRopes* node = tree[i];
		//if (node->depth < 0) break;
		KdTreeNodeSSBO nodeSSBO;

		if (i == 0) {
			LeafRopesUBO leafUBO; //is not actually leaf, just root
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
			//nodeSSBO.leftChild = node->leftChildId;
			//nodeSSBO.rightChild = node->rightChildId;
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

				LeafRopesUBO leafUBO;
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

				for (int i = 0;i < 6;i++) leafUBO.ropes[i] = node->ropes[i];

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
		for (int i = 0;i<tree.size();i++) free(tree[i]);
}

/*void KdTree::createTreeUbos(std::vector<KdTreeNode> tree) {

}*/

std::tuple<int, int> KdTreeRopes::getFaceIndexes(int dimension) {
	//0: left, 1: right, 2: bottom,3: top, 4: front,5: back
	switch (dimension) {
	case 0: //x
		return { 0,1 };
		break;

	case 1: //y
		return { 2,3 };
		//return { 4,5 };
		break;

	case 2: //z
		return { 4,5 };
		//return { 2,3 };
		break;

	default:
		return { -1,-1 };
		break;
	}
}

void KdTreeRopes::setRopes(KdTreeNodeRopes* node, int ropes[6]) {
	for (int i = 0;i < 6;i++) node->ropes[i] = ropes[i];
}