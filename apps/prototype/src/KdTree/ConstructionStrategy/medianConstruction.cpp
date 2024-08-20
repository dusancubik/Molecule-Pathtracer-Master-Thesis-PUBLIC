#include "medianConstruction.hpp"

std::vector<KdTreeNode_new*> MedianConstruction::constructTree(std::vector<SphereCPU*> spheres) {
	std::cout << "constructing tree - MEDIAN STRATEGY\n";
	//std::vector<Sphere> spheresSorted = spheres;
	std::vector<KdTreeNode_new*> tree_new;
	//std::sort(spheresSorted.begin(), spheresSorted.end(),smallerX(0));

	//std::vector<KdTreeNode_new> t(spheres.size() * 2);
	//tree = t;
	KdTreeNode_new* root = new KdTreeNode_new;
	root->dimSplit = 0;
	root->spheres = spheres;
	root->depth = 0;
	root->minAABB = glm::vec3(-372.f, -1.f, -372.f);
	root->maxAABB = glm::vec3(372.f, 42.f, 372.f);
	root->id = 0;
	int maxDepth = -1;
	int spPerLeaf = 4;
	int leavesCounter = 0;

	//std::vector<KdTreeNode_new*> visited;
	//visited.push_back(root);
	tree_new.push_back(root);
	std::stack<KdTreeNode_new*> unprocessed;
	unprocessed.push(root);
	while (!unprocessed.empty()) {

		//KdTreeNode_new _node = unprocessed.top();
		KdTreeNode_new* node = unprocessed.top();
		unprocessed.pop();


		if (node->depth > maxDepth) maxDepth = node->depth;
		if (node->spheres.size() == 0) {

			continue;
		}
		/*if (node->spheres.size() < 3) {

			node->isLeaf = true;

			continue;
		}*/
		std::vector<SphereCPU*> spheresSorted = node->spheres;

		std::sort(spheresSorted.begin(), spheresSorted.end(), sortSpheres(node->dimSplit));

		std::size_t const half_size = spheresSorted.size() / 2;
		std::vector half1(spheresSorted.begin(), spheresSorted.begin() + half_size);
		std::vector half2(spheresSorted.begin() + half_size, spheresSorted.end());

		//node->point = half1.size() > 0 ? half1[half1.size()-1].origin: half2[0].origin;
		if (half1.size() > 0) {
			node->point = half1[half1.size() - 1]->origin;
			//half2.push_back(half1[half1.size() - 1]);
		}
		else if (half2.size() > 0) {
			node->point = half2[0]->origin;
			//half1.push_back(half2[0]);
		}



		//half1.insert(half1.end(), extend2.begin(), extend2.end());
		//half2.insert(half2.end(), extend1.begin(), extend1.end());


		//Todo: pøidat na obì strany



		if (node->spheres.size() <= spPerLeaf) {

			node->isLeaf = true;

			continue;
		}

		std::vector<SphereCPU*> extend1 = getExtendingSpheres(half1, node->point, node->dimSplit);
		std::vector<SphereCPU*> extend2 = getExtendingSpheres(half2, node->point, node->dimSplit);

		//std::move(extend1.begin(), extend1.end(), std::back_inserter(half2));
		//std::move(extend2.begin(), extend2.end(), std::back_inserter(half1));

		KdTreeNode_new* leftChild = new KdTreeNode_new;
		if (half1.size() > 0) {

			leftChild->dimSplit = (node->dimSplit + 1) % 3;
			leftChild->spheres = half1;
			leftChild->depth = node->depth + 1;

			auto [newMinP_L, newMaxP_L] = computeBoundingPoint(true, node->dimSplit, node->point, node->minAABB, node->maxAABB);

			leftChild->minAABB = newMinP_L;
			leftChild->maxAABB = newMaxP_L;
			leftChild->parent = node->id;

			//tree[2 * i + 1] = leftChild;

		}

		KdTreeNode_new* rightChild = new KdTreeNode_new;
		if (half2.size() > 0) {

			rightChild->dimSplit = (node->dimSplit + 1) % 3;
			rightChild->spheres = half2;
			rightChild->depth = node->depth + 1;

			auto [newMinP_R, newMaxP_R] = computeBoundingPoint(false, node->dimSplit, node->point, node->minAABB, node->maxAABB);

			rightChild->minAABB = newMinP_R;
			rightChild->maxAABB = newMaxP_R;
			rightChild->parent = node->id;
			//tree[2 * i + 2] = rightChild;
		}
		leftChild->id = tree_new.size();
		leftChild->left = true;
		tree_new.push_back(leftChild);
		rightChild->id = tree_new.size();
		rightChild->left = false;
		tree_new.push_back(rightChild);

		unprocessed.push(leftChild);
		unprocessed.push(rightChild);
	}
	std::cout << "tree constructed - MEDIAN STRATEGY\n";
	return tree_new;
};

std::tuple<glm::vec3, glm::vec3> MedianConstruction::computeBoundingPoint(bool isLeft, int dimenstionSplit, glm::vec3 splitPoint, glm::vec3 minP, glm::vec3 maxP) {
	glm::vec3 newMinP = minP;
	glm::vec3 newMaxP = maxP;
	if (isLeft) {
		//newMinP = minP;

		switch (dimenstionSplit) {
		case 0: //x
			newMaxP.x = splitPoint.x;
			break;
		case 1: //y
			newMaxP.y = splitPoint.y;
			break;
		case 2: //z
			newMaxP.z = splitPoint.z;
			break;
		}

	}
	else {
		//newMaxP = maxP;

		switch (dimenstionSplit) {
		case 0: //x
			newMinP.x = splitPoint.x;
			break;
		case 1: //y
			newMinP.y = splitPoint.y;
			break;
		case 2: //z
			newMinP.z = splitPoint.z;
			break;
		}

	}

	return{ newMinP,newMaxP };
}

std::vector<SphereCPU*> MedianConstruction::getExtendingSpheres(std::vector<SphereCPU*> spheres, glm::vec3 point, int dimension) {
	std::vector<SphereCPU*> extending;

	for (auto sphere : spheres) {
		if (glm::abs(point[dimension] - sphere->origin[dimension]) < sphere->radius) extending.push_back(sphere);
	}

	return extending;

}