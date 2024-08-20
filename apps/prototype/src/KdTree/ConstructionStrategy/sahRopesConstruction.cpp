#include "sahRopesConstruction.hpp"

#define NODE_COST 3
#define TEST_COST 5
#define MAX_SPHERES 6
#define MIN_AREA 30.f
std::vector<KdTreeNodeRopes*> SahRopesConstruction::constructTree(std::vector<SphereCPU*> spheres) {
	std::cout << "constructing tree - SAH STRATEGY\n";
	//std::vector<Sphere> spheresSorted = spheres;
	std::vector<KdTreeNodeRopes*> tree_new;
	//std::sort(spheresSorted.begin(), spheresSorted.end(),smallerX(0));

	//std::vector<KdTreeNode_new> t(spheres.size() * 2);
	//tree = t;
	float scale = 1.f;
	KdTreeNodeRopes* root = new KdTreeNodeRopes;
	root->dimSplit = 0;
	root->spheres = spheres;
	root->depth = 0;
	//root->minAABB = glm::vec3(-713.5f, -598.f, -887.f);
	//root->maxAABB = glm::vec3(713.5f, 598.f, 887.f);
	//root->minAABB = glm::vec3(-20.5f, -2.f, -20.f);
	//root->maxAABB = glm::vec3(20.5f, 20.f, 20.f);
	root->minAABB = glm::vec3(scale *-300.7, scale *-243.845, scale *-217.98);
	root->maxAABB = glm::vec3(scale *300.7, scale *243.845, scale *217.98);

	int maxDepth = -1;
	int spPerLeaf = 3;
	int leavesCounter = 0;

	//std::vector<KdTreeNode_new*> visited;
	//visited.push_back(root);
	//tree_new.push_back(root);
	std::stack<KdTreeNodeRopes*> unprocessed;
	unprocessed.push(root);
	while (!unprocessed.empty()) {

		//KdTreeNode_new _node = unprocessed.top();
		KdTreeNodeRopes* node = unprocessed.top();
		unprocessed.pop();
		

		if (node->depth > maxDepth) maxDepth = node->depth;
		if (node->spheres.size() == 0) {
			node->dimSplit = 0;//dummy
			node->isLeaf = true;
			node->point = glm::vec3(-10000.f);
			node->id = tree_new.size();
			tree_new.push_back(node);
			continue;
		}
		/*if (node->spheres.size() < 3) {

			node->isLeaf = true;

			continue;
		}*/
		//std::vector<Sphere> spheresSorted = node->spheres;

		//std::sort(spheresSorted.begin(), spheresSorted.end(), sortSpheres(node->dimSplit));

		/*std::size_t const half_size = spheresSorted.size() / 2;
		std::vector half1(spheresSorted.begin(), spheresSorted.begin() + half_size);
		std::vector half2(spheresSorted.begin() + half_size, spheresSorted.end());*/

		SahDivision sd = findBestHalfsSAH(node->spheres, node);
		std::vector<SphereCPU*> half1 = sd.A;
		std::vector<SphereCPU*> half2 = sd.B;
		node->dimSplit = sd.dimension;
		//node->point = half1.size() > 0 ? half1[half1.size()-1].origin: half2[0].origin;
		node->point = glm::vec3(sd.splitPoint);
		
		//if (node->spheres.size() <= spPerLeaf) {
		if (sd.splitPoint == -10000.f) {
			node->isLeaf = true;
			node->point = glm::vec3(-10000.f);
			node->id = tree_new.size();
			tree_new.push_back(node);
			continue;
		}

		//auto [S_L, S_R] = getFaceIndexes(node->dimSplit);


		node->id = tree_new.size();
		tree_new.push_back(node);

		KdTreeNodeRopes* leftChild = new KdTreeNodeRopes;
		if (half1.size() >= 0) {

			//leftChild->dimSplit = (node->dimSplit + 1) % 3;
			leftChild->spheres = half1;
			leftChild->depth = node->depth + 1;

			auto [newMinP_L, newMaxP_L] = computeBoundingPoint(true, node->dimSplit, node->point, node->minAABB, node->maxAABB);

			leftChild->minAABB = newMinP_L;
			leftChild->maxAABB = newMaxP_L;
			leftChild->parent = node->id;

			
			
			//tree[2 * i + 1] = leftChild;

		}
		else {
			leftChild->depth = node->depth + 1;

			auto [newMinP_L, newMaxP_L] = computeBoundingPoint(true, node->dimSplit, node->point, node->minAABB, node->maxAABB);

			leftChild->minAABB = newMinP_L;
			leftChild->maxAABB = newMaxP_L;
			leftChild->parent = node->id;
		}

		KdTreeNodeRopes* rightChild = new KdTreeNodeRopes;
		if (half2.size() >= 0) {

			//rightChild->dimSplit = (node->dimSplit + 1) % 3;
			rightChild->spheres = half2;
			rightChild->depth = node->depth + 1;

			auto [newMinP_R, newMaxP_R] = computeBoundingPoint(false, node->dimSplit, node->point, node->minAABB, node->maxAABB);

			rightChild->minAABB = newMinP_R;
			rightChild->maxAABB = newMaxP_R;
			rightChild->parent = node->id;
			//tree[2 * i + 2] = rightChild;
		}
		else {
			rightChild->depth = node->depth + 1;

			auto [newMinP_R, newMaxP_R] = computeBoundingPoint(false, node->dimSplit, node->point, node->minAABB, node->maxAABB);

			rightChild->minAABB = newMinP_R;
			rightChild->maxAABB = newMaxP_R;
			rightChild->parent = node->id;
		}



		//leftChild->parent = node->id;
		//rightChild->parent = node->id;
		//leftChild->id = tree_new.size();
		leftChild->left = true;
		//tree_new.push_back(leftChild);
		//rightChild->id = tree_new.size();
		rightChild->left = false;
		//tree_new.push_back(rightChild);
		//node->leftChildId = leftChild->id;
		//node->rightChildId = rightChild->id;
		//add ropes
		/*setRopes(leftChild, node->ropes);
		setRopes(rightChild, node->ropes);
		

		leftChild->ropes[S_R] = rightChild->id;
		rightChild->ropes[S_L] = leftChild->id;*/

		//unprocessed.push(leftChild);
		unprocessed.push(rightChild);
		unprocessed.push(leftChild);
	}
	std::cout << "tree constructed - SAH STRATEGY\n";
	return tree_new;
}

SahDivision SahRopesConstruction::findBestHalfsSAH(std::vector<SphereCPU*> sortedSpheres, KdTreeNodeRopes* node) {
	//note: spheres are not sorted yet
	SahDivision minSD;
	std::vector<SphereCPU*> chosenSortedSpheres;
	std::vector<SphereCPU*> sortedSpheres_X = node->spheres;
	std::sort(sortedSpheres_X.begin(), sortedSpheres_X.end(), sortSpheres(0));
	std::vector<SphereCPU*> sortedSpheres_Y = node->spheres;
	std::sort(sortedSpheres_Y.begin(), sortedSpheres_Y.end(), sortSpheres(1));
	std::vector<SphereCPU*> sortedSpheres_Z = node->spheres;
	std::sort(sortedSpheres_Z.begin(), sortedSpheres_Z.end(), sortSpheres(2));

	float maxExtent = sortedSpheres_X[sortedSpheres_X.size() - 1]->origin.x - sortedSpheres_X[0]->origin.x;
	chosenSortedSpheres = sortedSpheres_X;
	int axis = 0;
	float yExtent = sortedSpheres_Y[sortedSpheres_Y.size() - 1]->origin.y - sortedSpheres_Y[0]->origin.y;
	if (yExtent > maxExtent) {
		maxExtent = yExtent;
		axis = 1;
		chosenSortedSpheres = sortedSpheres_Y;
	}
	float zExtent = sortedSpheres_Z[sortedSpheres_Z.size() - 1]->origin.z - sortedSpheres_Z[0]->origin.z;
	if (zExtent > maxExtent) {
		maxExtent = zExtent;
		axis = 2;
		chosenSortedSpheres = sortedSpheres_Z;
	}
	minSD = performSAH(chosenSortedSpheres, node, axis);
	//X
	/*std::vector<SphereCPU*> sortedSpheres_X = node->spheres;
	std::sort(sortedSpheres_X.begin(), sortedSpheres_X.end(), sortSpheres(0));
	SahDivision xSD = performSAH(sortedSpheres_X, node, 0);
	//Y
	std::vector<SphereCPU*> sortedSpheres_Y = node->spheres;
	std::sort(sortedSpheres_Y.begin(), sortedSpheres_Y.end(), sortSpheres(1));
	SahDivision ySD = performSAH(sortedSpheres_Y, node, 1);
	//Z
	std::vector<SphereCPU*> sortedSpheres_Z = node->spheres;
	std::sort(sortedSpheres_Z.begin(), sortedSpheres_Z.end(), sortSpheres(2));
	SahDivision zSD = performSAH(sortedSpheres_Z, node, 2);

	float minCost = xSD.cost;
	SahDivision minSD = xSD;
	if (ySD.cost < minCost) {
		minCost = ySD.cost;
		minSD = ySD;
	}
	if (zSD.cost < minCost)
	{
		minCost = zSD.cost;
		minSD = zSD;
	}*/

	return minSD;

}

SahDivision SahRopesConstruction::performSAH(std::vector<SphereCPU*> sortedSpheres, KdTreeNodeRopes* node, int dimSplit) {
	//if (sortedSpheres.size() > 4) {

		//int dimSplit = node->dimSplit;
		float t_step = 0.01f;
		//float t = node->minAABB[dimSplit];

		//Compute volume?

		float width = abs(node->maxAABB.x - node->minAABB.x);
		float height = abs(node->maxAABB.y - node->minAABB.y);
		float length = abs(node->maxAABB.z - node->minAABB.z);
		if (width == 0.0f) {
			std::cout << "width == 0.0f\n";
		}
		if (height == 0.0f) {
			std::cout << "height == 0.0f\n";
		}
		if (length == 0.0f) {
			std::cout << "length == 0.0f\n";
		}
		glm::vec3 vol = glm::vec3(width, height, length);

		float areaMain =  (vol[0] * vol[1] + vol[0] * vol[2] + vol[1] * vol[2]);

		std::vector<Sphere> minA;
		std::vector<Sphere> minB;
		float initMinCost = TEST_COST * sortedSpheres.size();
		float minCost = initMinCost;//std::numeric_limits<float>::max();//cost of making a leaf
		float minT = -10000.f;
		float eps = 0.0001f;
		std::cout << "performing SAH\n";
		//while (t < node->maxAABB[dimSplit]) {

		//for (int j = 0;j < sortedSpheres.size();j++) {
		for (float t = sortedSpheres[0]->origin[dimSplit]-1.f;t < sortedSpheres[sortedSpheres.size()-1]->origin[dimSplit]+1.f;t=t+t_step){
			if (sortedSpheres[sortedSpheres.size() - 1]->origin[dimSplit] + 1.f < t) {
				std::cout << "< t\n";
			}
			int countInA = 0;
			int countInB = 0;
			//float t = sortedSpheres[j]->origin[dimSplit] - sortedSpheres[j]->radius - eps;
			if (t < node->minAABB[dimSplit] || t>node->maxAABB[dimSplit]) continue;
			
			/*int a = 0;
			int b = sortedSpheres.size();
			int c = 0;

			int i = 0;
			while (i<10) {
				int c = (a + b) / 2;
				if (sortedSpheres[c]->origin[dimSplit]<t) {
					a = c;
				}
				else {
					b = c;
				}
				i++;
			}
			
			countInA = c + 1;
			countInB = sortedSpheres.size() - countInA;*/

			for (int i = 0;i < sortedSpheres.size();i++) {
				float pos = sortedSpheres[i]->origin[dimSplit];
				float radius = sortedSpheres[i]->radius;
				if (pos <= t || pos - radius <= t) {
					countInA++;
				}
				if (pos > t || pos + radius > t ) {
					countInB++;
				}
				//std::cout << "countInA: "<< countInA<< "| countInB: " << countInB << "\n";
			}
			//int countInB = sortedSpheres.size() - countInA;

			glm::vec3 volA = vol;
			glm::vec3 volB = vol;

			volA[dimSplit] = t - node->minAABB[dimSplit];
			volB[dimSplit] = node->maxAABB[dimSplit] - t;

			if (volA[dimSplit] < 0.f) {
				std::cout << "VolA < 0.f" << "\n";
			}

			if (volB[dimSplit] < 0.f) {
				std::cout << "VolB < 0.f" << "\n";
			}

			float areaA = (volA[0] * volA[1] + volA[0] * volA[2] + volA[1] * volA[2]);
			float areaB =  (volB[0] * volB[1] + volB[0] * volB[2] + volB[1] * volB[2]);

			float p_A = areaA / areaMain;
			float p_B = areaB / areaMain;

			//float cost = 1 + p_A * (2 * countInA) + p_B * (2 * countInB);
			float b = 0;
			//if (countInB == 0 || countInA == 0) b = 0.5;
			float cost = NODE_COST + TEST_COST*(1-b) * (areaA * countInA + areaB * countInB) / areaMain;
			
			if (cost < minCost) {
				minCost = cost;
				minT = t;
			}

			//t += 0.1;
		}

		std::cout << "Vol: (" << vol.x << ","<< vol.y<<","<< vol.z << ")\n";
		std::cout << "Area: "<< areaMain  <<"\n";
		std::cout << "Spheres: " << sortedSpheres.size() << "\n";
		std::cout << "initMinCost: " << TEST_COST * sortedSpheres.size() << "\n";
		std::cout << "minCost: " << minCost << "\n";
		SahDivision sd;
		//if (sortedSpheres.size() <= MAX_SPHERES || initMinCost < minCost || areaMain <= MIN_AREA) {
		if (initMinCost <= minCost || areaMain <= MIN_AREA) {
			std::cout << "leaf best\n";
			sd.A = sortedSpheres;
			sd.B = sortedSpheres;
			sd.cost = minCost;
			sd.splitPoint = -10000.f;
			sd.dimension = dimSplit;
		}
		else {
			std::vector<SphereCPU*> A;
			std::vector<SphereCPU*> B;
			int countInA = 0;
			for (int i = 0;i < sortedSpheres.size();i++) {
				float pos = sortedSpheres[i]->origin[dimSplit];
				float radius = sortedSpheres[i]->radius;
				if (pos <= minT || pos - radius <= minT) {
					A.push_back(sortedSpheres[i]);
					/*if (node->depth < 2) {
						if (pos - radius < minT) {
							B.push_back(sortedSpheres[i]);
						}
					}*/
				}
				if (pos > minT  || pos + radius > minT) {
					B.push_back(sortedSpheres[i]);
					/*if (node->depth < 2) {
						if (pos + radius > minT) {
							A.push_back(sortedSpheres[i]);
						}
					}*/
				}
			}
			sd.A = A;
			sd.B = B;
			sd.dimension = dimSplit;
			sd.cost = minCost;
			sd.splitPoint = minT;

		}
		std::cout << "---------------------------\n";
		return sd;
	/*}
	else {
		float splitPoint = (sortedSpheres[sortedSpheres.size() / 2])->origin[dimSplit];

		std::vector<SphereCPU*> A;
		std::vector<SphereCPU*> B;
		int countInA = 0;
		for (int i = 0;i < sortedSpheres.size();i++) {
			float pos = sortedSpheres[i]->origin[dimSplit];
			float radius = sortedSpheres[i]->radius;
			if (pos <= splitPoint ) {
				A.push_back(sortedSpheres[i]);
				
			}
			if (pos > splitPoint ) {
				B.push_back(sortedSpheres[i]);
				
			}
		}
		SahDivision sd;
		sd.A = A;
		sd.B = B;
		sd.dimension = dimSplit;
		sd.cost = 2 * sortedSpheres.size();
		sd.splitPoint = splitPoint;
		return sd;
	}*/
	//for t cyklus: 
		//
		// 
		// 
		//for sphere? cyklus2:
			//is sphere in A or B?
			// 
		//Compute volumus
		//compute P_a && P_b
		//compute cost
		// is min? 
		//t++

}

std::tuple<glm::vec3, glm::vec3> SahRopesConstruction::computeBoundingPoint(bool isLeft, int dimenstionSplit, glm::vec3 splitPoint, glm::vec3 minP, glm::vec3 maxP) {
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

	if (newMinP.x == newMaxP.x) {
		std::cout << "VnewMinP.x == newMaxP.x\n";
	}

	return{ newMinP,newMaxP };
}

std::vector<SphereCPU*> SahRopesConstruction::getExtendingSpheres(std::vector<SphereCPU*> spheres, glm::vec3 point, int dimension) {
	std::vector<SphereCPU*> extending;

	for (auto sphere : spheres) {
		if (glm::abs(point[dimension] - sphere->origin[dimension]) < sphere->radius) extending.push_back(sphere);
	}

	return extending;

}

/*float SahConstruction::computeSurfaceArea(int dimension, glm::vec3 atributes) {
	float width = atributes[0];
	float height = atributes[1];
	float length = atributes[2];
	

	switch (dimension) {
		case 0:
			return 2 * 
		break;
		case 1:

		break;
		case 2:

		break;
		default:
			std::cout << "SAH:can't compute surface area: dimenation = -1\n";
			return -1.f;
		break;
	}
}*/


std::tuple<int, int> SahRopesConstruction::getFaceIndexes(int dimension) {
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
			return { 4,5};
			//return { 2,3 };
		break;

		default:
			return { -1,-1 };
		break;
	}
}

void SahRopesConstruction::setRopes(KdTreeNodeRopes* node, int ropes[6]) {
	for (int i = 0;i < 6;i++) node->ropes[i] = ropes[i];
}