#include "bvh4.hpp"
#include "bvh.cpp"
#include "../float16/half.cpp"
void BVH4::construct(std::vector<SphereCPU*> spheres) {
	constructBVH4(spheres);
	tranformToSSBO();
	std::cout << "BVH in BVH4 constructed\n";
}

void BVH4::tranformToSSBO() {
    for (int i = 0; i < BVH4Construct.size(); i++) {
        BVH4ConstructNode* node = BVH4Construct[i];
        BVH4NodeSSBO nodeSSBO;
        nodeSSBO.axis = node->dimSplit;
        if (i != 0) {
            if (node->child1) {
                
                bvh4SSBOs[node->parent].child[0] = i;
                bvh4SSBOs[node->parent].bbox[0] = FLOAT16::ToFloat16(node->minAABB.x);
                bvh4SSBOs[node->parent].bbox[1] = FLOAT16::ToFloat16(node->minAABB.y);
                bvh4SSBOs[node->parent].bbox[2] = FLOAT16::ToFloat16(node->minAABB.z);

                bvh4SSBOs[node->parent].bbox[3] = FLOAT16::ToFloat16(node->maxAABB.x);
                bvh4SSBOs[node->parent].bbox[4] = FLOAT16::ToFloat16(node->maxAABB.y);
                bvh4SSBOs[node->parent].bbox[5] = FLOAT16::ToFloat16(node->maxAABB.z);
            }
            else if (node->child2) {
                bvh4SSBOs[node->parent].child[1] = i;
                bvh4SSBOs[node->parent].bbox[6] = FLOAT16::ToFloat16(node->minAABB.x);
                bvh4SSBOs[node->parent].bbox[7] = FLOAT16::ToFloat16(node->minAABB.y);
                bvh4SSBOs[node->parent].bbox[8] = FLOAT16::ToFloat16(node->minAABB.z);

                bvh4SSBOs[node->parent].bbox[9] = FLOAT16::ToFloat16(node->maxAABB.x);
                bvh4SSBOs[node->parent].bbox[10] = FLOAT16::ToFloat16(node->maxAABB.y);
                bvh4SSBOs[node->parent].bbox[11] = FLOAT16::ToFloat16(node->maxAABB.z);
            }
            else if (node->child3) {
                bvh4SSBOs[node->parent].child[2] = i;
                bvh4SSBOs[node->parent].bbox[12] = node->minAABB.x;
                bvh4SSBOs[node->parent].bbox[13] = node->minAABB.y;
                bvh4SSBOs[node->parent].bbox[14] = node->minAABB.z;

                bvh4SSBOs[node->parent].bbox[15] = node->maxAABB.x;
                bvh4SSBOs[node->parent].bbox[16] = node->maxAABB.y;
                bvh4SSBOs[node->parent].bbox[17] = node->maxAABB.z;
            }
            else if (node->child4) {
                bvh4SSBOs[node->parent].child[3] = i;
                bvh4SSBOs[node->parent].bbox[18] = node->minAABB.x;
                bvh4SSBOs[node->parent].bbox[19] = node->minAABB.y;
                bvh4SSBOs[node->parent].bbox[20] = node->minAABB.z;

                bvh4SSBOs[node->parent].bbox[21] = node->maxAABB.x;
                bvh4SSBOs[node->parent].bbox[22] = node->maxAABB.y;
                bvh4SSBOs[node->parent].bbox[23] = node->maxAABB.z;
            }
        }
        if (node->isLeaf) {
            int firstId = -1;

            for (int j = 0;j < node->spheres.size();j++) {
                SphereCPU* sphereCPU = node->spheres[j];
                //leafUBO.spheres[j] = node->spheres[j];
                //if (sphereCPU->id < 0) {
                    //is not in the sphere buffer yet

                sphereCPU->id = sphereSSBOs.size();
                if (j == 0) firstId = sphereCPU->id;
                //Sphere sphere{ sphereCPU->origin,sphereCPU->radius,sphereCPU->color };
                Sphere sphere{ glm::vec4(sphereCPU->origin,1.f) };
                sphereSSBOs.push_back(sphere);
            }
            nodeSSBO.child[0] = -1 * firstId;
            nodeSSBO.numberOfSpheres = node->spheres.size();
        }
        if(!node->isLeaf) bvh4SSBOs.push_back(nodeSSBO);
    }
    std::cout << "BVH transformed to SSBOs\n";
}

void BVH4::constructBVH4(std::vector<SphereCPU*> spheres) {
    BVH4ConstructNode* root = new BVH4ConstructNode;
    root->spheres = spheres;
    root->id = BVH4Construct.size();
    std::stack<BVH4ConstructNode*> unprocessed;
    root->depth = 0;
    unprocessed.push(root);
    BVH4Construct.push_back(root);
    while (!unprocessed.empty()) {
        BVH4ConstructNode* node = unprocessed.top();
        unprocessed.pop();
        updateAABB(node, node->spheres);
        BVH4Division bvhDiv;
        if (node->spheres.size()>8)
            bvhDiv = subdivide(node);
        else {
            node->isLeaf = true;
            //BVHConstruct.push_back(node);
            continue;
        }
        if (node->depth > totalDepth) totalDepth = node->depth;



        //bvhDiv = subdivide(node);
        /*if (bvhDiv.leaf) {
            node->isLeaf = true;
            continue;
        }*/


        BVH4ConstructNode* child1 = new BVH4ConstructNode;
        child1->spheres = bvhDiv.spheres1;

        BVH4ConstructNode* child2 = new BVH4ConstructNode;
        child2->spheres = bvhDiv.spheres2;

        BVH4ConstructNode* child3 = new BVH4ConstructNode;
        child3->spheres = bvhDiv.spheres3;

        BVH4ConstructNode* child4 = new BVH4ConstructNode;
        child4->spheres = bvhDiv.spheres4;

        //node->id = BVHConstruct.size();

        child1->parent = node->id;
        child1->child1 = true;
        
        child2->parent = node->id;
        child2->child2 = true;

        child3->parent = node->id;
        child3->child3 = true;

        child4->parent = node->id;
        child4->child4 = true;


        child1->id = BVH4Construct.size();
        BVH4Construct.push_back(child1);

        child2->id = BVH4Construct.size();
        BVH4Construct.push_back(child2);

        child3->id = BVH4Construct.size();
        BVH4Construct.push_back(child3);

        child4->id = BVH4Construct.size();
        BVH4Construct.push_back(child4);



        unprocessed.push(child1);
        unprocessed.push(child2);
        unprocessed.push(child3);
        unprocessed.push(child4);

        child1->depth = node->depth + 1;
        child2->depth = node->depth + 1;
        child3->depth = node->depth + 1;
        child4->depth = node->depth + 1;
        //unprocessed.push(leftChild);
    }
    std::cout << "BVH CPU constructed\n";
}

BVH4Division BVH4::subdivide(BVH4ConstructNode* node) {
    glm::vec3 extent = node->maxAABB - node->minAABB;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    float splitPos1 = node->minAABB[axis] + extent[axis] * 0.25f;
    float splitPos2 = node->minAABB[axis] + extent[axis] * 0.5f;
    float splitPos3 = node->minAABB[axis] + extent[axis] * 0.75f;
    std::vector<SphereCPU*> spheres = node->spheres;
    std::sort(spheres.begin(), spheres.end(), sortSpheresBVH(axis));

    std::vector<SphereCPU*> spheres1;
    std::vector<SphereCPU*> spheres2;
    std::vector<SphereCPU*> spheres3;
    std::vector<SphereCPU*> spheres4;
    for (int i = 0; i < spheres.size(); i++)
    {
        SphereCPU* sphere = spheres[i];
        if (sphere->origin[axis] < splitPos1) {
            spheres1.push_back(sphere);
        }
        else if (sphere->origin[axis] > splitPos1 && sphere->origin[axis] < splitPos2){
            spheres2.push_back(sphere);
        }
        else if (sphere->origin[axis] > splitPos2 && sphere->origin[axis] < splitPos3) {
            spheres3.push_back(sphere);
        }
        else {
            spheres4.push_back(sphere);
        }

    }
    node->dimSplit = axis;

    BVH4Division bvhDiv;
    bvhDiv.spheres1 = spheres1;
    bvhDiv.spheres2 = spheres2;
    bvhDiv.spheres3 = spheres3;
    bvhDiv.spheres4 = spheres4;
    //bvhDiv.rightSpheres = rightChildSpheres;

    return bvhDiv;
}

/*BVH4Division BVH4::subdivideSAH(BVHConstructNode* node) {
    BVH4Division bvhDiv;
    if (node->spheres.size() <= 2) {
        //return subdivide(node);
        bvhDiv.leaf = true;
    }
    else {
        glm::vec3 extent = node->maxAABB - node->minAABB;
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent[axis]) axis = 2;
        struct BucketInfo {
            int count = 0;
            glm::vec3 minAABB = glm::vec3(9999.f);;
            glm::vec3 maxAABB = glm::vec3(-9999.f);;
        };
        std::vector<SphereCPU*> spheres = node->spheres;
        const int nBuckets = 64;
        BucketInfo buckets[nBuckets];

        for (int i = 0; i < spheres.size(); i++)
        {
            SphereCPU* sphere = spheres[i];
            float radius = sphere->radius;
            int b = nBuckets * offset(node->minAABB, node->maxAABB, sphere->origin)[axis];

            if (b == nBuckets) b = nBuckets - 1;

            buckets[b].count++;
            updateBounds(buckets[b].minAABB, buckets[b].maxAABB, sphere->origin - radius, sphere->origin + radius);
        }

        float cost[nBuckets];
        for (int i = 0; i < nBuckets; i++)
        {
            glm::vec3 minB0 = glm::vec3(9999.f);
            glm::vec3 maxB0 = glm::vec3(-9999.f);

            glm::vec3 minB1 = glm::vec3(9999.f);
            glm::vec3 maxB1 = glm::vec3(-9999.f);
            int count0 = 0, count1 = 0;
            for (int j = 0; j <= i; ++j) {
                updateBounds(minB0, maxB0, buckets[j].minAABB, buckets[j].maxAABB);
                count0 += buckets[j].count;
            }
            for (int j = i + 1; j < nBuckets; ++j) {
                updateBounds(minB1, maxB1, buckets[j].minAABB, buckets[j].maxAABB);
                count1 += buckets[j].count;
            }

            cost[i] = NODE_COST + TEST_COST * (count0 * getSurfaceArea(minB0, maxB0) + count1 * getSurfaceArea(minB1, maxB1))
                / getSurfaceArea(node->minAABB, node->maxAABB);
        }


        float minCost = cost[0];
        int minCostSplitBucket = 0;
        for (int i = 1; i < nBuckets - 1; ++i) {
            if (cost[i] < minCost) {
                minCost = cost[i];
                minCostSplitBucket = i;
            }
        }

        //leaf or not
        float leafCost = TEST_COST * node->spheres.size();
        if (minCost <= leafCost || leafCost < SPHERE_THR) {
            std::vector<SphereCPU*> leftChildSpheres;
            std::vector<SphereCPU*> rightChildSpheres;
            for (int i = 0; i < spheres.size(); i++)
            {
                SphereCPU* sphere = spheres[i];
                int b = nBuckets * offset(node->minAABB, node->maxAABB, sphere->origin)[axis];
                if (b == nBuckets) b = nBuckets - 1;

                if (b <= minCostSplitBucket) {
                    leftChildSpheres.push_back(sphere);
                }
                else {
                    rightChildSpheres.push_back(sphere);
                }

            }

            //bvhDiv.leftSpheres = leftChildSpheres;
            //bvhDiv.rightSpheres = rightChildSpheres;
        }
        else {
            bvhDiv.leaf = true;
        }
    }

    return bvhDiv;
}*/