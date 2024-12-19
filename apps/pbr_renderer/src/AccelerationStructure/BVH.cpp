#include "../../include/AccelerationStructure/BVH.hpp"


#define NODE_COST 10.
#define TEST_COST 4.
#define SPHERE_THR 3

BVH::~BVH() {
    for (BVHConstructNode* node : BVHConstruct) delete node;
    //for (BVHNodeSSBO node : bvhSSBOs) delete &node;
    //for (SphereGPU sphere : spheresGPU_SSBOs) delete &sphere;
}
void BVH::construct(std::vector<std::shared_ptr<SphereCPU>> spheres) {
    constructBVH(spheres);
    tranformToSSBO();
}

void BVH::tranformToSSBO() {
    for (int i = 0; i < BVHConstruct.size(); i++) {
        BVHConstructNode* node = BVHConstruct[i];
        BVHNodeSSBO nodeSSBO;
        nodeSSBO.minAABB = node->minAABB;
        nodeSSBO.maxAABB = node->maxAABB;

        if (i != 0) {
            if (node->left) bvhSSBOs[node->parent].leftChild = i;
        }
        if (node->isLeaf) {
            int firstId = -1;

            for (int j = 0;j < node->spheres.size();j++) {
                std::shared_ptr<SphereCPU> sphereCPU = node->spheres[j];
                //leafUBO.spheres[j] = node->spheres[j];
                //if (sphereCPU->id < 0) {
                    //is not in the sphere buffer yet

                sphereCPU->id = spheresGPU_SSBOs.size();
                if (j == 0) firstId = sphereCPU->id;
                SphereGPU sphere{ glm::vec4(sphereCPU->origin,sphereCPU->color.w) };
                //Sphere sphere{ glm::vec4(sphereCPU->origin,1.f) };
                spheresGPU_SSBOs.push_back(sphere);
            }
            nodeSSBO.leftChild = -1 * firstId;
            nodeSSBO.numberOfSpheres = node->spheres.size();
        }
        bvhSSBOs.push_back(nodeSSBO);
        //delete node;
    }
    std::cout << "BVH transformed to SSBOs\n";
}

void BVH::constructBVH(std::vector<std::shared_ptr<SphereCPU>> spheres) {
    BVHConstructNode* root = new BVHConstructNode;
    root->spheres = spheres;
    root->id = BVHConstruct.size();
    std::stack<BVHConstructNode*> unprocessed;
    root->depth = 0;
    unprocessed.push(root);
    BVHConstruct.push_back(root);
    while (!unprocessed.empty()) {
        BVHConstructNode* node = unprocessed.top();
        unprocessed.pop();
        updateAABB(node, node->spheres);
        BVHDivision bvhDiv;
        /*if (node->spheres.size()>4)
            bvhDiv = subdivide(node);
        else {
            node->isLeaf = true;
            //BVHConstruct.push_back(node);
            continue;
        }*/
        if (node->depth > totalDepth) totalDepth = node->depth;



        bvhDiv = subdivideSAH(node);
        if (bvhDiv.leaf) {
            node->isLeaf = true;
            continue;
        }


        BVHConstructNode* leftChild = new BVHConstructNode;
        leftChild->spheres = bvhDiv.leftSpheres;

        BVHConstructNode* rightChild = new BVHConstructNode;
        rightChild->spheres = bvhDiv.rightSpheres;

        //node->id = BVHConstruct.size();

        leftChild->parent = node->id;
        leftChild->left = true;
        rightChild->parent = node->id;

        leftChild->id = BVHConstruct.size();
        BVHConstruct.push_back(leftChild);

        rightChild->id = BVHConstruct.size();
        BVHConstruct.push_back(rightChild);
        unprocessed.push(leftChild);
        unprocessed.push(rightChild);

        leftChild->depth = node->depth + 1;
        rightChild->depth = node->depth + 1;
        //unprocessed.push(leftChild);
    }
    std::cout << "BVH CPU constructed\n";
}

void BVH::updateAABB(BVHConstructNode* node, std::vector<std::shared_ptr<SphereCPU>> spheres) {

    node->minAABB = glm::vec3(9999.f);
    node->maxAABB = glm::vec3(-9999.f);
    for (int i = 0; i < spheres.size(); i++)
    {
        std::shared_ptr<SphereCPU> sphere = spheres[i];
        float radius = 1.f;//sphere->radius;
        updateBounds(node->minAABB, node->maxAABB, sphere->origin - radius, sphere->origin + radius);
    }
}

BVHDivision BVH::subdivide(BVHConstructNode* node) {
    glm::vec3 extent = node->maxAABB - node->minAABB;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    float splitPos = node->minAABB[axis] + extent[axis] * 0.5f;
    std::vector<std::shared_ptr<SphereCPU>> spheres = node->spheres;
    std::sort(spheres.begin(), spheres.end(), sortSpheresBVH(axis));

    std::vector<std::shared_ptr<SphereCPU>> leftChildSpheres;
    std::vector<std::shared_ptr<SphereCPU>> rightChildSpheres;
    for (int i = 0; i < spheres.size(); i++)
    {
        std::shared_ptr<SphereCPU> sphere = spheres[i];
        if (sphere->origin[axis] < splitPos) {
            leftChildSpheres.push_back(sphere);
        }
        else {
            rightChildSpheres.push_back(sphere);
        }

    }
    node->dimSplit = axis;

    BVHDivision bvhDiv;
    bvhDiv.leftSpheres = leftChildSpheres;
    bvhDiv.rightSpheres = rightChildSpheres;

    return bvhDiv;
}

BVHDivision BVH::subdivideSAH(BVHConstructNode* node) {
    BVHDivision bvhDiv;
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
        std::vector<std::shared_ptr<SphereCPU>> spheres = node->spheres;
        const int nBuckets = 64;
        BucketInfo buckets[nBuckets];

        for (int i = 0; i < spheres.size(); i++)
        {
            std::shared_ptr<SphereCPU> sphere = spheres[i];
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
            std::vector<std::shared_ptr<SphereCPU>> leftChildSpheres;
            std::vector<std::shared_ptr<SphereCPU>> rightChildSpheres;
            for (int i = 0; i < spheres.size(); i++)
            {
                std::shared_ptr<SphereCPU> sphere = spheres[i];
                int b = nBuckets * offset(node->minAABB, node->maxAABB, sphere->origin)[axis];
                if (b == nBuckets) b = nBuckets - 1;

                if (b <= minCostSplitBucket) {
                    leftChildSpheres.push_back(sphere);
                }
                else {
                    rightChildSpheres.push_back(sphere);
                }

            }

            bvhDiv.leftSpheres = leftChildSpheres;
            bvhDiv.rightSpheres = rightChildSpheres;
        }
        else {
            bvhDiv.leaf = true;
        }
    }

    return bvhDiv;
}

void BVH::updateBounds(glm::vec3& minAABB, glm::vec3& maxAABB, glm::vec3 minCentroid, glm::vec3 maxCentroid) {
    minAABB.x = glm::min(minAABB.x, minCentroid.x);
    minAABB.y = glm::min(minAABB.y, minCentroid.y);
    minAABB.z = glm::min(minAABB.z, minCentroid.z);


    maxAABB.x = glm::max(maxAABB.x, maxCentroid.x);
    maxAABB.y = glm::max(maxAABB.y, maxCentroid.y);
    maxAABB.z = glm::max(maxAABB.z, maxCentroid.z);
}

glm::vec3 BVH::offset(glm::vec3 minAABB, glm::vec3 maxAABB, glm::vec3 centroid) {
    glm::vec3 o = centroid - minAABB;
    if (maxAABB.x > minAABB.x) o.x /= maxAABB.x - minAABB.x;
    if (maxAABB.y > minAABB.y) o.y /= maxAABB.y - minAABB.y;
    if (maxAABB.z > minAABB.z) o.z /= maxAABB.z - minAABB.z;
    return o;
}

float BVH::getSurfaceArea(glm::vec3 minAABB, glm::vec3 maxAABB) {
    glm::vec3 d = maxAABB - minAABB;
    return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
}