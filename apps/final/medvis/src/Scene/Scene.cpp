#include "../../include/Scene/Scene.hpp"

bool Scene::init(BVH* _bvh, SkyboxesContainer* _skyboxesContainer) {
	camera = new Camera(1280, 720, glm::vec3(0.f,0.f,0.f));
	std::cout << "Scene init\n";
	bvh = _bvh;
	skyboxesContainer = _skyboxesContainer;
	CubemapData cd = ResourcesLoader::loadCubemapData("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\analyst\\skybox\\");
	skybox = new Skybox(cd);
	return true;
}

void Scene::update(/*deltaTime*/){
}

void Scene::keyPressed(int key, int scancode, int action, int mods){
	camera->onKeyPressed(key, scancode, action, mods);
}

void Scene::onMouseMove(double x, double y) {
	camera->onMouseMove(x, y);
}


void Scene::onMouseButton(int key, int action, int mods)
{
	camera->onMouseButton(key, action, mods);
}

void Scene::changeBVH(BVH* _bvh) {
	bvh = _bvh;
	SetBVHChanged(true);
}
