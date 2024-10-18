#pragma once
#include <iostream>
#include <vector>
#include "../../include/AccelerationStructure/BVH.hpp"
#include "Camera.hpp"
#include "Skybox.hpp"
#include "../Utils/ResourcesLoader.hpp"
#include "../Renderer/SkyboxesContainer.hpp"
class Scene {
	public:
		bool init(BVH* _bvh, SkyboxesContainer* _skyboxesContainer/*lights,cameraInitPos*/);
		void update(/*deltaTime*/);

		void keyPressed(int key, int scancode, int action, int mods);
		void onMouseMove(double x, double y);
		void onMouseButton(int key, int action, int mods);

		//getBVH()
		//getLights()
		SkyboxesContainer* getSkyboxesContainer() { return skyboxesContainer; }
		Camera* getCamera() { return camera; }
		BVH* getBVH() { return bvh; }
		Skybox* getSkybox() { return skybox; }
		void changeBVH(BVH* _bvh);
		bool isBVHChanged() { return BVHChanged; }
		void SetBVHChanged(bool _BVHChanged) { BVHChanged = _BVHChanged; }
	private:
		Camera *camera;
		BVH* bvh;
		Skybox* skybox;
		SkyboxesContainer* skyboxesContainer;
		bool BVHChanged = false;
		//vector lights
		//BVH
};