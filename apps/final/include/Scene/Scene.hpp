#pragma once
#include <iostream>
#include <vector>
#include "../../include/AccelerationStructure/BVH.hpp"
#include "Camera.hpp"
#include "Skybox.hpp"
#include "../Utils/ResourcesLoader.hpp"

class Scene {
	public:
		bool init(BVH* _bvh/*lights,cameraInitPos*/);
		void update(/*deltaTime*/);

		void keyPressed(int key, int scancode, int action, int mods);
		void onMouseMove(double x, double y);
		void onMouseButton(int key, int action, int mods);

		//getBVH()
		//getLights()
		Camera* getCamera() { return camera; }
		BVH* getBVH() { return bvh; }
		Skybox* getSkybox() { return skybox; }
	private:
		Camera *camera;
		BVH* bvh;
		Skybox* skybox;
		//vector lights
		//BVH
};