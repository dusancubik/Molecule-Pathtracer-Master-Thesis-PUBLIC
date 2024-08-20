#pragma once
#include <iostream>
#include "application.h"
#include "glfw_factory.h"
#include "wgpu_context.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_wgpu.h"
#include "../Utils/MoleculeLoader.hpp"
#include "../Scene/Scene.hpp"
#include "../AccelerationStructure/BVH.hpp"
#include "../Renderer/Renderer.hpp"

using namespace visitlab;

class PTApplication {
	public:
		bool init(int width, int height);

        void onFrame();

        void update(float delta);

        void onFinish();

        bool isRunning();


        //input handling
        void keyPressed(int key, int scancode, int action, int mods);
        void onMouseMove(double x, double y);
        void onMouseButton(int key, int action, int mods);


        bool initWindowAndDevice(int width, int height);

        WGPUDevice device;
        glfw::GLFWFactory glfw_factory;
        webgpu::WGPUContext context;
    private:
        Scene* currentScene;
        Renderer* renderer;
};