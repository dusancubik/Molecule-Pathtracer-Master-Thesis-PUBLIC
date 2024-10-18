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
#include <thread>
//#include "nativefiledialog/src/nfd_common.h"
#include "nfd_common.h"

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
        void render_ui(WGPURenderPassEncoder* renderPass);
        WGPUDevice device;
        glfw::GLFWFactory glfw_factory;
        webgpu::WGPUContext context;
    private:
        
        Scene* currentScene;
        Renderer* renderer;
        SkyboxesContainer* skyboxesContainer;
        const char* skyboxNames[4] = { "Skybox 1", "Skybox 2", "Skybox 3", "Skybox 4" };
        void OnSkyboxChanged(char* _skyboxName);

        //loading
        std::thread loaderThread;
        std::atomic<bool> loadingMoleculesComplete;

        void WhilePrint();
        void OpenMoleculeDialog();
        void LoadMolecules(const std::filesystem::path& path);

        //
        bool cursorEnabled = false;
        void setCursorEnabled(bool _cursorEnabled);
        //config
        Config* config;
        

        //debug
        DebugConfig* debugConfig;
        bool debugMode = false;
        void setDebugMode(bool _debugMode);
        void startCollectingSamplesDebug();

        //bilateralFilterConfig
        BilateralFilterConfig* bilateralFilterConfig;
        bool bilateralFilterOn = false;
        void bilateralFilterConfigCombobox(bool value);
};