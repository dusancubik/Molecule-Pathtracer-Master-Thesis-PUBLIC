/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: PTApplication.hpp
 *
 *  Description: 
 *  The PTApplication class serves as the core of the application.
 *  It manages the application's initialization, sets up the Renderer class, and handles input and user interface interactions.
 *  Also, it instructs the MoleculeLoader to load atoms.
 * -----------------------------------------------------------------------------
 */

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
#include "../Utils/Timer.hpp"

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
        bool panelOpen = true;
        Scene* currentScene;
        Renderer* renderer;
        SkyboxesContainer* skyboxesContainer;
        const char* skyboxNames[4] = { "Skybox 1", "Skybox 2", "Skybox 3", "Skybox 4" };
        void OnSkyboxChanged(char* _skyboxName);
        BVH* currentBVH;
        //loading
        std::thread loaderThread;
        std::atomic<bool> loadingMoleculesComplete;
        bool isLoadingMolecules = false;
        void WhilePrint();
        void OpenMoleculeDialog();
        void LoadMolecules(const std::filesystem::path& path);
        float scale = 1.f;
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
        void stopCollectingSamplesDebug();
        //bilateralFilterConfig
        BilateralFilterConfig* bilateralFilterConfig;
        bool bilateralFilterOn = false;
        void bilateralFilterConfigCombobox(bool value);

        //GUI
        void drawLabelWithDots(std::string text);
        int dotsState = 0;
        float timeAccumulatorDots = 0.0f;

        //timer
        Timer timer;
        int numberOfSpheres = 0;
};