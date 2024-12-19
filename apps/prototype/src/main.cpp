// ###############################################################################
// 
// Visitlab Playground - Visualization & Computer Graphics Toolkit.
// 
// Copyright (c) 2021-2022 Visitlab (https://visitlab.fi.muni.cz)
// All rights reserved.
// 
// ################################################################################

#include "MainApplication.cpp"
#include "application.h"
#include "glfw_factory.h"
#include "wgpu_context.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_wgpu.h"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE //this needs to be set because of vulkan??
#define GLM_FORCE_LEFT_HANDED
#include "glm/glm.hpp"
#include "glm/ext.hpp"

#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <array>

namespace fs = std::filesystem;

//WGPUShaderModule loadShaderModule(const fs::path& path, WGPUDevice device);
//bool loadGeometry(const fs::path& path, std::vector<float>& pointData, std::vector<uint16_t>& indexData, int geometry);
namespace visitlab
{

	
}

/**
 * The common entry point for all platforms.
 * <p>
 * We are using __main__ as a workaround for Emscripten that requires an asynchronous start.
 *
 * @param 	argc	the count of program arguments in argv.
 * @param 	argv	the program arguments.
 * @return	An int.
 */
extern "C" int __main__PROTO(int argc, char* argv[])
{
	using namespace visitlab;
	MainApplication application;
	application.onInit(1280, 720);

	
	double last_glfw_time = glfwGetTime() * 1000.0;


	while (application.isRunning()) {
		glfwPollEvents();

		const double current_time = glfwGetTime() * 1000.0; // from seconds to milliseconds
		const double elapsed_time = current_time - last_glfw_time;
		last_glfw_time = current_time;

		application.update(static_cast<float>(elapsed_time));


		application.onFrame();
	}
	std::cout << "finish\n";
	

	return 0;
}




