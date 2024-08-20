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
extern "C" int __main__(int argc, char* argv[])
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
	/*glfw_factory.create(800, 600, "Visitlab Playground");
	glfw_factory.show();
	
	

	const bool result = context.initialize(glfw_factory, glfw_factory.get_width(), glfw_factory.get_height());
	if (result)
	{
		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.IniFilename = nullptr;

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();

		// Setup Platform/Renderer backends
		ImGui_ImplGlfw_InitForOther(glfw_factory.get_glfw_handle(), true);
		ImGui_ImplWGPU_Init(context.get_device(), 3, context.get_default_swap_chain_format());
		setup();

		MainApplication application;
		application.onInit(context.get_device(),glfw_factory);


		glfw_factory.loop(redraw_local);
	}

	#ifndef __EMSCRIPTEN__ // TODO update to avoid using macros
	// Shutdown the Platform/Renderer backends.
	if (result)
	{
		ImGui_ImplWGPU_Shutdown();
		ImGui_ImplGlfw_Shutdown();
	}
	#endif*/

	return 0;
}



/*WGPUShaderModule loadShaderModule(const fs::path& path, WGPUDevice device) {
	std::ifstream file(path);
	if (!file.is_open()) {
		std::cout << "Cannot load module" << std::endl;
		return nullptr;
	}
	file.seekg(0, std::ios::end);
	size_t size = file.tellg();
	std::string shaderSource(size, ' ');
	file.seekg(0);
	file.read(shaderSource.data(), size);

	WGPUShaderModuleWGSLDescriptor shaderCodeDesc = {};
	shaderCodeDesc.chain.next = nullptr;
	shaderCodeDesc.chain.sType = WGPUSType::WGPUSType_ShaderModuleWGSLDescriptor;
	shaderCodeDesc.source = shaderSource.c_str();
	WGPUShaderModuleDescriptor shaderDesc = {};
	shaderDesc.nextInChain = &shaderCodeDesc.chain;
#ifdef WEBGPU_BACKEND_WGPU
	shaderDesc.hintCount = 0;
	shaderDesc.hints = nullptr;
#endif

	return wgpuDeviceCreateShaderModule(device, &shaderDesc);
}

bool loadGeometry(const fs::path& path, std::vector<float>& pointData, std::vector<uint16_t>& indexData, int dimensions) {
	std::ifstream file(path);
	if (!file.is_open()) {
		return false;
	}

	pointData.clear();
	indexData.clear();

	enum class Section {
		None,
		Points,
		Indices,
	};
	Section currentSection = Section::None;

	float value;
	uint16_t index;
	std::string line;
	while (!file.eof()) {
		getline(file, line);
		if (line == "[points]") {
			currentSection = Section::Points;
		}
		else if (line == "[indices]") {
			currentSection = Section::Indices;
		}
		else if (line[0] == '#' || line.empty()) {
			// Do nothing, this is a comment
		}
		else if (currentSection == Section::Points) {
			std::istringstream iss(line);
			// Get x, y, r, g, b
			for (int i = 0; i < dimensions+3; ++i) {
				iss >> value;
				pointData.push_back(value);
			}
		}
		else if (currentSection == Section::Indices) {
			std::istringstream iss(line);
			// Get corners #0 #1 and #2
			for (int i = 0; i < 3; ++i) {
				iss >> index;
				indexData.push_back(index);
			}
		}
	}
	return true;
}*/
