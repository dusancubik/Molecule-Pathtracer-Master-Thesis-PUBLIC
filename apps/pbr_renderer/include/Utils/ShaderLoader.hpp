/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: ShaderLoader.hpp
 *
 *  Description:
 *  The ShaderLoader class is responsible for loading WebGPU shader modules from a file.
 *  Method loadShaderModule is taken from Elie Michel's LearnWebGPU-Code repository 
 *  (https://github.com/eliemichel/LearnWebGPU-Code/blob/step037-vanilla/ResourceManager.cpp) which is licensed under the MIT License.
 * -----------------------------------------------------------------------------
 */
#pragma once
#include "wgpu_context.h"
#include <filesystem>
#include <fstream>

class ShaderLoader {
public:
	using path = std::filesystem::path;
	static WGPUShaderModule loadShaderModule(const path& path, WGPUDevice device);
};