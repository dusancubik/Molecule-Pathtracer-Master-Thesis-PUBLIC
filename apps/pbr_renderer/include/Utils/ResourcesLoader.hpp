/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: ResourcesLoader.hpp
 *
 *  Description:
 *  The ResourcesLoader class is responsible for loading WebGPU shader modules from a file.
 *  Method loadCubemapData is taken from Elie Michel's LearnWebGPU-Code repository
 *  (https://github.com/eliemichel/LearnWebGPU-Code/blob/step117/ResourceManager.cpp) which is licensed under the MIT License.
 * -----------------------------------------------------------------------------
 */
#pragma once

#include "wgpu_context.h"
#include "glm/glm.hpp"
#include "../Renderer/Texture.hpp"
#include <vector>
#include <filesystem>
#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include <regex>




class ResourcesLoader {
	public:
		using path = std::filesystem::path;
		static void loadPixelData(const path& path, unsigned char* pixelData);
		static CubemapData loadCubemapData(const path& path);
};