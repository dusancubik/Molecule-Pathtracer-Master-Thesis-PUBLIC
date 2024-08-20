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
		static void test();
};