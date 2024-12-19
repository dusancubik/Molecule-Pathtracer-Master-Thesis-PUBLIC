/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: MoleculeLoader.hpp
 *
 *  Description:
 *  The MoleculeLoader class is responsible for loading molecular data.
 * -----------------------------------------------------------------------------
 */
#pragma once
#include "Molecule.hpp"
#include "wgpu_context.h"
#include "glm/glm.hpp"
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
#include <memory>
#include <random>
class MoleculeLoader {
	public:
		static std::vector<std::shared_ptr<SphereCPU>> loadAtoms(const std::filesystem::path& path,std::atomic<bool>& taskComplete, float _scale);
		static void parseLineToAtom(std::string& line, std::shared_ptr<SphereCPU> &atom, float _scale);
};