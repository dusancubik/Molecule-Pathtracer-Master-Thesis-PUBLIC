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

class MoleculeLoader {
	public:
		static std::vector<std::shared_ptr<SphereCPU>> loadAtoms(const std::filesystem::path& path,std::atomic<bool>& taskComplete);
		static void parseLineToAtom(std::string& line, std::shared_ptr<SphereCPU> &atom);
};