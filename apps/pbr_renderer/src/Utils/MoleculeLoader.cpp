#include "../../include/Utils/MoleculeLoader.hpp"

std::vector<std::shared_ptr<SphereCPU>> MoleculeLoader::loadAtoms(const std::filesystem::path& path, std::atomic<bool>& taskComplete,float _scale) {
	taskComplete = false;

	std::ifstream file;
	file.open(path);

	std::vector<std::shared_ptr<SphereCPU>> atoms;

	if (!file.is_open()) {
		std::cerr << "Error opening file" << std::endl;
		return atoms;
	}


	std::string line;
	glm::vec3 minAABB = glm::vec3(9999999.f);
	glm::vec3 maxAABB = glm::vec3(-9999999.f);
	while (getline(file, line)) {
		std::string firstWord = line.substr(0, line.find_first_of(" "));
		if (firstWord == "ATOM") {

			std::shared_ptr<SphereCPU> atom = std::make_shared<SphereCPU>();
			parseLineToAtom(line, atom,_scale);
			//break;
			if (atom->radius == 1.f) {
				minAABB = glm::min(minAABB, atom->origin);
				maxAABB = glm::max(maxAABB, atom->origin);
				atoms.push_back(atom);
			}
		}

	}
	glm::vec3 midAABB = (minAABB + maxAABB) / 2.f;
	std::cerr << "minAABB: (" << minAABB.x << "," << minAABB.y << "," << minAABB.z << ")" << std::endl;
	std::cerr << "maxAABB: (" << maxAABB.x << "," << maxAABB.y << "," << maxAABB.z << ")" << std::endl;
	std::cerr << "-----------NEW-----------" << std::endl;
	minAABB = minAABB - midAABB;
	maxAABB = maxAABB - midAABB;
	std::cerr << "midAABB: (" << midAABB.x << "," << midAABB.y << "," << midAABB.z << ")" << std::endl;
	std::cerr << "minAABB: (" << minAABB.x << "," << minAABB.y << "," << minAABB.z << ")" << std::endl;
	std::cerr << "maxAABB: (" << maxAABB.x << "," << maxAABB.y << "," << maxAABB.z << ")" << std::endl;

	for (int i = 0;i < atoms.size();i++) atoms[i]->origin -= midAABB;
	//taskComplete = true;
	return atoms;
}
void MoleculeLoader::parseLineToAtom(std::string& line, std::shared_ptr<SphereCPU> &atom, float _scale) {
	std::regex pdb_regex(R"(^ATOM\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S)\s+(\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+(\S+))");
	std::smatch matches;
	float scale = _scale;
	if (std::regex_search(line, matches, pdb_regex)) {
		float r1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float r2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float r3 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dist(1, 4);
		int random_number = dist(gen);

		atom->color = glm::vec4(0.926f, 0.721f, 0.504f, float(random_number));
		atom->radius = 1.f;
		atom->origin = glm::vec3(scale * std::stof(matches[6].str()), scale * std::stof(matches[7].str()), scale * std::stof(matches[8].str()));
	}
}