/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: SkyboxesContainer.hpp
 *
 *  Description:
 *  The SkyboxesContainer class manages multiple skyboxes in the application. 
 *
 * -----------------------------------------------------------------------------
 */
#pragma once
#include "BindGroup/BindGroup.hpp"
#include <string>
#include "Texture.hpp"
#include <unordered_map>
class SkyboxesContainer {
	public:
		SkyboxesContainer(WGPUDevice _device);
		void addSkybox(CubemapData cubemapData, std::string skyboxName);
		BindGroup* getSkyboxBindGroup(std::string skyboxName);
		BindGroup* getCurrentSkyboxBindGroup() { return skyboxesBindGroupsMap[currentSkybox]; };
		void setCurrentSkybox(std::string newCurrentSkybox) { currentSkybox = newCurrentSkybox; }
		std::vector<std::string> getSkyboxNames() { return skyboxNames; };
	private:
		std::unordered_map<std::string, BindGroup*> skyboxesBindGroupsMap;
		std::vector<BindGroupLayoutEntry> layoutEntries;
		WGPUDevice device;
		std::vector<std::string> skyboxNames;
		std::string currentSkybox = "Sky";
};