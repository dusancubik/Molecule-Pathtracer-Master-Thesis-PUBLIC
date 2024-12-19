/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: Renderer.hpp
 *
 *  Description:
 *  The Renderer class orchestrates the rendering process in the application, managing WebGPU resources, pipelines, and configurations. 
 *
 * -----------------------------------------------------------------------------
 */
#pragma once
#include "application.h"
#include "glfw_factory.h"
#include "wgpu_context.h"
#include "../Utils/ShaderLoader.hpp"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_wgpu.h"
#include "BindGroup/BindGroup.hpp"
#include "Texture.hpp"
#include "Buffer.hpp"
#include "../Scene/Scene.hpp"
#include "../MaterialSets/MaterialSet.hpp"
struct DebugData {
	glm::vec4 data = glm::vec4(10.f,0.f,0.f,0.f);
	glm::vec4 values[10];
};

struct DebugConfig {
	//x,y, collecting samples on/off
	glm::vec4 pixelCoordinates = glm::vec4(0., 0.,0.,0.);
	glm::vec4 visOption = glm::vec4(0., 0., 0., 0.); //current sample, showAll, bounce, showOnlyOneBounce
	glm::vec4 cameraPosition = glm::vec4(0., 0., 0., 0.); //camera
	//int collectingSamples = 0;

};

struct BilateralFilterConfig {
	int accumulationFinished = 0;
	int on = 1;
	float sigmaS = 2.f;
	float sigmaL = 0.2f;
};

struct Config {
	int currentIteration = -1;
	int maxIterations = -1;
	int currentSample = -1;
	int maxSamples = -1;
	float time = 0.f;
	float uniformRandom = 0.f;
	int debugMode = 0;
	int debugCollectingMode = 0;
	int debugRayIndex = 0;
};

class Renderer {
	public:
		void init(WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format, Scene* _scene,Config* _config, BilateralFilterConfig* _bilateralFilterConfig, DebugConfig* _debugConfig);

		void render(WGPUTextureView& nextTexture);

		Config* getConfig() { return config; }

		void resetSamples(){ sampleId = 0; iteration = 0; bilateralFilterConfig->accumulationFinished = 0;}

		void setDebugMode(bool _debugMode) { 
			debugMode = _debugMode; config->debugMode = _debugMode ? 1 : 0; resetSamples();
		};

		void setCollectingDebugMode(int _debugCollectingMode) {
			config->debugCollectingMode = _debugCollectingMode; resetSamples();
		};

		void setMaterialSetIndex(int index) { materialSetIndex = index; }
		int getMaterialSetIndex() { return materialSetIndex; }

		void loadNewSpheres();
		void initBVHDataBindGroup();
	private:
		
		WGPUDevice device;
		WGPUQueue queue;
		WGPUTextureFormat swap_chain_default_format;
		Scene* scene;
		Config* config;
		BilateralFilterConfig* bilateralFilterConfig;
		int iteration = 0;
		int sampleId = 0;

		WGPUBindGroup bindGroup;
		WGPURenderPipeline pipeline;
		WGPUTextureView depthTextureView = nullptr;

		//Materials
		Buffer* materialsBuffer;
		std::vector<MaterialSet> materialSets;
		void initMaterialSets();
		int materialSetIndex = 0;
		//Buffers
		Buffer* bvhBuffer;
		Buffer* spheresBuffer;
		
		Buffer* bufferBVH = nullptr;
		Buffer* bufferSpheres = nullptr;

		Buffer* cameraBuffer;

		Texture* colorTextureAlpha;
		Texture* colorTextureBeta;
		Buffer* configBuffer;
		Buffer* bilateralFilterConfigBuffer;

		BindGroup* pathtracingDataBindGroup;
		BindGroup* computeTexturesBindGroupAlpha;
		BindGroup* computeTexturesBindGroupBeta;
		BindGroup* bvhDataBindGroup;
		BindGroup* configBindGroup;
		BindGroup* sampleBindGroupAlpha;
		BindGroup* sampleBindGroupBeta;
		BindGroup* accumulationTexturesBindGroupAlpha;
		BindGroup* accumulationTexturesBindGroupBeta;
		BindGroup* debugCounterBindGroupAlpha;
		BindGroup* debugCounterBindGroupBeta;

		WGPUComputePipeline pathtracingPipeline;
		WGPURenderPipeline accumulationPipeline;

		void initPathtracingPipeline();
		void initAccumulationPipeline();

		//Debug
		DebugConfig* debugConfig;
		BindGroup* debugDataBindGroup;
		BindGroup* debugScreenBindGroup;
		WGPURenderPipeline debugScreenPipeline;
		WGPUComputePipeline debugComputePipeline;
		Texture* colorTextureDebug;
		Buffer* debugConfigBuffer;
		Buffer* debugLineBuffer;
		Buffer* debugLineArrayBuffer;
		Buffer* debugIndexAtomicBuffer;
		bool debugMode = false;
		void render_debug(WGPUTextureView& nextTexture);
		void initDebugPipeline();
};