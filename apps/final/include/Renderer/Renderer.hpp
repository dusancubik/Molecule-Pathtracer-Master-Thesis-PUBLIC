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

struct Config {
	int currentIteration = -1;
	int maxIterations = -1;
	int currentSample = -1;
	int maxSamples = -1;
	float time = 0.f;
	float uniformRandom = 0.f;
};

class Renderer {
	public:
		void init(WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format, Scene* _scene);

		void render(WGPUTextureView& nextTexture);

	private:
		WGPUDevice device;
		WGPUQueue queue;
		WGPUTextureFormat swap_chain_default_format;
		Scene* scene;
		Config config;
		int iteration = 0;
		int sampleId = 0;

		WGPUBindGroup bindGroup;
		WGPURenderPipeline pipeline;
		WGPUTextureView depthTextureView = nullptr;



		//Buffers
		Buffer* cameraBuffer;

		Texture* colorTextureAlpha;
		Texture* colorTextureBeta;
		Buffer* configBuffer;

		BindGroup* pathtracingDataBindGroup;
		BindGroup* computeTexturesBindGroupAlpha;
		BindGroup* computeTexturesBindGroupBeta;

		BindGroup* configBindGroup;
		BindGroup* sampleBindGroupAlpha;
		BindGroup* sampleBindGroupBeta;
		BindGroup* accumulationTexturesBindGroupAlpha;
		BindGroup* accumulationTexturesBindGroupBeta;

		WGPUComputePipeline pathtracingPipeline;
		WGPURenderPipeline accumulationPipeline;

		void initPathtracingPipeline();
		void initAccumulationPipeline();
};