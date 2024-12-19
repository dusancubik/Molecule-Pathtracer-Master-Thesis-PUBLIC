/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: rendererBase.hpp
 *
 *  Description:
 *  The RendererBase class is a base class for managing a rendering pipeline.
 *  It includes WebGPU functionalities such as initializing the rendering pipeline, creating buffers/textures and bind groups.
 *  Setup of these WebGPU functionalities is based on Elie Michel's LearnWebGPU-Code repository 
 *  (https://github.com/eliemichel/LearnWebGPU-Code/) which is licensed under the MIT License.
 * -----------------------------------------------------------------------------
 */
#pragma once
#include "renderPipelineInterface.hpp"


class RendererBase : public RenderPipelineInterface {
	public:
		~RendererBase() {}

		void init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format);

		void render(WGPUTextureView& nextTexture) {}

		int64_t getFrameTimeNS() { return frameTimeNS; }

		std::shared_ptr<PROTO_Camera> getCamera() { return nullptr; }

		void createBindingLayout(uint32_t binding, uint64_t minBindingSize, WGPUBindGroupLayoutEntry& bindingLayout, WGPUBufferBindingType bufferType, WGPUShaderStageFlags shaderFlags);
		void createBindGroupEntry(WGPUBindGroupEntry& bindGroupEntry, uint32_t binding, WGPUBuffer buffer, uint64_t offset, uint64_t size);
		void createBuffer(WGPUBuffer& buffer, WGPUBufferUsageFlags flags, uint64_t size, const void* data);
		void initDepthBuffer();
	protected:
		//Timestamp
		std::shared_ptr<Timestamp<WGPURenderPassTimestampWrite>> timestamp;
		static void readBufferMap(WGPUBufferMapAsyncStatus status, void* userdata);
		int64_t frameTimeNS = 0;

		WGPUDevice device;
		WGPUQueue queue;
		WGPUTextureFormat swap_chain_default_format;

		//
		std::vector<SphereCPU*> spheres;

		// Depth Buffer
		WGPUTextureFormat depthTextureFormat = WGPUTextureFormat_Depth24Plus;
		WGPUTexture depthTexture = nullptr;
		WGPUTextureView depthTextureView = nullptr;
		WGPUDepthStencilState depthStencilState{};

		virtual void initUniforms(){}
		virtual void initBindGroup(WGPUBindGroup& bindGroup, WGPUBindGroupLayout bindGroupLayout) {}
};