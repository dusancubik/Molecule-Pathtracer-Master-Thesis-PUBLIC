/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: rendererPipeline.hpp
 *
 *  Description:
 *  The RenderPipeline class is a base class for managing a rendering pipeline.
 *  It includes WebGPU functionalities such as initializing the rendering pipeline, creating buffers/textures and bind groups.
 *  Setup of these WebGPU functionalities is based on Elie Michel's LearnWebGPU-Code repository
 *  (https://github.com/eliemichel/LearnWebGPU-Code/) which is licensed under the MIT License.
 * -----------------------------------------------------------------------------
 */
#pragma once
#include "Pipelines/renderPipelineInterface.hpp"


class RenderPipeline : public RenderPipelineInterface {
	public:
		void init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format) override;

		void render(WGPUTextureView &nextTexture) override;

        int getFrameTimeNS() { return frameTimeNS; }

        std::shared_ptr<PROTO_Camera> getCamera() { return camera; };
	private:
        //from app
        WGPUDevice device;
        WGPUQueue queue;
        WGPUTextureFormat swap_chain_default_format;
        
        //webgpu::WGPUContext context;

        void initDepthBuffer();
        void initUniforms();
        void initBindGroup();
        
        WGPUTextureFormat depthTextureFormat = WGPUTextureFormat_Depth24Plus;
        WGPUTexture depthTexture = nullptr;
        WGPUTextureView depthTextureView = nullptr;
        WGPUDepthStencilState depthStencilState{};
        
        WGPUBindGroupLayout bindGroupLayout = nullptr;
        WGPUShaderModule shaderModule = nullptr;
        WGPURenderPipeline pipeline = nullptr;

        
        WGPUBindGroup bindGroup = nullptr;

       
        WGPUBuffer uniformBuffer = nullptr;

        
        std::vector<SphereCPU*> spheres;
        WGPUBuffer spheresStorageBuffer = nullptr;
        //Timestamp
        std::shared_ptr<Timestamp<WGPURenderPassTimestampWrite>> timestamp;

        //Camera
        std::shared_ptr<PROTO_Camera> camera;
        bool initCamera();

        static void readBufferMap(WGPUBufferMapAsyncStatus status, void* userdata);
        int frameTimeNS = 0;
};