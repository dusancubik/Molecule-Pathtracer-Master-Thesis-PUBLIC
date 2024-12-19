/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: computeRendererBVHAccumulator.hpp
 *
 *  Description:
 *  The ComputeRendererBVHAccumulator is derived from RendererBase and handles rendering pipeline for BVH with samples accumulation.
 * -----------------------------------------------------------------------------
 */
#pragma once
#include "rendererBase.hpp"
#include "../KdTree/kdTreeRopes.hpp"
#include "../BVH/bvh.hpp"

struct Config {
    int currentIteration = -1;
    int maxIterations = -1;
    int currentSample = -1;
    int maxSamples = -1;
    float time = 0.f;
    float uniformRandom = 0.f;
};

class ComputeRendererBVHAccumulator : public RendererBase {
	public:
		void init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format) override;

		void render(WGPUTextureView &nextTexture) override;

        std::shared_ptr<PROTO_Camera> getCamera() override { return camera; };

        void setKdTree(std::shared_ptr<KdTreeRopes> _kdTree)  { kdTree = _kdTree; }
        void setBVH(std::shared_ptr<PROTO_BVH> _bvh) { bvh = _bvh; }
	private:
        Config config;
        WGPUBuffer configBuffer = nullptr;
        int iteration = 0;
        int sampleId = 0;
        WGPUTexture color_buffer;
        WGPUTextureView color_buffer_view;

        WGPUTexture color_buffer2;
        WGPUTextureView color_buffer_view2;

        WGPUSampler sampler;

        WGPUTexture origin_buffer;
        WGPUTextureView origin_buffer_view;
        
        WGPUTexture origin_buffer2;
        WGPUTextureView origin_buffer_view2;
        //WGPUSampler sampler;

        WGPUTexture direction_buffer;
        WGPUTextureView direction_buffer_view;
        
        WGPUTexture direction_buffer2;
        WGPUTextureView direction_buffer_view2;

        //accumulation buffers
        WGPUTexture accumulation_buffer;
        WGPUTextureView accumulation_buffer_view;

        WGPUTexture accumulation_buffer2;
        WGPUTextureView accumulation_buffer_view2;


        WGPUShaderModule screenShaderModule = nullptr;
        WGPUShaderModule raytracingKernelModule = nullptr;

        WGPURenderPipeline pipeline = nullptr;

        WGPURenderPipeline screenPipeline = nullptr;
        WGPUBindGroup screenBindGroup = nullptr;
        WGPUBindGroupLayout screenDataTextureBindLayout = nullptr;
        WGPUBindGroup screenDataTextureBindGroup1 = nullptr;
        WGPUBindGroup screenDataTextureBindGroup2 = nullptr;

        //accumulation
        WGPUBindGroupLayout screenAccumulationBindLayout = nullptr;
        WGPUBindGroup screenAccumulationBindGroup1 = nullptr;
        WGPUBindGroup screenAccumulationBindGroup2 = nullptr;

        //------
        WGPUComputePipeline raytracingPipeline = nullptr;
        WGPUBindGroup raytracingBindGroup = nullptr;

        WGPUBindGroupLayout rayTracingDataTexturesBindLayout = nullptr;
        WGPUBindGroup rayTracingDataTexturesBindGroup1 = nullptr;
        WGPUBindGroup rayTracingDataTexturesBindGroup2 = nullptr;

        // Uniforms
        WGPUBuffer uniformBuffer = nullptr;

        //Spheres
        //std::vector<SphereCPU*> spheres;
        WGPUBuffer spheresStorageBuffer = nullptr;
        //Timestamp
        //std::shared_ptr<Timestamp> timestamp;

        //Camera
        std::shared_ptr<PROTO_Camera> camera;
        bool initCamera();

        //static void readBufferMap(WGPUBufferMapAsyncStatus status, void* userdata);
        //int frameTimeNS = 0;
        //kdTree
        std::shared_ptr<KdTreeRopes> kdTree;
        //bvh
        std::shared_ptr<PROTO_BVH> bvh;
        WGPUBuffer bvhStorageBuffer = nullptr;
        WGPUBuffer leavesStorageBuffer = nullptr;

        void initRaytraycingDataTexturesBindGroups();
        void initScreenDataTexturesBindGroups();
        void initScreenAccumulationBindGroups();
       // void createBindingLayout(uint32_t binding, uint64_t minBindingSize, WGPUBindGroupLayoutEntry& bindingLayout, WGPUBufferBindingType bufferType, WGPUShaderStageFlags shaderFlags);
    
        //cube map
        void prepareCubemap();
        WGPUTexture cubemapTexture;
        WGPUTextureView cubemapTextureView;
        WGPUSampler cubemapSampler;
        protected:
            void initUniforms() override;
            //void initBindGroup(WGPUBindGroup& bindGroup, WGPUBindGroupLayout bindGroupLayout) override;
            void initRaytracingBindGroup(WGPUBindGroup& bindGroup, WGPUBindGroupLayout bindGroupLayout);
            void initScreenBindGroup(WGPUBindGroup& bindGroup, WGPUBindGroupLayout bindGroupLayout);

};