#pragma once
#include <iostream>
#include "application.h"
#include "glfw_factory.h"
#include "wgpu_context.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_wgpu.h"
//#include "ResourceManager.hpp"
#include "camera.hpp"
#include "timeStamp.hpp"
#include "renderPipeline.hpp"
#define GLM_FORCE_DEPTH_ZERO_TO_ONE //this needs to be set because of vulkan??
#define GLM_FORCE_LEFT_HANDED
#include "glm/glm.hpp"
#include "glm/ext.hpp"

#include "Pipelines/computeRenderer.hpp"
#include "Pipelines/computeRendererBVH.hpp"
#include "Pipelines/bvhFragmentRenderer.hpp"
#include "Pipelines/bvh4FragmentRenderer.hpp"
#include "Pipelines/kdTreeRenderPipeline.hpp"
#include "Pipelines/kdTreeRopesRenderPipeline.hpp"
#include "Pipelines/computeRendererBVH_accumulator.hpp"
#include "Factories/kdTreeStandardFactory.hpp"
#include "Factories/kdTreeRopesFactory.hpp"
#include <random>
#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include <functional>
using namespace visitlab;



class MainApplication {
public:
    float fps_cpu = 0;
    float fps_gpu = 0;

    using mat4x4 = glm::mat4x4;
    using vec4 = glm::vec4;
    using vec3 = glm::vec3;
    using vec2 = glm::vec2;
    // A function called only once at the beginning. Returns false is init failed.
    bool onInit(int width, int height);

    // A function called at each frame, guaranteed never to be called before `onInit`.
    void onFrame();

    void update(float delta);

    // A function called only once at the very end.
    void onFinish();

    bool isRunning();

    std::vector<SphereCPU*> spheres;
    std::vector<SphereCPU*> generateSpheres(int number);
    WGPUBuffer spheresStorageBuffer = nullptr;

    struct VertexAttributes {
        vec3 position;
        //vec3 normal;
        vec3 color;
        //vec2 uv;
    };

    //input handling
    void keyPressed(int key, int scancode, int action, int mods);
    void onMouseMove(double x, double y);
    void onMouseButton(int key, int action, int mods);
    static void readBufferMap(WGPUBufferMapAsyncStatus status, void* userdata);
private:
    bool renderTimer = false;
    const float PI = 3.14159265358979323846f;
    using mat4x4 = glm::mat4x4;
    using vec4 = glm::vec4;
    using vec3 = glm::vec3;
    using vec2 = glm::vec2;
    // Everything that is initialized in `onInit` and needed in `onFrame`.
    //wgpu::Instance m_instance = nullptr;
    //wgpu::Surface m_surface = nullptr;
    float randomFloat(float a, float b);

    bool initWindowAndDevice(int width, int height);
    bool initDepthBuffer();
    bool initRenderPipeline();
    bool initGeometry();
    bool initUniforms();
    bool initBindGroup();

    WGPUDevice device;
    glfw::GLFWFactory glfw_factory;
    webgpu::WGPUContext context;
    // [...]

    struct MyUniforms {
        // We add transform matrices
        mat4x4 projectionMatrix;
        mat4x4 viewMatrix;
        mat4x4 modelMatrix;
        vec4 color;
        float time;
        float _pad[3];
    };
    //Camera
    std::shared_ptr<Camera> camera;
    bool initCamera();
    // Depth Buffer
    WGPUTextureFormat depthTextureFormat = WGPUTextureFormat_Depth24Plus;
    WGPUTexture depthTexture = nullptr;
    WGPUTextureView depthTextureView = nullptr;
    WGPUDepthStencilState depthStencilState{};
    // Render Pipeline
    WGPUBindGroupLayout bindGroupLayout = nullptr;
    WGPUShaderModule shaderModule = nullptr;
    WGPURenderPipeline pipeline = nullptr;

    // Texture
    /*wgpu::Sampler m_sampler = nullptr;
    wgpu::Texture m_texture = nullptr;
    wgpu::TextureView m_textureView = nullptr;*/

    // Geometry
    WGPUBuffer indexBuffer = nullptr;
    WGPUBuffer vertexBuffer = nullptr;
    int m_vertexCount = 0;
    std::vector<float> vertexData;
    std::vector<uint16_t> indexData;
    int indexCount = 0;
    // Uniforms
    WGPUBuffer uniformBuffer = nullptr;
    MyUniforms uniforms;

    // Bind Group
    WGPUBindGroup bindGroup = nullptr;

    

    //UI
    WGPURenderPipeline UIpipeline = nullptr;
    void setupUI();
    void render_ui(WGPURenderPassEncoder* renderPass);
    bool cursorOn = false;
    bool usingKdTree = false;

    //Timestamp
    std::shared_ptr<Timestamp> timestamp;
    //void readBufferMap(WGPUBufferMapAsyncStatus status, void *userdata);

    //rendering
    std::shared_ptr<RenderPipeline> basicRenderPipeline;
    std::shared_ptr<KdTreeRenderPipeline> kdTreeRenderPipeline;
    std::shared_ptr<KdTreeRopesRenderPipeline> kdTreeRopesRenderPipeline;


    std::shared_ptr<RendererBase> currentRenderPipeline;
    bool usingKdTrees = false;
};