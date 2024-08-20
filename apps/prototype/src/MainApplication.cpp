#include "MainApplication.hpp"
//#include "ResourceManager.cpp"
//#include "camera.cpp"
//#include "timeStamp.cpp"
#include "renderPipeline.cpp"
#include "Pipelines/kdTreeRenderPipeline.cpp"
#include "Pipelines/kdTreeRopesRenderPipeline.cpp"
#include "Pipelines/computeRenderer.cpp"
#include "Pipelines/computeRendererBVH.cpp"
#include "Factories/kdTreeStandardFactory.cpp"
#include "Factories/kdTreeRopesFactory.cpp"
#include "Pipelines/bvhFragmentRenderer.cpp"
#include "Pipelines/bvh4FragmentRenderer.cpp"
#include "Pipelines/computeRendererBVH_accumulator.cpp"
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void cursorPosCallback(GLFWwindow* window, double x, double y);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
/*void gg(WGPUBufferMapAsyncStatus status, void* userdata) {
	int32_t* times =
		(int32_t*)wgpuBufferGetMappedRange(timestamp->getStagingBuffer(), 0, 8 * 2);

}*/
float MainApplication::randomFloat(float a, float b) {
	return a + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (b - a)));
}

std::vector<SphereCPU*> MainApplication::generateSpheres(int number) {

	std::vector<SphereCPU*> newSpheres;
	for (int i = 0;i < number;i++) {
		SphereCPU* newSphere = new SphereCPU;
		newSphere->origin.x = i == number / 2 ? 22.f : randomFloat(-20.f, 20.f);
		newSphere->origin.y = randomFloat(0.5f, 5.f);
		newSphere->origin.z = randomFloat(-20.f, 20.f);

		newSphere->radius = i == number / 2 ? 5.f : 1.f;//randomFloat(0.05f, 0.3f);

		newSphere->color.x = i == number / 2 ? 1.f : randomFloat(0.5f, 1.f);
		newSphere->color.y = i == number / 2 ? 1.f : randomFloat(0.5f, 1.f);
		newSphere->color.z = i == number / 2 ? 1.f : randomFloat(0.5f, 1.f);
		newSphere->color.w = i==number/2?0.f:1.0f;

		newSpheres.push_back(newSphere);
	}
	
	return newSpheres;
}

bool MainApplication::onInit(int width, int height) {
	//Timestamp timeStamp();
	std::cout << "MainApplication created\n";
	std::srand(std::time(nullptr));


	//spheres = ResourceManager::loadAtoms("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\analyst\\data\\1aon.pdb");
	//return false;
	
	spheres = generateSpheres(50);
	
	

	if (!initWindowAndDevice(width, height)) return false;
	
	//init glfw  keys
	glfwSetWindowUserPointer(glfw_factory.get_glfw_handle(), this);
	glfwSetKeyCallback(glfw_factory.get_glfw_handle(), key_callback);
	glfwSetCursorPosCallback(glfw_factory.get_glfw_handle(), cursorPosCallback);
	glfwSetMouseButtonCallback(glfw_factory.get_glfw_handle(), mouse_button_callback);
	glfwSetInputMode(glfw_factory.get_glfw_handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	
	/*KdTreeStandardFactory kdTreeStandardFactory;
	kdTreeRenderPipeline = kdTreeStandardFactory.createRenderPipeline();
	std::shared_ptr<KdTree> kd = kdTreeStandardFactory.createKdTree();

	kdTreeRenderPipeline->setKdTree(kd);
	kdTreeRenderPipeline->init(spheres, context.get_device(), context.get_queue(), context.get_default_swap_chain_format());
	currentRenderPipeline = kdTreeRenderPipeline;*/
	
	
	/*KdTreeRopesFactory kdTreeRopesFactory;
	kdTreeRopesRenderPipeline = kdTreeRopesFactory.createRenderPipeline();
	std::shared_ptr<KdTreeRopes> kd = kdTreeRopesFactory.createKdTree();

	kdTreeRopesRenderPipeline->setKdTree(kd);
	kdTreeRopesRenderPipeline->init(spheres, context.get_device(), context.get_queue(), context.get_default_swap_chain_format());
	currentRenderPipeline = kdTreeRopesRenderPipeline;*/

	//KdTreeRopesFactory kdTreeRopesFactory;
	//kdTreeRopesRenderPipeline = std::make_shared<KdTreeRopesRenderPipeline>();//kdTreeRopesFactory.createRenderPipeline();
	
	

	/*std::shared_ptr<BVH> bvh = std::make_shared<BVH>();
	std::shared_ptr<ComputeRendererBVH> cr = std::make_shared<ComputeRendererBVH>();

	cr->setBVH(bvh);

	cr->init(spheres, context.get_device(), context.get_queue(), context.get_default_swap_chain_format());
	currentRenderPipeline = cr;*/
	
	std::shared_ptr<BVH> bvh = std::make_shared<BVH>();
	std::shared_ptr<ComputeRendererBVHAccumulator> cr = std::make_shared<ComputeRendererBVHAccumulator>();

	cr->setBVH(bvh);

	cr->init(spheres, context.get_device(), context.get_queue(), context.get_default_swap_chain_format());
	currentRenderPipeline = cr;

	/*std::shared_ptr<BVH> bvh = std::make_shared<BVH>();
	std::shared_ptr<BVHFragmentRenderer> cr = std::make_shared<BVHFragmentRenderer>();

	cr->setBVH(bvh);

	cr->init(spheres, context.get_device(), context.get_queue(), context.get_default_swap_chain_format());
	currentRenderPipeline = cr;*/

	/*std::shared_ptr<BVH4> bvh4 = std::make_shared<BVH4>();
	std::shared_ptr<BVH4FragmentRenderer> cr = std::make_shared<BVH4FragmentRenderer>();

	cr->setBVH(bvh4);
	cr->init(spheres, context.get_device(), context.get_queue(), context.get_default_swap_chain_format());
	currentRenderPipeline = cr;
	*/
	//renderTimer = true;
	for (int i = 0;i < spheres.size();i++) free(spheres[i]);

	return true;
}
bool MainApplication::initCamera() {
	camera = std::make_shared<Camera>(1280, 720, glm::vec3(0.f, 0.f, 300.0f));
	return true;
}
bool MainApplication::isRunning() {
	return !glfwWindowShouldClose(glfw_factory.get_glfw_handle());
}

void MainApplication::update(float delta) {
	fps_cpu = 1000 / delta;
}

void MainApplication::onFrame() {


	//camera->updateCamera();
	//wgpuQueueWriteBuffer(context.get_queue(), uniformBuffer, 0, camera->getCameraUbo(), sizeof(CameraUBO));
	
	WGPUTextureView nextTexture = wgpuSwapChainGetCurrentTextureView(context.get_swap_chain());
	/*if (usingKdTree)
		currentRenderPipeline = kdTreeRenderPipeline;
	else
		currentRenderPipeline = basicRenderPipeline;*/

	currentRenderPipeline->render(nextTexture);

	render_ui(nullptr);

	
	wgpuTextureViewRelease(nextTexture);
	wgpuSwapChainPresent(context.get_swap_chain());
	
	//wgpuBufferUnmap(timestamp->getStagingBuffer());
}

void MainApplication::readBufferMap(WGPUBufferMapAsyncStatus status, void *userdata) {
	MainApplication* pThis = (MainApplication*)userdata;
	int64_t* times =
		(int64_t*)wgpuBufferGetConstMappedRange(pThis->timestamp->getStagingBuffer(), 0,sizeof(int64_t) * 2);
	//WGPUProcBufferGetMappedRange()
	//WGPUProcBufferGetConstMappedRange();
	//wgpuBufferGetCon
		
	if (times != nullptr) {
		std::cout << "Frametime: " << (times[1] - times[0]) << "\n";
	}
		
	std::cout << "readBufferMap callback" << "\n";
	wgpuBufferUnmap(pThis->timestamp->getStagingBuffer());
	//mapped
}
/*void MainApplication::onFinish() {
	//std::cout << "MainApplication created\n";
	//return true;
}*/

bool MainApplication::initWindowAndDevice(int width, int height) {
	glfw_factory.create(width, height, "Visitlab Playground");
	glfw_factory.show();
	const bool result = context.initialize(glfw_factory, glfw_factory.get_width(), glfw_factory.get_height());
	if (result) {
		std::cout << "context created\n";
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.IniFilename = nullptr;

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();

		// Setup Platform/Renderer backends
		ImGui_ImplGlfw_InitForOther(glfw_factory.get_glfw_handle(), true);
		ImGui_ImplWGPU_Init(context.get_device(), 3, context.get_default_swap_chain_format());

		//glfw_factory.loop(onFrame);
		//glfw_factory.
		return true;
	}
	return false;
}


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	auto* application = static_cast<MainApplication*>(glfwGetWindowUserPointer(window));

	application->keyPressed(key, scancode, action, mods);
	//application->keyPressed(key, scancode, action, mods);
	/*if (key == GLFW_KEY_SPACE) {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}*/
}

void cursorPosCallback(GLFWwindow* window, double x, double y) {
	auto* application = static_cast<MainApplication*>(glfwGetWindowUserPointer(window));
	application->onMouseMove(x, y);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	auto* application = static_cast<MainApplication*>(glfwGetWindowUserPointer(window));
	application->onMouseButton(button,action,mods);
}

void MainApplication::keyPressed(int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_E && action == GLFW_PRESS) {
		usingKdTree = !usingKdTree;
	}
		
	
	/*if (camera == nullptr) {
		std::cout << "camera nullptr\n";

	}*/
	//camera->onKeyPressed(key, scancode, action, mods);
}

void MainApplication::onMouseButton(int key, int action, int mods)
{
	currentRenderPipeline->getCamera()->onMouseButton(key, action, mods);
}


void MainApplication::onMouseMove(double x, double y) {
	
	currentRenderPipeline->getCamera()->onMouseMove(x, y);
	
}

void MainApplication::render_ui(WGPURenderPassEncoder* renderPass) {
	//ImG
	ImGui_ImplWGPU_NewFrame();
	ImGui_ImplGlfw_NewFrame();

	ImGui::NewFrame();

	const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(ImVec2(main_viewport->WorkPos.x, main_viewport->WorkPos.y), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(320, 120), ImGuiCond_Always);
	//ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, {800.f,600.f });
	
	ImGui::Begin("My First Tool", nullptr, ImGuiWindowFlags_NoDecoration);
	ImGui::Text(std::to_string(fps_cpu).c_str());
	ImGui::Text(renderTimer? std::to_string(currentRenderPipeline->getFrameTimeNS()).c_str() : "0");
	const char* items[] = { "Basic", "KdTree" };
	static const char* current_item = "Basic";
	int n = usingKdTree ? 1 : 0;
	ImGui::Text(std::string(items[n]).c_str());
	ImGui::End();


	//ImGui::EndFrame();
	ImGui::Render();


	WGPURenderPassColorAttachment color_attachments = {};
	color_attachments.loadOp = WGPULoadOp_Load;
	color_attachments.storeOp = WGPUStoreOp_Store;
	color_attachments.clearValue = { 0.0f, 0.0f, 0.0f, 0.0f };
	color_attachments.view = wgpuSwapChainGetCurrentTextureView(context.get_swap_chain());
	WGPURenderPassDescriptor render_pass_desc = {};
	render_pass_desc.colorAttachmentCount = 1;
	render_pass_desc.colorAttachments = &color_attachments;
	render_pass_desc.depthStencilAttachment = nullptr;

	WGPUCommandEncoderDescriptor enc_desc = {};
	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(context.get_device(), &enc_desc);

	WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_desc);
	ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);
	wgpuRenderPassEncoderEnd(pass);

	WGPUCommandBufferDescriptor cmd_buffer_desc = {};
	WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(encoder, &cmd_buffer_desc);
	WGPUQueue queue = wgpuDeviceGetQueue(context.get_device());
	wgpuQueueSubmit(queue, 1, &cmd_buffer);


	//ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), *renderPass);
	//ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

void MainApplication::setupUI() {
	
}