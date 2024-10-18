#include "../../include/Application/PTApplication.hpp"

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void cursorPosCallback(GLFWwindow* window, double x, double y);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void windowCloseCallback(GLFWwindow* window);

bool PTApplication::init(int width, int height) {
	std::cout << "PTApplication init(" << width << "," << height << ")\n";
	initWindowAndDevice(width, height);
	config = new Config;
	config->currentIteration = 0;
	config->currentSample = 0;

	config->maxIterations = 10;
	config->maxSamples = 128;

	bilateralFilterConfig = new BilateralFilterConfig;
	bilateralFilterConfig->on = 0;
	bilateralFilterConfig->sigmaS = 2.0f;
	bilateralFilterConfig->sigmaL = 0.2f;

	debugConfig = new DebugConfig;


	skyboxesContainer = new SkyboxesContainer(context.get_device());
	skyboxesContainer->addSkybox(ResourcesLoader::loadCubemapData("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\analyst\\skybox\\"), "Default");
	skyboxesContainer->addSkybox(ResourcesLoader::loadCubemapData("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\analyst\\skybox2\\"), "Dark");
	//std::thread dataLoader(&PTApplication::WhilePrint, this);
	std::atomic<bool> isTaskComplete(false);
	std::vector<std::shared_ptr<SphereCPU>> spheres = MoleculeLoader::loadAtoms("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\medvis\\resources\\Proteins\\1fjk.pdb",std::ref(isTaskComplete));

	BVH* bvh = new BVH;
	bvh->construct(spheres);


	currentScene = new Scene;
	if (!currentScene->init(bvh, skyboxesContainer)) { return false; }

	//for (int i = 0;i < spheres.size();i++) free(spheres[i]);
	//delete currentScene;

	renderer = new Renderer;
	renderer->init(context.get_device(), context.get_queue(), context.get_default_swap_chain_format(), currentScene, config, bilateralFilterConfig, debugConfig);


	delete bvh;
	return true;
}

bool PTApplication::initWindowAndDevice(int width, int height) {
	glfw_factory.create(width, height, "PBR PathTracer");
	glfw_factory.show();
	const bool result = context.initialize(glfw_factory, glfw_factory.get_width(), glfw_factory.get_height());
	if (result) {
		glfwSetWindowUserPointer(glfw_factory.get_glfw_handle(), this);
		glfwSetKeyCallback(glfw_factory.get_glfw_handle(), keyCallback);
		glfwSetCursorPosCallback(glfw_factory.get_glfw_handle(), cursorPosCallback);
		glfwSetMouseButtonCallback(glfw_factory.get_glfw_handle(), mouseButtonCallback);
		glfwSetInputMode(glfw_factory.get_glfw_handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwSetWindowCloseCallback(glfw_factory.get_glfw_handle(), windowCloseCallback);

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.IniFilename = nullptr;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableSetMousePos;
		io.WantCaptureKeyboard = true;
		
		ImGui::StyleColorsDark();

		
		ImGui_ImplGlfw_InitForOther(glfw_factory.get_glfw_handle(), true);
		ImGui_ImplWGPU_Init(context.get_device(), 3, context.get_default_swap_chain_format());

		
		return true;
	}
	return false;
}

void PTApplication::onFrame() {
	if (loadingMoleculesComplete) {
		loaderThread.join();
		loadingMoleculesComplete = false;
	}
	WGPUTextureView nextTexture = wgpuSwapChainGetCurrentTextureView(context.get_swap_chain());
	renderer->render(nextTexture);
	render_ui(nullptr);

	wgpuTextureViewRelease(nextTexture);
	wgpuSwapChainPresent(context.get_swap_chain());
}

void PTApplication::update(float delta) {

}

void PTApplication::onFinish() {

}

bool PTApplication::isRunning() {
	return !glfwWindowShouldClose(glfw_factory.get_glfw_handle());
}

void PTApplication::keyPressed(int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		setCursorEnabled(!cursorEnabled);
	}
	if (key == GLFW_KEY_D && action == GLFW_PRESS) {
		setDebugMode(!debugMode);
	}
	if (key == GLFW_KEY_C && action == GLFW_PRESS) {
		renderer->setCollectingDebugMode(config->debugCollectingMode == 0 ? 1 : 0);
	}



	if(!cursorEnabled) currentScene->keyPressed(key, scancode, action, mods);
}

void PTApplication::onMouseButton(int key, int action, int mods)
{
	currentScene->onMouseButton(key, action, mods);
}


void PTApplication::onMouseMove(double x, double y) {
	if(!cursorEnabled) currentScene->onMouseMove(x, y);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	auto* application = static_cast<PTApplication*>(glfwGetWindowUserPointer(window));

	application->keyPressed(key, scancode, action, mods);

}

void cursorPosCallback(GLFWwindow* window, double x, double y) {
	auto* application = static_cast<PTApplication*>(glfwGetWindowUserPointer(window));
	application->onMouseMove(x, y);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	auto* application = static_cast<PTApplication*>(glfwGetWindowUserPointer(window));
	application->onMouseButton(button, action, mods);
}

void windowCloseCallback(GLFWwindow* window) {
	glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void PTApplication::render_ui(WGPURenderPassEncoder* renderPass) {
	//ImG
	ImGui_ImplWGPU_NewFrame();
	ImGui_ImplGlfw_NewFrame();

	ImGui::NewFrame();

	const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(ImVec2(main_viewport->WorkPos.x, main_viewport->WorkPos.y), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(320, 320), ImGuiCond_Always);
	//ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, {800.f,600.f });

	ImGui::Begin("My First Tool", nullptr, ImGuiWindowFlags_NoDecoration);
	//ImGui::Text(std::to_string(fps_cpu).c_str());
	//ImGui::Text(renderTimer ? std::to_string(currentRenderPipeline->getFrameTimeNS()).c_str() : "0");

	if(debugMode) ImGui::Text("DEBUG");
	static int current_item = 0; 
	int previous_item = current_item;
	std::vector<std::string> items = skyboxesContainer->getSkyboxNames();
	if (ImGui::BeginCombo("Dropdown Menu", items[current_item].c_str()))
	{
		for (int n = 0; n < items.size(); n++)
		{
			bool is_selected = (current_item == n);
			if (ImGui::Selectable(items[n].c_str(), is_selected))
				current_item = n; 
			if (is_selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	std::string cs = "Current Sample: ";
	std::string cs_final = cs + std::to_string(renderer->getConfig()->currentSample);
	ImGui::Text(cs_final.c_str());
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	//int v = 0;
	int prevMaxIterations = config->maxIterations;
	ImGui::SliderInt("Iterations", &config->maxIterations, 1, 10);
	if (prevMaxIterations != config->maxIterations) {
		renderer->resetSamples();
	}
	//int samples = 0;
	int prevMaxSamples = config->maxSamples;
	ImGui::SliderInt("Samples", &config->maxSamples, 1, 512);
	if (prevMaxSamples != config->maxSamples) {
		renderer->resetSamples();
	}
	if (ImGui::Checkbox("Bilateral Filter", &bilateralFilterOn)) {
		bilateralFilterConfigCombobox(bilateralFilterOn);
	}
	ImGui::SliderFloat("SigmaS", &bilateralFilterConfig->sigmaS, 0.f, 16.f);
	ImGui::SliderFloat("SigmaL", &bilateralFilterConfig->sigmaL, 0.f, 4.f);
	if (ImGui::Button("Load Molecule"))
	{
		OpenMoleculeDialog();
	}
	//ImGui::SliderInt("Debug Index Ray", &config->debugRayIndex, 0, 32);
	//DEBUG
	if (ImGui::CollapsingHeader("DEBUG"))
	{
		int x = debugConfig->pixelCoordinates.x;
		int y = debugConfig->pixelCoordinates.y;
		ImGui::InputInt("X:",&x);
		ImGui::InputInt("Y:", &y);
		debugConfig->pixelCoordinates.x = x;
		debugConfig->pixelCoordinates.y = y;
		if (ImGui::Button("START DEBUG"))
		{
			startCollectingSamplesDebug();
		}

		ImGui::SliderInt("Debug Index Ray", &config->debugRayIndex, 0, 32);
		int sample = debugConfig->visOption.x;
		ImGui::SliderInt("Current Sample", &sample, 0, 127);
		debugConfig->visOption.x = sample;
		bool showAll = debugConfig->visOption.y == 1.f ? true : false;
		ImGui::Checkbox("Show All Paths",&showAll);
		debugConfig->visOption.y = showAll ? 1.f : 0.f;
		int bounce = debugConfig->visOption.z;
		ImGui::SliderInt("Bounce", &bounce, 0, 10);
		debugConfig->visOption.z = bounce;
		bool showOneBounce = debugConfig->visOption.w == 1.f ? true : false;
		ImGui::Checkbox("Show Only 1 Bounce", &showOneBounce);
		debugConfig->visOption.w = showOneBounce ? 1.f : 0.f;
	}


	ImGui::End();

	if (previous_item != current_item)
	{
		// Call your method here
		std::cout << "item changhed!!!\n";
		skyboxesContainer->setCurrentSkybox(items[current_item]);
		renderer->resetSamples();
	}

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

void PTApplication::OpenMoleculeDialog() {
	nfdchar_t* outPath = NULL;
	nfdresult_t result = NFD_OpenDialog(NULL, NULL, &outPath);

	if (result == NFD_OKAY) {
		puts("Success!");
		puts(outPath);
		
		
		loaderThread = std::thread(&PTApplication::LoadMolecules,this, outPath);
		//"E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\medvis\\resources\\Proteins\\1fjk.pdb"
		//free(outPath);
	}
	else if (result == NFD_CANCEL) {
		puts("User pressed cancel.");
	}
	else {
		printf("Error: %s\n", NFD_GetError());
	}
}

void PTApplication::LoadMolecules(const std::filesystem::path& path) {
	//std::thread loaderThread(MoleculeLoader::loadAtoms,path);
	//loaderThread = std::thread(MoleculeLoader::loadAtoms, path, std::ref(loadingMoleculesComplete));
	//TODO: memory leak
	std::vector<std::shared_ptr<SphereCPU>> newSpheres = MoleculeLoader::loadAtoms(path, std::ref(loadingMoleculesComplete));
	//loaderThread.join();
	BVH* bvh = new BVH;
	bvh->construct(newSpheres);

	//set to the scene + notify renderer
	currentScene->changeBVH(bvh);

	loadingMoleculesComplete = true;
}


void PTApplication::setCursorEnabled(bool _cursorEnabled) {
	cursorEnabled = _cursorEnabled;
	if (cursorEnabled) {
		glfwSetInputMode(glfw_factory.get_glfw_handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
	else {
		glfwSetInputMode(glfw_factory.get_glfw_handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
}

void PTApplication::setDebugMode(bool _debugMode) {
	debugMode = _debugMode;
	//set renderer
	renderer->setDebugMode(debugMode);
}

void PTApplication::startCollectingSamplesDebug() {
	renderer->resetSamples();
	debugConfig->pixelCoordinates.z = 1.f;
	debugConfig->cameraPosition = glm::vec4(currentScene->getCamera()->getPosition(), 0.f);
}

void PTApplication::bilateralFilterConfigCombobox(bool value) {
	bilateralFilterOn = value;
	bilateralFilterConfig->on = bilateralFilterOn ? 1 : 0;
}