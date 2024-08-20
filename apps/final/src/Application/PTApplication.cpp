#include "../../include/Application/PTApplication.hpp"

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void cursorPosCallback(GLFWwindow* window, double x, double y);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void windowCloseCallback(GLFWwindow* window);

bool PTApplication::init(int width, int height) {
	std::cout << "PTApplication init(" << width << "," << height << ")\n";
	initWindowAndDevice(width, height);

	std::vector<std::shared_ptr<SphereCPU>> spheres = MoleculeLoader::loadAtoms("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\analyst\\data\\1cqw.pdb");

	BVH* bvh = new BVH;
	bvh->construct(spheres);


	currentScene = new Scene;
	if (!currentScene->init(bvh)) { return false; }

	//for (int i = 0;i < spheres.size();i++) free(spheres[i]);
	//delete currentScene;

	renderer = new Renderer;
	renderer->init(context.get_device(), context.get_queue(), context.get_default_swap_chain_format(), currentScene);

	delete bvh;
	return true;
}

bool PTApplication::initWindowAndDevice(int width, int height) {
	glfw_factory.create(width, height, "PBR PathTracer");
	glfw_factory.show();
	const bool result = context.initialize(glfw_factory, glfw_factory.get_width(), glfw_factory.get_height());
	if (result) {
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.IniFilename = nullptr;

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();

		// Setup Platform/Renderer backends
		ImGui_ImplGlfw_InitForOther(glfw_factory.get_glfw_handle(), true);
		ImGui_ImplWGPU_Init(context.get_device(), 3, context.get_default_swap_chain_format());

		glfwSetWindowUserPointer(glfw_factory.get_glfw_handle(), this);
		glfwSetKeyCallback(glfw_factory.get_glfw_handle(), keyCallback);
		glfwSetCursorPosCallback(glfw_factory.get_glfw_handle(), cursorPosCallback);
		glfwSetMouseButtonCallback(glfw_factory.get_glfw_handle(), mouseButtonCallback);
		glfwSetInputMode(glfw_factory.get_glfw_handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwSetWindowCloseCallback(glfw_factory.get_glfw_handle(), windowCloseCallback);
		return true;
	}
	return false;
}

void PTApplication::onFrame() {

	WGPUTextureView nextTexture = wgpuSwapChainGetCurrentTextureView(context.get_swap_chain());
	renderer->render(nextTexture);


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
	currentScene->keyPressed(key, scancode, action, mods);
}

void PTApplication::onMouseButton(int key, int action, int mods)
{
	currentScene->onMouseButton(key, action, mods);
}


void PTApplication::onMouseMove(double x, double y) {
	currentScene->onMouseMove(x, y);
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