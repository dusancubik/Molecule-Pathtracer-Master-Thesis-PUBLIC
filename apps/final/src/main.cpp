// ###############################################################################
// 
// Visitlab Playground - Visualization & Computer Graphics Toolkit.
// 
// Copyright (c) 2021-2022 Visitlab (https://visitlab.fi.muni.cz)
// All rights reserved.
// 
// ################################################################################
#define STB_IMAGE_IMPLEMENTATION
#include "../include/Application/PTApplication.hpp"

namespace visitlab
{
	
}

/**
 * The common entry point for all platforms.
 * <p>
 * We are using __main__ as a workaround for Emscripten that requires an asynchronous start.
 *
 * @param 	argc	the count of program arguments in argv.
 * @param 	argv	the program arguments.
 * @return	An int.
 */
extern "C" int __main__(int argc, char* argv[])
{
	using namespace visitlab;
	PTApplication application;
	application.init(1280, 720);


	double last_glfw_time = glfwGetTime() * 1000.0;


	while (application.isRunning()) {
		glfwPollEvents();

		const double current_time = glfwGetTime() * 1000.0; 
		const double elapsed_time = current_time - last_glfw_time;
		last_glfw_time = current_time;

		application.update(static_cast<float>(elapsed_time));


		application.onFrame();
	}
	std::cout << "finish\n";
	return 0;
}
