#pragma once
#include <iostream>
#include"glm/glm.hpp"
#include"glm/ext.hpp"

#include "application.h"
#include "glfw_factory.h"
#include "wgpu_context.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_wgpu.h"
#include <memory>

struct CameraUBO {
	glm::mat4 projection;
	glm::mat4 view;
	glm::vec4 position;
	glm::mat4 inversePV;
};

class Camera
{
	float lastX = 400, lastY = 300;
public:
	bool didMove() { 
		bool tmp = did_move;
		did_move = false;
		return tmp;
	}
	// Stores the main vectors of the camera
	glm::vec3 position;
	glm::vec3 orientation = glm::vec3(0.0f, 0.0f, 0.f);
	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::mat4 view = glm::mat4(1.0f);
	CameraUBO cameraUBO = { .projection = glm::mat4(1.0f),.view = glm::mat4(1.0f),.position = glm::vec4(0.0f),.inversePV = glm::mat4(1.0f) };
	// Prevents the camera from jumping around when first clicking left click
	bool firstClick = true;

	// Stores the width and height of the window
	//default?
	int width = 800;
	int height = 600;
	//1280,860
		// Adjust the speed of the camera and it's sensitivity when looking around
	float speed = 0.001f;
	float sensitivity = 100.0f;

	float FOVdeg = 0;
	float nearPlane = 0;
	float farPlane = 0;

	float yaw = 0.f;
	float pitch = 0.f;

	bool is_zooming = false;
	float distance = 150.f;
	bool did_move = false;
	// Camera constructor to set up initial values
	Camera(int width, int height, glm::vec3 _position);
	//Camera() {};
	// Updates and exports the camera matrix to the Vertex Shader
	void initCameraUBO(float FOVdeg, float nearPlane, float farPlane);

	glm::mat4 getViewMat() { return view; }
	glm::vec3 getPosition() { return position; };
	glm::vec3 getOrientation() { return orientation; };
	//void updateCameraUBO(float FOVdeg, float nearPlane, float farPlane);
	// Handles camera inputs
	//void Inputs(GLFWwindow* window);
	CameraUBO* getCameraUbo() { return &cameraUBO; }
	void updateCamera();

	void onMouseMove(double x, double y);
	void onMouseButton(int button, int action, int mods);
	void onKeyPressed(int key, int scancode, int action, int mods);

};