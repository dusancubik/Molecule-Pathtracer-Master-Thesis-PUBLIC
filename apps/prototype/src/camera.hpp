/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: PROTO_Camera.hpp
 *
 *  Description:
 *  The PROTO_Camera class represents a camera, handling its position, orientation based on user's mouse movement.
 *  The PROTO_Camera class is based on the implementation and concepts from LearnOpenGL's Camera tutorial
 *  (https://learnopengl.com/Getting-started/Camera).
 * -----------------------------------------------------------------------------
 */
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

class PROTO_Camera
{
	float lastX = 400, lastY = 300;
public:
	bool didMove() { 
		bool tmp = did_move;
		did_move = false;
		return tmp;
	}

	glm::vec3 position;
	glm::vec3 orientation = glm::vec3(0.0f, 0.0f, 0.f);
	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::mat4 view = glm::mat4(1.0f);
	CameraUBO cameraUBO = { .projection = glm::mat4(1.0f),.view = glm::mat4(1.0f),.position = glm::vec4(0.0f),.inversePV = glm::mat4(1.0f) };

	bool firstClick = true;


	int width = 800;
	int height = 600;
	//1280,860

	float speed = 0.001f;
	float sensitivity = 100.0f;

	float FOVdeg = 0;
	float nearPlane = 0;
	float farPlane = 0;

	float yaw = 0.f;
	float pitch = 35.f;

	bool is_zooming = false;
	float distance = 55.f;
	bool did_move = false;
	
	PROTO_Camera(int width, int height, glm::vec3 _position);
	
	void initCameraUBO(float FOVdeg, float nearPlane, float farPlane);

	glm::mat4 getViewMat() { return view; }
	glm::vec3 getPosition() { return position; };
	glm::vec3 getOrientation() { return orientation; };

	CameraUBO* getCameraUbo() { return &cameraUBO; }
	void updateCamera();

	void onMouseMove(double x, double y);
	void onMouseButton(int button, int action, int mods);
	void onKeyPressed(int key, int scancode, int action, int mods);

};