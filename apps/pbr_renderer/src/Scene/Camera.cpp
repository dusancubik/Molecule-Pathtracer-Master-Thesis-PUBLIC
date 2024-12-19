#include "../../include/Scene/Camera.hpp"



Camera::Camera(int width, int height, glm::vec3 _position)
{
	is_zooming = false;
	width = width;
	height = height;
	position = _position;

	FOVdeg = 90.f;
	nearPlane = 0.01f;
	farPlane = 1000.0f;
	initCameraUBO(FOVdeg, nearPlane, farPlane);
}

void Camera::initCameraUBO(float FOVdeg, float nearPlane, float farPlane)
{

	glm::mat4 view = glm::mat4(1.0f);
	glm::mat4 projection = glm::mat4(1.0f);
	//position = glm::vec3(0.f,0.f,0.f);

	view = glm::lookAt(position, glm::vec3(0.0f, 0.0f, 0.0f), up);
	//std::cout << "width: " << width << "height: " << height<< std::endl;
	projection = glm::perspective(glm::radians(FOVdeg), static_cast<float>(width) / static_cast<float>(height), nearPlane, farPlane);

	cameraUBO.position = glm::vec4(position, 1.f);
	cameraUBO.view = view;
	cameraUBO.projection = projection;
	cameraUBO.inversePV = inverse(/*projection */ view);
}
void Camera::updateCamera()
{
	glm::mat4 projection = glm::mat4(1.0f);
	view = glm::lookAt(position, glm::vec3(0.0f, 0.0f, 0.0f), up);

	projection = glm::perspective(glm::radians(FOVdeg), static_cast<float>(width) / static_cast<float>(height), nearPlane, farPlane);
	cameraUBO.position = glm::vec4(position, 1.f);
	cameraUBO.view = view;
	cameraUBO.projection = projection;
	cameraUBO.inversePV = inverse(/*projection */ view);
}


void Camera::onMouseMove(double x, double y) {
	did_move = true;



	float xoffset = x - lastX;
	float yoffset = lastY - y;
	lastX = x;
	lastY = y;

	if (is_zooming) {
		distance *= (1.0f + yoffset * 0.005f);

		
		if (distance < 1.f)
			distance = 1.f;

	}
	else {
		const float sensitivity = 0.2f;
		xoffset *= sensitivity;
		yoffset *= sensitivity;

		//std::cout << "x: " << xoffset << "\n";
		//std::cout << "y: " << yoffset << "\n";

		yaw += xoffset;
		pitch += yoffset;
	}

	position.x = distance * sin(glm::radians(pitch)) * cos(glm::radians(yaw));
	position.y = distance * cos(glm::radians(pitch));
	position.z = distance * sin(glm::radians(pitch)) * sin(glm::radians(yaw));
	updateCamera();

}

void Camera::onMouseButton(int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		if (action == GLFW_PRESS) {
			is_zooming = true;
		}
		else {
			is_zooming = false;
		}
	}
}

void Camera::onKeyPressed(int key, int scancode, int action, int mods) {


}