#pragma once

// Std. Includes
#include <vector>

// GL Includes
#include <GL/glew.h>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <iostream>
using std::cout;
using std::endl;

// An abstract camera class that processes input and calculates the corresponding Eular Angles, Vectors and Matrices for use in OpenGL
class Camera
{
	// Camera Attributes
	glm::vec3 position_;
	glm::vec3 center_;
	glm::vec3 up_;
	float zoom_;
	float windowWidth_;
	float windowHeight_;

public:
	// Constructor with vectors
	Camera(float width, float height,
		    glm::vec3 position = glm::vec3(0.5f, -2.7f, -1.2f),
			//glm::vec3 position = glm::vec3(0.0f, 0.5f, -3.0f),
		    glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f),
		    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
		    float zoom = 45.0f)
		: position_(position)
		, center_(center)
		, up_(up)
		, zoom_(zoom) 
		, windowWidth_(width)
		, windowHeight_(height) {}

	// Returns the view matrix calculated using Eular Angles and the LookAt Matrix
	glm::mat4 GetViewMatrix() const {
		return glm::lookAt(this->position_, this->center_, this->up_);
	}

	void setCameraArgs(
			glm::vec3 position = glm::vec3(0.0f, 0.5f, -3.0f),
			glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f),
			glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f) ) {
		position_ = position;
		center_ = center;
		up_ = up;
	}

	glm::mat4 GetProjectionMatrix(float width, float height) const {
		return glm::perspective(glm::radians(zoom_), width / height, 0.1f, 100.0f);
	}

	glm::mat4 GetProjectionMatrix() const {
		return GetProjectionMatrix(windowWidth_, windowHeight_);
	}

	/* ===================================
     * 摄像头方向调节
	 * 目前只考虑center不变的情况
	 * TODO: 未来会加入centor改变的情况
	 * ===================================
	 */

private:
	void rotateBase(float angleX, float angleY) {
		glm::mat4 transUp(1.0f);
		glm::mat4 transPos(1.0f);
		if (angleX != 0) {
			transPos = glm::rotate(transPos, glm::radians(angleX), this->up_);
		}
		if (angleY != 0) {
			glm::vec3 right = glm::cross(this->up_, this->position_);
			glm::mat4 trans(1.0f);
			transUp = glm::rotate(transUp, glm::radians(angleY), right);
			transPos = glm::rotate(transPos, glm::radians(angleY), right);
		}
		this->position_ = transPos * glm::vec4(this->position_, 1.0);
		this->up_ = transUp * glm::vec4(this->up_, 1.0);
	}

public:
	enum class RotateDirection {
		Horizontal,
		Vertical,
		ErrorArg
	};

	template <RotateDirection D, typename... Args>
	void rotate(Args&&... arg) {
		switch(D) {
		case RotateDirection::Horizontal:
			rotateBase(std::forward<Args>(arg)..., 0);
			break;
		case RotateDirection::Vertical:
			rotateBase(0, std::forward<Args>(arg)...);
			break;
		}
	}
};