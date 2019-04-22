#pragma once
#include <vector>
#include <glm.hpp>
#include <memory>
#include "particle.h"
#include "Shader.h"
#include "Camera.h" 
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class CfwDlg;
enum class ArgType {
	Scalar = 0,
	Color,
	Vector3,
	ColorGroup,
	Vec3Group
};

class FwBase {
	friend class CfwDlg;
protected:

	struct Attr {
		ArgType type;
		std::wstring name;
		size_t start;
		size_t offset;
		size_t stride;
		static size_t idx;
		static size_t groupOffset;
		static size_t groupNums;
		static size_t groupStep;
		Attr(ArgType type, std::wstring name)
			: type(type)
			, name(name)
			, start(idx)
			, offset(0)
			, stride(0) {
			switch (type) {
			case ArgType::Scalar:
				idx += 1;
				break;
			case ArgType::Color:
			case ArgType::Vector3:
				idx += 3;
				break;
			case ArgType::ColorGroup:
			case ArgType::Vec3Group:
				offset = groupOffset;
				groupOffset += groupStep;
				stride = groupNums * groupOffset;
			default:
				break;
			}
		}

		static void startGroup(int n, int step) {
			FW_ASSERT(n > 0 && step > 0);
			groupOffset = 0;
			groupNums = n;
			groupStep = step;
		}

		static void stopGroup() {
			FW_ASSERT(groupNums * groupStep == groupOffset) << "Check whether"
					"the number of Attr is the same as set in startGroup";
			idx += 49 * groupNums * groupStep;
		}
	};

protected:
	std::vector<Attr> attrs_;
	float* args_;
	size_t nFrames_;

protected:
	GLuint vbo = 0;
	GLuint ebo = 0;
	GLuint vao = 0;
	size_t eboSize;
	struct cudaGraphicsResource *cuda_vbo_resource, *cuda_ebo_resource;

protected:
	FwBase(float* args) : args_(args) {}

protected:
	std::unique_ptr<Shader> shader_;

	/* ==========================================
	 * 子类需实现以下几个方法
	 * ==========================================
	 */
protected:
	// 与opengl有关的初始化放在这里，以保证其在glewInit之后执行
	// 子类改写时候必须首先显式的调用本方法。
	virtual void initialize() {
		shader_.reset(new Shader("fw.vs", "fw.fs"));
	}

	void genBuffer(size_t vboSize, size_t eboSize) {
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER,
			vboSize * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
			cudaGraphicsMapFlagsWriteDiscard);

		glGenBuffers(1, &ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER,
			eboSize * sizeof(int), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&cuda_ebo_resource, ebo,
			cudaGraphicsMapFlagsWriteDiscard);
	}

	// 如果子类在初始化时候调用了genBuffer
	// 那么必须在析构函数中调用deleteBuffer
	void deleteBuffer() {
		glDeleteBuffers(1, &vbo);
		glDeleteBuffers(1, &ebo);
		glDeleteVertexArrays(1, &vao);
	}

public:
	virtual void GetParticles(int frameIdx) = 0;

public:

	size_t getTotalFrame() {
		return nFrames_;
	}

	float* getArgs(int idx, int frame = 0) {
		return args_ + attrs_[idx].start +
			attrs_[idx].offset + frame * attrs_[idx].stride;
	}
	void RenderScene(const Camera& camera) {
		shader_->use();
		shader_->setMat4("view", camera.GetViewMatrix());
		shader_->setMat4("projection", camera.GetProjectionMatrix());

		glBindVertexArray(vao);
		// draw points 0-3 from the currently bound VAO with current in-use shader;
		glDrawElements(GL_TRIANGLES, eboSize, GL_UNSIGNED_INT, 0);

		glBindVertexArray(0);
	}

	virtual ~FwBase() = default;
};

enum class FireWorkType {
	Normal = 0
};

// 构造所有FireWork类的唯一入口函数
FwBase* getFirework(FireWorkType type, float* args);

// useful definations used in sub classes
#define AddValue(_name) \
	attrs_.push_back(Attr(ArgType::Scalar, L##_name));

#define AddColor(_name) \
	attrs_.push_back(Attr(ArgType::Color, L##_name));

#define AddVec3(_name) \
	attrs_.push_back(Attr(ArgType::Vector3, L##_name));

#define BeginGroup(_n, _step) Attr::startGroup((_n), (_step))
#define EndGroup() Attr::stopGroup()

#define AddColorGroup(_name) \
	attrs_.push_back(Attr(ArgType::ColorGroup, L##_name));

#define AddVec3Group(_name) \
	attrs_.push_back(Attr(ArgType::Vec3Group, L##_name));