#pragma once
#include "firework.h"
#include "kernels.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

class NormalFirework : public FwBase {
	friend FwBase* getFirework(FireWorkType type, float* args);
	GLuint vbo = 0;
	GLuint ebo = 0;
	GLuint vao = 0;
	size_t pointSize;
	struct cudaGraphicsResource *cuda_vbo_resource, *cuda_ebo_resource;

private:
	NormalFirework(float* args) : FwBase(args) {
		BeginGroup(4, 3);
			AddVec3Group("顶点1");
			AddVec3Group("顶点2");
			AddVec3Group("顶点3");
			AddVec3Group("顶点4");
		EndGroup(12);
		BeginGroup(4, 3);
			AddVec3Group("颜色1");
			AddVec3Group("颜色2");
			AddVec3Group("颜色3");
			AddVec3Group("颜色4");
		EndGroup(12);
	}

	void initialize() override {
		// 调用父类的初始化
		FwBase::initialize();

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 
			100000 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
			cudaGraphicsMapFlagsWriteDiscard);

		glGenBuffers(1, &ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
			100000 * sizeof(int), nullptr , GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&cuda_ebo_resource, ebo,
			cudaGraphicsMapFlagsWriteDiscard);
	}

	void GetParticles(int frameIdx) override {
		// 映射buffer的内存指针
		cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
		cudaGraphicsMapResources(1, &cuda_ebo_resource, 0);
		void *pVboData, *pEboData;
		size_t sizeVbo, sizeEbo;
		cudaGraphicsResourceGetMappedPointer(
			&pVboData, &sizeVbo, cuda_vbo_resource);
		cudaGraphicsResourceGetMappedPointer(
			&pEboData, &sizeEbo, cuda_ebo_resource);

		// 具体计算三角形面片的代码实现在这里
		float *dPoints, *dColors, *dSizes;
		size_t *dGroupOffsets;
		size_t groupOffsets[2]{ 0, 4 };
		cudaMalloc(&dPoints, 12 * sizeof(float));
		cudaMalloc(&dColors, 12 * sizeof(float));
		cudaMalloc(&dSizes, 4 * sizeof(float));
		cudaMalloc(&dGroupOffsets, 2 * sizeof(size_t));
		cudaMemcpy(dPoints, args_ + 12 * frameIdx,
			12 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dColors, args_ + 49 * 12 + 12 * frameIdx,
			12 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dSizes, args_ + 49 * 24 + 4 * frameIdx,
			4 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dGroupOffsets, groupOffsets,
			2 * sizeof(size_t), cudaMemcpyHostToDevice);

		pointSize = pointToLine(dPoints, dSizes, dColors, 4, dGroupOffsets, 1,
			static_cast<float*>(pVboData), static_cast<GLuint*>(pEboData));

		cudaFree(dPoints);
		cudaFree(dColors);
		cudaFree(dSizes);

		// 释放对buffer的内存指针映射
		cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
		cudaGraphicsUnmapResources(1, &cuda_ebo_resource, 0);

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
			6 * sizeof(GLfloat), (GLvoid*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
			6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	void RenderParticles(const Camera& camera) override {
		shader_->use();
		shader_->setMat4("view", camera.GetViewMatrix());
		shader_->setMat4("projection", camera.GetProjectionMatrix());

		glBindVertexArray(vao);
		// draw points 0-3 from the currently bound VAO with current in-use shader;
		glDrawElements(GL_TRIANGLES, pointSize, GL_UNSIGNED_INT, 0);
		//glDrawArrays(GL_TRIANGLES, 0, 3);

		glBindVertexArray(0);
	}

public:
	~NormalFirework() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};