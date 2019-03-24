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
	struct cudaGraphicsResource *cuda_vbo_resource, *cuda_ebo_resource;

private:
	NormalFirework(float* args) : FwBase(args) {
		printf("--- cons normal firework\n");
		//BeginGroup(4, 3);

		//EndGroup(12);

		/*for (int i = 0; i < 4; ++i) {
			AddVec3("顶点" + std::to_wstring(i));
		}

		for (int i = 0; i < 4; ++i) {
			AddColor("颜色" + std::to_wstring(i));
		}

		for (int i = 0; i < 4; ++i) {
			AddValue("尺寸" + std::to_wstring(i));
		}*/
		printf("--- cons normal firework done\n");
	}

	void initialize() override {
		// 调用父类的初始化
		FwBase::initialize();

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 
			24 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
			cudaGraphicsMapFlagsWriteDiscard);

		glGenBuffers(1, &ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
			12 * sizeof(int), nullptr , GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&cuda_ebo_resource, ebo,
			cudaGraphicsMapFlagsWriteDiscard);

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
		cudaMalloc(&dPoints, 12 * sizeof(float));
		cudaMalloc(&dColors, 12 * sizeof(float));
		cudaMalloc(&dSizes, 4 * sizeof(float));
		cudaMemcpy(dPoints, args_ + 12 * frameIdx,
			12 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dColors, args_ + 49 * 12 + 12 * frameIdx,
			12 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dSizes, args_ + 49 * 24 + 4 * frameIdx,
			4 * sizeof(float), cudaMemcpyHostToDevice);

		sizeEbo = getTrianglesAndIndices(static_cast<float*>(pVboData),
			static_cast<GLuint*>(pEboData), dPoints, dColors, dSizes, 4);

		cudaFree(dPoints);
		cudaFree(dColors);
		cudaFree(dSizes);

		// 释放对buffer的内存指针映射
		cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
		cudaGraphicsUnmapResources(1, &cuda_ebo_resource, 0);
	}

	void RenderParticles(const Camera& camera) override {
		shader_->use();
		shader_->setMat4("view", camera.GetViewMatrix());
		shader_->setMat4("projection", camera.GetProjectionMatrix());
		glBindVertexArray(vao);
		// draw points 0-3 from the currently bound VAO with current in-use shader;
		glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0);
		//glDrawArrays(GL_TRIANGLES, 0, 3);
		glBindVertexArray(0);
	}

public:
	~NormalFirework() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};