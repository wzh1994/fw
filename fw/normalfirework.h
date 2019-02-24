#pragma once
#include "firework.h"

class NormalFirework : public FwBase {
	friend FwBase* getFirework(FireWorkType type, float* args);
	GLuint vbo = 0;
	GLuint cbo = 0;
	GLuint vao = 0;
	float* points;
	float* colors;
private:
	NormalFirework(float* args) {
		points = new float[9];
		colors = new float[9];
		AddVec3("顶点1", -0.5f, -0.5f, 0.0f);
		AddVec3("顶点2", 0.5f, -0.5f, 0.0f);
		AddVec3("顶点3", 0.0f, 0.5f, 0.0f);
		AddColor("颜色左下", 1.0f, 0.0f, 0.0f);
		AddColor("颜色右下", 0.0f, 1.0f, 0.0f);
		AddColor("颜色上", 0.0f, 0.0f, 1.0f);
	}

	void GetParticles() override {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				points[i * 3 + j] = attrs_[i].value[j];
				printf("%f ", points[i * 3 + j]);
			}
			printf("\n");
		}
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				colors[i * 3 + j] = attrs_[i + 3].value[j];
				printf("%f ", colors[i * 3 + j]);
			}
			printf("\n");
		}

		glDeleteBuffers(1, &vbo);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(float), points, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		glDeleteBuffers(1, &cbo);
		glGenBuffers(1, &cbo);
		glBindBuffer(GL_ARRAY_BUFFER, cbo);
		glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(float), colors, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDeleteVertexArrays(1, &vao);
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, cbo);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	void RenderParticles(const Camera& camera) override {
		shader_->use();
		shader_->setMat4("view", camera.GetViewMatrix());
		shader_->setMat4("projection", camera.GetProjectionMatrix());
		glBindVertexArray(vao);
		// draw points 0-3 from the currently bound VAO with current in-use shader
		glDrawArrays(GL_TRIANGLES, 0, 3);
		glBindVertexArray(0);
	}

public:
	~NormalFirework() {
		delete[] points;
		delete[] colors;
		glDeleteBuffers(1, &vbo);
		glDeleteBuffers(1, &cbo);
	}
};