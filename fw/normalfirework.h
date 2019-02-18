#pragma once
#include "firework.h"


// useful definations
#define AddValue(_name, _value) \
	attrs_.push_back(Attr(L##_name, _value));
#define AddColor(_name, _r, _g, _b) \
	attrs_.push_back(Attr(L##_name, glm::vec3(_r, _g, _b)));

class NormalFirework : public FwBase {
	friend FwBase* getFirework(FireWorkType type, float* args);
	GLuint vbo = 0;
	GLuint cbo = 0;
	GLuint vao = 0;
	float* points;
	float* colors;
private:
	NormalFirework(float* args) {
		points = new float[9]{
			-0.5f, -0.5f, 0.0f,
			0.5f, -0.5f, 0.0f,
			0.0f,  0.5f, 0.0f
		};
		colors = new float[9];
		AddColor("颜色左下", 1.0f, 0.0f, 0.0f);
		AddColor("颜色右下", 0.0f, 1.0f, 0.0f);
		AddColor("颜色上", 0.0f, 0.0f, 1.0f);
	}

	void GetParticles() override {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				colors[i * 3 + j] = attrs_[i].color[j];
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

	void RenderParticles() override {
		printf("here\n");
		shader_->use();
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