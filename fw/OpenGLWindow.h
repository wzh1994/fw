#pragma once
#include "OpenGLWidget.h"

class OpenGLWindow : public OpenGLWidget {
public:
	OpenGLWindow() = default;
	~OpenGLWindow() = default;
	bool Initialize() override {

		return true;
	}
	void RenderScene() override {
		glClearColor(1, 0.2, 1.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		float points[] = {
			-0.5f, -0.5f, 0.0f,
			 0.5f, -0.5f, 0.0f,
			 0.0f,  0.5f, 0.0f
		};
		GLuint vbo = 0;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(float), points, GL_STATIC_DRAW);
		GLuint vao = 0;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		const char* vertex_shader =
			"#version 400\n"
			"in vec3 vp;"
			"void main() {"
			"  gl_Position = vec4(vp, 1.0);"
			"}";
		const char* fragment_shader =
			"#version 400\n"
			"out vec4 frag_colour;"
			"void main() {"
			"  frag_colour = vec4(0.5, 0.0, 0.5, 1.0);"
			"}";
		GLuint vs = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vs, 1, &vertex_shader, NULL);
		glCompileShader(vs);
		GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs, 1, &fragment_shader, NULL);
		glCompileShader(fs);
		GLuint shader_programme = glCreateProgram();
		glAttachShader(shader_programme, fs);
		glAttachShader(shader_programme, vs);
		glLinkProgram(shader_programme);
		glUseProgram(shader_programme);
		glBindVertexArray(vao);
		// draw points 0-3 from the currently bound VAO with current in-use shader
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}
};