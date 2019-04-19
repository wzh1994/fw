
// play.cpp: 使用glfw窗口播放烟花效果。
//

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <thread>
#include "firework.h"

// constant global variables
namespace {
	const int screenWidth = 600;
	const int screenHeight = 600;
	GLFWwindow* window;
}

void init() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	window = glfwCreateWindow(screenWidth, screenHeight, "LearnOpenGL", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	// Options
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		FW_THROW(NotInitialized) << "Init failed!";
	};
	glEnable(GL_DEPTH_TEST);
}

void playInNewThread() {
	init();
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glfwSwapBuffers(window);
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
	glfwTerminate();
}

extern "C" __declspec(dllexport) void play() {
	playInNewThread();
}

int main() {
	std::thread t(playInNewThread);
	t.join();
}
