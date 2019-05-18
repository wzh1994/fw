
// play.cpp: 使用glfw窗口播放烟花效果。
//

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <thread>
#include "firework.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include "kernels.h"
#include "utils.h"
#include "shader.h"
#include "timer.h"

// constant global variables
namespace {
	const int screenWidth = 400;
	const int screenHeight = 400;
	GLFWwindow* window;
}

void pushImg(std::vector<cv::Mat>& imgs) {
	cv::Mat img(400, 400, CV_8UC3);
	//use fast 4-byte alignment (default anyway) if possible
	glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
	//set length of one complete row in destination data (doesn't need to equal img.cols)
	glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());
	glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_UNSIGNED_BYTE, img.data);
	cv::flip(img, img, 0);
	imgs.push_back(img);
}

void init(size_t w = screenWidth, size_t h = screenHeight) {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 8);

	window = glfwCreateWindow(w, h, "Firework", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	// Options
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		FW_NO_THROW(NotInitialized) << "Init failed!";
	};
	glEnable(GL_DEPTH_TEST);
}

void playInNewThread(firework::FireWorkType type,
		float* args, bool genVideo, std::string fname) {
	init();

	firework::FwBase* fw = firework::getFirework(type, args);
	fw->initialize();
	fw->prepare();
	Camera c(screenWidth, screenHeight);
	int i = 0;
	std::vector<cv::Mat> imgs;
	// glPolygonMode(GL_FRONT, GL_LINE);
	Timer timer1;
	Timer timer2;
	AverageTime averageTime1(49);
	AverageTime averageTime2(49);
	while (!glfwWindowShouldClose(window)) {
		timer1.start();
		timer2.start();
		glfwPollEvents();
		glfwSwapBuffers(window);
		glClearColor(0, 0, 0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_MULTISAMPLE);
		fw->GetParticles(i++);
		averageTime1.append(timer1.stop());
		fw->RenderScene(c);
		averageTime2.append(timer2.stop());
		cout << averageTime1.averageTime() << " " << averageTime2.averageTime() << endl;
		if (genVideo) {
			pushImg(imgs);
		}
		if (i == fw->getTotalFrame()) {
			if (genVideo)
				break;
			else
				i = 0;
		}
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);
	}

	if (genVideo) {
		cv::VideoWriter writer(fname + ".avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
			24.0, cv::Size(400, 400));
		for (int i = 0; i < imgs.size(); i++)
			writer << imgs[i];
	}
	glfwTerminate();
}

extern "C" __declspec(dllexport)
void play(firework::FireWorkType type, float* args) {
	playInNewThread(type, args, false, "");
}

extern "C" __declspec(dllexport)
void playAndSave(firework::FireWorkType type, float* args, const char* str, size_t len) {
	std::string fname(str, len);
	playInNewThread(type, args, true, fname);
}

void showPointToLine() {
	init(1200, 1200);
	GLuint vbo=0, ebo=0, vao=0;
	size_t vboSize = 1000000, eboSize = 1000000;
	struct cudaGraphicsResource *cuda_vbo_resource_, *cuda_ebo_resource_;

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER,
		vboSize * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	CUDACHECK(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_, vbo,
		cudaGraphicsMapFlagsWriteDiscard));

	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
		eboSize * sizeof(int), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	CUDACHECK(cudaGraphicsGLRegisterBuffer(&cuda_ebo_resource_, ebo,
		cudaGraphicsMapFlagsWriteDiscard));

	CUDACHECK(cudaDeviceSynchronize());

	void *pVboData, *pEboData;
	size_t sizeVbo, sizeEbo;
	CUDACHECK(cudaGraphicsMapResources(1, &cuda_vbo_resource_, 0));
	CUDACHECK(cudaGraphicsMapResources(1, &cuda_ebo_resource_, 0));
	CUDACHECK(cudaGraphicsResourceGetMappedPointer(
		&pVboData, &sizeVbo, cuda_vbo_resource_));
	CUDACHECK(cudaGraphicsResourceGetMappedPointer(
		&pEboData, &sizeEbo, cuda_ebo_resource_));
	CUDACHECK(cudaDeviceSynchronize());

	float points[15]{-0.3, 0, 0, -0.15, 0, 0, 0, 0.03, 0, 0.15, 0.09, 0, 0.3, 0.06, 0.06};
	float colors[15]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0 };
	float sizes[5]{ 0.2, 0.1, 0.08, 0.12, 0.16};
	size_t offsets[2]{ 0, 5 };
	float *dPoints, *dColors, *dSizes;
	size_t *dOffsets;
	cudaKernel::cudaMallocAndCopy(dPoints, points, 15);
	cudaKernel::cudaMallocAndCopy(dColors, colors, 15);
	cudaKernel::cudaMallocAndCopy(dSizes, sizes, 5);
	cudaKernel::cudaMallocAndCopy(dOffsets, offsets, 2);
	size_t newEboSize = cudaKernel::pointToLine(dPoints, dSizes, dColors,
		5, dOffsets, 1, static_cast<float*>(pVboData),
		static_cast<uint32_t*>(pEboData), 1, 0, 0);

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource_, 0);
	cudaGraphicsUnmapResources(1, &cuda_ebo_resource_, 0);

	glDeleteVertexArrays(1, &vao);
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
		7 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE,
		7 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	Camera camera(1200, 1200);
	Shader shader("fw.vs", "fw.fs");
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glfwSwapBuffers(window);
		glClearColor(1, 1, 1, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_MULTISAMPLE);
		shader.use();
		shader.setMat4("view", camera.GetViewMatrix());
		shader.setMat4("projection", camera.GetProjectionMatrix());
		glBindVertexArray(vao);
		// draw points 0-3 from the currently bound VAO with current in-use shader;
		//glEnable(GL_BLEND);
		//glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
		//glBlendEquation(GL_MAX);
		glDepthMask(GL_TRUE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawElements(GL_TRIANGLES, newEboSize, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}
	glfwTerminate();
}

int main() {
	std::thread t(showPointToLine);
	t.join();
}
