
// play.cpp: 使用glfw窗口播放烟花效果。
//

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <thread>
#include "firework.h"
#include <opencv2/opencv.hpp>
#include <vector>

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


void init() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 8);

	window = glfwCreateWindow(screenWidth, screenHeight, "Firework", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	// Options
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		FW_THROW(NotInitialized) << "Init failed!";
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
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glfwSwapBuffers(window);
		glClearColor(0, 0, 0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_MULTISAMPLE);
		fw->GetParticles(i++);
		fw->RenderScene(c);
		if (genVideo) {
			pushImg(imgs);
		}
		if (i == fw->getTotalFrame()) {
			if (genVideo)
				break;
			else
				i = 0;
		}
	}
	if (genVideo) {
		//CV_FOURCC('M', 'J', 'P', 'G');
		cv::VideoWriter writer(fname + ".avi", cv::CAP_ANY, 24.0, cv::Size(400, 400));
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

int main() {
	float* args = new float[1000]{
		0.572549045085907, 0.250980406999588, 0.0941176488995552, 0.5098039507865906, 0.250980406999588, 0.16862745583057404, 0.5529412031173706, 0.3490196168422699, 0.2823529541492462, 0.572549045085907, 0.250980406999588, 0.0941176488995552, 0.5098039507865906, 0.250980406999588, 0.16862745583057404, 0.5529412031173706, 0.3490196168422699, 0.2823529541492462, 0.572549045085907, 0.250980406999588, 0.0941176488995552, 0.5098039507865906, 0.250980406999588, 0.16862745583057404, 0.5529412031173706, 0.3490196168422699, 0.2823529541492462, 0.572549045085907, 0.250980406999588, 0.0941176488995552, 0.5098039507865906, 0.250980406999588, 0.16862745583057404, 0.5529412031173706, 0.3490196168422699, 0.2823529541492462, 0.572549045085907, 0.250980406999588, 0.0941176488995552, 0.5098039507865906, 0.250980406999588, 0.16862745583057404, 0.5529412031173706, 0.3490196168422699, 0.2823529541492462, 0.572549045085907, 0.250980406999588, 0.0941176488995552, 0.5098039507865906, 0.250980406999588, 0.16862745583057404, 0.5529412031173706, 0.3490196168422699, 0.2823529541492462, 0.572549045085907, 0.250980406999588, 0.0941176488995552, 0.5098039507865906, 0.250980406999588, 0.16862745583057404, 0.5529412031173706, 0.3490196168422699, 0.2823529541492462, 0.572549045085907, 0.250980406999588, 0.0941176488995552, 0.5098039507865906, 0.250980406999588, 0.16862745583057404, 0.5529412031173706, 0.3490196168422699, 0.2823529541492462, 0.572549045085907, 0.250980406999588, 0.0941176488995552, 0.5098039507865906, 0.250980406999588, 0.16862745583057404, 0.5529412031173706, 0.3490196168422699, 0.2823529541492462, 0.572549045085907, 0.250980406999588, 0.0941176488995552, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 0.8799999952316284, 0.9300000071525574, 0.0, 10.0, 0.0, 35.0, 0.15000000596046448, 28.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.800000011920929, 0.800000011920929, 0.800000011920929, 0.800000011920929, 0.800000011920929, 0.800000011920929, 0.800000011920929, 0.800000011920929, 0.800000011920929, 0.800000011920929, 0.800000011920929, 0.7599999904632568, 0.7200000286102295, 0.6800000071525574, 0.6399999856948853, 0.6000000238418579, 0.5600000023841858, 0.5199999809265137, 0.47999998927116394, 0.4399999976158142, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.4000000059604645, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, 70.0, 70.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.800000011920929, 0.0, 0.0, 10.0, 0.0, 35.0, 0.699999988079071, 50.0
	};
	std::thread t(playInNewThread, firework::FireWorkType::Mixture, args, false, "");
	delete[] args;
	t.join();
}
