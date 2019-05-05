#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <vector>
#include <string>
#include "../fw/exceptions.h"

class VideoProc {
	using string_t = std::string;
	cv::VideoCapture cap_;
	std::vector<cv::Mat> photos_;
	size_t totalFrame_;
	string_t fname_;
	int x_, y_;
	int width_, height_;
	size_t currFrame_;
public:
	VideoProc(string_t name);

	template <class T>
	void saveIter(T start,T end) {
		cv::VideoWriter video(fname_ + ".avi", cv::CAP_ANY, 24.0, cv::Size(width_, height_));
		for (T s = start; s != end; ++s) {
			video << *s;
		}
	}

	void save(int start, int end) {
		FW_ASSERT(start < end);
		saveIter(photos_.begin() + start, photos_.begin() + end);
	}

	void save(int start = 0) {
		save(start, totalFrame_);
	}

	void reverse_save(int start, int end) {
		FW_ASSERT(start < end);
		int length = end - start;
		int rOffset = totalFrame_ - end;
		saveIter(photos_.rbegin() + rOffset, photos_.rbegin() + rOffset + length);
	}

	void reverse_save(int start = 0) {
		reverse_save(start, totalFrame_);
	}
	
	void rFrame() {
		if (currFrame_ < totalFrame_ - 1) {
			++currFrame_;
			std::cout << currFrame_ << std::endl;
			show();
		}
	}

	void lFrame() {
		if (currFrame_ > 1) {
			--currFrame_;
			std::cout << currFrame_ << std::endl;
			show();
		}
	}

	void move(int x, int y) {
		x_ += x;
		y_ += y;
		if (x_ > height_ - 400) {
			x = height_ - 400;
		}
		if (x_ < 0) {
			x_ = 0;
		}
		if (y_ > width_ - 400) {
			y = width_ - 400;
		}
		if (y_ < 0) {
			y_ = 0;
		}
		std::cout << "(" << x_ << "," << y_ << ")" << std::endl;
	}

	void show() {
		size_t width = std::min(400, width_ - y_);
		size_t height = std::min(400, height_ - x_);
		width = std::min(width, height);
		height = width;
		cv::Mat imageROI = photos_[currFrame_](cv::Rect(x_, y_, height, width));
		cv::imshow(fname_.c_str(), imageROI);
	}

	void toPics() {
		for (size_t i = 0; i < totalFrame_; ++i) {
			string_t name = fname_ + "/" + std::to_string(i) + ".png";
			cv::imwrite(name, photos_[i]);
		}
	}
};

void opencv_mouse_callback(int event, int x, int y, int flags, void* ustc) {
	VideoProc* vp = static_cast<VideoProc*>(ustc);
	static int lx, ly;
	if (event == CV_EVENT_LBUTTONDOWN) {
		lx = x;
		ly = y;
	}
	else if (event == CV_EVENT_LBUTTONUP) {
		lx = lx - x;
		ly = ly - y;
		vp->move(lx, ly);
		vp->show();
	}
	if (event == CV_EVENT_LBUTTONDBLCLK) {
		vp->lFrame();
	}
	else if (event == CV_EVENT_RBUTTONDBLCLK) {
		vp->rFrame();
	}
}

VideoProc::VideoProc(string_t name) : fname_(name), x_(0), y_(0), currFrame_(0) {
	cap_.open(name + ".avi");
	FW_ASSERT(cap_.isOpened()) << "Error open movies given: " << name;
	totalFrame_ = cap_.get(cv::CAP_PROP_FRAME_COUNT);
	width_ = cap_.get(cv::CAP_PROP_FRAME_WIDTH);
	height_ = cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
	for (size_t i = 0; i < totalFrame_; ++i) {
		cv::Mat frame;
		cap_ >> frame;
		FW_ASSERT(!frame.empty()) << sstr("Error get frames: ", i);
		photos_.push_back(frame);
	}
	cvNamedWindow(fname_.c_str(), 0);//设置窗口名
	cvResizeWindow(fname_.c_str(), width_, height_);
	// 设置回调函数，用于拾色器鼠标移动事件
	cv::setMouseCallback(fname_.c_str(), opencv_mouse_callback, this);
}
