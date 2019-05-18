#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <string>
#include <windows.h>
#include <iostream>

using namespace std;
using namespace cv;

class Difference {
public:
	Difference() = default;

	void extract(string videoName) {
		diffs.clear();
		oris.clear();
		diffs.reserve(100);
		oris.reserve(100);
		getDiffs(videoName);
		string topath = "extract/original";
		string tdpath = "extract/difference";
		for (int i = 0; i < diffs.size(); ++i) {
			string index = (i < 10 ? "00" : i < 100 ? "0" : "") + to_string(i);
			Mat m(400, 400, CV_8UC3, Scalar(0, 0, 0));
			resize(oris[i], m, Size(400, 400));
			imwrite(topath + "/" + index + ".jpg", m);
		}
		for (int i = 0; i < diffs.size(); ++i) {
			string index = (i < 10 ? "00" : i < 100 ? "0" : "") + to_string(i);
			Mat m(400, 400, CV_8UC3, Scalar(0, 0, 0));
			resize(diffs[i], m, Size(400, 400));
			imwrite(tdpath + "/" + index + ".jpg", m);
		}
	}

private:
	vector<Mat> diffs;
	vector<Mat> oris;
	void getDiffs(string path, int startFrame = 0, int endFrame = 48) {
		VideoCapture capture = VideoCapture(path);
		capture.set(cv::CAP_PROP_POS_FRAMES, startFrame); //设置开始帧为1
		Mat lastFrame;
		Mat currentFrame;
		Mat current, last;
		//读第一帧
		capture.read(lastFrame);
		
		{  // ostu
			cv::Mat m1(lastFrame.cols, lastFrame.rows, CV_8UC1, cv::Scalar(0));
			cv::cvtColor(lastFrame.clone(), m1, CV_BGR2GRAY);
			cv::threshold(m1.clone(), m1, 0, 255, cv::THRESH_OTSU);
			cv::Mat m2(m1.cols, m1.rows, CV_8UC3, cv::Scalar(0, 0, 0));
			lastFrame.copyTo(m2, m1);
			m2.copyTo(lastFrame);
		}

		diffs.push_back(lastFrame.clone());
		oris.push_back(lastFrame.clone());
		cvtColor(lastFrame, last, cv::COLOR_BGR2GRAY);
		threshold(last, last, 0, 255, cv::THRESH_OTSU);
		for (int i = startFrame + 1; i <= endFrame; i++) {
			capture.read(currentFrame);
			{  // ostu
				cv::Mat m1(currentFrame.cols, currentFrame.rows, CV_8UC1, cv::Scalar(0));
				cv::cvtColor(currentFrame.clone(), m1, CV_BGR2GRAY);
				cv::threshold(m1.clone(), m1, 0, 255, cv::THRESH_OTSU);
				cv::Mat m2(m1.cols, m1.rows, CV_8UC3, cv::Scalar(0, 0, 0));
				currentFrame.copyTo(m2, m1);
				m2.copyTo(currentFrame);
			}
			oris.push_back(currentFrame.clone());
			cvtColor(currentFrame, current, cv::COLOR_BGR2GRAY);
			threshold(current, current, 0, 255, cv::THRESH_OTSU);
			getDiffAndChange(current, last, currentFrame);
			diffs.push_back(currentFrame.clone());
			current.copyTo(last);
		}
	}
	void getDiffAndChange(Mat src1, Mat src2, Mat& dif) {
		//计算差值
		int nRows = dif.rows;
		int nCols = dif.cols;
		uchar *p, *q, *r;
		for (int j = 0; j < nRows; ++j)
		{
			p = dif.ptr<uchar>(j);
			q = src1.ptr<uchar>(j);
			r = src2.ptr<uchar>(j);
			for (int k = 0; k < nCols; ++k)
			{
				p[3 * k] = q[k] > r[k] ? p[3 * k] : 0;
				p[3 * k + 1] = q[k] > r[k] ? p[3 * k + 1] : 0;
				p[3 * k + 2] = q[k] > r[k] ? p[3 * k + 2] : 0;
			}
		}
	}
};
