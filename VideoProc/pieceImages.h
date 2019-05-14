#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>

inline void pieceImages(std::vector<std::string> fnames,
		int nCol, int nRow, std::string fanme = "result.png",
		size_t row=400, size_t col=400, size_t padding = 20) {
	cv::Mat result(
		padding * (nCol + 1) + col * nCol, padding * (nRow + 1) + row * nRow,
		CV_8UC3, cv::Scalar(255, 255, 255));
	std::cout << result.size() << std::endl;
	for (size_t i = 0; i < nCol; ++i) {
		for (size_t j = 0; j < nRow; ++j) {
			size_t index = i * nRow + j;
			std::cout << i << " " << j << " "<< padding * (j + 1) + row * j  <<" " << padding * (i + 1) + col * i <<  " " << fnames[index] << std::endl;
			cv::Mat imageROI = result(cv::Rect(
				padding * (j + 1) + row * j,
				padding * (i + 1) + col * i,
				row, col));
			cv::Mat r = cv::imread(fnames[index]);
			if (r.size().width != row || r.size().height != col) {
				cv::Mat m(400, 400, CV_8UC3, cv::Scalar(255, 255, 255));
				cv::resize(r, m, cv::Size(row, col));
				m.copyTo(imageROI);
			}
			else {
				r.copyTo(imageROI);
			}
		}	
	}
	cv::imwrite(fanme, result);
}