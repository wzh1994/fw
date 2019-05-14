#include "VideoProc.h"
#include "pieceImages.h"

void picToVideo(std::vector<std::string> names, std::string outName) {
	cv::Mat m = cv::imread("fws/" + names[0]);
	cv::imshow("1", m);
	cv::waitKey();
	cv::VideoWriter video(outName + ".avi", cv::CAP_ANY, 24.0, cv::Size(m.cols, m.rows));
	for (const auto& i : names) {
		video << cv::imread("fws/" + i);
	}
	video.release();
}

void convPic(std::string name) {
	cv::Mat m = cv::imread(name + ".png");
	cv::Mat m2(m.cols, m.rows, CV_8UC1);
	cv::cvtColor(m.clone(), m2, CV_BGR2GRAY);
	cv::threshold(m2.clone(), m2, 0, 255, cv::THRESH_OTSU);
	cv::Mat m3(m.cols, m.rows, CV_8UC3);
	m.copyTo(m3, m2);
	cv::imshow("1", m3);
	cv::waitKey();
}

int main() {
	/*VideoProc vp("fw", ".mp4");
	vp.toPics();*/
	/*picToVideo(std::vector<std::string>{
		"109.png", "111.png", "113.png", "115.png", "117.png", "119.png", "121.png", "123.png", "125.png", "127.png", "129.png", "131.png", "133.png", "135.png", "137.png", "139.png", "141.png", "143.png", "145.png", "147.png", "149.png", "151.png", "153.png", "155.png", "157.png", "159.png", "161.png", "163.png", "165.png", "167.png", "169.png", "171.png", "173.png", "175.png", "177.png", "179.png", "181.png", "183.png", "185.png", "187.png", "189.png", "191.png", "193.png", "194.png", "195.png", "196.png", "197.png", "198.png", "199.png"},
		"fw");*/
	/*vp.save(11, 70);*/
	/*size_t starts[8]{4, 7, 6, 10, 16, 4, 11, 29};
	size_t ends[8]{54, 50, 60, 60, 62, 58, 46, 78};*/
	for (int i = 1; i < 2; ++i) {
		VideoProc vp("mv_" + std::to_string(i), ".avi");
		vp.toPics(true);
		//vp.save(starts[i - 3] - 1, ends[i - 3]);
	}
	{
		size_t idx[7]{4, 6, 8, 10, 12, 14};
		std::vector<std::string> r;
		for (size_t i = 1; i < 2; ++i) {
			for (size_t j = 0; j < 6; ++j) {
				r.push_back("mv_" + std::to_string(i) + "/" + std::to_string(idx[j]) + ".png");
			}
		}
		pieceImages(r, 1, 6, "result1.png");
	}
	// convPic("1");
}