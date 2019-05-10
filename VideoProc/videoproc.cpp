#include "VideoProc.h"
#include "pieceImages.h"

int main() {
	//VideoProc vp("mv_3");
	//vp.toPics();
	//vp.save();
	size_t starts[8]{4, 7, 6, 10, 16, 4, 11, 29};
	size_t ends[8]{54, 50, 60, 60, 62, 58, 46, 78};
	for (int i = 3; i < 11; ++i) {
		VideoProc vp(std::to_string(i), ".mp4");
		vp.toPics();
		vp.save(starts[i - 3] - 1, ends[i - 3]);
	}
	//{
	//	size_t idx[7]{7, 14, 28, 35, 42, 48};
	//	std::vector<std::string> r;
	//	for (size_t i = 3; i < 4; ++i) {
	//		for (size_t j = 0; j < 6; ++j) {
	//			r.push_back("mv_" + std::to_string(i) + "/" + std::to_string(idx[j]) + ".png");
	//		}
	//	}
	//	pieceImages(r, 1, 6, "mixture3.png");
	//}

}