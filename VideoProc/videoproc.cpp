#include "VideoProc.h"
#include "pieceImages.h"

int main() {
	VideoProc vp("mv_2");
	vp.toPics();
	//vp.save();
	//for (int i = 0; i < 2; ++i) {
	//	VideoProc vp("mv_" + std::to_string(i));
	//	vp.toPics();
	//	//vp.save();
	//}
	{
		size_t idx[7]{7, 14, 28, 35, 42, 48};
		std::vector<std::string> r;
		for (size_t i = 2; i < 3; ++i) {
			for (size_t j = 0; j < 6; ++j) {
				r.push_back("mv_" + std::to_string(i) + "/" + std::to_string(idx[j]) + ".png");
			}
		}
		pieceImages(r, 1, 6, "renderShow.png");
	}

}