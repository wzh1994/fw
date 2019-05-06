#include "VideoProc.h"
#include "pieceImages.h"

int main() {
	//for (int i = 100; i < 102; ++i) {
	//	VideoProc vp("mv_" + std::to_string(i));
	//	vp.toPics();
	//	//vp.save();
	//}
	{
		size_t idx[7]{27};
		std::vector<std::string> r;
		for (size_t i = 100; i < 102; ++i) {
			for (size_t j = 0; j < 1; ++j) {
				r.push_back("mv_" + std::to_string(i) + "/" + std::to_string(idx[j]) + ".png");
			}
		}
		pieceImages(r, 1, 2, "compareInner.png");
	}

}