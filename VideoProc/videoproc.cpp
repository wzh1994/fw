#include "VideoProc.h"
#include "pieceImages.h"

int main() {
	for (int i = 10; i < 12; ++i) {
		VideoProc vp("mv_" + std::to_string(i));
		vp.toPics();
		//vp.save();
	}
	{
		size_t idx[7]{3, 10, 17, 24, 31, 37, 44};
		std::vector<std::string> r;
		for (size_t i = 10; i < 12; ++i) {
			for (size_t j = 0; j < 7; ++j) {
				r.push_back("mv_" + std::to_string(i) + "/" + std::to_string(idx[j]) + ".png");
			}
		}
		pieceImages(r, 2, 7, "dlModelResult.png");
	}

}