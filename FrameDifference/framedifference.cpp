#include "framedifference.h"
#include <iostream>
#include <string>
using namespace std;


extern "C" __declspec(dllexport)
void extract(char* name, int len) {
	string fname(name, len);
	Difference dt;
	dt.extract(fname);
}

int main(int argc, char* argv[]) {
	Difference dt;
#pragma omp parallel for
	for (int i = 0; i < 1000; ++i) {
		string name = "";
		if (i < 10) name += "0000" + to_string(i);
		else if (i < 100) name += "000" + to_string(i);
		else if (i < 1000) name += "00" + to_string(i);
		else if (i < 10000) name += "0" + to_string(i);
		else name += to_string(i);
		dt.extract(name);
	}
	return 0;
}
