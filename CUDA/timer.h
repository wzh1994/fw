#ifndef FW_KERNEL_UTILS_TIMER_HPP
#define FW_KERNEL_UTILS_TIMER_HPP

#include<chrono>
#include <iostream>
using namespace std::chrono;
using std::cout;
using std::endl;
namespace cudaKernel {
class Timer {

private:
	time_point<system_clock> start_time;
public:
	void start() {
		start_time = system_clock::now();
	}

	double stop() {
		return duration_cast<milliseconds>(
			system_clock::now() - start_time).count();
	}

	void pstop(std::string s) {
		double r = duration_cast<milliseconds>(
			system_clock::now() - start_time).count();
		cout << s << " cost: " << r << " ms" << endl;
	}
};
}
#endif