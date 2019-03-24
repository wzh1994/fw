#pragma once
#include<chrono>
#include <iostream>
using namespace std::chrono;
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
		printf("%s cost: %f ms\n", s.c_str(), r);
	}
};
