#ifndef FW_KERNEL_UTILS_TIMER_HPP
#define FW_KERNEL_UTILS_TIMER_HPP

#include<chrono>
#include <queue>
#include <iostream>
#include <string>
using namespace std::chrono;
using std::cout;
using std::endl;

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

class AverageTime {
	std::deque<double> times;
	size_t maxSize_;
public:
	AverageTime(size_t maxSize):maxSize_(maxSize){}

	void clear() {
		std::deque<double> empty;
		std::swap(empty, times);
	}

	void append(double time) {
		if (times.size() < maxSize_) {
			times.push_back(time);
		} else {
			times.pop_front();
			times.push_back(time);
		}
	}

	double averageTime() {
		double t = 0;
		for (auto it = times.begin(); it != times.end(); ++it) {
			t += *it;
		}
		return t / static_cast<double>(times.size());
	}
};

#endif