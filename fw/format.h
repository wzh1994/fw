#pragma once
#include <sstream>

#define FW_STRING(x) #x
#define FW_MSTRING(x) FW_STRING(x)

#define FW_CONCATD(x1, x2) x1##x2
#define FW_CONCAT(x1, x2) FW_CONCATD(x1, x2)

template<class A>
inline void insert_to_stream(std::ostream& os, A&& x) {
	os << x;
}

template<class A, class... Rest>
inline void insert_to_stream(std::ostream& os, A&& x, Rest&&... rest) {
	os << x;
	insert_to_stream(os, std::forward<Rest>(rest)...);
}

template<class... Args>
inline std::string sstr(Args&&... args) {
	std::ostringstream ss;
	insert_to_stream(ss, std::forward<Args>(args)...);
	return ss.str();
}