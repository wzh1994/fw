#include "bfc.h"
#include <iostream>
using namespace std;

//int main() {
//	memory::BfcAllocator& a = memory::cudaAllocator();
//	void* r1 = a.allocate(100);
//	cout << a.allocatedBytes() << " " << a.cachedBytes() << endl;
//	void* r2 = a.allocate(100000000);
//	cout << a.allocatedBytes() << " " << a.cachedBytes() << endl;
//	a.release(r2);
//	cout << a.allocatedBytes() << " " << a.cachedBytes() << endl;
//	a.release(r1);
//	cout << a.allocatedBytes() << " " << a.cachedBytes() << endl;
//}