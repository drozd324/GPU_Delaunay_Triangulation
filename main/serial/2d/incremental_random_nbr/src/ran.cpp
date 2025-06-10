#include "ran.h"

unsigned long long Ran::int64() {
	// Return 64-bit random integer. See text for explanation of method.
	u = u * 2862933555777941757LL + 7046029254386353087LL;
	v ^= v >> 17;
	v ^= v << 31;
	v ^= v >> 8;
	w = 4294957665U*(w & 0xffffffff) + (w >> 32);
	unsigned long long x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
	return (x + v) ^ w;
}

double Ran::doub() {
	// Return random double-precision floating value in the range 0. to 1.
	return 5.42101086242752217e-20 * int64();
}

unsigned int Ran::int32() {
	// Return 32-bit random integer.
	return (unsigned int)int64(); 
}


void shuffle(int* array, int n, int seed=69420) {
	Ran ran(seed); 
	int temp, ran_i;
	for (int i=0; i<n; ++i) {
		ran_i = ran.int64() % n;
		temp = array[ran_i];
		array[ran_i] = array[i];
		array[i] = temp;
	}
}
