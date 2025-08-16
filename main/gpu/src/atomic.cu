#include "atomic.h"

__device__ float atomicAddFloat(float* address, float val) {
	int* address_as_ull = (int*)address;
	int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __float_as_int(val + __int_as_float(assumed)));
	} while (assumed != old);
	return __int_as_float(old);
}

__device__ float atomicMaxFloat(float* address, float val) {
	int* address_as_ull = (int*)address;
	int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __float_as_int(max(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__device__ float atomicMinFloat(float* address, float val) {
	int* address_as_ull = (int*)address;
	int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __float_as_int(min(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}




__device__ double atomicAddDouble(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__device__ double atomicMaxDouble(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(max(val,__longlong_as_double(assumed))));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__device__ double atomicMinDouble(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(min(val,__longlong_as_double(assumed))));
	} while (assumed != old);
	return __longlong_as_double(old);
}

