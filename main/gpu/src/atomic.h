#ifndef ATOMIC_H
#define ATOMIC_H

/*
	Atomic CUDA functions for floating point numbers 
*/

__device__ float atomicAddFloat(float* address, float val);
__device__ float atomicMaxFloat(float* address, float val);
__device__ float atomicMinFloat(float* address, float val);

#endif
