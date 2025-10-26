#ifndef ATOMIC_H
#define ATOMIC_H

#include "types.h"

// Atomic CUDA functions for single precision numbers 
__device__ float atomicAddFloat(float* address, float val);
__device__ float atomicMaxFloat(float* address, float val);
__device__ float atomicMinFloat(float* address, float val);

// Atomic CUDA functions for double precision numbers 
__device__ double atomicAddDouble(double* address, double val);
__device__ double atomicMaxDouble(double* address, double val);
__device__ double atomicMinDouble(double* address, double val);

#endif
