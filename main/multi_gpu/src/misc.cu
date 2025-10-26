#include "misc.h"

/*
 * Adds a value 'val' to each element of an array.
 */
__global__ void arrayAddVal(int* array, int* val, int mult, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n) {
		array[idx] += mult * (*val);
	}
}

__global__ void arraySubVal(int* array, int* val, int mult, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n) {
		array[idx] -= mult * (*val);
	}
}

__global__ void sortPass(int* array, int n, int parity, int* sorted) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * idx + parity;
	int minval, maxval;

    if (i + 1 < n) {
        int l = array[i];
        int r = array[i + 1];

		maxval = max(l, r);
		minval = min(l, r);

		array[i] = maxval;
		array[i + 1] = minval;

		atomicOr(sorted, !(l==maxval));  // Mark that a swap happened
    }
}


/*
 * @param array Device pointer to array.
 * @param n Device pointer to length of array.
 */
void gpuSort(int* array, int* n) {
	thrust::device_ptr<int> thrust_ptr_d(array);

	int N;
    cudaMemcpy(&N, n, sizeof(int), cudaMemcpyDeviceToHost);
	if (N <= 1) {
		return;
	}

    thrust::sort(thrust_ptr_d, thrust_ptr_d + N, thrust::greater<int>());
}

///*
// * @param array Device pointer to array.
// * @param n Device pointer to length of array.
// */
//void gpuSort(int* array, int* n) {
//    int N;
//    cudaMemcpy(&N, n, sizeof(int), cudaMemcpyDeviceToHost);
//
//	if (N <= 1) {
//		return;
//	}
//
//    int sorted[1];
//    int* sorted_d;
//    cudaMalloc(&sorted_d, sizeof(int));
//
//    dim3 threadsPerBlock(32);
//    dim3 numBlocks((N / 2 + threadsPerBlock.x - 1) / threadsPerBlock.x);
//
//    do {
//        *sorted = 0;
//        cudaMemcpy(sorted_d, sorted, sizeof(int), cudaMemcpyHostToDevice);
//        sortPass<<<numBlocks, threadsPerBlock>>>(array, N, 0, sorted_d);
//        sortPass<<<numBlocks, threadsPerBlock>>>(array, N, 1, sorted_d);
//        cudaMemcpy(sorted, sorted_d, sizeof(int), cudaMemcpyDeviceToHost);
//    } while (*sorted != 0); // continue while swaps occurred
//
//    cudaFree(sorted_d);
//}
