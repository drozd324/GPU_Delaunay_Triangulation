#include <stdio.h>
#include <stdlib.h>

__device__ int* array;

__host__ __device__ void print_array(int* vec, int n) {

    for (int i = 0; i < n; ++i) {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

__global__ void allocArray(int* a) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx == 0) {

		a[0]++;

		array = new int[5];
		for (int i=0; i<5; ++i) {
			array[i] = i;
		}
	}
}

__global__ void printAndFree(int* a) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx == 0) {

		a[0]++;

		print_array(array, 5);
		delete[] array;
	}

}

int main() {
	int N = 100;
	int a[1] = {0};
	int *a_d;

	cudaMalloc(&a_d, sizeof(int));
	cudaMemcpy(a_d, a, sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(32);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));

	printf("before alloc: %d\n", a[0]);

	allocArray<<<numBlocks, threadsPerBlock>>>(a_d);
	cudaMemcpy(a, a_d, sizeof(int), cudaMemcpyDeviceToHost);
	printf("after alloc: %d\n", a[0]);

    cudaDeviceSynchronize();

	printAndFree<<<numBlocks, threadsPerBlock>>>(a_d);
	cudaMemcpy(a, a_d, sizeof(int), cudaMemcpyDeviceToHost);
	printf("after free: %d\n", a[0]);	

    cudaDeviceSynchronize();

    return 0;
}
