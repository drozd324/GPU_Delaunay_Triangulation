#include <stdio.h>

// Size of array
#define N 10

// Kernel
__global__ void add_vectors(double* a, double* b, double* c) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) { 
		c[id] = a[id] + b[id];
	}
}

void print_array(double* array, int n) {
	for (int i=0; i<n; ++i) {
		printf("%lf ", array[i]);
	}
	printf("\n");
}

// Main program
int main() {
    // Number of bytes to allocate for N doubles

    double *a = (double*)malloc(N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *c = (double*)malloc(N * sizeof(double));

    for(int i=0; i<N; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(double));
    cudaMalloc(&d_b, N * sizeof(double));
    cudaMalloc(&d_c, N * sizeof(double));

    cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(32);
    dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));
    add_vectors<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, N * sizeof(double), cudaMemcpyDeviceToHost);

	print_array(a, N);
	print_array(b, N);
	print_array(c, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("N                 = %d\n", N);
    printf("Threads Per Block = %d\n", threadsPerBlock);
    printf("Blocks In Grid    = %d\n", numBlocks);
    printf("---------------------------\n\n");

    return 0;
}
