#include <stdio.h>
#include <stdlib.h>

#define N 10000

__global__ void my_atomic(int* vect, int* sum) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(sum, vect[idx]);
    }
}

int main() {
    int sum[1] = {0};
    int* a = (int*)malloc(N * sizeof(int));

    int expected_sum = 0;
    for (int i = 0; i < N; i++) {
        a[i] = i;
        expected_sum += i;
    }

    int *a_d, *sum_d;
    cudaMalloc(&a_d, N * sizeof(int));
    cudaMalloc(&sum_d, sizeof(int));

    cudaMemcpy(a_d, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sum_d, sum, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    printf("Before: %d\n", sum[0]);

    my_atomic<<<numBlocks, threadsPerBlock>>>(a_d, sum_d);
    cudaDeviceSynchronize();

    cudaMemcpy(sum, sum_d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("After: %d (Expected: %d)\n", sum[0], expected_sum);

    cudaFree(a_d);
    cudaFree(sum_d);
    free(a);

    return 0;
}

