#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10000

__global__ void my_atomic(float* vect, float* added) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) { 
        atomicAdd(added, vect[idx]);
    }
}

void print_array(float* array, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

int main() {
    float added = 0.0f;
    float* a = (float*)malloc(N * sizeof(float));

    float expected_num = 0;
    for(int i = 0; i < N; i++) {
        a[i] = (float)i;
        expected_num += i;
    }

    //print_array(a, N);

    // Device allocations
    float *a_d, *added_d;
    cudaMalloc(&a_d, N * sizeof(float));
    cudaMalloc(&added_d, sizeof(float));

    // Copy data to device
    cudaMemcpy(a_d, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(added_d, &added, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int thr_per_blk = 256;
    int blk_in_grid = (int)ceil((float)N / thr_per_blk);

    printf("Before: %f\n", added);

    my_atomic<<<blk_in_grid, thr_per_blk>>>(a_d, added_d);
    cudaDeviceSynchronize();  // Always good for debugging

    // Copy result back
    cudaMemcpy(&added, added_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("After: %f (Expected: %f)\n", added, expected_num);

    // Cleanup
    cudaFree(a_d);
    cudaFree(added_d);
    free(a);

    return 0;
}

