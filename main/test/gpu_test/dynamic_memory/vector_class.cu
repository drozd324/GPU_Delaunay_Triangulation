#include <stdio.h>

int N = 10;



__global__ void add_vectors(int* a, int* b, int n, int* c) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < n) { 
		c[id] = a[id] + b[id];
	}
}

void print_array(int* array, int n) {
	for (int i=0; i<n; ++i) {
		printf("%lf ", array[i]);
	}
	printf("\n");
}


struct VectorAdder {
    int *a_d, *b_d, *c_d;
    int size;

    VectorAdder(int* a, int* b, int n) :  size(n) {
		cudaMalloc(&a_d, size * sizeof(int));
		cudaMalloc(&b_d, size * sizeof(int));
		cudaMalloc(&c_d, size * sizeof(int));

		cudaMemcpy(a_d, a, size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(b_d, b, size * sizeof(int), cudaMemcpyHostToDevice);

		compute();
    }

    ~VectorAdder() {
		cudaFree(a_d);
		cudaFree(b_d);
		cudaFree(c_d);
    }


    void compute() {
		// Set execution configuration parameters
		//      thr_per_blk: number of CUDA threads per grid block
		//      blk_in_grid: number of blocks in grid
		int thr_per_blk = 256;
		int blk_in_grid = ceil( float(size) / thr_per_blk );
		add_vectors<<<blk_in_grid, thr_per_blk>>>(a_d, b_d, size, c_d);
    }

    void getResult(int* c) {
		cudaMemcpy(c, c_d, (size-2) * sizeof(int), cudaMemcpyDeviceToHost);
    }
};


// Main program
int main() {
    // Number of bytes to allocate for N ints
    int *a = new int[N];
    int *b = new int[N];
    int *c = new int[N];

    // Fill host arrays A and B
    for(int i=0; i<N; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }

	VectorAdder vecadd(a, b, N);
	vecadd.getResult(c);

	print_array(a, N);
    print_array(b, N);
	print_array(c, N);

    delete[] a;
    delete[] b;
    delete[] c;
		
    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("N                 = %d\n", N);
//    printf("Threads Per Block = %d\n", thr_per_blk);
//    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");

    return 0;
}
