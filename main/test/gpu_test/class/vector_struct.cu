#include <stdio.h>
#include <math.h>
#define SQR(x) ((x)*(x)) 

// Size of array
#define N 10

struct Point {
	double x[2];

	__host__ __device__ Point(double x0=0.0, double x1=0.0) {
		x[0] = x0;
		x[1] = x1;
	}
};

__host__ __device__ double dist(Point a, Point b) {
	return sqrt( SQR(a.x[0] - b.x[0]) + SQR(a.x[1] - b.x[1]) );
}

__global__ void pts_dist(Point* a, Point* b, double* c) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) { 
		c[id] = dist(a[id], b[id]);
	}
}

void print_array(double* array, int n) {
	for (int i=0; i<n; ++i) {
		printf("%lf ", array[i]);
	}
	printf("\n");
}

void print_array_point(Point* array, int n) {
	for (int i=0; i<n; ++i) {
		printf("(%lf, %lf) ", array[i].x[0], array[i].x[1]);
	}
	printf("\n");
}

// Main program
int main() {
    // Number of bytes to allocate for N doubles

    Point *a = (Point*)malloc(N * sizeof(Point));
    Point *b = (Point*)malloc(N * sizeof(Point));
    double *c = (double*)malloc(N * sizeof(double));

    for(int i=0; i<N; i++) {
        a[i].x[0] = 0;
        a[i].x[1] = 1;
        b[i].x[0] = 1;
        b[i].x[1] = 0;
    }

    Point *d_a;
	Point *d_b;
	double *d_c;
    cudaMalloc(&d_a, N * sizeof(Point));
    cudaMalloc(&d_b, N * sizeof(Point));
    cudaMalloc(&d_c, N * sizeof(double));

    cudaMemcpy(d_a, a, N * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(Point), cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    pts_dist<<<blk_in_grid, thr_per_blk >>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, N * sizeof(double), cudaMemcpyDeviceToHost);

	print_array_point(a, N);
	print_array_point(b, N);
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
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");

    return 0;
}
