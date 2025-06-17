#include "point.h"

__host__ __device__ real dist(Point a, Point b) {
	return sqrt( SQR(a.x[0] - b.x[0]) + SQR(a.x[1] - b.x[1]) );
}
