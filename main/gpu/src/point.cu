#include "point.h"

__host__ __device__ float dist(Point a, Point b) {
	return sqrtf( (a.x[0] - b.x[0])*(a.x[0] - b.x[0])
			    + (a.x[1] - b.x[1])*(a.x[1] - b.x[1]) );
}
