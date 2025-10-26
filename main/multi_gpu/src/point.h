#ifndef POINT_H
#define POINT_H

#include "math.h"
#include "types.h"

/* 
 * Basic 2d point stucture. 
 */
struct Point {
	REAL x[2];
};

//__host__ __device__ inline REAL dist(Point a, Point b);


__host__ __device__ inline REAL dist(Point a, Point b) {
	return sqrtf( (a.x[0] - b.x[0])*(a.x[0] - b.x[0])
			    + (a.x[1] - b.x[1])*(a.x[1] - b.x[1]) );
}
#endif
