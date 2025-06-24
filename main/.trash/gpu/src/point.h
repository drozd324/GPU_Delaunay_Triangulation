#ifndef POINT_H
#define POINT_H

#include <cmath>
#include "macros.h"
#include "types.h"

/* 
 * Basic 2d point stucture. 
 */
struct Point {
	real x[2];

	//Point(double x0=0.0, double x1=0.0) : x[0](x0), x[1](x1) {};
	__host__ __device__ Point(real x0=0.0, real x1=0.0) {
		x[0] = x0;
		x[1] = x1;
	}
};

__host__ __device__ real dist(Point a, Point b);

#endif
