#ifndef POINT_H
#define POINT_H

#include "math.h"
#include "types.h"

/* 
 * Basic 2d point stucture. 
 */
struct Point {
	float x[2];

//	Point() {
//		x[0] = 0;
//		x[1] = 0;
//	}
//	Point(float a, float b) {
//		x[0] = a;
//		x[1] = a;
//	}

//	Point& operator=(Point& rhs) {
//		x[0] = rhs.x[0];
//		x[1] = rhs.x[1];
//	}
};

__host__ __device__ float dist(Point a, Point b);

#endif
