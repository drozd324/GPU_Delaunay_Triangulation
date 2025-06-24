#ifndef CIRCLE_H
#define CIRCLE_H

#include <cmath>
#include <iostream>
#include "point.h" 
#include "macros.h"
#include "types.h"

/*
 * Basic circle struct.
 */
struct Circle {
	Point center;
	real radius;

	__host__ __device__ Circle(const Point &cen, real rad) : center(cen), radius(rad) {};
};

__host__ __device__ Circle circumcircle(Point a, Point b, Point c);
__host__ __device__ real incircle(Point d, Point a, Point b, Point c);

#endif
