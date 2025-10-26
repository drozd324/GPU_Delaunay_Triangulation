#ifndef CIRCLE_H
#define CIRCLE_H

#include <cmath>
#include <iostream>
#include "point.h" 
#include "macros.h"
#include "types.h"

__host__ __device__ void circumcircle(Point a, Point b, Point c, Point* center, REAL* r);
__host__ __device__ REAL incircle(Point d, Point a, Point b, Point c);

__host__ __device__ void circumcircle_rad(Point a, Point b, Point c, REAL* r);
__host__ __device__ void circumcircle_center(Point a, Point b, Point c, Point* center);

#endif
