#ifndef CIRCLE_H
#define CIRCLE_H

#include <cmath>
#include <iostream>
#include "point.h" 
#include "macros.h"
#include "types.h"

__device__ void circumcircle(Point a, Point b, Point c, Point* center, float* r);
__device__ float incircle(Point d, Point a, Point b, Point c);

__device__ void circumcircle_rad(Point a, Point b, Point c, float* r);
__device__ void circumcircle_center(Point a, Point b, Point c, Point* center);

#endif
