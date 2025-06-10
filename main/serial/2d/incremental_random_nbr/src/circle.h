#ifndef CIRCLE_H
#define CIRCLE_H

#include <cmath>
#include "point.h" 
#include "macros.h"
#include "types.h"

/*
 * Basic circle struct.
 */
struct Circle {
	Point center;
	real radius;

	//Circle() : center(Point()), radius(1) {};
	Circle(const Point &cen, real rad) : center(cen), radius(rad) {};
};

Circle circumcircle(Point a, Point b, Point c);
real incircle(Point d, Point a, Point b, Point c);

#endif
