#ifndef POINT_H
#define POINT_H

#include <cmath>
#include <iostream>
#include "macros.h"
#include "types.h"

/* 
 * Basic 2d point stucture. 
 */
struct Point {
	real x[2];

	Point(real x0=0.0, real x1=0.0) {
		x[0] = x0;
		x[1] = x1;
	}

	Point(const Point& p) {
		x[0] = p.x[0];
		x[1] = p.x[1];
	}

	Point& operator=(const Point& p) {
		x[0] = p.x[0];
		x[1] = p.x[1];

		return *this;
	}

	void print() {
		std::cout << "(" << x[0] << ", " << x[1] << ")";
	}
	
};

real dist(Point a, Point b);

#endif
