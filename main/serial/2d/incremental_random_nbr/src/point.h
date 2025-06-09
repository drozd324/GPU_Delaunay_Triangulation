#ifndef POINT_H
#define POINT_H

#include "types.h"
#include <cmath>

/* 
 * Basic 2d point stucture. 
 */
struct Point {
	real x[2];

	//Point(double x0=0.0, double x1=0.0) : x[0](x0), x[1](x1) {};
	Point(real x0=0.0, real x1=0.0) {
		x[0] = x0;
		x[1] = x1;
	}
};

real dist(Point a, Point b) {
	reuturn sqrt( SQR(a.x[0] - b.x[0]) SQR(a.x[1] - b.x[1]) );
}

#endif
