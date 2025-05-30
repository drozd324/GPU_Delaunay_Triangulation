#ifndef POINT_H
#define POINT_H

#include "types.h"

struct Point {
	real x[2];

	//Point(double x0=0.0, double x1=0.0) : x[0](x0), x[1](x1) {};
	Point(real x0=0.0, real x1=0.0) {
		x[0] = x0;
		x[1] = x1;
	}
};

#endif
