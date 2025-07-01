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
//struct Circle {
//	Point center;
//	float radius;
//};

//__host__ __device__ inline void circumcircle(Point a, Point b, Point c, Point* center, float* r);
//__host__ __device__ inline float incircle(Point d, Point a, Point b, Point c);
//
//__host__ __device__ inline void circumcircle_rad(Point a, Point b, Point c, float* r);
//__host__ __device__ inline void circumcircle_center(Point a, Point b, Point c, Point* center);


/*
 * Contructs the circumcircle of a triangle from 3 given points
 * and returns a circle struct, which is the circumcircle.
 */	

__device__ inline void circumcircle(Point a, Point b, Point c, Point* center, float* r) {

	float ba0 = b.x[0] - a.x[0];
	float ba1 = b.x[1] - a.x[1];
	float ca0 = c.x[0] - a.x[0];
	float ca1 = c.x[1] - a.x[1];

	float det = ba0*ca1 - ca0*ba1;

	if (det == 0.0) {
		printf("DET=0");
	}

	det = 0.5 / det;
	float asq = ba0*ba0 + ba1*ba1;
	float csq = ca0*ca0 + ca1*ca1;
	float ctr0 = det*(asq*ca1 - csq*ba1);
	float ctr1 = det*(csq*ba0 - asq*ca0);

	*r = sqrt(ctr0*ctr0 + ctr1*ctr1);
	center->x[0] = ctr0 + a.x[0];
	center->x[1] = ctr1 + a.x[1];
}

/*
 *	
 */
__device__ inline float incircle(Point d, Point a, Point b, Point c){
	// +: inside  | flip
	// 0: on      |
	// -: outside | dont flip

	Point center;
	float rad;
	circumcircle(a, b, c, &center, &rad);

	// distance from center to d
	//float dist_sqr = SQR(d.x[0] - cc.center.x[0]) + SQR(d.x[1] - cc.center.x[1]); 
	float dist_sqr = (d.x[0] - center.x[0])*(d.x[0] - center.x[0]) 
				   + (d.x[1] - center.x[1])*(d.x[1] - center.x[1]); 

	//return SQR(cc.radius)- dist_sqr;
	return (rad*rad - dist_sqr);
}

__device__ inline void circumcircle_rad(Point a, Point b, Point c, float* r) {

	float ba0 = b.x[0] - a.x[0];
	float ba1 = b.x[1] - a.x[1];
	float ca0 = c.x[0] - a.x[0];
	float ca1 = c.x[1] - a.x[1];

	float det = ba0*ca1 - ca0*ba1;

	if (det == 0.0) {
		printf("DET=0");
	}

	det = 0.5 / det;
	float asq = ba0*ba0 + ba1*ba1;
	float csq = ca0*ca0 + ca1*ca1;
	float ctr0 = det*(asq*ca1 - csq*ba1);
	float ctr1 = det*(csq*ba0 - asq*ca0);
	*r = sqrt(ctr0*ctr0 + ctr1*ctr1);
}

__device__ inline void circumcircle_center(Point a, Point b, Point c, Point* center) {

	float ba0 = b.x[0] - a.x[0];
	float ba1 = b.x[1] - a.x[1];
	float ca0 = c.x[0] - a.x[0];
	float ca1 = c.x[1] - a.x[1];

	float det = ba0*ca1 - ca0*ba1;

	if (det == 0.0) {
		printf("DET=0");
	}

	det = 0.5 / det;
	float asq = ba0*ba0 + ba1*ba1;
	float csq = ca0*ca0 + ca1*ca1;
	float ctr0 = det*(asq*ca1 - csq*ba1);
	float ctr1 = det*(csq*ba0 - asq*ca0);

	center->x[0] = ctr0 + a.x[0];
	center->x[1] = ctr1 + a.x[1];
}

#endif
