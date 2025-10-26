#include "circle.h"

/*
 * Contructs the circumcircle of a triangle from 3 given points
 * and returns a circle struct, which is the circumcircle.
 */	

__host__ __device__ void circumcircle(Point a, Point b, Point c, Point* center, REAL* r) {

	REAL ba0 = b.x[0] - a.x[0];
	REAL ba1 = b.x[1] - a.x[1];
	REAL ca0 = c.x[0] - a.x[0];
	REAL ca1 = c.x[1] - a.x[1];

	REAL det = ba0*ca1 - ca0*ba1;

	if (det == 0.0) {
		printf("DET=0 | a: (%f, %f), b: (%f, %f), c: (%f, %f)\n", a.x[0], a.x[1], b.x[0], b.x[1], c.x[0], c.x[1]);
	}

	det = 0.5 / det;
	REAL asq = ba0*ba0 + ba1*ba1;
	REAL csq = ca0*ca0 + ca1*ca1;
	REAL ctr0 = det*(asq*ca1 - csq*ba1);
	REAL ctr1 = det*(csq*ba0 - asq*ca0);

	*r = sqrt(ctr0*ctr0 + ctr1*ctr1);
	center->x[0] = ctr0 + a.x[0];
	center->x[1] = ctr1 + a.x[1];
}

/*
 * Checks whether the point d is inside the circle created by the points a, b and c. Returns a positve number if
 * d lies on the inside, 0 if on the circle ad a negativee number if it lies on the outside.
 */
__host__ __device__ REAL incircle(Point d, Point a, Point b, Point c){
	// +: inside  | flip
	// 0: on      |
	// -: outside | dont flip

	Point center;
	REAL rad;
	circumcircle(a, b, c, &center, &rad);

	// distance from center to d
	REAL dist_sqr = (d.x[0] - center.x[0])*(d.x[0] - center.x[0]) 
				   + (d.x[1] - center.x[1])*(d.x[1] - center.x[1]); 

	return (rad*rad - dist_sqr);
}

__host__ __device__ void circumcircle_rad(Point a, Point b, Point c, REAL* r) {

	REAL ba0 = b.x[0] - a.x[0];
	REAL ba1 = b.x[1] - a.x[1];
	REAL ca0 = c.x[0] - a.x[0];
	REAL ca1 = c.x[1] - a.x[1];

	REAL det = ba0*ca1 - ca0*ba1;

	if (det == 0.0) {
		printf("DET=0 | a: (%f, %f), b: (%f, %f), c: (%f, %f)\n", a.x[0], a.x[1], b.x[0], b.x[1], c.x[0], c.x[1]);
	}

	det = 0.5 / det;
	REAL asq = ba0*ba0 + ba1*ba1;
	REAL csq = ca0*ca0 + ca1*ca1;
	REAL ctr0 = det*(asq*ca1 - csq*ba1);
	REAL ctr1 = det*(csq*ba0 - asq*ca0);
	*r = sqrt(ctr0*ctr0 + ctr1*ctr1);
}

__host__ __device__ void circumcircle_center(Point a, Point b, Point c, Point* center) {

	REAL ba0 = b.x[0] - a.x[0];
	REAL ba1 = b.x[1] - a.x[1];
	REAL ca0 = c.x[0] - a.x[0];
	REAL ca1 = c.x[1] - a.x[1];

	REAL det = ba0*ca1 - ca0*ba1;

	if (det == 0.0) {
		printf("DET=0 | a: (%f, %f), b: (%f, %f), c: (%f, %f)\n", a.x[0], a.x[1], b.x[0], b.x[1], c.x[0], c.x[1]);
	}

	det = 0.5 / det;
	REAL asq = ba0*ba0 + ba1*ba1;
	REAL csq = ca0*ca0 + ca1*ca1;
	REAL ctr0 = det*(asq*ca1 - csq*ba1);
	REAL ctr1 = det*(csq*ba0 - asq*ca0);

	center->x[0] = ctr0 + a.x[0];
	center->x[1] = ctr1 + a.x[1];
}
