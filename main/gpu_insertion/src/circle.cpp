#include "circle.h"

/*
 * Contructs the circumcircle of a triangle from 3 given points
 * and returns a circle struct, which is the circumcircle.
 */	
Circle circumcircle(Point a, Point b, Point c) {
	// equation (21.3.7) in "Numerical Recipeies"
	// equation (21.3.8)

	real ba0, ba1, ca0, ca1;
	real asq, csq;
	real ctr0, ctr1, rad; // center0, center1, radius
	real det; 

	ba0 = b.x[0] - a.x[0];
	ba1 = b.x[1] - a.x[1];
	ca0 = c.x[0] - a.x[0];
	ca1 = c.x[1] - a.x[1];

	det = ba0*ca1 - ca0*ba1;

	if (det == 0.0) {
		std::cout << "points ((" << a.x[0]  << "," << a.x[1] << "), (" << b.x[0] << "," << b.x[1] << "), (" << c.x[0] << "," << c.x[1] << ")" << " | "
				  << ba0 << "," << ca1 << "," << ca0 << "," << ba1 << "\n";
	}

	det = 0.5 / det;
	asq = SQR(ba0) + SQR(ba1);
	csq = SQR(ca0) + SQR(ca1);
	ctr0 = det*(asq*ca1 - csq*ba1);
	ctr1 = det*(csq*ba0 - asq*ca0);
	rad = sqrt(SQR(ctr0) + SQR(ctr1));

	return Circle(Point(ctr0 + a.x[0], ctr1 + a.x[1]), rad);
}

/*
 *	
 */
real incircle(Point d, Point a, Point b, Point c){
	// +: inside  | flip
	// 0: on      |
	// -: outside | dont flip

	Circle cc = circumcircle(a, b, c);
	// distance from center to d
	real dist_sqr = SQR(d.x[0] - cc.center.x[0]) + SQR(d.x[1] - cc.center.x[1]); 

	return SQR(cc.radius) - dist_sqr;
}
