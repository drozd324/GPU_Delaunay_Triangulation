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
		std::cout << "[ERROR][DET = 0] points ((" << a.x[0]  << "," << a.x[1] << "), (" << b.x[0] << "," << b.x[1] << "), (" << c.x[0] << "," << c.x[1] << ")" << " | "
				  << ba0 << "," << ca1 << "," << ca0 << "," << ba1 << "\n";
	}

	det = 0.5 / det;
	//asq = SQR(ba0) + SQR(ba1);
	asq = ba0*ba0 + ba1*ba1;
	//csq = SQR(ca0) + SQR(ca1);
	csq = ca0*ca0 + ca1*ca1;
	ctr0 = det*(asq*ca1 - csq*ba1);
	ctr1 = det*(csq*ba0 - asq*ca0);
	//rad = sqrt(SQR(ctr0) + SQR(ctr1));
	rad = sqrt(ctr0*ctr0 + ctr1*ctr1);

	return Circle(Point(ctr0 + a.x[0], ctr1 + a.x[1]), rad);
}

//Circle circumcircle(Point a, Point b, Point c) {
//	real a0,a1,c0,c1,det,asq,csq,ctr0,ctr1,rad2;
//	a0 = a.x[0] - b.x[0]; a1 = a.x[1] - b.x[1];
//	c0 = c.x[0] - b.x[0]; c1 = c.x[1] - b.x[1];
//	det = a0*c1 - c0*a1;
//	if (det == 0.0) { std::cout << "[ERROR][DET = 0]\n"; }
//	det = 0.5/det;
//	asq = a0*a0 + a1*a1;
//	csq = c0*c0 + c1*c1;
//	ctr0 = det*(asq*c1 - csq*a1);
//	ctr1 = det*(csq*a0 - asq*c0);
//	rad2 = ctr0*ctr0 + ctr1*ctr1;
//	return Circle(Point(ctr0 + b.x[0], ctr1 + b.x[1]), sqrt(rad2));
//}
//

/*
 *	
 */
real incircle(Point d, Point a, Point b, Point c){
	// +: inside  | flip
	// 0: on      |
	// -: outside | dont flip

	Circle cc = circumcircle(a, b, c);
	// distance from center to d
	//real dist_sqr = SQR(d.x[0] - cc.center.x[0]) + SQR(d.x[1] - cc.center.x[1]); 
	real dist_sqr = (d.x[0] - cc.center.x[0])*(d.x[0] - cc.center.x[0]) 
				  + (d.x[1] - cc.center.x[1])*(d.x[1] - cc.center.x[1]); 

	//return SQR(cc.radius)- dist_sqr;
	return (cc.radius*cc.radius - dist_sqr);
}
