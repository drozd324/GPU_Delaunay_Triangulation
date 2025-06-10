#include "tri.h"

// if there is not center reutrn -1 else reutrn the index to pts list
int Tri::get_center() {
	
	// calculute actual center of circumcircle for comparison
	Circle cc = circumcircle(pts[p[0]], pts[p[1]], pts[p[2]]);
	Point true_center = cc.center;

	center = -1;
	int npts_inside = 0;

	// loop through all points
	for (int k=0; k<npts; ++k) { 
		real area;
		
		// check if point is inside this triangle
		for (int i=0; i<3; ++i) {
			int j = (i+1) % 3;
			area = (pts[p[j]].x[0] - pts[p[i]].x[0])*(pts[k].x[1] - pts[p[i]].x[1]) - 
			        (pts[p[j]].x[1] - pts[p[i]].x[1])*(pts[k].x[0] - pts[p[i]].x[0]);

			if (area <= 0) {
				break;
			}
		}

		// if this is true then point is not inside triangle 
		if (area <= 0) {
			continue;
		}

		npts_inside++;

		if (npts_inside == 1) {
			center = k;
		}
		else if (dist(pts[k], true_center) < dist(pts[center], true_center)) {
			// check if its closer to the center than prevoius point
			center = k;
		}
	}

	return center;
}

void Tri::print() {
	std::cout << "| stat      : " << status
			  << "| points    : " << p[0] << ", " << p[1] << ", " << p[2]
			  << "| daughters : " << d[0] << ", " << d[1] << ", " << d[2]
			  << "| opposite  : " << o[0] << ", " << o[1] << ", " << o[2]
			  << "| neighbours: " << o[0] << ", " << o[1] << ", " << o[2]
			  << "\n";
}
