#include "tri.h"

void Tri::writeTri(Point* pts, int npts, Point triPts[3], int triNeighbours[3], int triOpposite[3]) {

	for (int i=0; i<3; ++i) {
		p[i] = triPts[i];
		n[i] = triNeighbours[i];
		o[i] = triOpposite[i];
	}

	find_pts_inside(pts, npts);

	//std::cout << "printing lpts: ";
//	for (int i=0; i<nlpts; ++i) {
//		lpts[i].print();
//	}
	//std::cout << "\n";

	get_center();
	
	tag++;
}

// if there is not center reutrn -1 else reutrn the index to pts list
/* 
 *
 */
void Tri::find_pts_inside(Point* pts_to_check, int npts) {

	if (isAllocated_lpts == true) {
		delete[] lpts;
	}

	Point* temp_lpts = new Point[npts];
	nlpts = 0;

	for (int k=0; k<npts; ++k) {
		real area;
		
		// check if point is inside this triangle
		for (int i=0; i<3; ++i) {
			int j = (i+1) % 3;
			area = (p[j].x[0] - p[i].x[0])*(pts_to_check[k].x[1] - p[i].x[1]) - 
			       (p[j].x[1] - p[i].x[1])*(pts_to_check[k].x[0] - p[i].x[0]);

			if (area <= 0) {
				break;
			}
		}

		// if this is true then point is not inside triangle 
		if (area <= 0) {
			continue;
		}

		temp_lpts[nlpts] = pts_to_check[k]; 
//		std::cout << "pt to save: <<  pts_to_check[k] << "\n"; 
//		temp_lpts[nlpts] =

		nlpts++;
	}

	lpts = new Point[nlpts];
	isAllocated_lpts = true;
	//std::cout << "nlpts: " << nlpts << "\n";
	for (int i=0; i<nlpts; ++i) {
		lpts[i] = temp_lpts[i];
	}

	delete[] temp_lpts;
}

/*
 * Sets the 'center' (the point nearest the circumcenter). Returns the index in the array
 * lpts of the 'center' point and -1 if this triangle has no center.
 * Only to be called by the set triangle function.
 */
int Tri::get_center() {
	// calculute actual center of circumcircle for comparison
	Circle cc = circumcircle(p[0], p[1], p[2]);
	Point true_center = cc.center;

	if (nlpts == 0) {
		return -1;
	}
	center = 0;

	// loop through all points
	for (int k=0; k<nlpts; ++k) {
		if (dist(lpts[k], true_center) <= dist(lpts[center], true_center)) {
			// check if its closer to the center than prevoius point
			center = k;
			//std::cout << "CLOSER\n";
		}
	}

	return center;
}

void Tri::print() {
	std::cout << "| points    : (" << p[0].x[0] << ", " << p[0].x[1] << "), (" << p[1].x[0] << ", " << p[1].x[1] << "), (" << p[2].x[0] << ", " << p[2].x[0] << "), "
			  << "| neighbours: " << n[0] << ", " << n[1] << ", " << n[2]
			  << "| opposite  : " << o[0] << ", " << o[1] << ", " << o[2]
			  << "\n";
}
