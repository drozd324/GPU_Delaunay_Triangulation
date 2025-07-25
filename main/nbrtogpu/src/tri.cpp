#include "tri.h"

/*
 * Writes key data to the triangle 'Tri' struct. 
 * 
 * @param gpts Global points. Every point involved in the tringulation
 * @param ngpts Number of global points.
 * @param triPts An array to intergers indexing the location to 3 points in the initialis
 * @param triNeighbours 
 * @param triOpposite
 */
void Tri::writeTri(Point* gpts, int ngpts, int* searchpts, int nsearchpts,
			int triPts[3], int triNeighbours[3], int triOpposite[3])
{
	pts = gpts;
	npts = ngpts;
	
	for (int i=0; i<3; ++i) {
		p[i] = triPts[i];
		n[i] = triNeighbours[i];
		o[i] = triOpposite[i];
	}

	tag++;
	flip = -1;
	

	// needed for flipping
	if (spts_alloc == true) {
		delete[] spts;
		nspts = 0;
		spts_alloc = false;
	}

	if (nsearchpts > 0) { 
		nspts = nsearchpts;
		//std::cout << "nsearchpts: " << nsearchpts << "\n";
		spts = new int[nsearchpts];
		for (int i=0; i<nsearchpts; ++i) {
			spts[i] = searchpts[i];
		}
		spts_alloc = true;
	} 

	get_center();
}

/* 
 * Checks whether the point "point" is contained in this triangle. Computed by making 
 * use of the property of barycentric coordinates of a triangle. In this algorithm
 * this is reducted to checking any of the constants determining this coordinate 
 * transform is negative, if so, then return -1 which means this point is not
 * contained inside this triangle. Otherwise the point is on the boundary 
 * and returns 0 or outside and returns 1. 
 */
int Tri::contains(Point point) {
	// Return 1 if point is in the triangle, 0 if on boundary, 1 if outside. (CCW triangle is assumed.)
	real area;
	int i, j, ztest=0;

	for (i=0; i<3; ++i) {
		j = (i+1) % 3;
		// area = area of triangle (21.3.2) (21.3.10) 
		area = (pts[p[j]].x[0] - pts[p[i]].x[0])*(point.x[1] - pts[p[i]].x[1]) - 
			(pts[p[j]].x[1] - pts[p[i]].x[1])*(point.x[0] - pts[p[i]].x[0]);

		if (area < 0.0) {
			return -1;
		}
		if (area == 0.0) {
			ztest = 1;
		}
	}

	return (ztest? 0:1);
}

///*
// * Finds points inside this triangle from an array of intergers indexing points in the global array.
// * 
// * @param spts Array of 'search points'. We check whether these points are inside this triangle.
// * @param nspts Lenght of array.
// */
//void Tri::find_pts_inside() {
//
//	if (lpts_alloc == true) {
//		delete[] lpts;
//		lpts_alloc = false;
//	}
//	
//	nlpts = 0;
//	int* temp_lpts = new int[nspts];
//
//	// loop through all points
//	for (int k=0; k<nspts; ++k) { 
//		// if this is true then point is not inside triangle 
//		if (contains(pts[spts[k]]) <= 0) {
//			continue;
//		}
//
//		temp_lpts[nlpts] = spts[k];
//		nlpts++;
//	}
//
//	lpts = new int[nlpts];
//	lpts_alloc = true;
//
//	for (int i=0; i<nlpts; ++i) {
//		lpts[i] = temp_lpts[i];
//	}
//
//	delete[] temp_lpts;
//}
//
//
//
//int Tri::get_center() {
//
//	find_pts_inside();
//
//	// calculute actual center of circumcircle for comparison
//	Circle cc = circumcircle(pts[p[0]], pts[p[1]], pts[p[2]]);
//	Point true_center = cc.center;
//
//	center = -1;
//	for (int k=0; k<nlpts; ++k) { 
//		if (k == 0 || (dist(pts[lpts[k]], true_center) < dist(pts[center], true_center)) ) {
//			// check if its closer to the center than prevoius point
//			center = lpts[k];
//		}
//	}
//
//	//delete[] lpts;
//	return center;
//}


int Tri::get_center() {

	if (lpts_alloc == true) {
		delete[] lpts;
		lpts_alloc = false;
	}
	
	nlpts = 0;

	if (spts_alloc == false) {
		return -1;
	}

	// loop through all points
	for (int k=0; k<nspts; ++k) { 
		// if this is true then point is not inside triangle 
		if (contains(pts[spts[k]]) <= 0) {
			continue;
		}

		//temp_lpts[nlpts] = spts[k];
		spts[nlpts] = spts[k];
		nlpts++;
	}

	//std::cout << "nlpts: " << nlpts << "\n";
	lpts = new int[nlpts];
	lpts_alloc = true;

	for (int i=0; i<nlpts; ++i) {
		lpts[i] = spts[i];
	}

	// calculute actual center of circumcircle for comparison
	Circle cc = circumcircle(pts[p[0]], pts[p[1]], pts[p[2]]);
	Point true_center = cc.center;

	center = -1;
	for (int k=0; k<nlpts; ++k) { 
		if (k == 0 || (dist(pts[lpts[k]], true_center) < dist(pts[center], true_center)) ) {
			// check if its closer to the center than prevoius point
			center = lpts[k];
		}
	}

	//delete[] lpts;
	return center;
}







void Tri::print() {
	std::cout << "| points    : " << p[0] << ", " << p[1] << ", " << p[2]
			  << "| neighbours: " << n[0] << ", " << n[1] << ", " << n[2]
			  << "| opposite  : " << o[0] << ", " << o[1] << ", " << o[2]
			  << "\n";
}
