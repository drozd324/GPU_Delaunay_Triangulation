#include "tri.h"

/*
 * Data structure needed for Point instertion algorithm. Its main features are
 * that it holds a pointer to an array of points which will be used for the triangulation,
 * the index of those points as ints which form this triangle, its daughter triangles 
 * which are represented as ints which belong to an array of all triangle elements and
 * whether this triangle is used in the trianglulation constructed so far.
 */
void Tri::get_ptsInside() {
	int idx_in = 0;
	for (int k=0; k<npts; ++k) {
		for (int i=0; i<3; ++i) {
			int j = (i+1) % 3;
			real area = (pts[p[j]].x[0] - pts[p[i]].x[0])*(pts[k].x[1] - pts[p[i]].x[1]) - 
						(pts[p[j]].x[1] - pts[p[i]].x[1])*(pts[k].x[0] - pts[p[i]].x[0]);

			if (area < 0) {
				break;
			}
		}
		
		ptsInside[idx_in++] = k;
	}

	get_Center();
}

int Tri::get_Center() {
	Circle cc = circumcircle(p[0], p[1], p[2]);
	Point true_center = cc.center;
	center = 0;

	//for (int i=1; i<nptsInside; ++i) {
	for (int i=1; i<npts; ++i) {
		//if (dist(ptsInside[i], true_center) < dist(center, true_center)) {
		if (dist(pts[i], true_center) < dist(pts[center], true_center)) {
			center = i;
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
