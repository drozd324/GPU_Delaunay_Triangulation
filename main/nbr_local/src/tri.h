#ifndef TRI_H
#define TRI_H

#include <iostream>
#include <fstream>

#include "types.h"
#include "point.h" 
#include "circle.h"

/*
 * Data structure needed for Point instertion algorithm. Its main features are
 * that it holds a pointer to an array of points which will be used for the triangulation,
 * the index of those points as ints which form this triangle, its daughter triangles 
 * which are represented as ints which belong to an array of all triangle elements and
 * whether this triangle is used in the trianglulation constructed so far.
 */
struct Tri {
	//Point* pts; int npts;
	Point* lpts; int nlpts;
	bool isAllocated_lpts = false; 

	Point p[3]; // indexes of points in pts list
	int n[3]; // idx to Tri neighbours of this triangle
	int o[3]; // index in the Tri noted by the int n[i] of opposite point of current Tri
	int center = -1;
	int tag = 0;

	Tri() {}
	~Tri() {
		if (isAllocated_lpts == true) {
			delete[] lpts;
		}
	}

	void writeTri(Point* pts, int npts, Point triPts[3], int triNeighbours[3], int triOpposite[3]);
	void find_pts_inside(Point* pts_to_check, int npts);
	int get_center();
	void print();

};

#endif
