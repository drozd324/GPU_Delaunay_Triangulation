#ifndef TRI_H
#define TRI_H

#include <iostream>
#include <fstream>
#include <format>

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
	Point* pts; int npts;

	int p[3]; // indexes of points in pts list
	int n[3]; // idx to Tri neighbours of this triangle
	int o[3]; // index in the Tri noted by the int n[i] of opposite point of current Tri
	int center = -1;
	int status = -1;
	int tag = -1;

	int get_center();

	void print();

};

#endif
