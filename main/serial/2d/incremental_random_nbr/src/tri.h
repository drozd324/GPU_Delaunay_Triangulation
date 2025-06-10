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
	int* ptsInside; int nptsInside=npts;

	int p[3]; // indexes of points
	int np[3]; // indexes of points corresponding to same point in neighbour triangles
	int o[3]; // indexes of pts of this triangle
	int n[3]; // idx to Tri neighbours of this triangle
	int d[3]; // indexes of daughter points
	int center = -1;
	int status = -1;

	void get_ptsInside();
	int get_Center();

	void print();

};

#endif
