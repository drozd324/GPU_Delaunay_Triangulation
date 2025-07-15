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
	int p[3];  // indexes of points in pts list
	int n[3]; // idx to Tri neighbours of this triangle
	int o[3]; // index in the Tri noted by the int n[i] of opposite point of current Tri

	bool insert;   
	int insertPt;  
	float insertPt_dist; 
	int flip; // marks an edge to flip 0,1 or 2, -1 if not to flip any edge
	int flipThisIter;
	int configIdx;

};

struct Quad {
	int p[4];  // indexes of points in pts list
	int n[4]; // idx to Tri neighbours of this triangle
	int o[4]; // index in the Tri noted by the int n[i] of opposite point of current Tri
};

struct Node {
	// -1 if inactive, >=0 if active
	int t;  // triangle (int indexing triangle in triList)
	int d[3]; // indexes of daugnter nodes in nodes array
};

#endif
