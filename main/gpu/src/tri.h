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

	bool insert = false;   
	int insertPt;  
	float insertPt_dist; 
	Point circumcenter; 
	int flip; // marks an edge to flip 0,1 or 2, -1 if not to flip any edge

	int node; // index of node in nodes array

	void writeCircumenter();

//	~Tri() { 
//		if (lpts_alloc == true) { delete[] lpts; }
//		if (spts_alloc == true) { delete[] spts; }
//	}

//	void writeTri(Point* gpts, int ngpts, int* searchpts, int nsearchpts, int triPts[3], int triNeighbours[3], int triOpposite[3]);
//
//	int contains(Point point);
//	void find_pts_inside();
//	int get_center();
//
//	void print();
};

struct Node {
	// -1 if inactive, >=0 if active
	int t;  // triangle (int indexing triangle in triList)
	int d[3]; // indexes of daugnter nodes in nodes array
};

#endif
