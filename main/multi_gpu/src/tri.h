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
 * whether this triangle is used in the trianglulation constructed so far. Aligned to
 * 64 bytes for more efficient accesing of memory.
 */
struct __align__(64)  Tri {
	int p[3]; // indexes of points in pts list
	int n[3]; // idx to Tri neighbours of this triangle
	int o[3]; // index in the Tri noted by the int n[i] of opposite point of current Tri

	int insert;         // takes values 0 or 1 for marking if it shouldn't or should respectively be inserted into 
	int insertPt;       // the index of the point to insert
	REAL insertPt_dist; // entry for the minimum distance between point and circumcenter
	int flip;           // marks an edge to flip 0,1 or 2
	int flipThisIter;   // mark whether this triangle should flip in the current iteration of flipping
	int configIdx;      // the minimum index for both triangles which could be involved in a flip  
};

/*
 * Data structure used in the flipping algorithm. This qualrilateral data structure
 * holds information about the intermediate state of two triangles involved in a configuration
 * currently being flipped. This struct is used in the construction of the two new triangles
 * created and in the updating of neighbouring triangles data. Aligned to 64 bytes for more
 * efficient accesing of memory.
 */
struct __align__(64) Quad  {
	int p[4]; // indexes of points in pts list
	int n[4]; // idx to Tri neighbours across the edge
	int o[4]; // index in the Tri noted by the int n[i] of opposite point of current Tri
}; 

#endif
