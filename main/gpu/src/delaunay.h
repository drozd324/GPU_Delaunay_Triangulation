#ifndef DELAUNAY_H
#define DELAUNAY_H

#include <iostream>
#include <fstream>
#include <math.h>

#include "macros.h"
#include "types.h"
#include "mymath.h"
#include "point.h" 
#include "circle.h"
#include "tri.h"

/*
 * Struct for creating a delaunay triangulation from a given vector of points. Consists of 
 */
struct Delaunay {
	int    npts, npts_d;
	Point* pts , pts_d;

	int  nTri   , nTri_d; 
	int  nTriMax, nTriMax_d; 
	Tri* triList, triList_d; 

	int num_tris_to_insert, num_tris_to_insert_d;
	Point avgPoint, avgPoint_d;

	std::ofstream saveFile;

	Delaunay(Point* points, int n);
	~Delaunay();

	// compute options
	void gpu_compute();
	
	__device__ void initSuperTri();

	// point insertion functions
	__device__ int checkInsert();
	__device__ int insert(int i);
	__device__ int insert();
	__device__ int insertInTri(int i);
	__device__ int insertPtInTri(int r, int i);

	// flipping functions
	__device__ int flip(int a, int edge);
	__device__ int flip_after_insert();

	void saveToFile(bool end=false);

	int iter = 0;
	int tag_num = 0;
};

#endif
