#ifndef DELAUNAY_H
#define DELAUNAY_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>

#include "macros.h"
#include "types.h"
#include "mymath.h"
#include "point.h" 
#include "circle.h"
#include "tri.h"
#include "atomic.h"
#include "misc.h"

/*
 * Struct for creating a delaunay triangulation from a given vector of points. Consists of 
 */
struct Delaunay {
	int    npts[1] ; int* npts_d;
	Point* pts     ; Point* pts_d;

	int  nTri   [1]; int* nTri_d; 
	int  nTriMax[1]; int* nTriMax_d; 
	Tri* triList   ; Tri* triList_d; 

	int* ptToTri          ; int* ptToTri_d;
	int* triWithInsert    ; int* triWithInsert_d; 
	int  nTriWithInsert[1]; int* nTriWithInsert_d;

	int* nTriToFlip; int* nTriToFlip_d;
	int* triToFlip ; int* triToFlip_d;

	int iter = 0; int* iter_d;

	int num_tris_to_insert; int* num_tris_to_insert_d;

	FILE* file;

	Delaunay(Point* points, int n);
	~Delaunay();

	int ntpb = 128;
	void compute();
	
	void initSuperTri();
	void prepForInsert();
	void insert();
	void flipAfterInsert();

	void printInfo();
	void printTri();

	void updatePointLocations();
	void saveToFile(bool end=false);
};

__host__ __device__ void writeTri(Tri* tri, int* p, int* n, int* o);

/* INIT */
__global__ void sumPoints(Point* pts, int* npts, Point* avgPoint);
__global__ void computeMaxDistPts(Point* pts, int* npts, float* largest_dist);

/* PREP FOR INSERT */
__global__ void resetInsertPtInTris(Tri* triList, int* nTriMax);
__global__ void setInsertPtsDistance(Point* pts, int* npts, Tri* triList, int* ptToTri);
__global__ void setInsertPts        (Point* pts, int* npts, Tri* triList, int* ptToTri);
__global__ void prepTriWithInsert(Tri* triList, int* nTri, int* triWithInsert, int* nTriWithInsert);

/* INSERT */
__global__ void insertKernel(Tri* triList, int* nTri, int* triWithInsert, int* nTriWithInsert, int* ptToTri);
__device__ int insertInTri(int i, Tri* triList, int newTriIdx, int* ptToTri);
__device__ int insertPtInTri(int r, int i, Tri* triList, int newTriIdx, int* ptToTri);
__global__ void checkInsertPoint(Tri* triList, int* triWithInsert, int* nTriWithInsert);
__global__ void resetBiggestDistInTris(Tri* triList, int* nTriMax);

/* FLIP */
__global__ void flipKernel(int* triToFlip, int* nTriToFlip, Tri* triList);
__device__ int flip(int a, Tri* triList);
__global__ void storeTriToFlip(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri, Point* pts);

/* UPDATE POINTS */
__global__ void updatePointLocationsKernel(Point* pts, int* npts, Tri* triList, int* nTri, int* ptToTri);
__device__ int contains(int t, int r, Tri* triList, Point* pts);

/* MISC */
//__global__ void arrayAddVal(int* array, int* val, int mult, int n);
//
//void gpuSort(int* array, int* n);
//__global__ void sortPass(int* array, int n, int parity, int* sorted);

#endif
