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
	int  nTriWithInsert[1]; int* nTriWithInsert_d;
	int* triWithInsert    ; int* triWithInsert_d; 

	int nTriToFlip[1]; int* nTriToFlip_d;
	int* triToFlip   ; int* triToFlip_d;

	int iter = 0; int* iter_d;
	bool verbose = false; // gives detail info to std out about state of the triangulation

	FILE* file;

	Delaunay(Point* points, int n);
	~Delaunay();

	int ntpb = 128; // number of threads per block
	void compute();
	
	void initSuperTri();
	void prepForInsert();
	void insert();

	void flipAfterInsert();
	void storeTriToFlip();
	void checkFlipAndLegality();
	void checkFlipConflicts();

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
__global__ void checkFlipKernel(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri, Point* pts);
__device__ int checkFlip(int a, int flip_edge, int b, Tri* triList); 

__global__ void checkLegalityKernel(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri, Point* pts);
__device__ int checkLegality(int a, int flip_edge, int b, Tri* triList, Point* pts); 

__global__ void prepForConflicts(Tri* triList, int* nTri);
__global__ void setConfigIdx(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri);
__global__ void storeNonConflictConfigs(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri);

__global__ void resetTriToFlipThisIter(Tri* triList, int* nTri);
__global__ void markTriToFlipThisIter(int* triToFlip, int* nTriToFlip, Tri* triList);

__global__ void flipKernel(int* triToFlip, int* nTriToFlip, Tri* triList);
__device__ void flip(int a, int e, int b, Tri* triList);

__global__ void resetTriToFlip(Tri* triList, int* nTri);

/* UPDATE POINTS */
__global__ void updatePointLocationsKernel(Point* pts, int* npts, Tri* triList, int* nTri, int* ptToTri);
__device__ int contains(int t, int r, Tri* triList, Point* pts);


__device__ void printQuad(int* p, int* n, int* o);
#endif


