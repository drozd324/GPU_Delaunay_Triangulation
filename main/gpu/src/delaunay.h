#ifndef DELAUNAY_H
#define DELAUNAY_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctime>

#include <iostream>
#include <fstream>
#include <random>	

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

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
	int    npts[1]        ; int* npts_d;
	Point* pts            ; Point* pts_d;
	int nptsInserted[1]   ; int* nptsInserted_d;
	int* ptsUninserted    ; int* ptsUninserted_d; 
	int nptsUninserted[1] ; int* nptsUninserted_d; 

	int  nTri   [1]	 ; int* nTri_d; 
	int  nTriMax[1]  ; int* nTriMax_d; 
	Tri* triList     ; Tri* triList_d; 
	Tri* triList_prev; Tri* triList_prev_d; 

	int* ptToTri          ; int* ptToTri_d;
	int  nTriWithInsert[1]; int* nTriWithInsert_d;
	int* triWithInsert    ; int* triWithInsert_d; 

	int nTriToFlip[1]; int* nTriToFlip_d;
	int* triToFlip   ; int* triToFlip_d;

	int* subtract_nTriToFlip_d;
	Quad* quadList_d;

	int iter = 0;
	bool saveHistory = true; 
	bool info = true;
	bool saveCSV = true;

	struct cudaDeviceProp device;
	FILE* trifile;
	FILE* csvfile;
	FILE* insertedPerIterfile;
	FILE* flipedPerIterfile;
	FILE* errorfile; 

	int seed = -1;
	int distribution = -1;

	Delaunay(Point* points, int n);
	Delaunay(Point* points, int n, int numThreadsPerBlock);
	Delaunay(Point* points, int n, int numThreadsPerBlock, int seed_mark, int distribution_mark);
	void constructor(Point* points, int n);

	~Delaunay();

	//int ntpb = 128; // number of threads per block
	int ntpb = 128;
	//int ntpb = 1;
	void compute();
	
	void initSuperTri();
	void prepForInsert();
	void insert();

	int flipAfterInsert();
	void storeTriToFlip();
	void checkFlipAndLegality();
	void checkFlipConflicts();

	void printInfo();
	void printTri();

	int delaunayCheck();

	int flip();
	void quadFlip();
	void checkIncircleAll();

	void updatePointLocations();
	void saveToFile(bool end=false);

};

__host__ __device__ void writeTri(Tri* tri, int* p, int* n, int* o);

/* INIT */
__global__ void sumPoints(Point* pts, int* npts, Point* avgPoint);
void CalcAvgPoint(Point& avgPoint, Point* pts_d, int* npts);
__global__ void computeMaxDistPts(Point* pts, int* npts, REAL* largest_dist);

/* PREP FOR INSERT */
__global__ void resetInsertPtInTris(Tri* triList, int* nTriMax);
__global__ void setInsertPtsDistance(Point* pts, int* npts, Tri* triList, int* ptToTri, int* ptsUninserted, int* nptsUninserted);
__global__ void setInsertPts        (Point* pts, int* npts, Tri* triList, int* ptToTri, int* ptsUninserted, int* nptsUninserted);
__global__ void prepTriWithInsert(Tri* triList, int* nTri, int* triWithInsert, int* nTriWithInsert);

__global__ void resetFlipUsageInTris(Tri* triList, int* nTriMax);

/* INSERT */
__global__ void insertKernel(Tri* triList, int* nTri, int* nTriMax, int* triWithInsert, int* nTriWithInsert, int* ptToTr, Tri* triList_previ);
__device__ int insertInTri(int i, Tri* triList, int newTriIdx, int* ptToTri, Tri* triList_prev);
__device__ int insertPtInTri(int r, int i, Tri* triList, int newTriIdx, Tri* triList_prev);

__global__ void updateNbrsAfterIsertKernel(Tri* triList, int* triWithInsert, int* nTriWithInsert, int* nTri, Tri* triList_prev);
__device__ void updateNbrsAfterIsert(int i, Tri* triList, int newTriIdx, Tri* triList_prev);

__global__ void checkInsertPoint(Tri* triList, int* triWithInsert, int* nTriWithInsert);
__global__ void resetBiggestDistInTris(Tri* triList, int* nTriMax);

__global__ void updatePtsUninserted(int* npts, int* ptToTri, int* ptsUninserted, int* nptsUninserted);

/* FLIP */
__global__ void checkIncircleAllKernel(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri, Point* pts);

//__global__ void prepForConflicts(Tri* triList, int* nTri, int* nTriMax);
__global__ void prepForConflicts( int* triToFlip, int* nTriToFlip, Tri* triList, int* nTriMax);
__global__ void setConfigIdx(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri);
__global__ void storeNonConflictConfigs(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri, int* subtract_nTriToFlip);

__global__ void resetTriToFlipThisIter(int* triToFlip, int* nTriToFlip, Tri* triList);
//__global__ void resetTriToFlipThisIter(Tri* triList, int* nTri);
__global__ void markTriToFlipThisIter(int* triToFlip, int* nTriToFlip, Tri* triList);

__global__ void updateNbrsAfterFlipKernel(int* triToFlip, int* nTriToFlip, Tri* triList, Quad* quad);
__device__ void updateNbrsAfterFlip(int a, int e, int b, Tri* triList, Quad* quad);

__global__ void resetTriToFlip(Tri* triList, int* nTri);

__global__ void writeQuadKernel(int* triToFlip, int* nTriToFlip, Tri* triList, Quad* quadList);
__device__ void writeQuads(int a, int e, int b, Tri* triList, Quad* quad);
__device__ void writeQuad(Quad* quad, int* p, int* n, int* o);
__global__ void flipFromQuadKernel(int* triToFlip, int* nTriToFlip, Tri* triList, Quad* quadList);
__device__ void flipFromQuad(int a, int e, int b, Tri* triList, Quad* quad);

/* UPDATE POINTS */
__global__ void updatePointLocationsKernel(Point* pts, int* npts, Tri* triList, int* nTri, int* ptToTri, int* ptsUninserted, int* nptsUninserted);
__device__ int contains(int t, int r, Tri* triList, Point* pts);

/* Delaunay Check */
__global__ void delaunayCheckKernel(Tri* triList, int* nTri, Point* pts, int* nEdges);

/* Timing */
float timeGPU(auto func);

#endif
