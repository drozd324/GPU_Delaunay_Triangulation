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

//#include "atomic.h"

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

	float largest_dist[1]; float* largest_dist_d;

	int iter = 0; int* iter_d;
	int tag_num = 0;

	int num_tris_to_insert; int* num_tris_to_insert_d;

	std::ofstream saveFile;

	Delaunay(Point* points, int n);

	// compute options
	//void gpu_compute();
	
	void initSuperTri();
	void prepForInsert();
	void insert();

	void saveToFile();
	void cpyToHost();

};


__host__ __device__ void writeTri(Tri* tri, int* p, int* n, int* o);

/* INIT */
__global__ void sumPoints(Point* pts, int npts, Point* avgPoint);
__global__ void computeMaxDistPts(Point* pts, int npts, float* largest_dist);

/* PREP FOR INSERT */
__global__ void setInsertPtsDistance(Point* pts, int npts, Tri* triList, int* ptToTri);
__global__ void setInsertPts        (Point* pts, int npts, Tri* triList, int* ptToTri);

/* INSERT */
__global__ void insertKernel(Tri* triList, int nTri, int* triWithInsert, int nTriWithInsert);
__device__ int insertInTri(int i, Tri* triList, int newTriIdx);
__device__ int insertPtInTri(int r, int i, Tri* triList, int newTriIdx);


/* UPDATE */
//__global__ void updatePointsTriangles();

/* PRINTING */
__global__ void printTri(Tri* triList, int nTriMax);

/* MISC */
__global__ void arrayAddVal(int* array, int val, int n);

/* ATOMIC FUNCTIONS */
__device__ float atomicAddFloat(float* address, float val);
__device__ float atomicMaxFloat(float* address, float val);
__device__ float atomicMinFloat(float* address, float val);


#endif
