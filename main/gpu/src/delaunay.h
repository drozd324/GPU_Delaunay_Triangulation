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

	int* ptToTri; int* ptToTri_d;
	//int** sptsList; int** sptsList_d;

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


	// point insertion functions
//	__device__ int checkInsert();
//	__device__ int insert(int i);
//	__device__ int insert();
//	__device__ int insertInTri(int i);
//	__device__ int insertPtInTri(int r, int i);
//
//	// flipping functions
//	__device__ int flip(int a, int edge);
//	__device__ int flip_after_insert();

	void saveToFile();
	void cpyToHost();

};


__host__ __device__ void writeTri(Tri* tri, int* p, int* n, int* o);
__global__ void writeTriKernel(Tri* tri, int* p, int* n, int* o);

__global__ void sumPoints(Point* pts_d, int npts, Point* avgPoint_d);
__global__ void computeMaxDistPts(Point* pts_d, int npts, float* largest_dist);

__global__ void setInsertPtsDistance(Point* pts, int npts, Tri* triList);
__global__ void setInsertPts(Point* pts, int npts, Tri* triList);

__device__ float atomicAddFloat(float* address, float val);
__device__ float atomicMaxFloat(float* address, float val);
__device__ float atomicMinFloat(float* address, float val);

//__global__ checkInsert_gpu() {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//	num_tris_to_insert_d = 0;
//	if (idx < nTri_d) {
//		if (triList_d[idx].spts_alloc == true) { // triList[i].nspts > 0 && 
//			triList_d[idx].get_center();
//			atomicAdd(&num_tris_to_insert_d, 1);
//		}
//	}
//}
//
//__global__ insert_gpu() {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//	//int num_inserted_tri = 0;
//	//int max = nTri;
//	if (idx<max) {
//		insertInTri(idx);
//	}
//}


#endif
