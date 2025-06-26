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
	int    npts ; int* npts_d;
	Point* pts  ; Point* pts_d;

	int  nTri   ; int* nTri_d; 
	int  nTriMax; int* nTriMax_d; 
	Tri* triList; Tri* triList_d; 

	int iter = 0; int* iter_d;
	int tag_num = 0;

	int num_tris_to_insert, num_tris_to_insert_d;

	std::ofstream saveFile;

	Delaunay(Point* points, int n);
	~Delaunay();

	// compute options
	//void gpu_compute();
	
	void initSuperTri();

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

	void saveToFile(bool end=false);
	void cpyToHost();

};

__global__ void setSptsAll(int npts_d, Tri* triList_d, int i=0);
__global__ void computeAvgPoint(Point* pts_d, int npts, Point *avgPoint_d);
__global__ void computeMaxDistPts(Point* pts_d, int npts, float* largest_dist);

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
