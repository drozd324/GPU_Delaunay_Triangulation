#ifndef MGPUDELAUNAY_H
#define MGPUDELAUNAY_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctime>
//
//#include <iostream>
//#include <fstream>
//#include <random>	
//
//#include <thrust/device_ptr.h>
//#include <thrust/reduce.h>
//
//#include "macros.h"
//#include "types.h"
//#include "mymath.h"
//#include "point.h" 
//#include "circle.h"
//#include "tri.h"
//#include "atomic.h"
//#include "misc.h"
#include "delaunay.h"

/*
 * Struct for creating a delaunay triangulation from a given vector of points. Consists of 
 */
struct mGPUDelaunay: public delaunay {

	cudaDeviceProp* devices;

	Delaunay(Point* points, int n, int numThreadsPerBlock, int seed_mark, int distribution_mark);

	Tri* triangles_;
};


#endif
