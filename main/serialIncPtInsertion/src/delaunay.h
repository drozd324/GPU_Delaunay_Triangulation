#ifndef DELAUNAY_H
#define DELAUNAY_H

#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctime>

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
	int npts;
	Point* pts;

	int nTri; 
	int nTriMax; 
	Tri* triList; 

	//Node nodes[10000]; // big num figure out later

	//std::ofstream saveFile;
	FILE* trifile;
	FILE* csvfile;

	int seed = -1;
	int distribution = -1;

	Delaunay(Point* points, int n);
	Delaunay(Point* points, int n, int seed_mark, int distribution_mark);
	void constructor(Point* points, int n);
	~Delaunay();

	// compute options
	void incPtIns();
	void onlyPointInsert();
	void notparallel();
	void gpu_compute();
	
	// point insertion functions
	int checkInsert();
	int insert(int i);
	int insert();
	int insertInTri(int i);
	int insertPt(int r);
	int insertPtInTri(int r, int i);

	// flipping functions
	int flip(int a, int edge);
	int flip_after_insert();
	int legalize(int a, int e);
	int legalize();

	void writeTri(int index, int triPts[3], int triNeighbours[3], int triOpposite[3]);

	void initSuperTri();
	void saveToFile(bool end=false);
	//void saveToFile();

	int iter = 0;
	int tag_num = 0;
};

#endif
