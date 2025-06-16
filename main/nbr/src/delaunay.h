#ifndef DELAUNAY_H
#define DELAUNAY_H

#include <iostream>
#include <fstream>

#include "macros.h"
#include "types.h"
#include "math.h"
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

	std::ofstream saveFile;

	Delaunay(Point* points, int n);
	~Delaunay();
	
	int insert(int i);
	int insert();

	int flip(int a, int edge);
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
