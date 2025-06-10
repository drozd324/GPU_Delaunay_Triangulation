#ifndef DELAUNAY_H
#define DELAUNAY_H

#include <iostream>
#include <fstream>
#include <format>

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
	
	int insert();
	void storeTriangle(int index, int tag, int triPts[3], int triNeighbours[3], int triOpposite[3]);

	void initSuperTri();
	void saveToFile(std::ofstream& file);

	int iter = 0;
};

#endif
