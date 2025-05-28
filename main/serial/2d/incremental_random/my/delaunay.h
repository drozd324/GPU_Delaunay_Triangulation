#ifndef DELAUNAY_H
#define DELAUNAY_H

#include <iostream>
#include <cmath>
#include "vector.h"
#include "ran.h"
#include "hash.h"
#include "point.h" 
#include "circle.h"
#include "macros.h"
#endif

struct Tri {
	vector<Point> pts;
	int p[3]; // index of points
	int d[3]; // index of Tri structs contained inside 
	int stat; // indicated whether this stuct is being used if nonzero

	void set(int a, int b, int c,  


