#ifndef DELAUNAY_H
#define DELAUNAY_H

#include <iostream>
#include <fstream>
#include <format>
#include <cmath>
#include "vector.h"
#include "ran.h"
#include "hash.h"
#include "point.h" 
#include "circle.h"
#include "macros.h"
#include "types.h"

int counter = 0;

/*
 * Data structure needed for Point instertion algorithm. Its main features are
 * that it holds a pointer to an array of points which will be used for the triangulation,
 * the index of those points as ints which form this triangle, its daughter triangles 
 * which are represented as ints which belong to an array of all triangle elements and
 * whether this triangle is used in the trianglulation constructed so far.
 */
struct Triel {
	Point *pts;
	int p[3]; // indexes of points
	int d[3]; // indexes of daughter points
	int stat; // status

	/*
	 * Inititalisation function for this struct. Sets the triangles
	 * 3 points (a, b, c), marks it as being used with stat=1 and
	 * marks its daughters and inactive with -1.
	 */
	void setme(int a, int b, int c, Point *ptss) {
		pts = ptss;
		p[0] = a;
		p[1] = b;
		p[2] = c;

		d[0] = d[1] = d[2] = -1;
		stat = 1;
	}

	/* 
	 * Checks whether the point "point" is contained in this triangle. Computed by making 
	 * use of the property of barycentric coordinates of a triangle. In this algorithm
	 * this is reducted to checking any of the constants determining this coordinate 
	 * transform is negative, if so, then return -1 which means this point is not
	 * contained inside this triangle. Otherwise the point is on the boundary 
	 * and returns 0 or outside and returns 1. 
	 */
	int contains(Point point) {
		real d;
		int i, j, ztest=0;

		for (i=0; i<3; ++i) {
			j = (i+1) % 3;
			// d = area of triangle (21.3.2) (21.3.10) 
			d = (pts[p[j]].x[0] - pts[p[i]].x[0])*(point.x[1] - pts[p[i]].x[1]) - 
				(pts[p[j]].x[1] - pts[p[i]].x[1])*(point.x[0] - pts[p[i]].x[0]);

			if (d < 0.0) {
				return -1;
			}
			if (d == 0.0) {
				ztest = 1;
			}
		}
   
		return (ztest? 0:1);
	}

	/*
	 * Print function for debugging.
	 */
	void print() {
		std::cout << "| stat     : " << stat
		          << "| points   : " << p[0] << ", " << p[1] << ", " << p[2]
		          << "| daughters: " << d[0] << ", " << d[1] << ", " << d[2]
		          << "\n";
	}
};

/*
 * Struct for creating a delaunay triangulation from a given vector of points. Consists of 
 */
struct Delaunay {
	int npts, ntri, ntree, ntreemax;
	vector<Point> pts;
	vector<Triel> triList;

	Hash<unsigned long long, int, Nullhash> *linehash;
	Hash<unsigned long long, int, Nullhash> *trihash;
	int *perm;

	Delaunay(vector<Point> &pvec);
	Ranhash hashfn;
	
	void insertapoint(int r);
	int whichcontainspt(const Point &p);
	int storetriangle(int a, int b, int c);
	void erasetriangle(int a, int b, int c, int d0, int d1, int d2);

	void initSuperTri(vector<Point> &pvec);
	void savetofile(int iter);

	static unsigned int jran; // Random number counter.
	static const real fuzz;
	static const real bigscale;
};
unsigned int Delaunay::jran = 69420; // Random number counter.
const real Delaunay::fuzz = 1.0e-6;
const real Delaunay::bigscale = 2.0;


/*
 * Constructor which creates the delaunay triagulation from a vector of points
 */
Delaunay::Delaunay(vector<Point> &pvec) :
		npts(pvec.size()), ntri(0), ntree(0), ntreemax(10*npts+1000), 
		pts(npts+3), triList(ntreemax) 
{
	// Construct Delaunay triangulation from an array of points pvec. If bit 0 in options is nonzero, 
	// hash memories used in the construction are deleted. (Some applications may want to use them
	// and will set options to 1.)
 
	linehash = new Hash<unsigned long long, int, Nullhash>(6*npts+12, 6*npts+12);
	trihash  = new Hash<unsigned long long, int, Nullhash>(2*npts+6, 2*npts+6);
	perm     = new int[npts]; //Permutation for randomizing point order.

	for (int j=0; j<npts; j++) {
		pts[j] = pvec[j];
		perm[j] = j;
	}
//	shuffle(perm, npts);
	initSuperTri(pvec);


	for (int j=0; j<npts; j++) { 
		std::cout << "========= ITER: " << j << "\n"; 
		insertapoint(perm[j]); 
	}

	for (int j=0; j<ntree; j++) {	// Delete the huge root triangle and all of its con-necting edges.
		if (triList[j].stat > 0) {
			if (triList[j].p[0] >= npts || triList[j].p[1] >= npts || triList[j].p[2] >= npts) {
				triList[j].stat = -1;
				ntri--;
			}
		}
	}

	savetofile(counter++);

	delete[] perm;
	delete   trihash;
	delete   linehash;
}

void Delaunay::initSuperTri(vector<Point> &pvec) {
	real x_low, y_low, x_high, y_high;
	x_low = x_high = pvec[0].x[0]; 
	y_low = y_high = pvec[0].x[1];

	for (int i=0; i<npts-2; i++) {
		for (int j=i+1; j<npts-1; j++) {
			for (int k=j+1; k<npts; k++) {
				Circle cc = circumcircle(pvec[i], pvec[j], pvec[k]);
				std::cout << i << " " << j << " " << k << "\n";
			
				if (cc.center.x[0] - cc.radius < x_low)
					x_low  = cc.center.x[0] - cc.radius; 
				if (cc.center.x[0] + cc.radius > x_high)
					x_high = cc.center.x[0] + cc.radius; 
				if (cc.center.x[1] - cc.radius < y_low)
					y_low  = cc.center.x[1] - cc.radius; 
				if (cc.center.x[1] + cc.radius > y_high)
					y_high = cc.center.x[1] + cc.radius; 
			}
		}
	}



	real center_x = (x_high + x_low) / 2;
	real center_y = (y_high + y_low) / 2;
	real radius = sqrt( SQR(center_x - x_high) + SQR(center_y - y_high) );
			
	std::cout << x_low << " " << x_high << " " << y_low << " " << y_high << "\n";
	std::cout << center_x << " " << center_y << "\n";
	std::cout << radius << "\n";

	pts[npts]   = Point(center_x, center_y + 2*radius);
	pts[npts+1] = Point(center_x - radius*sqrt(3), center_y - radius);
	pts[npts+2] = Point(center_x + radius*sqrt(3), center_y - radius);

	storetriangle(npts, npts+1, npts+2);

//	real xl, xh, yl, yh;
//	// Copy points to local store and calculate their bounding box.
//
//	for (int j=0; j<npts; j++) {
//
//		if (pvec[j].x[0] < xl) xl = pvec[j].x[0];
//		if (pvec[j].x[0] > xh) xh = pvec[j].x[0];
//		if (pvec[j].x[1] < yl) yl = pvec[j].x[1];
//		if (pvec[j].x[1] > yh) yh = pvec[j].x[1];
//	}
//	
//	// Store bounding box dimensions,  then construct
//	// the three fictitious points and store them.
//	real delx = xh - xl; 
//	real dely = yh - yl;	
//
//	pts[npts]   = Point(0.5*(xl + xh), yh + bigscale*dely);
//	pts[npts+1] = Point(xl - 0.5*bigscale*delx, yl - 0.5*bigscale*dely);
//	pts[npts+2] = Point(xh + 0.5*bigscale*delx, yl - 0.5*bigscale*dely);
//
//	storetriangle(npts, npts+1, npts+2);
}


void Delaunay::insertapoint(int r) {
	//Add the point with index r incrementally to the Delaunay triangulation.

	int i, j, k, l, tno, d0, d1, d2;
	unsigned long long key;

	for (j=0; j<3; j++) { // Find triangle containing point. Fuzz if it lies on an edge.
		tno = whichcontainspt(pts[r]);
		
		if (tno >= 0) // The desired result: Point is OK. 
			break; 
//		pts[r].x[0] += fuzz * delx * (hashfn.doub(jran++)-0.5);
//		pts[r].x[1] += fuzz * dely * (hashfn.doub(jran++)-0.5);
	}

	// store index of points of mother Triel
	i = triList[tno].p[0];
	j = triList[tno].p[1];
	k = triList[tno].p[2];

	int taski[50], taskj[50];
	int ntask = 0;
	// Create three triangles and queue them for legal edge tests.
	d0 = storetriangle(r, i, j);
	taski[++ntask] = i; taskj[ntask] = j;
	d1 = storetriangle(r, j, k);
	taski[++ntask] = j; taskj[ntask] = k;
	d2 = storetriangle(r, k, i);
	taski[++ntask] = k; taskj[ntask] = i;

	erasetriangle(i, j, k, d0, d1, d2); // Erase the old triangle and init the 3 new daughters

	while (ntask) { // Legalize edges
		savetofile(counter++);
		i = taski[ntask]; j = taskj[ntask--];
		key = hashfn.int64(j) - hashfn.int64(i); //  Look up fourth point.
		std::cout << "ntask: " << ntask << "| checking (" << r << ", " << i << ", " << j << ")\n";

		if ( ! linehash->get(key, l) ) {
			std::cout << "no triangle on other side\n";
			continue; // Case of no triangle on other side.
		} 

		if (incircle(pts[l], pts[j], pts[r], pts[i]) > 0.0) {
			std::cout << "INCIRCLE POSITIVE\n";

			// Create two new triangles
			d0 = storetriangle(r, l, j);
			d1 = storetriangle(r, i, l);
			// and erase old ones.
			erasetriangle(r, i, j, d0, d1, -1);
			erasetriangle(l, j, i, d0, d1, -1);

			// Erase line in both directions.
			key = hashfn.int64(i) - hashfn.int64(j); 
			linehash->erase(key);
			key = 0 - key; // Unsigned, hence binary minus.
			linehash->erase(key);
			
			// Two new edges now need checking:
			taski[++ntask] = l; taskj[ntask] = j;
			taski[++ntask] = i; taskj[ntask] = l;
		}
	}
}

int Delaunay::whichcontainspt(const Point &p) { 
	// Given point p,  return index in triList of the triangle in the triangulation that contains it,  or
	// return -1 for failure. 
	
	int i, j, k=0;

	// loop through dead triangles
	int count = 0;
	//std::cout << "while in whichcontainspt" << "\n";
	// loop is never entered wtf
	
	while (triList[k].stat <= 0) { 
		//std::cout << "count while in whichcontainspt: " << count++ << "\n";
		for (i=0; i<3; i++) { // Check up to three daughters.
			if ((j = triList[k].d[i]) < 0)
				continue; // Daughter doesn’t exist.
						  
			if (triList[j].contains(p) >= 0) 
				break;
		}

		if (i == 3) {
			return -1; // No daughters contain the point.
		}

		k = j; // Set new mother.
	}

	return k;
}

void Delaunay::erasetriangle(int a, int b, int c, int d0, int d1, int d2) {
	// Erase triangle abc in trihash and inactivate it in triList after setting its daughters.
	unsigned long long key;
	int j;
	
	key = hashfn.int64(a) ^ hashfn.int64(b) ^ hashfn.int64(c);
	//std::cout << "key to find : " << key << "\n";

	std::cout << "ERASING| " << a << "," << b << ", " << c << "\n";
	//std::cout << "      ERASING: " << key << "\n";
	if (trihash->get(key, j) == 0) {
		std::cout << "FAILED TO ERASE| " << a << "," << b << ", " << c << "\n";
		throw("nonexistent triangle");
	}

	trihash->erase(key);

	// set its daughters
	//std::cout << "\nIN ERASETRIANGLE\n";
	triList[j].d[0] = d0; triList[j].d[1] = d1; triList[j].d[2] = d2;
	triList[j].stat = 0; // kill/deacitvate this Tiel

	//std::cout << "mother pts (d0, d1, d2): " << d0 << ", " << d1 << ", " << d2 << "\n";
	//std::cout << "j: " << j << "\n";
	//triList[j].print();

	ntri--;
}



int Delaunay::storetriangle(int a, int b, int c) {
	// Store a triangle with vertices a, b, c in trihash. Store its points in linehash under keys to
	// opposite sides. Add it to triList,  returning its index there.

	unsigned long long key;
	triList[ntree].setme(a, b, c, &pts[0]);
	
	// save Triel location
	key = hashfn.int64(a) ^ hashfn.int64(b) ^ hashfn.int64(c);
	trihash->set(key, ntree);
	//std::cout << "STORING | key: " << key << "| ntree: " << ntree << "\n";
	std::cout << "STORING | " << a << "," << b << ", " << c << "\n";
	
	// save opposite points locations
	key = hashfn.int64(b) - hashfn.int64(c);
	linehash->set(key, a);
	key = hashfn.int64(c) - hashfn.int64(a);
	linehash->set(key, b);
	key = hashfn.int64(a) - hashfn.int64(b);
	linehash->set(key, c);
	
	if (++ntree == ntreemax) 
		throw("triList is sized too small");
	
	ntri++;
	return (ntree - 1);
}

void Delaunay::savetofile(int iter=0) {
	std::string tri_fname = std::format("./data/triangles_iterations/triangles_{}.txt", iter);
	std::string pts_fname = std::format("./data/points.txt", iter);

	std::ofstream triFile(tri_fname);
	std::ofstream ptsFile(pts_fname);

	for (int i=0; i<triList.size(); ++i) {
		if (triList[i].stat == 1) {
			// return triangles
			for (int j=0; j<3; ++j) {
				triFile << triList[i].p[j] << " "; 
			} 
			triFile << "\n"; 
		}
	}
	
	for (int i=0; i<pts.size(); ++i) {
		ptsFile << pts[i].x[0] << " " << pts[i].x[1] << "\n";
	}

	triFile.close();
	ptsFile.close();
}

#endif
