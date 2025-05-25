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

struct Triel {
	Point *pts;
	int p[3]; // indexes of points
	int d[3]; // daughters of this triel 
	int stat;

	void setme(int a, int b, int c, Point *ptss) {
		pts = ptss;
		p[0] = a;
		p[1] = b;
		p[2] = c;

		d[0] = d[1] = d[2] = -1;
		stat = 1;
	}

	// Return 1 if point is in the triangle, 0 if on boundary, -1 if outside. 
	// (CCW triangle is assumed) equations (21.3.10) and the paragraph below.
	int contains(Point point) {
		double d;
		int i, j, ztest=0;

		for (i=0; i<3; ++i) {
			j = (i+1) % 3;
			d = (pts[p[j]].x[0] - pts[p[i]].x[0])*(point.x[1] - pts[p[i]].x[1]) - 
				(pts[p[j]].x[1] - pts[p[i]].x[1])*(point.x[0] - pts[p[i]].x[0]);

			if (d < 0.0)
				return -1;
			if (d == 0.0) 
				ztest = 1;
		}
   
		return (ztest? 0:1);
	}
};


struct Nullhash {
	Nullhash(int nn) {}
	inline unsigned long long fn(const void *key) const {
		return *((unsigned long long *)key);
	}
};

                                    
struct Delaunay {
	//Structure for constructing a Delaunay triangulation from a given set of points.

	int npts, ntri, ntree, ntreemax, opt;
	double delx, dely;
	vector<Point> pts;
	vector<Triel> thelist;

	Hash<unsigned long long, int, Nullhash> *linehash;
	Hash<unsigned long long, int, Nullhash> *trihash;
	int *perm;

	Delaunay(vector<Point> &pvec, int options=0);
	Ranhash hashfn;
	
	//double interpolate(const Point &p, const vector<double> &fnvals, double defaultval=0.0);

	void insertapoint(int r);
	int whichcontainspt(const Point &p, int strict=0);

	int storetriangle(int a, int b, int c);
	void erasetriangle(int a, int b, int c, int d0, int d1, int d2);

	//void save_to_file(char* filename);

	static unsigned int jran; // Random number counter.
	static const double fuzz;
	static const double bigscale;
};
unsigned int Delaunay::jran = 14921620; // Random number counter.
const double Delaunay::fuzz = 1.0e-6;
const double Delaunay::bigscale = 1000.0;

Delaunay::Delaunay(vector<Point> &pvec, int options) :
		npts(pvec.size()), ntri(0), ntree(0), ntreemax(10*npts+1000), 
		opt(options), pts(npts+3), thelist(ntreemax) 
{
	// Construct Delaunay triangulation from an array of points pvec. If bit 0 in options is nonzero, 
	// hash memories used in the construction are deleted. (Some applications may want to use them
	// and will set options to 1.)
 
	linehash = new Hash<unsigned long long, int, Nullhash>(6*npts+12, 6*npts+12);
	trihash  = new Hash<unsigned long long, int, Nullhash>(2*npts+6, 2*npts+6);
	perm     = new int[npts]; //Permutation for randomizing point order.
							  //
	// Copy points to local store and calculate their bounding box.
	double xl, yl, xh, yh;
	xl = xh = pvec[0].x[0]; 
	yl = yh = pvec[0].x[1];

	for (int j=0; j<npts; j++) {
		pts[j] = pvec[j];
		perm[j] = j;

		if (pvec[j].x[0] < xl)
			xl = pvec[j].x[0];
		if (pvec[j].x[0] > xh)
			xh = pvec[j].x[0];
		if (pvec[j].x[1] < yl)
			yl = pvec[j].x[1];
		if (pvec[j].x[1] > yh)
			yh = pvec[j].x[1];
	}
	
	// Store bounding box dimensions,  then construct
	// the three fictitious points and store them.
	delx = xh - xl; 
	dely = yh - yl;	

	pts[npts]   = Point(0.5*(xl + xh), yh + bigscale*dely);
	pts[npts+1] = Point(xl - 0.5*bigscale*delx, yl - 0.5*bigscale*dely);
	pts[npts+2] = Point(xh + 0.5*bigscale*delx, yl - 0.5*bigscale*dely);

	storetriangle(npts, npts+1, npts+2);

	// Create a random permutation:
	for (int j=npts; j>0; j--)
		SWAP(perm[j-1], perm[hashfn.int64(jran++) % j]);
	
	for (int j=0; j<npts; j++) 
		insertapoint(perm[j]); //All the action is here!

	for (int j=0; j<ntree; j++) {	// Delete the huge root triangle and all of its con-necting edges.
		if (thelist[j].stat > 0) {
			if (thelist[j].p[0] >= npts || thelist[j].p[1] >= npts || thelist[j].p[2] >= npts) {
				thelist[j].stat = -1;
				ntri--;
			}
		}
	}

	if (!(opt & 1)) { //Clean up,  unless option bit says not to.
		delete[] perm;
		delete   trihash;
		delete   linehash;
	}
}



void Delaunay::insertapoint(int r) {
	//Add the point with index r incrementally to the Delaunay triangulation.

	int i, j, k, l, s, tno, ntask, d0, d1, d2;
	unsigned long long key;
	int tasks[50], taski[50], taskj[50]; // Stacks (3 vertices) for legalizing edges.
	for (j=0; j<3; j++) { // Find triangle containing point. Fuzz if it lies on an edge.
		tno = whichcontainspt(pts[r], 1);
		if (tno >= 0) // The desired result: Point is OK. 
			break; 
							 
		pts[r].x[0] += fuzz * delx * (hashfn.doub(jran++)-0.5);
		pts[r].x[1] += fuzz * dely * (hashfn.doub(jran++)-0.5);
	}

	if (j == 3) 
		throw("points degenerate even after fuzzing");
	

	ntask = 0;
	
	i = thelist[tno].p[0]; j = thelist[tno].p[1]; k = thelist[tno].p[2];
	// The following line is relevant only when the indicated bit in opt is set. This feature is used
	// by the convex hull application and causes any points already known to be interior to the
	// convex hull to be omitted from the triangulation,  saving time (but giving in an incomplete
	// triangulation).
	if (opt & 2 && i < npts && j < npts && k < npts)
		return;
	
	// Create three triangles and queue them for legal edge tests.
	d0 = storetriangle(r, i, j); 
	tasks[++ntask] = r; taski[ntask] = i; taskj[ntask] = j;

	d1 = storetriangle(r, j, k);
	tasks[++ntask] = r; taski[ntask] = j; taskj[ntask] = k;

	d2 = storetriangle(r, k, i);
	tasks[++ntask] = r; taski[ntask] = k; taskj[ntask] = i;

	erasetriangle(i, j, k, d0, d1, d2); //Erase the old triangle.

	while (ntask) { // Legalize edges recursively.
		s = tasks[ntask]; i = taski[ntask]; j = taskj[ntask--];
		key = hashfn.int64(j) - hashfn.int64(i); //  Look up fourth point.

		if ( ! linehash->get(key, l) )
			continue; // Case of no triangle on other side.

		if (incircle(pts[l], pts[j], pts[s], pts[i]) > 0.0){ // Needs legalizing?
			d0 = storetriangle(s, l, j); // Create two new triangles
			d1 = storetriangle(s, i, l);

			erasetriangle(s, i, j, d0, d1, -1); // and erase old ones.
			erasetriangle(l, j, i, d0, d1, -1);

			key = hashfn.int64(i) - hashfn.int64(j); // Erase line in both directions.
			linehash->erase(key);

			key = 0 - key; // Unsigned,  hence binary minus.
			linehash->erase(key);
			
			// Two new edges now need checking:
			tasks[++ntask] = s; taski[ntask] = l; taskj[ntask] = j;
			tasks[++ntask] = s; taski[ntask] = i; taskj[ntask] = l;
		}
	}
}

int Delaunay::whichcontainspt(const Point &p,  int strict) { 
	// Given point p,  return index in thelist of the triangle in the triangulation that contains it,  or
	// return 1 for failure. If strict is nonzero,  require strict containment,  otherwise allow the point
	// to lie on an edge.
	
	int i, j, k=0;

	while (thelist[k].stat <= 0) { // Descend in tree until reach a “live” triangle.
		for (i=0; i<3; i++) { // Check up to three daughters.
			if ((j = thelist[k].d[i]) < 0)
				continue; // Daughter doesn’t exist.
						  
			if (strict) {
				if (thelist[j].contains(p) > 0) 
					break;
			} else { // Yes,  descend on this branch.
				if (thelist[j].contains(p) >= 0) 
					break;
			}
		}

		if (i == 3) 
			return -1; // No daughters contain the point.
		k = j; // Set new mother.
	}

	return k; // Normal return.
}


void Delaunay::erasetriangle(int a, int b, int c, int d0, int d1, int d2) {
	// Erase triangle abc in trihash and inactivate it in thelist after setting its daughters.
	unsigned long long key;
	int j;
	
	key = hashfn.int64(a) ^ hashfn.int64(b) ^ hashfn.int64(c);
	if (trihash->get(key, j) == 0)
		throw("nonexistent triangle");

	trihash->erase(key);

	thelist[j].d[0] = d0; thelist[j].d[1] = d1; thelist[j].d[2] = d2;
	thelist[j].stat = 0;

	ntri--;
}



int Delaunay::storetriangle(int a, int b, int c) {
	// Store a triangle with vertices a, b, c in trihash. Store its points in linehash under keys to
	// opposite sides. Add it to thelist,  returning its index there.

	unsigned long long key;
	thelist[ntree].setme(a, b, c, &pts[0]);

	key = hashfn.int64(a) ^ hashfn.int64(b) ^ hashfn.int64(c);
	trihash->set(key, ntree);
	
	key = hashfn.int64(b) - hashfn.int64(c);
	linehash->set(key, a);
	key = hashfn.int64(c) - hashfn.int64(a);
	linehash->set(key, b);
	key = hashfn.int64(a) - hashfn.int64(b);
	linehash->set(key, c);
	
	if (++ntree == ntreemax) 
		throw("thelist is sized too small");
	
	
	ntri++;
	return (ntree - 1);
}

//
//void Delaunay::store(char* filename) {
//	FILE* fp;
//
//	fptr = fopen("filename.txt", "w");
//	for (int i=0; i<thelist.size(); ++i) {
//		for (int i=0; i<thelist.size(); ++i) {
//			fprintf(fptr, "%lf ", );
//		}
//	}
//
//	fclose(fptr);
//}

#endif
