#ifndef DELAUNAY_H
#define DELAUNAY_H

#include <iostream>
#include <fstream>
#include <format>
#include <cmath>
#include "ran.h"
#include "point.h" 
#include "circle.h"
#include "macros.h"
#include "types.h"

/*
 * Data structure needed for Point instertion algorithm. Its main features are
 * that it holds a pointer to an array of points which will be used for the triangulation,
 * the index of those points as ints which form this triangle, its daughter triangles 
 * which are represented as ints which belong to an array of all triangle elements and
 * whether this triangle is used in the trianglulation constructed so far.
 */
struct Tri {
	Point* pts; int npts;
	int* ptsInside; int nptsInside=npts;

	int p[3]; // indexes of points
	int np[3]; // indexes of points corresponding to same point in neighbour triangles
	int o[3]; // indexes of pts of this triangle
	int n[3]; // idx to Tri neighbours of this triangle
	int d[3]; // indexes of daughter points
	int center = -1;
	int status = -1;

	void get_ptsInside() {
		int idx_in = 0;
		for (int k=0; k<npts; ++i) {
			for (int i=0; i<3; ++i) {
				int j = (i+1) % 3;
				real area = (pts[p[j]].x[0] - pts[p[i]].x[0])*(point.x[1] - pts[p[i]].x[1]) - 
				            (pts[p[j]].x[1] - pts[p[i]].x[1])*(point.x[0] - pts[p[i]].x[0]);

				if (area < 0) {
					break;
				}
			}
			
			ptsInside[idx_in++] = k;
		}

		get_Center();
	}

	int get_Center() {
		Circle cc = circumcircle(p.x[0], p.x[1], p.x[2]);
		Point true_center = cc.center;
		center = 0;

		//for (int i=1; i<nptsInside; ++i) {
		for (int i=1; i<npts; ++i) {
			//if (dist(ptsInside[i], true_center) < dist(center, true_center)) {
			if (dist(pts[i], true_center) < dist(pts[center], true_center)) {
				center = i;
			}
		}

		return center;
	}

	void print() {
		std::cout << "| stat      : " << stat
		          << "| points    : " << p[0] << ", " << p[1] << ", " << p[2]
		          << "| daughters : " << d[0] << ", " << d[1] << ", " << d[2]
		          << "| opposite  : " << o[0] << ", " << o[1] << ", " << o[2]
		          << "| neighbours: " << o[0] << ", " << o[1] << ", " << o[2]
		          << "\n";
	}


};

/*
 * Struct for creating a delaunay triangulation from a given vector of points. Consists of 
 */
struct Delaunay {
	Point* pts      ; int npts;
	Tri*   triList  ; int nTri; 
	int*   activeTri; int nActiveTri; 
	std::ofstream saveFile;

	Delaunay(Point* points);
	
	void insert();
	int storeTriangle(int a, int b, int c);
	void erasetriangle(int a, int b, int c, int d0, int d1, int d2);

	void initSuperTri(vector<Point> &points);
	void saveToFile(std::ofstream& file);

	int iter = 0;
};


/*
 * Constructor which creates the delaunay triagulation from a vector of points
 */
Delaunay::Delaunay(Point* points, int n) : npts(n), nTri(0), {

	int nTriMax = 0;
	for (int i=0; i<npts; ++i) {
		nTriMax += pow(3, i); 
	}

	pts       = new Point[npts + 3];
	triList   = new Tri  [nTriMax];
	activeTri = new int  [pow(3, npts)];

	for (int i=0; i<npts; i++) {
		pts[j] = points[j];
	}

	saveFile.open("./data/data.txt", std::ios_base::app);
	initSuperTri(points);

	// save points data to file
	saveFile << npts << "\n";
	for (int i=0; i<npts; ++i) {
		saveFile << pts[i].x[0] << " " << pts[i].x[1] << "\n";
	}
	saveFile << "\n"; 

	for (int i=0; i<npts; i++) { 
		std::cout << "============ ITER " << i << "============ \n"; 
		insert(); 
	}

	saveToFile(saveFile);

	pts       = delete;
	triList   = delete; 
	activeTri = delete; 
}



void Delaunay::insert() {
	/*
	 * Algo
	 * 
	 * FOR EACH ACTIVE TRIANGLE
	 *     insert 3 triangles
	 *     make active triangle inactive
	*/

	int max = nTri;
	int p[3]; // points of root triangle
	int n[3]; // neighbours of root triangle
	int o[3]; // index in the Tri noted by the int n[i] of opposite point of current Tri

	for (int i=0; i<max; ++i) {
		if (triList[i].status == 1) {
			triList[i].status = -1;

			int center = nTri[i].get_Center();
			for (int j=0; j<3; ++j) {

				p = {center,
					 nTri[i].p[j % 3],
					 nTri[i].p[(j+1) % 3]};

				n = {triList[i].n[j],
					 nTri+1 + ((j+1)%3),
					 nTri+1 + ((j+2)%3)};

				o = {triList[i].o[j], 2, 1};

				// try to make some ascii art diagrams maybe good for explenation
				storeTriangle(p, n, o);

				// updates neighbour points opposite point
				triList[n[0]].o[(triList[i].o[0] + 1) % 3] = 0;
			}

			nTri += 3;		
		}
	}
}

void Delaunay::initSuperTri(vector<Point> &points) {
	real x_low, y_low, x_high, y_high;
	x_low = x_high = points[0].x[0]; 
	y_low = y_high = points[0].x[1];

	for (int i=0; i<npts-2; i++) {
		for (int j=i+1; j<npts-1; j++) {
			for (int k=j+1; k<npts; k++) {
				Circle cc = circumcircle(points[i], points[j], points[k]);
			
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
			
	pts[npts    ] = Point(center_x + radius*sqrt(3), center_y - radius  );
	pts[npts + 1] = Point(center_x                 , center_y + 2*radius);
	pts[npts + 2] = Point(center_x - radius*sqrt(3), center_y - radius  );

	storeTriangle({npts, npts+1, npts+2});
	ntri++;
}

void Delaunay::storeTriangle(int* triPts, int* triNeighbours={-1, -1, -1}, int* triOpposite={-1, -1, -1}) {
	triList[nTri].pts = pts;
	triList[nTri].npts = npts;
	triList[nTri].p = triPts;

	triList[nTri].n = triNeighbours;
	triList[nTri].o = triOpposite;

	triList[nTri].status = 1;
}


void Delaunay::saveToFile(std::ofstream& file) {

	file << iter << " " << nTriMax << "\n";
	for (int i=0; i<nTri); ++i) {
		if (triList[i].status == 1) {
			// return triangles
			for (int j=0; j<3; ++j) {
				file << triList[i].p[j] << " "; 
			} 
			file << "\n"; 
		}
	}

	file << "\n"; 
	iter++;
}

#endif
