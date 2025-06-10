#include "delaunay.h"

/*
 * Constructor which creates the delaunay triagulation from a vector of points
 */
Delaunay::Delaunay(Point* points, int n) : npts(n), nTri(0) {

	int nTriMax = 0;
	for (int i=0; i<npts; ++i) {
		nTriMax += pow(3, i); 
	}

	pts       = new Point[npts + 3];
	triList   = new Tri  [nTriMax];
	activeTri = new int  [pow(3, npts)];

	for (int i=0; i<npts; i++) {
		pts[i] = points[i];
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

	delete pts;
	delete triList; 
	delete activeTri; 
}

void Delaunay::insert() {
	int max = nTri;
//	int p[3]; // points of root triangle
//	int n[3]; // neighbours of root triangle as index in triList
//	int o[3]; // index in the Tri noted by the int n[i] of opposite point of current Tri

	for (int i=0; i<max; ++i) {
		if (triList[i].status == 1) {
			triList[i].status = -1;

			int center = triList[i].get_Center();
			for (int j=0; j<3; ++j) {

				int p[3] = {center,
					 triList[i].p[j % 3],
					 triList[i].p[(j+1) % 3]};

				int n[3] = {triList[i].n[j],
					 nTri+1 + ((j+1)%3),
					 nTri+1 + ((j+2)%3)};

				int o[3] = {triList[i].o[j], 2, 1};

				storeTriangle(p, n, o);
				// try to make some ascii art diagrams maybe good for explenation

				// updates neighbour points opposite point
				triList[n[0]].o[(triList[i].o[0] + 1) % 3] = 0;
			}

			nTri += 3;		
		}
	}
}

void Delaunay::initSuperTri(Point* points) {
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

	int p[3] = {npts, npts+1, npts+2};
	int n[3] = {-1, -1, -1}; 
	int o[3] = {-1, -1, -1}; 
	storeTriangle(p, n, o);
	nTri++;
}

void Delaunay::storeTriangle(int triPts[3], int triNeighbours[3], int triOpposite[3]) {
	triList[nTri].pts = pts;
	triList[nTri].npts = npts;

	for (int i=0; i<3; ++i) {
		triList[nTri].p[i] = triPts[i];
		triList[nTri].n[i] = triNeighbours[i];
		triList[nTri].o[i] = triOpposite[i];
	}


	triList[nTri].status = 1;
}


void Delaunay::saveToFile(std::ofstream& file) {

	file << iter << " " << nTri << "\n";
	for (int i=0; i<nTri; ++i) {
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
