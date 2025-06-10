#include "delaunay.h"

/*
 * Constructor which creates the delaunay triagulation from a vector of points
 */
Delaunay::Delaunay(Point* points, int n) : npts(n), nTri(0), nTriMax(0), nActiveTri(0) {

	for (int i=0; i<npts; ++i) {
		nTriMax += pow(3, i); 
	}

	pts       = new Point[npts + 3];
	triList   = new Tri  [nTriMax];

	std::cout << "[ALLOCATING] \n";
	std::cout << "npts + 3     = " << npts + 3 << "\n";
	std::cout << "nTriMax      = " <<  nTriMax << "\n";

	for (int i=0; i<npts; i++) {
		pts[i] = points[i];
	}

	saveFile.open("./data/data.txt", std::ios_base::app);

	std::cout << "INITSUPERTRIANGLE\n";
	initSuperTri(points);

	// save points data to file
	saveFile << npts+3 << "\n";
	for (int i=0; i<npts+3; ++i) {
		saveFile << pts[i].x[0] << " " << pts[i].x[1] << "\n";
	}
	saveFile << "\n"; 

	saveToFile(saveFile);

	for (int i=0; i<npts; ++i) { 
		int inserted = insert();
		std::cout << "============ ITER " << i << "============ \n"; 
		std::cout << "nActiveTri: " << nActiveTri << "\n";
		std::cout << "inserted: " << inserted << "\n";
		
		if (inserted == 0) {
			break; 
		}
		saveToFile(saveFile);
	}

	delete[] pts;
	delete[] triList; 
}

int Delaunay::insert() {
//	int p[3]; // points of root triangle
//	int n[3]; // neighbours of root triangle as index in triList
//	int o[3]; // index in the Tri noted by the int n[i] of opposite point of current Tri
	int num_inserted_tri = 0;

	int max = nTri;
	for (int i=0; i<max; ++i) {
		std::cout << "i: " << i << "\n";
		if (triList[i].status == 1) {
			int center = triList[i].get_center();
			std::cout << "center = " << center << "\n";

			if (center == -1) { // if center doesnt exist, continue
				continue;
			}

			triList[i].status = -1;
			nActiveTri--;
			
			for (int j=0; j<3; ++j) {

				int p[3] = {center,
				            triList[i].p[j % 3],
				            triList[i].p[(j+1) % 3]};

				int n[3] = {triList[i].n[j],
					        nTri+1 + ((j+1) % 3),
					        nTri+1 + ((j+2) % 3)};

				int o[3] = {triList[i].o[j], 2, 1};

				// try to make some ascii art diagrams maybe good for explenation
				storeTriangle(nTri+1 + j, p, n, o);

				// updates neighbour points opposite point
				triList[n[0]].o[(triList[i].o[0] + 1) % 3] = 0;
			}

			nActiveTri += 3;
			nTri += 3;		
			num_inserted_tri += 3;
		}
	}

	return num_inserted_tri;
}

void Delaunay::initSuperTri(Point* points) {
	real x_low, y_low, x_high, y_high;
	x_low = x_high = points[0].x[0]; 
	y_low = y_high = points[0].x[1];
//
//	for (int i=0; i<npts-2; i++) {
//		for (int j=i+1; j<npts-1; j++) {
//			for (int k=j+1; k<npts; k++) {
//				std::cout << "(i,j,k) = (" << i << "," << j << "," << k << ")\n";
//				//std::cout << "(i,j,k) = (" << points[i] << "," << points[j] << "," << points[k] << ")\n";
//
//				Circle cc = circumcircle(points[i], points[j], points[k]);
//			
//				if (cc.center.x[0] - cc.radius < x_low)
//					x_low  = cc.center.x[0] - cc.radius; 
//				if (cc.center.x[0] + cc.radius > x_high)
//					x_high = cc.center.x[0] + cc.radius; 
//				if (cc.center.x[1] - cc.radius < y_low)
//					y_low  = cc.center.x[1] - cc.radius; 
//				if (cc.center.x[1] + cc.radius > y_high)
//					y_high = cc.center.x[1] + cc.radius; 
//			}
//		}
//	}
//
	x_low = 0;
	x_high = 1;
	y_low = 0;
	y_high = 1;

	real center_x = (x_high + x_low) / 2;
	real center_y = (y_high + y_low) / 2;
	real radius = sqrt( SQR(center_x - x_high) + SQR(center_y - y_high) );
			
	pts[npts    ] = Point(center_x + radius*sqrt(3), center_y - radius  );
	pts[npts + 1] = Point(center_x                 , center_y + 2*radius);
	pts[npts + 2] = Point(center_x - radius*sqrt(3), center_y - radius  );

	int p[3] = {npts, npts+1, npts+2};
	int n[3] = {-1, -1, -1}; 
	int o[3] = {-1, -1, -1}; 
	storeTriangle(nTri, p, n, o);
	nTri++;
	nActiveTri++;
}

void Delaunay::storeTriangle(int index, int triPts[3], int triNeighbours[3], int triOpposite[3]) {
	triList[index].pts = pts;
	triList[index].npts = npts;

	for (int i=0; i<3; ++i) {
		triList[index].p[i] = triPts[i];
		triList[index].n[i] = triNeighbours[i];
		triList[index].o[i] = triOpposite[i];
	}

	triList[index].status = 1;
}


void Delaunay::saveToFile(std::ofstream& file) {

	file << iter << " " << nActiveTri << "\n";
	for (int i=0; i<nTriMax; ++i) {
		if (triList[i].status == 1) {
			std::cout << "FOUND ACTIVE TRIANGLE TO SAVE\n";
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
