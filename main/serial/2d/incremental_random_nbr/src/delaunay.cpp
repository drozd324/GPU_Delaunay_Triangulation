#include "delaunay.h"

/*
 * Constructor which creates the delaunay triagulation from a vector of points
 */
Delaunay::Delaunay(Point* points, int n) :
	npts(n), pts(new Point[npts + 3]),
	nTri(0), nTriMax(2*npts - 2 - 3), triList(new Tri[nTriMax]),
	saveFile("./data/data.txt", std::ios_base::app)
{
	std::cout << "[ALLOCATING] \n";
	std::cout << "npts + 3     = " << npts + 3 << "\n";
	std::cout << "nTriMax      = " <<  nTriMax << "\n";

	for (int i=0; i<npts; i++) {
		pts[i] = points[i];
	}

	std::cout << "INITSUPERTRIANGLE\n";
	initSuperTri();

	// save points data to file
	saveFile << npts+3 << "\n";
	for (int i=0; i<npts+3; ++i) {
		saveFile << pts[i].x[0] << " " << pts[i].x[1] << "\n";
	}
	saveFile << "\n"; 

	saveToFile(saveFile);

	for (int i=0; i<npts; ++i) { 
		std::cout << "============ ITER " << i << "============ \n"; 
		int inserted = insert();
		std::cout << "inserted: " << inserted << "\n";
		std::cout << "nTri " << nTri << "/" << nTriMax << "\n";
		for (int k=0; k<nTri; ++k) {
			std::cout << k;
			triList[k].print();
		}
		
		if (inserted == 0) {
			break; 
		}
		saveToFile(saveFile);
	}

}

Delaunay::~Delaunay() {
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
		int center = triList[i].get_center();
		std::cout << "center=" << center << "\n";
		std::cout << i;
		triList[i].print();

		if (center == -1) { // if center doesnt exist, continue
			continue;
		}
		
		for (int j=2; j>=0; --j) {
			std::cout << "j=" << j << "\n";
			int p[3] = {center,
						triList[i].p[j % 3],
						triList[i].p[(j+1) % 3]};

			int n[3] = {triList[i].n[j],
						nTri+1 + ((j+1) % 3),
						nTri+1 + ((j+2) % 3)};

			int o[3] = {triList[i].o[j], 2, 1};		

			std::cout << "i=" << i << " |j=" << j << " |p: {" << p[0] << "," << p[1] << "," << p[2] << "}\n";
			std::cout << "i=" << i << " |j=" << j << " |n: {" << n[0] << "," << n[1] << "," << n[2] << "}\n";
			std::cout << "i=" << i << " |j=" << j << " |o: {" << o[0] << "," << o[1] << "," << o[2] << "}\n";

			int index = j==0 ? i : nTri+j-1;
			std::cout << "index=" << index << "\n";
			storeTriangle(index, nTri+j, p, n, o);

			// updates neighbour points opposite point
			triList[n[0]].o[(triList[i].o[0] + 1) % 3] = 0;
			// try to make some ascii art diagrams maybe good for explenation
		}

		nTri += 2;		
		num_inserted_tri += 2;
	}

	return num_inserted_tri;
}

void Delaunay::initSuperTri() {
//	real x_low, y_low, x_high, y_high;
//	x_low = x_high = pts[0].x[0]; 
//	y_low = y_high = pts[0].x[1];

//
//	for (int i=0; i<npts-2; i++) {
//		for (int j=i+1; j<npts-1; j++) {
//			for (int k=j+1; k<npts; k++) {
//				std::cout << "(i,j,k) = (" << i << "," << j << "," << k << ")\n";c
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
	Point avg; 
	for (int i=0; i<npts; ++i) {
		for (int k=0; k<2; ++k) {
			avg.x[k] += pts[i].x[k];
		}
	}
	for (int k=0; k<2; ++k) {
		avg.x[k] = avg.x[k]/npts;
	}

	real largest_dist = 0;
	real sample_dist;
	for (int i=0; i<npts; ++i) {
		for (int j=0; j<npts; ++j) {
			sample_dist = dist(pts[i], pts[j]);
			if (largest_dist < sample_dist) {
				largest_dist = sample_dist;
			}
		}
	}

//	x_low = 0;
//	x_high = 1;
//	y_low = 0;
//	y_high = 1;
//
//	real center_x = (x_high + x_low) / 2;
//	real center_y = (y_high + y_low) / 2;
//	real radius = sqrt( SQR(center_x - x_high) + SQR(center_y - y_high) );
			
	real center_x = avg.x[0];
	real center_y = avg.x[1];
	real radius = 2*largest_dist;

	pts[npts    ] = Point(center_x + radius*sqrt(3), center_y - radius  );
	pts[npts + 1] = Point(center_x                 , center_y + 2*radius);
	pts[npts + 2] = Point(center_x - radius*sqrt(3), center_y - radius  );

	int p[3] = {npts, npts+1, npts+2};
	int n[3] = {-1, -1, -1}; 
	int o[3] = {-1, -1, -1}; 
	storeTriangle(nTri, nTri, p, n, o);
	nTri++;
}

void Delaunay::storeTriangle(int index, int tag, int triPts[3], int triNeighbours[3], int triOpposite[3]) {
	triList[index].pts = pts;
	triList[index].npts = npts;

	for (int i=0; i<3; ++i) {
		triList[index].p[i] = triPts[i];
		triList[index].n[i] = triNeighbours[i];
		triList[index].o[i] = triOpposite[i];
	}

	triList[index].status = 1;
	triList[index].tag = tag;
}


void Delaunay::saveToFile(std::ofstream& file) {
//	file << iter << " " << nTri << "\n";
//	for (int i=0; i<nTriMax; ++i) {
//		if (triList[i].status == 1) {
//			// return triangles
//			for (int j=0; j<3; ++j) {
//				file << triList[i].p[j] << " "; 
//			} 
//			file << "\n"; 
//		}
//	}

	file << iter << " " << nTri << "\n";
	for (int i=0; i<nTri; ++i) {
		for (int j=0; j<3; ++j) {
			file << triList[i].p[j] << " "; 
		} 
		file << "\n"; 
	}

	file << "\n"; 
	iter++;
}
