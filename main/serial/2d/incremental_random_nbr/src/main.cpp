#include "delaunay.h"
#include "ran.h"

int main(){

	Ran ran(123);
	int n = 10;
	Point* points = new Point[n];
	for (int i=0; i<n; ++i) {
		points[i].x[0] = ran.doub();
		points[i].x[1] = ran.doub();
	}
	
	try {
		Delaunay delaunay(points, n);
	} 
	catch (const char* msg) {
		std::cerr << "CAUGHT: " << msg << "\n";
	}

	delete[] points;

//	int side = 4;
//	Point* square = new Point[side*side];
//	for (int i=0; i<side; ++i) {
//		for (int j=0; j<side; ++j) {
//			square[i*side + j].x[0] = (double)i/(double)side;
//			square[i*side + j].x[1] = (double)j/(double)side;
//		}
//	}
//
//	for (int i=0; i<square.size(); ++i) {
//		std::cout << square[i].x[0] << " ";
//	}
//	std::cout << "\n"; 
//	
//	try {
//	  Delaunay delaunay(square, side*side);
//	} 
//	catch (const char* msg) {
//		std::cerr << "CAUGHT: " << msg << "\n";
//	}

	return 0;	
}
