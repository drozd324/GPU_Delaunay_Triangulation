#include "src/delaunay.h"
#include "src/vector.h"
#include "src/ran.h"

int main(){

	Ran ran(1234);
	int n = 10;
	vector<Point> points(n);
	for (int i=0; i<n; ++i) {
		points[i].x[0] = ran.doub();
		points[i].x[1] = ran.doub();
	}
	
	try {
		Delaunay delaunay(points);
	} 
	catch (const char* msg) {
		std::cerr << "CAUGHT: " << msg << "\n";
	}

					
//	int side = 4;
//	vector<Point> square(side*side);
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
//	  Delaunay delaunay(square);
//	} 
//	catch (const char* msg) {
//		std::cerr << "CAUGHT: " << msg << "\n";
//	}

	return 0;	
}
