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
				
	return 0;	
}
