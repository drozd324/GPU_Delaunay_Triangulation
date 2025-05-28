#include "delaunay.h"
#include "vector.h"
#include "ran.h"

int main(){

	Ran ran(123);

	int n = 10;
	vector<Point> points(n);
	for (int i=0; i<n; ++i) {
		points[i].x[0] = ran.doub();
		points[i].x[1] = ran.doub();
	}
	
	for (int i=0; i<n; ++i) {
		std::cout << "(" << points[i].x[0] << ", " << points[i].x[1] << ")\n";
	}
	std::cout << std::endl;
	
	try {
		Delaunay delaunay(points);
	} 
	catch (const char* msg) {
		std::cerr << msg;
	}
				
	//delaunay.store("triangulation.txt")
	return 0;	

}
