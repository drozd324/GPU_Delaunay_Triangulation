#include <unistd.h>
#include "delaunay.h"
#include "ran.h"

int main(int argc, char *argv[]) {

	int n = 10;
	int seed = 69420;
	
	int option;
    while ((option = getopt(argc, argv, "n:s:")) != -1) {
        switch (option) {
            case 'n': // set num cols m of matrix
	            n = atoi(optarg);
				break;
            case 's': // set num cols m of matrix
	            seed = atoi(optarg);
				break;
        }
    }

	Ran ran(seed);
	Point* points = new Point[n];
	for (int i=0; i<n; ++i) {
		float x, y;
		ran.circle(x, y);

		points[i].x[0] = x;
		points[i].x[1] = y;

//		points[i].x[0] = ran.doub();
//		points[i].x[1] = ran.doub();
	}
	
	Delaunay delaunay(points, n);

	delete[] points;

//	int n = 4;
//	Point* square = new Point[n*n];
//	for (int i=0; i<n; ++i) {
//		for (int j=0; j<n; ++j) {
//			square[i*n + j].x[0] = (float)i/(float)n;
//			square[i*n + j].x[1] = (float)j/(float)n;
//		}
//	}
//
//	for (int i=0; i<square.size(); ++i) {
//		std::cout << square[i].x[0] << " ";
//	}
//	std::cout << "\n"; 
//	
//	Delaunay delaunay(square, n*n);

	return 0;	
}
