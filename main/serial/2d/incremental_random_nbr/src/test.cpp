#include <unistd.h>
#include "delaunay.h"
#include "ran.h"

int main(int argc, char *argv[]) {

	int option;
	int n = 3;
    while ((option = getopt(argc, argv, "n:")) != -1) {
        switch (option) {
            case 'n': // set num cols m of matrix
	            n = atoi(optarg);
				break;
        }
    }

	Ran ran(123);
	Point* points = new Point[n];
	for (int i=0; i<n; ++i) {
		points[i].x[0] = ran.doub();
		points[i].x[1] = ran.doub();
	}
	
	Delaunay delaunay(points, n);

	delete[] points;

//	int n = 4;
//	Point* square = new Point[n*n];
//	for (int i=0; i<n; ++i) {
//		for (int j=0; j<n; ++j) {
//			square[i*n + j].x[0] = (double)i/(double)n;
//			square[i*n + j].x[1] = (double)j/(double)n;
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
