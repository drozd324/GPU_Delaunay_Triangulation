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
		double x, y;
		ran.circle(x, y);

		points[i].x[0] = x;
		points[i].x[1] = y;
	}
	
	Delaunay delaunay(points, n);
	delaunay

	delete[] points;
	return 0;	
}
