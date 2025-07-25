#include <unistd.h>
#include "delaunay.h"
#include "ran.h"

#include <random>	

int main(int argc, char *argv[]) {

	int n = 5;
	int seed = 69420;
	int distribution = 1; 
	// 0: uniform square, 1: uniform disk, 2: gaussian u=0 var=1, 3: 

	int option;
    while ((option = getopt(argc, argv, "n:s:d:t:")) != -1) {
        switch (option) {
            case 'n': // set num cols m of matrix
	            n = atoi(optarg);
				break;
            case 's': // set num cols m of matrix
	            seed = atoi(optarg);
				break;
            case 'd': // distribution of points
	            distribution = atoi(optarg);
				break;
        }
    }

	Point* points = (Point*) malloc(n * sizeof(Point));

	if (distribution == 0) {
		Ran ran(seed);
		for (int i=0; i<n; ++i) {
			points[i].x[0] = ran.doub();
			points[i].x[1] = ran.doub();
		}

	} else if (distribution == 1) {
		Ran ran(seed);
		for (int i=0; i<n; ++i) {
			double x, y;
			ran.circle(x, y);

			points[i].x[0] = x;
			points[i].x[1] = y;
		}
	} else if (distribution == 2) {
		std::random_device rd{};
		std::mt19937 gen{rd()};
		std::normal_distribution d{0.0, 1.0};
		auto rand_normal = [&d, &gen]{ return d(gen); };

		for (int i=0; i<n; ++i) {
			points[i].x[0] = rand_normal();
			points[i].x[1] = rand_normal();
		}
	}
	
	//Delaunay delaunay(points, n, ntpb);
	Delaunay delaunay(points, n, seed, distribution);

	free(points);

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
