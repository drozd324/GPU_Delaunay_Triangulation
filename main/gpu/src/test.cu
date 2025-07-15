#include <unistd.h>
#include <random>	

#include "delaunay.h"
#include "point.h"
#include "ran.h"

int main(int argc, char *argv[]) {
  
	int n = 5;
	int seed = 69420;
	int distribution = 1; 
	// 0: uniform square, 1: uniform disk, 2: gaussian u=0 var=1, 3: 
	int ntpb = 128;

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
            case 't': // number of threads per block for gpu compute
	            ntpb = atoi(optarg);
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
			float x, y;
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
	Delaunay delaunay(points, n, ntpb, seed, distribution);

	free(points);
	return 0;
}
