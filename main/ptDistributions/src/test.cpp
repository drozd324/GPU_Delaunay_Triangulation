#include <unistd.h>
#include <random>	
#include <stdio.h>

#include "delaunay.h"
#include "point.h"
#include "ran.h"

int main(int argc, char *argv[]) {
  
	int n = 5;
	int seed = 69420;
	int distribution = 1; // 0: uniform square, 1: uniform disk, 2: sphere disk  3: gaussian
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


	Ran ran(seed);
	switch (distribution) {
		       case 0:
			for (int i=0; i<n; ++i) {
				points[i].x[0] = ran.doub();
				points[i].x[1] = ran.doub();
			}

		break; case 1:
			for (int i=0; i<n; ++i) {
				float x, y;
				ran.disk(x, y);

				points[i].x[0] = x;
				points[i].x[1] = y;
			}

		break; case 2:
			for (int i=0; i<n; ++i) {
				float x, y;
				ran.proj_sphere(x, y);

				points[i].x[0] = x;
				points[i].x[1] = y;
			}

		break; case 3:
			for (int i=0; i<n; ++i) {
				float x, y;
				ran.gaussain(x, y);

				points[i].x[0] = x;
				points[i].x[1] = y;
			}

		break;
	}

	FILE* points_file = fopen("./points.txt", "w");
	fclose(points_file);
	FILE* points_file = fopen("./points.txt", "a");
	fprintf(points_file, "%d\n", n); 
	for (int i=0; i<n; ++i) {
		fprintf(points_file, "%f %f\n", points[i].x[0], points[i].x[0]);
	}

	free(points);
	return 0;
}
