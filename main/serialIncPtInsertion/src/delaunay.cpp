#include "delaunay.h"


Delaunay::Delaunay(Point* points, int n) {
	constructor(points, n);	
}

Delaunay::Delaunay(Point* points, int n, int seed_mark, int distribution_mark) {
	seed = seed_mark;
	distribution = distribution_mark;

	constructor(points, n);	
}

/*
 * Constructor which creates the delaunay triagulation from an array of 'points'
 * and its lenght 'n'.
 */
//Delaunay::Delaunay(Point* points, int n) :
void Delaunay::constructor(Point* points, int n) {
	npts = n;
	pts = new Point[npts + 3];

	nTri = 0;
	nTriMax = 2*(npts+3) - 2 - 3;
	triList = new Tri[nTriMax];

	// Opening files to save data
	trifile = fopen("./data/tri.txt", "w");
	fclose(trifile);
	trifile = fopen("./data/tri.txt", "a");

	csvfile = fopen("./data/coredata.csv", "w");

	// assigning copyting points
	for (int i=0; i<npts; i++) {
		pts[i] = points[i];
	}

	// initialize super triangle
	initSuperTri();

    // save points data to trifile
	fprintf(trifile, "%d\n", npts + 3);
	for (int i=0; i<npts + 3; ++i) {
		fprintf(trifile, "%f %f\n", pts[i].x[0], pts[i].x[1]);
	}
	fprintf(trifile, "\n");

	saveToFile();
	// main compute which performs the incremental point insertion
	clock_t t0 = clock();

	incPtIns();

	clock_t t1 = clock() - t0;
	float totalRuntime = (float)t1 / CLOCKS_PER_SEC;


	// check if triangulation is delaunay
	int nflips = -1;
	while (nflips != 0) {
		if ((nflips = legalize()) == 0) {
			break;
		} else {
			std::cout << "Triangluation is NOT Delaunay\n";
		}
		std::cout << "Performed	" << nflips  << " additional flips\n"; 
	}

	std::cout << "Triangluation is Delaunay\n";

	// saves final triangulion with extra points removed
	saveToFile(true);

	fprintf(csvfile, "npts,nTriMax,totalRuntime,seed,distribution,\n");
	fprintf(csvfile, "%d,%d,%f,%d,%d,\n", npts, nTriMax, totalRuntime, seed, distribution);
}

/*
 * Basic destructor.
 */
Delaunay::~Delaunay() {
	delete[] triList; 
	delete[] pts;
	
	fclose(trifile);
	fclose(csvfile);
}

void Delaunay::incPtIns() {
	std::cout << "\n====[INCREMENTAL POINT INSERTION]====\n" ;
	for (int i=0; i<npts; ++i) {
		std::cout << "\rNumber of points inserted: " << i+1 << "/" << npts;
		insertPt(i);
		
		int nflips = -1;
		while (nflips != 0) {
			nflips = flip_after_insert();
		}
	}
	std::cout << "\n" ;
}

/*
 * Performs a flip operation on a triangle 'a' and one of its edges/neighbours 'e' denoted by
 * index 0, 1 or 2. Returns 1 if the flip was performed, reuturns -1 if no flip was 
 * performed.
 * 
 * @param a Index in triList of chosen triangle
 * @param e Index in chosen triangle of edge/neigbour. This is an int in 0, 1 or 2.
 */
int Delaunay::flip(int a, int e) {

	if (e == -1) {
		return 0;
	}

	// if neighbour doesnt exist, then exit
	int b = triList[a].n[e]; // index in triList of nei
	if (b == -1) {
		return 0;
	}

	int opp_idx = triList[a].o[e]; 

	// check if we should flip
	if (0 > incircle(pts[triList[b].p[opp_idx]],
				     pts[triList[a].p[0]],
				     pts[triList[a].p[1]],
				     pts[triList[a].p[2]])) 
	{
		return 0;
	}

 
 	// temporary qaud "struct" data  just to make it readable
	int p[4] = {triList[a].p[(e-1 + 3)%3], triList[a].p[e]                , triList[b].p[opp_idx], triList[a].p[(e + 1)%3]};
	int n[4] = {triList[a].n[(e-1 + 3)%3], triList[b].n[(opp_idx-1 + 3)%3], triList[b].n[opp_idx], triList[a].n[(e + 1)%3]}; 
	int o[4] = {triList[a].o[(e-1 + 3)%3], triList[b].o[(opp_idx-1 + 3)%3], triList[b].o[opp_idx], triList[a].o[(e + 1)%3]}; 

	int ap[3] = {p[0], p[1], p[2]};
	int an[3] = {n[0], n[1], b};
	int ao[3] = {o[0], o[1], 1};

	int bp[3] = {p[2], p[3], p[0]};
	int bn[3] = {n[2], n[3], a};
	int bo[3] = {o[2], o[3], 1};

	triList[a].writeTri(pts, npts, ap, an, ao);
	triList[b].writeTri(pts, npts, bp, bn, bo);

	if (n[0] >= 0) {
		triList[n[0]].n[(o[0]+1)%3] = a;	
		triList[n[0]].o[(o[0]+1)%3] = 2;	
	}

	if (n[1] >= 0) {
		triList[n[1]].n[(o[1]+1)%3] = a;	
		triList[n[1]].o[(o[1]+1)%3] = 0;	
	}

	if (n[2] >= 0) {
		triList[n[2]].n[(o[2]+1)%3] = b;	
		triList[n[2]].o[(o[2]+1)%3] = 2;	
	}

	if (n[3] >= 0) {
		triList[n[3]].n[(o[3]+1)%3] = b;	
		triList[n[3]].o[(o[3]+1)%3] = 0;	
	}

	saveToFile();
	return 1;
}

/*
 * Function to legalize a given triangle in triList with index 'a', with edge 'e'.
 */
int Delaunay::legalize(int a, int e) {
	if (flip(a, e) == 0) {
		return 0;
	}

	int nflips = 1;
	nflips += legalize(a, 1);
	nflips += legalize(triList[a].n[2], 0);

	return nflips;
}

/*
 * Legalize the whole trianglulation by brute force.
 */
int Delaunay::legalize() {
	int nflips = 0;
	for (int i=0; i<nTri; ++i) {
		for (int j=0; j<3; ++j) {
			nflips += legalize(i, j);
		}

		//saveToFile();
	}

	return nflips;
}

int Delaunay::flip_after_insert() {
	int nflips = 0;
	for (int i=0; i<nTri; ++i) {
		if (flip(i, triList[i].flip) == 1) { 	// if flip succesfull, mark next two edges
			nflips++;

			triList[i].flip = 1;
			triList[triList[i].n[2]].flip = 0;
			saveToFile();
		}
		else {                                  // else mark for no flipping
			triList[i].flip = -1;
		}

	}

	return nflips;
}

/*
 * Pick a point by index 'r' in pts and insert it into a triangle which cointains it.
 * Returns the number of a new triangles created. 
 */
int Delaunay::insertPt(int r) {
	int i; //index of triangle in triList
	for (int k=0; k<nTri; ++k) {
		if (triList[k].contains(pts[r]) == 1) {
			i = k;
			break;
		}
	}

	insertPtInTri(r, i);


	return 2;
}
/*
 * Inserts a point into triangle indexed by 'i' (splits the triangle into 3 creating 
 * two new triangles) if possible. Returns the number of a new triangles created.
 *
 * @param i Index of triangle in the array triList.
 */
int Delaunay::insertPtInTri(int r, int i) {

	//std::cout << "[INSERTING] point: " << r << " in triangle: " << i << "\n";

	int p[3] = {triList[i].p[0],
				triList[i].p[1],
				triList[i].p[2]};

	int n[3] = {triList[i].n[0],
				triList[i].n[1],
				triList[i].n[2]};

	int o[3] = {triList[i].o[0],
				triList[i].o[1],
				triList[i].o[2]};

	int p0[3] = {r, p[0], p[1]};
	int n0[3] = {nTri+1, n[0], nTri};
	int o0[3] = {1, o[0], 2};

	int p1[3] = {r, p[1], p[2]};
	int n1[3] = {i, n[1], nTri+1};
	int o1[3] = {1, o[1], 2};

	int p2[3] = {r, p[2], p[0]};
	int n2[3] = {nTri, n[2], i};
	int o2[3] = {1, o[2], 2};

	triList[nTri  ].writeTri(pts, npts, p1, n1, o1);
	triList[nTri+1].writeTri(pts, npts, p2, n2, o2);
	triList[i     ].writeTri(pts, npts, p0, n0, o0);

	// marking edge for flipping
	triList[nTri  ].flip = 1;
	triList[nTri+1].flip = 1;
	triList[i     ].flip = 1;

	// updates neighbour points opposite point if they exist
	if (n[0] >= 0) {
		triList[n[0]].o[(o[0]+1) % 3] = 0;
		triList[n[0]].n[(o[0]+1) % 3] = i;
	}

	if (n[1] >= 0) {
		triList[n[1]].o[(o[1]+1) % 3] = 0;
		triList[n[1]].n[(o[1]+1) % 3] = nTri;
	}

	if (n[2] >= 0) {
		triList[n[2]].o[(o[2]+1) % 3] = 0;
		triList[n[2]].n[(o[2]+1) % 3] = nTri+1;
	}
	
	nTri += 2;		

	// try to make some ascii art diagrams maybe good for explenation
	saveToFile();
	return 2;
}

void Delaunay::initSuperTri() {
	Point avg; 
	for (int i=0; i<npts; ++i) {
		for (int k=0; k<2; ++k) {
			avg.x[k] += pts[i].x[k];
		}
	}
	for (int k=0; k<2; ++k) {
		avg.x[k] = avg.x[k]/npts;
	}

	//std::cout << "Avg point: (" << avg.x[0] << ", " << avg.x[1] << ")\n"; 

	real largest_dist = 0;
	real sample_dist;
	for (int i=0; i<npts; ++i) {
		for (int j=0; j<npts; ++j) {
			sample_dist = dist(pts[i], pts[j]);
			if (largest_dist < sample_dist) {
				largest_dist = sample_dist;
			}
		}
	}

	//std::cout << "largest_dist: " << largest_dist << "\n";

	real center_x = avg.x[0];
	real center_y = avg.x[1];
	real radius = largest_dist;

	pts[npts    ] = Point(center_x + radius*sqrt(3), center_y - radius  );
	pts[npts + 1] = Point(center_x                 , center_y + 2*radius);
	pts[npts + 2] = Point(center_x - radius*sqrt(3), center_y - radius  );

	int p[3] = {npts, npts+1, npts+2};
	int n[3] = {-1, -1, -1}; 
	int o[3] = {-1, -1, -1}; 

	triList[nTri].writeTri(pts, npts, p, n, o);

	nTri++;
}

void Delaunay::saveToFile(bool end) {
	if (end == false) { // save all triangles
		fprintf(trifile, "%d %d\n", iter, nTri);
		for (int i=0; i<nTri; ++i) {
			for (int j=0; j<3; ++j) { fprintf(trifile, "%d ", triList[i].p[j]); } 
			for (int j=0; j<3; ++j) { fprintf(trifile, "%d ", triList[i].n[j]); } 
			for (int j=0; j<3; ++j) { fprintf(trifile, "%d ", triList[i].o[j]); } 
			fprintf(trifile, "%d ", triList[i].flip);
//			fprintf(trifile, "%d ", triList[i].insert);
//			fprintf(trifile, "%d ", triList[i].flipThisIter);

			fprintf(trifile, "\n");
		}

		fprintf(trifile, "\n");
		iter++;
	} 
	else {
		int nTriFinal = 0;	
		// count number of triangles which do not contain the supertriangle points
		for (int i=0; i<nTri; ++i) {
			int cont = 0;
			for (int k=0; k<3; ++k) {
				for (int l=0; l<3; ++l) {
					if (triList[i].p[k] == (npts + l)) {
						cont = -1;
					}
				}
			}
			if (cont == -1) { continue; }
	
			nTriFinal++;
		}
	
		fprintf(trifile, "%d %d\n", iter, nTriFinal);
		//saveFile << iter << " " << nTriFinal << "\n";
		for (int i=0; i<nTri; ++i) {
			// if any point in this triangle is on the boundary dont save
			int cont = 0;
			for (int k=0; k<3; ++k) {
				for (int l=0; l<3; ++l) {
					if (triList[i].p[k] == (npts + l)) {
						cont = -1;
					}
				}
			}
	
			if (cont == -1) {
				continue;
			}
	
			for (int j=0; j<3; ++j) { fprintf(trifile, "%d ", triList[i].p[j]); } 
			for (int j=0; j<3; ++j) { fprintf(trifile, "%d ", triList[i].n[j]); } 
			for (int j=0; j<3; ++j) { fprintf(trifile, "%d ", triList[i].o[j]); } 
			fprintf(trifile, "%d ", triList[i].flip);
//			fprintf(trifile, "%d ", triList[i].insert);
//			fprintf(trifile, "%d ", triList[i].flipThisIter);

			fprintf(trifile, "\n");
		}
 
		fprintf(trifile, "\n");
		iter++;
	}
}
