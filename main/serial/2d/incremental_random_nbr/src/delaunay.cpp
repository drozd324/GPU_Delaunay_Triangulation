#include "delaunay.h"

/*
 * Constructor which creates the delaunay triagulation from an array of 'points' and its lenght 'n'.
 */
Delaunay::Delaunay(Point* points, int n) :
	npts(n), pts(new Point[npts + 3]),
	nTri(0), nTriMax(2*(npts+3) - 2 - 3), triList(new Tri[nTriMax]),
	saveFile("./data/data.txt", std::ios_base::app)
{
	std::cout << "[ALLOCATING] \n";
	std::cout << "npts + 3     = " << npts + 3 << "\n";
	std::cout << "nTriMax      = " <<  nTriMax << "\n";

	for (int i=0; i<npts; i++) {
		pts[i] = points[i];
	}

	std::cout << "INITSUPERTRIANGLE\n";
	initSuperTri();

	// save points data to file
	saveFile << npts+3 << "\n";
	for (int i=0; i<npts+3; ++i) {
		saveFile << pts[i].x[0] << " " << pts[i].x[1] << "\n";
	}
	saveFile << "\n"; 

	saveToFile();

	for (int i=0; i<npts; ++i) { 
		std::cout << "============  PASS " << i << "============ \n"; 
		int inserted = insert();
		std::cout << "inserted: " << inserted << "\n";
		std::cout << "nTri " << nTri << "/" << nTriMax << "\n";
		for (int k=0; k<nTri; ++k) {
			std::cout << k;
			triList[k].print();
		}
		
		if (inserted == 0) {
			break; 
		}

		saveToFile();
	}

	legalize();
}

/*
 * Basic destructor.
 */
Delaunay::~Delaunay() {
	delete[] pts;
	delete[] triList; 
}


/*
 * Performs a flip operation on a triangle 'a' and one of its neighbours denoted by
 * index 0, 1 or 2. Returns 0 if the flip was performed, reuturns -1 if no flip was 
 * performed.
 * 
 * @param a   Index in triList of chosen triangle
 * @param nbr Index in chosen triangle of neighbour. This is an int in 0, 1 or 2.
 * @out Index in chosen triangle of neighbour. This is an int in 0, 1 or 2.
 */
int Delaunay::flip(int a, int edge) {
	int i = edge;

	int b = triList[a].n[i]; // index in triList of nei
	if (b == -1) {
		return -1;
	}

	int opp_idx = triList[a].o[i]; 

	// check if we should flip
	if (0 > incircle(pts[triList[b].p[opp_idx]],
				      pts[triList[a].p[0]],
				      pts[triList[a].p[1]],
				      pts[triList[a].p[2]])) 
	{
		return -1;
	}

	std::cout << "Flipping triangle: " << a << ", egde: " << i << "\n"; 

	std::cout << "t" << a << " ";
	triList[a].print();

	std::cout << "idk | ";
	for (int k=0; k<5; ++k) {
		std::cout << triList[a].p[(k-1)%3] << " ";
	}
	std::cout << "\n";

	// temporary qaud "struct" data  just to make it readable
	int p[4] = {triList[a].p[(i-1 + 3)%3], triList[a].p[i], triList[b].p[opp_idx], triList[a].p[(i+1)%3]};

	int n[4] = {triList[a].n[(i-1 + 3)%3], triList[b].n[(opp_idx-1 + 3)%3],
				triList[b].n[opp_idx], triList[a].n[(i+1)%3]}; 

	int o[4] = {triList[a].o[(i-1 + 3)%3], triList[b].o[(opp_idx-1 + 3)%3],
				triList[b].o[opp_idx], triList[a].o[(i+1)%3]}; 

	std::cout << "QUAD STRUCT: "; 
	for (int k=0; k<4; ++k) {
		std::cout << p[k] << " "; 
	}
	std::cout << "\n"; 

	int ap[3] = {p[0], p[1], p[2]};
	int an[3] = {n[0], n[1], b};
	int ao[3] = {o[0], o[1], 1};

	int bp[3] = {p[2], p[3], p[0]};
	int bn[3] = {n[2], n[3], a};
	int bo[3] = {o[2], o[3], 1};

	std::cout << "OVERWRITING: a "; 
	for (int k=0; k<3; ++k) {
		std::cout << triList[a].p[k] << " "; 
	}
	std::cout << "\n"; 
	std::cout << "OVERWRITING: b "; 
	for (int k=0; k<3; ++k) {
		std::cout << triList[b].p[k] << " "; 
	}
	std::cout << "\n"; 

	std::cout << "WRITING: a "; 
	for (int k=0; k<3; ++k) {
		std::cout << ap[k] << " "; 
	}
	std::cout << "\n"; 
	std::cout << "WRITING: b "; 
	for (int k=0; k<3; ++k) {
		std::cout << bp[k] << " "; 
	}
	std::cout << "\n"; 

	writeTri(a, ap, an, ao);
	writeTri(b, bp, bn, bo);

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

	std::cout << "COMPLETE\n";
	saveToFile();
	return 0;
}

/*
 * Function to legalize a given triangle in triList with index 'a', with edge 'e'.
 */
void Delaunay::legalize(int a, int e) {
	if (flip(a, e) == -1) {
		return;
	}

	legalize(a, 1);
	legalize(triList[a].n[2], 0);
}

/*
 * Function to legalize a given triangle in triList with index a.
 */
int Delaunay::legalize() {
	for (int i=0; i<nTri; ++i) {
		for (int j=0; j<3; ++j) {
			legalize(i, j);
		}
	}

	return 0;
}

/*
 * Inserts a point into triangles which contain points inside of them. The point is 
 * chosen to be the closest to the circumcenter of this triangle if available. This 
 * function also reutrns the number of triangles added into the triangulation.
 */
int Delaunay::insert() {
	int num_inserted_tri = 0;

	int max = nTri;
	for (int i=0; i<max; ++i) {
		int center = triList[i].get_center();
		//std::cout << "center=" << center << "\n";
		//std::cout << i;
		triList[i].print();

		if (center == -1) { // if center doesnt exist, continue
			continue;
		}

		int p[3] = {triList[i].p[0],
					triList[i].p[1],
					triList[i].p[2]};

		int n[3] = {triList[i].n[0],
					triList[i].n[1],
					triList[i].n[2]};

		int o[3] = {triList[i].o[0],
					triList[i].o[1],
					triList[i].o[2]};

		int p0[3] = {center, p[0], p[1]};
		int n0[3] = {nTri+1, n[0], nTri};
		int o0[3] = {1, o[0], 2};

		int p1[3] = {center, p[1], p[2]};
		int n1[3] = {i, n[1], nTri+1};
		int o1[3] = {1, o[1], 2};

		int p2[3] = {center, p[2], p[0]};
		int n2[3] = {nTri, n[2], i};
		int o2[3] = {1, o[2], 2};


		writeTri(nTri, p1, n1, o1);
		writeTri(nTri+1, p2, n2, o2);
		writeTri(i     , p0, n0, o0);

		// updates neighbour points opposite point if they exist
		//[ nbr tri  ]  [                          ]
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


		// try to make some ascii art diagrams maybe good for explenation

		nTri += 2;		
		num_inserted_tri += 2;

		//saveToFile();

	}

	return num_inserted_tri;
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

	real center_x = avg.x[0];
	real center_y = avg.x[1];
	real radius = largest_dist;

	pts[npts    ] = Point(center_x + radius*sqrt(3), center_y - radius  );
	pts[npts + 1] = Point(center_x                 , center_y + 2*radius);
	pts[npts + 2] = Point(center_x - radius*sqrt(3), center_y - radius  );

	int p[3] = {npts, npts+1, npts+2};
	int n[3] = {-1, -1, -1}; 
	int o[3] = {-1, -1, -1}; 
	writeTri(nTri, p, n, o);
	nTri++;
}

void Delaunay::writeTri(int index, int triPts[3], int triNeighbours[3], int triOpposite[3]) {
	triList[index].pts = pts;
	triList[index].npts = npts;

	for (int i=0; i<3; ++i) {
		triList[index].p[i] = triPts[i];
		triList[index].n[i] = triNeighbours[i];
		triList[index].o[i] = triOpposite[i];
	}

	triList[index].status = 1;
	triList[index].tag = tag_num++;
}

void Delaunay::saveToFile() {
	saveFile << iter << " " << nTri << "\n";
	for (int i=0; i<nTri; ++i) {
		for (int j=0; j<3; ++j) {
			saveFile << triList[i].p[j] << " "; 
		} 
		for (int j=0; j<3; ++j) {
			saveFile << triList[i].n[j] << " "; 
		} 
		for (int j=0; j<3; ++j) {
			saveFile << triList[i].o[j] << " "; 
		} 
		saveFile << "\n"; 
	}

	saveFile << "\n"; 
	iter++;
}
