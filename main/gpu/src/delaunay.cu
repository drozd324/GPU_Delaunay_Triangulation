#include "delaunay.h"


/*
 * Constructor which creates the delaunay triagulation from an array of 'points'
 * and its lenght 'n'.
 */
Delaunay::Delaunay(Point* points, int n) :
	npts(n), pts(new Point[npts + 3]),
	nTri(0), nTriMax(2*(npts+3) - 2 - 3), triList(new Tri[nTriMax]),
	saveFile("./data/data.txt", std::ios_base::app) 
{
	for (int i=0; i<npts; i++) {
		pts[i] = points[i];
	}

	// alloc points
	cudaMalloc(&npts_d, sizeof(int));
	cudaMalloc(&pts_d , (npts+3) * sizeof(Point));

	// alloc triangles
	cudaMalloc(&nTri_d   , sizeof(int));
	cudaMalloc(&nTriMax_d, sizeof(int));
	cudaMalloc(&triList_d, nTriMax * sizeof(Tri));
	
	// counters
	cudaMalloc(&num_tris_to_insert_d, sizeof(int));

	// copying exitsting info to gpu
	cudaMemcpy(pts_d, pts, (npts+3) * sizeof(Point), cudaMemcpyHostToDevice);
	cudaMemcpy(triList_d, triList, sizeof(Tri), cudaMemcpyHostToDevice);

//	dim3 threadsPerBlock(warpSize);
//	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));

	initSuperTri();

	for (int i=0; i<npts; ++i) { 
		std::cout << "============[PASS " << i << "]============ \n"; 
		
		// ==== MARK TRIANGLES FOR INSERTION ====
		int num_to_insert  = checkInsert();
		std::cout << "num to insert: " << num_to_insert << "\n";

		//cudaMemcpy(triList_d, triList, sizeof(Tri), cudaMemcpyDeviceToHost);

		// ==== INSERT ====
		int num_inserted_tri = insert();
		std::cout << "number of inserted intrangles: " << num_inserted_tri << "\n";

		if (num_inserted_tri == 0) {
			break; 
		}

		std::cout << "nTri " << nTri << "/" << nTriMax << "\n";
	}
	//
		
	cudaFree(pts_d);
	cudaFree(triList_d);

	// save points data to file
	saveFile << npts+3 << "\n";
	for (int i=0; i<npts+3; ++i) {
		saveFile << pts[i].x[0] << " " << pts[i].x[1] << "\n";
	}
	saveFile << "\n"; 
	saveToFile();


//	int nflips = -1;
//	while (nflips != 0) {
//		nflips = legalize();
//		std::cout << "Performed	" << nflips  << " additional flips\n"; 
//	}
//
//	std::cout << "Triangluation is Delaunay\n";

	saveToFile(true);
}

/*
 * Basic destructor.
 */
Delaunay::~Delaunay() {
	delete[] triList; 
	delete[] pts;
}

void Delaunay::notparallel() {
	std::cout << "\n====[\"PARALLEL\"]====\n" ;

	for (int i=0; i<npts; ++i) { 
		std::cout << "============[PASS " << i << "]============ \n"; 
		
		// ==== MARK TRIANGLES FOR INSERTION ====
		int num_to_insert  = checkInsert();
		std::cout << "num to insert: " << num_to_insert << "\n";

		// ==== INSERT ====
		int num_inserted_tri = insert();
		std::cout << "number of inserted intrangles: " << num_inserted_tri << "\n";

		if (num_inserted_tri == 0) {
			break; 
		}

		// ==== FLIP ====
		int total_flips = 0;
		int nflips = -1;
		while (nflips != 0) {
			nflips = flip_after_insert();
			total_flips += nflips;
		}
		std::cout << "num flips: " << total_flips << "\n";

		std::cout << "nTri " << nTri << "/" << nTriMax << "\n";
	}
}


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

	int nspts = triList[a].nspts + triList[b].nspts;
	int* spts = new int[nspts];
	for (int k=0; k<triList[a].nspts; ++k) {
		spts[k] = triList[a].spts[k];
	}
	for (int k=0; k<triList[b].nspts; ++k) {
		spts[triList[a].nspts + k] = triList[b].spts[k];
	}

	triList[a].writeTri(pts, npts, spts, nspts, ap, an, ao);
	triList[b].writeTri(pts, npts, spts, nspts, bp, bn, bo);

	delete[] spts;

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

__device__ int Delaunay::flip_after_insert() {
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
 * Pick a triangle by index 'i' in triList and insert its center point.
 * Returns the number of a new triangles created. 
 */
__device__ int Delaunay::insertInTri(int i) {
	int r = triList[i].center;

	if (r == -1) { // if no points inside this triangle, continue
		return 0;
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
__device__ int Delaunay::insertPtInTri(int r, int i) {

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


	int nspts = triList[i].nlpts;
	int* spts = new int[nspts];
	for (int k=0; k<nspts; ++k) {
		spts[k] = triList[i].lpts[k];
	}

	triList[nTri  ].writeTri(pts, npts, spts, nspts, p1, n1, o1);
	triList[nTri+1].writeTri(pts, npts, spts, nspts, p2, n2, o2);
	triList[i     ].writeTri(pts, npts, spts, nspts, p0, n0, o0);

	// marking edge for flipping
	triList[nTri  ].flip = 1;
	triList[nTri+1].flip = 1;
	triList[i     ].flip = 1;

	delete[] spts;

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
	//saveToFile();

	return 2;
}

/*
 * Inserts a point into triangles which contain points inside of them. The point is 
 * chosen to be the closest to the circumcenter of this triangle if available. This 
 * function also reutrns the number of triangles added into the triangulation.
 */
__device__ int Delaunay::insert() {
	int num_inserted_tri = 0;

	int max = nTri;
	for (int i=0; i<max; ++i) {
		num_inserted_tri += insertInTri(i);
	}

	return num_inserted_tri;
}

/*
 *
 */
__device__ int Delaunay::checkInsert() {
	int num_to_insert = 0;
	for (int i=0; i<nTri; ++i) {
		if (triList[i].spts_alloc == true) { // triList[i].nspts > 0 && 
			triList[i].get_center();
			num_to_insert++;
		}
	}
	
	return num_to_insert;
}

//void Delaunay::initSuperTri() {
void Delaunay::initSuperTri() {

	avgPoint.x[0] = 0; avgPoint.x[1] = 0;
	cudaMalloc(&avgPoint_d, sizeof(Point));
	cudaMemcpy(&avgPoint_d, &avgPoint, sizeof(Point), cudaMemcpyHostToDevice);

	//cudaFree(avgPoint);

	dim3 threadsPerBlock(warpSize);
	dim3 numBlocks((2*npts)/threadsPerBlock.x + (!((2*npts) % threadsPerBlock.x) ? 0:1));
	computeAvgPoint<<<numBlocks, threadsPerBlock>>>(pts_d, npts_d, &avgPoint_d);

	avgPoint.x[0] = 0; avgPoint.x[1] = 0;
	cudaMalloc(&avgPoint_d, sizeof(Point));
	cudaMemcpy(&avgPoint_d, &avgPoint, sizeof(Point), cudaMemcpyHostToDevice);

	//cudaFree(avgPoint);
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

	//sqrt(3)~=1.73205
	pts[npts    ] = Point(center_x + radius*1.73205, center_y - radius  );
	pts[npts + 1] = Point(center_x                 , center_y + 2*radius);
	pts[npts + 2] = Point(center_x - radius*1.73205, center_y - radius  );

	int p[3] = {npts, npts+1, npts+2};
	int n[3] = {-1, -1, -1}; 
	int o[3] = {-1, -1, -1}; 

	int nspts = npts;
	int* spts = new int[nspts];
	//std::cout << "flip nspts: " << nspts << "\n";
	for (int k=0; k<npts; ++k) {
		spts[k] = k;
	}

	triList[nTri].writeTri(pts, npts, spts, nspts, p, n, o);

//	head_node.t = 0;
//	head_node.n[0] = -1;
//	head_node.n[1] = -1;
//	head_node.n[2] = -1;

	delete[] spts;
	nTri++;
}

void Delaunay::saveToFile(bool end) {

	if (end == false) { // save all triangles

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
			saveFile << triList[i].flip << " "; 

			saveFile << "\n"; 
		}

		saveFile << "\n"; 
		iter++;
	}

	else { // save triangulation with super triangle points removed

		// count number of triangles which do not contain the supertriangle points
		int nTriFinal = 0;	
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
	
			nTriFinal++;	
		}
	
		// save
		saveFile << iter << " " << nTriFinal << "\n";
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
	
			for (int j=0; j<3; ++j) {
				saveFile << triList[i].p[j] << " "; 
			} 
			for (int j=0; j<3; ++j) {
				saveFile << triList[i].n[j] << " "; 
			} 
			for (int j=0; j<3; ++j) {
				saveFile << triList[i].o[j] << " "; 
			} 
			saveFile << triList[i].flip << " "; 
			saveFile << "\n"; 
		}
	
		saveFile << "\n"; 
		iter++;
	}
}

/* ============================================= GPU CODE ============================================= */

void Delaunay::gpu_compute() {
}

__global__ computeAvgPoint(Point* pts_d, int npts_d, Point *avgPoint_d) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int size = 2*npts_d;

	// x coord
	if (i < npts) {
		atomicAdd(&(avgPoint[0], pts[i].x[0]);
	}

	// y coord
	if (npts < i && i < (2*npts)) {
		atomicAdd(&(avgPoint[1], pts[i].x[1]);
	}

	__syncthreads();

	// division
	if (i < 2) {
		avg.x[i] = avg.x[i]/npts;
	}
}

__device__ double atomicMinDouble(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(min(val,__longlong_as_double(assumed))));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void apply_atomic_operation( double *input, int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		atomicAddDouble (&(input[idx]),1.5f);
		//atomicMaxDouble (&(input[idx]),5.5f);
		//atomicMinDouble (&(input[idx]),5.5f);
	}
}
__global__ checkInsert_gpu() {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	num_tris_to_insert_d = 0;
	if (idx < nTri_d) {
		if (triList_d[idx].spts_alloc == true) { // triList[i].nspts > 0 && 
			triList_d[idx].get_center();
			atomicAdd(&num_tris_to_insert_d, 1);
		}
	}
}

__global__ insert_gpu() {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//int num_inserted_tri = 0;
	//int max = nTri;
	if (idx<max) {
		insertInTri(idx);
	}
}
