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
	cudaMemcpy(pts_d, pts, npts * sizeof(Point), cudaMemcpyHostToDevice);

	initSuperTri();


	// save points data to file
	saveFile << npts+3 << "\n";
	for (int i=0; i<npts+3; ++i) {
		saveFile << pts[i].x[0] << " " << pts[i].x[1] << "\n";
	}
	saveFile << "\n"; 

	cudaMemcpy(triList, triList_d, nTriMax * sizeof(Tri), cudaMemcpyDeviceToHost);
	saveToFile();

//	for (int i=0; i<npts; ++i) { 
//		std::cout << "============[PASS " << i << "]============ \n"; 
//		
//		// ==== MARK TRIANGLES FOR INSERTION ====
//		int num_to_insert  = checkInsert();
//		std::cout << "num to insert: " << num_to_insert << "\n";
//
//		//cudaMemcpy(triList_d, triList, sizeof(Tri), cudaMemcpyDeviceToHost);
//
//		// ==== INSERT ====
//		int num_inserted_tri = insert();
//		std::cout << "number of inserted intrangles: " << num_inserted_tri << "\n";
//
//		if (num_inserted_tri == 0) {
//			break; 
//		}
//
//		std::cout << "nTri " << nTri << "/" << nTriMax << "\n";
//	}
	//

	// copy everything back to host
		
	cudaFree(pts_d);
	cudaFree(npts_d);

	cudaFree(triList_d);
	cudaFree(nTri_d);
	cudaFree(nTriMax_d);
	

//	int nflips = -1;
//	while (nflips != 0) {
//		nflips = legalize();
//		std::cout << "Performed	" << nflips  << " additional flips\n"; 
//	}
//
//	std::cout << "Triangluation is Delaunay\n";

	//saveToFile(true);
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

/*
 * Pick a triangle by index 'i' in triList and insert its center point.
 * Returns the number of a new triangles created. 
 */
//__device__ int Delaunay::insertInTri(int i) {
//	int r = triList[i].center;
//
//	if (r == -1) { // if no points inside this triangle, continue
//		return 0;
//	}
//
//	insertPtInTri(r, i);
//
//	return 2;
//}

/*
 * Inserts a point into triangle indexed by 'i' (splits the triangle into 3 creating 
 * two new triangles) if possible. Returns the number of a new triangles created.
 *
 * @param i Index of triangle in the array triList.
 */
//__device__ int Delaunay::insertPtInTri(int r, int i) {
//
//	int p[3] = {triList[i].p[0],
//				triList[i].p[1],
//				triList[i].p[2]};
//
//	int n[3] = {triList[i].n[0],
//				triList[i].n[1],
//				triList[i].n[2]};
//
//	int o[3] = {triList[i].o[0],
//				triList[i].o[1],
//				triList[i].o[2]};
//
//	int p0[3] = {r, p[0], p[1]};
//	int n0[3] = {nTri+1, n[0], nTri};
//	int o0[3] = {1, o[0], 2};
//
//	int p1[3] = {r, p[1], p[2]};
//	int n1[3] = {i, n[1], nTri+1};
//	int o1[3] = {1, o[1], 2};
//
//	int p2[3] = {r, p[2], p[0]};
//	int n2[3] = {nTri, n[2], i};
//	int o2[3] = {1, o[2], 2};
//
//
//	int nspts = triList[i].nlpts;
//	int* spts = new int[nspts];
//	for (int k=0; k<nspts; ++k) {
//		spts[k] = triList[i].lpts[k];
//	}
//
//	triList[nTri  ].writeTri(pts, npts, spts, nspts, p1, n1, o1);
//	triList[nTri+1].writeTri(pts, npts, spts, nspts, p2, n2, o2);
//	triList[i     ].writeTri(pts, npts, spts, nspts, p0, n0, o0);
//
//	// marking edge for flipping
//	triList[nTri  ].flip = 1;
//	triList[nTri+1].flip = 1;
//	triList[i     ].flip = 1;
//
//	delete[] spts;
//
//	// updates neighbour points opposite point if they exist
//	if (n[0] >= 0) {
//		triList[n[0]].o[(o[0]+1) % 3] = 0;
//		triList[n[0]].n[(o[0]+1) % 3] = i;
//	}
//
//	if (n[1] >= 0) {
//		triList[n[1]].o[(o[1]+1) % 3] = 0;
//		triList[n[1]].n[(o[1]+1) % 3] = nTri;
//	}
//
//	if (n[2] >= 0) {
//		triList[n[2]].o[(o[2]+1) % 3] = 0;
//		triList[n[2]].n[(o[2]+1) % 3] = nTri+1;
//	}
//	
//	nTri += 2;		
//
//	// try to make some ascii art diagrams maybe good for explenation
//	//saveToFile();
//
//	return 2;
//}

/*
 * Inserts a point into triangles which contain points inside of them. The point is 
 * chosen to be the closest to the circumcenter of this triangle if available. This 
 * function also reutrns the number of triangles added into the triangulation.
 */
//__device__ int Delaunay::insert() {
//	int num_inserted_tri = 0;
//
//	int max = nTri;
//	for (int i=0; i<max; ++i) {
//		num_inserted_tri += insertInTri(i);
//	}
//
//	return num_inserted_tri;
//}

/*
 *
 */
//__device__ int Delaunay::checkInsert() {
//	int num_to_insert = 0;
//	for (int i=0; i<nTri; ++i) {
//		if (triList[i].spts_alloc == true) { // triList[i].nspts > 0 && 
//			triList[i].get_center();
//			num_to_insert++;
//		}
//	}
//	
//	return num_to_insert;
//}

void Delaunay::initSuperTri() {

	Point* avgPoint;
	Point* avgPoint_d;

	// computing the average point
	avgPoint.x[0] = 0;
	avgPoint.x[1] = 0;
	cudaMalloc(&avgPoint_d, sizeof(Point));
	cudaMemcpy(avgPoint_d, avgPoint, sizeof(Point), cudaMemcpyHostToDevice);

	printf("avgPoint before: (%d, %d)\n", avgPoint.x[0], avgPoint.x[1]);

	dim3 threadsPerBlock(warpSize);
	dim3 numBlocks1((2*npts)/threadsPerBlock.x + (!((2*npts) % threadsPerBlock.x) ? 0:1));
	computeAvgPoint<<<numBlocks1, threadsPerBlock>>>(pts_d, npts_d, &avgPoint_d);

	cudaMemcpy(avgPoint, avgPoint_d, sizeof(Point), cudaMemcpyDeviceToHost);
	printf("avgPoint After: (%d, %d)\n", avgPoint.x[0], avgPoint.x[1]);

//	// computing the largest distance bewtween two points
//	int  largest_dist;
//	int* largest_dist_d;
//	cudaMalloc(&largest_dist_d, sizeof(float));
//
//	int ncomps = (npts*(npts+1)) / 2; //number of comparisons
//	dim3 numBlocks2(ncomps/threadsPerBlock.x + (!(ncomps % threadsPerBlock.x) ? 0:1));
//	computeMaxDistPts<<<numBlocks2, threadsPerBlock>>>(pts_d, npts_d, &largest_dist);
//
//	cudaMemcpy(&largest_dist, &largest_dist_d, sizeof(float), cudaMemcpyDeviceToHost);
//
//	// writing supertriangle points to pts
//	float center_x = avgPoint.x[0];
//	float center_y = avgPoint.x[1];
//	float radius = largest_dist;
//
//	pts[npts    ] = Point(center_x + radius*1.73205, center_y - radius  );
//	pts[npts + 1] = Point(center_x                 , center_y + 2*radius);
//	pts[npts + 2] = Point(center_x - radius*1.73205, center_y - radius  );
//
//	// copying supertriangle points to device
//	cudaMemcpy(&(pts_d[npts]), &(pts[npts]), 3 * sizeof(Point), cudaMemcpyHostToDevice);
//	cudaFree(&avgPoint);
//	cudaFree(&largest_dist_d);
//
//	// writing supertriangle on host
//	int p[3] = {npts, npts+1, npts+2};
//	int n[3] = {-1, -1, -1}; 
//	int o[3] = {-1, -1, -1}; 
//
//	// writing supertriangle on host
//	triList[nTri].writeTri(pts, npts, spts, npts, p, n, o);
//
//	cudaMemcpy(&(triList_d[0]), &(triList[0]), sizeof(Tri), cudaMemcpyHostToDevice);
//	setPtsAll<<<1, 1>>>(npts_d, triList_d);
//
//	nTri++;
//	cudaMemcpy(&nTri_d, &nTri, sizeof(int), cudaMemcpyHostToDevice);
}

void Delaunay::cpyToHost() {
	cudaMemcpy(triList_d, triList, nTri * sizeof(Tri), cudaMemcpyDeviceToHost);
}

/* ============================================= GPU CODE ============================================= */

//void Delaunay::gpu_compute() {}

/*
 * Sets the spts (search points) of the chosen triangle to be all of the points 
 * of the final triangulation.
 */
__global__ void setSptsAll(int* npts_d, Tri* triList_d, int i=0) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0) {
		triList_d[i].nspts = npts_d[0];

		triList_d[i].spts = new int[npts_d[0]];
		for (int i=0; i<npts_d[0]; ++i) {
			triList_d[i].spts[i] = i;
		}
	}
}

__global__ void computeAvgPoint(Point* pts_d, int* npts_d, Point *avgPoint_d) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < 2*npts_d[0]) {
		atomicAdd(&(avgPoint_d->x[idx%2]), pts_d[idx/2].x[idx%2]);
		//atomicAdd(&(avgPoint_d.x[idx%2]), pts_d[idx/2].x[idx%2]);
		//atomicAdd(&(avgPoint_d->(idx%2)), pts_d[idx/2].x[idx%2]);
		//atomicAddFloat(&(avgPoint_d[i%2]), pts_d[i/2].x[i%2]);
	}

	__syncthreads();

	// division
	if (idx < 2) {
		avgPoint_dx[idx] = avgPoint_d.x[idx]/npts_d[0];
	}
}

__global__ void computeMaxDistPts(Point* pts_d, int* npts_d, float* largest_dist_d) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < ((npts_d[0]*(npts_d[0] + 1)) / 2)) {
		// for now yoinnked from chet gipiti but uses triangilar number math to get convenient indexing
		int i = (int)((2*npts_d[0] - 1 - sqrt((2*npts_d[0] - 1) * (2*npts_d[0] - 1) - 8*idx)) / 2);
		int j = idx - (i*(2*npts_d[0] - i - 1) / 2) + i + 1;

		atomicMaxFloat(largest_dist_d, dist(pts_d[i], pts_d[j])); 
	}
}

/* =============================== ATOMIC ================================ */
__device__ float atomicAddFloat(float* address, float val) {
	int* address_as_ull = (int*)address;
	int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __float_as_int(val + __int_as_float(assumed)));
	} while (assumed != old);
	return __int_as_float(old);
}

__device__ float atomicMaxFloat(float* address, float val) {
	int* address_as_ull = (int*)address;
	int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__float_as_int(max(val,__int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__device__ float atomicMinFloat(float* address, float val) {
	int* address_as_ull = (int*)address;
	int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__float_as_int(min(val,__int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

//__global__ checkInsert_gpu() {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//	num_tris_to_insert_d = 0;
//	if (idx < nTri_d) {
//		if (triList_d[idx].spts_alloc == true) { // triList[i].nspts > 0 && 
//			triList_d[idx].get_center();
//			atomicAdd(&num_tris_to_insert_d, 1);
//		}
//	}
//}
//
//__global__ insert_gpu() {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//	//int num_inserted_tri = 0;
//	//int max = nTri;
//	if (idx<max) {
//		insertInTri(idx);
//	}
//}


/* ====================================================================================================== */


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

