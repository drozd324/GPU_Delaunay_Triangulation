#include "delaunay.h"


/*
 * Constructor which creates the delaunay triagulation from an array of 'points'
 * and its lenght 'n'.
 */
Delaunay::Delaunay(Point* points, int n) : 
	saveFile("./data/data.txt", std::ios_base::app)
{
	// ============= allocation on host ============  
	// alloc points
	*npts = n;
	pts = (Point*) malloc(((*npts) + 3) * sizeof(Point));

	// alloc triangles
	*nTri = 0;
	*nTriMax = 2*((*npts)+3) - 2 - 3;
	triList = (Tri*) malloc((*nTriMax) * sizeof(Tri));

	// maps
	ptToTri = (int*) malloc(npts[0] * sizeof(int));
	triWithInsert = (int*) malloc((*nTriMax) * sizeof(int));

	for (int i=0; i<(*npts); i++) {
		pts[i] = points[i];
	}

	// setting default values
	*nTriWithInsert = 0;

	// ============= alloc on device ============ 
	// alloc points
	cudaMalloc(&npts_d,               sizeof(int)  );
	cudaMalloc(&pts_d , ((*npts)+3) * sizeof(Point));

	// alloc triangles
	cudaMalloc(&nTri_d   , sizeof(int));
	cudaMalloc(&nTriMax_d, sizeof(int));
	cudaMalloc(&triList_d, (*nTriMax) * sizeof(Tri));
	
	// maps
	cudaMalloc(&ptToTri_d      , (*npts) * sizeof(Point));
	cudaMalloc(&triWithInsert_d, (*nTriMax) * sizeof(int));

	// counters
	cudaMalloc(&nTriWithInsert_d, sizeof(int));

	// copying exitsting info to gpu
	cudaMemcpy(pts_d           , pts            , (*npts) * sizeof(Point), cudaMemcpyHostToDevice);
	cudaMemcpy(nTriMax_d       , nTriMax        ,           sizeof(int)  , cudaMemcpyHostToDevice);
	cudaMemcpy(nTriWithInsert_d, nTriWithInsert ,           sizeof(int)  , cudaMemcpyHostToDevice);

	cudaMemset(triWithInsert_d, -1, (*nTriMax) * sizeof(int));

	// ============= initialize super triangle ============ 
	initSuperTri();
	cudaDeviceSynchronize();

	// save points data to file
	saveFile << (*npts) + 3 << "\n";
	for (int i=0; i<(*npts) + 3; ++i) {
		saveFile << pts[i].x[0] << " " << pts[i].x[1] << "\n";
	}
	saveFile << "\n"; 

	cudaMemcpy(triList, triList_d, (*nTriMax) * sizeof(Tri), cudaMemcpyDeviceToHost);
	saveToFile();

	// =============== Main compute loop ================ 

	//for (int i=0; i<*npts; ++i) { 
	for (int i=0; i<1; ++i) {
		std::cout << "============[PASS " << i << "]============ \n"; 

		prepForInsert();
		cpyToHost();

		cudaDeviceSynchronize();

		std::cout << "ptToTri: ";
		for (int l=0; l<(*npts); ++l) {
			std::cout << ptToTri[l] << ", ";
		}
		std::cout << "\n";

//		insert();
//		
//		// ==== MARK TRIANGLES FOR INSERTION ====
//		int num_to_insert  = checkInsert();
//		std::cout << "num to insert: " << num_to_insert << "\n";
//
//		cudaMemcpy(triList_d, triList, sizeof(Tri), cudaMemcpyDeviceToHost);
//
//		// ==== INSERT ====
//		int num_inserted_tri = insert();
//		std::cout << "number of inserted intrangles: " << num_inserted_tri << "\n";
//
//		if (num_inserted_tri == 0) {
//			break; 
//		}

		std::cout << "nTri " << *nTri << "/" << *nTriMax << "\n";
	}

	cudaFree(pts_d);
	cudaFree(npts_d);

	cudaFree(triList_d);
	cudaFree(nTri_d);
	cudaFree(nTriMax_d);

	cudaFree(ptToTri_d);
	cudaFree(triWithInsert_d);

	cudaFree(nTriWithInsert);
	
	free(triList); 
	free(pts);
	free(ptToTri);

//	int nflips = -1;
//	while (nflips != 0) {
//		nflips = legalize();
//		std::cout << "Performed	" << nflips  << " additional flips\n"; 
//	}
//
//	std::cout << "Triangluation is Delaunay\n";

	//saveToFile(true);
}

/* ================================== COPY TO HOST ======================================== */

void Delaunay::cpyToHost() {
	cudaMemcpy(triList, triList_d, (*nTri) * sizeof(Tri), cudaMemcpyDeviceToHost);
	cudaMemcpy(ptToTri, ptToTri_d, (*npts) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(nTri   , nTri_d   , sizeof(int), cudaMemcpyDeviceToHost);
}

/* ================================== INIT SUPERTRIANGLE ======================================== */

/*
 *
 */
void Delaunay::initSuperTri() {

	Point avgPoint[1];
	Point* avgPoint_d;

	// computing the average point
	avgPoint->x[0] = 0;
	avgPoint->x[1] = 0;
	cudaMalloc(&avgPoint_d, sizeof(Point));
	cudaMemcpy(avgPoint_d, avgPoint, sizeof(Point), cudaMemcpyHostToDevice);

	printf("avgPoint before: (%f, %f)\n", avgPoint->x[0], avgPoint->x[1]);

	//dim3 threadsPerBlock(warpSize);
	int valsToAdd = 2*(*npts); 
	dim3 threadsPerBlock(32);
	dim3 numBlocks1(valsToAdd/threadsPerBlock.x + (!(valsToAdd % threadsPerBlock.x) ? 0:1));

	sumPoints<<<numBlocks1, threadsPerBlock>>>(pts_d, (*npts), avgPoint_d);

	cudaMemcpy(avgPoint, avgPoint_d, sizeof(Point), cudaMemcpyDeviceToHost);

	avgPoint->x[0] /= npts[0];
	avgPoint->x[1] /= npts[0];

	printf("avgPoint After: (%f, %f)\n", avgPoint->x[0],
										 avgPoint->x[1]);

	// computing the largest distance bewtween two points
	largest_dist[0] = 0;
	cudaMalloc(&largest_dist_d, sizeof(float));
	cudaMemcpy(largest_dist_d, largest_dist, sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemset(largest_dist_d, 0, sizeof(float));

	printf("before largest_dist: %f\n", *largest_dist);

	int ncomps = ((*npts)*((*npts) - 1)) / 2; //number of comparisons
	dim3 numBlocks2(ncomps/threadsPerBlock.x + (!(ncomps % threadsPerBlock.x) ? 0:1));
	computeMaxDistPts<<<numBlocks2, threadsPerBlock>>>(pts_d, (*npts), largest_dist_d);

	cudaMemcpy(largest_dist, largest_dist_d, sizeof(float), cudaMemcpyDeviceToHost);

	printf("after largest_dist: %f\n", *largest_dist);

	// writing supertriangle points to pts
	float center_x = avgPoint->x[0];
	float center_y = avgPoint->x[1];
	float radius = *largest_dist;

	pts[npts[0]    ].x[0] = center_x + radius*1.73205;
	pts[npts[0]    ].x[1] = center_y - radius; 

	pts[npts[0] + 1].x[0] = center_x;
	pts[npts[0] + 1].x[1] = center_y + 2*radius;

	pts[npts[0] + 2].x[0] = center_x - radius*1.73205;
	pts[npts[0] + 2].x[1] = center_y - radius;

	// copying supertriangle points to device
	cudaMemcpy(&(pts_d[(*npts)]), &(pts[(*npts)]), 3 * sizeof(Point), cudaMemcpyHostToDevice);
	cudaFree(avgPoint_d);
	cudaFree(largest_dist_d);

	// writing supertriangle on host
	int p[3] = {(*npts), (*npts) + 1, (*npts) + 2};
	int n[3] = {-1, -1, -1}; 
	int o[3] = {-1, -1, -1}; 

	// writing supertriangle on host
	writeTri(&(triList[0]), p, n, o);
	triList[0].insertPt_dist = *largest_dist;

	memset(ptToTri, 0, (*npts) * sizeof(int));
	cudaMemset(ptToTri_d, 0, (*npts) * sizeof(int));
	cudaMemcpy(&(triList_d[0]), &(triList[0]), sizeof(Tri), cudaMemcpyHostToDevice);

	cudaMemset(nTriWithInsert_d, 1, sizeof(int));
	cudaMemset(triWithInsert_d, 0, sizeof(int)); // set fist element of this array to be 

	cudaMemset(triWithInsert_d, 0, sizeof(int)); // set fist element of this array to be 0, ie the super triangle just constructed 

	(*nTri)++;
	cudaMemcpy(nTri_d, nTri, sizeof(int), cudaMemcpyHostToDevice);
}

/*
 * Computes a vector sum of points provided in pts and stores them in avgPoint.
 */
__global__ void sumPoints(Point* pts, int npts, Point *avgPoint) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < 2*npts) {
		atomicAddFloat(&(avgPoint->x[idx%2]), pts[idx/2].x[idx%2]);
	}
}

/*
 * Computes the maximum distance bewteen two points in the array pts and stores this value in largest_dist.
 */
__global__ void computeMaxDistPts(Point* pts, int npts, float* largest_dist) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;
	float dist; 

	int count = (npts*(npts - 1)) / 2; 
	if (idx < count) {
		// for now yoinnked from chet gipiti but uses triangilar number math to get convenient indexing
		i = (int)((2*npts - 1 - sqrtf((2*npts - 1) * (2*npts - 1) - 8*idx)) / 2);
		j = idx - (i*(2*npts - i - 1) / 2) + i + 1;

		printf("index: %d | count: %d | i,j =(%d, %d)\n", idx, count, i, j);

		dist = sqrtf( (pts[i].x[0] - pts[j].x[0])*(pts[i].x[0] - pts[j].x[0]) +
				      (pts[i].x[1] - pts[j].x[1])*(pts[i].x[1] - pts[j].x[1]));

		if (dist > (*largest_dist)) {
			atomicMaxFloat(largest_dist, dist); 
		}
	}
}

__host__ __device__ void writeTri(Tri* tri, int* p, int* n, int* o) {
	for (int i=0; i<3; i++) {
		tri->p[i] = p[i];
		tri->n[i] = n[i];
		tri->o[i] = o[i];
	}

}

/* ================================== PREP FOR INSERT ======================================== */

/*
 * Prepares points to be inserted by finding the shortest distance bewteen a point in a triangle and
 * circumcenter of this triangle, then searches for the point which has this distance. 
 */
void Delaunay::prepForInsert() {

	dim3 threadsPerBlock(32);
	dim3 numBlocks((*npts)/threadsPerBlock.x + (!((*npts) % threadsPerBlock.x) ? 0:1));

	setInsertPtsDistance<<<numBlocks, threadsPerBlock>>>(pts_d, *npts, triList_d, ptToTri_d);
	setInsertPts<<<numBlocks, threadsPerBlock>>>(pts_d, *npts, triList_d, ptToTri_d);
}

/*
 * Writes the shortest distance between a point in each triangle and the center
 * of its circumcircle to triangles slot named 'insertPt_dist'.
 */
__global__ void setInsertPtsDistance(Point* pts, int npts, Tri* triList, int* ptToTri) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < npts) {
		if (ptToTri[idx] >= 0) {      // for uninserted points
			int idxTri = ptToTri[idx];

			Point center;
			circumcircle_center(pts[triList[idxTri].p[0]], 
					            pts[triList[idxTri].p[1]],
					            pts[triList[idxTri].p[2]],
					            &center);

			triList[idxTri].circumcenter.x[0] = center.x[0]; 
			triList[idxTri].circumcenter.x[1] = center.x[1]; 

			float ptDist = dist(center, pts[idx]); 
			atomicMinFloat(&(triList[idxTri].insertPt_dist), ptDist);
		}
	}
}

/*
 * Writes the index of the point int the array 'pts' to be inserted into each triangle
 * in the triangles slot named 'insertPt'.
 */
__global__ void setInsertPts(Point* pts, int npts, Tri* triList, int* ptToTri) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < npts) {
		if (ptToTri[idx] >= 0) {      // for uninserted points
			int idxTri = ptToTri[idx];

			float ptDist = dist(pts[triList[idxTri].insertPt], pts[idx]); 
			if (ptDist == triList[idxTri].insertPt_dist) {
				atomicExch(&(triList[idxTri].insertPt), idx);
			}
		}
	}
}


/* ================================== INSERT ======================================== */

/*
 * Main function for inserting points in triangles in parallel.
 */
void Delaunay::insert() {
	int N = *nTriWithInsert;
	dim3 threadsPerBlock(32);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));

	insertKernel<<<numBlocks, threadsPerBlock>>>(triList_d, *nTri, triWithInsert_d, *nTriWithInsert);

	cudaDeviceSynchronize();
	arrayAddVal<<<1, 1>>>(nTri_d, *nTriWithInsert_d, 1); 

	cudaMemset(triWithInsert_d, -1, (*nTriMax) * sizeof(int));
}

/*
 * Inserts points in parallel into triangles marked for insertion.
 */
__global__ void insertKernel(Tri* triList, int nTri, int* triWithInsert, int nTriWithInsert) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < nTriWithInsert) {
		int triIdx = triWithInsert[idx];
		insertInTri(triIdx, triList, nTri + 2*idx);
	}
}

/*
 * Pick a triangle by index 'i' in triList and insert its center point.
 * Returns the number of a new triangles created. 
 */
__device__ int insertInTri(int i, Tri* triList, int newTriIdx) {
	int r = triList[i].insertPt;

	if (r == -1) { // if no points inside this triangle, continue
		printf("Neighbour doesent exist\n");
	}

	insertPtInTri(r, i, triList, newTriIdx);
	return 0;
}

/*
 * Inserts a point into triangle indexed by 'i' (splits the triangle into 3 creating 
 * two new triangles) if possible. Returns the number of a new triangles created.
 *
 * @param r Index of point in the array pts.
 * @param i Index of triangle in the array triList.
 * @param triList Array of triangles
 * @param newTriIdx Index at which to store the new triangles in array 'triList'.  
 */
__device__ int insertPtInTri(int r, int i, Tri* triList, int newTriIdx) {

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
	int n0[3] = {newTriIdx + 1, n[0], newTriIdx};
	int o0[3] = {1, o[0], 2};

	int p1[3] = {r, p[1], p[2]};
	int n1[3] = {i, n[1], newTriIdx + 1};
	int o1[3] = {1, o[1], 2};

	int p2[3] = {r, p[2], p[0]};
	int n2[3] = {newTriIdx, n[2], i};
	int o2[3] = {1, o[2], 2};

	writeTri(&(triList[newTriIdx    ]), p1, n1, o1);
	writeTri(&(triList[newTriIdx + 1]), p2, n2, o2);
	writeTri(&(triList[i            ]), p0, n0, o0);

	// marking edge for flipping
	triList[newTriIdx    ].flip = 1;
	triList[newTriIdx + 1].flip = 1;
	triList[i            ].flip = 1;

	// ==================== updates neighbour points opposite point if they exist ==================== 
	int nbrnbr[3] = {i, newTriIdx, newTriIdx + 1};

	for (int k=0; k<3; ++k) {
		int mvInsNbr = (o[k] + 1) % 3;

		if (n[k] >= 0) { // if nbr exist
			//if (triList[n[k]].center != -1) { // if nbr was marked for insertion
			if (triList[n[k]].insert == true) { // if nbr was marked for insertion
				int idxNbr_k = n[k];

				// move anticlockwise in the nbr tri which just was inserted into 
				for (int i=0; i<mvInsNbr; ++i) {
					idxNbr_k = triList[idxNbr_k].n[2];
				}

				triList[idxNbr_k].o[1] = 0;
				triList[idxNbr_k].n[1] = nbrnbr[k];

			} // else { 
			if (triList[n[k]].insert == false) { // nbr was not marked for insertion and is the same from prev state
				triList[n[k]].o[mvInsNbr] = 0;
				triList[n[k]].n[mvInsNbr] = nbrnbr[k];
			}
		}
	}

	return 0;
}

/* ===================================== UPDATE POINT LOCATIONS ========================================= */

//void Delaunay::updatePointLocations() {
//
//}
//
//__global__ void updatePointLocationsKernel() {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	
//	if (idx < *npts) {
//		ptToTri[idx] = 
//	}
//
//
///* ===================================== PRINTING ========================================= */
//
//__global__ void printTri(Tri* triList, int nTriMax) {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (idx < nTriMax) {
//		printf("triList[%d].insert = %d", idx, triList[idx].insert); 
//	}	
//}

/* ====================================== ADD VECTOR ============================================= */


/*
 * Adds a value 'val' to each element of an array.
 */
__global__ void arrayAddVal(int* array, int val, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n) {
		array[idx] += val;
	}
}

/* ====================================== ATOMICS ============================================= */

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
		old = atomicCAS(address_as_ull, assumed, __float_as_int(max(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__device__ float atomicMinFloat(float* address, float val) {
	int* address_as_ull = (int*)address;
	int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __float_as_int(min(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

void Delaunay::saveToFile() {

	saveFile << iter << " " << nTri[0] << "\n";
	for (int i=0; i<nTri[0]; ++i) {
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
