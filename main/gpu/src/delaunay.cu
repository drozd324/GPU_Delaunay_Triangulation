#include "delaunay.h"

void Delaunay::compute() {
    // =============== Main compute loop ================ 

    for (int i=0; i<(*npts); ++i) { 
	//for (int i=0; i<3; ++i) {
		std::cout << "\n============[PASS " << i << "]============ \n"; 
		printInfo();

		prepForInsert();
		printf("\nPREP FOR INSERT [%d]\n", i);
		printInfo();

		printf("\nINSERT [%d]\n", i);
		insert();
		printInfo();

		printf("\nUPDATE POINT LOCATIONS [%d]\n", i);
		updatePointLocations();
		printInfo();
		saveToFile();

		cudaMemcpy(nTri, nTri_d, sizeof(int), cudaMemcpyDeviceToHost);
		if (*nTri == *nTriMax) {
			break;
		}
	}
}

/*
 * Constructor which creates the delaunay triagulation from an array of 'points'
 * and its lenght 'n'.
 */
Delaunay::Delaunay(Point* points, int n) 
	//saveFile("./data/data.txt", std::ios_base::app)
{
	// ============= init save file ============  
	file = fopen("./data/data.txt", "w");
	fclose(file);
	file = fopen("./data/data.txt", "a");

    // ============= allocation on host ============  
    // alloc points
	*npts = n;
	pts = (Point*) malloc(((*npts) + 3) * sizeof(Point));

	// alloc triangles
	*nTri = 0;
	*nTriMax = 2*((*npts)+3) - 2 - 3;
	triList = (Tri*) malloc((*nTriMax) * sizeof(Tri));

	// maps
	ptToTri = (int*) malloc((*npts) * sizeof(int));
	triWithInsert = (int*) malloc((*nTriMax) * sizeof(int));
	memset(triWithInsert, -1, (*nTriMax) * sizeof(int));

	for (int i=0; i<(*npts); i++) {
		pts[i] = points[i];
	}

	// setting default values
	*nTriWithInsert = 0;
	float biggest_float = (float)(unsigned long)-1;
	for (int i=0; i<(*nTriMax); ++i) {
		triList[i].flip = -1;
		triList[i].insert = false;
		triList[i].insertPt = -1;
		triList[i].insertPt_dist = biggest_float;

		for (int j=0; j<3; ++j) { triList[i].p[j] = -1; }
		for (int j=0; j<3; ++j) { triList[i].n[j] = -1; }
		for (int j=0; j<3; ++j) { triList[i].o[j] = -1; }
	}

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
	cudaMemcpy(pts_d           , pts            , (*npts)    * sizeof(Point), cudaMemcpyHostToDevice);
	cudaMemcpy(npts_d          , npts           ,              sizeof(int)  , cudaMemcpyHostToDevice);

	cudaMemcpy(triList_d       , triList        , (*nTriMax) * sizeof(Tri)  , cudaMemcpyHostToDevice);
	cudaMemcpy(nTriMax_d       , nTriMax        ,              sizeof(int)  , cudaMemcpyHostToDevice);
	cudaMemcpy(triWithInsert_d , triWithInsert  , (*nTriMax) * sizeof(int)  , cudaMemcpyHostToDevice);
	cudaMemcpy(nTriWithInsert_d, nTriWithInsert ,              sizeof(int)  , cudaMemcpyHostToDevice);

	// ============= initialize super triangle ============ 
	initSuperTri();
	cudaDeviceSynchronize();

    // save points data to file
	fprintf(file, "%d\n", (*npts) + 3);
	for (int i=0; i<(*npts) + 3; ++i) {
		fprintf(file, "%f %f\n", pts[i].x[0], pts[i].x[1]);
	}
	fprintf(file, "\n");

	saveToFile();

	compute();

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

Delaunay::~Delaunay() {
	fclose(file);

	cudaFree(pts_d);
	cudaFree(npts_d);

	cudaFree(triList_d);
	cudaFree(nTri_d);
	cudaFree(nTriMax_d);

	cudaFree(ptToTri_d);
	cudaFree(triWithInsert_d);
	cudaFree(nTriWithInsert_d);
	
	free(triList); 
	free(pts);
	free(ptToTri);
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

//	printf("avgPoint before: (%f, %f)\n", avgPoint->x[0], avgPoint->x[1]);

	//dim3 threadsPerBlock(warpSize);
	int valsToAdd = 2*(*npts); 
	dim3 threadsPerBlock1(32);
	dim3 numBlocks1(valsToAdd/threadsPerBlock1.x + (!(valsToAdd % threadsPerBlock1.x) ? 0:1));

	sumPoints<<<numBlocks1, threadsPerBlock1>>>(pts_d, npts_d, avgPoint_d);

	cudaMemcpy(avgPoint, avgPoint_d, sizeof(Point), cudaMemcpyDeviceToHost);

	avgPoint->x[0] /= npts[0];
	avgPoint->x[1] /= npts[0];

//	printf("avgPoint After: (%f, %f)\n", avgPoint->x[0], avgPoint->x[1]);

	// computing the largest distance bewtween two points
	float largest_dist[1] = {0};
	float *largest_dist_d;
	cudaMalloc(&largest_dist_d, sizeof(float));
	cudaMemcpy(largest_dist_d, largest_dist, sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemset(largest_dist_d, 0, sizeof(float));

//	printf("before largest_dist: %f\n", *largest_dist);

	int ncomps = ((*npts)*((*npts) - 1)) / 2; //number of comparisons
	dim3 threadsPerBlock2(32);
	dim3 numBlocks2(ncomps/threadsPerBlock2.x + (!(ncomps % threadsPerBlock2.x) ? 0:1));
	computeMaxDistPts<<<numBlocks2, threadsPerBlock2>>>(pts_d, npts_d, largest_dist_d);

	cudaMemcpy(largest_dist, largest_dist_d, sizeof(float), cudaMemcpyDeviceToHost);

//	printf("after largest_dist: %f\n", *largest_dist);

	// writing supertriangle points to pts
	float center_x = avgPoint->x[0];
	float center_y = avgPoint->x[1];
	float radius = *largest_dist;

	pts[(*npts)    ].x[0] = center_x + radius*1.73205;
	pts[(*npts)    ].x[1] = center_y - radius; 

	pts[(*npts) + 1].x[0] = center_x;
	pts[(*npts) + 1].x[1] = center_y + 2*radius;

	pts[(*npts) + 2].x[0] = center_x - radius*1.73205;
	pts[(*npts) + 2].x[1] = center_y - radius;

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
	triList[0].insert = true;

	//memset(ptToTri, 0, (*npts) * sizeof(int));
	cudaMemset(ptToTri_d, 0, (*npts) * sizeof(int));
	cudaMemcpy(&(triList_d[0]), &(triList[0]), sizeof(Tri), cudaMemcpyHostToDevice);

	cudaMemset(triWithInsert_d, 0, sizeof(int)); // set fist element of this array to be 0, ie the super triangle just constructed 

	(*nTri)++;
	cudaMemcpy(nTri_d, nTri, sizeof(int), cudaMemcpyHostToDevice);
}

/*
 * Computes a vector sum of points provided in pts and stores them in avgPoint.
 */
__global__ void sumPoints(Point* pts, int* npts, Point *avgPoint) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < 2*(*npts)) {
		atomicAddFloat(&(avgPoint->x[idx%2]), pts[idx/2].x[idx%2]);
	}
}

/*
 * Computes the maximum distance bewteen two points in the array pts and stores this value in largest_dist.
 */
__global__ void computeMaxDistPts(Point* pts, int* npts, float* largest_dist) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;
	float dist; 

	int count = ((*npts)*((*npts) - 1)) / 2; 
	if (idx < count) {
		// for now yoinnked from chet gipiti but uses triangilar number math to get convenient indexing
		i = (int)((2*(*npts) - 1 - sqrtf((2*(*npts) - 1) * (2*(*npts) - 1) - 8*idx)) / 2);
		j = idx - (i*(2*(*npts) - i - 1) / 2) + i + 1;

//		printf("index: %d | count: %d | i,j =(%d, %d)\n", idx, count, i, j);

		dist = sqrtf( (pts[i].x[0] - pts[j].x[0])*(pts[i].x[0] - pts[j].x[0]) +
				      (pts[i].x[1] - pts[j].x[1])*(pts[i].x[1] - pts[j].x[1]));

		if (dist > (*largest_dist)) {
			atomicMaxFloat(largest_dist, dist); 
		}
	}
}

__host__ __device__ void writeTri(Tri* tri, int* p, int* n, int* o) {
	for (int i=0; i<3; i++) {
//		printf("tri->p[%d] = p[%d]: %d\n", i, i, tri->p[i] = p[i]);
//		printf("tri->n[%d] = n[%d]: %d\n", i, i, tri->n[i] = n[i]);
//		printf("tri->o[%d] = o[%d]: %d\n", i, i, tri->o[i] = o[i]);

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
	int N;

	N = *nTriMax;
	dim3 threadsPerBlock0(32);
	dim3 numBlocks0(N/threadsPerBlock0.x + (!(N % threadsPerBlock0.x) ? 0:1));

	resetInsertPtInTris<<<numBlocks0, threadsPerBlock0>>>(triList_d, nTriMax_d);

	N = *npts;
	printf("%d\n", N);
	dim3 threadsPerBlock1(32);
	dim3 numBlocks1(N/threadsPerBlock1.x + (!(N%threadsPerBlock1.x) ? 0:1));

	setInsertPtsDistance<<<numBlocks1, threadsPerBlock1>>>(pts_d, npts_d, triList_d, ptToTri_d);
	setInsertPts<<<numBlocks1, threadsPerBlock1>>>(pts_d, npts_d, triList_d, ptToTri_d);

	N = *nTri;
	printf("%d\n", N);
	dim3 threadsPerBlock2(32);
	dim3 numBlocks2(N/threadsPerBlock2.x + (!(N%threadsPerBlock2.x) ? 0:1));

	cudaMemset(nTriWithInsert_d, 0, sizeof(int));
	prepTriWithInsert<<<numBlocks2, threadsPerBlock2>>>(triList_d, nTri_d, triWithInsert_d, nTriWithInsert_d);

	cudaMemcpy(triWithInsert, triWithInsert_d, (*nTriMax) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(nTriWithInsert, nTriWithInsert_d,  sizeof(int), cudaMemcpyDeviceToHost);
	printf("nTriWithInsert: %d\n", *nTriWithInsert);
	printf("BEFORE SORT triWithInsert: ");
	for (int i=0; i<(*nTri); ++i) {
		printf("%d ", triWithInsert[i]);
	}
	printf("\n");

	gpuSort(triWithInsert_d, nTri_d); 

	cudaDeviceSynchronize();

	cudaMemcpy(triWithInsert, triWithInsert_d, (*nTriMax) * sizeof(int), cudaMemcpyDeviceToHost);
	printf("AFTER SORT triWithInsert: ");
	for (int i=0; i<(*nTri); ++i) {
		printf("%d ", triWithInsert[i]);
	}
	printf("\n");

	cudaDeviceSynchronize();
//	checkInsertPoint<<<numBlocks2, threadsPerBlock2>>>(triList, triWithInsert, nTriWithInsert);
//	gpuSort(triWithInsert_d, nTriWithInsert_d); 

	resetBiggestDistInTris<<<numBlocks2, threadsPerBlock2>>>(triList_d, nTriMax_d);
}

__global__ void resetInsertPtInTris(Tri* triList, int* nTriMax) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTriMax)) {
		triList[idx].insertPt = -1;
	}
}

/*
 * Writes the shortest distance between a point in each triangle and the center
 * of its circumcircle to triangles slot named 'insertPt_dist'.
 */
__global__ void setInsertPtsDistance(Point* pts, int* npts, Tri* triList, int* ptToTri) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*npts)) {
		if (ptToTri[idx] >= 0) {      // for uninserted points
			int idxTri = ptToTri[idx];

			// find circumcenter for the trianle this point is inside of
			Point  circumcenter;
			circumcircle_center(pts[triList[idxTri].p[0]], pts[triList[idxTri].p[1]], pts[triList[idxTri].p[2]], &circumcenter);

			float ptDist = dist(circumcenter , pts[idx]); 
			atomicMinFloat(&(triList[idxTri].insertPt_dist), ptDist);
		}
	}
}

/*
 * Writes the index of the point int the array 'pts' to be inserted into each triangle
 * in the triangles slot named 'insertPt'.
 */
__global__ void setInsertPts(Point* pts, int* npts, Tri* triList, int* ptToTri) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*npts)) {
		if (ptToTri[idx] >= 0) {      // for uninserted points
			int idxTri = ptToTri[idx];

			Point circumcenter;
			circumcircle_center(pts[triList[idxTri].p[0]], pts[triList[idxTri].p[1]], pts[triList[idxTri].p[2]], &circumcenter);

			float ptDist = dist(circumcenter, pts[idx]); 
			if (ptDist == triList[idxTri].insertPt_dist) {
				//printf("INSERT POINT: %d\n", triList[idxTri].insertPt);
				atomicExch(&(triList[idxTri].insertPt), idx);
			}
		}
	}
}

__global__ void prepTriWithInsert(Tri* triList, int* nTri, int* triWithInsert, int* nTriWithInsert) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTri)) {
		// "append" index of triange to 'triWithInsert' if insert=true
		int insert_stat = triList[idx].insert && (triList[idx].insertPt != -1);

		printf("TRI IDX=%d INSERT STAT: %d\n", idx, insert_stat);

		// write -1 if insert is false and write the index number if insert is true
		triWithInsert[idx] = idx*(insert_stat == 1) + (-(insert_stat == 0));
		atomicAdd(nTriWithInsert, insert_stat);
		//atomicAdd(nTriWithInsert, (triList[idx].insert == true) + 0*(triWithInsert[idx] = idx) );
	}
}

__global__ void checkInsertPoint(Tri* triList, int* triWithInsert, int* nTriWithInsert) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int N = *nTriWithInsert;

	if (idx < N) {
		if (triList[nTriWithInsert[idx]].insertPt == -1) {
			triList[nTriWithInsert[idx]].insert = false;
			nTriWithInsert[idx] = -1;
			atomicAdd(nTriWithInsert, -1);
		}
	}
}

__global__ void resetBiggestDistInTris(Tri* triList, int* nTriMax) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTriMax)) {
		triList[idx].insertPt_dist = (float)(unsigned long)-1;
	}
}

/* ================================== INSERT ======================================== */

/*
 * Main function for inserting points in triangles in parallel.
 */
void Delaunay::insert() {

	cudaMemcpy(nTriWithInsert, nTriWithInsert_d, sizeof(int), cudaMemcpyDeviceToHost);
	int N = *nTriWithInsert;
	dim3 threadsPerBlock(32);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));

//	std::cout << "N: " << N << "\n";
//	std::cout << "threadsPerBlock.x: " << threadsPerBlock.x << "\n";
//	std::cout << "numBlocks.x: " << numBlocks.x << "\n";

	insertKernel<<<numBlocks, threadsPerBlock>>>(triList_d, nTri_d, triWithInsert_d, nTriWithInsert_d, ptToTri_d);

	cudaDeviceSynchronize();

	// update number of triangles in triList
	arrayAddVal<<<1, 1>>>(nTri_d, nTriWithInsert_d, 2, 1); 
	
	// reset triWithInsert_d for next iteraiton
	cudaMemset(triWithInsert_d, -1, (*nTri) * sizeof(int));
}

/*
 * Inserts points in parallel into triangles marked for insertion.
 */
__global__ void insertKernel(Tri* triList, int* nTri, int* triWithInsert, int* nTriWithInsert, int* ptToTri) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTriWithInsert)) {
		int triIdx = triWithInsert[idx];
		insertInTri(triIdx, triList, (*nTri) + 2*idx, ptToTri);
	}
}

/*
 * Pick a triangle by index 'i' in triList and insert its center point.
 * Returns the number of a new triangles created. 
 */
__device__ int insertInTri(int i, Tri* triList, int newTriIdx, int* ptToTri) {
	int r = triList[i].insertPt;
	ptToTri[r] = -1;

	if (r == -1) { // if no points inside this triangle, continue
		printf("POINT DOESENT EXIST IN INSERTINTRI\n");
		return -1;
	}

	insertPtInTri(r, i, triList, newTriIdx, ptToTri);
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
__device__ int insertPtInTri(int r, int i, Tri* triList, int newTriIdx, int* ptToTri) {

//	printf("====================================INSERT========================================\n");
//	printf("i: %d\n", i);

	printf("newTriIdx: %d and %d\n", newTriIdx, newTriIdx+1);
	
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

	writeTri(&(triList[i            ]), p0, n0, o0);
	writeTri(&(triList[newTriIdx    ]), p1, n1, o1);
	writeTri(&(triList[newTriIdx + 1]), p2, n2, o2);

	// marking edge for flipping
	triList[i            ].flip = 1;
	triList[newTriIdx    ].flip = 1;
	triList[newTriIdx + 1].flip = 1;

	triList[i            ].insert = true;
	triList[newTriIdx    ].insert = true;
	triList[newTriIdx + 1].insert = true;

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

void Delaunay::updatePointLocations() {
	int N;

	N = *npts;
	dim3 threadsPerBlock2(32);
	dim3 numBlocks2(N/threadsPerBlock2.x + (!(N % threadsPerBlock2.x) ? 0:1));
	updatePointLocationsKernel<<<numBlocks2, threadsPerBlock2>>>(pts_d, npts_d, triList_d, nTri_d, ptToTri_d);
}


/*
 * For each point, in the array ptToTri, index of the triangle in which the point is contained in is
 * written in the corresponding index.
 * 
 * @param pts Array of Points in this triangulation.
 * @param npts Number of points in the triangulation.
 * @param triList Array of Tri which contain information about the triangles in the triangulation.  
 * @param nTri Number of triangles in use by triList.
 * @param ptToTri Array which maps each point index to a triangle index.
 */
__global__ void updatePointLocationsKernel(Point* pts, int* npts, Tri* triList, int* nTri, int* ptToTri) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*npts)) {
		if (ptToTri[idx] >= 0) {
			for (int t=0; t<(*nTri); ++t) {
				if (contains(t, idx, triList, pts) == 1) {
					ptToTri[idx] = t;
				}
			}
		}
	}
}

/*
 * Checks if a triangle with index 't' contains point with index 'r'. Returns 1 if the
 * point is inside or if on the boundary and -1 if its on the outside.
 */
__device__ int contains(int t, int r, Tri* triList, Point* pts) {

	float area;
	int i, j;
	int bit = 0;

	for (i=0; i<3; ++i) {
		j = (i+1) % 3;
		// area = area of triangle (21.3.2) (21.3.10) 
		area = (pts[triList[t].p[j]].x[0] - pts[triList[t].p[i]].x[0])*(pts[r].x[1] - pts[triList[t].p[i]].x[1]) - 
			   (pts[triList[t].p[j]].x[1] - pts[triList[t].p[i]].x[1])*(pts[r].x[0] - pts[triList[t].p[i]].x[0]);

		bit = bit | (area < 0.0);
	}

	return (-(bit == 1)) + (bit == 0);

}


/* ===================================== PRINTING ========================================= */

void Delaunay::printInfo() {
	//cudaMemcpy(pts, pts_d, sizeof(Tri), cudaMemcpyDeviceToHost);
	cudaMemcpy(triList, triList_d, (*nTriMax) * sizeof(Tri), cudaMemcpyDeviceToHost);
	cudaMemcpy(nTri   , nTri_d   ,              sizeof(int), cudaMemcpyDeviceToHost);

	for (int i=0; i<(*nTriMax); ++i) { printf("triList[%d].insert: %d\n"       , i, triList[i].insert); }
	for (int i=0; i<(*nTriMax); ++i) { printf("triList[%d].insertPt: %d\n"     , i, triList[i].insertPt); }
	for (int i=0; i<(*nTriMax); ++i) { printf("triList[%d].insertPt_dist: %f\n", i, triList[i].insertPt_dist); }

	cudaMemcpy(ptToTri       , ptToTri_d       , (*npts)    * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(triWithInsert , triWithInsert_d , (*nTriMax) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(nTriWithInsert, nTriWithInsert_d,              sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i=0; i<(*npts   ); ++i) { printf("ptToTri[%d]: %d\n"      , i, ptToTri[i]); }
	for (int i=0; i<(*nTriMax); ++i) { printf("triWithInsert[%d]: %d\n", i, triWithInsert[i]); }
	for (int i=0; i<1         ; ++i) { printf("nTriWithInsert[%d]: %d\n", i, nTriWithInsert[i]); }
}

void Delaunay::printTri() {
	cudaMemcpy(triList, triList_d, (*nTriMax) * sizeof(Tri), cudaMemcpyDeviceToHost);
	cudaMemcpy(nTri   , nTri_d   ,              sizeof(int), cudaMemcpyDeviceToHost);

	for (int i=0; i<(*nTriMax); ++i) {
		printf("triList[%d].p[0]: %d\n"  , i, triList[i].p[0]);
		printf("triList[%d].p[1]: %d\n"  , i, triList[i].p[1]);
		printf("triList[%d].p[2]: %d\n"  , i, triList[i].p[2]);

		printf("triList[%d].n[0]: %d\n"  , i, triList[i].n[0]);
		printf("triList[%d].n[1]: %d\n"  , i, triList[i].n[1]);
		printf("triList[%d].n[2]: %d\n"  , i, triList[i].n[2]);

		printf("triList[%d].o[0]: %d\n"  , i, triList[i].o[0]);
		printf("triList[%d].o[1]: %d\n"  , i, triList[i].o[1]);
		printf("triList[%d].o[2]: %d\n\n", i, triList[i].o[2]);
	}
}


/* ====================================== MISC ============================================= */


/*
 * Adds a value 'val' to each element of an array.
 */
__global__ void arrayAddVal(int* array, int* val, int mult, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n) {
		array[idx] += mult * (*val);
	}
}



/*
 * @param array Device pointer to array.
 * @param n Device pointer to length of array.
 */
void gpuSort(int* array, int* n) {
    int N;
    cudaMemcpy(&N, n, sizeof(int), cudaMemcpyDeviceToHost);

	if (N <= 1) {
		return;
	}

    int sorted[1];
    int* sorted_d;
    cudaMalloc(&sorted_d, sizeof(int));

    dim3 threadsPerBlock(32);
    dim3 numBlocks((N / 2 + threadsPerBlock.x - 1) / threadsPerBlock.x);

    //int parity = 0;
    do {
        *sorted = 0;
        cudaMemcpy(sorted_d, sorted, sizeof(int), cudaMemcpyHostToDevice);
        sortPass<<<numBlocks, threadsPerBlock>>>(array, N, 0, sorted_d);
        sortPass<<<numBlocks, threadsPerBlock>>>(array, N, 1, sorted_d);
        cudaMemcpy(sorted, sorted_d, sizeof(int), cudaMemcpyDeviceToHost);
        //parity = 1 - parity;  // alternate parity 0/1
    } while (*sorted != 0); // continue while swaps occurred

    cudaFree(sorted_d);
}

__global__ void sortPass(int* array, int n, int parity, int* sorted) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * idx + parity;
	int minval, maxval;

    if (i + 1 < n) {
        int l = array[i];
        int r = array[i + 1];

		maxval = max(l, r);
		minval = min(l, r);

		array[i] = maxval;
		array[i + 1] = minval;

		atomicOr(sorted, !(l==maxval));  // Mark that a swap happened
    }
}

/* ====================================== SAVE TO FILE ============================================= */

void Delaunay::saveToFile(bool end) {
	cudaMemcpy(nTri   , nTri_d   ,              sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(triList, triList_d, (*nTriMax) * sizeof(Tri), cudaMemcpyDeviceToHost);

	if (end == false) { // save all triangles
		fprintf(file, "%d %d\n", iter, *nTri);
		for (int i=0; i<(*nTri); ++i) {
			for (int j=0; j<3; ++j) { fprintf(file, "%d ", triList[i].p[j]); } 
			for (int j=0; j<3; ++j) { fprintf(file, "%d ", triList[i].n[j]); } 
			for (int j=0; j<3; ++j) { fprintf(file, "%d ", triList[i].o[j]); } 
			fprintf(file, "%d ", triList[i].flip);
			fprintf(file, "%d ", triList[i].insert);

			fprintf(file, "\n");
		}

		fprintf(file, "\n");
		iter++;
	} 
	else {
		int nTriFinal = 0;	
		// count number of triangles which do not contain the supertriangle points
		for (int i=0; i<(*nTri); ++i) {
			int cont = 0;
			for (int k=0; k<3; ++k) {
				for (int l=0; l<3; ++l) {
					if (triList[i].p[k] == ((*npts) + l)) {
						cont = -1;
					}
				}
			}
			if (cont == -1) { continue; }
	
			nTriFinal++;
		}
	
		fprintf(file, "%d %d\n", iter, nTriFinal);
		//saveFile << iter << " " << nTriFinal << "\n";
		for (int i=0; i<(*nTri); ++i) {
			// if any point in this triangle is on the boundary dont save
			int cont = 0;
			for (int k=0; k<3; ++k) {
				for (int l=0; l<3; ++l) {
					if (triList[i].p[k] == ((*npts) + l)) {
						cont = -1;
					}
				}
			}
	
			if (cont == -1) {
				continue;
			}
	
			for (int j=0; j<3; ++j) { fprintf(file, "%d ", triList[i].p[j]); } 
			for (int j=0; j<3; ++j) { fprintf(file, "%d ", triList[i].n[j]); } 
			for (int j=0; j<3; ++j) { fprintf(file, "%d ", triList[i].o[j]); } 
			fprintf(file, "%d ", triList[i].flip);
			fprintf(file, "%d ", triList[i].insert);

			fprintf(file, "\n");
		}

		fprintf(file, "\n");
		iter++;
	}
}
