#include "delaunay.h"

int min_ntpb = 32;

/*
 * Main compute function which strings together each parallel operation
 * involved in the parallel point insertion.
 */
void Delaunay::compute() {
	float prepForInsertTimeTot = 0;
	float insertTimeTot = 0;
	float flipTimeTot = 0;
	float updatePtsTimeTot = 0;
	int numConfigsFlippedTot = 0;

	float prepForInsertTime;
	float insertTime;
	float flipTime;
	float updatePtsTime;
	int numConfigsFlipped;

	float totalGPUTime;
	float totalCPUTime;
	float totalRuntime;
	clock_t t0 = clock();

	for (int i=0; i<(*npts); ++i) { 
		if (info    == true) {
			printf("============== [%d] PASS ----------------- ==============\n", i); 
			cudaMemcpy(nTri, nTri_d, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(nptsInserted, nptsInserted_d, sizeof(int), cudaMemcpyDeviceToHost);
			printf("    No. of triangles in triangulation: %d/%d\n", *nTri, *nTriMax);
			printf("    No. of points currently inserted: %d/%d\n", *nptsInserted, *npts);
		}

		cudaMemcpy(nTri, nTri_d, sizeof(int), cudaMemcpyDeviceToHost);
		if (*nTri == *nTriMax) {
			break;
		}

		if (info    == true) {
			printf("============== [%d] PREP FOR INSERT ------ ==============\n", i);
			cudaMemcpy(nTriWithInsert, nTriWithInsert_d, sizeof(int), cudaMemcpyDeviceToHost);
			printf("    No. of triangles with points to insert: %d\n", *nTriWithInsert);
			printf("    time: %f\n", prepForInsertTime);
		}
		prepForInsertTime = timeGPU([this] () { prepForInsert(); });

		if (info == true) {
			printf("============== [%d] INSERT --------------- ==============\n", i);
			printf("    time: %f\n", insertTime);
		}
		insertTime = timeGPU([this] () { insert(); });

		#ifndef NOFLIP
		if (saveHistory == true) { saveToFile(); }

		if (info == true) {
			printf("============== [%d] FLIP ----------------- ==============\n", i);
		}
		flipTime = timeGPU([this, &numConfigsFlipped] () { numConfigsFlipped = flip(); });
		numConfigsFlippedTot += numConfigsFlipped;
		if (info == true) {
			printf("    No. of configurations flipped: %d\n", numConfigsFlipped);
			printf("    time: %f\n", flipTime);
		}
		#endif

		updatePtsTime = timeGPU([this] () { updatePointLocations(); });
		if (info == true) {
			printf("============== [%d] UPDATE POINT LOCATIONS ==============\n", i);
			printf("    time: %f\n", updatePtsTime);
		}

		if (saveHistory == true) { saveToFile(); }

		prepForInsertTimeTot += prepForInsertTime;
		insertTimeTot        += insertTime;
		flipTimeTot          += flipTime;
		updatePtsTimeTot     += updatePtsTime;
	}


	clock_t t1 = clock() - t0;
	totalRuntime = (float)t1 / CLOCKS_PER_SEC;

	if (delaunayCheck() > 0) {
		if (info == true) { printf("Attempting to perform additional flips\n"); }
	}

	if (info == true) {

		printf("All time measured in seconds\n");
		printf("prepForInsertTimeTot__________: %f\n", prepForInsertTimeTot);
		printf("insertTimeTot_________________: %f\n", insertTimeTot);
		printf("flipTimeTot___________________: %f\n", flipTimeTot);
		printf("updatePtsTimeTot______________: %f\n", updatePtsTimeTot);

		printf("\n");

		totalGPUTime = prepForInsertTimeTot + insertTimeTot + flipTimeTot + updatePtsTimeTot;
		totalCPUTime = totalRuntime - totalGPUTime; 
		printf("Total run time of cuda kernels: %f\n", totalGPUTime);
//		printf("Total run time of host code   : %f\n", totalCPUTime);
		printf("Total run time of all code    : %f\n", totalRuntime);

	}

	if (saveCSV == true) {
		fprintf(csvfile, "ntpb,npts,nTriMax,totalRuntime,totalCPUTime,totalGPUTime,prepForInsertTimeTot,insertTimeTot,flipTimeTot,updatePtsTimeTot,seed,distribution,deviceName\n");
		fprintf(csvfile, "%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%d,%d,%s\n",
		  ntpb, (*npts), (*nTriMax), totalRuntime, totalCPUTime, totalGPUTime, prepForInsertTimeTot, insertTimeTot, flipTimeTot, updatePtsTimeTot, seed, distribution, device.name);
	}
}

/* constructors */
Delaunay::Delaunay(Point* points, int n) {
	constructor(points, n);	
}

Delaunay::Delaunay(Point* points, int n, int numThreadsPerBlock) {
	ntpb = numThreadsPerBlock;

	constructor(points, n);	
}

Delaunay::Delaunay(Point* points, int n, int numThreadsPerBlock, int seed_mark, int distribution_mark) {
	ntpb = numThreadsPerBlock;
	seed = seed_mark;
	distribution = distribution_mark;

	constructor(points, n);	
}

/*
 * Constructor which creates the delaunay triagulation from an array of 'points'
 * and its lenght 'n'.
 * 
 * @param points Array of point points involved in the triangulation
 * @param n Number of points in array
 */
void Delaunay::constructor(Point* points, int n) {

	// ============= DEVICE INFO ==================

	//struct cudaDeviceProp device;
	int numDevices = 0;
	if (cudaGetDeviceCount(&numDevices) != cudaSuccess ) {
		printf("No CUDA-enabled devices were found\n"); 
		return;
	}

	if (info == true) {
		printf("[DEVICE INFO]\n");
		printf("Found %d CUDA-enabled devices\n", numDevices);

		cudaGetDeviceProperties(&device, 0);
		printf("GPU compute capability: %d.%d\n", device.major, device.minor);
		printf("GPU model name: %s\n"           , device.name);
		printf("\n");
	}

	// ============= INITIALIZE FILES TO SAVE DATA TO ============  

	trifile = fopen("./data/tri.txt", "w");
	fclose(trifile);
	trifile = fopen("./data/tri.txt", "a");

	csvfile = fopen("./data/coredata.csv", "w");

	FILE* iPIfile = fopen("./data/insertedPerIter.txt", "w");
	insertedPerIterfile = iPIfile;
	fclose(insertedPerIterfile);
	insertedPerIterfile  = fopen("./data/insertedPerIter.txt", "a");

	flipedPerIterfile = fopen("./data/flipedPerIter.txt", "w");
	fclose(flipedPerIterfile);
	flipedPerIterfile = fopen("./data/flipedPerIter.txt", "a");

    // ============= ALLOCATION ON HOST ============  
    // alloc points
	long long int totMemAlloc = 0;
	long long int totMemAlloc_onDev = 0;

	*npts = n;
	*nptsInserted = 0;
	totMemAlloc += ((*npts) + 3) * sizeof(Point);
	pts = (Point*) malloc(((*npts) + 3) * sizeof(Point));

	// alloc triangles
	*nTri = 0;
	*nTriMax = 2*((*npts)+3) - 2 - 3;
	*nEdgesMax = 3*((*npts)+3) - 3 - 3;
	triList = (Tri*) malloc((*nTriMax) * sizeof(Tri));
	totMemAlloc += (*nTriMax) * sizeof(Tri);

	// maps
	ptToTri = (int*) malloc((*npts) * sizeof(int));
	triWithInsert = (int*) malloc((*nTriMax) * sizeof(int));
	memset(triWithInsert, -1, (*nTriMax) * sizeof(int));
	totMemAlloc += (*npts) * sizeof(int);
	totMemAlloc += (*nTriMax) * sizeof(int);

	triToFlip = (int*) malloc((*nTriMax) * sizeof(int)); 
	totMemAlloc += (*nTriMax) * sizeof(int);

	ptsUninserted  = (int*) malloc((*npts) * sizeof(int));
	*nptsUninserted = 0;
	memset(ptsUninserted, -1, (*npts) * sizeof(int));

	for (int i=0; i<(*npts); i++) {
		pts[i] = points[i];
	}

	// setting default values
	*nTriWithInsert = 0;
	for (int i=0; i<(*nTriMax); ++i) {
		triList[i].flip = -1;
		triList[i].insert = 1;
		triList[i].insertPt = -1;
		triList[i].insertPt_dist = (REAL)(unsigned long)-1;
		triList[i].flipThisIter = -1;

		for (int j=0; j<3; ++j) { triList[i].p[j] = -1; }
		for (int j=0; j<3; ++j) { triList[i].n[j] = -1; }
		for (int j=0; j<3; ++j) { triList[i].o[j] = -1; }
	}

	// ============= ALLOC ON DEVICE ============ 
	// alloc points
	totMemAlloc_onDev += (*nTriMax) * sizeof(int);
	cudaMalloc(&npts_d,               sizeof(int));
	cudaMalloc(&pts_d , ((*npts)+3) * sizeof(Point));
	cudaMalloc(&nptsInserted_d,       sizeof(int));
	cudaMemcpy(nptsInserted_d, nptsInserted, sizeof(int), cudaMemcpyHostToDevice);

	// alloc triangles
	totMemAlloc_onDev += (*nTriMax) * sizeof(Tri);
	cudaMalloc(&nTri_d   , sizeof(int));
	cudaMalloc(&nTriMax_d, sizeof(int));
	cudaMalloc(&triList_d, (*nTriMax) * sizeof(Tri));
	cudaMalloc(&triList_prev_d, (*nTriMax) * sizeof(Tri));
	
	// maps
	totMemAlloc_onDev += (*npts) * sizeof(Point);
	totMemAlloc_onDev += (*nTriMax) * sizeof(int);
	cudaMalloc(&ptToTri_d      , (*npts) * sizeof(Point));
	cudaMalloc(&triWithInsert_d, (*nTriMax) * sizeof(int));
	cudaMalloc(&ptsUninserted_d, (*npts) * sizeof(int));
	cudaMalloc(&nptsUninserted_d, sizeof(int));

	cudaMalloc(&nTriToFlip_d, sizeof(int));
	cudaMalloc(&triToFlip_d, (*nTriMax) * sizeof(int));

	cudaMalloc(&quadList_d, (((*nTriMax)/2) + 1) * sizeof(Quad));

	// counters
	cudaMalloc(&nTriWithInsert_d, sizeof(int));
	cudaMalloc(&subtract_nTriToFlip_d, sizeof(int));

	// copying exitsting info to gpu
	cudaMemcpy(pts_d           , pts            , (*npts)    * sizeof(Point), cudaMemcpyHostToDevice);
	cudaMemcpy(npts_d          , npts           ,              sizeof(int)  , cudaMemcpyHostToDevice);

	cudaMemcpy(triList_d       , triList        , (*nTriMax) * sizeof(Tri)  , cudaMemcpyHostToDevice);
	cudaMemcpy(nTriMax_d       , nTriMax        ,              sizeof(int)  , cudaMemcpyHostToDevice);
	cudaMemcpy(triWithInsert_d , triWithInsert  , (*nTriMax) * sizeof(int)  , cudaMemcpyHostToDevice);
	cudaMemcpy(nTriWithInsert_d, nTriWithInsert ,              sizeof(int)  , cudaMemcpyHostToDevice);

	cudaMemcpy(ptsUninserted_d , ptsUninserted  , (*npts) * sizeof(int)     , cudaMemcpyHostToDevice);
	cudaMemcpy(nptsUninserted_d, nptsUninserted ,           sizeof(int)     , cudaMemcpyHostToDevice);

	//printf("Total global memory used: %f GB \n", ((REAL)totMemAlloc_onDev) * 1e-9 );

	initSuperTri();

	fprintf(trifile, "%d\n", (*npts) + 3);
	for (int i=0; i<(*npts) + 3; ++i) {
		fprintf(trifile, "%f %f\n", pts[i].x[0], pts[i].x[1]);
	}
	fprintf(trifile, "\n");

	if (saveHistory == true) { saveToFile(); }

	compute();

	saveToFile(true);
}

Delaunay::~Delaunay() {

	fclose(trifile);
	fclose(csvfile);
	fclose(insertedPerIterfile);
	fclose(flipedPerIterfile); 

	cudaFree(pts_d);
	cudaFree(npts_d);
	cudaFree(nptsInserted_d);

	cudaFree(triList_d);
	cudaFree(triList_prev_d);
	cudaFree(nTri_d);
	cudaFree(nTriMax_d);

	cudaFree(ptToTri_d);
	cudaFree(triWithInsert_d);
	cudaFree(nTriWithInsert_d);
	cudaFree(ptsUninserted_d);
	cudaFree(nptsUninserted_d);

	cudaFree(nTriToFlip_d);
	cudaFree(triToFlip_d);
	
	free(triList); 
	free(pts);
	free(ptToTri);
	free(triToFlip);

	cudaFree(quadList_d);
	cudaFree(subtract_nTriToFlip_d);
}

/*
 * Function to initialize the super triangle containing all points which will be involved in the
 * final triangulation.
 */
void Delaunay::initSuperTri() {

	Point avgPoint[1];
	Point* avgPoint_d;

	// computing the average point
	avgPoint->x[0] = 0;
	avgPoint->x[1] = 0;
	cudaMalloc(&avgPoint_d, sizeof(Point));
	cudaMemcpy(avgPoint_d, avgPoint, sizeof(Point), cudaMemcpyHostToDevice);

	CalcAvgPoint(*avgPoint, pts_d, npts);

	// computing the largest distance bewtween two points
	REAL largest_dist[1] = {0};
	REAL *largest_dist_d;
	cudaMalloc(&largest_dist_d, sizeof(REAL));
	cudaMemcpy(largest_dist_d, largest_dist, sizeof(REAL), cudaMemcpyHostToDevice);
	//cudaMemset(largest_dist_d, 0, sizeof(REAL));

	// computing the max distance, need for loop here as the number of threads it spaws is insane with large n
	//int ncomps = (*npts) * ((*npts) - 1) / 2; // total comparisons

	dim3 threadsPerBlock2(ntpb);
	dim3 numBlocks2((*npts)/threadsPerBlock2.x + (!((*npts) % threadsPerBlock2.x) ? 0:1));
	computeMaxDistPts<<<numBlocks2, threadsPerBlock2>>>(pts_d, npts_d, largest_dist_d);

	cudaMemcpy(largest_dist, largest_dist_d, sizeof(REAL), cudaMemcpyDeviceToHost);
	//*largest_dist = 2;

	// writing supertriangle points to pts
	REAL center_x = avgPoint->x[0];
	REAL center_y = avgPoint->x[1];
	REAL radius = *largest_dist;

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


struct PointReduce {
	__host__ __device__
	Point operator()(const Point &a, const Point &b) {
		Point res;
		res.x[0] = a.x[0] + b.x[0];
		res.x[1] = a.x[1] + b.x[1];
		return res;
	};
};

/*
 * Calculated the average point of a set of points.
 *
 * @param avgPoint storage to average points.
 * @param pts Array of points.
 * @param npts Number of points in pts.
 */
void CalcAvgPoint(Point& avgPoint, Point* pts_d, int* npts) {
    int N = (*npts);

    thrust::device_ptr<Point> thrust_pts(pts_d);
	PointReduce myreduce; 

    //Point result = thrust::reduce(thrust_pts, thrust_pts + N, avgPoint, PointSum());
    Point result = thrust::reduce(
		thrust_pts,
		thrust_pts + N,
		avgPoint,
		myreduce
	);

	avgPoint.x[0] = avgPoint.x[0] / N;  
	avgPoint.x[1] = avgPoint.x[1] / N;  
}


/*
 * Computes the maximum distance bewteen two points in the array pts and stores this value in largest_dist.
 *
 * @param pts Array of Point structs on the device
 * @param npts lenght of pts array on device
 * @param laegrest_dist Should be initialized to be 0, this is output of the programms containing the largest distance bewtween all points pts 
 */
__global__ void computeMaxDistPts(Point* pts, int* npts, REAL* largest_dist) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	REAL dist; 
	REAL local_largest_dist = 0;

	if (idx < (*npts)) {
		int i = idx;
		for (int j=0; j<(*npts); j++) {

			dist = sqrtf( (pts[i].x[0] - pts[j].x[0])*(pts[i].x[0] - pts[j].x[0]) +
						  (pts[i].x[1] - pts[j].x[1])*(pts[i].x[1] - pts[j].x[1]));

			if (dist > (local_largest_dist)) {
				local_largest_dist = dist; 
			}
		}
	}

	#ifdef REALFLOAT
		atomicMaxFloat(largest_dist, local_largest_dist); 
	#endif
	#ifdef REALDOUBLE
		atomicMaxDouble(largest_dist, local_largest_dist); 
	#endif
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
	int N;
	N = *nTriMax;
	dim3 threadsPerBlock0(ntpb);
	dim3 numBlocks0(N/threadsPerBlock0.x + (!(N % threadsPerBlock0.x) ? 0:1));

	// reset index of the point to insert in each triangle
	resetInsertPtInTris<<<numBlocks0, threadsPerBlock0>>>(triList_d, nTriMax_d);

	N = *npts;
	dim3 threadsPerBlock1(ntpb);
	dim3 numBlocks1(N/threadsPerBlock1.x + (!(N%threadsPerBlock1.x) ? 0:1));

	cudaMemset(nptsUninserted_d, 0, sizeof(int)); // set counter for number of points uninserted to be 0
	updatePtsUninserted<<<numBlocks1, threadsPerBlock1>>>(npts_d, ptToTri_d, ptsUninserted_d, nptsUninserted_d);
	gpuSort(ptsUninserted_d, npts_d);

	// caclualtes and writes the smallest distance to circumcenter of triangle
	setInsertPtsDistance<<<numBlocks1, threadsPerBlock1>>>(pts_d, npts_d, triList_d, ptToTri_d, ptsUninserted_d, nptsUninserted_d);
	// finda and writes the index of point with smallelst distance to circumcenter of triangle
	setInsertPts<<<numBlocks1, threadsPerBlock1>>>(pts_d, npts_d, triList_d, ptToTri_d, ptsUninserted_d, nptsUninserted_d);

	N = *nTri;
	dim3 threadsPerBlock2(ntpb);
	dim3 numBlocks2(N/threadsPerBlock2.x + (!(N%threadsPerBlock2.x) ? 0:1));

	cudaMemset(nTriWithInsert_d, 0, sizeof(int)); // resets counter of the number of poinnts to insert
	// counts the number of triangles which are marked for insertion
	prepTriWithInsert<<<numBlocks2, threadsPerBlock2>>>(triList_d, nTri_d, triWithInsert_d, nTriWithInsert_d);

	//cudaMemset(nTriWithInsert_d, 0, sizeof(int));

	// save innsert num to file for each iteration
	cudaMemcpy(nTriWithInsert, nTriWithInsert_d, sizeof(int), cudaMemcpyDeviceToHost);
	if (saveHistory == true) {
		fprintf(insertedPerIterfile, "%d\n", (*nTriWithInsert));
	}

	// sorts the array triWithInsert for efficient thread launches
	gpuSort(triWithInsert_d, nTri_d); 

	// resets the value of the distance of point to circumcenter in each triangle
	resetBiggestDistInTris<<<numBlocks2, threadsPerBlock2>>>(triList_d, nTriMax_d);
}


/*
 * Resets the default vaulue of the index of the point to insert into the triangle.
 */
__global__ void resetInsertPtInTris(Tri* triList, int* nTriMax) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTriMax)) {
		triList[idx].insertPt = -1;
	}
}


/*
 * Updates the ptsUninserted array for use by later functions. ptsUninserted needs to be
 * sorted after this kernel is called
 */
__global__ void updatePtsUninserted(int* npts, int* ptToTri, int* ptsUninserted, int* nptsUninserted) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	// definitely can optimize this part, reduce thread divergence
	if (idx < (*npts)) {
		if (ptToTri[idx] == -1) {
			ptsUninserted[idx] = -1;
		} 
		else if (ptToTri[idx] >= 0) {
			ptsUninserted[idx] = idx;
			atomicAdd(nptsUninserted, 1);
		}
	}
}

/*
 * Writes the shortest distance between a point in each triangle and the center
 * of its circumcircle to triangles slot named 'insertPt_dist'.
 */
__global__ void setInsertPtsDistance(Point* pts, int* npts, Tri* triList, int* ptToTri, int* ptsUninserted, int* nptsUninserted) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nptsUninserted)) {
		int ptIdx = ptsUninserted[idx];
		int idxTri = ptToTri[ptIdx];

		// find circumcenter for the trianle this point is inside of
		Point circumcenter;
		circumcircle_center(pts[triList[idxTri].p[0]], pts[triList[idxTri].p[1]], pts[triList[idxTri].p[2]], &circumcenter);

		//REAL ptDist = dist(circumcenter , pts[idx]); 
		REAL ptDist = dist(circumcenter , pts[ptIdx]); 
		#ifdef REALFLOAT
			atomicMinFloat(&(triList[idxTri].insertPt_dist), ptDist);
		#endif
		#ifdef REALDOUBLE
			atomicMinDouble(&(triList[idxTri].insertPt_dist), ptDist);
		#endif
	}
}

/*
 * Writes the index of the point int the array 'pts' to be inserted into each triangle
 * in the triangles slot named 'insertPt'.
 */
__global__ void setInsertPts(Point* pts, int* npts, Tri* triList, int* ptToTri, int* ptsUninserted, int* nptsUninserted) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//if (idx < (*npts)) {
	if (idx < (*nptsUninserted)) {
		int ptIdx = ptsUninserted[idx];
		int idxTri = ptToTri[ptIdx];

		Point circumcenter;
		circumcircle_center(pts[triList[idxTri].p[0]], pts[triList[idxTri].p[1]], pts[triList[idxTri].p[2]], &circumcenter);

		REAL ptDist = dist(circumcenter, pts[ptIdx]); 
		if (ptDist == triList[idxTri].insertPt_dist) {
			atomicExch(&(triList[idxTri].insertPt), ptIdx);
		}
	}
}

/*
 *	 
 */
__global__ void prepTriWithInsert(Tri* triList, int* nTri, int* triWithInsert, int* nTriWithInsert) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTri)) {
		int insert_stat = triList[idx].insert && (triList[idx].insertPt != -1);

		triWithInsert[idx] = idx*(insert_stat == 1) + (-(insert_stat == 0));
		if (insert_stat) {
			atomicAdd(nTriWithInsert, insert_stat);
		}

	}
}

/*
 * Resets the distance of the point to insert in the tri struct to the largest
 * number possible. Neccesary for the next iteration of finding the next point
 * with minimun distance to circumcenter.
 */
__global__ void resetBiggestDistInTris(Tri* triList, int* nTriMax) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTriMax)) {
		triList[idx].insertPt_dist = (REAL)(unsigned long)-1;
	}
}

/* ================================== INSERT ======================================== */

/*
 * Main function for inserting points in triangles in parallel.
 */
void Delaunay::insert() {
	int N;

	cudaMemcpy(nTriWithInsert, nTriWithInsert_d, sizeof(int), cudaMemcpyDeviceToHost);
	N = *nTriWithInsert;
	dim3 threadsPerBlock(ntpb);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));
	cudaMemcpy(triList_prev_d, triList_d, (*nTriMax) * sizeof(Tri), cudaMemcpyDeviceToDevice);

	// Insert point in marked triangles
	insertKernel<<<numBlocks, threadsPerBlock>>>(triList_d, nTri_d, nTriMax_d, triWithInsert_d, nTriWithInsert_d, ptToTri_d, triList_prev_d);
	// Update neighbours
	updateNbrsAfterIsertKernel<<<numBlocks, threadsPerBlock>>>(triList_d, triWithInsert_d, nTriWithInsert_d, nTri_d, triList_prev_d);
 
	// update number of triangles in triList
	// Update number of triangles and number of points inserted
	arrayAddVal<<<1, 1>>>(nTri_d        , nTriWithInsert_d, 2, 1);
	arrayAddVal<<<1, 1>>>(nptsInserted_d, nTriWithInsert_d, 1, 1);
	
	// reset triWithInsert_d for next iteraiton
	cudaMemset(triWithInsert_d, -1, (*nTri) * sizeof(int));
}

/*
 * Inserts points in parallel into triangles marked for insertion.
 */
__global__ void insertKernel(Tri* triList, int* nTri, int* nTriMax, int* triWithInsert, int* nTriWithInsert, int* ptToTri, Tri* triList_prev) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTriWithInsert)) {
		int triIdx = triWithInsert[idx];
		insertInTri(triIdx, triList, (*nTri) + 2*idx, ptToTri, triList_prev);
	}
}

/*
 * Pick a triangle by index 'i' in triList and insert its center point.
 * Returns the number of a new triangles created. 
 */
__device__ int insertInTri(int i, Tri* triList, int newTriIdx, int* ptToTri, Tri* triList_prev) {
	int r = triList[i].insertPt;
	ptToTri[r] = -1;

	insertPtInTri(r, i, triList, newTriIdx, triList_prev);
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
__device__ int insertPtInTri(int r, int i, Tri* triList, int newTriIdx, Tri* triList_prev) {
	int p[3] = {triList_prev[i].p[0],
				triList_prev[i].p[1],
				triList_prev[i].p[2]};

	int n[3] = {triList_prev[i].n[0],
				triList_prev[i].n[1],
				triList_prev[i].n[2]};

	int o[3] = {triList_prev[i].o[0],
				triList_prev[i].o[1],
				triList_prev[i].o[2]};

	int p0[3] = {r            , p[0], p[1]};
	int n0[3] = {newTriIdx + 1, n[0], newTriIdx};
	int o0[3] = {1            , o[0], 2};

	int p1[3] = {r, p[1], p[2]         };
	int n1[3] = {i, n[1], newTriIdx + 1};
	int o1[3] = {1, o[1], 2            };

	int p2[3] = {r        , p[2], p[0]};
	int n2[3] = {newTriIdx, n[2], i   };
	int o2[3] = {1        , o[2], 2   };

	writeTri(&(triList[i            ]), p0, n0, o0);
	writeTri(&(triList[newTriIdx    ]), p1, n1, o1);
	writeTri(&(triList[newTriIdx + 1]), p2, n2, o2);

	// marking edge for flipping
	triList[i            ].flip = 1;
	triList[newTriIdx    ].flip = 1;
	triList[newTriIdx + 1].flip = 1;

	// maybe insead of this use the insertPt array and reserve maybe -9 as a marker for insert == true
	triList[i            ].insert = true;
	triList[newTriIdx    ].insert = true;
	triList[newTriIdx + 1].insert = true;

	return 0;
}


//=================================================================================================

__global__ void updateNbrsAfterIsertKernel(Tri* triList, int* triWithInsert, int* nTriWithInsert, int* nTri, Tri* triList_prev) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTriWithInsert)) {
		int triIdx = triWithInsert[idx];
		updateNbrsAfterIsert(triIdx, triList, (*nTri) + 2*idx, triList_prev);
	}
}
__device__ void updateNbrsAfterIsert(int i, Tri* triList, int newTriIdx, Tri* triList_prev) {

	int n[3] = {triList_prev[i].n[0],
				triList_prev[i].n[1],
				triList_prev[i].n[2]};

	int o[3] = {triList_prev[i].o[0],
				triList_prev[i].o[1],
				triList_prev[i].o[2]};


	int triIdx[3] = {i, newTriIdx, newTriIdx + 1};
		
	for (int k=0; k<3; ++k) {
		int mvInsNbr = (o[k] + 1) % 3;

		if (n[k] >= 0) { // if nbr exist, this will almost allways happen with large number of points
			if (triList[n[k]].insert & (triList[n[k]].insertPt != -1)) {
				int idxNbr_k = n[k];

				// move anticlockwise in the nbr tri which just was inserted into 
				for (int i=0; i<mvInsNbr; ++i) {
					idxNbr_k = triList[idxNbr_k].n[2];
				}

				triList[idxNbr_k].o[1] = 0;
				triList[idxNbr_k].n[1] = triIdx[k];
			} else { 
				triList[n[k]].o[mvInsNbr] = 0;
				triList[n[k]].n[mvInsNbr] = triIdx[k];
			}
		}
	}
}



/* ===================================== UPDATE POINT LOCATIONS ========================================= */

/*
 * Updates the array which makes a map from the insex of a point in 'pts' to the triangle in 'triList' it
 * is sits in.
 */
void Delaunay::updatePointLocations() {
	cudaMemcpy(nptsUninserted, nptsUninserted_d, sizeof(int), cudaMemcpyDeviceToHost);
	int N = *nptsUninserted;
	dim3 threadsPerBlock2(ntpb);
	dim3 numBlocks2(N/threadsPerBlock2.x + (!(N % threadsPerBlock2.x) ? 0:1));
	updatePointLocationsKernel<<<numBlocks2, threadsPerBlock2>>>(pts_d, npts_d, triList_d, nTri_d, ptToTri_d, ptsUninserted_d, nptsUninserted_d);
//
//	cudaMemcpy(nptsUninserted, nptsUninserted_d, sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(nTri, nTri_d, sizeof(int), cudaMemcpyDeviceToHost);
//	int N = (*nptsUninserted)*(*nTri);
//	printf("================================= N=%d | (*nptsUninserted)=%d | (*nTri)=%d \n", N, (*nptsUninserted), (*nTri));
//	dim3 threadsPerBlock2(ntpb);
//	dim3 numBlocks2(N/threadsPerBlock2.x + (!(N % threadsPerBlock2.x) ? 0:1));
//	updatePointLocationsKernel<<<numBlocks2, threadsPerBlock2>>>(pts_d, npts_d, triList_d, nTri_d, ptToTri_d, ptsUninserted_d, nptsUninserted_d);


//	cudaMemcpy(nTri, nTri_d, sizeof(int), cudaMemcpyDeviceToHost);
//	int N = (*nTri);
////	cudaMemcpy(nptsUninserted, nptsUninserted_d, sizeof(int), cudaMemcpyDeviceToHost);
////	int N = *nptsUninserted;
//	dim3 threadsPerBlock2(ntpb);
//	dim3 numBlocks2(N/threadsPerBlock2.x + (!(N % threadsPerBlock2.x) ? 0:1));
//	updatePointLocationsKernel<<<numBlocks2, threadsPerBlock2>>>(pts_d, npts_d, triList_d, nTri_d, ptToTri_d, ptsUninserted_d, nptsUninserted_d);
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
__global__ void updatePointLocationsKernel(Point* pts, int* npts, Tri* triList, int* nTri, int* ptToTri, int* ptsUninserted, int* nptsUninserted) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	// definitely can optimize this part, reduce thread divergence
	if (idx < (*nptsUninserted)) {
		int ptIdx = ptsUninserted[idx];
		int local_t = 0;
		int used = 0;
		int cont;

		// definitely can optimize this
		for (int t=0; t<(*nTri); ++t) { // for each triangle check if point is contained inside it
//			if (contains(t, ptIdx, triList, pts) == 1) {
//				used++;
//				local_t = t;
//			}
			cont = (contains(t, ptIdx, triList, pts) == 1);
			used += cont;
			local_t = local_t*(cont == 0) + t*(cont == 1);
		}

		if (used > 1) { printf("POINT %d LIES IN MORE THAN 1 TRIANGLE\n", ptIdx); }

		if (used == 1) {
			//atomicExch(&(ptToTri[ptIdx]), local_t);
			ptToTri[ptIdx] = local_t;
		}
	}
}

//__global__ void updatePointLocationsKernel(Point* pts, int* npts, Tri* triList, int* nTri, int* ptToTri, int* ptsUninserted, int* nptsUninserted) {
//	int idx = blockIdx.x*blockDim.x + threadIdx.x;
//
//	// definitely can optimize this part, reduce thread divergence
//	if (idx < (*nptsUninserted)*(*nTri)) {
//	//if (idx < (*nptsUninserted)) {
//		//int ptIdx = ptsUninserted[idx];
//		int ptIdx = ptsUninserted[idx / (*nTri)];
//		int t = idx % (*nTri);
//		int local_t = 0;
//		int used = 0;
//		int cont;
//		Tri tri;
//
//		// definitely can optimize this
//		//for (int t=0; t<(*nTri); ++t) { // for each triangle check if point is contained inside it
//			for (int i=0; i<3; ++i) {
//				tri.p[i] = triList[t].p[i];
//			}
//			
//			cont = (contains(&tri, ptIdx, pts) == 1);
//			used += cont;
//			local_t = local_t*(cont == 0) + t*(cont == 1);
//		//}
//
//		if (used > 1) { printf("POINT %d LIES IN MORE THAN 1 TRIANGLE\n", ptIdx); }
//
//		if (used == 1) {
//			//atomicExch(&(ptToTri[ptIdx]), local_t);
//			ptToTri[ptIdx] = local_t;
//		}
//	}
//}
//
//__global__ void updatePointLocationsKernel(Point* pts, int* npts, Tri* triList, int* nTri, int* ptToTri, int* ptsUninserted, int* nptsUninserted) {
//	int idx = blockIdx.x*blockDim.x + threadIdx.x;
//
//	// definitely can optimize this part, reduce thread divergence
//	if (idx < (*nTri)) {
//		//int ptIdx = ptsUninserted[idx];
//		//int ptIdx = ptsUninserted[idx / (*nTri)];
//		int loc_ptIdx;
//		int t = idx;// % (*nTri);
//		int local_t = 0;
//		int used = 0;
//		int cont;
//		Tri tri;
//		for (int i=0; i<3; ++i) {
//			tri.p[i] = triList[t].p[i];
//		}
//
//		// definitely can optimize this
//		for (int ptIdx=0; ptIdx<(*nptsUninserted); ++ptIdx) { // for each triangle check if point is contained inside it
//			cont = (contains(&tri, ptIdx, pts) == 1);
//			used += cont;
//			//local_t = local_t*(cont == 0) + t*(cont == 1);
//			loc_ptIdx = loc_ptIdx *(cont == 0) + ptIdx*(cont == 1);
//
//			if (used > 1) { printf("POINT %d LIES IN MORE THAN 1 TRIANGLE\n", ptIdx); }
//		}
//
//		if (used == 1) {
//			//atomicExch(&(ptToTri[ptIdx]), local_t);
//			ptToTri[loc_ptIdx] = local_t;
//		}
//	}
//}

//__global__ void updatePointLocationsKernel(Point* pts, int* npts, Tri* triList, int* nTri, int* ptToTri, int* ptsUninserted, int* nptsUninserted) {
//	int t = blockIdx.x*blockDim.x + threadIdx.x;
//
//
//	// definitely can optimize this part, reduce thread divergence
//	if (t < (*nTri)) {
//	//if (idx < (*nptsUninserted)) {
//		//int ptIdx = ptsUninserted[idx];
//		//int local_t = 0;
//		int local_ptIdx = 0;
//		int used = 0;
//		int cont;
//		Tri tri;
//		for (int i=0; i<3; i++) {
//			tri.p[i] = triList[t].p[i];
//		}	
//
//		// definitely can optimize this
//		//for (int t=0; t<(*nTri); ++t) { // for each triangle check if point is contained inside it
//		for (int ptIdx=0; ptIdx<(*nptsUninserted); ++ptIdx) { // for each triangle check if point is contained inside it
//			cont = (contains(&tri, ptIdx, pts) == 1);
//			used += cont;
//			//local_t = local_t*(cont == 0) + t*(cont == 1);
//			local_ptIdx = local_ptIdx*(cont == 0) + ptIdx*(cont == 1);
//		}
//
//		if (used > 1) { printf("POINT %d LIES IN MORE THAN 1 TRIANGLE\n", local_ptIdx); }
//
//		if (used == 1) {
//			//atomicExch(&(ptToTri[ptIdx]), local_t);
//			//ptToTri[ptIdx] = local_t;
//			//atomicExch(&(ptToTri[local_ptIdx]), t);
//			ptToTri[local_ptIdx] = t;
//		}
//	}
//}
//


/*
 * Checks if a triangle with index 't' contains point with index 'r'. Returns 1 if the
 * point is inside or if on the boundary and -1 if its on the outside.
 */
__device__ int contains(Tri* tri, int r, Point* pts) {

	REAL area;
	int i, j;
	int not_contained = 0;

	for (i=0; i<3; ++i) {
		j = (i+1) % 3;
		// area = area of triangle (21.3.2) (21.3.10) 
		area = (pts[tri->p[j]].x[0] - pts[tri->p[i]].x[0])*(pts[r].x[1] - pts[tri->p[i]].x[1]) - 
			   (pts[tri->p[j]].x[1] - pts[tri->p[i]].x[1])*(pts[r].x[0] - pts[tri->p[i]].x[0]);

		not_contained = not_contained | (area <= 0.0);
	}

	return (-(not_contained == 1)) + (not_contained == 0);
}

/*
 * Checks if a triangle with index 't' contains point with index 'r'. Returns 1 if the
 * point is inside or if on the boundary and -1 if its on the outside.
 */
__device__ int contains(int t, int r, Tri* triList, Point* pts) {

	REAL area;
	int i, j;
	int not_contained = 0;

	for (i=0; i<3; ++i) {
		j = (i+1) % 3;
		// area = area of triangle (21.3.2) (21.3.10) 
		area = (pts[triList[t].p[j]].x[0] - pts[triList[t].p[i]].x[0])*(pts[r].x[1] - pts[triList[t].p[i]].x[1]) - 
			   (pts[triList[t].p[j]].x[1] - pts[triList[t].p[i]].x[1])*(pts[r].x[0] - pts[triList[t].p[i]].x[0]);

		not_contained = not_contained | (area <= 0.0);
	}

	return (-(not_contained == 1)) + (not_contained == 0);

}

/*
 * Performs parallel flipping operations until the triangulation is Delaunay or gets stuck flipping.
 */
int Delaunay::flip() {
	int numConfigsFlipped = 0;
	int N;
	cudaMemcpy(nTri, nTri_d, sizeof(int), cudaMemcpyDeviceToHost);

	*nTriToFlip = 0;
	cudaMemcpy(nTriToFlip_d, nTriToFlip, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(triToFlip_d, -1, (*nTriMax) * sizeof(int));

	checkIncircleAll();   // Perform incircle checks on all triangles
	checkFlipConflicts(); // Take account for possible flip conflicts 

	cudaMemcpy(nTriToFlip, nTriToFlip_d, sizeof(int), cudaMemcpyDeviceToHost);

	int flipIter = 0;
	int timesTheSame = 0; // safeguard for getting stuck in flipping
	// While there are configurations to flip
	int prev_nTriToFlip;
	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);
	while ((*nTriToFlip) > 0 && (timesTheSame < 2)) {
		cudaEventRecord(start, 0);
		timesTheSame += (prev_nTriToFlip == (*nTriToFlip));
		prev_nTriToFlip = (*nTriToFlip); 
		N = *nTriToFlip;
		dim3 threadsPerBlock1(ntpb);
		dim3 numBlocks1(N/threadsPerBlock1.x + (!(N % threadsPerBlock1.x) ? 0:1));

		numConfigsFlipped += N;
		flipIter++;
		if (info == true) {
			printf("    [Performing flip iteration %d]\n"     , flipIter);
			printf("        Flipping %d configurations\n"     , N);
			//printf("        Iter : %d\n"                      , iter);
		}

		if (saveHistory == true) {
			fprintf(flipedPerIterfile, "%d ", N);
		}

		float quadFlipTime = timeGPU([this] () { quadFlip(); }); // Flip marked configurations

		if (saveHistory == true) { saveToFile(); }

		// Reset  nTriToFlip and triToFlip_d to -1 in each entry 
		*nTriToFlip = 0;
		//cudaMemcpy(nTriToFlip_d, nTriToFlip, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(nTriToFlip_d, 0, sizeof(int));
		cudaMemset(triToFlip_d, -1, (*nTriMax) * sizeof(int));

		float checkIncircleAllTime   = timeGPU([this] () { checkIncircleAll()  ; }); // Perform incircle checks on all triangles
		float checkFlipConflictsTime = timeGPU([this] () { checkFlipConflicts(); }); // Take account for possible flip conflicts 

		//cudaMemcpy(nTriToFlip, nTriToFlip_d, sizeof(int), cudaMemcpyDeviceToHost);
		float memcpyTime = timeGPU([this] () { cudaMemcpy(nTriToFlip, nTriToFlip_d, sizeof(int), cudaMemcpyDeviceToHost); }); // Flip marked configurations

		cudaEventRecord(finish, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(finish);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, finish);
		if (info == true) {
			printf("        Time taken quadFlipTime           : %f\n", quadFlipTime);
			printf("        Time taken checkIncircleAllTime   : %f\n", checkIncircleAllTime );
			printf("        Time taken checkFlipConflictsTime : %f\n", checkFlipConflictsTime );
			printf("        Time taken memcpyTime             : %f\n", memcpyTime  );
			printf("        Total Time taken                  : %f\n", elapsedTime/1000.0);
		}
	}

	if (saveHistory == true) {
		if (flipIter == 0) {
			fprintf(flipedPerIterfile, "%d ", 0);
		}
		fprintf(flipedPerIterfile, "\n");
		fprintf(flipedPerIterfile, "%d %d\n", flipIter, numConfigsFlipped);
		//fprintf(flipedPerIterfile, "%f\n", flipIter, numConfigsFlipped);
		fprintf(flipedPerIterfile, "\n");
	}


	N = *nTri;
	dim3 threadsPerBlock2(ntpb);
	dim3 numBlocks2(N/threadsPerBlock2.x + (!(N % threadsPerBlock2.x) ? 0:1));
	// Reset mark for flipping in tri struct 
	resetTriToFlip<<<numBlocks2, threadsPerBlock2>>>(triList_d, nTri_d);

	return numConfigsFlipped;
}


/* ====================================================================  */ 

void Delaunay::checkIncircleAll() {
	cudaMemcpy(nTri, nTri_d, sizeof(int), cudaMemcpyDeviceToHost);
	int N = *nTri;
	dim3 threadsPerBlock(N < ntpb ? min_ntpb : ntpb);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));
	checkIncircleAllKernel<<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, nTri_d, pts_d);
	gpuSort(triToFlip_d, nTri_d);
}

/*
 * Performs an incricle check on each edge of each triangle currently in the triangulation
 * and sets '.flip' of each triagle to be an edge which should be flipped if there are any.
 */
__global__ void checkIncircleAllKernel(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri, Point* pts) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int edgeToFlip = -1;

	// for each triangle
	if (idx < (*nTri)) {
		int a = idx;
		int opp_idx;
		int flip;
		int b;

		for (int edge=0; edge<3; ++edge) {
			b = triList[a].n[ edge ]; // b is the index of the triangle across the edge

			if (b >= 0) { //true most of the time, nbr exists most of the time
				opp_idx = triList[a].o[ edge ]; // index of opposite point in neighbour
				flip = (0 < incircle(pts[triList[b].p[opp_idx]],
										 pts[triList[a].p[0      ]],
										 pts[triList[a].p[1      ]],
										 pts[triList[a].p[2      ]] ));

				// if one of the edges should be flipped, mark it
				//edges_to_flip += flip;
				//if (flip) { edgeToFlip = edge; }
				edgeToFlip = (flip == 1)*edge + (flip == 0)*edgeToFlip; 

			}
		}

		if (edgeToFlip >= 0) {
			triList[a].flip = edgeToFlip;
			triToFlip[a] = a;
			atomicAdd(nTriToFlip, 1);
		}
	}
}

/* ====================================================================  */ 


void Delaunay::checkFlipConflicts() {
	int N;

	N = *nTri;
	dim3 threadsPerBlock(ntpb);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));

	prepForConflicts<<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, nTriMax_d);
	setConfigIdx    <<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, nTri_d);
	
	cudaMemset(subtract_nTriToFlip_d, 0, sizeof(int));
	storeNonConflictConfigs<<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, nTri_d, subtract_nTriToFlip_d);

	arraySubVal<<<1, 1>>>(nTriToFlip_d, subtract_nTriToFlip_d, 1, 1); 

	gpuSort(triToFlip_d, nTri_d);

	resetTriToFlipThisIter<<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d);
	markTriToFlipThisIter <<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d);
}

/*
 * Sets the configuration index of each triangle in triList with a default value, here the largest value it 
 * can be in this iteration.
 */
__global__ void prepForConflicts( int* triToFlip, int* nTriToFlip, Tri* triList, int* nTriMax) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//if (idx < (*nTri)) {
	if (idx < (*nTriToFlip)) {
		int a = triToFlip[idx];
		triList[a].configIdx = (*nTriMax);
		//triList[idx].configIdx = (*nTriMax);
	}
}

/*
 * Writes the minimun configuration index to each triangle marked for fliping  
 */
__global__ void setConfigIdx(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < (*nTriToFlip)) {
		int a = triToFlip[idx];
		int b = triList[a].n[ triList[a].flip ];

		atomicMin(&(triList[a].configIdx), min(a, b));
		atomicMin(&(triList[b].configIdx), min(a, b));
	}
}

__global__ void storeNonConflictConfigs(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri, int* subtract_nTriToFlip) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < (*nTriToFlip)) {

		int a = triToFlip[idx];
		int b = triList[a].n[ triList[a].flip ];

		int flip = ((a == min(a, b)) &&
					(triList[a].configIdx == min(a, b)) &&
		       	   	(triList[b].configIdx == min(a, b))
		);
	
		if (!flip) { // overwrite the previous checks if the triangle is not to be flipped this round
			atomicAdd(subtract_nTriToFlip, 1);
			atomicExch(&(triToFlip[idx]), -1);
		}
	}
}

/*
 * Resets the 'flipThisIter' (flip this iteration) field in the 'Tri' struct to -1
 * so that the followig kernel can mark the triangles which have been set for insertion
 * an iteraion of flipping.
 */
__global__ void resetTriToFlipThisIter(int* triToFlip, int* nTriToFlip, Tri* triList) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//if (idx < (*nTri)) {
	if (idx < (*nTriToFlip)) {
		int a = triToFlip[idx];
		triList[a].flipThisIter = -1;
		//triList[idx].flipThisIter = -1;
	}
}

/*
 * Marks the 'flipThisIter' (flip this iteration) field in the 'Tri' struct to 1
 * if a triangle is to be flipped this iteration. The triangle performing the 
 * flip and the neighbour its flipping with is marked.
 */
__global__ void markTriToFlipThisIter(int* triToFlip, int* nTriToFlip, Tri* triList) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < (*nTriToFlip)) {
		int a = triToFlip[idx];
		int flip_edge = triList[a].flip; // edge of triangle to flip across
		int b = triList[a].n[ flip_edge ]; // b is the index of the triangle across the edge

		atomicExch(&(triList[a].flipThisIter), 1);
		atomicExch(&(triList[b].flipThisIter), 1);
		atomicExch(&(triList[b].flip), ((triList[a].o[flip_edge] + 1) % 3)); // lets the flipping companion know where its going to flip
	}
}

void Delaunay::quadFlip() {
	cudaMemcpy(nTriToFlip, nTriToFlip_d, sizeof(int), cudaMemcpyDeviceToHost);

	int N = *nTriToFlip;
	dim3 threadsPerBlock(ntpb);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));

	// Writes relevant quadrilaterals 
	writeQuadKernel          <<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, quadList_d);
	// Overwrites new triangles internal structure
	flipFromQuadKernel       <<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, quadList_d);
	// Updates neighbours information
	updateNbrsAfterFlipKernel<<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, quadList_d);
}

__global__ void writeQuadKernel(int* triToFlip, int* nTriToFlip, Tri* triList, Quad* quadList) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < (*nTriToFlip)) {
		int a = triToFlip[idx];
		int flip_edge = triList[a].flip; // edge of triangle to flip across
		int b = triList[a].n[ flip_edge ]; // b is the index of the triangle across the edge

		writeQuads(a, flip_edge, b, triList, &(quadList[idx]));
	}
}

__device__ void writeQuads(int a, int e, int b, Tri* triList, Quad* quad) {
	
	int opp_idx = triList[a].o[e]; // index in neighbour of point opposite the edge marked for flipping in 'a'.

 	// temporary qaud "struct" data just to make it readable
	int p[4] = {triList[a].p[(e-1 + 3)%3], triList[a].p[e]                , triList[b].p[opp_idx], triList[a].p[(e + 1)%3]};
	int n[4] = {triList[a].n[(e-1 + 3)%3], triList[b].n[(opp_idx-1 + 3)%3], triList[b].n[opp_idx], triList[a].n[(e + 1)%3]};
	int o[4] = {triList[a].o[(e-1 + 3)%3], triList[b].o[(opp_idx-1 + 3)%3], triList[b].o[opp_idx], triList[a].o[(e + 1)%3]};

	// making sure nbr and opposite points are good and correcting then if not
	for (int k=0; k<4; ++k) {
		if (n[k] >= 0) { // true most of the time
			if (triList[n[k]].flipThisIter == 1) { // uhhh
				// the neighbouring flip will be performed in the direction of this flip, SCREAM
				if (triList[n[k]].flip == ((o[k] + 1) % 3)) {
					printf("[WARNING] CONFIG IDX %d | NEIGHBOURING TRIANGLE (%d) WILL FLIP INTO THIS CONFIGURATION (%d, %d)\n", triList[a].configIdx, n[k], a, b);
				}

				// in this case nbr will change in the neighbours flip
				else if (triList[n[k]].flip == ((o[k]    ) % 3)) {
					n[k] = triList[n[k]].n[ triList[n[k]].flip ]; // the index of the triangle which neighbour n[k] will be flipping with.
					o[k] = 0;
				}

				// in this case nbr stay the same in the neighbours flip
				else if (triList[n[k]].flip == ((o[k] + 2) % 3)) {
					//n[k] = n[k];
					o[k] = 2;
				}
			}
		}
	}

	// writeQuad
	for (int i=0; i<4; ++i) { 
		quad->p[i] = p[i];	
		quad->n[i] = n[i];	
		quad->o[i] = o[i];	
	}
}

__global__ void flipFromQuadKernel(int* triToFlip, int* nTriToFlip, Tri* triList, Quad* quadList) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < (*nTriToFlip)) {
		int a = triToFlip[idx];
		int flip_edge = triList[a].flip; // edge of triangle to flip across
		int b = triList[a].n[ flip_edge ]; // b is the index of the triangle across the edge

		flipFromQuad(a, flip_edge, b, triList, &(quadList[idx]));
	}
}

__device__ void flipFromQuad(int a, int e, int b, Tri* triList, Quad* quad) {

	int p[4];
	int n[4];
	int o[4];

	// copy from quad
	for (int i=0; i<4; ++i) {
		p[i] = quad->p[i];
		n[i] = quad->n[i];
		o[i] = quad->o[i];
	}

	int ap[3] = {p[0], p[1], p[2]};
	int an[3] = {n[0], n[1], b   };
	int ao[3] = {o[0], o[1], 1   };

	int bp[3] = {p[2], p[3], p[0]};
	int bn[3] = {n[2], n[3], a   };
	int bo[3] = {o[2], o[3], 1   };

	writeTri(&(triList[a]), ap, an, ao);
	writeTri(&(triList[b]), bp, bn, bo);
}


__global__ void updateNbrsAfterFlipKernel(int* triToFlip, int* nTriToFlip, Tri* triList, Quad* quadList) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < (*nTriToFlip)) {
		int a = triToFlip[idx];
		int flip_edge = 2 ; // edge of triangle which now shares the now flipped pair
		int b = triList[a].n[ flip_edge ]; // b is the index of the triangle across the edge

		//printf("idx: %d | flip pair: (%d, %d)\n", idx, a, b);

		updateNbrsAfterFlip(a, flip_edge, b, triList, &(quadList[idx]));

		triList[a].flipThisIter = -1;
		triList[b].flipThisIter = -1;
	}
}

// maybe need to mark non conflicting update nbrs step
__device__ void updateNbrsAfterFlip(int a, int e, int b, Tri* triList, Quad* quad) {
	
	//int p[4];
	int n[4];
	int o[4];

	// read quad
	for (int i=0; i<4; ++i) {
		//p[i] = quad->p[i];
		n[i] = quad->n[i];
		o[i] = quad->o[i];
	}

	int nbrs[4] = {a, a, b, b};
	int opps[4] = {2, 0, 2, 0};

	for (int k=0; k<4; ++k) {
		int mvIdx = (o[k] + 1) % 3;
		if (n[k] >= 0)  { // n[k] is almost always >= 0 with large number of points
			triList[n[k]].n[mvIdx] = nbrs[k];
			triList[n[k]].o[mvIdx] = opps[k];
		}
	}
}

__global__ void resetTriToFlip(Tri* triList, int* nTri) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < (*nTri)) {
		triList[idx].flip = -1;
	}
}

//__global__ void resetTriToFlip(Tri* triList, int* nTri) {
//	int idx = blockIdx.x*blockDim.x + threadIdx.x;
//
//	if (idx < (*nTri)) {
//		triList[idx].flip = -1;
//	}
//}


// ========== DELAUNAY CHECK =========

/* 
 * Looks into each triangle and checks each of its edges for whether it should 
 * be flipped or not. Returns the number of edges that need flipping.
 */
int Delaunay::delaunayCheck() {
	cudaMemcpy(nTri, nTri_d, sizeof(int), cudaMemcpyDeviceToHost);

	int N = *nTri;
	dim3 threadsPerBlock(ntpb > 32 ? 32 : ntpb);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));

	int nEdges[1] = {0};
	int *nEdges_d;
	cudaMalloc(&nEdges_d, sizeof(int)); 
	cudaMemcpy(nEdges_d, nEdges, sizeof(int), cudaMemcpyHostToDevice);
	delaunayCheckKernel<<<numBlocks, threadsPerBlock>>>(triList_d, nTri_d, pts_d, nEdges_d);

	cudaMemcpy(nEdges, nEdges_d, sizeof(int), cudaMemcpyDeviceToHost);
	if ((*nEdges) > 0) {

		if (info == true) {
			//printf("\nTriangulation is NOT Delaunay, with %d illegal edges\n\n", *nEdges);
			printf("\nTriangulation is NOT Delaunay, is %.4f%% Delaunay, with %d illegal edges\n\n", 100*((float)((*nEdgesMax) - (*nEdges)) / (float)(*nEdgesMax)) , *nEdges);
		}
	}
	else {
		if (info == true) {
			printf("\nTriangulation is Delaunay\n\n");
		}
	}

	cudaFree(nEdges_d);

	return *nEdges;
}

__global__ void delaunayCheckKernel(Tri* triList, int* nTri, Point* pts, int* nEdges) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int edges_to_flip = 0;

	if (idx < (*nTri)) {

		// for each edge
		for (int edge=0; edge<3; ++edge) {
			int a = idx;
			int b = triList[a].n[ edge ]; // b is the index of the triangle across the edge

			if (b >= 0) {
				int opp_idx = triList[a].o[ edge ];
				//int to_flip = (0 < checkLegality(a, edge, b, triList, pts));
				int flip = (0 < incircle(pts[triList[b].p[opp_idx]],
										 pts[triList[a].p[0      ]],
										 pts[triList[a].p[1      ]],
										 pts[triList[a].p[2      ]] ));

				flip = flip && (a == min(a, b));
				
				edges_to_flip += flip;
				//if (flip) { printf("in tri: %d, edge: %d needs to flip\n", idx, edge); }
			}
		}

		atomicAdd(nEdges, edges_to_flip);
	}
}

/* ====================================== SAVE TO FILE ============================================= */

void Delaunay::saveToFile(bool end) {
	cudaMemcpy(nTri   , nTri_d   ,              sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(triList, triList_d, (*nTriMax) * sizeof(Tri), cudaMemcpyDeviceToHost);

	if (end == false) { // save all triangles
		fprintf(trifile, "%d %d\n", iter, *nTri);
		for (int i=0; i<(*nTri); ++i) {
			for (int j=0; j<3; ++j) { fprintf(trifile, "%d ", triList[i].p[j]); } 
			for (int j=0; j<3; ++j) { fprintf(trifile, "%d ", triList[i].n[j]); } 
			for (int j=0; j<3; ++j) { fprintf(trifile, "%d ", triList[i].o[j]); } 
			fprintf(trifile, "%d ", triList[i].flip);
			fprintf(trifile, "%d ", triList[i].insert);
			fprintf(trifile, "%d ", triList[i].flipThisIter);

			fprintf(trifile, "\n");
		}

		fprintf(trifile, "\n");
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
	
		fprintf(trifile, "%d %d\n", iter, nTriFinal);
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

			// ==== check legality ====  
			int edge;
			int a = i;
			for (int e=0; e<3; ++e) {
				edge = e;
				int b = triList[a].n[ edge ]; // b is the index of the triangle across the edge

				if (b >= 0) { //true most of the time, nbr exists most of the time
					int opp_idx = triList[a].o[edge]; // index of opposite point in neighbour
					int flip = (0 < incircle(pts[triList[b].p[opp_idx]],
											 pts[triList[a].p[0      ]],
											 pts[triList[a].p[1      ]],
											 pts[triList[a].p[2      ]] ));

					// if one of the edges should be flipped print it
//					if (flip) {
//						printf("tri: %d should have edge: %d flipped\n", a, e);
//					}
				}
			}
	
			for (int j=0; j<3; ++j) { fprintf(trifile, "%d ", triList[i].p[j]); } 
			for (int j=0; j<3; ++j) { fprintf(trifile, "%d ", triList[i].n[j]); } 
			for (int j=0; j<3; ++j) { fprintf(trifile, "%d ", triList[i].o[j]); } 
			fprintf(trifile, "%d ", triList[i].flip);
			fprintf(trifile, "%d ", triList[i].insert);
			fprintf(trifile, "%d ", triList[i].flipThisIter);

			fprintf(trifile, "\n");
		}
 
		fprintf(trifile, "\n");
		iter++;
	}
}

/* ====================================== CALCULATING TIMINGS ============================================= */

/*
 * A wrapper for funtions composed of cuda kernels which outputs their runtime time in seconds.
 * This wrapper makes use of 'cudaEvent' functions which allow for more reliable and precise timings
 * of device functions.
 * 
 * @param func A function to be wrapped around by cuda event timing functions. Might be neccesary to 
 *             wrap this function in a lamda as its passed into this function.
 */
float timeGPU(auto func) {
	
	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);

	func();

	cudaEventRecord(finish, 0);

	// Synchronize events
	cudaEventSynchronize(start);
	cudaEventSynchronize(finish);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, finish);

	return elapsedTime/1000; // to give time in seconds 
}
