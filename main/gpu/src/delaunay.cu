#include "delaunay.h"

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
		if (verbose == true) printInfo();
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

		prepForInsertTime = timeGPU([this] () { prepForInsert(); });
		if (verbose == true) printInfo();
		if (info    == true) {
			printf("============== [%d] PREP FOR INSERT ------ ==============\n", i);
			cudaMemcpy(nTriWithInsert, nTriWithInsert_d, sizeof(int), cudaMemcpyDeviceToHost);
			printf("    No. of triangles with points to insert: %d\n", *nTriWithInsert);
			printf("    time: %f\n", prepForInsertTime);
		}

		insertTime = timeGPU([this] () { insert(); });
		if (verbose == true) printInfo();
		if (info == true) {
			printf("============== [%d] INSERT --------------- ==============\n", i);
			printf("    time: %f\n", insertTime);
		}

		if (saveHistory == true) { saveToFile(); }

		//flipTime = timeGPU([this] () { flipAfterInsert(); });
		//flipTime = timeGPU([this, &numConfigsFlipped] () { numConfigsFlipped = flipAfterInsert(); });
		//flipTime = timeGPU([this] () { bruteFlip(); });
		flipTime = timeGPU([this, &numConfigsFlipped] () { numConfigsFlipped = bruteFlip(); });
		numConfigsFlippedTot += numConfigsFlipped;
		if (verbose == true) printInfo();
		if (info == true) {
			printf("============== [%d] FLIP ----------------- ==============\n", i);
			printf("    No. of configurations flipped: %d\n", numConfigsFlipped);
			printf("    time: %f\n", flipTime);
		}

		updatePtsTime = timeGPU([this] () { updatePointLocations(); });
		if (verbose == true) printInfo();
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


	//cudaDeviceSynchronize();
	if (delaunayCheck() > 0) {
		if (info == true) {
			printf("Attempting to perform additional flips\n");
		}

		//cudaDeviceSynchronize();
		bruteFlip();
		//cudaDeviceSynchronize();
		delaunayCheck();
	}

	//cudaDeviceSynchronize();
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
		printf("Total run time of host code   : %f\n", totalCPUTime);
		printf("Total run time of all code____: %f\n", totalRuntime);

	}

	if (saveHistory == true) {
		fprintf(csvfile, "ntpb,npts,nTriMax,totalRuntime,totalCPUTime,totalGPUTime,prepForInsertTimeTot,insertTimeTot,flipTimeTot,updatePtsTimeTot,seed,distribution,\n");
		fprintf(csvfile, "%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%d,%d,\n", 
		  ntpb, (*npts), (*nTriMax), totalRuntime, totalCPUTime, totalGPUTime, prepForInsertTimeTot, insertTimeTot, flipTimeTot, updatePtsTimeTot, seed, distribution);
	}

	//if (inserted_out) {
}

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
 */
void Delaunay::constructor(Point* points, int n) {
	// ============= INITIALIZE FILES TO SAVE DATA TO ============  

	if (saveHistory == true) {
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
	}


	// ============= DEVICE INFO ==================

	struct cudaDeviceProp device;
	int numDevices = 0;


	if (info == true) {
		printf("[DEVICE INFO]\n");
		printf("Found %d CUDA-enabled devices\n", numDevices);

		cudaGetDeviceProperties(&device, 0);
		printf("GPU compute capability: %d.%d\n", device.major, device.minor);
		printf("GPU model name: %s\n"           , device.name);

		printf("GPU total global memory: %lf GB\n"                       ,  (double)(device.totalGlobalMem) * 1e-9);
		printf("GPU Shared memory available per block in GB: %lf\n"      ,  (double)(device.sharedMemPerBlock) * 1e-9);
		printf("GPU 32-bit registers available per block: %d\n"                    , device.regsPerBlock);
		printf("GPU L2 cache size: %d bytes\n"                                     , device.l2CacheSize);

		printf("\n");
	}

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

	for (int i=0; i<(*npts); i++) {
		pts[i] = points[i];
	}

	// setting default values
	*nTriWithInsert = 0;
	for (int i=0; i<(*nTriMax); ++i) {
		triList[i].flip = -1;
		triList[i].insert = 1;
		triList[i].insertPt = -1;
		triList[i].insertPt_dist = (float)(unsigned long)-1;
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

	cudaMalloc(&nTriToFlip_d, sizeof(int));
	cudaMalloc(&triToFlip_d, (*nTriMax) * sizeof(int));

	// counters
	cudaMalloc(&nTriWithInsert_d, sizeof(int));

	// copying exitsting info to gpu
	cudaMemcpy(pts_d           , pts            , (*npts)    * sizeof(Point), cudaMemcpyHostToDevice);
	cudaMemcpy(npts_d          , npts           ,              sizeof(int)  , cudaMemcpyHostToDevice);

	cudaMemcpy(triList_d       , triList        , (*nTriMax) * sizeof(Tri)  , cudaMemcpyHostToDevice);
	cudaMemcpy(nTriMax_d       , nTriMax        ,              sizeof(int)  , cudaMemcpyHostToDevice);
	cudaMemcpy(triWithInsert_d , triWithInsert  , (*nTriMax) * sizeof(int)  , cudaMemcpyHostToDevice);
	cudaMemcpy(nTriWithInsert_d, nTriWithInsert ,              sizeof(int)  , cudaMemcpyHostToDevice);


	//printf("Total global memory used: %f GB \n", ((float)totMemAlloc_onDev) * 1e-9 );

	// ============= INITIALIZE ============ 

	initSuperTri();

    // save points data to trifile
	fprintf(trifile, "%d\n", (*npts) + 3);
	for (int i=0; i<(*npts) + 3; ++i) {
		fprintf(trifile, "%f %f\n", pts[i].x[0], pts[i].x[1]);
	}
	fprintf(trifile, "\n");

	saveToFile();

	// ============= COMPUTE ============ 
	compute();

	saveToFile(true);
}

Delaunay::~Delaunay() {

	if (saveHistory == true) {
		fclose(trifile);
		fclose(csvfile);
		fclose(insertedPerIterfile);
		fclose(flipedPerIterfile); 
	}

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

	cudaFree(nTriToFlip_d);
	cudaFree(triToFlip_d);
	
	free(triList); 
	free(pts);
	free(ptToTri);
	free(triToFlip);
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

	int valsToAdd = 2*(*npts); 
	dim3 threadsPerBlock1(ntpb);
	dim3 numBlocks1(valsToAdd/threadsPerBlock1.x + (!(valsToAdd % threadsPerBlock1.x) ? 0:1));

	sumPoints<<<numBlocks1, threadsPerBlock1>>>(pts_d, npts_d, avgPoint_d);

	cudaMemcpy(avgPoint, avgPoint_d, sizeof(Point), cudaMemcpyDeviceToHost);

	avgPoint->x[0] /= npts[0];
	avgPoint->x[1] /= npts[0];

	// computing the largest distance bewtween two points
	float largest_dist[1] = {0};
	float *largest_dist_d;
	cudaMalloc(&largest_dist_d, sizeof(float));
	cudaMemcpy(largest_dist_d, largest_dist, sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemset(largest_dist_d, 0, sizeof(float));

	int ncomps = ((*npts)*((*npts) - 1)) / 2; //number of comparisons
	dim3 threadsPerBlock2(ntpb);
	dim3 numBlocks2(ncomps/threadsPerBlock2.x + (!(ncomps % threadsPerBlock2.x) ? 0:1));
	if (numBlocks2.x > 65535) { printf("GRID SIZE TOO BIG\n"); }
	//computeMaxDistPts<<<numBlocks2, threadsPerBlock2>>>(pts_d, npts_d, largest_dist_d);
	cudaMemcpy(largest_dist, largest_dist_d, sizeof(float), cudaMemcpyDeviceToHost);
	*largest_dist = 2;


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
	int N;

	N = *nTriMax;
	dim3 threadsPerBlock0(ntpb);
	dim3 numBlocks0(N/threadsPerBlock0.x + (!(N % threadsPerBlock0.x) ? 0:1));

	
	//cudaDeviceSynchronize();
	resetInsertPtInTris<<<numBlocks0, threadsPerBlock0>>>(triList_d, nTriMax_d);

	N = *npts;
	dim3 threadsPerBlock1(ntpb);
	dim3 numBlocks1(N/threadsPerBlock1.x + (!(N%threadsPerBlock1.x) ? 0:1));

	//cudaDeviceSynchronize();
	setInsertPtsDistance<<<numBlocks1, threadsPerBlock1>>>(pts_d, npts_d, triList_d, ptToTri_d);

	//cudaDeviceSynchronize();
	setInsertPts<<<numBlocks1, threadsPerBlock1>>>(pts_d, npts_d, triList_d, ptToTri_d);

	N = *nTri;
	dim3 threadsPerBlock2(ntpb);
	dim3 numBlocks2(N/threadsPerBlock2.x + (!(N%threadsPerBlock2.x) ? 0:1));

	cudaMemset(nTriWithInsert_d, 0, sizeof(int));
	//cudaDeviceSynchronize();
	prepTriWithInsert<<<numBlocks2, threadsPerBlock2>>>(triList_d, nTri_d, triWithInsert_d, nTriWithInsert_d);

	//cudaMemset(nTriWithInsert_d, 0, sizeof(int));

	// save innsert num to file for each iteration
	cudaMemcpy(nTriWithInsert, nTriWithInsert_d, sizeof(int), cudaMemcpyDeviceToHost);
	fprintf(insertedPerIterfile, "%d\n", (*nTriWithInsert));

	//cudaDeviceSynchronize();
	gpuSort(triWithInsert_d, nTriMax_d); 

	//cudaDeviceSynchronize();
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
				//atomicExch(&(triList[idxTri].insertPt), idx);
				triList[idxTri].insertPt = idx;

			}
		}
	}
}

__global__ void prepTriWithInsert(Tri* triList, int* nTri, int* triWithInsert, int* nTriWithInsert) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTri)) {
		// "append" index of triange to 'triWithInsert' if insert=true
		int insert_stat = triList[idx].insert && (triList[idx].insertPt != -1);

		// write -1 if insert is false and write the index number if insert is true
		triWithInsert[idx] = idx*(insert_stat == 1) + (-(insert_stat == 0));
		atomicAdd(nTriWithInsert, insert_stat);
	}
}

__global__ void checkInsertPoint(Tri* triList, int* triWithInsert, int* nTriWithInsert) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTriWithInsert)) {
		int triIdx = triWithInsert[idx];
		if (triList[triIdx].insertPt == -1) {
			//triList[triIdx].insert = 0;
			triList[triIdx].insert = -1;
			triWithInsert[idx] = -1;
			atomicSub(nTriWithInsert, 1);
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
	dim3 threadsPerBlock(ntpb);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));

	cudaMemcpy(triList_prev_d, triList_d, (*nTriMax) * sizeof(Tri), cudaMemcpyDeviceToDevice);

	//cudaDeviceSynchronize();
	insertKernel<<<numBlocks, threadsPerBlock>>>(triList_d, nTri_d, nTriMax_d, triWithInsert_d, nTriWithInsert_d, ptToTri_d);

	//cudaDeviceSynchronize();
	updateNbrsAfterIsertKernel<<<numBlocks, threadsPerBlock>>>(triList_d, triWithInsert_d, nTriWithInsert_d, nTri_d, triList_prev_d);
 
	//cudaDeviceSynchronize();
	// update number of triangles in triList
	arrayAddVal<<<1, 1>>>(nTri_d        , nTriWithInsert_d, 2, 1); 
	arrayAddVal<<<1, 1>>>(nptsInserted_d, nTriWithInsert_d, 1, 1); 
	
	// reset triWithInsert_d for next iteraiton
	cudaMemset(triWithInsert_d, -1, (*nTri) * sizeof(int));
}

/*
 * Inserts points in parallel into triangles marked for insertion.
 */
__global__ void insertKernel(Tri* triList, int* nTri, int* nTriMax, int* triWithInsert, int* nTriWithInsert, int* ptToTri) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTriWithInsert)) {
		int triIdx = triWithInsert[idx];
		if (triIdx < 0) printf("idx %d, triIdx %d\n", idx, triIdx);
		if ((*nTri) + 2*idx >= (*nTriMax)) printf("(*nTri) + 2*idx: %d\n", (*nTri) + 2*idx);
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

__global__ void updateNbrsAfterIsertKernel(Tri* triList, int* triWithInsert, int* nTriWithInsert, int* nTri, Tri* triList_prev) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (*nTriWithInsert)) {
		int triIdx = triWithInsert[idx];
		//updateNbrsAfterIsert(triIdx, triList, (*nTri) + 2*idx);
		updateNbrsAfterIsert_wprev(triIdx, triList, (*nTri) + 2*idx, triList_prev);
	}
}

__device__ void updateNbrsAfterIsert(int i, Tri* triList, int newTriIdx) {

	// ==================== updates neighbour points opposite point if they exist ==================== 
	int nbrnbr[3] = {i, newTriIdx, newTriIdx + 1};
	int n[3] = {(triList[nbrnbr[0]].n[1] + 1) % 3, (triList[nbrnbr[1]].n[1] + 1) % 3, (triList[nbrnbr[2]].n[1] + 1) % 3};

	for (int k=0; k<3; ++k) {
		int mvInsNbr[3] = {(triList[nbrnbr[0]].o[1] + 1) % 3, (triList[nbrnbr[1]].o[1] + 1) % 3, (triList[nbrnbr[2]].o[1] + 1) % 3};

		if (n[k] >= 0) { // if nbr exist, this will almost allways happen with large number of points
			if (triList[n[k]].insert & (triList[n[k]].insertPt != -1)) {
				int idxNbr_k = n[k];

				// move anticlockwise in the nbr tri which just was inserted into 
				for (int i=0; i<mvInsNbr[k]; ++i) {
					idxNbr_k = triList[idxNbr_k].n[2];
				}

				triList[idxNbr_k].o[1] = 0;
				triList[idxNbr_k].n[1] = nbrnbr[k];

			} else { 
			//} if (triList[n[k]].insert == false) { // nbr was not marked for insertion and is the same from prev state
				triList[n[k]].o[mvInsNbr[k]] = 0;
				triList[n[k]].n[mvInsNbr[k]] = nbrnbr[k];
			}
		}
	}
}

__device__ void updateNbrsAfterIsert_wprev(int i, Tri* triList, int newTriIdx, Tri* triList_prev) {

	int n[3] = {triList_prev[i].n[0],
				triList_prev[i].n[1],
				triList_prev[i].n[2]};

	int o[3] = {triList_prev[i].o[0],
				triList_prev[i].o[1],
				triList_prev[i].o[2]};

	// ==================== updates neighbour points opposite point if they exist ==================== 

	// ==================== updates neighbour points opposite point if they exist ==================== 
	int nbrnbr[3] = {i, newTriIdx, newTriIdx + 1};

	for (int k=0; k<3; ++k) {
		int mvInsNbr = (o[k] + 1) % 3;

		//printf("insert point: %d, triidx: %d | n[%d]= %d\n", r, i, k, n[k]);
		
		if (n[k] >= 0) { // if nbr exist, this will almost allways happen with large number of points
			if (triList[n[k]].insert & (triList[n[k]].insertPt != -1)) {
				int idxNbr_k = n[k];

				// move anticlockwise in the nbr tri which just was inserted into 
				for (int i=0; i<mvInsNbr; ++i) {
					idxNbr_k = triList[idxNbr_k].n[2];
				}

				triList[idxNbr_k].o[1] = 0;
				triList[idxNbr_k].n[1] = nbrnbr[k];

			} else { 
				triList[n[k]].o[mvInsNbr] = 0;
				triList[n[k]].n[mvInsNbr] = nbrnbr[k];
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
	int N;

	N = *npts;
	dim3 threadsPerBlock2(ntpb);
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

	// definitely can optimize this part, reduce thread divergence
	if (idx < (*npts)) {
		if (ptToTri[idx] >= 0) { // picks out points not already inserted
			// for each triangle check if point is contained inside it
			for (int t=0; t<(*nTri); ++t) {
				if (contains(t, idx, triList, pts) == 1) {
					//printf("TRI: %d contains POINT: %d\n", t, idx);
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
	int not_contained = 0;

	for (i=0; i<3; ++i) {
		j = (i+1) % 3;
		// area = area of triangle (21.3.2) (21.3.10) 
		area = (pts[triList[t].p[j]].x[0] - pts[triList[t].p[i]].x[0])*(pts[r].x[1] - pts[triList[t].p[i]].x[1]) - 
			   (pts[triList[t].p[j]].x[1] - pts[triList[t].p[i]].x[1])*(pts[r].x[0] - pts[triList[t].p[i]].x[0]);

		not_contained = not_contained | (area <= 0.0);
	}

	//printf("for t: %d, and r: %d, contains spits out: %d and not_contained is %d\n", t, r, (-(not_contained == 1)) + (not_contained == 0), not_contained);
	return (-(not_contained == 1)) + (not_contained == 0);

}

/* ===================================== FLIPPING ========================================= */

int Delaunay::flipAfterInsert() {
	int numConfigsFlipped = 0;
	cudaMemcpy(nTri, nTri_d, sizeof(int), cudaMemcpyDeviceToHost);

	*nTriToFlip = 0;
	cudaMemcpy(nTriToFlip_d, nTriToFlip, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(triToFlip_d, -1, (*nTriMax) * sizeof(int));
	checkFlipAndLegality();
	checkFlipConflicts();
	cudaMemcpy(nTriToFlip, nTriToFlip_d, sizeof(int), cudaMemcpyDeviceToHost);

	// storing which triangles we should flip in this iteration
	int flipIter = 0;
	while ((*nTriToFlip) > 0) {
		int N = *nTriToFlip;
		dim3 threadsPerBlock1(ntpb);
		dim3 numBlocks1(N/threadsPerBlock1.x + (!(N % threadsPerBlock1.x) ? 0:1));

		numConfigsFlipped += N;
		if (info == true) {
			printf("    [Performing flip iteration %d]\n"     , flipIter++);
			printf("        Flipping %d configurations\n"     , N);
			printf("        Number of threads per block: %d\n", threadsPerBlock1.x);
			printf("        Number of blocks: %d\n"           , numBlocks1.x);
		}

		flipKernel<<<numBlocks1, threadsPerBlock1>>>(triToFlip_d, nTriToFlip_d, triList_d);

		saveToFile();

		*nTriToFlip = 0;
		cudaMemcpy(nTriToFlip_d, nTriToFlip, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(triToFlip_d, -1, (*nTriMax) * sizeof(int));
		checkFlipAndLegality();
		checkFlipConflicts();
		cudaMemcpy(nTriToFlip, nTriToFlip_d, sizeof(int), cudaMemcpyDeviceToHost);

	}

	int N = *nTri;
	dim3 threadsPerBlock2(ntpb);
	dim3 numBlocks2(N/threadsPerBlock2.x + (!(N % threadsPerBlock2.x) ? 0:1));
	resetTriToFlip<<<numBlocks2, threadsPerBlock2>>>(triList_d, nTri_d);

	return numConfigsFlipped;
}


/* ================================== STEP 1 ==================================  */ 

void Delaunay::checkFlipAndLegality() {
	int N;

	N = *nTri;
	dim3 threadsPerBlock0(ntpb);
	dim3 numBlocks0(N/threadsPerBlock0.x + (!(N % threadsPerBlock0.x) ? 0:1));
	 
	//printf("N: %d\n", N);
	//printf("threadsPerBlock0: %d, numBlocks0: %d\n", threadsPerBlock0.x, numBlocks0.x);
	checkFlipKernel<<<numBlocks0, threadsPerBlock0>>>(triToFlip_d, nTriToFlip_d, triList_d, nTri_d, pts_d);
	gpuSort(triToFlip_d, nTriMax_d);
	
	cudaMemcpy(nTriToFlip, nTriToFlip_d, sizeof(int), cudaMemcpyDeviceToHost);

	N = *nTriToFlip;
	dim3 threadsPerBlock1(ntpb);
	dim3 numBlocks1(N/threadsPerBlock1.x + (!(N % threadsPerBlock1.x) ? 0:1));

	checkLegalityKernel<<<numBlocks0, threadsPerBlock0>>>(triToFlip_d, nTriToFlip_d, triList_d, nTri_d, pts_d);
	gpuSort(triToFlip_d, nTriMax_d);
}

/* 
 * Performs Flip checks on each triangle currently in triList.
 */
__global__ void checkFlipKernel(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri, Point* pts) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x; // idx of triangle

	if (idx < (*nTri)) {
		int a = idx;
		int flip_edge = triList[a].flip;                     // edge to flip in a 
		int b = triList[a].n[flip_edge];                     // neighbour across edge 

		int flip = checkFlip(a, flip_edge, b, triList);

		if (flip) {
			triToFlip[a] = a;
			atomicAdd(nTriToFlip, 1);
		}
	}
}

/*
 * Checks if a triangle is marked for flipping has an eligible neibour to flip with. 
 */
__device__ int checkFlip(int a, int flip_edge, int b, Tri* triList) { 
	int noflip = 0;

	noflip = noflip | (flip_edge == -1); // is it marked for not flipping?
	noflip = noflip | (b == -1);         // does its neighbour exist?

	if (noflip) {
		triList[a].flip = -1;
	} //else {
//		triList[a].flip = 1;
//	}

	return !noflip;
}

/* 
 * Performs Legality checks on each triangle previously marked for flipping.
 */
__global__ void checkLegalityKernel(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri, Point* pts) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x; // idx of triangle

	if (idx < (*nTriToFlip)) {
		int a = triToFlip[idx];

		int flip_edge = triList[a].flip; // edge to flip in a 
		int b = triList[a].n[flip_edge]; // neighbour across edge 

		//printf("idx: %d, a: %d, flip_edge: %d, b: %d\n", idx, a, flip_edge, b);

		int flip = checkLegality(a, flip_edge, b, triList, pts);

		if (!flip) {
			//triList[a].flip = -1;
			triToFlip[idx] =  -1;
			atomicSub(nTriToFlip, 1);
		}
	}
}

	
//
///* 
// * Performs Legality checks on each triangle previously marked for flipping.
// */
//__global__ void checkLegalityKernel(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri, Point* pts) {
//	int idx = blockIdx.x*blockDim.x + threadIdx.x; // idx of triangle
//
//	if (idx < (*nTriToFlip)) {
//		int a = triToFlip[idx];
//		int flip; // 'boolean' for whether to flip this edge or not
//		int flip_edge = 0; // index of edge to flip across in triangle a
//
//		// for each edge of the triangle check whether it should be flipped,
//		// and if so mark this edge for flipping
//		for (int edge=0; edge<3; ++edge) {
//			int b = triList[a].n[edge]; // neighbour across edge 
//
//			flip = checkLegality(a, edge, b, triList, pts);
//
//			if (flip) {
//				//printf("Found egde %d to flip in tri %d\n", edge, a);
//				flip_edge = edge;
//				break;
//			}
//		}
//
//		triList[a].flip = flip_edge;
//
//		if (!flip) {
//			triList[a].flip = -1;
//			triToFlip[idx] =  -1;
//			atomicSub(nTriToFlip, 1);
//		}
//	}
//}


/*
 * Checks whether a triangle marked for flipping should be flipped. Returns 0 if the edge should
 * not be flipped (edge is legal), and returns 1 if the edge should be flipped (edge is illegal).
 */
__device__ int checkLegality(int a, int flip_edge, int b, Tri* triList, Point* pts) { 
	int opp_idx = triList[a].o[flip_edge]; // index of opposite point in neighbour

	if (flip_edge == -1) {
		// this shouldnt happen
		printf("PASSING ILLEGAL EDGE TO CHECKLEGALITY");
		return 0;
	}

	int noflip = (0 > incircle(pts[triList[b].p[opp_idx]],
							   pts[triList[a].p[0      ]],
							   pts[triList[a].p[1      ]],
							   pts[triList[a].p[2      ]] ));

//	if (noflip == 1) {
//		triList[a].flip = -1;
//	}

	return !noflip;
}

/* ================================== STEP 2 ==================================  */ 

void Delaunay::checkFlipConflicts() {
	int N;

	N = *nTri;
	dim3 threadsPerBlock(ntpb);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));

	prepForConflicts       <<<numBlocks, threadsPerBlock>>>(triList_d, nTri_d);
	setConfigIdx           <<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, nTri_d);
	storeNonConflictConfigs<<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, nTri_d);

	gpuSort(triToFlip_d, nTriMax_d);

	resetTriToFlipThisIter<<<numBlocks, threadsPerBlock>>>(triList_d, nTri_d);
	markTriToFlipThisIter<<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d);
}

/*
 * Sets the configuration index of each triangle in triList with a default value, here the largest value it 
 * can be in this iteration.
 */
__global__ void prepForConflicts(Tri* triList, int* nTri) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < (*nTri)) {
		triList[idx].configIdx = (*nTri);
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

__global__ void storeNonConflictConfigs(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < (*nTriToFlip)) {
		int a = triToFlip[idx];
		int b = triList[a].n[ triList[a].flip ];

		// checks whether the index of both triangles of the configuration hold the 
		// same configuration index, if so they are chosen to be flipped, else not flipped
		int flip = 0;

		flip = (a == min(a, b)                    &
		        triList[a].configIdx == min(a, b) &
			    triList[b].configIdx == min(a, b));
	
		// subtract the newly realised unflippable configurations
		if (!flip) { // overwrite the previous checks if the triangle is not to be flipped this round
			triToFlip[idx] = -1;
			atomicSub(nTriToFlip, 1);
		}

		//atomicSub(nTriToFlip, (flip == 0));
	}
}

/*
 * Resets the 'flipThisIter' (flip this iteration) field in the 'Tri' struct to -1
 * so that the followig kernel can mark the triangles which have been set for insertion
 * an iteraion of flipping.
 */
__global__ void resetTriToFlipThisIter(Tri* triList, int* nTri) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < (*nTri)) {
		triList[idx].flipThisIter = -1;
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

		triList[a].flipThisIter = 1;
		triList[b].flipThisIter = 1;
		triList[b].flip = ((triList[a].o[flip_edge] + 1) % 3); // lets the flipping companion know where its going to flip
	}
}
/* ================================== STEP 3 ==================================  */ 

__global__ void flipKernel(int* triToFlip, int* nTriToFlip, Tri* triList) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < (*nTriToFlip)) {
		int a = triToFlip[idx];
		int flip_edge = triList[a].flip; // edge of triangle to flip across
		int b = triList[a].n[ flip_edge ]; // b is the index of the triangle across the edge

		//printf("idx: %d | flip pair: (%d, %d)\n", idx, a, b);
		flip(a, flip_edge, b, triList);

		// disabled for updateNbrsAfterFlip
		//triList[a].flipThisIter = -1;
		//triList[b].flipThisIter = -1;

		// disabled for brute flip
		//triList[a].flip = 1;
		//triList[b].flip = 0;
	}
}

__global__ void flipKernel(int* triToFlip, int* nTriToFlip, Tri* triList, Quad* quadList) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < (*nTriToFlip)) {
		int a = triToFlip[idx];
		int flip_edge = triList[a].flip; // edge of triangle to flip across
		int b = triList[a].n[ flip_edge ]; // b is the index of the triangle across the edge

		//printf("idx: %d | flip pair: (%d, %d)\n", idx, a, b);
		flip(a, flip_edge, b, triList, &(quadList[idx]));

		// disabled for updateNbrsAfterFlip
		//triList[a].flipThisIter = -1;
		//triList[b].flipThisIter = -1;

		// disabled for brute flip
		//triList[a].flip = 1;
		//triList[b].flip = 0;
	}
}

/*
 * Performs a flip operation on a triangle 'a' and one of its edges/neighbours 'e' denoted by
 * index 0, 1 or 2. Returns 1 if the flip was performed, reuturns -1 if no flip was 
 * performed.
 * 
 * @param a Index in triList of chosen triangle
 * @param e Index in chosen triangle of edge/neigbour. This is an int in 0, 1 or 2.
 */
__device__ void flip(int a, int e, int b, Tri* triList) {
	
	int opp_idx = triList[a].o[e]; // index in neighbour of point opposite the edge marked for flipping in 'a'.

 	// temporary qaud "struct" data just to make it readable
	int p[4] = {triList[a].p[(e-1 + 3)%3], triList[a].p[e]                , triList[b].p[opp_idx], triList[a].p[(e + 1)%3]};
	int n[4] = {triList[a].n[(e-1 + 3)%3], triList[b].n[(opp_idx-1 + 3)%3], triList[b].n[opp_idx], triList[a].n[(e + 1)%3]};
	int o[4] = {triList[a].o[(e-1 + 3)%3], triList[b].o[(opp_idx-1 + 3)%3], triList[b].o[opp_idx], triList[a].o[(e + 1)%3]};

	// making sure nbr and opposite points are good and correcting then if not
	for (int k=0; k<4; ++k) {
		if (triList[n[k]].flipThisIter == 1) {
			// the neighbouring flip will be performed in the direction of this flip, SCREAM
			if (triList[n[k]].flip == ((o[k] + 1) % 3)) {
				printf("CONFIG IDX %d | NEIGHBOURING TRIANGLE (%d) WILL FLIP INTO THIS CONFIGURATION (%d, %d) | CONFIG IDX of nbr %d \n", triList[a].configIdx, n[k], a, b, triList[n[k]].configIdx);
				//printf("IDX %d | a: (%f, %f), b: (%f, %f)\n", a, triList[a].p[0].x[0], triList[a].x[1], triList[b].x[0], triList[b].x[1]);
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

	int ap[3] = {p[0], p[1], p[2]};
	int an[3] = {n[0], n[1], b   };
	int ao[3] = {o[0], o[1], 1   };

	int bp[3] = {p[2], p[3], p[0]};
	int bn[3] = {n[2], n[3], a   };
	int bo[3] = {o[2], o[3], 1   };

	writeTri(&(triList[a]), ap, an, ao);
	writeTri(&(triList[b]), bp, bn, bo);


	int nbrs[4] = {a, a, b, b};
	int opps[4] = {2, 0, 2, 0};

	for (int k=0; k<4; ++k) {
		int mvIdx = (o[k] + 1) % 3;
		// n[k] is almost always >= 0 with large number of points
		if (n[k] >= 0)  {
			triList[n[k]].n[mvIdx] = nbrs[k];
			triList[n[k]].o[mvIdx] = opps[k];
		}
	}
}



/* 
 * One with quad structs for splitting the eupdate nbr step
 */
__device__ void flip(int a, int e, int b, Tri* triList, Quad* quad) {
	
	int opp_idx = triList[a].o[e]; // index in neighbour of point opposite the edge marked for flipping in 'a'.

 	// temporary qaud "struct" data just to make it readable
	int p[4] = {triList[a].p[(e-1 + 3)%3], triList[a].p[e]                , triList[b].p[opp_idx], triList[a].p[(e + 1)%3]};
	int n[4] = {triList[a].n[(e-1 + 3)%3], triList[b].n[(opp_idx-1 + 3)%3], triList[b].n[opp_idx], triList[a].n[(e + 1)%3]};
	int o[4] = {triList[a].o[(e-1 + 3)%3], triList[b].o[(opp_idx-1 + 3)%3], triList[b].o[opp_idx], triList[a].o[(e + 1)%3]};

	/* ===================================== THIS SHIT IS FUNKY =============================================================*/ 

	// making sure nbr and opposite points are good and correcting then if not
	for (int k=0; k<4; ++k) {
		if (triList[n[k]].flipThisIter == 1) {
			// the neighbouring flip will be performed in the direction of this flip, SCREAM
			if (triList[n[k]].flip == ((o[k] + 1) % 3)) {
				printf("CONFIG IDX %d | NEIGHBOURING TRIANGLE (%d) WILL FLIP INTO THIS CONFIGURATION (%d, %d) | CONFIG IDX of nbr %d \n", triList[a].configIdx, n[k], a, b, triList[n[k]].configIdx);
				//printf("IDX %d | a: (%f, %f), b: (%f, %f)\n", a, triList[a].p[0].x[0], triList[a].x[1], triList[b].x[0], triList[b].x[1]);
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

	// write to trilistNext instead of triList, then swap pointers
	/* ===================================== THIS SHIT IS FUNKY =============================================================*/ 

	// writeQuad
	for (int i=0; i<4; ++i) { 
		//quad->p[i] = p[i];	
		quad->n[i] = n[i];	
		quad->o[i] = o[i];	
	}

	//printf("flip | base a: %d | n : %d, %d, %d, %d\n", a, n[0], n[1], n[2], n[3]);
	//printf("flip | base a: %d | o : %d, %d, %d, %d\n", a, o[0], o[1], o[2], o[3]);

	// is a cause of the divergent threads that the rewriting step is performed multiple times?

	int ap[3] = {p[0], p[1], p[2]};
	int an[3] = {n[0], n[1], b   };
	int ao[3] = {o[0], o[1], 1   };

	int bp[3] = {p[2], p[3], p[0]};
	int bn[3] = {n[2], n[3], a   };
	int bo[3] = {o[2], o[3], 1   };

	writeTri(&(triList[a]), ap, an, ao);
	writeTri(&(triList[b]), bp, bn, bo);


//	int nbrs[4] = {a, a, b, b};
//	int opps[4] = {2, 0, 2, 0};
//
//	for (int k=0; k<4; ++k) {
//		int mvIdx = (o[k] + 1) % 3;
//		// n[k] is almost always >= 0 with large number of points
//		if (n[k] >= 0)  {
//			triList[n[k]].n[mvIdx] = nbrs[k];
//			triList[n[k]].o[mvIdx] = opps[k];
//		}
//	}
}

void Delaunay::quadFlip() {
	cudaMemcpy(nTriToFlip, nTriToFlip_d, sizeof(int), cudaMemcpyDeviceToHost);

	int N = *nTriToFlip;
	dim3 threadsPerBlock(ntpb);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));

	Quad* quadList_d;
	cudaMalloc(&quadList_d, (*nTriToFlip) * sizeof(Quad));

	writeQuadKernel          <<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, quadList_d);
	flipFromQuadKernel       <<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, quadList_d);
	updateNbrsAfterFlipKernel<<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, quadList_d);

	cudaFree(quadList_d);
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
		if (triList[n[k]].flipThisIter == 1) {
			// the neighbouring flip will be performed in the direction of this flip, SCREAM
			if (triList[n[k]].flip == ((o[k] + 1) % 3)) {
				printf("CONFIG IDX %d | NEIGHBOURING TRIANGLE (%d) WILL FLIP INTO THIS CONFIGURATION (%d, %d) | CONFIG IDX of nbr %d \n", triList[a].configIdx, n[k], a, b, triList[n[k]].configIdx);
				//printf("IDX %d | a: (%f, %f), b: (%f, %f)\n", a, triList[a].p[0].x[0], triList[a].x[1], triList[b].x[0], triList[b].x[1]);
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

// ========== DELAUNAY CHECK =========

/* 
 * Looks into each triangle and checks each of its edges for whether it should 
 * be flipped or not. Returns the number of edges that need flipping.
 */
int Delaunay::delaunayCheck() {
	cudaMemcpy(nTri, nTri_d, sizeof(int), cudaMemcpyDeviceToHost);

	int N = *nTri;
	dim3 threadsPerBlock(ntpb);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));

	int nEdges[1] = {0};
	int *nEdges_d;
	cudaMalloc(&nEdges_d, sizeof(int)); 
	cudaMemcpy(nEdges_d, nEdges, sizeof(int), cudaMemcpyHostToDevice);
	delaunayCheckKernel<<<numBlocks, threadsPerBlock>>>(triList_d, nTri_d, pts_d, nEdges_d);

	cudaMemcpy(nEdges, nEdges_d, sizeof(int), cudaMemcpyDeviceToHost);
	if ((*nEdges) > 0) {

		if (info == true) {
			printf("\nTriangulation is NOT Delaunay, with %d illegal edges\n\n", *nEdges);
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
			if (b != -1) {
				edges_to_flip += (0 < checkLegality(a, edge, b, triList, pts));
			}
		}

		atomicAdd(nEdges, edges_to_flip);
	}
}


// ====================  BRUTE FORCE FLIP =================== 


// check every triangle for an edge to flip
//     if it has an edge to flip mark it in ".flip" and add the index of the triangle to the triToFlip array
// check conflicts
// flipKernel

int Delaunay::bruteFlip() {
	int numConfigsFlipped = 0;
	int N;
	cudaMemcpy(nTri, nTri_d, sizeof(int), cudaMemcpyDeviceToHost);

	*nTriToFlip = 0;
	cudaMemcpy(nTriToFlip_d, nTriToFlip, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(triToFlip_d, -1, (*nTriMax) * sizeof(int));
//
//		N = *nTri;
//		dim3 threadsPerBlock0(ntpb);
//		dim3 numBlocks0(N/threadsPerBlock0.x + (!(N % threadsPerBlock0.x) ? 0:1));
//	resetTriToFlipThisIter<<<numBlocks0, threadsPerBlock0>>>(triList_d, nTri_d);
//
	//cudaDeviceSynchronize();
	checkIncircleAll();

	//cudaDeviceSynchronize();
	checkFlipConflicts();

	cudaMemcpy(nTriToFlip, nTriToFlip_d, sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemset(nTriToFlip_d, 0, sizeof(int));

//	if (saveHistory == true) {
//		fprintf(flipedPerIterfile, "%d %d\n", -1, -1);
//	}

	int flipIter = 0;
	while ((*nTriToFlip) > 0) {
		N = *nTriToFlip;
		dim3 threadsPerBlock1(ntpb);
		dim3 numBlocks1(N/threadsPerBlock1.x + (!(N % threadsPerBlock1.x) ? 0:1));

		numConfigsFlipped += N;
		flipIter++;
		if (info == true) {
			printf("    [Performing flip iteration %d]\n"     , flipIter);
			printf("        Flipping %d configurations\n"     , N);
			//printf("        Number of threads per block: %d\n", threadsPerBlock1.x);
			//printf("        Number of blocks: %d\n"           , numBlocks1.x);
			printf("        Iter : %d\n"                      , iter);

//			cudaMemcpy(triToFlip, triToFlip_d, (*nTriMax) * sizeof(int), cudaMemcpyDeviceToHost);
//			printf("Flipping configurations with index:  ");
//			for (int k=0; k<(*nTriToFlip); ++k) {
//				printf("%d ", triToFlip[k]);
//			}
//			printf("\n");
		}

		//cudaDeviceSynchronize();
		if (saveHistory == true) {
			fprintf(flipedPerIterfile, "%d ", N);
		}

		quadFlip();

//		Quad* quadList_d;
//		cudaMalloc(&quadList_d, (*nTriMax) * sizeof(Quad));
		//cudaMemset(quadList_d, -1, sizeof(Quad));

		//cudaDeviceSynchronize();


//		flipKernel<<<numBlocks1, threadsPerBlock1>>>(triToFlip_d, nTriToFlip_d, triList_d, quadList_d);
//
//		//cudaDeviceSynchronize();
//		updateNbrsAfterFlipKernel<<<numBlocks1, threadsPerBlock1>>>(triToFlip_d, nTriToFlip_d, triList_d, quadList_d);

//		cudaFree(quadList_d);
		
//		N = *nTri;
//		dim3 threadsPerBlock2(ntpb);
//		dim3 numBlocks2(N/threadsPerBlock2.x + (!(N % threadsPerBlock2.x) ? 0:1));
//		resetTriToFlipThisIter<<<numBlocks2, threadsPerBlock2>>>(triList_d, nTri_d);

		saveToFile();

		*nTriToFlip = 0;
		cudaMemcpy(nTriToFlip_d, nTriToFlip, sizeof(int), cudaMemcpyHostToDevice);
		//cudaMemset(nTriToFlip_d, 0, sizeof(int));
		cudaMemset(triToFlip_d, -1, (*nTriMax) * sizeof(int));

		//cudaDeviceSynchronize();
		checkIncircleAll();

		//cudaDeviceSynchronize();
		checkFlipConflicts();

		cudaMemcpy(nTriToFlip, nTriToFlip_d, sizeof(int), cudaMemcpyDeviceToHost);
	}

	if (saveHistory == true) {
		if ( flipIter == 0) {
			fprintf(flipedPerIterfile, "%d ", 0);
		}
		fprintf(flipedPerIterfile, "\n");
		// should move back to beginning of text block
		//fseek(flipedPerIterfile, -9 - 5*(flipIter), SEEK_CUR);
		// write lenght
		fprintf(flipedPerIterfile, "%d %d\n", flipIter, numConfigsFlipped);
		// go back to original position
		//fseek(flipedPerIterfile, 9 + 5*(flipIter), SEEK_CUR);
		fprintf(flipedPerIterfile, "\n");
	}

	N = *nTri;
	dim3 threadsPerBlock2(ntpb);
	dim3 numBlocks2(N/threadsPerBlock2.x + (!(N % threadsPerBlock2.x) ? 0:1));
	//cudaDeviceSynchronize();
	resetTriToFlip<<<numBlocks2, threadsPerBlock2>>>(triList_d, nTri_d);

	return numConfigsFlipped;
}

void Delaunay::checkIncircleAll() {
	cudaMemcpy(nTri, nTri_d, sizeof(int), cudaMemcpyDeviceToHost);
	int N = *nTri;
	dim3 threadsPerBlock(ntpb);
	dim3 numBlocks(N/threadsPerBlock.x + (!(N % threadsPerBlock.x) ? 0:1));
	//for (int edge=0; edge<3; ++edge) {
	checkIncircleAllKernel<<<numBlocks, threadsPerBlock>>>(triToFlip_d, nTriToFlip_d, triList_d, nTri_d, pts_d);
	//}

	//cudaDeviceSynchronize();
	gpuSort(triToFlip_d, nTriMax_d);
}

/*
 * Performs an incricle check on each edge of each triangle currently in the triangulation
 * and sets '.flip' of each triagle to be an edge which should be flipped if there are any.
 */
//__global__ void checkIncircleAllKernel(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri, Point* pts) {
//	int idx = blockIdx.x*blockDim.x + threadIdx.x;
//	//int edge_to_flip = 0;
//
//	// for each triangle
//	if (idx < (*nTri)) {
//
//		int edge;
//		int a = idx;
//		int nsucces = 0;
//		//int flip_edges[3];
//
//		// for each edge
//		for (int e=0; e<3; ++e) {
//			edge = e;
//			int b = triList[a].n[ edge ]; // b is the index of the triangle across the edge
//
//			if (b != -1) { //true most of the time, nbr exists most of the time
//				int opp_idx = triList[a].o[edge]; // index of opposite point in neighbour
//				int flip = (0 < incircle(pts[triList[b].p[opp_idx]],
//										 pts[triList[a].p[0      ]],
//										 pts[triList[a].p[1      ]],
//										 pts[triList[a].p[2      ]] ));
//
//				// if one of the edges should be flipped, mark it and exit
//				nsucces += flip;
//			}
//
//		triList[a].flip = nsucces*(nsucces > 0) - (nsucces == 0);
//		triToFlip[a] = a*(nsucces > 0) - (nsucces == 0);
//		atomicAdd(nTriToFlip, (nsucces > 0));
//
//		}
//	}
//}

__global__ void checkIncircleAllKernel(int* triToFlip, int* nTriToFlip, Tri* triList, int* nTri, Point* pts) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	// for each triangle
	if (idx < (*nTri)) {

		int edge;
		int a = idx;

		// for each edge
		for (int e=0; e<3; ++e) {
			edge = e;
			int b = triList[a].n[ edge ]; // b is the index of the triangle across the edge

			if (b != -1) { //true most of the time, nbr exists most of the time
				int opp_idx = triList[a].o[edge]; // index of opposite point in neighbour
				int flip = (0 < incircle(pts[triList[b].p[opp_idx]],
										 pts[triList[a].p[0      ]],
										 pts[triList[a].p[1      ]],
										 pts[triList[a].p[2      ]] ));

				// if one of the edges should be flipped, mark it and exit
				if (flip) {
					triList[a].flip = edge;
					triToFlip[a] = a;
					atomicAdd(nTriToFlip, 1);
					break;
				}
			}
		}
	}
}

/* ===================================== PRINTING ========================================= */

void Delaunay::printInfo() {
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

	printf("\nPRINTING TRI INFO\n");

	for (int i=0; i<(*nTri); ++i) {
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

/* ====================================== SAVE TO FILE ============================================= */

void Delaunay::saveToFile(bool end) {
	//cudaDeviceSynchronize();
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

	// Record events around kernel launch
	cudaEventRecord(start, 0); // We use 0 here because it is the "default" stream

	func();

	cudaEventRecord(finish, 0);

	// Synchronize events
	cudaEventSynchronize(start); // This is optional, we shouldn't need it
	cudaEventSynchronize(finish); // This isn't - we need to wait for the event to finish

	// Calculate time
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, finish);

	return elapsedTime/1000; // to give time in seconds 

}
