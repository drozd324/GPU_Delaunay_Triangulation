#ifndef MISC_H
#define MISC_H

__global__ void arrayAddVal(int* array, int* val, int mult, int n);
__global__ void sortPass(int* array, int n, int parity, int* sorted);
void gpuSort(int* array, int* n);

#endif
