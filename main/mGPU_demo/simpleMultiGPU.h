/*
 * This application demonstrates how to use the CUDA API to use multiple GPUs.
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the
 * application. On the other side, you can still extend your desktop to screens
 * attached to both GPUs.
 */

#ifndef SIMPLEMULTIGPU_H
#define SIMPLEMULTIGPU_H

struct TGPUplan {
    // Host-side input data
    int    dataN;
    float *h_Data;

    // Partial sum for this GPU
    float *h_Sum;

    // Device buffers
    float *d_Data, *d_Sum;

    // Reduction copied back from GPU
    float *h_Sum_from_device;

    // Stream for asynchronous command execution
    cudaStream_t stream;

};

//extern "C" void launch_reduceKernel(float *d_Result, float *d_Input, int N, int BLOCK_N, int THREAD_N, cudaStream_t &s);

#endif
