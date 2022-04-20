#include <iostream>
#include <random>
#include <stdlib.h>
#include <time.h>
#include <sstream>
#include <chrono>
#include<windows.h>
//for __syncthreads()
#include <device_functions.h>

#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

using namespace std;

const size_t DEFAULT_TESTS_COUNT = 100;
const unsigned int DEFAULT_THREADS_COUNT_PER_BLOCK = 128;

const unsigned int INITIAL_INFECTED_COUNT = 1;

__device__ float d_probs[20];

__global__
void monte_simp(size_t* result, size_t n)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t s;
    curand_init(clock64(), threadId, 0 ,&s);

    for (size_t i = threadId; i < n; i += gridDim.x * blockDim.x)
    {
        int infectedCount = INITIAL_INFECTED_COUNT;

        do
        {
            atomicAdd(result, (size_t)1);

            int newCasesCount = 0;
            for (int uninfectedCount = 20 - infectedCount; uninfectedCount--;)
            {
                float prob = curand_uniform(&s);
                newCasesCount += prob < d_probs[infectedCount];
            }

            infectedCount += newCasesCount;
        } while ((infectedCount -= 5) > 0);
    }
}

size_t get_unsigned_num(string msg, size_t defaultValue)
{
    cout << msg;
    string input;
    getline(cin, input);
    stringstream ss(input);
    size_t result;
    if (ss >> result)
        return result;
    return defaultValue;
}

int main()
{
    // Get devices infos
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }

    int device_idx;
    cudaGetDevice(&device_idx);
    device_idx = get_unsigned_num("device index: ", device_idx);

    cudaSetDevice(device_idx);
    // Get numSMs and device
    int numSMs;
    CUDA_CALL(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device_idx));
    
    // Get inputs
    const size_t tests_count = get_unsigned_num("tests_count: ", DEFAULT_TESTS_COUNT);
    const size_t tests_count_2 = get_unsigned_num("tests_count_2: ", DEFAULT_TESTS_COUNT);
    const size_t threads_count_per_block = get_unsigned_num("threads_per_block: ", DEFAULT_THREADS_COUNT_PER_BLOCK);

    const unsigned int default_blocks_count = (tests_count + threads_count_per_block - 1) / threads_count_per_block;
    const size_t blocks_count_per_numSMs = get_unsigned_num("blocks_count_per_numSMs: ", (default_blocks_count + numSMs - 1) / numSMs);

    const unsigned int threads_count = numSMs * blocks_count_per_numSMs * threads_count_per_block;

    // Print datas
    cout << endl;
    cout << "Datas:" << endl;
    cout << "   device_idx = " << device_idx << endl;
    cout << "   tests_count = " << tests_count << endl;
    cout << "   tests_count_2 = " << tests_count_2 << endl;
    cout << "   blocks_count_per_numSMs = " << blocks_count_per_numSMs << endl;
    cout << "   numSMs = " << numSMs << endl;
    cout << "   threads_count_per_block = " << threads_count_per_block << endl;
    cout << "   threads_count = " << threads_count << endl;
    cout << endl;

    // Initialize d_A
    size_t* d_A;
    CUDA_CALL(cudaMalloc((void**)&d_A, sizeof(size_t)));
    CUDA_CALL(cudaMemset(d_A, 0, sizeof(size_t)));

    // Calculate probs
    float h_probs[20];
    float currentProb = 1.;
    for (int i = 0; i < 20; i++)
    {
        h_probs[i] = 1. - currentProb;
        currentProb *= .9;
    }

    CUDA_CALL(cudaMemcpyToSymbol(d_probs, h_probs, 20 * sizeof(float)));

    // Run simulation kernel
    size_t h_A;
    size_t h_A_sum = 0;
    cout << "Running kernel \"monte_simp()\"..." << endl;
    clock_t tStart = clock();
    for (int i = 0; i < tests_count_2; i++)
    {
        monte_simp << <numSMs * blocks_count_per_numSMs, threads_count_per_block >> > (d_A, tests_count);
        CUDA_CALL(cudaMemcpy(&h_A, d_A, sizeof(size_t), cudaMemcpyDeviceToHost));
        h_A_sum += h_A;
        printf("  %llu / %llu / %d = %.2f ", h_A_sum, tests_count,i, (float)(h_A) / tests_count / (i+1));
        printf("ETA: %.2fs\n", ((double)(clock() - tStart) / CLOCKS_PER_SEC) * (double)(tests_count_2/(i+1)));
    }
    CUDA_CALL(cudaThreadSynchronize());
    printf("Time taken: %.2fs\n\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    
    // print result from device memory
    //CUDA_CALL(cudaMemcpy(&h_A, d_A, sizeof(size_t), cudaMemcpyDeviceToHost));

    cout << "Results: " << endl;
    printf("    A) %llu / %llu = %.2f\n", h_A_sum, tests_count * tests_count_2, (float)(h_A) / tests_count / tests_count_2);

    // Free memory
    cudaFree(d_A);

    // Check for any errors
    cudaCheckErrors("Don't know wtf");
}