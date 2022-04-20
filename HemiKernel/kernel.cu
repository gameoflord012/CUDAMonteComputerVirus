#include <iostream>
#include <random>
#include <stdlib.h>
#include <time.h>
#include <sstream>
#include <chrono>

#include "hemi/hemi.h"
#include "hemi/launch.h"
#include "hemi/grid_stride_range.h"
#include "hemi/execution_policy.h"

//for __syncthreads()
#include <device_functions.h>

#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>

#define DEBUG

using namespace std;

const size_t DEFAULT_TESTS_COUNT = 10;

const unsigned int INITIAL_INFECTED_COUNT = 1;

__device__ float d_probs[20];

HEMI_LAUNCHABLE
void monte_simp(size_t* result, size_t n)
{
    curandState_t s;
    curand_init(clock64(), hemi::globalThreadIndex(), 0, &s);

    for (auto i : hemi::grid_stride_range<size_t>(0, n))
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

    int numSMs;
    checkCuda(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device_idx));

    const size_t tests_count = get_unsigned_num("tests_count: ", DEFAULT_TESTS_COUNT);
    const size_t threads_count_per_block = get_unsigned_num("threads_per_block: ", 0);
    const size_t blocks_count_per_numSMs = get_unsigned_num("blocks_count_per_numSMs: ", 0);

    // Print datas
    cout << endl;
    cout << "Datas:" << endl;
    cout << "   device_idx = " << device_idx << endl;
    cout << "   tests_count = " << tests_count << endl;
    cout << "   threads_count_per_block = " << (threads_count_per_block ? to_string(threads_count_per_block) : "Hemi decide") << endl;
    cout << "   blocks_count_per_numSMs = " << (blocks_count_per_numSMs ? to_string(blocks_count_per_numSMs) : "Hemi decide") << endl;
    cout << "   numSMs = " << numSMs << endl;
    cout << endl;

    // Initialize d_A
    size_t* d_A;
    checkCuda(cudaMalloc((void**)&d_A, sizeof(size_t)));
    checkCuda(cudaMemset(d_A, 0, sizeof(size_t)));

    // Calculate probs
    float h_probs[20];
    float currentProb = 1.;
    for (int i = 0; i < 20; i++)
    {
        h_probs[i] = 1. - currentProb;
        currentProb *= .9;
    }

    checkCuda(cudaMemcpyToSymbol(d_probs, h_probs, 20 * sizeof(float)));

    // Run simulation kernel
    cout << "Running kernel \"monte_simp()\"..." << endl;
    clock_t tStart = clock();

    hemi::ExecutionPolicy ep;
    if (blocks_count_per_numSMs) ep.setGridSize(blocks_count_per_numSMs * numSMs);
    if (threads_count_per_block) ep.setBlockSize(threads_count_per_block);

    hemi::cudaLaunch(monte_simp, d_A, tests_count);

    checkCuda(cudaThreadSynchronize());
    printf("Time taken: %.2fs\n\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    // print result from device memory
    size_t h_A;
    checkCuda(cudaMemcpy(&h_A, d_A, sizeof(size_t), cudaMemcpyDeviceToHost));

    cout << "Results: " << endl;
    printf("    A) %llu / %llu = %.2f\n", h_A, tests_count, (float)(h_A) / tests_count);

    // Free memory
    checkCuda(cudaFree(d_A));

    // Check for any errors
    checkCudaErrors();
}