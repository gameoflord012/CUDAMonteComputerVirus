#include <iostream>
#include <random>
#include <stdlib.h>
#include <time.h>
#include <sstream>
#include <chrono>

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
const unsigned int DEFAULT_BLOCKS_COUNT_PER_numSMs = 32;

const unsigned int INITIAL_INFECTED_COUNT = 1;

__device__ float d_probs[20];

__global__
void uniform_dis(curandState_t* states, unsigned int* result, float p)
{
    float prob = curand_uniform(states + threadIdx.x);
    atomicAdd(result, prob < p);
}

__global__
void monte_simp(curandState_t* states, unsigned int* counters, size_t* result, size_t n)
{
    unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = threadId; i < n; i += gridDim.x * blockDim.x)
    {
        int infectedCount = INITIAL_INFECTED_COUNT;

        do
        {
            atomicAdd(result, (size_t)1);

            counters[threadId] = 0;
            uniform_dis << <1, 20 >> > (states + threadId * 20, counters + threadId, d_probs[infectedCount]);
            infectedCount += counters[threadId];
        } while ((infectedCount -= 5) > 0);
    }
}

__global__
void init_kernel_states(curandState_t* states, int seed)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, i, 0, states + i);
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
    // Get inputs
    const size_t tests_count = get_unsigned_num("tests_count: ", DEFAULT_TESTS_COUNT);
    const size_t blocks_count_per_numSMs = get_unsigned_num("blocks_count_per_numSMs: ", DEFAULT_BLOCKS_COUNT_PER_numSMs);
    const size_t threads_count_per_block = get_unsigned_num("threads_per_block: ", DEFAULT_THREADS_COUNT_PER_BLOCK);
    
    // Get numSMs and device
    int numSMs, device;
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    const unsigned int threads_count = numSMs * blocks_count_per_numSMs * threads_count_per_block;

    // Print datas
    cout << endl;
    cout << "Datas:" << endl;
    cout << "   blocks_count_per_numSMs = " << blocks_count_per_numSMs << endl;
    cout << "   threads_count_per_block = " << threads_count_per_block << endl;
    cout << "   tests_count = " << tests_count << endl;
    cout << "   threads_count = " << threads_count << endl;
    cout << endl;

    // Init kernels
    srand(time(NULL));

    curandState_t* d_states;
    CUDA_CALL(cudaMalloc((void**)&d_states, 20 * threads_count * sizeof(curandState)));

    clock_t tStart = clock();
    cout << "Start kernel \"init_kernel_states()\"" << endl;
    init_kernel_states << <numSMs * blocks_count_per_numSMs * 20, threads_count_per_block >> > (d_states, rand());
    CUDA_CALL(cudaThreadSynchronize());
    printf("Time taken: %.2fs\n\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    unsigned int* counters;
    CUDA_CALL(cudaMalloc((void**)&counters, threads_count * sizeof(unsigned int)));

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
    cout << "Start kernel \"monte_simp()\"..." << endl;
    tStart = clock();
    monte_simp << <numSMs * blocks_count_per_numSMs, threads_count_per_block >> > (d_states, counters, d_A, tests_count);
    CUDA_CALL(cudaThreadSynchronize());
    printf("Time taken: %.2fs\n\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    // print result from device memory
    size_t h_A;
    CUDA_CALL(cudaMemcpy(&h_A, d_A, sizeof(size_t), cudaMemcpyDeviceToHost));

    cout << "Results: " << endl;
    printf("    %llu / %llu = %.2f\n", h_A, tests_count, (float)(h_A) / tests_count);

    // Free memory
    cudaFree(d_states);
    cudaFree(d_A);

    // Check for any errors
    cudaCheckErrors("Don't know wtf");
}