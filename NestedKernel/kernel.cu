#include <iostream>
#include <random>
#include <stdlib.h>
#include <time.h>

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

const size_t EXPECTED_TEST_COUNT = 100;
const unsigned int THREADS_PER_BLOCK = 128;
const unsigned int BLOCKS_COUNT_PER_numSMs = 32;

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

int main()
{
    // Init kernels
    srand(time(NULL));

    int numSMs, device;
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

    unsigned int THREADS_COUNT = numSMs * BLOCKS_COUNT_PER_numSMs * THREADS_PER_BLOCK;
    curandState_t* d_states;
    CUDA_CALL(cudaMalloc((void**)&d_states, 20 * THREADS_COUNT * sizeof(curandState)));

    init_kernel_states << <numSMs * BLOCKS_COUNT_PER_numSMs * 20, THREADS_PER_BLOCK >> > (d_states, rand());

    unsigned int* counters;
    CUDA_CALL(cudaMalloc((void**)&counters, THREADS_COUNT * sizeof(unsigned int)));

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
    monte_simp << <numSMs * BLOCKS_COUNT_PER_numSMs, THREADS_PER_BLOCK >> > (d_states, counters, d_A, EXPECTED_TEST_COUNT);

    // print result from device memory
    size_t h_A;
    CUDA_CALL(cudaMemcpy(&h_A, d_A, sizeof(size_t), cudaMemcpyDeviceToHost));

    printf("Threads count: %d\n", THREADS_COUNT);
    printf("%llu %llu\n", h_A, EXPECTED_TEST_COUNT);
    printf("%.2f\n", (float)(h_A) / EXPECTED_TEST_COUNT);

    // Free memory
    cudaFree(d_states);
    cudaFree(d_A);

    // Check for any errors
    cudaCheckErrors("Don't know wtf");
}