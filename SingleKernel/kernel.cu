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
const unsigned int INITIAL_INFECTED_COUNT = 1;
const unsigned int THREADS_PER_BLOCK = 128;
const unsigned int BLOCKS_COUNT_PER_numSMs = 32;

__device__ float d_probs[20];

__global__
void monte_simp(curandState_t* states, size_t* result, size_t n)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = threadId; i < n; i += gridDim.x * blockDim.x)
    {
        int infectedCount = INITIAL_INFECTED_COUNT;

        do
        {
            atomicAdd(result, (size_t)1);

            int newCasesCount = 0;
            for (int uninfectedCount = 20 - infectedCount; uninfectedCount--;)
            {
                float prob = curand_uniform(&states[threadId]);
                newCasesCount += prob < d_probs[infectedCount];
            }

            infectedCount += newCasesCount;
        } while ((infectedCount -= 5) > 0);
    }
}

__global__
void curand_init_kernel(curandState_t* states, int seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, i, 0, &states[i]);
}

int main()
{
    // Init kernel
    srand(time(NULL));

    int numSMs, device;
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

    int THREADS_COUNT = numSMs * BLOCKS_COUNT_PER_numSMs * THREADS_PER_BLOCK;
    curandState_t* d_states;
    CUDA_CALL(cudaMalloc((void**)&d_states, THREADS_COUNT * sizeof(curandState)));

    curand_init_kernel << <numSMs * BLOCKS_COUNT_PER_numSMs, THREADS_PER_BLOCK >> > (d_states, rand());

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
    monte_simp<<<numSMs * BLOCKS_COUNT_PER_numSMs, THREADS_PER_BLOCK>>>(d_states, d_A, EXPECTED_TEST_COUNT);

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