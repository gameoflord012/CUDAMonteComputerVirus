#include <iostream>
#include <random>
#include <sstream>

#include <cuda.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <curand_kernel.h>

#include "hemi/hemi.h"

using namespace std;

const int COMPUTER_COUNT = 20;
const int TEST_COUNT = 1000;

__constant__ double d_probsTable[COMPUTER_COUNT + 1][COMPUTER_COUNT + 1];
__constant__ double d_probs[COMPUTER_COUNT + 1];

struct matrix
{
	int ar[COMPUTER_COUNT + 1];
};

const matrix INIT_MATRIX{ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} };

struct randomGenerator : public thrust::unary_function<matrix, uint64_t>
{
	int seed;
	randomGenerator(int _seed) : seed(_seed) {};

	__device__ matrix operator()(const uint64_t& x)
	{
		curandState_t s;
		curand_init(seed, x, 0, &s);
		
		matrix result;
		for (int nInfect = 0; nInfect < COMPUTER_COUNT + 1; nInfect++)
		{
			result.ar[nInfect] = nInfect;

			for (int nUninfect = COMPUTER_COUNT - nInfect; nUninfect--;)
			{
				double p = curand_uniform(&s);
				result.ar[nInfect] += p < d_probs[nInfect];
			}

			result.ar[nInfect] -= result.ar[nInfect] > 5 ? 5 : result.ar[nInfect];
		}

		return result;
	}
};

struct matrixFunctor : thrust::binary_function<matrix, matrix, matrix>
{
	__device__ matrix operator()(const matrix& a, const matrix& b)
	{
		matrix c;

		for (int i = 0; i < COMPUTER_COUNT + 1; i++)
		{
			c.ar[i] = b.ar[a.ar[i]];
		}

		return c;
	}
};

typedef thrust::transform_iterator<
	randomGenerator,
	thrust::counting_iterator<uint64_t>, matrix> tramsformIterator_t;

uint64_t getUnsigned(string msg, uint64_t defaultValue)
{
	cout << msg;
	string input;
	getline(cin, input);
	stringstream ss(input);
	uint64_t result;
	if (ss >> result)
		return result;
	return defaultValue;
}

void test_1()
{
	auto first = thrust::make_transform_iterator(
		thrust::counting_iterator<uint64_t>(0),
		randomGenerator(rand()));

	for (int i = 0; i < 10; i++)
	{
		thrust::for_each<tramsformIterator_t>(
			first + i,
			first + i + 1,
			[]__device__(const matrix & x)
		{
			for (int i = 0; i < COMPUTER_COUNT + 1; i++)
			{
				printf("%2d:%2d | ", i, x.ar[i]);
			}
			printf("\n");
		});
	}

	matrix result = thrust::reduce<tramsformIterator_t>(
		first,
		first + 10,
		INIT_MATRIX,
		matrixFunctor());

	for (int i = 0; i < COMPUTER_COUNT + 1; i++)
	{
		printf("%2d:%2d | ", i, result.ar[i]);
	}
	printf("\n");
}

int main()
{
	srand(time(NULL));

	double probsTable[COMPUTER_COUNT + 1][COMPUTER_COUNT + 1];
	double probs[COMPUTER_COUNT + 1];

	// Calc facs
	uint64_t facs[COMPUTER_COUNT + 1];
	facs[0] = 1;
	for (int i = 1; i < COMPUTER_COUNT + 1; i++)
	{
		facs[i] = facs[i - 1] * i;
	}

	
	// Calc probs
	double inv_p = 1;
	for (int n = 0; n < COMPUTER_COUNT + 1; n++)
	{
		probs[n] = (1 - inv_p);

		double check = .0;

		for (int r = 0; r <= n; r++)
		{
			probsTable[n][r] = (double)
				facs[n] / facs[r] / facs[n - r] *
				pow(.1, r) *
				pow(.9, n - r);

			check += probsTable[n][r];
		}

		inv_p *= .9;
	}

	checkCuda(cudaMemcpyToSymbol(d_probs, probs, (COMPUTER_COUNT + 1) * sizeof(double)));
	checkCuda(cudaMemcpyToSymbol(d_probsTable, probsTable, (COMPUTER_COUNT + 1) * (COMPUTER_COUNT + 1) * sizeof(double)));

	int testCnt = 0;
	testCnt = getUnsigned("testCnt: ", TEST_COUNT);
	printf("\n");
	printf("Data:\n");
	printf("	testCnt: %d\n", testCnt);

	clock_t start = clock();

	uint64_t aCnt = 0;

	for (int itest = testCnt; itest--;)
	{
		uint64_t sz = 1;

		auto first = thrust::make_transform_iterator(
			thrust::counting_iterator<uint64_t>(0),
			randomGenerator(rand()));

		auto getResult = [&first](uint64_t sz) -> int
		{
			matrix result = thrust::reduce<tramsformIterator_t>(
				first,
				first + sz,
				INIT_MATRIX,
				matrixFunctor()
				);
			return result.ar[0];
		};

		while (getResult(sz) > 0)
		{
			sz *= 2;
		}

		uint64_t l = sz / 2 + 1, r = sz;
		while (l < r)
		{
			int m = l + r >> 1;
			if (getResult(m) > 0)
				l = m + 1;
			else
				r = m;

		}

		aCnt += l;
	}

	printf("%.2f\n", (float)aCnt / testCnt);

	printf("\nTime elapsed %f in seconds\n", ((float)clock() - start) / CLOCKS_PER_SEC);
}