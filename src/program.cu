#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>

__global__ void AddInts(int32_t* a, int32_t* b, int32_t count)
{
	int32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < count) {
		a[id] += b[id];
	}
}
int main( void ) {
	cudaError err = cudaSuccess;

	srand(time(NULL));
	const int16_t count = 100;
	int32_t* h_a = new int32_t[count];
	int32_t* h_b = new int32_t[count];
	for (int32_t i = 0; i < count; i++)
	{
		h_a[i] = rand() % 1000;
		h_b[i] = rand() % 1000;
	}
	std::cout << "Prior to addition : " << std::endl;
	for (int32_t i = 0; i < 5; i++)
	{
		std::cout << h_a[i] << " " << h_b[i] << std::endl;
	}
	int32_t* d_a,* d_b;

	err = cudaMalloc(&d_a, sizeof(int32_t) * count);
	err = cudaMalloc(&d_b, sizeof(int32_t) * count);

	err = cudaMemcpy(d_a, h_a, sizeof(int32_t) * count, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_b, h_b, sizeof(int32_t) * count, cudaMemcpyHostToDevice);

	AddInts<<<count/256 + 1, 256>>>(d_a, d_b,count);	

	err = cudaMemcpy(h_a, d_a,sizeof(int32_t) * count, cudaMemcpyDeviceToHost);
	
	for (int32_t i = 0; i < 5; i++)
	{
		std::cout << "After addition : " << h_a[i] << std::endl;
	}
	cudaFree(d_a);
	cudaFree(d_b);

	delete[] h_a;
	delete[] h_b;

	return EXIT_SUCCESS;
}