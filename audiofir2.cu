// RIM_FIR_CPU.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define K 512

#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_timer.h>


__global__ static void audiofir_kernel(
	float* yout, float* yin, float* coeff, int n, int len)
{
	float output = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		for (int k = 0; k <= n; k++) {
			if (i >= k) output += yin[i - k] * coeff[k];
		}
		yout[i] = output;
	}
}


void audiofir(float* yout, float* yin, float* coeff, int n, int len, ...)
{

	float* coeffd, * yind, * youtd;
	auto M = (len + K - 1) / K;
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc(&coeffd, sizeof(float) * (n + 1)));
	checkCudaErrors(cudaMalloc(&yind, sizeof(float) * (2 * len)));
	checkCudaErrors(cudaMalloc(&youtd, sizeof(float) * (2 * len)));
	checkCudaErrors(cudaMemcpy(coeffd, coeff, sizeof(float) * (n + 1), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(yind, yin, sizeof(float) * (2 * len), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(&youtd, yout, 2 * len, cudaMemcpyHostToDevice));
	cudaEvent_t start1, stop1; // pomiar czasu wykonania jądra
	checkCudaErrors(cudaEventCreate(&start1));
	checkCudaErrors(cudaEventCreate(&stop1));
	checkCudaErrors(cudaEventRecord(start1, 0));
	audiofir_kernel << <M, K >> > (youtd, yind, coeffd, n, len);
	audiofir_kernel << <M, K >> > (youtd + len, yind + len, coeffd, n, len);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventRecord(stop1, 0));
	checkCudaErrors(cudaEventSynchronize(stop1));
	float elapsedTime;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime,
		start1, stop1));
	checkCudaErrors(cudaEventDestroy(start1));
	checkCudaErrors(cudaEventDestroy(stop1));
	checkCudaErrors(cudaDeviceSynchronize());
	printf("GPU (kernel) time = %.3f ms (%6.3f GFLOP/s)\n",
		elapsedTime,
		1e-6 * 2 * ((double)n + 1) * 2 * ((double)len) /
		elapsedTime);
	checkCudaErrors(cudaMemcpy(&yout, youtd, 2 * len, cudaMemcpyDeviceToHost));
	if (IsDebuggerPresent()) getchar();
	cudaFree(&coeffd);
	cudaFree(&yind);
	cudaFree(&youtd);
}