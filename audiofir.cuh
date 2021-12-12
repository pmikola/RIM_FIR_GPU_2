#pragma once
#if defined __cplusplus
extern "C"
#endif
void audiofir(
	float* yout, float* yin, float* coeff, int n, int len, ...);