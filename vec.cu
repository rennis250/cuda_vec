// #include <iostream>

#include <optix.h>
#include <optix_device.h>
#include <vector_types.h>

#include "rob_optix.h"
#include "launch_params.h"
#include "vec.h"

extern "C"
{
	__constant__ Params params;
}

//int main() {
//	int nvecs = 100;
//	size_t vec_size = nvecs * sizeof(vec3);
//
//	vec3 *xs, *ys;
//	cudaMallocManaged(&xs, vec_size);
//	cudaMallocManaged(&ys, vec_size);
//
//	for (int i = 0; i < nvecs; i++) {
//		xs[i] = vec3(1.0f);
//		ys[i] = vec3(0.0f);
//	}
//
//	norm_vecs<<<1, 1>>>(nvecs, xs, ys);
//	
//	// Wait for GPU to finish before accessing on host
//	cudaDeviceSynchronize();
//
//	printf("This first vec is %f\n", xs[0].x);
//	printf("This first normed vec is %f\n", ys[0].x);
//
//	cudaFree(xs);
//	cudaFree(ys);
//
//	return 0;
//}