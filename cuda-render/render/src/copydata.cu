// #include <helper_cuda.h>
#include <stdio.h>

// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val + __longlong_as_double(assumed)));
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
	}while (assumed != old);
	return __longlong_as_double(old);
}
#endif

// 4 float values, rgba, where a is the distance factor
texture<float4, 2, cudaReadModeElementType> inTex2;

__global__ void cudaProcess2(float *data_txhxwx3, int timebin, int imgh,
		int imgw, int sz, int maxdepth, int mindepth) {

	/////////////////////////////////////////////////////////////////
	// index
	// heiidx * (sz * width) + wididx

	int IMGH = sz * imgh;
	int IMGW = sz * imgw;

	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;

	int wididx = presentthread % IMGW;
	int heiidx = (presentthread - wididx) / IMGW;
	if (heiidx >= IMGH || wididx >= IMGW) {
		return;
	}

	//////////////////////////////////////////////////////////////
	// 1) what;s the corresponding value?
	float4 res = tex2D(inTex2, wididx, heiidx);
	float r = res.x;
	float g = res.y;
	float b = res.z;
	float dv = res.w;

	float ress[4];
	ress[0] = r;
	ress[1] = g;
	ress[2] = b;
	ress[3] = dv;

	if (dv > 1.0f / (1.0f + 0.0f + 1e-2f) || dv < 1.0f / (1.0f + 6.0f - 1e-2f))
		return;

	//////////////////////////////////////////////////////////////////////
	// depth: depthvalue = 1 / (1 + depth)
	float depth = 1.0f / dv - 1.0f;

	float tidx = depth * 100;
	int tidxleft = static_cast<int>(std::floor(tidx));
	int tidxright = tidxleft + 1;

	if (tidxright >= maxdepth)
		return;

	////////////////////////////////////////////////////////////
	// 2) which place should it be added
	wididx = wididx % imgw;
	heiidx = heiidx % imgh;

	// notice that for ogl, height begins from bottom to top
	heiidx = imgh - 1 - heiidx;

	///////////////////////////////////////////////////////////////////////
	// interpolation
	int idxleft = tidxleft * imgh * imgw * 3 + heiidx * imgw * 3 + wididx * 3;
	int idxright = tidxright * imgh * imgw * 3 + heiidx * imgw * 3 + wididx * 3;
	// linear

	// linear intepolation
	float weightleft = tidxright - tidx;
	float weightright = tidx - tidxleft;
	for (int kc = 0; kc < 3; kc++) {
		// data_txhxwx3[idxleft + kc] += weightleft * res[kc];
		// data_txhxwx3[idxright + kc] += weightright * res[kc];
		atomicAdd(data_txhxwx3 + idxleft + kc, weightleft * ress[kc]);
		atomicAdd(data_txhxwx3 + idxright + kc, weightright * ress[kc]);
	}
}

__global__ void cudaProcess3(float *data_hxwx3, int imgh, int imgw, int sz) {

	/////////////////////////////////////////////////////////////////
	// index
	// heiidx * width + wididx

	int IMGH = imgh;
	int IMGW = imgw;

	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;

	int wididx = presentthread % IMGW;
	int heiidx = (presentthread - wididx) / IMGW;
	if (heiidx >= IMGH || wididx >= IMGW) {
		return;
	}

	/////////////////////////////////////////////////////////////////
	int totalidx1 = heiidx * imgw + wididx;
	int totalidx3 = totalidx1 * 3;

	////////////////////////////////////////////////////////////////
	// each thread will touch sz * sz data

	// inverse height
	heiidx = imgh - 1 - heiidx;
	for (int i = 0; i < sz; i++)
		for (int j = 0; j < sz; j++) {
			int hbig = i * imgh + heiidx;
			int wbig = j * imgw + wididx;

			//////////////////////////////////////////////////////////////
			// 1) what;s the corresponding value?
			float4 res = tex2D(inTex2, wbig, hbig);
			float r = res.x;
			float g = res.y;
			float b = res.z;

			data_hxwx3[totalidx3 + 0] += r;
			data_hxwx3[totalidx3 + 1] += g;
			data_hxwx3[totalidx3 + 2] += b;
		}
}

extern "C" void launch_cudaProcess2(cudaArray *g_data_array,
		float *g_output_data, int timebin, int imgh, int imgw, int sz,
		int maxdepth, int mindepth) {

	/////////////////////////////////////////////////////////////
	// for input
	// g_data_array
	// it is a (sz*imgh)*(sz*imgw)*4 image
	// for output
	// g_output_data
	// it is a timebin*imgh*imgw*3 image

	/////////////////////////////////////////////////////////////
	// threadnum
	const int totalthread = sz * imgh * sz * imgw;
	const int threadnum = 1024;
	const int blocknum = static_cast<int>(std::ceil(
			1.0f * totalthread / threadnum));

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);

	/////////////////////////////////////////////////////////////
	// checkCudaErrors(cudaBindTextureToArray(inTex2, g_data_array));
	cudaBindTextureToArray(inTex2, g_data_array);

	struct cudaChannelFormatDesc desc;
	// checkCudaErrors(cudaGetChannelDesc(&desc, g_data_array));
	cudaGetChannelDesc(&desc, g_data_array);

	/*
	 printf("CUDA Array channel descriptor, bits per component:\n");
	 printf("X %d Y %d Z %d W %d, kind %d\n", desc.x, desc.y, desc.z, desc.w,
	 desc.f);

	 printf("Possible values for channel format kind: i%d, u%d, f%d:\n",
	 cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned,
	 cudaChannelFormatKindFloat);

	 printf("%d\n", inTex2);
	 */

	cudaProcess2<<<blocks, threads>>>(g_output_data, timebin, imgw, imgh, sz,
			maxdepth, mindepth);

}

extern "C" void launch_cudaProcess3(cudaArray *g_data_array,
		float *g_output_data, int imgh, int imgw, int sz) {

/////////////////////////////////////////////////////////////
// for input
// g_data_array
// it is a (sz*imgh)*(sz*imgw)*4 image
// for output
// g_output_data
// it is a imgh*imgw*3 image

/////////////////////////////////////////////////////////////
// threadnum
	const int totalthread = imgh * imgw;
	const int threadnum = 1024;
	const int blocknum = static_cast<int>(std::ceil(
			1.0f * totalthread / threadnum));

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);

/////////////////////////////////////////////////////////////
// checkCudaErrors(cudaBindTextureToArray(inTex2, g_data_array));
	cudaBindTextureToArray(inTex2, g_data_array);

	struct cudaChannelFormatDesc desc;
// checkCudaErrors(cudaGetChannelDesc(&desc, g_data_array));
	cudaGetChannelDesc(&desc, g_data_array);

	/*
	 printf("CUDA Array channel descriptor, bits per component:\n");
	 printf("X %d Y %d Z %d W %d, kind %d\n", desc.x, desc.y, desc.z, desc.w,
	 desc.f);

	 printf("Possible values for channel format kind: i%d, u%d, f%d:\n",
	 cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned,
	 cudaChannelFormatKindFloat);

	 printf("%d\n", inTex2);
	 */

	cudaProcess3<<<blocks, threads>>>(g_output_data, imgw, imgh, sz);

}

