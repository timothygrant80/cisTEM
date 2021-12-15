/*
 * Histogram.cu
 *
 *  Created on: Aug 29, 2019
 *      Author: himesb
 */

#include "gpu_core_headers.h"


__global__ void histogram_smem_atomics(const __half* in, int4 dims, float *out, int n_bins, const __half bin_min, const __half bin_inc, const int max_padding);


__global__ void histogram_smem_atomics(const  __half* in, int4 dims, float *out, int n_bins, const __half bin_min, const __half bin_inc, const int max_padding)
{
  // pixel coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // grid dimensions
  int nx = blockDim.x * gridDim.x;
  int ny = blockDim.y * gridDim.y;

  // linear thread index within 2D block
  int t = threadIdx.x + threadIdx.y * blockDim.x;

  // total threads in 2D block
  int nt = blockDim.x * blockDim.y;

  // initialize temporary accumulation array in shared memory
  extern __shared__  int smem[];
  for (int i = t; i < n_bins; i += nt) smem[i] = 0;
  __syncthreads();

  int pixel_idx;
  // process pixels
  // updates our block's partial histogram in shared memory
  for (int col = x; col < dims.x - max_padding ; col += nx)
  {
	if (col < max_padding) continue;
    for (int row = y; row < dims.y*dims.z - max_padding; row += ny)
    {
      if (row < max_padding) continue;
//      pixel_idx = (int)(floor((in[row * dims.w + col]-bin_min) / bin_inc));
      pixel_idx = __half2int_rd((in[row * dims.w + col]-bin_min) / bin_inc);

	  // use nvidia integer instrinic max/min
      //pixel_idx = max(min(pixel_idx,n_bins-1),0);
	  if (pixel_idx >= 0 && pixel_idx < n_bins)
	  {
   	    atomicAdd(&smem[pixel_idx], 1);	
	  }

    }
  }
  __syncthreads();

  // write partial histogram into the global memory
  // Converting to long was super slow. Given that I don't care about representing the number exactly,
  // but do care about overflow, just switch the bins to flaot
  out += (blockIdx.x + blockIdx.y * gridDim.x) * n_bins;
  for (int i = t; i < n_bins; i += nt) {
    out[i] += (float)smem[i];
  }
}

__global__ void histogram_final_accum(float *in, float *out, int n_bins, int n_blocks);


__global__ void histogram_final_accum(float *in, float *out, int n_bins, int n_blocks)
{



	int lIDX = blockIdx.x * blockDim.x + threadIdx.x;

	if (lIDX < n_bins)
	{
		float total = 0.0f;
		for (int j = 0; j < n_blocks; j ++)
		{
		  total += in[lIDX + n_bins * j];
		}
		out[lIDX] += total;

	}

}


Histogram::Histogram()
{
	SetInitialValues();
	wxPrintf("\n\tInit histogram\n");
}

Histogram::Histogram(int histogram_n_bins, float histogram_min, float histogram_step)
{

	SetInitialValues();
	Init(histogram_n_bins, histogram_min, histogram_step);

}

Histogram::~Histogram()
{

	if (is_allocated_histogram)
	{
		cudaErr(cudaFree(histogram))
		cudaErr(cudaFree(cummulative_histogram));
	}

}
	//FIXME

void Histogram::Init(int histogram_n_bins, float histogram_min, float histogram_step)
{

	this->histogram_n_bins 	= histogram_n_bins;
	this->histogram_min 	= __float2half(histogram_min);
	this->histogram_step	= __float2half(histogram_step);
	this->max_padding = 2;

}

void Histogram::SetInitialValues()
{
	is_allocated_histogram = false;
	histogram_n_bins 	= 0;
	histogram_min 	= (__half)0.0;
	histogram_step	= (__half) 0.0;
}


void Histogram::BufferInit(NppiSize npp_ROI)
{


	// Set up grids for the kernels
	 threadsPerBlock_img = dim3(32, 32, 1);
	 gridDims_img = dim3((npp_ROI.width + threadsPerBlock_img.x - 1) / threadsPerBlock_img.x,
						 (npp_ROI.height+ threadsPerBlock_img.y - 1) / threadsPerBlock_img.y,1);

	 threadsPerBlock_accum_array = dim3(32, 1, 1);
	 gridDims_accum_array = dim3((histogram_n_bins + threadsPerBlock_accum_array.x - 1) / threadsPerBlock_accum_array.x,1,1);


	 size_of_temp_hist = (gridDims_img.x*gridDims_img.y*histogram_n_bins*sizeof(int));


	 // Array of temporary storage to accumulate the shared mem to
	cudaErr(cudaMalloc(&histogram, size_of_temp_hist));
	cudaErr(cudaMalloc(&cummulative_histogram, histogram_n_bins*sizeof(int)));


	// could bring in the context and then put this to an async op
	cudaErr(cudaMemset(histogram, 0 ,size_of_temp_hist));
	cudaErr(cudaMemset(cummulative_histogram, 0 , (histogram_n_bins)*sizeof(int)));


	is_allocated_histogram = true;


}

void Histogram::AddToHistogram(GpuImage &input_image)
{


    MyAssertTrue(input_image.is_in_memory_gpu, "The image to add to the histogram is not in gpu memory.");



	precheck
	histogram_smem_atomics<<< gridDims_img,threadsPerBlock_img, (histogram_n_bins)*sizeof(int), input_image.nppStream.hStream>>>((const __half*)input_image.real_values_16f, input_image.dims, histogram, histogram_n_bins,histogram_min,histogram_step,max_padding);
	postcheck


}

void Histogram::Accumulate(GpuImage &input_image)
{
	cudaErr(cudaStreamSynchronize(input_image.nppStream.hStream));
	precheck
	histogram_final_accum<<< gridDims_accum_array,threadsPerBlock_accum_array, 0, input_image.nppStream.hStream>>>(histogram, cummulative_histogram, histogram_n_bins,gridDims_img.x*gridDims_img.y);
	postcheck
}

void Histogram::CopyToHostAndAdd(long* array_to_add_to)
{

	// Make a temporary copy of the cummulative histogram on the host and then add on the host. TODO errorchecking
	float* tmp_array;
	cudaErr(cudaMallocHost(&tmp_array, histogram_n_bins*sizeof(float)));
	cudaErr(cudaMemcpy(tmp_array, this->cummulative_histogram,histogram_n_bins*sizeof(float),cudaMemcpyDeviceToHost));

	for (int iBin = 0; iBin < histogram_n_bins; iBin++)
	{
		array_to_add_to[iBin] += (long)tmp_array[iBin];

	}

	cudaErr(cudaFreeHost(tmp_array));
}





