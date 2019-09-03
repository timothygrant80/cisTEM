/*
 * Histogram.cu
 *
 *  Created on: Aug 29, 2019
 *      Author: himesb
 */

#include "gpu_core_headers.h"

__global__ void histogram_final_accum(const unsigned int *in, int n, unsigned int *out);

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
//	if (is_allocated_histogram_buffer)
//	{
//		checkCudaErrors(cudaFree(histogram_buffer));
//	}
//	if (is_allocated_histogram)
//	{
//		checkCudaErrors(cudaFree(histogram));
//	}
//	if (is_allocated_histogram_bin_values)
//	{
//		checkCudaErrors(cudaFree(histogram_bin_values));
//	}
}
	//FIXME

void Histogram::Init(int histogram_n_bins, float histogram_min, float histogram_step)
{

	this->histogram_n_bins 	= histogram_n_bins;
	this->histogram_min 	= histogram_min;
	this->histogram_step	= histogram_step;
	wxPrintf("\n\tInit histogram with vals\n");

}

void Histogram::SetInitialValues()
{

	is_allocated_histogram_buffer = false;
	int histogram_n_bins 	= 0;
	float histogram_min 	= 0.0f;
	float histogram_step	= 0.0f;
}


void Histogram::BufferInit(NppiSize npp_ROI)
{

	wxPrintf("\n\tInit histogram buffers vals %d %d\n",npp_ROI.width, npp_ROI.height);

	int n_elem;
    checkNppErrors(nppiHistogramRangeGetBufferSize_32f_C1R(npp_ROI, histogram_n_bins + 1, &n_elem));
    wxPrintf("Function asks for a min buffer of n elemen %d\n",n_elem);
    n_elem = 1490400;
    checkCudaErrors(cudaMalloc(&histogram_buffer, n_elem));

    is_allocated_histogram_buffer = true;
    checkCudaErrors(cudaMalloc(&histogram, histogram_n_bins*sizeof(Npp32s)));
    checkCudaErrors(cudaMalloc(&cummulative_histogram, histogram_n_bins*sizeof(Npp32s)));
    // could bring in the context and then put this to an async op
    checkCudaErrors(cudaMemset(histogram, 0 , (histogram_n_bins)*sizeof(Npp32s)));
    checkCudaErrors(cudaMemset(cummulative_histogram, 0 , (histogram_n_bins)*sizeof(Npp32s)));
    is_allocated_histogram = true;

    Npp32f* h_histogram_bin_values;
    checkCudaErrors(cudaMallocHost(&h_histogram_bin_values,(histogram_n_bins+1)*sizeof(Npp32f)));
    checkCudaErrors(cudaMalloc(&histogram_bin_values,(histogram_n_bins+1)*sizeof(Npp32f)));
    // Fill in the bin values;
    for (int iBin = 0; iBin < 1 + histogram_n_bins; iBin ++)
    {
    	h_histogram_bin_values[iBin] = (Npp32f)(histogram_min + (iBin * histogram_step));
//    	wxPrintf("Hist value %d is %f\n",iBin,h_histogram_bin_values[iBin]);
    }
    checkCudaErrors(cudaMemcpy(histogram_bin_values, h_histogram_bin_values, (histogram_n_bins+1)*sizeof(Npp32f),cudaMemcpyHostToDevice));
    is_allocated_histogram_bin_values = true;

    checkCudaErrors(cudaFreeHost(h_histogram_bin_values));


    vector_ROI.width = histogram_n_bins;
    vector_ROI.height = 1;

}



void Histogram::AddToHistogram(GpuImage &input_image )
{


    if ( ! is_allocated_histogram_buffer )
    {
    	input_image.NppInit();
    	BufferInit(input_image.npp_ROI);
    }

    MyAssertTrue(input_image.is_in_memory_gpu, "The image to add to the histogram is not in gpu memory.");

    // FIXME this function breaks under mutli-threading across multi gpus.
    checkNppErrors(nppiHistogramRange_32f_C1R_Ctx((const Npp32f*)input_image.real_values_gpu,
											  (int)input_image.pitch,
											  (NppiSize)input_image.npp_ROI,
											  (Npp32s *)histogram,
											  (const Npp32f*)histogram_bin_values,
											  (int)(histogram_n_bins + 1),
											  (Npp8u *)histogram_buffer, input_image.nppStream));

//    for (int iBin = 0; iBin < histogram_n_bins; iBin ++)
//    {
//    	cummulative_histogram[iBin] += histogram[iBin];
//    }
    checkNppErrors(nppsAdd_32s_ISfs_Ctx((const Npp32s*)histogram, (Npp32s*)cummulative_histogram, (int)histogram_n_bins, (int)0, input_image.nppStream));

}

void Histogram::CopyToHostAndAdd(long* array_to_add_to)
{
	// Make a temporary copy of the cummulative histogram on the host and then add on the host. TODO errorchecking
	long* tmp_array;
	checkCudaErrors(cudaMallocHost(&tmp_array, histogram_n_bins*sizeof(Npp32s)));
	checkCudaErrors(cudaMemcpy((void *)tmp_array, (void *)cummulative_histogram,histogram_n_bins*sizeof(Npp32s),cudaMemcpyDeviceToHost));

	for (int iBin = 0; iBin < histogram_n_bins; iBin++)
	{
		array_to_add_to[iBin] += tmp_array[iBin];
	}

	checkCudaErrors(cudaFreeHost(&tmp_array));
}


