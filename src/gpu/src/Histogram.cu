/*
 * Histogram.cu
 *
 *  Created on: Aug 29, 2019
 *      Author: himesb
 */

#include "gpu_core_headers.h"


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
	if (is_allocated_histogram_buffer)
	{
		checkCudaErrors(cudaFree(histogram_buffer));
	}
	if (is_allocated_histogram)
	{
		checkCudaErrors(cudaFree(histogram));
	}
	if (is_allocated_histogram_bin_values)
	{
		checkCudaErrors(cudaFree(histogram_bin_values));
	}
}

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
    checkCudaErrors(cudaMalloc(&histogram_buffer, n_elem));
    is_allocated_histogram_buffer = true;
    checkCudaErrors(cudaMallocManaged(&histogram, histogram_n_bins*sizeof(Npp32s)));
    is_allocated_histogram = true;

    float* h_histogram_bin_values;
    checkCudaErrors(cudaMallocHost(&h_histogram_bin_values,(histogram_n_bins+1)*sizeof(float)));
    checkCudaErrors(cudaMalloc(&histogram_bin_values,(histogram_n_bins+1)*sizeof(float)));
    // Fill in the bin values;
    for (int iBin = 0; iBin <= histogram_n_bins; iBin ++)
    {
    	h_histogram_bin_values[iBin] = histogram_min + (iBin * histogram_step);
//    	wxPrintf("Hist value %d is %f\n",iBin,h_histogram_bin_values[iBin]);
    }
    checkCudaErrors(cudaMemcpy(histogram_bin_values, h_histogram_bin_values, (histogram_n_bins+1)*sizeof(float),cudaMemcpyHostToDevice));
    is_allocated_histogram_bin_values = true;
    checkCudaErrors(cudaFreeHost(h_histogram_bin_values));



    vector_ROI.width = histogram_n_bins;
    vector_ROI.height = 1;

	MyPrintWithDetails("");
}

void Histogram::AddToHistogram(GpuImage &input_image)
{

	MyAssertTrue(input_image.is_npp_loaded, "The npp context is not set inside the image");

    if ( ! is_allocated_histogram_buffer )
    {
    	BufferInit(input_image.npp_ROI);

    }

    MyAssertTrue(input_image.is_in_memory_gpu, "The image to add to the histogram is not in gpu memory.");
    checkNppErrors(nppiHistogramRange_32f_C1R((const Npp32f*)input_image.real_values_gpu,
											  (int)input_image.pitch,
											  (NppiSize)input_image.npp_ROI,
											  (Npp32s *)histogram,
											  (const Npp32f*)histogram_bin_values,
											  (int)(histogram_n_bins + 1),
											  (Npp8u *)histogram_buffer));


}

void Histogram::AddToHistogram(GpuImage &input_image, Npp32s* cummulative_histogram )
{


    if ( ! is_allocated_histogram_buffer )
    {
    	input_image.NppInit();
    	BufferInit(input_image.npp_ROI);
    }
    MyPrintWithDetails("");

    MyAssertTrue(input_image.is_in_memory_gpu, "The image to add to the histogram is not in gpu memory.");

    checkNppErrors(nppiHistogramRange_32f_C1R((const Npp32f*)input_image.real_values_gpu,
											  (int)input_image.pitch,
											  (NppiSize)input_image.npp_ROI,
											  (Npp32s *)histogram,
											  (const Npp32f*)histogram_bin_values,
											  (int)(histogram_n_bins + 1),
											  (Npp8u *)histogram_buffer));

    input_image.Wait();
    MyPrintWithDetails("");
    for (int iBin = 0; iBin < histogram_n_bins; iBin ++)
    {
    	wxPrintf("Hist value %d is %d\n",iBin,cummulative_histogram[iBin]);
    	wxPrintf("Hist value %d is %d\n",iBin,histogram[iBin]);

    	cummulative_histogram[iBin] += histogram[iBin];
    }
//    checkNppErrors(nppsAdd_32s_ISfs((const Npp32s*)histogram,cummulative_histogram, histogram_n_bins, (int)0));
    MyPrintWithDetails("");

}


