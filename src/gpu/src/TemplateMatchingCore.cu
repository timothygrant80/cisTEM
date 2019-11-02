#include "gpu_core_headers.h"

#define DO_HISTOGRAM true

const unsigned int SUM_PIXELWISE_UNROLL = 1;
const unsigned int MIP_PIXELWISE_UNROLL = 1;

__global__ void  SumPixelWiseKernel(const cufftReal* correlation_output, Stats* my_stats, const int numel);
__global__ void MipPixelWiseKernel(const cufftReal*  correlation_output, Peaks* my_peaks, const int  numel,
                                   __half psi, __half theta, __half phi);


TemplateMatchingCore::TemplateMatchingCore() 
{

};

TemplateMatchingCore::TemplateMatchingCore(int number_of_jobs) 
{

  Init(number_of_jobs);


};



TemplateMatchingCore::~TemplateMatchingCore() 
{


// FIXME
//	if (is_allocated_cummulative_histogram)
//	{
//		cudaErr(cudaFree((void *)cummulative_histogram));
//		cudaErr(cudaFreeHost((void *)h_cummulative_histogram));
//	}

};

void TemplateMatchingCore::Init(int number_of_jobs)
{
  this->nThreads = 1;
	this->number_of_jobs_per_image_in_gui = 1;
  this->nGPUs = 1;


};

void TemplateMatchingCore::Init(Image &template_reconstruction,
                                Image &input_image,
                                Image &current_projection,
                                float pixel_size_search_range,
                                float pixel_size_step,
                                float pixel_size,
                                float defocus_search_range,
                                float defocus_step,
                                float defocus1,
                                float defocus2,
                                float psi_max,
                                float psi_start,
                                float psi_step,
                                AnglesAndShifts angles,
                                EulerSearch global_euler_search,
                    			float histogram_min_scaled,
                    			float histogram_step_scaled,
                    			int histogram_number_of_bins,
                    			int max_padding,
                                int first_search_position,
                                int last_search_position)
                                
{



	this->first_search_position = first_search_position;
	this->last_search_position  = last_search_position;
	this->angles = angles;
	this->global_euler_search = global_euler_search;

	this->psi_start = psi_start;
	this->psi_step  = psi_step;
	this->psi_max   = psi_max;



    // It seems that I need a copy for these - 1) confirm, 2) if already copying, maybe put straight into pinned mem with cudaHostMalloc
    this->template_reconstruction.CopyFrom(&template_reconstruction);
    this->input_image.CopyFrom(&input_image);
    this->current_projection.CopyFrom(&current_projection);

    d_input_image.Init(this->input_image);
    d_input_image.CopyHostToDevice();

    d_current_projection.Init(this->current_projection);

    d_padded_reference.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
    d_max_intensity_projection.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
    d_best_psi.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
    d_best_theta.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
    d_best_phi.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);



	d_sum1.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
	d_sumSq1.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
	d_sum2.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
	d_sumSq2.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
	d_sum3.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
	d_sumSq3.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);




    wxPrintf("Setting up the histogram\n\n");
	histogram.Init(histogram_number_of_bins, histogram_min_scaled, histogram_step_scaled);
	if (max_padding > 2) {histogram.max_padding = max_padding;}




    
    // For now we are only working on the inner loop, so no need to track best_defocus and best_pixel_size

    // At the outset these are all empty cpu images, so don't xfer, just allocate on gpuDev



    // Transfer the input image_memory_should_not_be_deallocated  

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

};

void TemplateMatchingCore::RunInnerLoop(Image &projection_filter, float c_pixel, float c_defocus, int threadIDX)
{


  

	// Make sure we are starting with zeros
	d_max_intensity_projection.Zeros();
	d_best_psi.Zeros();
	d_best_phi.Zeros();
	d_best_theta.Zeros();
	d_padded_reference.Zeros();

	d_sum1.Zeros();
	d_sumSq1.Zeros();
	d_sum2.Zeros();
	d_sumSq2.Zeros();
	d_sum3.Zeros();
	d_sumSq3.Zeros();

	this->c_defocus = c_defocus;
	this->c_pixel = c_pixel;
	total_number_of_cccs_calculated = 0;

	// Either do not delete the single precision, or add in a copy here so that each loop over defocus vals
	// have a copy to work with. Otherwise this will not exist on the second loop
	d_input_image.ConvertToHalfPrecision(false);

	cudaErr(cudaMalloc((void **)&my_peaks, sizeof(Peaks)*d_input_image.real_memory_allocated));
	cudaErr(cudaMalloc((void **)&my_stats, sizeof(Stats)*d_input_image.real_memory_allocated));

	cudaEvent_t projection_is_free_Event, gpu_work_is_done_Event;
	cudaErr(cudaEventCreateWithFlags(&projection_is_free_Event, cudaEventDisableTiming));
	cudaErr(cudaEventCreateWithFlags(&gpu_work_is_done_Event, cudaEventDisableTiming));

	int ccc_counter = 0;
	int current_search_position;
	float average_on_edge;

	int thisDevice;
	cudaGetDevice(&thisDevice);
	wxPrintf("Thread %d is running on device %d\n", threadIDX, thisDevice);

//	cudaErr(cudaFuncSetCacheConfig(SumPixelWiseKernel, cudaFuncCachePreferL1));

	bool first_loop_complete = false;

	for (current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++)
	{

		if ( current_search_position % 10 == 0)
		{
			wxPrintf("Starting position %d/ %d\n", current_search_position, last_search_position);
		}


		for (float current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step)
		{

			angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, 0.0, 0.0);

//			current_projection.SetToConstant(0.0f); // This also sets the FFT padding to zero
			template_reconstruction.ExtractSlice(current_projection, angles, 1.0f, false);
			current_projection.complex_values[0] = 0.0f + I * 0.0f;

			current_projection.SwapRealSpaceQuadrants();
			current_projection.MultiplyPixelWise(projection_filter);
			current_projection.BackwardFFT();
			average_on_edge = current_projection.ReturnAverageOfRealValuesOnEdges();


			// Make sure the device has moved on to the padded projection
			cudaStreamWaitEvent(cudaStreamPerThread,projection_is_free_Event, 0);

			if ( ! is_graph_allocated && first_loop_complete)
			{
				cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
				wxPrintf("\nBeginning stream capture for creation of graph\n");
				cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
			}


			if (first_loop_complete && is_graph_allocated)
			{
				cudaGraphLaunch(graphExec, cudaStreamPerThread);

			}
			else
			{



			//// TO THE GPU ////
			d_current_projection.CopyHostToDevice();

			d_current_projection.AddConstant(-average_on_edge);
			// The average in the full padded image will be different;
			average_on_edge *= (d_current_projection.number_of_real_space_pixels / (float)d_padded_reference.number_of_real_space_pixels);

			d_current_projection.MultiplyByConstant(rsqrtf(  d_current_projection.ReturnSumOfSquares() / (float)d_padded_reference.number_of_real_space_pixels - (average_on_edge * average_on_edge)));

			d_current_projection.ClipInto(&d_padded_reference, 0, false, 0, 0, 0, 0);
			cudaEventRecord(projection_is_free_Event, cudaStreamPerThread);


			d_padded_reference.ForwardFFT(false);
			//      d_padded_reference.ForwardFFTAndClipInto(d_current_projection,false);
			d_padded_reference.BackwardFFTAfterComplexConjMul(d_input_image.complex_values_16f, true);
//			d_padded_reference.BackwardFFTAfterComplexConjMul(d_input_image.complex_values_gpu, false);




			if (DO_HISTOGRAM)
			{
				if ( ! histogram.is_allocated_histogram )
				{
					d_padded_reference.NppInit();
					histogram.BufferInit(d_padded_reference.npp_ROI);
				}
				histogram.AddToHistogram(d_padded_reference);
			}



				this->MipPixelWise(d_padded_reference, __float2half_rn(current_psi) , __float2half_rn(global_euler_search.list_of_search_parameters[current_search_position][1]),
																					  __float2half_rn(global_euler_search.list_of_search_parameters[current_search_position][0]));
	//			this->MipPixelWise(d_padded_reference, float(current_psi) , float(global_euler_search.list_of_search_parameters[current_search_position][1]),
	//																			 	 float(global_euler_search.list_of_search_parameters[current_search_position][0]));
				this->SumPixelWise(d_padded_reference);
//			}





			ccc_counter++;
			total_number_of_cccs_calculated++;


			if ( ccc_counter % 10 == 0)
			{
				this->AccumulateSums(my_stats, d_sum1, d_sumSq1);
			}


			if ( ccc_counter % 100 == 0)
			{

				d_sum2.AddImage(d_sum1);
				d_sum1.Zeros();

				d_sumSq2.AddImage(d_sumSq1);
				d_sumSq1.Zeros();

			}

			if ( ccc_counter % 10000 == 0)
			{

				d_sum3.AddImage(d_sum2);
				d_sum2.Zeros();

				d_sumSq3.AddImage(d_sumSq2);
				d_sumSq2.Zeros();

			}


			current_projection.is_in_real_space = false;
			d_padded_reference.is_in_real_space = true;
//			d_padded_reference.Zeros();
			cudaEventRecord(gpu_work_is_done_Event, cudaStreamPerThread);

			} // end of non-graph launch loop


			if (! is_graph_allocated && first_loop_complete)
			{
				wxPrintf("\nEnding stream capture for creation of graph\n");
				cudaStreamEndCapture(cudaStreamPerThread, &graph);
				cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
				is_graph_allocated = true;
			}

			if (!first_loop_complete) first_loop_complete = true;

			} // loop over psi angles

      
 	} // end of outer loop euler sphere position

	wxPrintf("\t\t\ntotal number %d\n",ccc_counter);

    cudaStreamWaitEvent(cudaStreamPerThread,gpu_work_is_done_Event, 0);

	this->AccumulateSums(my_stats, d_sum1, d_sumSq1);

	d_sum2.AddImage(d_sum1);
	d_sumSq2.AddImage(d_sumSq1);

	d_sum3.AddImage(d_sum2);
	d_sumSq3.AddImage(d_sumSq2);

	this->MipToImage((const Peaks *)my_peaks, d_max_intensity_projection, d_best_psi, d_best_theta, d_best_phi);

	MyAssertTrue(histogram.is_allocated_histogram, "Trying to accumulate a histogram that has not been initialized!")
	histogram.Accumulate(d_padded_reference);

	cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

	cudaErr(cudaFree(my_peaks));
	cudaErr(cudaFree(my_stats));

}


void TemplateMatchingCore::SumPixelWise(GpuImage &image)
{

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
//	dim3 threadsPerBlock = dim3(1024, 1, 1);
//	dim3 gridDims = dim3((image.real_memory_allocated / SUM_PIXELWISE_UNROLL + threadsPerBlock.x - 1) / threadsPerBlock.x,1,1);

	image.ReturnLaunchParamtersLimitSMs(64,512);

    cudaErr(cudaFuncSetCacheConfig(SumPixelWiseKernel, cudaFuncCachePreferL1));
	SumPixelWiseKernel<< <image.gridDims, image.threadsPerBlock, 0,cudaStreamPerThread>> >((cufftReal *)image.real_values_gpu, my_stats,(int) image.real_memory_allocated - SUM_PIXELWISE_UNROLL + 1);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}


__global__ void SumPixelWiseKernel(const cufftReal* correlation_output, Stats* my_stats, const int  numel)
{
//    const int x = SUM_PIXELWISE_UNROLL * (blockIdx.x*blockDim.x + threadIdx.x);

    for ( int i = blockIdx.x*blockDim.x + threadIdx.x; i < numel; i+=blockDim.x * gridDim.x)
    {
    	const float val = correlation_output[i];
    	my_stats[i].sum += val;
    	my_stats[i].sq_sum += val*val;
    }
//	if ( x < numel )
//	{
//		#pragma unroll (SUM_PIXELWISE_UNROLL)
//		for ( int iVal = 0; iVal < SUM_PIXELWISE_UNROLL; iVal++)
//		{
//			const float val = correlation_output[x + iVal];
//			my_stats[x + iVal].sum += val;
//			my_stats[x + iVal].sq_sum += val*val;
// 		}
//	}

}



void TemplateMatchingCore::MipPixelWise(GpuImage &image, __half psi, __half theta, __half phi)
{

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
//	dim3 threadsPerBlock = dim3(1024, 1, 1);
//	dim3 gridDims = dim3((image.real_memory_allocated / MIP_PIXELWISE_UNROLL + threadsPerBlock.x - 1) / threadsPerBlock.x,1,1);
	int N = 64;

	image.ReturnLaunchParamtersLimitSMs(64,512);

	MipPixelWiseKernel<< <image.gridDims, image.threadsPerBlock,0,cudaStreamPerThread>> >((cufftReal *)image.real_values_gpu, my_peaks,(int) image.real_memory_allocated - MIP_PIXELWISE_UNROLL + 1, psi,theta, phi);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}

__global__ void MipPixelWiseKernel(const cufftReal*  correlation_output, Peaks* my_peaks, const int  numel,
									__half psi, __half theta, __half phi)
{

    for ( int i = blockIdx.x*blockDim.x + threadIdx.x; i < numel; i+=blockDim.x * gridDim.x)
    {
		const __half val = __float2half_rn(correlation_output[i]);

		if ( __hgt( val , my_peaks[i].mip) )
		{
			my_peaks[i].mip = val;
			my_peaks[i].psi = psi;
			my_peaks[i].theta = theta;
			my_peaks[i].phi = phi;

		}
    }
//
//    const unsigned int x = MIP_PIXELWISE_UNROLL *(blockIdx.x*blockDim.x + threadIdx.x);
//
//
//	if ( x < numel  )
//	{
//		#pragma unroll (MIP_PIXELWISE_UNROLL)
//		for (unsigned int iVal = 0; iVal < MIP_PIXELWISE_UNROLL; iVal++)
//		{
//
//			const __half val = __float2half_rn(correlation_output[x+iVal]);
//
//			if ( __hgt( val , my_peaks[x+iVal].mip) )
//			{
//				my_peaks[x+iVal].mip = val;
//				my_peaks[x+iVal].psi = psi;
//				my_peaks[x+iVal].theta = theta;
//				my_peaks[x+iVal].phi = phi;
//
//			}
//
//		}
//	}

}

__global__ void MipToImageKernel(const Peaks*, const int, cufftReal*, cufftReal*, cufftReal*, cufftReal*);

void TemplateMatchingCore::MipToImage(const Peaks* my_peaks, GpuImage &mip, GpuImage &psi, GpuImage &theta, GpuImage &phi)
{

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	dim3 threadsPerBlock = dim3(1024, 1, 1);
	dim3 gridDims = dim3((mip.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x,1,1);

	MipToImageKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >(my_peaks, mip.real_memory_allocated, mip.real_values_gpu, psi.real_values_gpu, theta.real_values_gpu, phi.real_values_gpu);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}

__global__ void  MipToImageKernel(const Peaks* my_peaks, const int numel, cufftReal* mip, cufftReal* psi, cufftReal* theta, cufftReal* phi)
{

    const int x = blockIdx.x*blockDim.x + threadIdx.x;

	if ( x < numel  )
	{
		mip[x] = 	(cufftReal)__half2float(my_peaks[x].mip);
		psi[x] = 	(cufftReal)__half2float(my_peaks[x].psi);
		theta[x] =	(cufftReal)__half2float(my_peaks[x].theta);
		phi[x] =	(cufftReal)__half2float(my_peaks[x].phi);

//		mip[x] = 	(cufftReal)(my_peaks[x].mip);
//		psi[x] = 	(cufftReal)(my_peaks[x].psi);
//		theta[x] =	(cufftReal)(my_peaks[x].theta);
//		phi[x] =	(cufftReal)(my_peaks[x].phi);

    }
}

__global__ void AccumulateSumsKernel(Stats* my_stats, const int numel, cufftReal* sum, cufftReal* sq_sum);

void TemplateMatchingCore::AccumulateSums(Stats* my_stats, GpuImage &sum, GpuImage &sq_sum)
{

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	dim3 threadsPerBlock = dim3(1024, 1, 1);
	dim3 gridDims = dim3((sum.real_memory_allocated / SUM_PIXELWISE_UNROLL + threadsPerBlock.x - 1) / threadsPerBlock.x,1,1);

	AccumulateSumsKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >(my_stats, sum.real_memory_allocated - SUM_PIXELWISE_UNROLL + 1, sum.real_values_gpu, sq_sum.real_values_gpu);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}

__global__ void AccumulateSumsKernel(Stats* my_stats, const int numel, cufftReal* sum, cufftReal* sq_sum)
{

    const unsigned int x = SUM_PIXELWISE_UNROLL * (blockIdx.x*blockDim.x + threadIdx.x);

	if ( x < numel )
	{
		#pragma unroll (SUM_PIXELWISE_UNROLL)
		for (unsigned int iVal = 0; iVal < SUM_PIXELWISE_UNROLL; iVal++)
		{
			sum[x+iVal] += my_stats[x+iVal].sum;
			sq_sum[x+iVal] += my_stats[x+iVal].sq_sum;

			my_stats[x+iVal].sum = 0.0f;
			my_stats[x+iVal].sq_sum = 0.0f;
		}

    }
}



