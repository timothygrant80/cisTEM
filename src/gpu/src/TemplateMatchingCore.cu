#include "gpu_core_headers.h"

#define DO_HISTOGRAM true


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
//		checkCudaErrors(cudaFree((void *)cummulative_histogram));
//		checkCudaErrors(cudaFreeHost((void *)h_cummulative_histogram));
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

    cudaStreamSynchronize(cudaStreamPerThread);

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

	d_input_image.ConvertToHalfPrecision(true);

	checkCudaErrors(cudaMalloc((void **)&my_stats, sizeof(Stats)*d_input_image.real_memory_allocated));

	cudaEvent_t projection_is_free_Event, gpu_work_is_done_Event;
	checkCudaErrors(cudaEventCreateWithFlags(&projection_is_free_Event, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&gpu_work_is_done_Event, cudaEventDisableTiming));


	long ccc_counter = 0;
	int current_search_position;
	float average_on_edge;
	float variance;

	int thisDevice;
	cudaGetDevice(&thisDevice);
	wxPrintf("Thread %d is running on device %d\n", threadIDX, thisDevice);

	for (current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++)
	{

		wxPrintf("Starting position %d/ %d\n", current_search_position, last_search_position);


		for (float current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step)
		{

			angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, 0.0, 0.0);

			template_reconstruction.ExtractSlice(current_projection, angles, 1.0f, false);
			current_projection.complex_values[0] = 0.0f + I * 0.0f;

			current_projection.SwapRealSpaceQuadrants();
			current_projection.MultiplyPixelWise(projection_filter);
			current_projection.BackwardFFT();
			average_on_edge = current_projection.ReturnAverageOfRealValuesOnEdges();

			// Make sure the device has moved on to the padded projection
			cudaStreamWaitEvent(cudaStreamPerThread,projection_is_free_Event, 0);


			//// TO THE GPU ////
			d_current_projection.CopyHostToDevice();

			d_current_projection.AddConstant(-average_on_edge);
			d_current_projection.MultiplyByConstant(1.0f / sqrtf(  d_current_projection.ReturnSumOfSquares() / (float)d_padded_reference.number_of_real_space_pixels - (average_on_edge * average_on_edge)));

			d_current_projection.ClipInto(&d_padded_reference, 0, false, 0, 0, 0, 0);
			cudaEventRecord(projection_is_free_Event, cudaStreamPerThread);


			d_padded_reference.ForwardFFT(false);

			//      d_padded_reference.ForwardFFTAndClipInto(d_current_projection,false);

			d_padded_reference.BackwardFFTAfterComplexConjMul(d_input_image.complex_values_16f, true);


			this->MipPixelWise(d_padded_reference, __float2half(current_psi) , __float2half(global_euler_search.list_of_search_parameters[current_search_position][1]),
																			 	 __float2half(global_euler_search.list_of_search_parameters[current_search_position][0]));


			if (DO_HISTOGRAM)
			{
				if ( ! histogram.is_allocated_histogram )
				{
					d_padded_reference.NppInit();
					histogram.BufferInit(d_padded_reference.npp_ROI);
				}
				histogram.AddToHistogram(d_padded_reference);
			}


			ccc_counter++;
			total_number_of_cccs_calculated++;


			if ( ccc_counter % 20 == 0)
			{
				this->AccumulateSums(my_stats, d_sum1, d_sumSq1);
			}


			if ( ccc_counter % 400 == 0)
			{

				d_sum2.AddImage(d_sum1);
				d_sum1.Zeros();

				d_sumSq2.AddImage(d_sumSq1);
				d_sumSq1.Zeros();

			}

			if ( ccc_counter % 160000 == 0)
			{

				d_sum3.AddImage(d_sum2);
				d_sum2.Zeros();

				d_sumSq3.AddImage(d_sumSq2);
				d_sumSq2.Zeros();

			}


			current_projection.is_in_real_space = false;
			d_padded_reference.is_in_real_space = true;
			cudaEventRecord(gpu_work_is_done_Event, cudaStreamPerThread);



			} // loop over psi angles


      
 	} // end of outer loop euler sphere position

	wxPrintf("\t\t\ntotal number %ld\n",ccc_counter);

    cudaStreamWaitEvent(cudaStreamPerThread,gpu_work_is_done_Event, 0);

	this->AccumulateSums(my_stats, d_sum1, d_sumSq1);

	d_sum2.AddImage(d_sum1);
	d_sumSq2.AddImage(d_sumSq1);

	d_sum3.AddImage(d_sum2);
	d_sumSq3.AddImage(d_sumSq2);

	this->MipToImage((const Stats *)my_stats, d_max_intensity_projection, d_best_psi, d_best_theta, d_best_phi);


	histogram.Accumulate(d_padded_reference);


    checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));


}

__global__ void MipPixelWiseKernel(const cufftReal*  correlation_output, Stats* my_stats, const long  numel,
                                   __half c_psi, __half c_phi, __half c_theta);

void TemplateMatchingCore::MipPixelWise(GpuImage &image, __half psi, __half theta, __half phi)
{

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	dim3 threadsPerBlock = dim3(1024, 1, 1);
	dim3 gridDims = dim3((image.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x,1,1);

	MipPixelWiseKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >((const cufftReal *)image.real_values_gpu, my_stats,(const long) image.real_memory_allocated, psi,theta, phi);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}


__global__ void MipPixelWiseKernel(const cufftReal*  correlation_output, Stats* my_stats, const long  numel,
                                   __half psi, __half theta, __half phi)
{

    const int x = blockIdx.x*blockDim.x + threadIdx.x;

    __half val = __float2half_rn(correlation_output[x]);

	if ( x < numel  )
	{

		my_stats[x].sum = __hadd(my_stats[x].sum, val);
		my_stats[x].sq_sum = __hadd(my_stats[x].sum, __hmul(val,val));



		if ( __hgt( val , my_stats[x].mip) )
		{

			my_stats[x].mip = val;
			my_stats[x].psi = psi;
			my_stats[x].theta = theta;
			my_stats[x].phi = phi;

		}

    }
}

__global__ void MipToImageKernel(const Stats*, const long, cufftReal*, cufftReal*, cufftReal*, cufftReal*);

void TemplateMatchingCore::MipToImage(const Stats* my_stats, GpuImage &mip, GpuImage &psi, GpuImage &theta, GpuImage &phi)
{

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	dim3 threadsPerBlock = dim3(1024, 1, 1);
	dim3 gridDims = dim3((mip.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x,1,1);

	MipToImageKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >(my_stats, (const long)mip.real_memory_allocated, mip.real_values_gpu, psi.real_values_gpu, theta.real_values_gpu, phi.real_values_gpu);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}

__global__ void  MipToImageKernel(const Stats* my_stats, const long numel, cufftReal* mip, cufftReal* psi, cufftReal* theta, cufftReal* phi)
{

    const int x = blockIdx.x*blockDim.x + threadIdx.x;

	if ( x < numel  )
	{
		mip[x] = 	(cufftReal)__half2float(my_stats[x].mip);
		psi[x] = 	(cufftReal)__half2float(my_stats[x].psi);
		theta[x] =	(cufftReal)__half2float(my_stats[x].theta);
		phi[x] =	(cufftReal)__half2float(my_stats[x].phi);

    }
}

__global__ void AccumulateSumsKernel(Stats* my_stats, const long numel, cufftReal* sum, cufftReal* sq_sum);

void TemplateMatchingCore::AccumulateSums(Stats* my_stats, GpuImage &sum, GpuImage &sq_sum)
{

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	dim3 threadsPerBlock = dim3(1024, 1, 1);
	dim3 gridDims = dim3((sum.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x,1,1);

	AccumulateSumsKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >(my_stats, sum.real_memory_allocated, sum.real_values_gpu, sq_sum.real_values_gpu);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}

__global__ void AccumulateSumsKernel(Stats* my_stats, const long numel, cufftReal* sum, cufftReal* sq_sum)
{

    const int x = blockIdx.x*blockDim.x + threadIdx.x;

	if ( x < numel )
	{
		sum[x] += (cufftReal)__half2float(my_stats[x].sum);
		sq_sum[x] += (cufftReal)__half2float(my_stats[x].sq_sum);

		my_stats[x].sum = 0.0f;
		my_stats[x].sq_sum = 0.0f;
    }
}



