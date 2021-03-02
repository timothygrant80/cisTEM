#include "gpu_core_headers.h"

#define DO_HISTOGRAM true


__global__ void MipPixelWiseKernel(__half* correlation_output, __half2* my_peaks, const int  numel,
                                   __half psi, __half theta, __half phi, __half2* my_stats, __half2* my_new_peaks);




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

void TemplateMatchingCore::Init(MyApp *parent_pointer,
								Image &template_reconstruction,
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
                                AnglesAndShifts &angles,
                                EulerSearch &global_euler_search,
                    			float histogram_min_scaled,
                    			float histogram_step_scaled,
                    			int histogram_number_of_bins,
                    			int max_padding,
                                int first_search_position,
                                int last_search_position,
                                ProgressBar *my_progress,
                                long total_correlation_positions,
                                bool is_running_locally)
                                
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

	this->my_progress = my_progress;
	this->total_correlation_positions = total_correlation_positions;
	this->is_running_locally = is_running_locally;

	this->parent_pointer = parent_pointer;
    
    // For now we are only working on the inner loop, so no need to track best_defocus and best_pixel_size

    // At the outset these are all empty cpu images, so don't xfer, just allocate on gpuDev



    // Transfer the input image_memory_should_not_be_deallocated  

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

};

void TemplateMatchingCore::RunInnerLoop(Image &projection_filter, float c_pixel, float c_defocus, int threadIDX, long &current_correlation_position)
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
	d_padded_reference.ConvertToHalfPrecision(false);


	cudaErr(cudaMalloc((void **)&my_peaks, sizeof(__half2)*d_input_image.real_memory_allocated));
	cudaErr(cudaMalloc((void **)&my_new_peaks, sizeof(__half2)*d_input_image.real_memory_allocated));
	cudaErr(cudaMalloc((void **)&my_stats, sizeof(__half2)*d_input_image.real_memory_allocated));
	cudaErr(cudaMemset(my_peaks,0,sizeof(__half2)*d_input_image.real_memory_allocated));
//	cudaErr(cudaMemset(my_stats,0,sizeof(Peaks)*d_input_image.real_memory_allocated));



	cudaEvent_t projection_is_free_Event, gpu_work_is_done_Event;
	cudaErr(cudaEventCreateWithFlags(&projection_is_free_Event, cudaEventDisableTiming));
	cudaErr(cudaEventCreateWithFlags(&gpu_work_is_done_Event, cudaEventDisableTiming));

	int ccc_counter = 0;
	int current_search_position;
	float average_on_edge;
	float temp_float;

	int thisDevice;
	cudaGetDevice(&thisDevice);
	wxPrintf("Thread %d is running on device %d\n", threadIDX, thisDevice);

//	cudaErr(cudaFuncSetCacheConfig(SumPixelWiseKernel, cudaFuncCachePreferL1));

//	bool make_graph = true;
//	bool first_loop_complete = false;

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

			//// TO THE GPU ////
			d_current_projection.CopyHostToDevice();

			d_current_projection.AddConstant(-average_on_edge);
			// The average in the full padded image will be different;
			average_on_edge *= (d_current_projection.number_of_real_space_pixels / (float)d_padded_reference.number_of_real_space_pixels);

			d_current_projection.MultiplyByConstant(rsqrtf(  d_current_projection.ReturnSumOfSquares() / (float)d_padded_reference.number_of_real_space_pixels - (average_on_edge * average_on_edge)));
			d_current_projection.ClipInto(&d_padded_reference, 0, false, 0, 0, 0, 0);
			cudaEventRecord(projection_is_free_Event, cudaStreamPerThread);

			// For the cpu code (MKL and FFTW) the image is multiplied by N on the forward xform, and subsequently normalized by 1/N
			// cuFFT multiplies by 1/root(N) forward and then 1/root(N) on the inverse. The input image is done on the cpu, and so has no scaling.
			// Stating false on the forward FFT leaves the ref = ref*root(N). Then we have root(N)*ref*input * root(N) (on the inverse) so we need a factor of 1/N to come out proper. This is included in BackwardFFTAfterComplexConjMul
			d_padded_reference.ForwardFFT(false);

			//      d_padded_reference.ForwardFFTAndClipInto(d_current_projection,false);
			d_padded_reference.BackwardFFTAfterComplexConjMul(d_input_image.complex_values_16f, true);

//			d_padded_reference.BackwardFFTAfterComplexConjMul(d_input_image.complex_values_gpu, false);
//			d_padded_reference.ConvertToHalfPrecision(false);



			if (DO_HISTOGRAM)
			{
				if ( ! histogram.is_allocated_histogram )
				{
					d_padded_reference.NppInit();
					histogram.BufferInit(d_padded_reference.npp_ROI);
				}
				histogram.AddToHistogram(d_padded_reference);
			}

//			if (make_graph && first_loop_complete)
//			{
//				wxPrintf("\nBeginning stream capture for creation of graph\n");
//				cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
//			}
//
//			if (first_loop_complete && ! make_graph)
//			{
//				cudaGraphLaunch(graphExec, cudaStreamPerThread);
//
//			}
//			else
//			{
				this->MipPixelWise( __float2half_rn(current_psi) , __float2half_rn(global_euler_search.list_of_search_parameters[current_search_position][1]),
																					  __float2half_rn(global_euler_search.list_of_search_parameters[current_search_position][0]));
	//			this->MipPixelWise(d_padded_reference, float(current_psi) , float(global_euler_search.list_of_search_parameters[current_search_position][1]),
	//																			 	 float(global_euler_search.list_of_search_parameters[current_search_position][0]));
//				this->SumPixelWise(d_padded_reference);
//			}


//			if (make_graph && first_loop_complete)
//			{
//				wxPrintf("\nEnding stream capture for creation of graph\n");
//				cudaStreamEndCapture(cudaStreamPerThread, &graph);
//				cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
//				make_graph = false;
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


//			first_loop_complete = true;

			if (is_running_locally)
			{
				if (ReturnThreadNumberOfCurrentThread() == 0)
				{
					current_correlation_position++;
					if (current_correlation_position > total_correlation_positions) current_correlation_position = total_correlation_positions;
					my_progress->Update(current_correlation_position);
				}
			}
			else
			{
				temp_float = current_correlation_position;
				JobResult *temp_result = new JobResult;
				temp_result->SetResult(1, &temp_float);
				parent_pointer->AddJobToResultQueue(temp_result);
			}
			} // loop over psi angles

      
 	} // end of outer loop euler sphere position


	wxPrintf("\t\t\ntotal number %d\n",ccc_counter);

    cudaStreamWaitEvent(cudaStreamPerThread,gpu_work_is_done_Event, 0);

	this->AccumulateSums(my_stats, d_sum1, d_sumSq1);

	d_sum2.AddImage(d_sum1);
	d_sumSq2.AddImage(d_sumSq1);

	d_sum3.AddImage(d_sum2);
	d_sumSq3.AddImage(d_sumSq2);

	this->MipToImage();

	MyAssertTrue(histogram.is_allocated_histogram, "Trying to accumulate a histogram that has not been initialized!")
	histogram.Accumulate(d_padded_reference);

	cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

	cudaErr(cudaFree(my_peaks));
	cudaErr(cudaFree(my_stats));

}




void TemplateMatchingCore::MipPixelWise( __half psi, __half theta, __half phi)
{

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);


	// N*
	d_padded_reference.ReturnLaunchParamtersLimitSMs(5.f,1024);


	MipPixelWiseKernel<< <d_padded_reference.gridDims, d_padded_reference.threadsPerBlock,0,cudaStreamPerThread>> >((__half *)d_padded_reference.real_values_16f, my_peaks,(int)d_padded_reference.real_memory_allocated , psi,theta, phi, my_stats, my_new_peaks);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}

__global__ void MipPixelWiseKernel(__half* correlation_output, __half2* my_peaks, const int  numel,
									__half psi, __half theta, __half phi, __half2* my_stats, __half2* my_new_peaks)
{

//	Peaks tmp_peak;

    for ( int i = blockIdx.x*blockDim.x + threadIdx.x; i < numel; i+=blockDim.x * gridDim.x)
    {


    	const __half half_val = correlation_output[i];
    	const __half2 input = __half2half2(half_val * __half(10000.0));
    	const __half2 mulVal = __halves2half2((__half)1.0, half_val);
//    	my_stats[i].sum = __hadd(my_stats[i].sum, half_val);
//    	my_stats[i].sq_sum = __hfma(__half(1000.)*half_val,half_val,my_stats[i].sq_sum);
    	my_stats[i] = __hfma2(input, mulVal, my_stats[i]);
//    	tmp_peak = my_peaks[i];
//		const __half half_val = __float2half_rn(val);

//			tmp_peak.psi = psi;
//			tmp_peak.theta = theta;
//			tmp_peak.phi = phi;
			if (  half_val > __low2half(my_peaks[i]) )
			{
//				tmp_peak.mip = half_val;
				my_peaks[i] = __halves2half2(half_val, psi);
				my_new_peaks[i] = __halves2half2(theta, phi);

//				my_peaks[i].mip = correlation_output[i];
//				my_peaks[i].psi = psi;
//				my_peaks[i].theta = theta;
//				my_peaks[i].phi = phi;
			}
    }
//


}

__global__ void MipToImageKernel(const __half2*, const __half2* my_new_peaks, const int, cufftReal*, cufftReal*, cufftReal*, cufftReal*);

void TemplateMatchingCore::MipToImage()
{

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	dim3 threadsPerBlock = dim3(1024, 1, 1);
	dim3 gridDims = dim3((d_max_intensity_projection.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x,1,1);

	MipToImageKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >(my_peaks, my_new_peaks, d_max_intensity_projection.real_memory_allocated,
			d_max_intensity_projection.real_values_gpu, d_best_psi.real_values_gpu, d_best_theta.real_values_gpu, d_best_phi.real_values_gpu);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}

__global__ void  MipToImageKernel(const __half2* my_peaks, const __half2* my_new_peaks, const int numel, cufftReal* mip, cufftReal* psi, cufftReal* theta, cufftReal* phi)
{

    const int x = blockIdx.x*blockDim.x + threadIdx.x;

	if ( x < numel  )
	{

		mip[x] = 	(cufftReal)__low2float(my_peaks[x]);
		psi[x] = 	(cufftReal)__high2float(my_peaks[x]);
		theta[x] =	(cufftReal)__low2float(my_new_peaks[x]);
		phi[x] =	(cufftReal)__high2float(my_new_peaks[x]);

    }
}

__global__ void AccumulateSumsKernel(__half2* my_stats, const int numel, cufftReal* sum, cufftReal* sq_sum);

void TemplateMatchingCore::AccumulateSums(__half2* my_stats, GpuImage &sum, GpuImage &sq_sum)
{

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	dim3 threadsPerBlock = dim3(1024, 1, 1);
	dim3 gridDims = dim3((sum.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x,1,1);

	AccumulateSumsKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >(my_stats, sum.real_memory_allocated, sum.real_values_gpu, sq_sum.real_values_gpu);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}

__global__ void AccumulateSumsKernel(__half2* my_stats, const int numel, cufftReal* sum, cufftReal* sq_sum)
{

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if ( x < numel )
	{

			sum[x] = __fmaf_rn(0.0001f, __low2float(my_stats[x]), sum[x]);
			sq_sum[x] = __fmaf_rn(0.0001f, __high2float(my_stats[x]), sq_sum[x]);

			my_stats[x] = __halves2half2((__half)0.,(__half)0.);


    }
}



