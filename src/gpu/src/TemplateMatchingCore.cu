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
    d_stats_reference.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
    d_max_intensity_projection.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
    d_best_psi.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
    d_best_theta.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
    d_best_phi.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);



	d_sum1.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
	d_sumSq1.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
	is_non_zero_sum_buffer = 1;




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


  
//  // TODO should this be swapped? Can it just be passed as FFT?
//  d_input_image.ForwardFFT(); // This is for the standalone test
//  template_reconstruction.ForwardFFT(); // This is for the standalone test
//  template_reconstruction.SwapRealSpaceQuadrants(); // This is for the standalone test
	// Make sure we are starting with zeros
	d_max_intensity_projection.Zeros();
	d_best_psi.Zeros();
	d_best_phi.Zeros();
	d_best_theta.Zeros();
	d_padded_reference.Zeros();
	d_stats_reference.Zeros();

	  d_sum1.Zeros();
	  d_sumSq1.Zeros();

	this->c_defocus = c_defocus;
	this->c_pixel = c_pixel;
	total_number_of_cccs_calculated = 0;


	cudaStream_t statsStream;
	cudaEvent_t projection_is_free_Event, gpu_work_is_done_Event, stats_are_done_Event, stats_copy_is_done_Event;
	checkCudaErrors(cudaStreamCreateWithFlags(&statsStream, cudaStreamDefault));
	checkCudaErrors(cudaEventCreateWithFlags(&projection_is_free_Event, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&gpu_work_is_done_Event, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&stats_are_done_Event, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&stats_copy_is_done_Event, cudaEventDisableTiming));


	d_stats_reference.SetStream(statsStream);
	d_sum1.SetStream(statsStream);d_sum2.SetStream(statsStream);d_sum3.SetStream(statsStream);d_sum4.SetStream(statsStream);d_sum5.SetStream(statsStream);
	d_sumSq1.SetStream(statsStream);d_sumSq2.SetStream(statsStream);d_sumSq3.SetStream(statsStream);d_sumSq4.SetStream(statsStream);d_sumSq5.SetStream(statsStream);


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

//      wxPrintf("Pos %d working on phi, theta, psi, %3.3f %3.3f %3.3f \n", current_search_position, (float)global_euler_search.list_of_search_parameters[current_search_position][0], (float)global_euler_search.list_of_search_parameters[current_search_position][1], (float)current_psi);
      // FIXME not padding enabled
      // HOST project. Note that the projection has mean set to zero but it is probably better to ensure this here as it is cheap.

      template_reconstruction.ExtractSlice(current_projection, angles, 1.0f, false);
      current_projection.complex_values[0] = 0.0f + I * 0.0f;

      current_projection.SwapRealSpaceQuadrants();
      current_projection.MultiplyPixelWise(projection_filter);
      current_projection.BackwardFFT();
      average_on_edge = current_projection.ReturnAverageOfRealValuesOnEdges();
//      variance = 1.0f / sqrtf( current_projection.ReturnSumOfSquares() * current_projection.number_of_real_space_pixels /
//              d_padded_reference.number_of_real_space_pixels - (average_on_edge*average_on_edge) );
//      current_projection.AddMultiplyConstant(-average_on_edge, variance);

//      // TODO add more?

      // Make sure the device has moved on to the padded projection
      cudaStreamWaitEvent(d_padded_reference.calcStream,projection_is_free_Event, 0);

    
      d_current_projection.CopyHostToDevice();

      d_current_projection.AddConstant(-average_on_edge);
      d_current_projection.MultiplyByConstant(1.0f / sqrtf(  d_current_projection.ReturnSumOfSquares() / (float)d_padded_reference.number_of_real_space_pixels - (average_on_edge * average_on_edge)));


      // Ensure the last padded ref is copied prior to over-writing it.
      cudaStreamWaitEvent(d_stats_reference.calcStream,stats_are_done_Event, 0);

      d_current_projection.ClipInto(&d_padded_reference, 0, false, 0, 0, 0, 0);
      cudaEventRecord(projection_is_free_Event, d_padded_reference.calcStream);

//      cudaStreamSynchronize(cudaStreamPerThread);
//      std::string fileNameOUT4 = "/tmp/checkPaddedRef" + std::to_string(threadIDX) + ".mrc";
//      d_padded_reference.QuickAndDirtyWriteSlices(fileNameOUT4, 1, 1); 

      pre_checkErrorsAndTimingWithSynchronization(d_padded_reference.calcStream);
      d_padded_reference.ForwardFFT();
      checkErrorsAndTimingWithSynchronization(d_padded_reference.calcStream);
      // The input image should have zero mean, so multipling also zeros the mean of the ref.
      d_padded_reference.MultiplyPixelWiseComplexConjugate(d_input_image);
      pre_checkErrorsAndTimingWithSynchronization(d_padded_reference.calcStream);
      d_padded_reference.BackwardFFT();
      checkErrorsAndTimingWithSynchronization(d_padded_reference.calcStream);


      cudaStreamWaitEvent(d_stats_reference.calcStream,stats_are_done_Event, 0);


      // TODO make this a Device to Device method
      checkCudaErrors(cudaMemcpyAsync(d_stats_reference.real_values_gpu,
    		  	  	  	  	  	  	  d_padded_reference.real_values_gpu,
    		  	  	  	  	  	  	  sizeof(cufftReal)*d_padded_reference.real_memory_allocated,
    		  	  	  	  	  	  	  cudaMemcpyDeviceToDevice, d_stats_reference.calcStream));


      cudaEventRecord(stats_copy_is_done_Event, d_stats_reference.calcStream);

//
      cudaStreamWaitEvent(d_stats_reference.calcStream,stats_are_done_Event, 0);

//		d_padded_reference.NppInit();
		if (DO_HISTOGRAM)
		{
			if ( ! histogram.is_allocated_histogram )
			{
				d_stats_reference.NppInit();
		    	histogram.BufferInit(d_stats_reference.npp_ROI);
			}
			histogram.AddToHistogram(d_stats_reference);
		}

		d_sum1.AddImage(d_stats_reference);
		d_sumSq1.AddSquaredImage(d_stats_reference);

//		d_sumSq[0].AddSquaredImage(d_padded_reference);


        ccc_counter++;
        total_number_of_cccs_calculated++;


		if ( ccc_counter % 10 == 0)
		{

			if ( is_non_zero_sum_buffer < 2 )
			{
				d_sum2.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);d_sum2.Zeros();
				d_sumSq2.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);d_sumSq2.Zeros();
				is_non_zero_sum_buffer = 2;
			}


			d_sum2.AddImage(d_sum1);
			d_sum1.Zeros();

			d_sumSq2.AddImage(d_sumSq1);
			d_sumSq1.Zeros();


		}


		if ( ccc_counter % 100 == 0)
		{

			if ( is_non_zero_sum_buffer < 3 )
			{
				d_sum3.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);d_sum3.Zeros();
				d_sumSq3.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);d_sumSq3.Zeros();
				is_non_zero_sum_buffer = 3;
			}

			d_sum3.AddImage(d_sum2);
			d_sum2.Zeros();

			d_sumSq3.AddImage(d_sumSq2);
			d_sumSq2.Zeros();

		}


		if ( ccc_counter % 10000 == 0)
		{
			if ( is_non_zero_sum_buffer < 4 )
			{
				d_sum4.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);d_sum4.Zeros();
				d_sumSq4.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);d_sumSq4.Zeros();
				is_non_zero_sum_buffer = 4;
			}

			d_sum4.AddImage(d_sum3);
			d_sum3.Zeros();

			d_sumSq4.AddImage(d_sumSq3);
			d_sumSq3.Zeros();

		}


		if ( ccc_counter % 100000000 == 0)
		{

			if ( is_non_zero_sum_buffer < 2 )
			{
				d_sum5.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);d_sum5.Zeros();
				d_sumSq5.Allocate(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);d_sumSq5.Zeros();
				is_non_zero_sum_buffer = 5;
			}

			d_sum5.AddImage(d_sum4);
			d_sum4.Zeros();

			d_sumSq5.AddImage(d_sumSq4);
			d_sumSq4.Zeros();

		}

		cudaEventRecord(stats_are_done_Event, d_stats_reference.calcStream);

	      d_max_intensity_projection.MipPixelWise(d_padded_reference, d_best_psi, d_best_phi, d_best_theta,
	                                              current_psi,
	                                              global_euler_search.list_of_search_parameters[current_search_position][0],
	                                              global_euler_search.list_of_search_parameters[current_search_position][1]);

      current_projection.is_in_real_space = false;
      d_padded_reference.is_in_real_space = true;

      cudaEventRecord(gpu_work_is_done_Event, d_padded_reference.calcStream);




		} // loop over psi angles



      
 	} // end of outer loop euler sphere position

	wxPrintf("\t\t\ntotal number %ld\n",ccc_counter);

    checkCudaErrors(cudaStreamSynchronize(d_padded_reference.calcStream));
    checkCudaErrors(cudaStreamSynchronize(d_stats_reference.calcStream));




  
}





