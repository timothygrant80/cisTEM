#include "../../core/core_headers.h"




class
GpuUtilTest : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();
	void TemplateMatchingStandalone(int nGPUs, int nThreads);
	void createImageAddOne();

	private:
};


IMPLEMENT_APP(GpuUtilTest)

// override the DoInteractiveUserInput

void GpuUtilTest::DoInteractiveUserInput()
{

}

// override the do calculation method which will be what is actually run..

bool GpuUtilTest::DoCalculation()
{

  wxPrintf("GpuUtilTest is running!\n");

//  this->createImageAddOne();
  int nThreads = 1;
  int nGPUs = 1;
  this->TemplateMatchingStandalone(nThreads, nGPUs);

  int gpuID = 0;
  wxPrintf("I made it here\n");

  

}	

void GpuUtilTest::TemplateMatchingStandalone(int nThreads, int nGPUs)
{

	int number_of_jobs_per_image_in_gui = 1;
	nThreads = 2;
	nGPUs = 1;
	int minPos = 0;
	int maxPos = 60;
	int incPos = 60 / (nThreads*nGPUs); // FIXME
//	DeviceManager gpuDev(nGPUs);
//    omp_set_num_threads(nThreads * gpuDev.nGPUs);  // create as many CPU threads as there are CUDA devices
//	#pragma omp parallel
//    {

		TemplateMatchingCore GPU[nThreads];
//    	TemplateMatchingCore GPU(number_of_jobs_per_image_in_gui);


		// Check the number of available gpus
		DeviceManager gpuDev;
		gpuDev.Init(nGPUs);


		#pragma omp parallel num_threads(nThreads)
		{

			int tIDX = ReturnThreadNumberOfCurrentThread();

	    	Image template_reconstruction;
	    	Image projection_filter;
	    	Image input_image;
	    	Image current_projection;
	    	Image padded_reference;
	    	Image max_intensity_projection;
	    	Image best_psi;
	    	Image best_phi;
	    	Image best_theta;
	    	Image best_defocus;
	    	Image best_pixel_size;

	    	ImageFile template_reconstruction_file;

	    	template_reconstruction_file.OpenFile("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/template_reconstruction.mrc", false);
	    	template_reconstruction.ReadSlices(&template_reconstruction_file, 1, template_reconstruction_file.ReturnNumberOfSlices());

			projection_filter.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/projection_filter.mrc",1);
			input_image.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/input_image.mrc",1);
			current_projection.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/current_projection.mrc",1);
			padded_reference.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/padded_reference.mrc",1);


			input_image.Resize(4096,4096,1,0.0f);
			padded_reference.CopyFrom(&input_image);
			// These are all blank to start
			max_intensity_projection.CopyFrom(&input_image);
			best_psi.CopyFrom(&input_image);
			best_phi.CopyFrom(&input_image);
			best_theta.CopyFrom(&input_image);
			best_pixel_size.CopyFrom(&input_image);
			best_defocus.CopyFrom(&input_image);

	//		// These should be in Fourier space, but were ifft to save
			template_reconstruction.ForwardFFT();
			template_reconstruction.SwapRealSpaceQuadrants();
			input_image.ForwardFFT();
			input_image.SwapRealSpaceQuadrants();
			projection_filter.ForwardFFT();



			// These also were set up prior to entering the GPU loop
			EulerSearch	global_euler_search;
			AnglesAndShifts angles;


			float angular_step = 2.5f;
			float psi_step = 1.5f;
			float pixel_size = 1.5;
			float pixel_size_search_range = 0.0f;
			float pixel_size_step = 0.001f;
			float defocus_search_range = 0.0f;
			float defocus_step = 200.0f;
			float defocus1 = 19880.0f;
			float defocus2 = 18910.0f;
			long first_search_position = 0 + (tIDX*incPos);
			long last_search_position = incPos + (tIDX*incPos);

			if (tIDX == (nThreads*nGPUs - 1)) last_search_position = maxPos;

			float high_resolution_limit_search = 2.0f * pixel_size;
			int best_parameters_to_keep = 1;
			float psi_start = 0.0f;
			float psi_max = 360.0f;
			ParameterMap parameter_map; // needed for euler search init
			parameter_map.SetAllTrue();
			global_euler_search.InitGrid("O", angular_step, 0.0, 0.0, psi_max, psi_step, psi_start, pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);
			global_euler_search.CalculateGridSearchPositions(false);

			wxDateTime 	overall_start;
			wxDateTime 	overall_finish;
			overall_start = wxDateTime::Now();
			gpuDev.SetGpu(tIDX);

			int max_padding = 0;
			const float histogram_min = -20.0f;
			const float histogram_max = 50.0f;
			const int histogram_number_of_points = 1024;
			float histogram_step;
			float histogram_min_scaled, histogram_step_scaled;
			histogram_step = (histogram_max - histogram_min) / float(histogram_number_of_points);

			histogram_min_scaled = histogram_min / double(sqrt(input_image.logical_x_dimension * input_image.logical_y_dimension));
			histogram_step_scaled = histogram_step / double(sqrt(input_image.logical_x_dimension * input_image.logical_y_dimension));

			GPU[tIDX].Init(template_reconstruction, input_image, current_projection,
					pixel_size_search_range, pixel_size_step, pixel_size,
					defocus_search_range, defocus_step, defocus1, defocus2,
					psi_max, psi_start, psi_step,
					angles, global_euler_search,
					histogram_min_scaled, histogram_step_scaled, histogram_number_of_points,
					max_padding, first_search_position, last_search_position);

			int size_i = 0;
			int defocus_i = 0;


			GPU[tIDX].RunInnerLoop(projection_filter, size_i, defocus_i, tIDX);

			long* histogram_data = new long[GPU[tIDX].histogram.histogram_n_bins];
			for (int iBin = 0; iBin < GPU[tIDX].histogram.histogram_n_bins; iBin++)
			{
				histogram_data[iBin] = 0;
			}
//			GPU[tIDX].histogram.CopyToHostAndAdd(histogram_data);
			std::string fileNameOUT4 = "/tmp/tmpMip" + std::to_string(tIDX) + ".mrc";
			max_intensity_projection.QuickAndDirtyWriteSlice(fileNameOUT4,1,true,1.5);

			wxPrintf("\n\n\tTimings: Overall: %s\n",(wxDateTime::Now()-overall_start).Format());


    } // end of omp block
}

void GpuUtilTest::createImageAddOne()
{


	bool do_all_tests = false;
	bool do_shift = false;
	bool do_fft = true;
	bool do_scale = false;
	bool do_swap = false;
	bool do_pad = false;


	int wanted_number_of_gpus = 1;
	int wanted_number_threads_per_gpu = 1;

	DeviceManager gpuDev(wanted_number_of_gpus);



//
//
//	wxPrintf("Found %d gpus to use\n",gDev.nGPUs);
//    ContextManager CM[gDev.nGPUs];
//
//
	#pragma omp parallel num_threads(wanted_number_threads_per_gpu * gpuDev.nGPUs)
    {

		int threadIDX = ReturnThreadNumberOfCurrentThread();
		gpuDev.SetGpu(threadIDX);

		Image cpu_image_half;
		Image cpu_work_half;
		GpuImage d_image_half;

		Image cpu_image_full;
		Image cpu_work_full;
		GpuImage d_image_full;

		EmpiricalDistribution eDist;

		float t_rmsd;


//		GpuImage mask;
//		cpu_image_full.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/oval_full.mrc",1);
//		d_image_full.Init(cpu_image_full);
//		d_image_full.CopyHostToDevice();
//		d_image_full.ForwardFFT(false);
//		float wm =  d_image_full.ReturnSumSquareModulusComplexValuesMask();
//		float nm = d_image_full.ReturnSumSquareModulusComplexValues();
//		wxPrintf("Compare the slow loop and the cublas %3.3e, %3.3e\n",wm,nm);
//		mask.Allocate(512,512,1,true);
//		mask.Wait();
//		mask.ReturnSumSquareModulusComplexValuesMask();
//		mask.mask_CSOS->printVal("after copy val 0 is",0);
//		mask.mask_CSOS->printVal("after copy val 100 is",100);
//		mask.mask_CSOS->QuickAndDirtyWriteSlices("/tmp/mask.mrc",1,1);


		if (do_shift || do_all_tests)
		{

			// full size image

			// Read in the CPU image
			cpu_image_full.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/oval_full.mrc",1);

			// Copy of the image to operate on using cpu method
			cpu_work_full.CopyFrom(&cpu_image_full);

			// Initialize gpu image, and copy to device
			d_image_full.Init(cpu_image_full);
			d_image_full.CopyHostToDevice();

			cpu_work_full.PhaseShift(10, -200,0);
			d_image_full.PhaseShift(10,-200,0);

			d_image_full.Wait();
//			d_image_full.CopyDeviceToHost(true, true);
			d_image_full.CopyDeviceToHost(true, true);

			d_image_full.Wait();
			d_image_full.QuickAndDirtyWriteSlices("/tmp/oval_full_shift_fromGPU.mrc",1,1);

			cpu_image_full.QuickAndDirtyWriteSlice("/tmp/oval_full_shift.mrc",1);

			cpu_work_full.SubtractImage(&cpu_image_full);

			t_rmsd = sqrtf(cpu_work_full.ReturnSumOfSquares(0,0,0,0,false));
			wxPrintf("RMSD between cpu phase shift and gpu phase shift for 512 x 512 is %3.3e\n", t_rmsd);


			// Half size image

			cpu_image_half.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/oval_half.mrc",1);
			// Copy of the image to operate on using cpu method
			cpu_work_half.CopyFrom(&cpu_image_half);

			// Initialize gpu image, and copy to device
			d_image_half.Init(cpu_image_half);
			d_image_half.CopyHostToDevice();

			cpu_work_half.PhaseShift(10, -200,0);
			d_image_half.PhaseShift(10,-200,0);

			d_image_half.Wait();
//			d_image_half.CopyDeviceToHost(true, true);
			d_image_half.CopyDeviceToHost(false, true);

			d_image_half.Wait();
			d_image_half.QuickAndDirtyWriteSlices("/tmp/oval_half_shift_fromGPU.mrc",1,1);

			cpu_image_half.QuickAndDirtyWriteSlice("/tmp/oval_half_shift.mrc",1);

			cpu_work_half.SubtractImage(&cpu_image_half);

			t_rmsd = sqrtf(cpu_work_half.ReturnSumOfSquares(0,0,0,0,false));
			wxPrintf("RMSD between cpu phase shift and gpu phase shift for 256 x 512 is %3.3e\n", t_rmsd);


		}


		if (do_fft || do_all_tests)
		{

			wxDateTime start;
			start = wxDateTime::Now();

			Image c;
			c.Allocate(514,514,1,true);


			// full size image

			// Read in the CPU image
			cpu_image_full.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/oval_full.mrc",1);
			cpu_image_full.AddConstant(-cpu_image_full.ReturnAverageOfRealValues(0.0f,false));

			// Copy of the image to operate on using cpu method
			cpu_work_full.CopyFrom(&cpu_image_full);

			// Re-use the image and library contexts - this needs to have debug asserts added'			'
			d_image_full.Init(cpu_image_full);

			d_image_full.CopyHostToDevice();

//			d_image_full.Mean();
//			d_image_full.AddConstant(-d_image_full.img_mean);
			bool doNorm = false;
			cpu_work_full.AddConstant(-cpu_work_full.ReturnAverageOfRealValues(0.0f,false));

			cpu_work_full.MultiplyByConstant(1.0f/sqrtf(cpu_work_full.ReturnSumOfSquares()));
			wxPrintf("cpu var before fft is %3.3e\n", (cpu_work_full.ReturnSumOfSquares()));

			cpu_work_full.ForwardFFT(doNorm);
			wxPrintf("cpu var after fft no norm is %3.3e or *= / n^2 %3.3e\n", (cpu_work_full.ReturnSumOfSquares()), cpu_work_full.ReturnSumOfSquares()/cpu_work_full.number_of_real_space_pixels/cpu_work_full.number_of_real_space_pixels);
			cpu_work_full.MultiplyByConstant(1.0f/sqrtf(cpu_work_full.ReturnSumOfSquares()));
			cpu_work_full.BackwardFFT();
			wxPrintf("cpu var after ifft with norm is %3.3e or *= / n^2 %3.3e\n", (cpu_work_full.ReturnSumOfSquares()), cpu_work_full.ReturnSumOfSquares()/(cpu_work_full.number_of_real_space_pixels)/cpu_work_full.number_of_real_space_pixels);


			d_image_full.MultiplyByConstant(1.0f/sqrtf(d_image_full.ReturnSumOfSquares()));
			wxPrintf("gpu var before fft is %3.3e\n", (d_image_full.ReturnSumOfSquares()));

			d_image_full.ForwardFFT(doNorm);
			wxPrintf("gpu var after fft no norm is %3.3e or *= / n %3.3e\n", (d_image_full.ReturnSumSquareModulusComplexValues()), d_image_full.ReturnSumSquareModulusComplexValues()/(d_image_full.number_of_real_space_pixels));
			d_image_full.MultiplyByConstant(1.0f/sqrtf(d_image_full.ReturnSumSquareModulusComplexValues()));
			d_image_full.BackwardFFT();
			wxPrintf("gpu var after ifft with norm is %3.3e or *= / n %3.3e\n", (d_image_full.ReturnSumOfSquares()), d_image_full.ReturnSumOfSquares()/(d_image_full.number_of_real_space_pixels));

			exit(0);



			cpu_work_full.ForwardFFT(true);
			cpu_work_full.BackwardFFT();

			d_image_full.ForwardFFT(true);
			d_image_full.BackwardFFT();

			d_image_full.Wait();

			d_image_full.CopyDeviceToHost(true, true);
			d_image_full.Wait();

			cpu_image_full.QuickAndDirtyWriteSlice("/tmp/oval_full_fft_ifft.mrc",1);
			cpu_work_full.QuickAndDirtyWriteSlice("/tmp/oval_full_work_fft_ifft.mrc",1);

			cpu_work_full.SubtractImage(&cpu_image_full);
//			cpu_work_full.DivideByConstant(cpu_work_full.number_of_real_space_pixels);

			t_rmsd = sqrtf(cpu_work_full.ReturnSumOfSquares(0,0,0,0,false));
			wxPrintf("RMSD between cpu fft/ifft and gpu  fft/ifft  for 512 x 512 is %3.3e\n", t_rmsd);


			// Half size image

			cpu_image_half.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/oval_half.mrc",1);

			// Copy of the image to operate on using cpu method
			cpu_work_half.CopyFrom(&cpu_image_half);

			// Re-use the image and library contexts - this needs to have debug asserts added
			d_image_half.CopyHostToDevice();

			cpu_work_half.ForwardFFT(true);
			cpu_work_half.BackwardFFT();

			d_image_half.ForwardFFT(true);
			d_image_half.BackwardFFT();

			d_image_half.Wait();

			d_image_half.CopyDeviceToHost(true, true);
			d_image_half.Wait();

			cpu_image_half.QuickAndDirtyWriteSlice("/tmp/oval_half_fft_ifft.mrc",1);
			cpu_work_half.QuickAndDirtyWriteSlice("/tmp/oval_half_work_fft_ifft.mrc",1);


			cpu_work_half.SubtractImage(&cpu_image_half);

			t_rmsd = sqrtf(cpu_work_half.ReturnSumOfSquares(0,0,0,0,false));
			wxPrintf("RMSD between cpu fft/ifft and gpu  fft/ifft  for 256 x 512 is %3.3e\n", t_rmsd);


		}

		if (do_scale || do_all_tests)
		{

			// full size image

			// Read in the CPU image
			cpu_image_full.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/oval_full.mrc",1);

			// Copy of the image to operate on using cpu method
			cpu_work_full.CopyFrom(&cpu_image_full);
			// Re-use the image and library contexts - this needs to have debug asserts added
			d_image_full.CopyHostToDevice();

			cpu_work_full.MultiplyByConstant(PIf);
			d_image_full.MultiplyByConstant(PIf);
			d_image_full.Wait();




			wxPrintf("Real sums are cpu: %f gpu: %f\n",cpu_work_full.ReturnSumOfRealValues(), d_image_full.ReturnSumOfRealValues());
			d_image_full.Wait();

			cpu_work_full.ForwardFFT(true);
			d_image_full.ForwardFFT(true);

			wxPrintf("Complex sums are cpu: %4.4e gpu: %4.4e\n",cpu_work_full.ReturnSumOfSquares(), d_image_full.ReturnSumSquareModulusComplexValues());
			d_image_full.Wait();

			cpu_work_full.BackwardFFT();
			d_image_full.BackwardFFT();

			d_image_full.Wait();



		}

		if (do_pad || do_all_tests)
		{

			// full size image

			// Read in the CPU image
			cpu_image_full.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/oval_full.mrc",1);

			// Copy of the image to operate on using cpu method
			cpu_work_full.CopyFrom(&cpu_image_full);
			// Re-use the image and library contexts - this needs to have debug asserts added
			d_image_full.CopyHostToDevice();

			Image padded;
			Image padded_work;
			padded.Allocate(1024,768,1,true);
			padded.SetToConstant(0.0f);
			padded_work.CopyFrom(&padded);
			GpuImage d_padded(padded);
			d_padded.CopyHostToDevice();

			cpu_work_full.ClipInto(&padded_work,1,false,0.0f,10,-30,0);

			d_image_full.ClipInto(&d_padded,1,false,0.0f,10,-30,0);
			d_padded.Wait();
			d_padded.CopyDeviceToHost(true, true);

			//			d_fft.ClipInto(&d_padded,-1,0,128,0);



			d_padded.Wait();


			padded_work.SubtractImage(&padded);
//			cpu_work_full.DivideByConstant(cpu_work_full.number_of_real_space_pixels);

			t_rmsd = sqrtf(padded_work.ReturnSumOfSquares(0,0,0,0,false));
			wxPrintf("RMSD between padded images 512/512-->1024x768 is %3.3e\n", t_rmsd);



		}
//

		if (do_swap || do_all_tests)
		{

			// full size image

			// Read in the CPU image
			cpu_image_full.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/oval_full.mrc",1);

			// Copy of the image to operate on using cpu method
			cpu_work_full.CopyFrom(&cpu_image_full);
			// Re-use the image and library contexts - this needs to have debug asserts added
			d_image_full.CopyHostToDevice();

			cpu_work_full.SwapRealSpaceQuadrants();
			d_image_full.SwapRealSpaceQuadrants();
			d_image_full.Wait();




			wxPrintf("Real sums after swapping are cpu: %f gpu: %f\n",cpu_work_full.ReturnSumOfRealValues(), d_image_full.ReturnSumOfRealValues());
			d_image_full.Wait();




		}

//	wxPrintf("Making an image of zeros in host memory\n");
//	Image zeros;
//	zeros.Allocate(512,512,512,true);
//	zeros.SetToConstant(1.f);
//
//	Image zeros2;
//	zeros2.real_values = zeros.real_values;
//
//	  wxPrintf("\n\nhost ZEROS2 %p with value %f\n\n", &zeros2.real_values,  zeros2.real_values[0]);
//
//
//	wxPrintf("Initializing a GpuImage\n");
//
//	GpuImage d_zeros;
//
//	wxPrintf("Set up a GpuImage with size %d\n", d_zeros.logical_x_dimension);
//	d_zeros.CopyFromCpuImage(zeros);
//
//
//	wxPrintf("copied the Image into the GpuImage with size %d\n", d_zeros.logical_x_dimension);
//
////	d_zeros.CopyVolumeHostToDevice();
//	d_zeros.CopyHostToDevice();
//
//	wxPrintf("I copied to the device\n");
//
//	wxPrintf("Now I'll try to multipy by 5\n");
//	d_zeros.MultiplyByScalar(5.0f);
//
//	wxPrintf("Now I'll try to copy to the host\n");
////	d_zeros.CopyVolumeDeviceToHost();
//	d_zeros.CopyDeviceToHost();
//
//
//	wxPrintf("Checking the values in the copied array %f\n",d_zeros.real_values[10]);
//
//	zeros.real_values = d_zeros.real_values;
//
//	zeros.QuickAndDirtyWriteSlices("TestGpuOutx5.mrc",1,512);

    } // end of parallel omp block

}
