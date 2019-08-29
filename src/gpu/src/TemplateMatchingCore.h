#ifndef TemplateMatchingCore_H_
#define TemplateMatchingCore_H_


class TemplateMatchingCore {

public:

	TemplateMatchingCore() ;
	TemplateMatchingCore(int number_of_jobs);
	virtual ~TemplateMatchingCore() ;

	void Init(int number_of_jobs);

	DeviceManager gpuDev;

	int nGPUs;
	int nThreads;
	int number_of_jobs_per_image_in_gui;

	// CPU images to be passed in -
	Image template_reconstruction;
	Image current_projection;
	Image input_image; // These will be modified on the host from withing Template Matching Core so Allocate locally



	// These are assumed to be empty containers at the outset, so xfer host-->device is skipped
	GpuImage d_max_intensity_projection;
	GpuImage d_best_psi;
	GpuImage d_best_phi;
	GpuImage d_best_theta;
	GpuImage d_best_defocus;
	GpuImage d_best_pixel_size;

	const int nSums = 5;
	GpuImage* d_sum;
	GpuImage* d_sumSq;
	bool is_allocated_sum_buffer = false;
	bool* is_non_zero_sum_buffer;

	//  GpuImage d_sum1, d_sum2, d_sum3, d_sum4, d_sum5;
	//  GpuImage d_sumSq1, d_sumSq2, d_sumSq3, d_sumSq4, d_sumSq5;

	// This will need to be copied in
	GpuImage d_input_image;
	GpuImage d_current_projection;
	GpuImage d_padded_reference;


	// Search range parameters
	float pixel_size_search_range;
	float pixel_size_step;
	float pixel_size;
	float defocus_search_range;
	float defocus_step;
	float defocus1;
	float defocus2;
	float psi_max;
	float psi_start;
	float psi_step;

	float c_defocus;
	float c_pixel;

	int current_search_position;
	int first_search_position;
	int last_search_position;
	long total_number_of_cccs_calculated;

	Histogram histogram;
	Npp32s* cummulative_histogram;
	bool is_allocated_cummulative_histogram = false;

	// Search objects
	AnglesAndShifts angles;
	EulerSearch global_euler_search;

	int dummy;


	void Init(Image &template_reconstruction,
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
			int first_search_position,
			int last_search_position);

	void RunInnerLoop(Image &projection_filter, float pixel_i, float defocus_i, int threadIDX);


private:


};

#endif
