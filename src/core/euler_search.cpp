#include "core_headers.h"

EulerSearch::EulerSearch()
{
	// Nothing to do until Init is called
	refine_top_N = 0;
	number_of_search_dimensions = 0;
	number_of_search_positions = 0;
	best_parameters_to_keep = 0;
	list_of_search_parameters = NULL;
	list_of_best_parameters = NULL;
//	kernel_index = NULL;
//	best_values = NULL;
//	starting_values = NULL;
	parameter_map = NULL;
	angular_step_size = 0.0;
	phi_max = 0.0;
	phi_start = 0.0;
	theta_max = 0.0;
	theta_start = 0.0;
	psi_max = 0.0;
	psi_step = 0.0;
	psi_start = 0.0;
	resolution_limit = 0.0;
//	best_score = - std::numeric_limits<float>::max();
	test_mirror = false;
	max_search_x = 0.0;
	max_search_y = 0.0;
}

EulerSearch::~EulerSearch()
{
//	if (best_values != NULL) delete [] best_values;
	if (parameter_map != NULL) delete [] parameter_map;
	if (list_of_search_parameters != NULL) Deallocate2DFloatArray(list_of_search_parameters, number_of_search_positions);
	if (list_of_best_parameters != NULL) Deallocate2DFloatArray(list_of_best_parameters, best_parameters_to_keep + 1);
}

void EulerSearch::Init(float wanted_resolution_limit, bool *wanted_parameter_map, int wanted_parameters_to_keep)
{
	MyDebugAssertTrue(number_of_search_positions == 0,"Euler search object is already setup");
	MyDebugAssertTrue(wanted_parameters_to_keep > 0,"Must at least keep one set of best parameters");

	int i;

//	best_values    						= 	new float[5];
	parameter_map						= 	new bool[5];

// Need to add 1 to index of input parameter map since this refers to line in parameter file
	for (i = 0; i < 5; i++) {parameter_map[i] = wanted_parameter_map[i + 1];};
	resolution_limit = wanted_resolution_limit;

	best_parameters_to_keep = wanted_parameters_to_keep;
	if (list_of_best_parameters != NULL) Deallocate2DFloatArray(list_of_best_parameters, best_parameters_to_keep + 1);
	Allocate2DFloatArray(list_of_best_parameters, best_parameters_to_keep + 1, 6);
}

void EulerSearch::InitGrid(wxString wanted_symmetry_symbol, float wanted_angular_step_size, float wanted_phi_start, float wanted_theta_start, float wanted_psi_max, float wanted_psi_step, float wanted_psi_start, float wanted_resolution_limit, bool *wanted_parameter_map, int wanted_parameters_to_keep)
{
	if (number_of_search_positions == 0) Init(wanted_resolution_limit, wanted_parameter_map, wanted_parameters_to_keep);

	angular_step_size = wanted_angular_step_size;
	phi_start = wanted_phi_start;
	theta_start = wanted_theta_start;
	psi_max = wanted_psi_max;
	psi_step = wanted_psi_step;
	psi_start = wanted_psi_start;
	symmetry_symbol = wanted_symmetry_symbol;

	SetSymmetryLimits();
	CalculateGridSearchPositions();
}

// This method has not been tested
void EulerSearch::InitRandom(wxString wanted_symmetry_symbol, float wanted_psi_step, int wanted_number_of_search_positions, float wanted_resolution_limit, bool *wanted_parameter_map, int wanted_parameters_to_keep)
{
	int i;

	if (number_of_search_positions == 0) Init(wanted_resolution_limit, wanted_parameter_map, wanted_parameters_to_keep);

	number_of_search_positions = wanted_number_of_search_positions;

	psi_step = wanted_psi_step;
	symmetry_symbol = wanted_symmetry_symbol;

	SetSymmetryLimits();
	CalculateRandomSearchPositions();
}

void EulerSearch::CalculateGridSearchPositions()
{
	int i;
	float phi_step = 360.0;
	float theta_step = 360.0;
	float phi;
	float theta;
	float theta_max_local;
	float theta_start_local;
	float phi_start_local;

//	phi_max = 360.0;
	theta_max_local = theta_max;

	if (! parameter_map[0]) phi_start_local = phi_start;
	if (parameter_map[1])
	{
//		theta_max_local = 90.0;
		// make sure that theta_step produces an integer number of steps
		theta_step = theta_max_local / int(theta_max_local / angular_step_size + 0.5);
		theta_start_local = fabsf(theta_step / 2.0 * global_random_number_generator.GetUniformRandom());
	}
	else
	{
		theta_start_local = theta_start;
		theta_max_local = theta_start;
	}

	if (list_of_search_parameters != NULL) Deallocate2DFloatArray(list_of_search_parameters, number_of_search_positions);
//	{
//		for (i = 0; i < 2; ++i)
//		{
//			delete [] list_of_search_parameters[i];						// delete inner arrays of floats
//		}
//		delete [] list_of_search_parameters;							// delete array of pointers to float arrays
//	}

	number_of_search_positions = 0;
	for (theta = theta_start_local; theta < theta_max_local + theta_step / 2.0; theta += theta_step)
	{
		if (parameter_map[0])
		{
	        if (theta == 0.0 || theta == 180.0)
	        {
	        	phi_step = phi_max;
	        }
	        else
	        {
	        	// angular sampling was adapted from Spider subroutine VOEA (Paul Penczek)
	        	phi_step = fabsf(angular_step_size / sinf(deg_2_rad(theta)));
	        	if (phi_step > phi_max) phi_step = phi_max;
	    		phi_step = phi_max / int(phi_max / phi_step + 0.5);
	        }
		}
        for (phi = 0; phi < phi_max; phi += phi_step)
		{
        	number_of_search_positions++;
		}
	}

	Allocate2DFloatArray(list_of_search_parameters, number_of_search_positions, 2);
//	list_of_search_parameters = new float* [2];							// dynamic array (size 2) of pointers to float

//	for (i = 0; i < 2; ++i)
//	{
//		list_of_search_parameters[i] = new float[number_of_search_positions];	// each i-th pointer is now pointing to dynamic array (size number_of_search_positions) of actual float values
//	}

	number_of_search_positions = 0;
	for (theta = theta_start_local; theta < theta_max_local + theta_step / 2.0; theta += theta_step)
	{
		if (parameter_map[0])
		{
	        if (theta == 0.0 || theta == 180.0)
	        {
	        	phi_step = phi_max;
	        }
	        else
	        {
	        	// angular sampling was adapted from Spider subroutine VOEA (Paul Penczek)
	        	phi_step = fabsf(angular_step_size / sinf(deg_2_rad(theta)));
	        	if (phi_step > phi_max) phi_step = phi_max;
	    		phi_step = phi_max / int(phi_max / phi_step + 0.5);
	    		phi_start_local = phi_step / 2.0 * global_random_number_generator.GetUniformRandom();
	        }
		}
        for (phi = 0.0; phi < phi_max; phi += phi_step)
		{
        	list_of_search_parameters[number_of_search_positions][0] = phi + phi_start_local;
        	list_of_search_parameters[number_of_search_positions][1] = theta;
        	number_of_search_positions++;
		}
	}

	if (! parameter_map[2])
	{
		test_mirror = false;
	}
	wxPrintf("Number of global search views = %i\n", number_of_search_positions);
}

void EulerSearch::CalculateRandomSearchPositions()
{
	int i;
	int trials = 0;
	int max_trials = 1000 * number_of_search_positions;
	int number_of_positions = 0;
	float theta;
	float phi;

//	phi_max = 360.0;
//	theta_max = 90.0;

	if (list_of_search_parameters != NULL) Deallocate2DFloatArray(list_of_search_parameters, number_of_search_positions);
//	{
//		for (i = 0; i < 2; ++i)
//		{
//			delete [] list_of_search_parameters[i];						// delete inner arrays of floats
//		}
//		delete [] list_of_search_parameters;							// delete array of pointers to float arrays
//	}

	Allocate2DFloatArray(list_of_search_parameters, number_of_search_positions, 2);
//	list_of_search_parameters = new float* [2];							// dynamic array (size 2) of pointers to float

//	for (i = 0; i < 2; ++i)
//	{
//		list_of_search_parameters[i] = new float[number_of_search_positions];	// each i-th pointer is now pointing to dynamic array (size number_of_positions) of actual float values
//	}

	while (number_of_positions < number_of_search_positions)
	{
		if (parameter_map[0])
		{
			phi = phi_max * (global_random_number_generator.GetUniformRandom() + 1.0) / 2.0;
		}
		else
		{
			phi = phi_start;
		}
		if (parameter_map[1])
		{
			theta = theta_max * (global_random_number_generator.GetUniformRandom() + 1.0) / 2.0;
		}
		else
		{
			theta = theta_start;
		}
		if (theta == 0.0)
		{
			list_of_search_parameters[number_of_positions][0] = phi;
			list_of_search_parameters[number_of_positions][1] = 0.0;
			number_of_positions++;
		}
		else
		if (global_random_number_generator.GetUniformRandom() > sinf(deg_2_rad(theta)) || trials > max_trials)
		{
			list_of_search_parameters[number_of_positions][0] = phi;
			list_of_search_parameters[number_of_positions][1] = theta;
			number_of_positions++;
		}
		trials++;
	}

	if (! parameter_map[2])
	{
		test_mirror = false;
	}
}

void EulerSearch::SetSymmetryLimits()
{
// Frealign limits, original code written by Richard Henderson
//    DATA  ASYMTEST/' CDTOI0123456789'/
//    DATA  THETASTORE/90.0,  90.0,  54.7,  54.7,  31.7/
//    DATA  PHISTORE/360.0,  360.0, 180.0,  90.0, 180.0/
//    DATA  JSTORE/2,1,1,1,1/

	wxChar   symmetry_type;
	long     symmetry_number;

	if (symmetry_symbol.Length() < 1)
	{
		MyPrintWithDetails("Error: Must specify symmetry symbol\n");
		abort();
	}
	symmetry_type = symmetry_symbol.Capitalize()[0];
	if (symmetry_symbol.Length() == 1)
	{
		symmetry_number = 0;
	}
	else
	{
		if (! symmetry_symbol.Mid(1).ToLong(&symmetry_number))
		{
			MyPrintWithDetails("Error: Invalid n after symmetry symbol\n");
			abort();
		}
	}

	if (symmetry_type == 'C')
	{
		if (symmetry_number == 0)
		{
			MyPrintWithDetails("Error: Invalid n after symmetry symbol\n");
			abort();
		}

		phi_max = 360.0 / symmetry_number;
		theta_max = 90.0;
		test_mirror = true;

		return;
	}

	if (symmetry_type == 'D')
	{
		if (symmetry_number == 0)
		{
			MyPrintWithDetails("Error: Invalid n after symmetry symbol\n");
			abort();
		}

		phi_max = 360.0 / symmetry_number;
		theta_max = 90.0;
		test_mirror = false;

		return;
	}

	if (symmetry_type == 'T')
	{
		phi_max = 180.0;
		theta_max = 54.7;
		test_mirror = false;

		return;
	}

	if (symmetry_type == 'O')
	{
		phi_max = 90.0;
		theta_max = 54.7;
		test_mirror = false;

		return;
	}

	if (symmetry_type == 'I')
	{
		phi_max = 180.0;
		theta_max = 31.7;
		test_mirror = false;

		return;
	}

	MyPrintWithDetails("Error: Invalid symmetry symbol\n");
	abort();
}

// Run the search
void EulerSearch::Run(Particle &particle, Image &input_3d, float *starting_values, Image *projections, Kernel2D **kernel_index)
{
	MyDebugAssertTrue(number_of_search_positions > 0,"EulerSearch not initialized");
	MyDebugAssertTrue(particle.particle_image->is_in_memory,"Particle image not allocated");
	MyDebugAssertTrue(input_3d.is_in_memory,"3D reference map not allocated");
	MyDebugAssertTrue(particle.particle_image->logical_x_dimension == input_3d.logical_x_dimension && particle.particle_image->logical_y_dimension == input_3d.logical_y_dimension, "Error: Image and 3D reference incompatible");

	int i;
	int j;
	int k;
	int sample_rate = 2;
	int pixel_counter;
	int psi_i;
	int psi_m;
	int number_of_psi_positions;
	int max_pix_x = max_search_x / particle.pixel_size;
	int max_pix_y = max_search_y / particle.pixel_size;
	float psi;
//	float psi_max = 360.0;
//	float psi_step = 1.0;
	float best_inplane_score;
	float best_inplane_values[3];
	float temp_float[6];
	bool mirrored_match;
	Peak found_peak;
	AnglesAndShifts angles;
	Image *flipped_image = new Image;
	Image *projection_image = new Image;
	Image *rotated_image = new Image;
	Image *correlation_map = new Image;
//	Image *sampled_image = new Image;
	Image *rotation_cache = NULL;
	flipped_image->Allocate(input_3d.logical_x_dimension, input_3d.logical_y_dimension, false);
	projection_image->Allocate(input_3d.logical_x_dimension, input_3d.logical_y_dimension, false);
	rotated_image->Allocate(input_3d.logical_x_dimension, input_3d.logical_y_dimension, false);
	correlation_map->Allocate(input_3d.logical_x_dimension, input_3d.logical_y_dimension, false);
//	sampled_image->Allocate(int(input_3d.logical_x_dimension / sample_rate), int(input_3d.logical_y_dimension / sample_rate), false);
	correlation_map->object_is_centred_in_box = false;
	#ifndef MKL
		float *temp_k1 = new float [flipped_image->real_memory_allocated];
//		float *temp_k2 = new float [flipped_image->real_memory_allocated];
		float *temp_k2; temp_k2 = temp_k1 + 1;
//		float *temp_k3 = new float [flipped_image->real_memory_allocated];
		float *real_a;
		float *real_b;
		float *real_c;
		float *real_d;
		float *real_r;
		float *real_i;
		real_a = projection_image->real_values;
		real_b = projection_image->real_values + 1;
		real_r = correlation_map->real_values;
		real_i = correlation_map->real_values + 1;
	#endif

//	temp_k1[0] = 10;
//	temp_k1[1] = 11;
//	temp_k1[2] = 12;
//	wxPrintf("k10, k11, k12, k20, k21 = %f, %f, %f, %f, %f\n",temp_k1[0], temp_k1[1], temp_k1[2], temp_k2[0], temp_k2[1]);
//	wxPrintf("0, 1, 2 = %g, %g, %g\n", particle.particle_image->real_values[0], particle.particle_image->real_values[1], particle.particle_image->real_values[2]);
//	wxPrintf("0, 1, 2 = %g, %g, %g\n", real_a[0], real_a[1], real_a[2]);
//	exit(0);

	for (i = 0; i < best_parameters_to_keep + 1; ++i)
	{
		ZeroFloatArray(list_of_best_parameters[i], 5);
		list_of_best_parameters[i][5] = - std::numeric_limits<float>::max();
	}

	if (parameter_map[2])
	{
		number_of_psi_positions = myroundint(psi_max / psi_step);
		if (number_of_psi_positions < 1) number_of_psi_positions = 1;
	}
	else
	{
		number_of_psi_positions = 1;
	}
	psi_i = number_of_psi_positions;
	if (test_mirror) psi_i *= 2;
	rotation_cache = new Image [psi_i];
	for (i = 0; i < psi_i; i++)
	{
		rotation_cache[i].Allocate(input_3d.logical_x_dimension, input_3d.logical_y_dimension, false);
	}

//	best_score = - std::numeric_limits<float>::max();
	flipped_image->CopyFrom(particle.particle_image);
//	flipped_image->PhaseFlipPixelWise(*particle.ctf_image);

//	psi_i = 0;
	psi_m = 0;
	for (psi_i = 0; psi_i < number_of_psi_positions; psi_i++)
	{
//		wxPrintf("rotation_cache[psi_m].logical_z_dimension = %i\n", rotation_cache[psi_m].logical_z_dimension);
		if (parameter_map[2])
		{
			flipped_image->RotateFourier2DFromIndex(rotation_cache[psi_m], kernel_index[psi_i]);
		}
		else
		{
			angles.GenerateRotationMatrix2D(psi_start);
			flipped_image->RotateFourier2D(rotation_cache[psi_m], angles);
		}
//		for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated / 2; pixel_counter++)
//		{
//			rotation_cache[psi_m].complex_values[pixel_counter] = conjf(rotation_cache[psi_m].complex_values[pixel_counter]);
//		}
//		psi_i++;
		psi_m++;
		if (test_mirror)
		{
			rotation_cache[psi_m - 1].MirrorYFourier2D(rotation_cache[psi_m]);
			psi_m++;
		}
	}

	for (i = 0; i < number_of_search_positions; i++)
	{
		if (projections == NULL)
		{
//			wxPrintf("i, phi, theta = %i, %f, %f\n", i, list_of_search_parameters[i][0], list_of_search_parameters[i][1]);
			angles.Init(list_of_search_parameters[i][0], list_of_search_parameters[i][1], 0.0, 0.0, 0.0);
			input_3d.ExtractSlice(*projection_image, angles, resolution_limit);
			projection_image->Whiten(resolution_limit);
		}
		else
		{
			projection_image->CopyFrom(&projections[i]);
			#ifndef MKL
				for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k1[pixel_counter] = real_a[pixel_counter] + real_b[pixel_counter];};
				for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k2[pixel_counter] = real_b[pixel_counter] - real_a[pixel_counter];};
			#endif
		}

		best_inplane_score = - std::numeric_limits<float>::max();
		psi_m = 0;
		for (psi_i = 0; psi_i < number_of_psi_positions; psi_i++)
		{
			#ifndef MKL
				real_c = rotation_cache[psi_m].real_values;
				real_d = rotation_cache[psi_m].real_values + 1;
			#endif

//			projection_image->RotateFourier2DFromIndex(*quad_rot_image, kernel_index[psi_m]);

			// test 90, 180 and 270 degree rotated references
//			for (quad_i = 0; quad_i < 3; quad_i++)
//			{
//				quad_rot_image->RotateQuadrants(*rotated_image, quad_i * 90);
//				(a + ib)(c -id) = ac + bd +i(bc - ad)
//				k1 = a(c - d)
//				k2 = d(a + b)
//				k3 = c(b - a)
//				R = k1 + k2 , and I = k1 + k3
//			for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k1[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]);};
//			for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k2[pixel_counter] = real_d[pixel_counter] * (real_a[pixel_counter] + real_b[pixel_counter]);};
//			for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k3[pixel_counter] = real_c[pixel_counter] * (real_b[pixel_counter] - real_a[pixel_counter]);};
			#ifdef MKL
				// Use the MKL
				vmcMulByConj(flipped_image->real_memory_allocated/2,reinterpret_cast <MKL_Complex8 *> (projection_image->complex_values),reinterpret_cast <MKL_Complex8 *> (rotation_cache[psi_m].complex_values),reinterpret_cast <MKL_Complex8 *> (correlation_map->complex_values),VML_EP|VML_FTZDAZ_ON|VML_ERRMODE_IGNORE);
			#else
				for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_d[pixel_counter] * temp_k1[pixel_counter];};
				for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_c[pixel_counter] * temp_k2[pixel_counter];};
			#endif
	//		for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_d[pixel_counter] * (real_a[pixel_counter] + real_b[pixel_counter]);};
	//		for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_c[pixel_counter] * (real_b[pixel_counter] - real_a[pixel_counter]);};
	//			for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated / 2; pixel_counter++)
	//				{correlation_map->complex_values[pixel_counter] = projection_image->complex_values[pixel_counter] * rotation_cache[psi_m].complex_values[pixel_counter];}
				correlation_map->is_in_real_space = false;
//				correlation_map->BackwardFFT();
//				correlation_map->SetToConstant(0.0);
//				correlation_map->real_values[47] = 1.0;
//				correlation_map->ForwardFFT();
	//			correlation_map->SampleFFT(*sampled_image, sample_rate);
//				sampled_image->SwapRealSpaceQuadrants();
	//			sampled_image->BackwardFFT();
//				sampled_image->QuickAndDirtyWriteSlice("junks.mrc",1);
/*				if (fabsf(list_of_search_parameters[i][1] - 90.0) < 10.0)
				{
				rotation_cache[psi_m].SwapRealSpaceQuadrants();
				rotation_cache[psi_m].BackwardFFT();
				rotation_cache[psi_m].QuickAndDirtyWriteSlice("part.mrc",1);
				projection_image->SwapRealSpaceQuadrants();
				projection_image->BackwardFFT();
				projection_image->QuickAndDirtyWriteSlice("proj.mrc",1);
				correlation_map->SwapRealSpaceQuadrants();
				correlation_map->BackwardFFT();
				correlation_map->QuickAndDirtyWriteSlice("cros.mrc",1);
				exit(0);
				}
				else
				{ */
				correlation_map->complex_values[0] = 0.0;
				correlation_map->BackwardFFT();
//				}
	//			found_peak = sampled_image->FindPeakAtOriginFast2D(0.5 * correlation_map->physical_address_of_box_center_x);
				found_peak = correlation_map->FindPeakAtOriginFast2D(max_pix_x, max_pix_y);
//				wxPrintf("peak  = %g  psi = %g  theta = %g  phi = %g  x = %g  y = %g\n", found_peak.value, 360.0 - (psi_i * psi_step + psi_start),
//						list_of_search_parameters[i][1], list_of_search_parameters[i][0], found_peak.x, found_peak.y);
				if (found_peak.value > best_inplane_score)
				{
					best_inplane_score =  found_peak.value;
					best_inplane_values[0] = 360.0 - (psi_i * psi_step + psi_start);
//					best_inplane_values[0] = 360.0 - (psi_i * psi_step + quad_i * 90);
					best_inplane_values[1] = found_peak.x;
					best_inplane_values[2] = found_peak.y;
					mirrored_match = false;
				}

				if (test_mirror)
				{
					psi_m++;

					#ifndef MKL
						real_c = rotation_cache[psi_m].real_values;
						real_d = rotation_cache[psi_m].real_values + 1;
					#endif

//					rotation_cache[psi_m].MirrorYFourier2D(*correlation_map);
//					for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated / 2; pixel_counter++)
//					{
//						correlation_map->complex_values[pixel_counter] *= conjf(flipped_image->complex_values[pixel_counter]);
//					}
//					for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k1[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]);};
//					for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k2[pixel_counter] = real_d[pixel_counter] * (real_a[pixel_counter] + real_b[pixel_counter]);};
//					for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k3[pixel_counter] = real_c[pixel_counter] * (real_b[pixel_counter] - real_a[pixel_counter]);};
					#ifdef MKL
						// Use the MKL
						vmcMulByConj(flipped_image->real_memory_allocated/2,reinterpret_cast <MKL_Complex8 *> (projection_image->complex_values),reinterpret_cast <MKL_Complex8 *> (rotation_cache[psi_m].complex_values),reinterpret_cast <MKL_Complex8 *> (correlation_map->complex_values),VML_EP|VML_FTZDAZ_ON|VML_ERRMODE_IGNORE);
					#else
						for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_d[pixel_counter] * temp_k1[pixel_counter];};
						for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_c[pixel_counter] * temp_k2[pixel_counter];};
					#endif
	//				for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_d[pixel_counter] * (real_a[pixel_counter] + real_b[pixel_counter]);};
	//				for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_c[pixel_counter] * (real_b[pixel_counter] - real_a[pixel_counter]);};
	//				for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated / 2; pixel_counter++)
	//					{correlation_map->complex_values[pixel_counter] = projection_image->complex_values[pixel_counter] * rotation_cache[psi_m].complex_values[pixel_counter];}
					correlation_map->is_in_real_space = false;
	//				correlation_map->SampleFFT(*sampled_image, sample_rate);
	//				sampled_image->BackwardFFT();
	//				found_peak = sampled_image->FindPeakAtOriginFast2D(0.5 * correlation_map->physical_address_of_box_center_x);
					correlation_map->complex_values[0] = 0.0;
					correlation_map->BackwardFFT();
					found_peak = correlation_map->FindPeakAtOriginFast2D(max_pix_x, max_pix_y);
//					wxPrintf("peakm = %g  psi = %g  theta = %g  phi = %g  x = %g  y = %g\n", found_peak.value, 360.0 - (psi_i * psi_step + psi_start),
//							list_of_search_parameters[i][1] + 180.0, list_of_search_parameters[i][0], found_peak.x, found_peak.y);
					if (found_peak.value > best_inplane_score)
					{
						best_inplane_score =  found_peak.value;
						best_inplane_values[0] = 360.0 - (psi_i * psi_step + psi_start);
						best_inplane_values[1] = found_peak.x;
						best_inplane_values[2] = found_peak.y;
						mirrored_match = true;
					}
				}
//			}
			psi_m++;
		}
		if (best_inplane_score > list_of_best_parameters[best_parameters_to_keep - 1][5])
		{
			list_of_best_parameters[best_parameters_to_keep - 1][5] =  best_inplane_score;
//			list_of_best_parameters[best_parameters_to_keep - 1][3] = (  best_inplane_values[1] * cosf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * sinf(deg_2_rad(best_inplane_values[0]))) * particle.pixel_size;
//			list_of_best_parameters[best_parameters_to_keep - 1][4] = (- best_inplane_values[1] * sinf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * cosf(deg_2_rad(best_inplane_values[0]))) * particle.pixel_size;
//			best_score =  best_inplane_score;
//			best_values[3] = (  best_inplane_values[1] * cosf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * sinf(deg_2_rad(best_inplane_values[0]))) * particle.pixel_size;
//			best_values[4] = (- best_inplane_values[1] * sinf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * cosf(deg_2_rad(best_inplane_values[0]))) * particle.pixel_size;
			if (mirrored_match)
			{
				list_of_best_parameters[best_parameters_to_keep - 1][0] = list_of_search_parameters[i][0];
				list_of_best_parameters[best_parameters_to_keep - 1][1] = list_of_search_parameters[i][1] + 180.0;
				list_of_best_parameters[best_parameters_to_keep - 1][2] = best_inplane_values[0];
				list_of_best_parameters[best_parameters_to_keep - 1][3] = (  best_inplane_values[1] * cosf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * sinf(deg_2_rad(best_inplane_values[0]))) * particle.pixel_size;
				list_of_best_parameters[best_parameters_to_keep - 1][4] = (- best_inplane_values[1] * sinf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * cosf(deg_2_rad(best_inplane_values[0]))) * particle.pixel_size;
//				best_values[0] = list_of_search_parameters[i][0];
//				best_values[1] = list_of_search_parameters[i][1] + 180.0;
//				best_values[2] = best_inplane_values[0];
//				wxPrintf("i = %i, best_score_m = %f, best values = %f, %f, %f, %f, %f\n", i, list_of_best_parameters[best_parameters_to_keep - 1][5],
//						list_of_best_parameters[best_parameters_to_keep - 1][2], list_of_best_parameters[best_parameters_to_keep - 1][1], list_of_best_parameters[best_parameters_to_keep - 1][0],
//						list_of_best_parameters[best_parameters_to_keep - 1][3], list_of_best_parameters[best_parameters_to_keep - 1][4]);
/*				if (i == 0)
				{
					rotation_cache[0].SwapRealSpaceQuadrants();
					rotation_cache[0].BackwardFFT();
					rotation_cache[0].QuickAndDirtyWriteSlice("part.mrc",1);
					rotation_cache[0].ForwardFFT();
					rotation_cache[0].SwapRealSpaceQuadrants();
					projection_image->SwapRealSpaceQuadrants();
					projection_image->BackwardFFT();
					projection_image->QuickAndDirtyWriteSlice("proj.mrc",1);
					projection_image->ForwardFFT();
					projection_image->SwapRealSpaceQuadrants();
				} */
			}
			else
			{
				list_of_best_parameters[best_parameters_to_keep - 1][0] = list_of_search_parameters[i][0];
				list_of_best_parameters[best_parameters_to_keep - 1][1] = list_of_search_parameters[i][1];
				list_of_best_parameters[best_parameters_to_keep - 1][2] = best_inplane_values[0];
				list_of_best_parameters[best_parameters_to_keep - 1][3] = (- best_inplane_values[1] * cosf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * sinf(deg_2_rad(best_inplane_values[0]))) * particle.pixel_size;
				list_of_best_parameters[best_parameters_to_keep - 1][4] = (  best_inplane_values[1] * sinf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * cosf(deg_2_rad(best_inplane_values[0]))) * particle.pixel_size;
//				best_values[0] = list_of_search_parameters[i][0];
//				best_values[1] = list_of_search_parameters[i][1];
//				best_values[2] = best_inplane_values[0];
//				wxPrintf("i = %i, best_score   = %f, best values = %f, %f, %f, %f, %f\n", i, list_of_best_parameters[best_parameters_to_keep - 1][5],
//						list_of_best_parameters[best_parameters_to_keep - 1][2], list_of_best_parameters[best_parameters_to_keep - 1][1], list_of_best_parameters[best_parameters_to_keep - 1][0],
//						list_of_best_parameters[best_parameters_to_keep - 1][3], list_of_best_parameters[best_parameters_to_keep - 1][4]);
			}
		}
		for (j = best_parameters_to_keep - 1; j > 0; j--)
		{
			if (list_of_best_parameters[j][5] > list_of_best_parameters[j - 1][5])
			{
				for (k = 0; k < 6; k++) {temp_float[k] = list_of_best_parameters[j - 1][k];}
				for (k = 0; k < 6; k++) {list_of_best_parameters[j - 1][k] = list_of_best_parameters[j][k];}
				for (k = 0; k < 6; k++) {list_of_best_parameters[j][k] = temp_float[k];}
//				if (j == 1)
//				{
//					wxPrintf("i = %i, best_score = %f, best values = %f, %f, %f, %f, %f\n", i, list_of_best_parameters[0][5],
//							list_of_best_parameters[0][2], list_of_best_parameters[0][1], list_of_best_parameters[0][0], list_of_best_parameters[0][3], list_of_best_parameters[0][4]);
//				}
			}
			else
			{
				break;
			}
		}
	}

	delete flipped_image;
	delete projection_image;
	delete rotated_image;
	delete correlation_map;
//	psi_i = myroundint(psi_max / psi_step);
	psi_i = number_of_psi_positions;
	if (test_mirror) psi_i *= 2;
	for (i = 0; i < psi_i; ++i)
	{
		rotation_cache[i].Deallocate();
	}
	delete [] rotation_cache;
	#ifndef MKL
		delete [] temp_k1;
	#endif
//	delete [] temp_k2;
//	delete [] temp_k3;
}
