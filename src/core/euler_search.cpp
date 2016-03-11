#include "core_headers.h"

EulerSearch::EulerSearch()
{
	// Nothing to do until Init is called
	refine_top_N = 0;
	number_of_search_dimensions = 0;
	number_of_search_positions = 0;
	list_of_search_parameters = NULL;
//	kernel_index = NULL;
	best_values = NULL;
	starting_values = NULL;
	parameter_map = NULL;
	angular_step_size = 0.0;
	psi_max = 360.0;
	psi_step = 0.0;
	resolution_limit = 0.0;
	best_score = - std::numeric_limits<float>::max();
	test_mirror = false;
}

EulerSearch::~EulerSearch()
{
	if (best_values != NULL) delete [] best_values;
	if (parameter_map != NULL) delete [] parameter_map;
	if (list_of_search_parameters != NULL)
	{
		for (int i = 0; i < 2; ++i)
		{
			delete [] list_of_search_parameters[i];						// delete inner arrays of floats
		}
		delete [] list_of_search_parameters;							// delete array of pointers to float arrays
	}
//	if (list_of_search_parameters != NULL) delete [] list_of_search_parameters;
}

void EulerSearch::Init(float wanted_resolution_limit, bool *wanted_parameter_map)
{
	MyDebugAssertTrue(number_of_search_positions == 0,"Euler search object is already setup");

	int i;

	best_values    						= 	new float[5];
	parameter_map						= 	new bool[5];

// Need to add 1 to index of input parameter map since this refers to line in parameter file
	for (i = 0; i < 5; i++) {parameter_map[i] = wanted_parameter_map[i + 1];};
	resolution_limit = wanted_resolution_limit;
}

void EulerSearch::InitGrid(float wanted_angular_step_size, float wanted_psi_step, float wanted_resolution_limit, bool *wanted_parameter_map)
{
	Init(wanted_resolution_limit, wanted_parameter_map);

	angular_step_size = wanted_angular_step_size;
	psi_step = wanted_psi_step;
	CalculateGridSearchPositions();
}

void EulerSearch::InitRandom(float wanted_psi_step, float *starting_values, int wanted_number_of_search_positions, float wanted_resolution_limit, bool *wanted_parameter_map)
{
	int i;

	Init(wanted_resolution_limit, wanted_parameter_map);

	if (list_of_search_parameters != NULL)
	{
		for (int i = 0; i < 2; ++i)
		{
			delete [] list_of_search_parameters[i];						// delete inner arrays of floats
		}
		delete [] list_of_search_parameters;							// delete array of pointers to float arrays
	}

	number_of_search_positions = wanted_number_of_search_positions;

	psi_step = wanted_psi_step;
	CalculateRandomSearchPositions();
}

void EulerSearch::CalculateGridSearchPositions()
{
	int i;
	int number_of_positions = 0;
	float phi_max = 0.0;
	float theta_max = 0.0;
	float phi_step;
	float theta_step;
	float phi = 0.0;
	float theta = 0.0;

	if (parameter_map[0]) phi_max = 360.0;
	if (parameter_map[1]) theta_max = 90.0;
	// make sure that theta_step produces an integer number of steps
	theta_step = theta_max / int(theta_max / angular_step_size);

	for (theta = 0.0; theta <= theta_max; theta += theta_step)
	{
        if (theta == 0.0 || theta == 180.0)
        {
        	phi_step = 360.0;
        }
        else
        {
        	// angular sampling was adapted from Spider subroutine VOEA (Paul Penczek)
    		phi_step = angular_step_size / sinf(deg_2_rad(theta));
    		phi_step = phi_max / int(phi_max / phi_step);
        }
        for (phi = 0.0; phi < phi_max; phi += phi_step)
		{
        	number_of_positions++;
		}
	}

	if (list_of_search_parameters != NULL)
	{
		for (i = 0; i < 2; ++i)
		{
			delete [] list_of_search_parameters[i];						// delete inner arrays of floats
		}
		delete [] list_of_search_parameters;							// delete array of pointers to float arrays
	}

	list_of_search_parameters = new float* [2];							// dynamic array (size 2) of pointers to float

	for (i = 0; i < 2; ++i)
	{
		list_of_search_parameters[i] = new float[number_of_positions];	// each i-th pointer is now pointing to dynamic array (size number_of_positions) of actual float values
	}
//	list_of_search_parameters  = new float* [2][number_of_positions];

	number_of_positions = 0;
	for (theta = 0.0; theta <= theta_max; theta += theta_step)
	{
        if (theta == 0.0 || theta == 180.0)
        {
        	phi_step = 360.0;
        }
        else
        {
        	// angular sampling was adapted from Spider subroutine VOEA (Paul Penczek)
    		phi_step = angular_step_size / sinf(deg_2_rad(theta));
    		phi_step = phi_max / int(phi_max / phi_step);
        }
        for (phi = 0.0; phi < phi_max; phi += phi_step)
		{
        	list_of_search_parameters[0][number_of_positions] = phi;
        	list_of_search_parameters[1][number_of_positions] = theta;
        	number_of_positions++;
		}
	}

	number_of_search_positions = number_of_positions;
	test_mirror = true;
	wxPrintf("Number of search positions = %i\n", number_of_search_positions);
}

void EulerSearch::CalculateRandomSearchPositions()
{
	int i;
	int trials = 0;
	int max_trials = 1000 * number_of_search_positions;
	int number_of_positions = 0;
	float theta;
	float phi;

	if (list_of_search_parameters != NULL)
	{
		for (i = 0; i < 2; ++i)
		{
			delete [] list_of_search_parameters[i];						// delete inner arrays of floats
		}
		delete [] list_of_search_parameters;							// delete array of pointers to float arrays
	}

	list_of_search_parameters = new float* [2];							// dynamic array (size 2) of pointers to float

	for (i = 0; i < 2; ++i)
	{
		list_of_search_parameters[i] = new float[number_of_search_positions];	// each i-th pointer is now pointing to dynamic array (size number_of_positions) of actual float values
	}
//	list_of_search_parameters  = new float[2][number_of_search_positions];

	while (number_of_positions < number_of_search_positions)
	{
		theta = 180.0 * global_random_number_generator.GetUniformRandom();
		phi = 360.0 * global_random_number_generator.GetUniformRandom();
		if (theta == 0.0)
		{
			list_of_search_parameters[0][number_of_positions] = phi;
			list_of_search_parameters[1][number_of_positions] = 0.0;
			number_of_positions++;
		}
		else
		if (global_random_number_generator.GetUniformRandom() > sinf(deg_2_rad(theta)) || trials > max_trials)
		{
			list_of_search_parameters[0][number_of_positions] = phi;
			list_of_search_parameters[1][number_of_positions] = theta;
			number_of_positions++;
		}
		trials++;
	}
}

// Run the search
void EulerSearch::Run(Particle &particle, Image &input_3d, Image *projections, Kernel2D **kernel_index)
{
	MyDebugAssertTrue(number_of_search_positions > 0,"EulerSearch not initialized");
	MyDebugAssertTrue(particle.particle_image->is_in_memory,"Particle image not allocated");
	MyDebugAssertTrue(input_3d.is_in_memory,"3D reference map not allocated");
	MyDebugAssertTrue(particle.particle_image->logical_x_dimension == input_3d.logical_x_dimension && particle.particle_image->logical_y_dimension == input_3d.logical_y_dimension, "Error: Image and 3D reference incompatible");

	int i;
	int pixel_counter;
	int psi_i;
	int psi_m;
	float psi;
//	float psi_max = 360.0;
//	float psi_step = 1.0;
	float best_inpane_score;
	float best_inplane_values[3];
	bool mirrored_match;
	Peak found_peak;
	AnglesAndShifts angles;
	Image *flipped_image = new Image;
	Image *projection_image = new Image;
	Image *rotated_image = new Image;
	Image *quad_rot_image = new Image;
	Image *correlation_map = new Image;
	Image *rotation_cache = NULL;
	flipped_image->Allocate(input_3d.logical_x_dimension, input_3d.logical_y_dimension, false);
	projection_image->Allocate(input_3d.logical_x_dimension, input_3d.logical_y_dimension, false);
	rotated_image->Allocate(input_3d.logical_x_dimension, input_3d.logical_y_dimension, false);
	quad_rot_image->Allocate(input_3d.logical_x_dimension, input_3d.logical_y_dimension, false);
	correlation_map->Allocate(input_3d.logical_x_dimension, input_3d.logical_y_dimension, false);
	correlation_map->object_is_centred_in_box = false;
	float *temp_k1 = new float [flipped_image->real_memory_allocated];
//	float *temp_k2 = new float [flipped_image->real_memory_allocated];
	float *temp_k2; temp_k2 = temp_k1 + 1;
//	float *temp_k3 = new float [flipped_image->real_memory_allocated];
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

//	temp_k1[0] = 10;
//	temp_k1[1] = 11;
//	temp_k1[2] = 12;
//	wxPrintf("k10, k11, k12, k20, k21 = %f, %f, %f, %f, %f\n",temp_k1[0], temp_k1[1], temp_k1[2], temp_k2[0], temp_k2[1]);
//	wxPrintf("0, 1, 2 = %g, %g, %g\n", particle.particle_image->real_values[0], particle.particle_image->real_values[1], particle.particle_image->real_values[2]);
//	wxPrintf("0, 1, 2 = %g, %g, %g\n", real_a[0], real_a[1], real_a[2]);
//	exit(0);

	psi_i = myroundint(psi_max / psi_step);
	if (test_mirror) psi_i *= 2;
	rotation_cache = new Image [psi_i];
	for (i = 0; i < psi_i; i++)
	{
		rotation_cache[i].Allocate(input_3d.logical_x_dimension, input_3d.logical_y_dimension, false);
	}

	best_score = - std::numeric_limits<float>::max();
	flipped_image->CopyFrom(particle.particle_image);
	flipped_image->PhaseFlipPixelWise(*particle.ctf_image);

	psi_i = 0;
	psi_m = 0;
	for (psi = 0.0; psi < psi_max; psi += psi_step)
	{
		flipped_image->RotateFourier2DFromIndex(rotation_cache[psi_m], kernel_index[psi_i]);
//		for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated / 2; pixel_counter++)
//		{
//			rotation_cache[psi_m].complex_values[pixel_counter] = conjf(rotation_cache[psi_m].complex_values[pixel_counter]);
//		}
		psi_i++;
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
			wxPrintf("i, phi, theta = %i, %f, %f\n", i, list_of_search_parameters[0][i], list_of_search_parameters[1][i]);
			angles.Init(list_of_search_parameters[0][i], list_of_search_parameters[1][i], 0.0, 0.0, 0.0);
			input_3d.ExtractSlice(*projection_image, angles, resolution_limit);
		}
		else
		{
			projection_image->CopyFrom(&projections[i]);
			for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k1[pixel_counter] = real_a[pixel_counter] + real_b[pixel_counter];};
			for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k2[pixel_counter] = real_b[pixel_counter] - real_a[pixel_counter];};
		}

		best_inpane_score = - std::numeric_limits<float>::max();
		psi_i = 0;
		for (psi = 0.0; psi < psi_max; psi += psi_step)
		{
			real_c = rotation_cache[psi_i].real_values;
			real_d = rotation_cache[psi_i].real_values + 1;

//			projection_image->RotateFourier2DFromIndex(*quad_rot_image, kernel_index[psi_i]);

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
			for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_d[pixel_counter] * temp_k1[pixel_counter];};
			for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_c[pixel_counter] * temp_k2[pixel_counter];};
	//		for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_d[pixel_counter] * (real_a[pixel_counter] + real_b[pixel_counter]);};
	//		for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_c[pixel_counter] * (real_b[pixel_counter] - real_a[pixel_counter]);};
	//			for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated / 2; pixel_counter++)
	//				{correlation_map->complex_values[pixel_counter] = projection_image->complex_values[pixel_counter] * rotation_cache[psi_i].complex_values[pixel_counter];}
				correlation_map->is_in_real_space = false;
				correlation_map->BackwardFFT();
				found_peak = correlation_map->FindPeakAtOriginFast2D(0.5 * correlation_map->physical_address_of_box_center_x);
//				found_peak = correlation_map->FindPeakWithIntegerCoordinates(0.0, 0.9);
				if (found_peak.value > best_inpane_score)
				{
					best_inpane_score =  found_peak.value;
					best_inplane_values[0] = 360.0 - psi;
//					best_inplane_values[0] = 360.0 - (psi + quad_i * 90);
					best_inplane_values[1] = found_peak.x;
					best_inplane_values[2] = found_peak.y;
					mirrored_match = false;
				}

//				if (test_mirror && fabs(list_of_search_parameters[1][i] - 90.0) > 1.0 && fabs(list_of_search_parameters[1][i] - 270.0) > 1.0)
				if (test_mirror)
				{
					psi_i++;

					real_c = rotation_cache[psi_i].real_values;
					real_d = rotation_cache[psi_i].real_values + 1;

//					rotation_cache[psi_i].MirrorYFourier2D(*correlation_map);
//					for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated / 2; pixel_counter++)
//					{
//						correlation_map->complex_values[pixel_counter] *= conjf(flipped_image->complex_values[pixel_counter]);
//					}
//					for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k1[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]);};
//					for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k2[pixel_counter] = real_d[pixel_counter] * (real_a[pixel_counter] + real_b[pixel_counter]);};
//					for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {temp_k3[pixel_counter] = real_c[pixel_counter] * (real_b[pixel_counter] - real_a[pixel_counter]);};
					for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_d[pixel_counter] * temp_k1[pixel_counter];};
					for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_c[pixel_counter] * temp_k2[pixel_counter];};
	//				for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_d[pixel_counter] * (real_a[pixel_counter] + real_b[pixel_counter]);};
	//				for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2) {real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_c[pixel_counter] * (real_b[pixel_counter] - real_a[pixel_counter]);};
	//				for (pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated / 2; pixel_counter++)
	//					{correlation_map->complex_values[pixel_counter] = projection_image->complex_values[pixel_counter] * rotation_cache[psi_i].complex_values[pixel_counter];}
					correlation_map->is_in_real_space = false;
					correlation_map->BackwardFFT();
					found_peak = correlation_map->FindPeakAtOriginFast2D(0.5 * correlation_map->physical_address_of_box_center_x);
//					found_peak = correlation_map->FindPeakWithIntegerCoordinates(0.0, 0.9);
					if (found_peak.value > best_inpane_score)
					{
						best_inpane_score =  found_peak.value;
						best_inplane_values[0] = 360.0 - psi;
//						best_inplane_values[0] = 360.0 - (psi + quad_i * 90);
						best_inplane_values[1] = found_peak.x;
						best_inplane_values[2] = found_peak.y;
						mirrored_match = true;
					}
				}
//			}
			psi_i++;
		}
		if (best_inpane_score > best_score)
		{
			best_score =  best_inpane_score;
			best_values[3] =   best_inplane_values[1] * cosf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * sinf(deg_2_rad(best_inplane_values[0]));
			best_values[4] = - best_inplane_values[1] * sinf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * cosf(deg_2_rad(best_inplane_values[0]));
			if (mirrored_match)
			{
				best_values[0] = list_of_search_parameters[0][i];
				best_values[1] = list_of_search_parameters[1][i] + 180.0;
				best_values[2] = best_inplane_values[0];
				wxPrintf("best_score_m = %f, best values = %f, %f, %f, %f, %f\n", best_score,
						best_values[2], best_values[1], best_values[0], best_values[3] * particle.pixel_size, best_values[4] * particle.pixel_size);
			}
			else
			{
				best_values[0] = list_of_search_parameters[0][i];
				best_values[1] = list_of_search_parameters[1][i];
				best_values[2] = best_inplane_values[0];
				wxPrintf("best_score   = %f, best values = %f, %f, %f, %f, %f\n", best_score,
						best_values[2], best_values[1], best_values[0], best_values[3] * particle.pixel_size, best_values[4] * particle.pixel_size);
			}
		}
	}

	delete flipped_image;
	delete projection_image;
	delete rotated_image;
	delete quad_rot_image;
	delete correlation_map;
	delete [] rotation_cache;
	delete [] temp_k1;
//	delete [] temp_k2;
//	delete [] temp_k3;
}
