#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayofRefinementResults);
WX_DEFINE_OBJARRAY(ArrayofClassRefinementResults);
WX_DEFINE_OBJARRAY(ArrayofRefinements);
WX_DEFINE_OBJARRAY(ArrayofShortRefinementInfos);

RefinementResult::RefinementResult()
{
	position_in_stack = -1;
	psi = 0;
	theta = 0;
	phi = 0;
	xshift = 0;
	yshift = 0;
	defocus1 = 0;
	defocus2 = 0;
	defocus_angle = 0;
	phase_shift = 0;
	occupancy = 0;
	logp = 0;
	sigma = 0;
	score = 0;
	image_is_active = 1;

}

RefinementResult::~RefinementResult()
{

}

ClassRefinementResults::ClassRefinementResults()
{
	low_resolution_limit = 0;
	high_resolution_limit = 0;
	mask_radius = 0;
	signed_cc_resolution_limit = 0;
	global_resolution_limit = 0;
	global_mask_radius = 0;
	number_results_to_refine = 0;
	angular_search_step = 0;
	search_range_x = 0;
	search_range_y = 0;
	classification_resolution_limit = 0;
	should_focus_classify = 0;
	sphere_x_coord = 0;
	sphere_y_coord = 0;
	sphere_z_coord = 0;
	sphere_radius = 0;
	should_refine_ctf = false;
	defocus_search_range = 0;
	defocus_search_step = 0;
	average_occupancy = 0.0f;
	estimated_resolution = 0.0f;
	reconstructed_volume_asset_id = -1;
	reconstruction_id = -1;
	should_auto_mask = false;
	should_refine_input_params = true;
	should_use_supplied_mask = false;
	mask_asset_id = -1;
	mask_edge_width = 10;
	outside_mask_weight = 0.0;
	should_low_pass_filter_mask = false;
	filter_resolution = 30.0;
}

ClassRefinementResults::~ClassRefinementResults()
{

}


ShortRefinementInfo::ShortRefinementInfo()
{

	refinement_id = -1;
	refinement_package_asset_id = -1;
	name = wxEmptyString;
	number_of_particles = 0;
	number_of_classes = 0;
}

ShortRefinementInfo & ShortRefinementInfo::operator = (const Refinement &other_refinement)
{
	*this = &other_refinement;
	return *this;
}

ShortRefinementInfo & ShortRefinementInfo::operator = (const Refinement *other_refinement)
{
	refinement_id = other_refinement->refinement_id;
	refinement_package_asset_id = other_refinement->refinement_package_asset_id;
	name = other_refinement->name;
	number_of_particles = other_refinement->number_of_particles;
	number_of_classes = other_refinement->number_of_classes;

	average_occupancy.Clear();
	reconstructed_volume_asset_ids.Clear();
	estimated_resolution.Clear();

	for (int counter = 0; counter < number_of_classes; counter++)
	{
		average_occupancy.Add(other_refinement->class_refinement_results[counter].average_occupancy);
		reconstructed_volume_asset_ids.Add(other_refinement->class_refinement_results[counter].reconstructed_volume_asset_id);
		if (other_refinement->resolution_statistics_are_generated == true) estimated_resolution.Add(0.0f);
		else estimated_resolution.Add(other_refinement->class_refinement_results[counter].class_resolution_statistics.ReturnEstimatedResolution());
	}

	return *this;
}

Refinement::Refinement()
{
	refinement_id = -1;
	refinement_package_asset_id = -1;
	name = "";
	resolution_statistics_are_generated = true;
	datetime_of_run = wxDateTime::Now();
	starting_refinement_id = -1;
	number_of_particles = 0;
	number_of_classes = 0;
	percent_used = 0.0;
	resolution_statistics_pixel_size = 0;
	resolution_statistics_box_size = 0;
	percent_used = 100.0f;
}

Refinement::~Refinement()
{

}


long Refinement::ReturnNumberOfActiveParticlesInFirstClass()
{
	long particle_counter;
	long number_active = 0;

	for (particle_counter = 0; particle_counter < number_of_particles; particle_counter ++ )
	{
		if (class_refinement_results[0].particle_refinement_results[particle_counter].image_is_active >= 0) number_active++;
	}

	return number_active;
}

void Refinement::FillAngularDistributionHistogram(wxString wanted_symmetry, int wanted_class, int number_of_theta_bins, int number_of_phi_bins, AngularDistributionHistogram &histogram_to_fill)
{
	long particle_counter;
	int symmetry_counter;
	// for symmetry;

	SymmetryMatrix symmetry_matrices;
	AnglesAndShifts angles_and_shifts;
	RotationMatrix temp_matrix;

	float proj_x;
	float proj_y;
	float proj_z;
	float north_pole_x = 0.0f;
	float north_pole_y = 0.0f;
	float north_pole_z = 1.0f;

	float sym_theta;
	float sym_phi;

	symmetry_matrices.Init(wanted_symmetry);
	histogram_to_fill.Init(number_of_theta_bins, number_of_phi_bins);

	for (particle_counter = 0; particle_counter < number_of_particles; particle_counter ++ )
	{
		if (class_refinement_results[wanted_class].particle_refinement_results[particle_counter].image_is_active >= 0)
		{
			if (ReturnClassWithHighestOccupanyForGivenParticle(particle_counter) == wanted_class)
			{
				// Setup a angles and shifts
				angles_and_shifts.Init(class_refinement_results[wanted_class].particle_refinement_results[particle_counter].phi, class_refinement_results[wanted_class].particle_refinement_results[particle_counter].theta, class_refinement_results[wanted_class].particle_refinement_results[particle_counter].psi,0.0,0.0);

				// Loop over symmetry-related views
				for (symmetry_counter = 0; symmetry_counter < symmetry_matrices.number_of_matrices; symmetry_counter ++ )
				{

					// Get the rotation matrix for the current orientation and current symmetry-related view
					temp_matrix = symmetry_matrices.rot_mat[symmetry_counter] * angles_and_shifts.euler_matrix;

					// Rotate a vector which initially points at the north pole
					temp_matrix.RotateCoords(north_pole_x, north_pole_y, north_pole_z, proj_x, proj_y, proj_z);

					// If we are in the southern hemisphere, we will need to plot the equivalent projection in the northen hemisphere
					if (proj_z < 0.0)
					{
						proj_z = - proj_z;
						proj_y = - proj_y;
						proj_x = - proj_x;
					}

					// Add to histogram

					sym_theta = ConvertProjectionXYToThetaInDegrees(proj_x, proj_y);
					sym_phi = ConvertXYToPhiInDegrees(proj_x, proj_y);

					histogram_to_fill.AddPosition(sym_theta, sym_phi);
				}
			}
		}
	}
}

ArrayofAngularDistributionHistograms Refinement::ReturnAngularDistributions(wxString desired_symmetry)
{
	ArrayofAngularDistributionHistograms all_histograms;
	AngularDistributionHistogram blank;

	all_histograms.Add(blank, number_of_classes);

	for (int class_counter = 0; class_counter < number_of_classes; class_counter++)
	{
		FillAngularDistributionHistogram(desired_symmetry, class_counter, 18, 72, all_histograms[class_counter]);
	}

	return all_histograms;
}

RefinementResult Refinement::ReturnRefinementResultByClassAndPositionInStack(int wanted_class, long wanted_position_in_stack)
{
	MyDebugAssertTrue(wanted_class >= 0 && wanted_class < number_of_classes, "wanted class (%i) out of range!", wanted_class);
	MyDebugAssertTrue(wanted_position_in_stack > 0 && wanted_class <= number_of_particles, "wanted position in stack out of range!");

	for (long counter = wanted_position_in_stack - 1; counter < number_of_particles; counter++)
	{
		if (class_refinement_results[wanted_class].particle_refinement_results[counter].position_in_stack == wanted_position_in_stack) return class_refinement_results[wanted_class].particle_refinement_results[counter];
	}

	for (long counter = 0; counter < wanted_position_in_stack; counter++)
	{
		if (class_refinement_results[wanted_class].particle_refinement_results[counter].position_in_stack == wanted_position_in_stack) return class_refinement_results[wanted_class].particle_refinement_results[counter];
	}

	MyDebugPrintWithDetails("Shouldn't get here, means i didn't find the particle - Class #%i, Pos = %li", wanted_class, wanted_position_in_stack);
	abort();
}

int Refinement::ReturnClassWithHighestOccupanyForGivenParticle(long wanted_particle) // starts at 0
{
	MyDebugAssertTrue(wanted_particle >= 0 && wanted_particle < number_of_particles, "wanted_particle (%li) is out of range", wanted_particle);
	float best_occupancy = -FLT_MAX;
	int best_class = 0;

	for (int class_counter = 0; class_counter < number_of_classes; class_counter++)
	{
		if (class_refinement_results[class_counter].particle_refinement_results[wanted_particle].occupancy > best_occupancy)
		{
			best_occupancy = class_refinement_results[class_counter].particle_refinement_results[wanted_particle].occupancy;
			best_class = class_counter;
		}
	}

	return best_class;
}

void Refinement::WriteSingleClassFrealignParameterFile(wxString filename,int wanted_class, float percent_used_overide, float sigma_override)
{
	float output_parameters[17];
	float parameter_average[17];
	float temp_float;
	long particle_counter;
	int parameter_counter;

	ZeroFloatArray(output_parameters, 17);
	ZeroFloatArray(parameter_average, 17);

	FrealignParameterFile *my_output_par_file = new FrealignParameterFile(filename, OPEN_TO_WRITE);

	my_output_par_file->WriteCommentLine("C           PSI   THETA     PHI       SHX       SHY     MAG  INCLUDE   DF1      DF2  ANGAST  PSHIFT     OCC      LogP      SIGMA   SCORE  CHANGE");

	for ( particle_counter = 0; particle_counter < number_of_particles; particle_counter++)
	{
		output_parameters[0] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].position_in_stack;
		output_parameters[1] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].psi;
		output_parameters[2] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].theta;
		output_parameters[3] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].phi;
		output_parameters[4] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].xshift;
		output_parameters[5] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].yshift;
		output_parameters[6] = 0.0;

		if (percent_used_overide < 1.0)
		{
			temp_float = global_random_number_generator.GetUniformRandom();
			if (temp_float < 1.0 - 2.0 * percent_used_overide)
			{
				output_parameters[7] = -1;
			}
			else
			{
				output_parameters[7] = 1;
			}
		}
		else
		{
			output_parameters[7] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].image_is_active;
		}

		output_parameters[8] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].defocus1;
		output_parameters[9] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].defocus2;
		output_parameters[10] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].defocus_angle;
		output_parameters[11] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].phase_shift;
		output_parameters[12] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].occupancy;
		output_parameters[13] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].logp;
		if (sigma_override > 0.0) output_parameters[14] = sigma_override;
		else output_parameters[14] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].sigma;
		output_parameters[15] = class_refinement_results[wanted_class].particle_refinement_results[particle_counter].score;
		output_parameters[16] = 0.0;

		for (parameter_counter = 0; parameter_counter < 17; parameter_counter++)
		{
			parameter_average[parameter_counter] += output_parameters[parameter_counter];
		}

		my_output_par_file->WriteLine(output_parameters);

	}

	for (parameter_counter = 0; parameter_counter < 17; parameter_counter++)
	{
		parameter_average[parameter_counter] /= float (number_of_particles);
	}

	my_output_par_file->WriteLine(parameter_average, true);
	my_output_par_file->WriteCommentLine("C  Total particles included, overall score, average occupancy " + wxString::Format("%11li %10.6f %10.6f", number_of_particles, parameter_average[15], parameter_average[12]));

	delete my_output_par_file;
}

wxArrayString Refinement::WriteFrealignParameterFiles(wxString base_filename, float percent_used_overide, float sigma_override)
{
	MyDebugAssertTrue(number_of_classes > 0, "Number of classes is not greater than 0!")
	wxArrayString output_filenames;
	wxString current_filename;


	int class_counter;


	for ( class_counter = 0; class_counter < number_of_classes; class_counter++)
	{
		current_filename = base_filename + wxString::Format("_%li_%i.par", refinement_id, class_counter + 1);
		output_filenames.Add(current_filename);
		WriteSingleClassFrealignParameterFile(current_filename, class_counter, percent_used_overide, sigma_override);
	}

	return output_filenames;
}

wxArrayString Refinement::WriteResolutionStatistics(wxString base_filename, float pssnr_division_factor)
{
	NumericTextFile *current_plot;
	int class_counter;

	wxString current_filename;
	wxArrayString output_filenames;


	for ( class_counter = 0; class_counter < number_of_classes; class_counter++)
	{
		current_filename = base_filename + wxString::Format("_%li_%i.txt", refinement_id, class_counter + 1);
		output_filenames.Add(current_filename);

		current_plot = new NumericTextFile(current_filename, OPEN_TO_WRITE, 7);
		class_refinement_results[class_counter].class_resolution_statistics.WriteStatisticsToFile(*current_plot, pssnr_division_factor);
		delete current_plot;
	}

	return output_filenames;
}

void Refinement::SizeAndFillWithEmpty(long wanted_number_of_particles, int wanted_number_of_classes)
{
	ClassRefinementResults junk_class_results;
	RefinementResult junk_result;

	//wxPrintf("Allocating for %i classes and %li particles\n", wanted_number_of_classes, wanted_number_of_particles);
	number_of_classes = wanted_number_of_classes;
	number_of_particles = wanted_number_of_particles;

	reference_volume_ids.Clear();
	reference_volume_ids.Add(-1, number_of_classes);

	class_refinement_results.Alloc(number_of_classes);
	class_refinement_results.Add(junk_class_results, number_of_classes);

	for (int class_counter = 0; class_counter < number_of_classes; class_counter++)
	{
		class_refinement_results[class_counter].particle_refinement_results.Alloc(number_of_particles);
		class_refinement_results[class_counter].particle_refinement_results.Add(junk_result, number_of_particles);
	}
}

float Refinement::ReturnChangeInAverageOccupancy(Refinement &other_refinement)
{
	MyDebugAssertTrue(number_of_classes == other_refinement.number_of_classes, "Number of classes is not the same")

	float change_in_average_occupancy = 0.0f;

	for (int class_counter = 0; class_counter < number_of_classes; class_counter++)
	{
		change_in_average_occupancy += fabsf(class_refinement_results[class_counter].average_occupancy - other_refinement.class_refinement_results[class_counter].average_occupancy);
	}

	return change_in_average_occupancy;// / float(number_of_classes);

}

wxArrayFloat Refinement::UpdatePSSNR()
{
	wxArrayFloat average_occupancies;


	if (this->number_of_classes > 1)
	{
		average_occupancies.Add(0.0, this->number_of_classes);

		long number_of_active_images = 0;
		float sum_ave_occ = 0.0f;
		int class_counter;
		long particle_counter;
		int point_counter;
		float sum_part_ssnr;
		float current_part_ssnr;


		for (class_counter = 0; class_counter < this->number_of_classes; class_counter++)
		{
			number_of_active_images = 0;
			average_occupancies[class_counter] = 0.0;

			for (particle_counter = 0; particle_counter < this->number_of_particles; particle_counter++)
			{
				if (this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_is_active >= 0)
				{
					average_occupancies[class_counter] += this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy;
					number_of_active_images++;
				}
			}

			average_occupancies[class_counter] /= float(number_of_active_images);
			sum_ave_occ += average_occupancies[class_counter];
		}

		for (point_counter = 0; point_counter < this->class_refinement_results[0].class_resolution_statistics.part_SSNR.number_of_points; point_counter++)
		{
			sum_part_ssnr = 0;
			for (class_counter = 0; class_counter < this->number_of_classes; class_counter++)
			{
				sum_part_ssnr += this->class_refinement_results[class_counter].class_resolution_statistics.part_SSNR.data_y[point_counter] * average_occupancies[class_counter];
			}

			current_part_ssnr = sum_part_ssnr / sum_ave_occ;

			for (class_counter = 0; class_counter < this->number_of_classes; class_counter++)
			{
				this->class_refinement_results[class_counter].class_resolution_statistics.part_SSNR.data_y[point_counter] = current_part_ssnr;
			}

		}
	}
	else average_occupancies.Add(100.00);

	return average_occupancies;
}

void Refinement::UpdateOccupancies(bool use_old_occupancies)
{
	if (this->number_of_classes > 1)
	{
		int class_counter;
		int particle_counter;
		int point_counter;

		float sum_probabilities;
		float occupancy;
		float max_logp;
		float average_occupancies[this->number_of_classes];
		long number_of_active_images = 0;
		float current_average_sigma;

		// calculate old average occupancies

		if (use_old_occupancies == true)
		{
			for (class_counter = 0; class_counter < this->number_of_classes; class_counter++)
			{
				average_occupancies[class_counter] = 0.0;
				number_of_active_images = 0;

				for (particle_counter = 0; particle_counter < this->number_of_particles; particle_counter++)
				{
					average_occupancies[class_counter] += this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy;
					number_of_active_images++;

				}

				average_occupancies[class_counter] /= float(number_of_active_images);
			}
		}
		else
		{
			for (class_counter = 0; class_counter < this->number_of_classes; class_counter++)
			{
				average_occupancies[class_counter] = 100.0 / float(this->number_of_classes);
			}
		}

		for (particle_counter = 0; particle_counter < this->number_of_particles; particle_counter++)
		{
			max_logp = -FLT_MAX;

			for (class_counter = 0; class_counter < this->number_of_classes; class_counter++)
			{
				if (this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp > max_logp) max_logp = this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp;
			}

			sum_probabilities = 0.0;

			for (class_counter = 0; class_counter < this->number_of_classes; class_counter++)
			{
				if (max_logp - this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp < 10.0)
				{
					sum_probabilities += exp(this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp  - max_logp) * average_occupancies[class_counter];
				}
			}

			current_average_sigma = 0.0;

			for (class_counter = 0; class_counter < this->number_of_classes; class_counter++)
			{
				if (max_logp -  this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp < 10.0)
				{
					occupancy = exp(this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp - max_logp) * average_occupancies[class_counter] / sum_probabilities *100.0;
				}
				else
				{
					occupancy = 0.0;
				}

				//occupancy = 1. * (occupancy - this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy) + this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy;
				this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy = occupancy;
				current_average_sigma +=  this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].sigma * (this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy / 100.0);
			}

			for (class_counter = 0; class_counter < this->number_of_classes; class_counter++)
			{
				this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].sigma = current_average_sigma;
			}
		}

		// calculate the new average occupancies..

		for (class_counter = 0; class_counter < this->number_of_classes; class_counter++)
		{
			number_of_active_images = 0;
			this->class_refinement_results[class_counter].average_occupancy = 0.0f;

			for (particle_counter = 0; particle_counter < this->number_of_particles; particle_counter++)
			{
				if (this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_is_active >= 0)
				{
					this->class_refinement_results[class_counter].average_occupancy += this->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy;
					number_of_active_images++;
				}
			}

			if (number_of_active_images > 0) this->class_refinement_results[class_counter].average_occupancy /= float(number_of_active_images);
		}

	}
}


