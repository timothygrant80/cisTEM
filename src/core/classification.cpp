#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayofClassificationResults);
WX_DEFINE_OBJARRAY(ArrayofClassifications);
WX_DEFINE_OBJARRAY(ArrayofShortClassificationInfos);

ClassificationResult::ClassificationResult()
{
	position_in_stack = -1;
	psi = 0.0;
	xshift = 0.0;
	yshift = 0.0;
	best_class = 0;
	sigma = 10.0;
	logp = 0.0;
	defocus_1 = 0.0f;
	defocus_2 = 0.0f;
	defocus_angle = 0.0f;
	phase_shift = 0.0f;
	beam_tilt_x = 0.0f;
	beam_tilt_y = 0.0f;
	image_shift_x = 0.0f;
	image_shift_y = 0.0f;
	pixel_size = 0.0f;
	microscope_voltage_kv = 0.0f;
	microscope_spherical_aberration_mm = 0.0f;
	amplitude_contrast = 0.0f;
}

ClassificationResult::~ClassificationResult()
{

}

ShortClassificationInfo::ShortClassificationInfo()
{
	classification_id = -1;
	refinement_package_asset_id = -1;
	name = wxEmptyString;
	number_of_particles = 0;
	number_of_classes = 0;
	class_average_file = "";
	high_resolution_limit = 0.0;
}

ShortClassificationInfo & ShortClassificationInfo::operator = (const Classification &other_classification)
{
	*this = &other_classification;
	return *this;
}

ShortClassificationInfo & ShortClassificationInfo::operator = (const Classification *other_classification)
{
	classification_id = other_classification->classification_id;
	refinement_package_asset_id = other_classification->refinement_package_asset_id;
	name = other_classification->name;
	number_of_particles = other_classification->number_of_particles;
	number_of_classes = other_classification->number_of_classes;
	class_average_file = other_classification->class_average_file;
	high_resolution_limit = other_classification->high_resolution_limit;

	return *this;
}

Classification::Classification()
{

	classification_id = -1;
	refinement_package_asset_id = -1;
	name = wxEmptyString;
	class_average_file = wxEmptyString;
	classification_was_imported_or_generated = true;
	datetime_of_run = wxDateTime::Now();
	starting_classification_id = -1;
	number_of_particles = 0;
	number_of_classes = 0;
	low_resolution_limit = 0;
	high_resolution_limit = 0;
	mask_radius = 0;
	angular_search_step = 0;
	search_range_x = 0;
	search_range_y = 0;
	smoothing_factor = 0;
	exclude_blank_edges = true;
	auto_percent_used = true;
	percent_used = 100;

	ArrayofClassificationResults classification_results;


}

Classification::~Classification()
{

}

void Classification::SizeAndFillWithEmpty(long wanted_number_of_particles)
{
	ClassificationResult junk_result;

	number_of_particles = wanted_number_of_particles;
	classification_results.Alloc(number_of_particles);
	classification_results.Add(junk_result, number_of_particles);
}

wxString Classification::WritecisTEMStarFile(wxString base_filename, RefinementPackage *parent_refinement_package)
{
	wxString output_filename;
	output_filename = base_filename + wxString::Format("_%li.star", classification_id);

	long particle_counter;



	cisTEMParameters output_params;
	output_params.parameters_to_write.SetActiveParameters(POSITION_IN_STACK | BEST_2D_CLASS | PSI | X_SHIFT | Y_SHIFT | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | PHASE_SHIFT | LOGP | SIGMA | PIXEL_SIZE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y);
	output_params.PreallocateMemoryAndBlank(number_of_particles);

	for ( particle_counter = 0; particle_counter < number_of_particles; particle_counter++)
	{
		output_params.all_parameters[particle_counter].position_in_stack = classification_results[particle_counter].position_in_stack;
		output_params.all_parameters[particle_counter].best_2d_class = classification_results[particle_counter].best_class;
		output_params.all_parameters[particle_counter].psi = classification_results[particle_counter].psi;
		output_params.all_parameters[particle_counter].x_shift = classification_results[particle_counter].xshift;
		output_params.all_parameters[particle_counter].y_shift = classification_results[particle_counter].yshift;
		output_params.all_parameters[particle_counter].defocus_1 = parent_refinement_package->ReturnParticleInfoByPositionInStack( classification_results[particle_counter].position_in_stack).defocus_1;
		output_params.all_parameters[particle_counter].defocus_2 = parent_refinement_package->ReturnParticleInfoByPositionInStack( classification_results[particle_counter].position_in_stack).defocus_2;
		output_params.all_parameters[particle_counter].defocus_angle = parent_refinement_package->ReturnParticleInfoByPositionInStack( classification_results[particle_counter].position_in_stack).defocus_angle;
		output_params.all_parameters[particle_counter].phase_shift = parent_refinement_package->ReturnParticleInfoByPositionInStack( classification_results[particle_counter].position_in_stack).phase_shift;
		output_params.all_parameters[particle_counter].logp = classification_results[particle_counter].logp;
		output_params.all_parameters[particle_counter].sigma = classification_results[particle_counter].sigma;
		output_params.all_parameters[particle_counter].pixel_size = classification_results[particle_counter].pixel_size;
		output_params.all_parameters[particle_counter].microscope_voltage_kv = classification_results[particle_counter].microscope_voltage_kv;
		output_params.all_parameters[particle_counter].microscope_spherical_aberration_mm = classification_results[particle_counter].microscope_spherical_aberration_mm;
		output_params.all_parameters[particle_counter].amplitude_contrast = classification_results[particle_counter].amplitude_contrast;
		output_params.all_parameters[particle_counter].beam_tilt_x = classification_results[particle_counter].beam_tilt_x;
		output_params.all_parameters[particle_counter].beam_tilt_y = classification_results[particle_counter].beam_tilt_y;
		output_params.all_parameters[particle_counter].image_shift_x = classification_results[particle_counter].image_shift_x;
		output_params.all_parameters[particle_counter].image_shift_y = classification_results[particle_counter].image_shift_y;
	}

	output_params.WriteTocisTEMStarFile(output_filename);

	return output_filename;
}

wxString Classification::WriteFrealignParameterFiles(wxString base_filename, RefinementPackage *parent_refinement_package)
{
	wxString output_filename;

	float output_parameters[17];
	float parameter_average[17];

	long particle_counter;
	int parameter_counter;

	ZeroFloatArray(output_parameters, 17);
	ZeroFloatArray(parameter_average, 17);

	output_filename = base_filename + wxString::Format("_%li.par", classification_id);
	FrealignParameterFile *my_output_par_file = new FrealignParameterFile(output_filename, OPEN_TO_WRITE);

	my_output_par_file->WriteCommentLine("C           PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST  PSHIFT     OCC      LogP      SIGMA   SCORE  CHANGE");

	for ( particle_counter = 0; particle_counter < number_of_particles; particle_counter++)
	{

		output_parameters[0] = classification_results[particle_counter].position_in_stack;
		output_parameters[1] = classification_results[particle_counter].psi;
		output_parameters[2] = 0.0;
		output_parameters[3] = 0.0;
		output_parameters[4] = classification_results[particle_counter].xshift;
		output_parameters[5] = classification_results[particle_counter].yshift;
		output_parameters[6] = 0.0;
		output_parameters[7] = classification_results[particle_counter].best_class;
		output_parameters[8] = parent_refinement_package->ReturnParticleInfoByPositionInStack( classification_results[particle_counter].position_in_stack).defocus_1;
		output_parameters[9] = parent_refinement_package->ReturnParticleInfoByPositionInStack( classification_results[particle_counter].position_in_stack).defocus_2;
		output_parameters[10] = parent_refinement_package->ReturnParticleInfoByPositionInStack( classification_results[particle_counter].position_in_stack).defocus_angle;
		output_parameters[11] = parent_refinement_package->ReturnParticleInfoByPositionInStack( classification_results[particle_counter].position_in_stack).phase_shift;
		output_parameters[12] = 100.0;
		output_parameters[13] = classification_results[particle_counter].logp;
		output_parameters[14] = classification_results[particle_counter].sigma;
		output_parameters[15] = 0.0;
		output_parameters[16] = 0.0;

		for (parameter_counter = 0; parameter_counter < 17; parameter_counter++)
		{
			parameter_average[parameter_counter] += output_parameters[parameter_counter];
		}


		my_output_par_file->WriteLine(output_parameters);

	}

	delete my_output_par_file;
	return output_filename;
}

float Classification::ReturnXShiftByPositionInStack(long wanted_position_in_stack)
{
	long particle_counter;

	long start_pos = wanted_position_in_stack - 1;
	if (start_pos >= number_of_particles) start_pos = number_of_particles - 1;

	for ( particle_counter = start_pos; particle_counter < number_of_particles; particle_counter++)
	{
		if (classification_results[particle_counter].position_in_stack == wanted_position_in_stack) return classification_results[particle_counter].xshift;
	}

	// if we got here, haven't found it yet, try again..

	for ( particle_counter = 0; particle_counter < start_pos; particle_counter++)
	{
		if (classification_results[particle_counter].position_in_stack == wanted_position_in_stack) return classification_results[particle_counter].xshift;
	}

	// if we got here, it isn't there..

	return 0.0;
}

float Classification::ReturnYShiftByPositionInStack(long wanted_position_in_stack)
{
	long particle_counter;

	long start_pos = wanted_position_in_stack - 1;
	if (start_pos >= number_of_particles) start_pos = number_of_particles - 1;

	for ( particle_counter = start_pos; particle_counter < number_of_particles; particle_counter++)
	{
		if (classification_results[particle_counter].position_in_stack == wanted_position_in_stack) return classification_results[particle_counter].yshift;
	}

	// if we got here, haven't found it yet, try again..

	for ( particle_counter = 0; particle_counter < start_pos; particle_counter++)
	{
		if (classification_results[particle_counter].position_in_stack == wanted_position_in_stack) return classification_results[particle_counter].yshift;
	}

	// if we got here, it isn't there..

	return 0.0;
}


