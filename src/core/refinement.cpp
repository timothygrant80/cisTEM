#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayofRefinementResults);
WX_DEFINE_OBJARRAY(ArrayofClassRefinementResults);

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
	occupancy = 0;
	logp = 0;
	sigma = 0;
	score = 0;
	score_change = 0;

}

RefinementResult::~RefinementResult()
{

}

Refinement::Refinement()
{
	refinement_id = -1;
	refinement_package_asset_id = -1;
	name = "";
	refinement_was_imported_or_generated = true;
	datetime_of_run = wxDateTime::Now();
	starting_refinement_id = -1;
	number_of_particles = 0;
	number_of_classes = 0;
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


}

Refinement::~Refinement()
{

}

wxArrayString Refinement::WriteFrealignParameterFiles(wxString base_filename)
{
	wxArrayString output_filenames;
	wxString current_filename;
	float output_parameters[16];
	float parameter_average[16];

	int class_counter;
	long particle_counter;
	int parameter_counter;


	for ( class_counter = 0; class_counter < number_of_classes; class_counter++)
	{
		ZeroFloatArray(output_parameters, 16);
		ZeroFloatArray(parameter_average, 16);

		current_filename = base_filename + wxString::Format("_%li_%i.par", refinement_id, class_counter + 1);
		output_filenames.Add(current_filename);

		FrealignParameterFile *my_output_par_file = new FrealignParameterFile(current_filename, OPEN_TO_WRITE);

		for ( particle_counter = 0; particle_counter < number_of_particles; particle_counter++)
		{

			output_parameters[0] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].position_in_stack;
			output_parameters[1] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].psi;
			output_parameters[2] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].theta;
			output_parameters[3] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].phi;
			output_parameters[4] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].xshift;
			output_parameters[5] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].yshift;
			output_parameters[6] = 0.0;
			output_parameters[7] = 0.0;
			output_parameters[8] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].defocus1;
			output_parameters[9] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].defocus2;
			output_parameters[10] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].defocus_angle;
			output_parameters[11] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy;
			output_parameters[12] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp;
			output_parameters[13] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].sigma;
			output_parameters[14] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].score;
			output_parameters[15] = class_refinement_results[class_counter].particle_refinement_results[particle_counter].score_change;

			for (parameter_counter = 0; parameter_counter < 16; parameter_counter++)
			{
				parameter_average[parameter_counter] += output_parameters[parameter_counter];
			}

			my_output_par_file->WriteLine(output_parameters);

		}

		for (parameter_counter = 0; parameter_counter < 16; parameter_counter++)
		{
			parameter_average[parameter_counter] /= float (number_of_particles);
		}

		my_output_par_file->WriteLine(parameter_average, true);
		my_output_par_file->WriteCommentLine("C  Total particles included, overall score, average occupancy " + wxString::Format("%11li %10.6f %10.6f", number_of_particles, parameter_average[14], parameter_average[11]));

		delete my_output_par_file;
	}

	return output_filenames;
}

void Refinement::WriteResolutionStatistics(wxString base_filename)
{
	NumericTextFile *current_plot;
	int class_counter;

	for ( class_counter = 0; class_counter < number_of_classes; class_counter++)
	{
		current_plot = new NumericTextFile(base_filename + wxString::Format("_%li_%i.txt", refinement_id, class_counter + 1), OPEN_TO_WRITE, 7);
		class_refinement_results[class_counter].class_resolution_statistics.WriteStatisticsToFile(*current_plot);
		delete current_plot;
	}
}

void Refinement::SizeAndFillWithEmpty(long wanted_number_of_particles, int wanted_number_of_classes)
{
	ClassRefinementResults junk_class_results;
	RefinementResult junk_result;

	//wxPrintf("Allocating for %i classes and %li particles\n", wanted_number_of_classes, wanted_number_of_particles);
	number_of_classes = wanted_number_of_classes;
	number_of_particles = wanted_number_of_particles;

	class_refinement_results.Alloc(number_of_classes);
	class_refinement_results.Add(junk_class_results, number_of_classes);

	for (int class_counter = 0; class_counter < number_of_classes; class_counter++)
	{
		class_refinement_results[class_counter].particle_refinement_results.Alloc(number_of_particles);
		class_refinement_results[class_counter].particle_refinement_results.Add(junk_result, number_of_particles);
	}


}
