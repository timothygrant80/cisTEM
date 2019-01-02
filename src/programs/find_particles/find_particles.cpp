#include "../../core/core_headers.h"

class
FindParticlesApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();


};

IMPLEMENT_APP(FindParticlesApp)

// override the DoInteractiveUserInput

void FindParticlesApp::DoInteractiveUserInput()
{

	UserInput *my_input = new UserInput("FindParticles", 0.0);

	wxString	micrograph_filename					=	my_input->GetFilenameFromUser("Input micrograph filename","The input micrograph, in which we will look for particles","micrograph.mrc",true);
	float		pixel_size							=	my_input->GetFloatFromUser("Micrograph pixel size","In Angstroms","1.0",0.0);
	float		acceleration_voltage_in_keV			=	my_input->GetFloatFromUser("Acceleration voltage","In keV","300.0",0.0);
	float		spherical_aberration_in_mm			=	my_input->GetFloatFromUser("Spherical aberration","In mm","2.7",0.0);
	float		amplitude_contrast					=	my_input->GetFloatFromUser("Amplitude contrast","As a fraction, e.g. 0.07 is 7%","0.07",0.0);
	float 		additional_phase_shift_in_radians	=	my_input->GetFloatFromUser("Additional phase shift", "In radians. For example due to a phase plate","0.0");
	float		defocus_1_in_angstroms				=	my_input->GetFloatFromUser("Micrograph defocus 1","In Angstroms. For underfocus, give a positive value.","15000.0");
	float		defocus_2_in_angstroms				=	my_input->GetFloatFromUser("Micrograph defocus 2","In Angstroms. For underfocus, give a positive value.","15000.0");
	float		astigmatism_angle_in_degrees		=	my_input->GetFloatFromUser("Micrograph astigmatism angle","In degrees, following CTFFIND convention","0.0");
	bool		already_have_templates				=	my_input->GetYesNoFromUser("Would you like to supply templates?","Say yes here if you already have a template or templates to use as references for picking, for example projections from an existing 3D reconstruction","no");
	wxString	templates_filename					=	"templates.mrc";
	bool 		rotate_templates					=	false;
	int			number_of_template_rotations		=	1;
	bool 		average_templates_radially			=	false;
	float 		typical_radius						=	25.0;
	if (already_have_templates)
	{
				templates_filename					=	my_input->GetFilenameFromUser("Input templates filename","Set of templates to use in the search. Must have same pixel size as the micrograph","templates.mrc",true);
				average_templates_radially			=	my_input->GetYesNoFromUser("Should the templates be radially averaged?","Say yes if the templates should be rotationally averaged","no");
				if (! average_templates_radially)
				{
					rotate_templates					=	my_input->GetYesNoFromUser("Would you like to also search for rotated versions of the templates?","If you answer yes, each template image will be rotated a number of times (see next question) and the micrograph will be searched for the rotated template","yes");
					if (rotate_templates)
					{
						number_of_template_rotations	=	my_input->GetIntFromUser("Number of in-plane rotations","Give 1 to only correlate against the templates as given. Give 2 to also correlate against a 180-degree-rotated version. Etc.","36",1);
					}
					else
					{
						number_of_template_rotations	=	1;
					}
				}
	}
	else
	{
		typical_radius								=	my_input->GetFloatFromUser("Typical radius of particles (in Angstroms)","An estimate of the typical or average radius of the particles to be found. This will be used to generate a featureless disc as a template.","25.0",0.0);
	}
	float		maximum_radius						=	my_input->GetFloatFromUser("Maximum radius of the particle (in Angstroms)","The maximum radius of the templates, in angstroms","32.0",0.0);
	float		highest_resolution_to_use			=	my_input->GetFloatFromUser("Highest resolution to use for picking","In Angstroms. Data at higher resolutions will be ignored in the picking process","15.0",pixel_size * 2.0);
	wxString	output_stack_filename				=	my_input->GetFilenameFromUser("Filename for output stack of candidate particles.","A stack of candidate particles will be written to disk","candidate_particles.mrc",false);
	int			output_stack_box_size				=	my_input->GetIntFromUser("Box size for output candidate particle images (pixels)","In pixels. Give 0 to skip writing particle images to disk.","256",0);
	int			minimum_distance_from_edges			=	my_input->GetIntFromUser("Minimum distance from edge (pixels)","In pixels, the minimum distance between the center of a box and the edge of the micrograph","129",0);
	float		picking_threshold					=	my_input->GetFloatFromUser("Picking threshold","The minimum peak height for candidate particles. In numbers of background noise standard deviations. Typically in the 3.0 to 15.0 range. For micrographs with good contrast, give higher values to avoid spurious peaks.","8.0",0.0);
	bool		avoid_low_variance_areas			=	my_input->GetYesNoFromUser("Avoid low variance areas?","Areas with low local variance should be avoided. This often works well to avoid false positive picks in empty areas.","yes");
	bool		avoid_high_variance_areas			=	my_input->GetYesNoFromUser("Avoid high variance areas?","Areas with abnormally high local variance should be avoided. This often works well to avoid the edges of support film, ice crystals etc.","yes");
	float		low_variance_threshold_in_fwhm		=	my_input->GetFloatFromUser("Low variance threshold","When the local variance is below this threshold, no particles can be found. Expressed in numbers of FWHM above the mode of the local variance in the image. A negative number indicates a local variance below the mode","-1.0");
	float		high_variance_threshold_in_fwhm		=	my_input->GetFloatFromUser("High variance threshold","When the local variance is above this threshold, no particles can be found. Expressed in numbers of FWHM above the mode of the local variance in the image.","2.0");
	bool 		avoid_high_low_mean_areas			=	my_input->GetYesNoFromUser("Avoid areas with abnormal local mean?","Areas with abnormal local mean are can be avoided. This often works well to avoid ice crystals, for example.","yes");
	int			algorithm_to_find_background		=	my_input->GetIntFromUser("Algorithm to find background areas (0 or 1)","0: lowest variance; 1: variance near mode","0",0,1);
	int			number_of_background_boxes			=	my_input->GetIntFromUser("Number of background boxes","This number of boxes will be extracted from the micrographs in areas devoid of particles or other features, to compute the background amplitude spectrum","50",1);
	bool		particles_are_white					=	my_input->GetYesNoFromUser("Particles are white on a dark background", "Answer yes here if contrast is inverted, i.e. particles have higher densities than the background","no");



	delete my_input;

	my_current_job.Reset(28);
	my_current_job.ManualSetArguments("tffffffffbtbiffftiifbbffbiib",	micrograph_filename.ToStdString().c_str(),
																	pixel_size,
																	acceleration_voltage_in_keV,
																	spherical_aberration_in_mm,
																	amplitude_contrast,
																	additional_phase_shift_in_radians,
																	defocus_1_in_angstroms,
																	defocus_2_in_angstroms,
																	astigmatism_angle_in_degrees,
																	already_have_templates,
																	templates_filename.ToStdString().c_str(),
																	average_templates_radially,
																	number_of_template_rotations,
																	typical_radius,
																	maximum_radius,
																	highest_resolution_to_use,
																	output_stack_filename.ToStdString().c_str(),
																	output_stack_box_size,
																	minimum_distance_from_edges,
																	picking_threshold,
																	avoid_low_variance_areas,
																	avoid_high_variance_areas,
																	low_variance_threshold_in_fwhm,
																	high_variance_threshold_in_fwhm,
																	avoid_high_low_mean_areas,
																	algorithm_to_find_background,
																	number_of_background_boxes,
																	particles_are_white
																	);


}

// override the do calculation method which will be what is actually run..
bool FindParticlesApp::DoCalculation()
{

	ParticleFinder particle_finder;

	// Get the arguments for this job..
	wxString	micrograph_filename 						= 	my_current_job.arguments[0].ReturnStringArgument();
	float		original_micrograph_pixel_size				=	my_current_job.arguments[1].ReturnFloatArgument();
	float		acceleration_voltage_in_keV					=	my_current_job.arguments[2].ReturnFloatArgument();
	float		spherical_aberration_in_mm					=	my_current_job.arguments[3].ReturnFloatArgument();
	float		amplitude_contrast							=	my_current_job.arguments[4].ReturnFloatArgument();
	float		additional_phase_shift_in_radians			=	my_current_job.arguments[5].ReturnFloatArgument();
	float		defocus_1_in_angstroms						=	my_current_job.arguments[6].ReturnFloatArgument();
	float		defocus_2_in_angstroms						=	my_current_job.arguments[7].ReturnFloatArgument();
	float		astigmatism_angle_in_degrees				=	my_current_job.arguments[8].ReturnFloatArgument();
	bool		already_have_templates						=	my_current_job.arguments[9].ReturnBoolArgument();
	wxString	templates_filename							= 	my_current_job.arguments[10].ReturnStringArgument();
	bool		average_templates_radially					=	my_current_job.arguments[11].ReturnBoolArgument();
	int			number_of_template_rotations				=	my_current_job.arguments[12].ReturnIntegerArgument();
	float		typical_radius_in_angstroms					=	my_current_job.arguments[13].ReturnFloatArgument();
	float		maximum_radius_in_angstroms					=	my_current_job.arguments[14].ReturnFloatArgument();
	float		highest_resolution_to_use					=	my_current_job.arguments[15].ReturnFloatArgument();
	wxString	output_stack_filename						=	my_current_job.arguments[16].ReturnStringArgument();
	int			output_stack_box_size						=	my_current_job.arguments[17].ReturnIntegerArgument();
	int			minimum_distance_from_edges_in_pixels		=	my_current_job.arguments[18].ReturnIntegerArgument();
	float		minimum_peak_height_for_candidate_particles = 	my_current_job.arguments[19].ReturnFloatArgument();
	bool		avoid_low_variance_areas					=	my_current_job.arguments[20].ReturnBoolArgument();
	bool		avoid_high_variance_areas					=	my_current_job.arguments[21].ReturnBoolArgument();
	float		low_variance_threshold_in_fwhm				=	my_current_job.arguments[22].ReturnFloatArgument();
	float		high_variance_threshold_in_fwhm				=	my_current_job.arguments[23].ReturnFloatArgument();
	bool		avoid_high_low_mean_areas					=	my_current_job.arguments[24].ReturnBoolArgument();
	int			algorithm_to_find_background				=	my_current_job.arguments[25].ReturnIntegerArgument();
	int			number_of_background_boxes					=	my_current_job.arguments[26].ReturnIntegerArgument();
	bool		particles_are_white							=	my_current_job.arguments[27].ReturnBoolArgument();


	particle_finder.SetAllUserParameters(   micrograph_filename,
                                            original_micrograph_pixel_size,
			                                acceleration_voltage_in_keV,
			                                spherical_aberration_in_mm,
			                                amplitude_contrast,
			                                additional_phase_shift_in_radians,
			                                defocus_1_in_angstroms,
			                                defocus_2_in_angstroms,
			                                astigmatism_angle_in_degrees,
			                                already_have_templates,
			                                templates_filename,
			                                average_templates_radially,
			                                number_of_template_rotations,
			                                typical_radius_in_angstroms,
			                                maximum_radius_in_angstroms,
			                                highest_resolution_to_use,
			                                output_stack_filename,
			                                output_stack_box_size,
			                                minimum_distance_from_edges_in_pixels,
			                                minimum_peak_height_for_candidate_particles,
			                                avoid_low_variance_areas,
											avoid_high_variance_areas,
											low_variance_threshold_in_fwhm,
											high_variance_threshold_in_fwhm,
			                                avoid_high_low_mean_areas,
			                                algorithm_to_find_background,
			                                number_of_background_boxes,
											particles_are_white);

	particle_finder.write_out_plt = is_running_locally;
	if (is_running_locally) wxPrintf("Running locally. Should write PLT out.\n");

	particle_finder.DoItAll();


	// Put together results

	float *result_array;

	if (particle_finder.results_x_y.number_of_points > 0) result_array = new float[ 5 * particle_finder.results_x_y.number_of_points ];

	long address = 0;
	for ( long counter = 0; counter < particle_finder.results_x_y.number_of_points; counter ++ )
	{
		result_array[address] = particle_finder.results_x_y.data_x[counter]; // x
		address ++;
		result_array[address] = particle_finder.results_x_y.data_y[counter]; // y
		address ++;
		result_array[address] = particle_finder.results_height_template.data_x[counter]; // peak height
		address ++;
		result_array[address] = particle_finder.results_height_template.data_y[counter]; // which template
		address ++;
		result_array[address] = particle_finder.results_rotation.data_x[counter]; // template rotation (in degrees)
		address ++;
	}

	my_result.SetResult(5*particle_finder.results_x_y.number_of_points, result_array);


	// Cleanup
	if (particle_finder.results_x_y.number_of_points > 0) delete [] result_array;

	return true;
}





