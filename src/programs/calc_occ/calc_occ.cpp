#include "../../core/core_headers.h"

class
        CalcOccApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(CalcOccApp)

// override the DoInteractiveUserInput

void CalcOccApp::DoInteractiveUserInput( ) {
    int      number_of_parameter_files;
    float    occupancy_change_multiplier = 1.0;
    wxString input_file_seed;
    wxString output_file_seed;

    UserInput* my_input = new UserInput("CalcOcc", 1.00);

    number_of_parameter_files   = my_input->GetIntFromUser("Number of parameter files (0 = all)", "The number of parameter files to process; enter 0 to read all that match the seed", "0", 0);
    occupancy_change_multiplier = my_input->GetFloatFromUser("Occupancy change multiplier", "The change in occupancies from the input files will be multiplied by this number", "1.0", 0.0, 1.0);
    input_file_seed             = my_input->GetFilenameFromUser("Seed for input parameter filenames", "The string common to the names of the input parameter files to be processed", "my_input_1_r.par", false);
    output_file_seed            = my_input->GetFilenameFromUser("Seed for output parameter filenames", "The string common to the names of the output parameter files", "my_output_1_r.par", false);

    delete my_input;

    //	my_current_job.Reset(4);
    my_current_job.ManualSetArguments("iftt", number_of_parameter_files,
                                      occupancy_change_multiplier,
                                      input_file_seed.ToUTF8( ).data( ),
                                      output_file_seed.ToUTF8( ).data( ));
}

// override the do calculation method which will be what is actually run..

bool CalcOccApp::DoCalculation( ) {
    int      number_of_parameter_files   = my_current_job.arguments[0].ReturnIntegerArgument( );
    float    occupancy_change_multiplier = my_current_job.arguments[1].ReturnFloatArgument( );
    wxString input_file_seed             = my_current_job.arguments[2].ReturnStringArgument( );
    wxString output_file_seed            = my_current_job.arguments[3].ReturnStringArgument( );

    int   i, j;
    int   number_of_files;
    int   count;
    float input_parameters[16];
    float particle_position;
    float max_logp;
    float average_sigma;
    float sum_probabilities;
    float occupancy;
    //wxFileName	parameter_file_name = input_file_seed;
    wxString extension = wxFileName(input_file_seed).GetExt( );
    wxString parameter_file;

    Refinement input_refinement;

    // count parameter files
    number_of_files = 0;
    parameter_file  = wxFileName::StripExtension(input_file_seed) + wxString::Format("%i", number_of_files + 1) + "." + extension;
    while ( DoesFileExist(parameter_file) ) {
        number_of_files++;
        parameter_file = wxFileName::StripExtension(input_file_seed) + wxString::Format("%i", number_of_files + 1) + "." + extension;
    }
    if ( number_of_files == 0 ) {
        MyPrintWithDetails("Error: Parameter file %s not found\n", parameter_file);
        DEBUG_ABORT;
    }
    if ( number_of_files > number_of_parameter_files && number_of_parameter_files != 0 )
        number_of_files = number_of_parameter_files;

    FrealignParameterFile* input_parameter_files = NULL;
    input_parameter_files                        = new FrealignParameterFile[number_of_files];
    float average_occupancies[number_of_files];

    wxPrintf("\nReading %i parameter files...\n", number_of_files);
    for ( i = 0; i < number_of_files; i++ ) {
        parameter_file = wxFileName::StripExtension(input_file_seed) + wxString::Format("%i", i + 1) + "." + extension;
        wxPrintf("\n%s", parameter_file);
        input_parameter_files[i].Open(parameter_file, OPEN_TO_READ);
        input_parameter_files[i].ReadFile( );
    }
    wxPrintf("\nFinished reading files\n\n");

    input_refinement.SizeAndFillWithEmpty(input_parameter_files[0].number_of_lines, number_of_files);

    for ( int class_counter = 0; class_counter < input_refinement.number_of_classes; class_counter++ ) {
        for ( long particle_counter = 0; particle_counter < input_refinement.number_of_particles; particle_counter++ ) {
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].position_in_stack = input_parameter_files[class_counter].ReadParameter(particle_counter, 0);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].psi               = input_parameter_files[class_counter].ReadParameter(particle_counter, 1);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].theta             = input_parameter_files[class_counter].ReadParameter(particle_counter, 2);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].phi               = input_parameter_files[class_counter].ReadParameter(particle_counter, 3);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].xshift            = input_parameter_files[class_counter].ReadParameter(particle_counter, 4);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].yshift            = input_parameter_files[class_counter].ReadParameter(particle_counter, 5);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_is_active   = input_parameter_files[class_counter].ReadParameter(particle_counter, 7);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].defocus1          = input_parameter_files[class_counter].ReadParameter(particle_counter, 8);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].defocus2          = input_parameter_files[class_counter].ReadParameter(particle_counter, 9);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].defocus_angle     = input_parameter_files[class_counter].ReadParameter(particle_counter, 10);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].phase_shift       = input_parameter_files[class_counter].ReadParameter(particle_counter, 11);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp              = input_parameter_files[class_counter].ReadParameter(particle_counter, 13);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy         = input_parameter_files[class_counter].ReadParameter(particle_counter, 12);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].sigma             = input_parameter_files[class_counter].ReadParameter(particle_counter, 14);
            input_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].score             = input_parameter_files[class_counter].ReadParameter(particle_counter, 15);
        }
    }

    input_refinement.UpdateOccupancies( );

    for ( long particle_counter = 0; particle_counter < 10; particle_counter++ ) {
        wxPrintf("%li, occ = %f, logp = %f, sigma = %f\n", input_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].position_in_stack, input_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].occupancy, input_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].logp, input_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].sigma);
    }

    /*

	// calculate average occupancies
	wxPrintf("Average occupancies:\n");
	for (i = 0; i < number_of_files; i++)
	{
		count = 0;
		average_occupancies[i] = 0.0;
		particle_position = 0.0;
		for (j = 0; j < input_parameter_files[i].number_of_lines; j++)
		{
			average_occupancies[i] += input_parameter_files[i].ReadParameter(j, 12);
			if (particle_position != input_parameter_files[i].ReadParameter(j, 0))
			{
				count++;
				particle_position = input_parameter_files[i].ReadParameter(j, 0);
			}
		}
		average_occupancies[i] /= count;
		wxPrintf("Parameter file %3i: %8.2f\n", i, average_occupancies[i]);
	}

	for (j = 0; j < input_parameter_files[0].number_of_lines; j++)
	{
		max_logp = - std::numeric_limits<float>::max();
		particle_position = input_parameter_files[0].ReadParameter(j, 0);
		for (i = 0; i < number_of_files; i++)
		{
			if (particle_position != input_parameter_files[i].ReadParameter(j, 0))
			{
				MyPrintWithDetails("Error: Inconsistent particle positions in line %i\n", j + 1);
				DEBUG_ABORT;
			}
			max_logp = std::max(max_logp,input_parameter_files[i].ReadParameter(j, 13));
		}
		sum_probabilities = 0.0;
		for (i = 0; i < number_of_files; i++)
		{
			if (max_logp - input_parameter_files[i].ReadParameter(j, 13) < 10.0)
			{
				sum_probabilities += exp(input_parameter_files[i].ReadParameter(j, 13) - max_logp) * average_occupancies[i];
			}
		}
		average_sigma = 0.0;
		for (i = 0; i < number_of_files; i++)
		{
			if (max_logp - input_parameter_files[i].ReadParameter(j, 13) < 10.0)
			{
				occupancy = exp(input_parameter_files[i].ReadParameter(j, 13) - max_logp) * average_occupancies[i] / sum_probabilities *100.0;
			}
			else
			{
				occupancy = 0.0;
			}
			occupancy = occupancy_change_multiplier * (occupancy - input_parameter_files[i].ReadParameter(j, 12)) + input_parameter_files[i].ReadParameter(j, 12);
			input_parameter_files[i].UpdateParameter(j, 12, occupancy);
			average_sigma += input_parameter_files[i].ReadParameter(j, 14) * occupancy / 100.0;
		}
		for (i = 0; i < number_of_files; i++)
		{
			input_parameter_files[i].UpdateParameter(j, 14, average_sigma);
		}
	}

	// Write parameter files
	wxPrintf("\nWriting %i parameter files...\n", number_of_files);
	for (i = 0; i < number_of_files; i++)
	{
		parameter_file = wxFileName::StripExtension(output_file_seed) + wxString::Format("%i", i + 1) + "." + extension;
		wxPrintf("\n%s", parameter_file);
		FrealignParameterFile output_parameter_file(parameter_file, OPEN_TO_WRITE);
		input_parameter_files[i].Rewind();
		for (j = 0; j < input_parameter_files[i].number_of_lines; j++)
		{
			input_parameter_files[i].ReadLine(input_parameters);
			output_parameter_file.WriteLine(input_parameters);
		}
	}
	wxPrintf("\n\nFinished writing files\n");

	wxPrintf("\nCalcOcc: Normal termination\n\n");

	*/

    return true;
}
