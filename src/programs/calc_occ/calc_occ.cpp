#include "../../core/core_headers.h"

class
CalcOccApp : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};

IMPLEMENT_APP(CalcOccApp)

// override the DoInteractiveUserInput

void CalcOccApp::DoInteractiveUserInput()
{
	int			number_of_parameter_files;
	float		occupancy_change_multiplier = 1.0;
	wxString	input_file_seed;
	wxString	output_file_seed;

	UserInput *my_input = new UserInput("CalcOcc", 1.00);

	number_of_parameter_files = my_input->GetIntFromUser("Number of parameter files (0 = all)", "The number of parameter files to process; enter 0 to read all that match the seed", "0", 0);
	occupancy_change_multiplier = my_input->GetFloatFromUser("Occupancy change multiplier", "The change in occupancies from the input files will be multiplied by this number", "1.0", 0.0, 1.0);
	input_file_seed = my_input->GetFilenameFromUser("Seed for input parameter filenames", "The string common to the names of the input parameter files to be processed", "my_input_1_r.par", false);
	output_file_seed = my_input->GetFilenameFromUser("Seed for output parameter filenames", "The string common to the names of the output parameter files", "my_output_1_r.par", false);

	delete my_input;

	my_current_job.Reset(4);
	my_current_job.ManualSetArguments("iftt",	number_of_parameter_files,
												occupancy_change_multiplier,
												input_file_seed.ToUTF8().data(),
												output_file_seed.ToUTF8().data());
}

// override the do calculation method which will be what is actually run..

bool CalcOccApp::DoCalculation()
{
	int		 number_of_parameter_files			= my_current_job.arguments[0].ReturnIntegerArgument();
	float 	 occupancy_change_multiplier		= my_current_job.arguments[1].ReturnFloatArgument();
	wxString input_file_seed 					= my_current_job.arguments[2].ReturnStringArgument();
	wxString output_file_seed 					= my_current_job.arguments[3].ReturnStringArgument();

	int			i, j;
	int			number_of_files;
	int			count;
	float 		input_parameters[16];
	float		particle_position;
	float		max_logp;
	float		average_sigma;
	float		sum_probabilities;
	float		occupancy;
	//wxFileName	parameter_file_name = input_file_seed;
	wxString	extension = wxFileName(input_file_seed).GetExt();
	wxString	parameter_file;

	// count parameter files
	number_of_files = 0;
	parameter_file = wxFileName::StripExtension(input_file_seed) + wxString::Format("%i", number_of_files + 1) + "." + extension;
	while (DoesFileExist(parameter_file))
	{
		number_of_files++;
		parameter_file = wxFileName::StripExtension(input_file_seed) + wxString::Format("%i", number_of_files + 1) + "." + extension;
	}
	if (number_of_files == 0)
	{
		MyPrintWithDetails("Error: Parameter file %s not found\n", parameter_file);
		abort();
	}
	if (number_of_files > number_of_parameter_files && number_of_parameter_files != 0) number_of_files = number_of_parameter_files;

	FrealignParameterFile *input_parameter_files = NULL;
	input_parameter_files = new FrealignParameterFile [number_of_files];
	float average_occupancies[number_of_files];

	wxPrintf("\nReading %i parameter files...\n", number_of_files);
	for (i = 0; i < number_of_files; i++)
	{
		parameter_file = wxFileName::StripExtension(input_file_seed) + wxString::Format("%i", i + 1) + "." + extension;
		wxPrintf("\n%s", parameter_file);
		input_parameter_files[i].Open(parameter_file, OPEN_TO_READ);
		input_parameter_files[i].ReadFile();
	}
	wxPrintf("\nFinished reading files\n\n");

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
				abort();
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

	return true;
}
