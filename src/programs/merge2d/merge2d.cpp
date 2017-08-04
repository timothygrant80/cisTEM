#include "../../core/core_headers.h"

class
Merge2DApp : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();
	void ReadArrayHeader(wxString filename);
	void ReadArrays(wxString filename);
	void AddArrays();

	int xy_dimensions;
	int number_of_classes;
	int number_of_nonzero_classes;
	int images_processed;
	int *list_of_nozero_classes;
	float sum_logp_total;
	float sum_snr;
	float pixel_size;
	float mask_radius;
	float mask_falloff;
	float log_range;
	float average_snr;
	float *class_logp;
	Image *class_averages;
	Image *CTF_sums;
	int temp_processed;
	float temp_logp_total;
	float temp_snr;
	float *temp_logp;
	Image *temp_averages;
	Image *temp_sums;
	wxString dump_file_seed;

	private:
};

IMPLEMENT_APP(Merge2DApp)

// override the DoInteractiveUserInput

void Merge2DApp::DoInteractiveUserInput()
{
	wxString	ouput_class_averages;

	UserInput *my_input = new UserInput("Merge2D", 1.00);

	ouput_class_averages = my_input->GetFilenameFromUser("Output class averages", "The refined 2D class averages", "my_refined_classes.mrc", false);
	dump_file_seed = my_input->GetFilenameFromUser("Seed for input dump filenames for intermediate arrays", "The seed name of the dump files with the intermediate 2D class sums", "dump_file_seed_.dat", false);

	delete my_input;

	my_current_job.Reset(2);
	my_current_job.ManualSetArguments("tt",	ouput_class_averages.ToUTF8().data(),
												dump_file_seed.ToUTF8().data());
}

// override the do calculation method which will be what is actually run..

bool Merge2DApp::DoCalculation()
{
	wxString ouput_class_averages				= my_current_job.arguments[0].ReturnStringArgument();
	dump_file_seed 								= my_current_job.arguments[1].ReturnStringArgument();

	int			i;
	int			current_class;
	int			image_counter;
	float		filter_constant;
	float		variance;
	float		occupancy;
	float		temp_float;
	wxFileName	dump_file_name = wxFileName::FileName(dump_file_seed);
	wxString	extension = dump_file_name.GetExt();
	wxString	dump_file;

	dump_file = wxFileName::StripExtension(dump_file_seed) + wxString::Format("%i", 1) + "." + extension;
	if (is_running_locally)
	{
		if (! DoesFileExist(dump_file))
		{
			MyPrintWithDetails("Error: Dump file %s not found\n", dump_file);
			abort();
		}
	}
	else
	{
		if (! DoesFileExistWithWait(dump_file, 30))
		{
			MyPrintWithDetails("Error: Dump file %s not found\n", dump_file);
			abort();
		}
	}

	ReadArrayHeader(dump_file);
	wxPrintf("\nNumber of classes = %i, nonzero classes = %i, box size = %i, pixel size = %f\n", number_of_classes, number_of_nonzero_classes, xy_dimensions, pixel_size);

	list_of_nozero_classes = new int [number_of_classes];
	class_logp = new float [number_of_nonzero_classes];
	temp_logp = new float [number_of_nonzero_classes];
	class_averages = new Image [number_of_nonzero_classes];
	temp_averages = new Image [number_of_nonzero_classes];
	CTF_sums = new Image [number_of_nonzero_classes];
	temp_sums = new Image [number_of_nonzero_classes];
	for (i = 0; i < number_of_nonzero_classes; i++)
	{
		class_averages[i].Allocate(xy_dimensions, xy_dimensions, false);
		temp_averages[i].Allocate(xy_dimensions, xy_dimensions, false);
		CTF_sums[i].Allocate(xy_dimensions, xy_dimensions, false);
		temp_sums[i].Allocate(xy_dimensions, xy_dimensions, false);
		class_averages[i].SetToConstant(0.0);
		CTF_sums[i].SetToConstant(0.0);
		class_logp[i] = - std::numeric_limits<float>::max();
	}

	wxPrintf("\nReading intermediate arrays...\n\n");

	i = 1;
	sum_logp_total = - std::numeric_limits<float>::max();
	sum_snr = 0.0;
	images_processed = 0;
	while (DoesFileExist(dump_file))
	{
		wxPrintf("%s\n", dump_file);
		ReadArrays(dump_file);
		AddArrays();
		i++;
		dump_file = wxFileName::StripExtension(dump_file_seed) + wxString::Format("%i", i) + "." + extension;
	}

	MRCFile output_classes(ouput_class_averages.ToStdString(), true);
	Image temp_image;
	temp_image.Allocate(xy_dimensions, xy_dimensions, true);

	image_counter = 0;
	temp_image.SetToConstant(0.0);
	for (current_class = 0; current_class < number_of_nonzero_classes; current_class++)
	{
		if (fabsf(class_logp[current_class]) <= log_range)
		{
			// Divide summed class likelihood by number of images
			occupancy = class_logp[current_class] - logf(images_processed);
			if (occupancy >= - log_range)
			{
				occupancy = exp(occupancy);
			}
			else
			{
				occupancy = 0.0;
			}
			if (occupancy > 0.0)
			{
				filter_constant = occupancy * sum_snr / images_processed;
				for (i = 0; i < class_averages[current_class].real_memory_allocated / 2; i++)
				{
					class_averages[current_class].complex_values[i] /= (abs(CTF_sums[current_class].complex_values[i]) + occupancy);
				}
				class_averages[current_class].BackwardFFT();
			}
			variance = class_averages[current_class].ReturnSumOfSquares();
//			wxPrintf("images_processed = %i, occupancy = %g, variance = %g\n", images_processed, occupancy, variance);
		}
		else
		{
			occupancy = 0.0;
		}
		while (image_counter < list_of_nozero_classes[current_class])
		{
			temp_image.WriteSlice(&output_classes, image_counter + 1);
			wxPrintf("Class = %4i, average occupancy = %10.4f\n", image_counter + 1, 0.0);
			image_counter++;
		}
		class_averages[current_class].WriteSlice(&output_classes, image_counter + 1);
		wxPrintf("Class = %4i, average occupancy = %10.4f\n", image_counter + 1, 100.0 * occupancy);
		image_counter++;
	}
	while (image_counter < number_of_classes)
	{
		temp_image.WriteSlice(&output_classes, image_counter + 1);
		image_counter++;
	}

	wxPrintf("\nTotal logp = %g\n", sum_logp_total);

	delete [] list_of_nozero_classes;
	delete [] class_averages;
	delete [] temp_averages;
	delete [] CTF_sums;
	delete [] temp_sums;
	delete [] class_logp;
	delete [] temp_logp;

	wxPrintf("\nMerge2D: Normal termination\n\n");

	return true;
}

void Merge2DApp::ReadArrayHeader(wxString filename)
{
	int i;
	int count = 4 * sizeof(int) + 6 * sizeof(float);
	float local_mask_radius;
	char temp_char[count];
	char *char_pointer;

	std::ifstream b_stream(filename.c_str(), std::fstream::in | std::fstream::binary);

	b_stream.read(temp_char, count);
	count = 0;
	char_pointer = (char *) &xy_dimensions;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &number_of_classes;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &number_of_nonzero_classes;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &temp_processed;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &sum_logp_total;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &sum_snr;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &pixel_size;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &mask_radius;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &mask_falloff;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &log_range;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};

	b_stream.close();
}

void Merge2DApp::ReadArrays(wxString filename)
{
	int i;
	int count = 4 * sizeof(int) + 6 * sizeof(float);
	char temp_char[count];
	char *char_pointer;
	int input_xy_dimensions;
	int input_number_of_classes;
	int input_number_of_nonzero_classes;
	float input_pixel_size;
	float input_mask_radius;
	float input_mask_falloff;
	float input_log_range;

	std::ifstream b_stream(filename.c_str(), std::fstream::in | std::fstream::binary);

	b_stream.read(temp_char, count);
	count = 0;
	char_pointer = (char *) &input_xy_dimensions;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_number_of_classes;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_number_of_nonzero_classes;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &temp_processed;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &temp_logp_total;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &temp_snr;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_pixel_size;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_mask_radius;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_mask_falloff;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_log_range;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};

	if (input_xy_dimensions != xy_dimensions || input_number_of_classes != number_of_classes || input_number_of_nonzero_classes != number_of_nonzero_classes || input_pixel_size != pixel_size)
	{
		MyPrintWithDetails("Error: Dump file incompatible with 2D class averages\n");
		abort();
	}
	char_pointer = (char *) list_of_nozero_classes;
	b_stream.read(char_pointer, sizeof(int) * number_of_classes);
	char_pointer = (char *) temp_logp;
	b_stream.read(char_pointer, sizeof(float) * number_of_nonzero_classes);

	for (i = 0; i < number_of_nonzero_classes; i++)
	{
		char_pointer = (char *) temp_averages[i].real_values;
		b_stream.read(char_pointer, sizeof(float) * temp_averages[i].real_memory_allocated);
		char_pointer = (char *) temp_sums[i].real_values;
		b_stream.read(char_pointer, sizeof(float) * temp_sums[i].real_memory_allocated);
	}

	b_stream.close();
}

void Merge2DApp::AddArrays()
{
	int current_class;

	images_processed += temp_processed;
	sum_snr += temp_snr;
	for (current_class = 0; current_class < number_of_nonzero_classes; current_class++)
	{
		sum_logp_total = ReturnSumOfLogP(sum_logp_total, temp_logp_total, log_range);
		class_logp[current_class] = ReturnSumOfLogP(class_logp[current_class], temp_logp[current_class], log_range);
		class_averages[current_class].AddImage(&temp_averages[current_class]);
		CTF_sums[current_class].AddImage(&temp_sums[current_class]);
	}
}
