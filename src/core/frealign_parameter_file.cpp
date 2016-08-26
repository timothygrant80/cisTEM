#include "core_headers.h"

/*FrealignParameterFile::FrealignParameterFile()
{
	parameter_file = NULL;
	MyPrintWithDetails("FrealignParameterFile has been declared with no filename.\n");
	abort();
} */

FrealignParameterFile::FrealignParameterFile()
{
	filename = "";
	access_type = 0;
	parameter_file = NULL;
	number_of_lines = 0;
	current_line = 0;
	parameter_cache = NULL;
	average_defocus = 0.0;
	defocus_coeff_a = 0.0;
	defocus_coeff_b = 0.0;
	records_per_line = 0;
}

FrealignParameterFile::FrealignParameterFile(wxString wanted_filename, int wanted_access_type, int wanted_records_per_line)
{
	Open(wanted_filename, wanted_access_type, wanted_records_per_line);
}

FrealignParameterFile::~FrealignParameterFile()
{
	Close();
}

void FrealignParameterFile::Open(wxString wanted_filename, int wanted_access_type, int wanted_records_per_line)
{
	filename = wanted_filename;
	access_type = wanted_access_type;
	number_of_lines = 0;
	current_line = 0;
	parameter_cache = NULL;
	average_defocus = 0.0;
	defocus_coeff_a = 0.0;
	defocus_coeff_b = 0.0;

	if (access_type == 1)
	{
		parameter_file = fopen(filename, "w");
		if (parameter_file == NULL)
		{
			MyPrintWithDetails("Error: Cannot open Frealign parameter file (%s) for write\n", wanted_filename);
			abort();
		}
	}
	else
	{
		parameter_file = fopen(filename, "r");
		if (parameter_file == NULL)
		{
			MyPrintWithDetails("Error: Cannot open Frealign parameter file (%s) for read\n", wanted_filename);
			abort();
		}
		records_per_line = 16;
	}

	records_per_line = wanted_records_per_line;
}
void FrealignParameterFile::Close()
{
	if (parameter_cache != NULL)
	{
		delete [] parameter_cache;
		parameter_cache = NULL;
	}
	if (access_type == 1) fclose(parameter_file);
}

void FrealignParameterFile::WriteCommentLine(wxString comment_string)
{
	if (comment_string.StartsWith("C") == false)
	{
		comment_string = "C " + comment_string;
	}
	fprintf(parameter_file, "%s\n", comment_string.ToUTF8().data());
}

void FrealignParameterFile::WriteLine(float *parameters, bool comment)
{
	MyDebugAssertTrue(access_type == 1, "File not opened for WRITE");

	if (records_per_line == 16)
	{
		if (comment)
		{
			fprintf(parameter_file, "C       %7.2f %7.2f %7.2f %9.2f %9.2f %7.0f %5i %8.1f %8.1f %7.2f %7.2f %9i %10.4f %7.2f %7.2f\n",
					float(parameters[1]), float(parameters[2]), float(parameters[3]), float(parameters[4]), float(parameters[5]),
					float(parameters[6]), int(parameters[7]), float(parameters[8]), float(parameters[9]), float(parameters[10]),
					float(parameters[11]), int(parameters[12]), float(parameters[13]), float(parameters[14]), float(parameters[15]));
		}
		else
		{
			fprintf(parameter_file, "%7i %7.2f %7.2f %7.2f %9.2f %9.2f %7.0f %5i %8.1f %8.1f %7.2f %7.2f %9i %10.4f %7.2f %7.2f\n",
					int(parameters[0]), float(parameters[1]), float(parameters[2]), float(parameters[3]), float(parameters[4]), float(parameters[5]),
					float(parameters[6]), int(parameters[7]), float(parameters[8]), float(parameters[9]), float(parameters[10]),
					float(parameters[11]), int(parameters[12]), float(parameters[13]), float(parameters[14]), float(parameters[15]));
		}
	}
	else
	{
		if (comment)
		{
			fprintf(parameter_file, "C       %7.2f %7.2f %7.2f %9.2f %9.2f %7.0f %5i %8.1f %8.1f %7.2f %7.2f %9i %10.4f %7.2f\n",
					float(parameters[1]), float(parameters[2]), float(parameters[3]), float(parameters[4]), float(parameters[5]),
					float(parameters[6]), int(parameters[7]), float(parameters[8]), float(parameters[9]), float(parameters[10]),
					float(parameters[11]), int(parameters[12]), float(parameters[13]), float(parameters[14]));
		}
		else
		{
			fprintf(parameter_file, "%7i %7.2f %7.2f %7.2f %9.2f %9.2f %7.0f %5i %8.1f %8.1f %7.2f %7.2f %9i %10.4f %7.2f\n",
					int(parameters[0]), float(parameters[1]), float(parameters[2]), float(parameters[3]), float(parameters[4]), float(parameters[5]),
					float(parameters[6]), int(parameters[7]), float(parameters[8]), float(parameters[9]), float(parameters[10]),
					float(parameters[11]), int(parameters[12]), float(parameters[13]), float(parameters[14]));
		}
	}
	number_of_lines++;
}

int FrealignParameterFile::ReadFile()
{
	MyDebugAssertTrue(access_type == 0, "File not opened for READ");

	long	file_size;
	int		line_length;
	int		i;
	int		line;
	int		elements_read;
	int		lines_read = 0;
	char	dataline[1000];
	char	*char_pointer;

	file_size = ReturnFileSizeInBytes(filename);

	dataline[0] = 'C';
	while (dataline[0] == 'C')
	{
		fgets(dataline, sizeof dataline, parameter_file);
		lines_read++;
	}
	line_length = strlen(dataline);
	number_of_lines = file_size / line_length;
	if (parameter_cache != NULL) delete [] parameter_cache;
	parameter_cache = new float[records_per_line * number_of_lines];

	current_line = 0;
	for (line = 0; line < number_of_lines + 1; line++)
//	for (line = 0; line < 10; line++)
	{
		if (dataline[0] != 'C')
		{
			if (strlen(dataline) != line_length) {wxPrintf("Warning: line %i has different length than first data line\n", lines_read);};
			elements_read = records_per_line * current_line;
			char_pointer = dataline;
			for (i = 0; i < records_per_line; i++) {parameter_cache[i + elements_read] = strtof(char_pointer, &char_pointer);};
			current_line++;
		}
		lines_read++;
		if (fgets(dataline, sizeof dataline, parameter_file) == NULL) break;
	}
	number_of_lines = current_line;
	wxPrintf("\n %i data lines read\n", number_of_lines);
	current_line = 0;

	fclose(parameter_file);

	return number_of_lines;
}

void FrealignParameterFile::ReadLine(float *parameters)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");

	int i;
	int elements_read = records_per_line * current_line;

	for (i = 0; i < records_per_line; i++) {parameters[i] = parameter_cache[i + elements_read];};
	current_line++;
}

float FrealignParameterFile::ReadParameter(int wanted_line_number, int wanted_parameter)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");

	int elements_read = records_per_line * wanted_line_number;

	return parameter_cache[wanted_parameter + elements_read];
}

void FrealignParameterFile::UpdateParameter(int wanted_line_number, int wanted_parameter, float wanted_value)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");

	int elements_read = records_per_line * wanted_line_number;

	parameter_cache[wanted_parameter + elements_read] = wanted_value;
}

void FrealignParameterFile::Rewind()
{
	current_line = 0;
}

float FrealignParameterFile::ReturnThreshold(float wanted_percentage)
{
	MyDebugAssertTrue(wanted_percentage < 1.0 && wanted_percentage >= 0.0, "Percentage out of range");

	int i;
	int line;
	int index;
	int number_of_bins = 10000;
	float average_occ = 0.0;
	float sum_occ;
	float increment = 100.0 / float(number_of_bins);
	float threshold;
	float percentage;

	for (line = 0; line < number_of_lines; line++)
	{
		average_occ += parameter_cache[11 + records_per_line * line];
	}
	average_occ /= number_of_lines;

	for (i = number_of_bins; i > 0; i--)
	{
		sum_occ = 0.0;
		threshold = float(i) * increment;
		for (line = 0; line < number_of_lines; line++)
		{
			index = records_per_line * line;
			if (parameter_cache[14 + index] >= threshold) sum_occ += parameter_cache[11 + index];
		}
		percentage = sum_occ / number_of_lines / average_occ;
		if (percentage >= wanted_percentage) break;
	}

	if (sum_occ == 0.0)
	{
		MyPrintWithDetails("Error: Number of particles selected = 0; please change score threshold\n");
		abort();
	}

	return threshold;
}

void FrealignParameterFile::CalculateDefocusDependence()
{
	int line;
	float s = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
	float delta;

	for (line = 0; line < number_of_lines; line++)
	{
		average_defocus = (parameter_cache[8 + records_per_line * line] + parameter_cache[9 + records_per_line * line]) / 2.0;
		s += parameter_cache[11 + records_per_line * line];
		sx += average_defocus * parameter_cache[11 + records_per_line * line];
		sy += parameter_cache[14 + records_per_line * line] * parameter_cache[11 + records_per_line * line];
		sxx += powf(average_defocus,2) * parameter_cache[11 + records_per_line * line];
		sxy += average_defocus * parameter_cache[14 + records_per_line * line] * parameter_cache[11 + records_per_line * line];
	}
	average_defocus = sx / s;
	delta = s * sxx - powf(sx,2);
	defocus_coeff_a = (sxx * sy - sx * sxy) / delta;
	defocus_coeff_b = (s * sxy - sx * sy) / delta;
//	wxPrintf("average_defocus = %g, defocus_coeff_a = %g, defocus_coeff_b = %g\n", average_defocus, defocus_coeff_a, defocus_coeff_b);
}

void FrealignParameterFile::AdjustScores()
{
	int line;
	float defocus;

	for (line = 0; line < number_of_lines; line++)
	{
		defocus = (parameter_cache[8 + records_per_line * line] + parameter_cache[9 + records_per_line * line]) / 2.0;
		parameter_cache[14 + records_per_line * line] -= ReturnScoreAdjustment(defocus);
	}
}

float FrealignParameterFile::ReturnScoreAdjustment(float defocus)
{
	MyDebugAssertTrue(average_defocus != 0.0 || defocus_coeff_b != 0.0, "Defous coefficients not determined");

	return (defocus - average_defocus) * defocus_coeff_b;
}
