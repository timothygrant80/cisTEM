#include "core_headers.h"

/*FrealignParameterFile::FrealignParameterFile()
{
	parameter_file = NULL;
	MyPrintWithDetails("FrealignParameterFile has been declared with no filename.\n");
	DEBUG_ABORT;
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
			DEBUG_ABORT;
		}
	}
	else
	{
		parameter_file = fopen(filename, "r");
		if (parameter_file == NULL)
		{
			MyPrintWithDetails("Error: Cannot open Frealign parameter file (%s) for read\n", wanted_filename);
			DEBUG_ABORT;
		}
		records_per_line = 17;
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

	if (records_per_line == 17)
	{
		if (comment)
		{
			fprintf(parameter_file, "C       %7.2f %7.2f %7.2f %9.2f %9.2f %7.0f %5i %8.1f %8.1f %7.2f %7.2f %7.2f %9i %10.4f %7.2f %7.2f\n",
					float(parameters[1]), float(parameters[2]), float(parameters[3]), float(parameters[4]), float(parameters[5]),
					float(parameters[6]), int(parameters[7]), float(parameters[8]), float(parameters[9]), float(parameters[10]),
					float(parameters[11]), float(parameters[12]), myroundint(parameters[13]), float(parameters[14]), float(parameters[15]), float(parameters[16]));
		}
		else
		{
			fprintf(parameter_file, "%7i %7.2f %7.2f %7.2f %9.2f %9.2f %7.0f %5i %8.1f %8.1f %7.2f %7.2f %7.2f %9i %10.4f %7.2f %7.2f\n",
					int(parameters[0]), float(parameters[1]), float(parameters[2]), float(parameters[3]), float(parameters[4]), float(parameters[5]),
					float(parameters[6]), int(parameters[7]), float(parameters[8]), float(parameters[9]), float(parameters[10]),
					float(parameters[11]), float(parameters[12]), myroundint(parameters[13]), float(parameters[14]), float(parameters[15]), float(parameters[16]));
		}
	}
	else
	{
		if (comment)
		{
			fprintf(parameter_file, "C       %7.2f %7.2f %7.2f %9.2f %9.2f %7.0f %5i %8.1f %8.1f %7.2f %7.2f %7.2f %9i %10.4f %7.2f\n",
					float(parameters[1]), float(parameters[2]), float(parameters[3]), float(parameters[4]), float(parameters[5]),
					float(parameters[6]), int(parameters[7]), float(parameters[8]), float(parameters[9]), float(parameters[10]),
					float(parameters[11]), float(parameters[12]), myroundint(parameters[13]), float(parameters[14]), float(parameters[15]));
		}
		else
		{
			fprintf(parameter_file, "%7i %7.2f %7.2f %7.2f %9.2f %9.2f %7.0f %5i %8.1f %8.1f %7.2f %7.2f %7.2f %9i %10.4f %7.2f\n",
					int(parameters[0]), float(parameters[1]), float(parameters[2]), float(parameters[3]), float(parameters[4]), float(parameters[5]),
					float(parameters[6]), int(parameters[7]), float(parameters[8]), float(parameters[9]), float(parameters[10]),
					float(parameters[11]), float(parameters[12]), myroundint(parameters[13]), float(parameters[14]), float(parameters[15]));
		}
	}
	number_of_lines++;
}

int FrealignParameterFile::ReadFile(bool exclude_negative_film_numbers, int particles_in_stack)
{
	MyDebugAssertTrue(access_type == 0, "File not opened for READ");

	long	file_size;
	int		line_length;
	int		i;
	int		line;
	int		elements_read;
	int		records_per_line_in_file;
	int		lines_read = 0;
	char	dataline[1000];
	char	*char_pointer;
	bool	one_warning_length = false;
	bool	one_warning_range = false;

	file_size = ReturnFileSizeInBytes(filename);

	dataline[0] = 'C';
	while (dataline[0] == 'C')
	{
		fgets(dataline, sizeof dataline, parameter_file);
		lines_read++;
	}
	line_length = strlen(dataline);
	number_of_lines = file_size / line_length + 1;
	if (parameter_cache != NULL) delete [] parameter_cache;
	parameter_cache = new float[records_per_line * number_of_lines];
	records_per_line_in_file = records_per_line;
	// Test if old Frealign format (phase shift missing)
	if (line_length < 142) records_per_line_in_file = 16;
	if (records_per_line_in_file < records_per_line) wxPrintf("\n Reading old parameter file...\n");

	current_line = 0;
	for (line = 0; line < number_of_lines + 1; line++)
//	for (line = 0; line < 10; line++)
	{
		if (dataline[0] != 'C')
		{
			if (strlen(dataline) != line_length && ! one_warning_length) {wxPrintf("Warning: line %i has different length than first data line\n", lines_read); one_warning_length = true;}
			elements_read = records_per_line * current_line;
			char_pointer = dataline;
			for (i = 0; i < records_per_line_in_file; i++) {parameter_cache[i + elements_read] = strtof(char_pointer, &char_pointer);}
			// Old Frealign format: need to shift last five parameters and add zero phase shift
			if (records_per_line_in_file < records_per_line)
			{
				for (i = records_per_line - 1; i > 11; i--) {parameter_cache[i + elements_read] = parameter_cache[i - 1 + elements_read];}
				parameter_cache[11 + elements_read] = 0.0;
			}
			if (parameter_cache[elements_read] < 1 || (parameter_cache[elements_read] > particles_in_stack && particles_in_stack > -1))
			{
				if (! one_warning_range) {wxPrintf("Warning: particle location in line %i is out of range\n", lines_read); one_warning_range = true;}
			}
			else if (parameter_cache[7 + elements_read] >= 0 || ! exclude_negative_film_numbers) current_line++;
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

void FrealignParameterFile::ReadLine(float *parameters, int wanted_line_number)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");
	MyDebugAssertTrue(current_line <= number_of_lines, "End of Frealign file reached");

	int i;
	int elements_read;

	if (wanted_line_number >= 0) elements_read = records_per_line * wanted_line_number;
	else {elements_read = records_per_line * current_line; current_line++;}

	for (i = 0; i < records_per_line; i++) {parameters[i] = parameter_cache[i + elements_read];};
}

float FrealignParameterFile::ReadParameter(int wanted_line_number, int wanted_index)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");

	int elements_read = records_per_line * wanted_line_number;

	return parameter_cache[wanted_index + elements_read];
}

void FrealignParameterFile::UpdateParameter(int wanted_line_number, int wanted_index, float wanted_value)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");

	int elements_read = records_per_line * wanted_line_number;

	parameter_cache[wanted_index + elements_read] = wanted_value;
}

void FrealignParameterFile::Rewind()
{
	current_line = 0;
}

float FrealignParameterFile::ReturnMin(int wanted_index, bool exclude_negative_film_numbers)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");
	MyDebugAssertTrue(wanted_index >= 0 && wanted_index < records_per_line, "Index out of range");

	int line;
	int index;
	float min;
	float temp_float;

	min = std::numeric_limits<float>::max();
	for (line = 0; line < number_of_lines; line++)
	{
		index = records_per_line * line;
		if (parameter_cache[7 + index] >= 0 || ! exclude_negative_film_numbers)
		{
			temp_float = parameter_cache[wanted_index + index];
			if (min > temp_float) min = temp_float;
		}
	}

	return min;
}

float FrealignParameterFile::ReturnMax(int wanted_index, bool exclude_negative_film_numbers)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");
	MyDebugAssertTrue(wanted_index >= 0 && wanted_index < records_per_line, "Index out of range");

	int line;
	int index;
	float max;
	float temp_float;

	max =  - std::numeric_limits<float>::max();
	for (line = 0; line < number_of_lines; line++)
	{
		index = records_per_line * line;
		if (parameter_cache[7 + index] >= 0 || ! exclude_negative_film_numbers)
		{
			temp_float = parameter_cache[wanted_index + index];
			if (max < temp_float) max = temp_float;
		}
	}

	return max;
}

double FrealignParameterFile::ReturnAverage(int wanted_index, bool exclude_negative_film_numbers)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");
	MyDebugAssertTrue(wanted_index >= 0 && wanted_index < records_per_line, "Index out of range");

	int line;
	int index;
	int sum_i = 0;
	double sum = 0.0;

	for (line = 0; line < number_of_lines; line++)
	{
		index = records_per_line * line;
		if (parameter_cache[7 + index] >= 0 || ! exclude_negative_film_numbers)
		{
			sum += parameter_cache[wanted_index + index];
			sum_i++;
		}
	}

	if (sum_i > 0) return sum / sum_i;
	else return 0.0;
}

void FrealignParameterFile::RemoveOutliers(int wanted_index, float wanted_standard_deviation, bool exclude_negative_film_numbers, bool reciprocal_square)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");
	MyDebugAssertTrue(wanted_index >= 0 && wanted_index < records_per_line, "Index out of range");
	MyDebugAssertTrue(wanted_standard_deviation > 0.0, "Invalid standard deviation");

	int line;
	int index;
	int sum_i = 0;
	double average = 0.0;
	double sum2 = 0.0;
	float std;
	float upper_threshold;
	float lower_threshold;
	float temp_float;

	for (line = 0; line < number_of_lines; line++)
	{
		index = records_per_line * line;
		if (parameter_cache[7 + index] >= 0 || ! exclude_negative_film_numbers)
		{
			temp_float = parameter_cache[wanted_index + index];
			if (reciprocal_square && temp_float > 0.0) temp_float = 1.0 / powf(temp_float, 2);
			average += temp_float;
			sum2 += powf(temp_float, 2);
			sum_i++;
		}
	}

	if (sum_i > 0)
	{
		average /= sum_i;
		std = sum2 / sum_i - powf(average / sum_i, 2);
	}

	if (std > 0.0)
	{
		// Remove extreme outliers and recalculate std
		std = sqrtf(std);
		upper_threshold = average + 2.0 * wanted_standard_deviation * std;
		lower_threshold = average - 2.0 * wanted_standard_deviation * std;
//		wxPrintf("0: average, std, upper, lower = %g %g %g %g\n", float(average), std, upper_threshold, lower_threshold);
		average = 0.0;
		sum2 = 0.0;
		sum_i = 0;
		for (line = 0; line < number_of_lines; line++)
		{
			index = records_per_line * line;
			if (parameter_cache[7 + index] >= 0 || ! exclude_negative_film_numbers)
			{
				temp_float = parameter_cache[wanted_index + index];
				if (reciprocal_square && temp_float > 0.0) temp_float = 1.0 / powf(temp_float, 2);
				if (temp_float <= upper_threshold && temp_float >= lower_threshold)
				{
					average += temp_float;
					sum2 += powf(temp_float, 2);
					sum_i++;
				}
			}
		}

		if (sum_i > 0)
		{
			average /= sum_i;
			std = sum2 / sum_i - powf(average / sum_i, 2);
		}

		// Now remove outliers according to (hopefully) more reasonable std
		std = sqrtf(std);
		upper_threshold = average + wanted_standard_deviation * std;
		lower_threshold = average - wanted_standard_deviation * std;
//		wxPrintf("1: average, std, upper, lower = %g %g %g %g\n", float(average), std, upper_threshold, lower_threshold);

		for (line = 0; line < number_of_lines; line++)
		{
			index = records_per_line * line;
			if (parameter_cache[7 + index] >= 0 || ! exclude_negative_film_numbers)
			{
				temp_float = parameter_cache[wanted_index + index];
				if (reciprocal_square)
				{
					if (temp_float > 0.0) temp_float = 1.0 / powf(temp_float, 2);
					else temp_float = average;
					if (temp_float > upper_threshold) temp_float = upper_threshold;
					if (temp_float < lower_threshold) temp_float = lower_threshold;
					temp_float = sqrtf(1.0 / temp_float);
				}
				else
				{
					if (temp_float > upper_threshold) temp_float = upper_threshold;
					if (temp_float < lower_threshold) temp_float = lower_threshold;
				}
				parameter_cache[wanted_index + index] = temp_float;
			}
		}
	}
}

float FrealignParameterFile::ReturnThreshold(float wanted_percentage, bool exclude_negative_film_numbers)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");
//	MyDebugAssertTrue(wanted_percentage < 1.0 && wanted_percentage >= 0.0, "Percentage out of range");

	int i;
	int line;
	int index;
	int number_of_bins = 10000;
	float average_occ = 0.0;
	float sum_occ;
	float increment;
	float threshold;
	float percentage;
	float min, max;

	min = ReturnMin(15, exclude_negative_film_numbers);
	max = ReturnMax(15, exclude_negative_film_numbers);
	average_occ = ReturnAverage(12, exclude_negative_film_numbers);
	increment = (min - max) / (number_of_bins - 1);
	if (increment == 0.0) return min;

	for (i = 0; i < number_of_bins; i++)
	{
		sum_occ = 0.0;
		threshold = float(i) * increment + max;
		for (line = 0; line < number_of_lines; line++)
		{
			index = records_per_line * line;
			if (parameter_cache[7 + index] >= 0 || ! exclude_negative_film_numbers)
			{
	 			if (parameter_cache[15 + index] >= threshold) sum_occ += parameter_cache[12 + index];
			}
		}
		percentage = sum_occ / number_of_lines / average_occ;

	//	wxPrintf("sum_occ = %f : threshold = %f\n", sum_occ, threshold);
		if (percentage >= wanted_percentage) break;
	}

	if (sum_occ == 0.0)
	{
		MyPrintWithDetails("Error: Number of particles selected = 0; please change score threshold\n");
		DEBUG_ABORT;
	}

	return threshold;
}

void FrealignParameterFile::CalculateDefocusDependence(bool exclude_negative_film_numbers)
{
	int line;
	int index;
	double s = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
	double delta;

	for (line = 0; line < number_of_lines; line++)
	{
		index = records_per_line * line;
		if (parameter_cache[7 + index] >= 0 || ! exclude_negative_film_numbers)
		{
			average_defocus = (parameter_cache[8 + index] + parameter_cache[9 + index]) / 2.0;
			s += parameter_cache[12 + index];
			sx += average_defocus * parameter_cache[12 + index];
			sy += parameter_cache[15 + index] * parameter_cache[12 + index];
			sxx += powf(average_defocus,2) * parameter_cache[12 + index];
			sxy += average_defocus * parameter_cache[15 + index] * parameter_cache[12 + index];
		}
	}
	average_defocus = sx / s;
	delta = s * sxx - powf(sx,2);
	defocus_coeff_a = (sxx * sy - sx * sxy) / delta;
	defocus_coeff_b = (s * sxy - sx * sy) / delta;
//	wxPrintf("average_defocus = %g, defocus_coeff_a = %g, defocus_coeff_b = %g\n", average_defocus, defocus_coeff_a, defocus_coeff_b);
}

void FrealignParameterFile::AdjustScores(bool exclude_negative_film_numbers)
{
	int line;
	int index;
	float defocus;

	for (line = 0; line < number_of_lines; line++)
	{
		index = records_per_line * line;
		if (parameter_cache[7 + index] >= 0 || ! exclude_negative_film_numbers)
		{
			defocus = (parameter_cache[8 + index] + parameter_cache[9 + index]) / 2.0;
			if (defocus != 0.0f) parameter_cache[15 + index] -= ReturnScoreAdjustment(defocus); // added 0 check for defocus sweep
		}
	}
}

float FrealignParameterFile::ReturnScoreAdjustment(float defocus)
{
	MyDebugAssertTrue(average_defocus != 0.0 || defocus_coeff_b != 0.0, "Defous coefficients not determined");

	return (defocus - average_defocus) * defocus_coeff_b;
}

void FrealignParameterFile::ReduceAngles()
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");

	int line;
	int index;
	float psi, theta, phi;

	for (line = 0; line < number_of_lines; line++)
	{
		index = records_per_line * line;
		psi = parameter_cache[1 + index];
		theta = parameter_cache[2 + index];
		phi = parameter_cache[3 + index];

		theta = fmodf(theta, 360.0);
		if (theta < 0.0) theta += 360.0;
		if (theta >= 180.0)
		{
			theta = 360.0 - theta;
			psi += 180.0;
			phi += 180.0;
		}
		psi = fmodf(psi, 360.0);
		if (psi < 0.0) psi += 360.0;
		phi = fmodf(phi, 360.0);
		if (phi < 0.0) phi += 360.0;
		parameter_cache[1 + index] = psi;
		parameter_cache[2 + index] = theta;
		parameter_cache[3 + index] = phi;
	}
}

float FrealignParameterFile::ReturnDistributionMax(int wanted_index, int selector, bool exclude_negative_film_numbers)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");
	MyDebugAssertTrue(wanted_index >= 0 && wanted_index < records_per_line, "Index out of range");

	int bin;
	int number_of_bins;
	int max_bin;
	int line;
	int index;
	int included_lines;
	float sum;
	float sum_max;
	float min;
	float max;
	float step;
	float temp_float;
	bool is_angle = false;

	if (selector > 0)
	{
		included_lines = 0;
		for (line = 0; line < number_of_lines; line++)
		{
			index = records_per_line * line;
			if (fabsf(parameter_cache[7 + index] - selector) < 0.01) included_lines++;
		}
	}
	else included_lines = number_of_lines;

	if (included_lines == 0) return 0.0;

	number_of_bins = myroundint(included_lines / 100.0);
	if (number_of_bins < 10) number_of_bins = std::min(included_lines, 10);
	if (wanted_index >= 1 && wanted_index <= 3) is_angle = true;
	if (is_angle)
	{
		min = 0.0;
		max = 360.0;
	}
	else
	{
		min = ReturnMin(wanted_index, exclude_negative_film_numbers);
		max = ReturnMax(wanted_index, exclude_negative_film_numbers);
	}

	if (min == max) return min;

	int *bins = new int [number_of_bins];
	float *bin_sum = new float [number_of_bins];
	ZeroIntArray(bins, number_of_bins);
	ZeroFloatArray(bin_sum, number_of_bins);
	step = (max - min) / (number_of_bins - 1);

	for (line = 0; line < number_of_lines; line++)
	{
		index = records_per_line * line;
		if (parameter_cache[7 + index] >= 0 || ! exclude_negative_film_numbers)
		{
			if (fabsf(parameter_cache[7 + index] - selector) < 0.01 || selector <= 0.0)
			{
				temp_float = parameter_cache[wanted_index + index];
				if (is_angle)
				{
					temp_float = fmodf(temp_float, 360.0);
					if (temp_float < 0.0) temp_float += 360.0;
				}
				bin = myroundint((temp_float - min) / step);
				if (bin < 0) bin = 0;
				if (bin >= number_of_bins) bin = number_of_bins - 1;
				bins[bin]++;
				bin_sum[bin] += temp_float;
			}
		}
	}

	max_bin = 0;
	for (bin = 0; bin < number_of_bins; bin++) if (bins[bin] > bins[max_bin]) max_bin = bin;
	temp_float = bin_sum[max_bin];
	max_bin = bins[max_bin];

	delete [] bins;
	delete [] bin_sum;

	return temp_float / max_bin;
}

float FrealignParameterFile::ReturnDistributionSigma(int wanted_index, float distribution_max, int selector, bool exclude_negative_film_numbers)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");
	MyDebugAssertTrue(wanted_index >= 0 && wanted_index < records_per_line, "Index out of range");

	int line;
	int index;
	int count = 0;
	float sum_of_squares = 0.0;
	float temp_float;
	bool is_angle = false;

	if (wanted_index >= 1 && wanted_index <= 3) is_angle = true;

	for (line = 0; line < number_of_lines; line++)
	{
		index = records_per_line * line;
		if (parameter_cache[7 + index] >= 0 || ! exclude_negative_film_numbers)
		{
			if (fabsf(parameter_cache[7 + index] - selector) < 0.01 || selector <= 0.0)
			{
				temp_float = parameter_cache[wanted_index + index] - distribution_max;
				if (is_angle)
				{
					temp_float = fmodf(temp_float, 360.0);
				}
				sum_of_squares += powf(temp_float, 2);
				count++;
			}
		}
	}

	if (count == 0) return 0.0;
	else return sqrtf(sum_of_squares / count);
}

void FrealignParameterFile::SetParameters(int wanted_index, float wanted_value, float wanted_sigma, int selector, bool exclude_negative_film_numbers)
{
	MyDebugAssertTrue(parameter_cache != NULL, "File has not been read into memory");
	MyDebugAssertTrue(wanted_index >= 0 && wanted_index < records_per_line, "Index out of range");

	int line;
	int index;

	for (line = 0; line < number_of_lines; line++)
	{
		index = records_per_line * line;
		if (parameter_cache[7 + index] >= 0 || ! exclude_negative_film_numbers)
		{
			if (fabsf(parameter_cache[7 + index] - selector) < 0.01 || selector <= 0.0)
			{
				parameter_cache[wanted_index + index] = wanted_value + wanted_sigma * global_random_number_generator.GetNormalRandom();
			}
		}
	}
}
