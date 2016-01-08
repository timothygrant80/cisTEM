#include "core_headers.h"

NumericTextFile::NumericTextFile()
{
	MyPrintWithDetails("NumericTextfile has been declared with no filename.\n");
	abort();
}

NumericTextFile::NumericTextFile(wxString Filename, long wanted_access_type, long wanted_records_per_line)
{
	strcpy(text_filename, Filename);
	strcat(text_filename, ".plt");
	access_type = wanted_access_type;

	if (access_type == OPEN_TO_WRITE)
	{
		records_per_line = wanted_records_per_line;
		if (records_per_line <= 0)
		{
			MyPrintWithDetails("NumericTextFile asked to OPEN_TO_WRITE, but with erroneous records per line\n");
			abort();
		}
	}

	Init();
}

NumericTextFile::~NumericTextFile()
{
	if (text_file != 0)
	{
		fclose(text_file);
	}
}

void NumericTextFile::Open(wxString Filename, long wanted_access_type)
{
	if (text_file != 0)
	{
		MyPrintWithDetails("File already Open\n");
		abort();
	}

	strcpy(text_filename, Filename);
	strcat(text_filename, ".plt");
	access_type = wanted_access_type;

	Init();
}

void NumericTextFile::Close()
{
	if (text_file != 0)
	{
		fclose(text_file);
	}
}

void NumericTextFile::Init()
{

	if (access_type == OPEN_TO_READ)
	{
		text_file = fopen(text_filename, "rb");
		if (text_file == 0)
		{
			MyPrintWithDetails("Attempt to access %s for reading failed\n",text_filename);
			abort();
		}

		// work out the records per line and how many lines

		float temp_float;
		long pos_check;
		long total_number_of_records = 0;

		int c = 0;

		records_per_line = 0;

		// this now needs to be quite complicated in case there are extra spaces at the end of a line..


		while(c != '\n' && c != '\r' && c != 10 && c != 13)
		{
			pos_check = 0;
			do
			{
				pos_check++;
			} while ((c = fgetc(text_file)) == ' ');

			if (c == '\n' || c == '\r' || c == 10 || c == 13) break;

			fseek(text_file, pos_check * -1, SEEK_CUR);

			pos_check = fscanf(text_file, "%f", &temp_float);
			if (pos_check == 0)
			{
				if (feof(text_file)) break;
				MyPrintWithDetails("File seems to contain non-numbers and therefore cannot be read\n");
				abort();
			}
			if (pos_check != EOF) records_per_line++;
			else break;
		}

		rewind(text_file);

		// this is a bit dodgy, i am going to assume that each line has the same number of records to make things easy here..

		while (fscanf (text_file, "%f", &temp_float) != EOF)
		{
			total_number_of_records++;
		}


		// divide total number by number per line to get number of lines..

		number_of_lines = total_number_of_records / records_per_line;

		rewind(text_file);

	}
	else
		if (access_type == OPEN_TO_WRITE)
		{
			text_file = fopen(text_filename, "w");
			if (text_file == 0)
			{
				MyPrintWithDetails("Attempt to access %s for writing, failed.\n",text_filename);
				abort();
			}
		}
		else
			if (access_type == OPEN_TO_APPEND)
			{
				text_file = fopen(text_filename, "a");
				if (text_file == 0)
				{
					MyPrintWithDetails("Attempt to access %s for appending, failed.\n",text_filename);
					abort();
				}
			}


}

void NumericTextFile::Rewind()
{
	if (text_file != 0)
	{
		rewind(text_file);
	}

}

void NumericTextFile::Flush()
{
	if (text_file !=0)
	{
		fflush(text_file);
	}
}

long NumericTextFile::ReadLine(double *data_array)
{

	float temp_float;
	long pos_check = 0;


	if (access_type != OPEN_TO_READ)
	{
		MyPrintWithDetails("Attempt to read %s however access type is not READ\n",text_filename);
		abort();
	}

	if (text_file == 0)
	{
		MyPrintWithDetails("Attempt to read %s however file is not open\n",text_filename);
		abort();
	}

	for (long counter = 1; counter <= records_per_line; counter++)
	{
		pos_check = fscanf (text_file, "%f", &temp_float);
		if (pos_check == EOF) return pos_check;

		data_array[counter - 1] = double(temp_float);
	}

	return pos_check;

}

void NumericTextFile::WriteLine(double *data_array)
{

	float temp_float;

	if (access_type == OPEN_TO_READ)
	{
		MyPrintWithDetails("Attempt to write to %s however access type is READ\n",text_filename);
		abort();
	}

	if (text_file == 0)
	{
		MyPrintWithDetails("Attempt to read %s however file is not open\n");
		abort();
	}

	for (long counter = 1; counter <= records_per_line; counter++)
	{
		temp_float = float(data_array[counter - 1]);
		fprintf (text_file, "%f ", temp_float);
	}

	fprintf(text_file, "\n");
}

void NumericTextFile::WriteCommentLine(const char * format, ...)
{
	va_list args;
	va_start(args, format);

	vfprintf(text_file, format, args);

	va_end(args);
}

std::string NumericTextFile::ReturnFilename()
{
	return text_filename;
}





