#include "core_headers.h"

NumericTextFile::NumericTextFile()
{
	MyPrintWithDetails("NumericTextfile has been declared with no filename.\n");
	DEBUG_ABORT;
}

NumericTextFile::NumericTextFile(wxString Filename, long wanted_access_type, long wanted_records_per_line)
{
	input_file_stream = NULL;
	input_text_stream = NULL;
	output_file_stream = NULL;
	output_text_stream = NULL;

	do_nothing = false;

	Open(Filename, wanted_access_type, wanted_records_per_line);
}

NumericTextFile::~NumericTextFile()
{
	Close();
}

void NumericTextFile::Open(wxString Filename, long wanted_access_type, long wanted_records_per_line)
{
	access_type = wanted_access_type;
	records_per_line = wanted_records_per_line;
	text_filename = Filename;

	do_nothing = text_filename.IsSameAs("/dev/null");

	if (!do_nothing)
	{
		if (access_type == OPEN_TO_READ)
		{
			if (input_file_stream != NULL)
			{
				if (input_file_stream->GetFile()->IsOpened() == true)
				{
					MyPrintWithDetails("File already Open\n");
					DEBUG_ABORT;
				}

			}
		}
		else
		if (access_type == OPEN_TO_WRITE)
		{
			records_per_line = wanted_records_per_line;

			if (records_per_line <= 0)
			{
				MyPrintWithDetails("NumericTextFile asked to OPEN_TO_WRITE, but with erroneous records per line\n");
				DEBUG_ABORT;
			}

			if (output_file_stream != NULL)
			{
				if (output_file_stream->GetFile()->IsOpened() == true)
				{
					MyPrintWithDetails("File already Open\n");
					DEBUG_ABORT;
				}

			}


		}
		else
		{
			MyPrintWithDetails("Unknown access type!\n");
			DEBUG_ABORT;
		}


		Init();
	}
}

void NumericTextFile::Close()
{
	if (input_text_stream != NULL) delete input_text_stream;
	if (output_text_stream != NULL) delete output_text_stream;

	if (output_file_stream != NULL)
	{
		if (output_file_stream->GetFile()->IsOpened() == true) output_file_stream->GetFile()->Close();
		delete output_file_stream;
	}

	if (input_file_stream != NULL)
	{
		if (input_file_stream->GetFile()->IsOpened() == true) input_file_stream->GetFile()->Close();
		delete input_file_stream;
	}

	input_file_stream = NULL;
	input_text_stream = NULL;
	output_file_stream = NULL;
	output_text_stream = NULL;
}

void NumericTextFile::Init()
{

	if (!do_nothing)
	{
		if (access_type == OPEN_TO_READ)
		{
			wxString current_line;
			wxString token;
			double temp_double;
			int current_records_per_line;
			int old_records_per_line = -1;

			input_file_stream = new wxFileInputStream(text_filename);
			input_text_stream = new wxTextInputStream(*input_file_stream);

			if (input_file_stream->IsOk() == false)
			{
				MyPrintWithDetails("Attempt to access %s for reading failed\n",text_filename);
				DEBUG_ABORT;
			}

			// work out the records per line and how many lines

			number_of_lines = 0;


			while (input_file_stream->Eof() == false)
			{
				current_line = input_text_stream->ReadLine();
				current_line.Trim(false);

				if (current_line.StartsWith("#") != true && current_line.StartsWith("C") != true && current_line.Length() > 0)
				{
					number_of_lines++;
					wxStringTokenizer tokenizer(current_line);

					current_records_per_line = 0;

					while ( tokenizer.HasMoreTokens() )
					{
						token = tokenizer.GetNextToken();

						if (token.ToDouble(&temp_double) == false)
						{
							MyPrintWithDetails("Failed on the following record : %s\n", token);
							DEBUG_ABORT;
						}
						else
						{
							current_records_per_line++;
						}
					}

					// we want to check records_per_line for consistency..

					if (old_records_per_line != -1)
					{
						if (old_records_per_line != current_records_per_line)
						{
							MyPrintWithDetails("Different records per line found");
							DEBUG_ABORT;
						}
					}

					old_records_per_line = current_records_per_line;


				}
			}

			records_per_line = current_records_per_line;

			// rewind the file..

			Rewind();

		}
		else
		if (access_type == OPEN_TO_WRITE)
		{
			// check if the file exists..

			if (DoesFileExist(text_filename) == true)
			{
				if (wxRemoveFile(text_filename) == false)
				{
					MyDebugPrintWithDetails("Cannot remove already existing text file");
				}
			}

			output_file_stream = new wxFileOutputStream(text_filename);
			output_text_stream = new wxTextOutputStream(*output_file_stream);
		}
	}
}

void NumericTextFile::Rewind()
{
	if (!do_nothing)
	{
		if (access_type == OPEN_TO_READ)
		{
			delete input_file_stream;
			delete input_text_stream;

			input_file_stream = new wxFileInputStream(text_filename);
			input_text_stream = new wxTextInputStream(*input_file_stream);

		}
		else
		output_file_stream->GetFile()->Seek(0);
	}

}

void NumericTextFile::Flush()
{
	if (!do_nothing)
	{
		if (access_type == OPEN_TO_READ) input_file_stream->GetFile()->Flush();
		else
		output_file_stream->GetFile()->Flush();
	}
}

void NumericTextFile::ReadLine(float *data_array)
{
	if (!do_nothing)
	{
		if (access_type != OPEN_TO_READ)
		{
			MyPrintWithDetails("Attempt to read from %s however access type is not READ\n",text_filename);
			DEBUG_ABORT;
		}

		wxString current_line;
		wxString token;
		double temp_double;

		while(input_file_stream->Eof() == false)
		{
			current_line = input_text_stream->ReadLine();
			current_line.Trim(false);

			if (current_line.StartsWith("C") == false && current_line.StartsWith("#") == false && current_line.Length() != 0) break;
		}

		wxStringTokenizer tokenizer(current_line);

		for (int counter = 0; counter < records_per_line; counter++ )
		{
			token = tokenizer.GetNextToken();

			if (token.ToDouble(&temp_double) == false)
			{
				MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", token.ToUTF8().data(), current_line.ToUTF8().data());
				DEBUG_ABORT;
			}
			else
			{
				data_array[counter] = temp_double;

			}
		}
	}
}

void NumericTextFile::WriteLine(float *data_array)
{
	if (!do_nothing)
	{
		if (access_type != OPEN_TO_WRITE)
		{
			MyPrintWithDetails("Attempt to read from %s however access type is not WRITE\n",text_filename);
			DEBUG_ABORT;
		}

		for (int counter = 0; counter < records_per_line; counter++ )
		{
	//		output_text_stream->WriteDouble(data_array[counter]);
			output_text_stream->WriteString(wxString::Format("%14.5f",data_array[counter]));
			if (counter != records_per_line - 1) output_text_stream->WriteString(" ");
		}

		output_text_stream->WriteString("\n");
	}
}

void NumericTextFile::WriteLine(double *data_array)
{
	if (!do_nothing)
	{
		if (access_type != OPEN_TO_WRITE)
		{
			MyPrintWithDetails("Attempt to read from %s however access type is not WRITE\n",text_filename);
			DEBUG_ABORT;
		}

		for (int counter = 0; counter < records_per_line; counter++ )
		{
			output_text_stream->WriteDouble(data_array[counter]);
			if (counter != records_per_line - 1) output_text_stream->WriteString(" ");
		}

		output_text_stream->WriteString("\n");
	}
}

void NumericTextFile::WriteCommentLine(const char * format, ...)
{
	if (!do_nothing)
	{
		va_list args;
		va_start(args, format);

		wxString comment_string;
		wxString buffer;


		comment_string.PrintfV(format, args);

		buffer = comment_string;
		buffer.Trim(false);

		if (buffer.StartsWith("#") == false && buffer.StartsWith("C") == false)
		{
			comment_string = "# " + comment_string;
		}

		output_text_stream->WriteString(comment_string);

		if (comment_string.EndsWith("\n") == false) output_text_stream->WriteString("\n");

		va_end(args);
	}
}

wxString NumericTextFile::ReturnFilename()
{
	return text_filename;
}





