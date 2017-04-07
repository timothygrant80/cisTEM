#include "core_headers.h"
TiffFile::TiffFile()
{
	tif = NULL;
	logical_dimension_x = 0;
	logical_dimension_y = 0;
	number_of_images = 0;
}

TiffFile::TiffFile(std::string wanted_filename, bool overwrite)
{
	OpenFile(wanted_filename, overwrite);
}

TiffFile::~TiffFile()
{
	CloseFile();
}

bool TiffFile::OpenFile(std::string wanted_filename, bool overwrite)
{

	MyDebugAssertFalse(tif != NULL,"File already open: %s",wanted_filename)

	bool file_already_exists = DoesFileExist(wanted_filename);

	bool return_value = true;

	// if overwrite is specified, then we delete the file nomatter what..
	if (overwrite) file_already_exists = false;

	if (file_already_exists)
	{
		// We open to read/write
		tif = TIFFOpen(wanted_filename.c_str(),"rc");
		return_value = ReadLogicalDimensionsFromDisk();
	}
	else
	{
		// We just open to write
		tif = TIFFOpen(wanted_filename.c_str(),"w");
	}

	filename = wanted_filename;
	return return_value;
}

void TiffFile::CloseFile()
{
	if (tif != NULL) TIFFClose(tif);
	tif = NULL;
}

void TiffFile::PrintInfo()
{
	wxPrintf("Filename  : %s\n",filename.GetFullName());
	wxPrintf("Dimensions: %i %i %i\n",ReturnXSize(),ReturnYSize(),ReturnZSize());
}

bool TiffFile::ReadLogicalDimensionsFromDisk()
{
	MyDebugAssertTrue(tif != NULL,"File must be open");
	// Loop through all the TIFF directories and check they all have the same x,y dimensions
	int set_dir_ret = TIFFSetDirectory(tif,0);
	bool return_value = (set_dir_ret == 1);
	int dircount = 1;
	uint32 current_x = 0;
	uint32 current_y = 0;

	TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&current_x);
	TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&current_y);

	logical_dimension_x = current_x;
	logical_dimension_y = current_y;

	while (TIFFReadDirectory(tif))
	{
		dircount++;
		TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&current_x);
		TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&current_y);

		if (logical_dimension_x != current_x || logical_dimension_y != current_y)
		{
			MyPrintfRed("Oops. Image %i of file %s has dimensions %i,%i, whereas previous images had dimensions %i,%i\n",dircount,current_x,current_y,logical_dimension_x,logical_dimension_y);
			return_value = false;
		}
	}

	number_of_images = dircount;

	return return_value;
}


void TiffFile::ReadSliceFromDisk(int slice_number, float * output_array)
{
	ReadSlicesFromDisk(slice_number, slice_number, output_array);
}


void TiffFile::ReadSlicesFromDisk(int start_slice, int end_slice, float *output_array)
{
	MyDebugAssertTrue(tif != NULL,"File must be open");
	MyDebugAssertTrue(start_slice > 0 && end_slice >= start_slice && end_slice <= number_of_images,"Bad start or end slice number");


	unsigned int bits_per_sample = 0;
	unsigned int samples_per_pixel = 0;
	unsigned int sample_format = 0;
	unsigned int rows_per_strip = 0;

	long output_counter;

	tstrip_t strip_counter;
	tmsize_t number_of_bytes_placed_in_buffer;

	for (unsigned int directory_counter = start_slice-1; directory_counter < end_slice; directory_counter ++ )
	{

		TIFFSetDirectory(tif,directory_counter);

		// We don't support tiles, only strips
		if (TIFFIsTiled(tif)) { MyPrintfRed("Error. Cannot read tiled TIF files. Filename = %s, Directory # %i. Number of tiles per image = %i\n",filename.GetFullPath(),directory_counter,TIFFNumberOfTiles(tif)); }

		// Get bit depth etc
		TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
		TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
		if (samples_per_pixel != 1) { MyPrintfRed("Error. Unsupported number of samples per pixel: %i. Filename = %s, Directory # %i\n",samples_per_pixel,filename.GetFullPath(),directory_counter); }
		TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sample_format);

		// How many rows per strip?
		TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);

		// Copy & cast data from these rows into the output array
		switch (sample_format)
		{
		case SAMPLEFORMAT_UINT:
			switch (bits_per_sample)
			{
			case 8:
			{
				uint8 * buf = new uint8[TIFFStripSize(tif)];

				for (strip_counter = 0; strip_counter < TIFFNumberOfStrips(tif); strip_counter++)
				{
					number_of_bytes_placed_in_buffer = TIFFReadEncodedStrip(tif, strip_counter, (char *) buf, (tsize_t) -1);
					MyDebugAssertTrue(number_of_bytes_placed_in_buffer == rows_per_strip * ReturnXSize(),"Unexpected number of bytes in buffer");

					output_counter = strip_counter * rows_per_strip * ReturnXSize() + ((directory_counter - start_slice + 1) * ReturnXSize() * ReturnYSize());
					for (long counter = 0; counter < number_of_bytes_placed_in_buffer; counter ++)
					{
						output_array[output_counter] = buf[counter];
						output_counter ++;
					}
				}

				delete [] buf;
			}
				break;
			case 16:
			{
				uint16 * buf = new uint16[TIFFStripSize(tif)/2];

				for (strip_counter = 0; strip_counter < TIFFNumberOfStrips(tif); strip_counter++)
				{
					number_of_bytes_placed_in_buffer = TIFFReadEncodedStrip(tif, strip_counter, (char *) buf, (tsize_t) -1);
					MyDebugAssertTrue(number_of_bytes_placed_in_buffer == rows_per_strip * ReturnXSize() * 2,"Unexpected number of bytes in buffer");

					output_counter = strip_counter * rows_per_strip * ReturnXSize() + ((directory_counter - start_slice + 1) * ReturnXSize() * ReturnYSize());
					for (long counter = 0; counter < number_of_bytes_placed_in_buffer/2; counter ++)
					{
						output_array[output_counter] = buf[counter];
						output_counter++;
					}
				}
				delete [] buf;
			}
				break;
			default:
				MyPrintfRed("Error. Unsupported bit depth: %i. Filename = %s, Directory # %i\n",bits_per_sample,filename.GetFullPath(),directory_counter);
				break;
			}
			break;
		default:
			MyPrintfRed("Error. Unsupported sample format: %i. Filename = %s, Directory # %i\n",sample_format,filename.GetFullPath(),directory_counter);
			break;
		}

		// Annoyingly, we need to swap the order of the lines to be "compatible" with MRC files etc
		{
			float * temp_line = new float[ReturnXSize()];
			long address_of_start_of_slice = (directory_counter - start_slice + 1) * ReturnXSize() * ReturnYSize();
			long address_of_start_of_line = address_of_start_of_slice;
			long address_of_start_of_other_line = address_of_start_of_slice + (ReturnYSize()-1)*ReturnXSize();
			long counter;

			for (long line_counter = 0; line_counter < ReturnYSize()/2; line_counter++)
			{
				// Copy current line to a buffer
				for (counter = 0; counter < ReturnXSize(); counter ++ )
				{
					temp_line[counter] = output_array[counter+address_of_start_of_line];
				}
				// Copy other line to current line
				for (counter = 0; counter < ReturnXSize(); counter ++ )
				{
					output_array[counter+address_of_start_of_line] = output_array[counter+address_of_start_of_other_line];
				}
				// Copy line from buffer to other line
				for (counter = 0; counter < ReturnXSize(); counter ++ )
				{
					output_array[counter+address_of_start_of_other_line] = temp_line[counter];
				}
				// Get ready for next iteration
				address_of_start_of_line += ReturnXSize();
				address_of_start_of_other_line -= ReturnXSize();
			}

			delete [] temp_line;
		}

	} // end of loop over slices

}

void TiffFile::WriteSliceToDisk(int slice_number, float * input_array)
{
	WriteSlicesToDisk(slice_number, slice_number, input_array);
}

void TiffFile::WriteSlicesToDisk(int start_slice, int end_slice, float * input_array)
{
	MyDebugAssertTrue(false,"Not implemented yet");
}
