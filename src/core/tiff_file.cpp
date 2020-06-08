#include "core_headers.h"
TiffFile::TiffFile()
{
	tif = NULL;
	logical_dimension_x = 0;
	logical_dimension_y = 0;
	number_of_images = 0;
	this_is_in_mastronarde_4bit_hack_format = false;
}

TiffFile::TiffFile(std::string wanted_filename, bool overwrite)
{
	OpenFile(wanted_filename, overwrite);
}

TiffFile::~TiffFile()
{
	CloseFile();
}


/*
 * By default, this method will check all the images within the file, to make sure
 * they are valid (not corrupted) and of the same dimensions. However, this can take
 * a long time. If you want to save that time, set check_only_the_first_image to true.
 * This is risky because we may not notice that the file is corrupt or has unusual dimensions.
 */
bool TiffFile::OpenFile(std::string wanted_filename, bool overwrite, bool wait_for_file_to_exist, bool check_only_the_first_image)
{

	MyDebugAssertFalse(tif != NULL,"File already open: %s",wanted_filename);
	MyDebugAssertFalse(wait_for_file_to_exist,"Waiting for file to exist not implemented for tif files");

	bool file_already_exists = DoesFileExist(wanted_filename);

	bool return_value = true;

	filename = wanted_filename;

	// if overwrite is specified, then we delete the file nomatter what..
	if (overwrite) file_already_exists = false;

	if (file_already_exists)
	{
		// We open to read/write
		// The first dictionary is automatically read in
		tif = TIFFOpen(wanted_filename.c_str(),"rc");
		if (tif)
		{
			return_value = ReadLogicalDimensionsFromDisk(check_only_the_first_image);
		}
		else
		{
			MyPrintfRed("Oops. File %s could not be opened, may be corrupted\n",wanted_filename);
			return_value = false;
		}
	}
	else
	{
		// We just open to write
		tif = TIFFOpen(wanted_filename.c_str(),"w");
		if (tif)
		{
			return_value = true;
		}
		else
		{
			MyPrintfRed("Oops. File %s could not be opened for writing\n",wanted_filename);
			return_value = false;
		}
	}
	pixel_size = 1.0; //TODO: work out out to grab the pixel size from SerialEM/IMOD-style TIFF files
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

/*
 * By default, this method will check all the images within the file, to make sure
 * they are valid (not corrupted) and of the same dimensions. However, this can take
 * a long time. If you want to save that time, set check_only_the_first_image to true.
 * This is risky because we may not notice that the file is corrupt or has unusual dimensions.
 */
bool TiffFile::ReadLogicalDimensionsFromDisk(bool check_only_the_first_image)
{
	MyDebugAssertTrue(tif != NULL,"File must be open");

	/*
	 * Since the file was already open, TiffOpen has already read in the first dictionary
	 * and it must be valid, else we would have returned an error at open-time already
	 */
	int set_dir_ret = TIFFSetDirectory(tif,0);
	bool return_value = (set_dir_ret == 1);
	int dircount = 1;

	uint32 original_x = 0;
	uint32 current_x = 0;
	uint32 current_y = 0;

	TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&current_x);
	TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&current_y);

	// Serial EM has the option to write out 4-bit compressed tif.  They way that this is done is to do it as an 8-bit with
	// half the x dimension. If the file has specific dimensions, it is assumed to be 4-bit.  I've copied the check directly
	// from David Mastronarde. - see function comments for sizeCanBe4BitK2SuperRes

	if (sizeCanBe4BitK2SuperRes(current_x, current_y) == 1)
	{
		logical_dimension_x = current_x * 2;
		logical_dimension_y = current_y;
		this_is_in_mastronarde_4bit_hack_format = true;
	}
	else
	{
		logical_dimension_x = current_x;
		logical_dimension_y = current_y;
		this_is_in_mastronarde_4bit_hack_format = false;
	}

	original_x = current_x;

	if (!check_only_the_first_image)
	{
		const bool check_dimensions_of_every_image = true;
		if (check_dimensions_of_every_image)
		{
			// Loop through all the TIFF directories and check they all have the same x,y dimensions
			// This takes ages when importing TIFFs
			while (!TIFFLastDirectory(tif))
			{
				// The current directory is not the last one, so let's go to the next
				dircount++;
				set_dir_ret = TIFFSetDirectory(tif,dircount-1);

				if (set_dir_ret != 1)
				{
					MyPrintfRed("Warning: Image %i of file %s seems to be corrupted\n",dircount,filename.GetFullName());
					return_value = false;
					dircount--;
					break;
				}
				else
				{
					TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&current_x);
					TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&current_y);

					if (current_x != original_x || logical_dimension_y != current_y)
					{
						MyPrintfRed("Warning: Image %i of file %s has dimensions %i,%i, whereas previous images had dimensions %i,%i\n",dircount,filename.GetFullName(),current_x,current_y,original_x,logical_dimension_y);
						return_value = false;
						dircount--;
						break;
					}
				}
			}
			// we return the number of valid images
			number_of_images = dircount;
		}
		else
		{
			// This doesnt' seem to be significantly faster than the other way to do it.
			number_of_images = TIFFNumberOfDirectories(tif);
		}
	}
	else
	{
		// We only checked the first image, so we don't know how many images are in the file
		number_of_images = -1;
	}

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

				// Serial EM has the option to write out 4-bit compressed tif.  They way that this is done is to do it as an 8-bit with
				// half the x dimension. If the file has specific dimensions, it is assumed to be 4-bit.  I've copied the check directly
				// from David Mastronarde. - see function comments for sizeCanBe4BitK2SuperRes

				if (this_is_in_mastronarde_4bit_hack_format == true) // this is 4-bit
				{

					uint8 hi_4bits;
					uint8 low_4bits;

					strip_counter = 0;

					number_of_bytes_placed_in_buffer = TIFFReadEncodedStrip(tif, strip_counter, (char *) buf, (tsize_t) -1);
					//wxPrintf("%i %i %i\n",int(number_of_bytes_placed_in_buffer),int(rows_per_strip), int(rows_per_strip * ReturnXSize()));
					if (strip_counter < TIFFNumberOfStrips(tif) - 1) MyDebugAssertTrue(number_of_bytes_placed_in_buffer == rows_per_strip * ReturnXSize(),"Unexpected number of bytes in uint8 buffer");


					for (strip_counter = 0; strip_counter < TIFFNumberOfStrips(tif); strip_counter++)
					{

						number_of_bytes_placed_in_buffer = TIFFReadEncodedStrip(tif, strip_counter, (char *) buf, (tsize_t) -1);
						if (strip_counter < TIFFNumberOfStrips(tif) - 1) MyDebugAssertTrue(number_of_bytes_placed_in_buffer == rows_per_strip * ReturnXSize() / 2,"Unexpected number of bytes in uint8 buffer");

						output_counter = strip_counter * rows_per_strip * ReturnXSize() + ((directory_counter - start_slice + 1) * ReturnXSize() * ReturnYSize());

						for (long counter = 0; counter < number_of_bytes_placed_in_buffer; counter ++)
						{
							low_4bits = buf[counter] & 0x0F;
							hi_4bits = (buf[counter]>>4) & 0x0F;

							output_array[output_counter] = float(low_4bits);
							output_counter ++;
							output_array[output_counter] = float(hi_4bits);
							output_counter ++;
						}
					}
				}
				else // not 4-bit
				{
					for (strip_counter = 0; strip_counter < TIFFNumberOfStrips(tif); strip_counter++)
					{

						number_of_bytes_placed_in_buffer = TIFFReadEncodedStrip(tif, strip_counter, (char *) buf, (tsize_t) -1);
						if (strip_counter < TIFFNumberOfStrips(tif) - 1) MyDebugAssertTrue(number_of_bytes_placed_in_buffer == rows_per_strip * ReturnXSize(),"Unexpected number of bytes in uint8 buffer");

						output_counter = strip_counter * rows_per_strip * ReturnXSize() + ((directory_counter - start_slice + 1) * ReturnXSize() * ReturnYSize());
						for (long counter = 0; counter < number_of_bytes_placed_in_buffer; counter ++)
						{
							output_array[output_counter] = buf[counter];
							output_counter ++;
						}
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
					MyDebugAssertTrue(number_of_bytes_placed_in_buffer == rows_per_strip * ReturnXSize() * 2,"Unexpected number of bytes in uint16 buffer");

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
				MyPrintfRed("Error. Unsupported uint bit depth: %i. Filename = %s, Directory # %i\n",bits_per_sample,filename.GetFullPath(),directory_counter);
				break;
			}
			break;
		case SAMPLEFORMAT_INT:
			switch (bits_per_sample)
			{
			case 16:
			{
				int16 * buf = new int16[TIFFStripSize(tif)/2];

				for (strip_counter = 0; strip_counter < TIFFNumberOfStrips(tif); strip_counter++)
				{
					number_of_bytes_placed_in_buffer = TIFFReadEncodedStrip(tif, strip_counter, (char *) buf, (tsize_t) -1);
					MyDebugAssertTrue(number_of_bytes_placed_in_buffer == rows_per_strip * ReturnXSize() * 2,"Unexpected number of bytes in uint16 buffer");

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
				MyPrintfRed("Error. Unsupported int bit depth: %i. Filename = %s, Directory # %i\n",bits_per_sample,filename.GetFullPath(),directory_counter);
				break;
			}
			break;
		case SAMPLEFORMAT_IEEEFP:
		{
			switch (bits_per_sample)
			{
			case 32:
			{
				float * buf = new float[TIFFStripSize(tif)/4];

				for (strip_counter = 0; strip_counter < TIFFNumberOfStrips(tif); strip_counter++)
				{
					number_of_bytes_placed_in_buffer = TIFFReadEncodedStrip(tif, strip_counter, (char *) buf, (tsize_t) -1);
					MyDebugAssertTrue(number_of_bytes_placed_in_buffer == rows_per_strip * ReturnXSize() * 4,"Unexpected number of bytes in float buffer");

					output_counter = strip_counter * rows_per_strip * ReturnXSize() + ((directory_counter - start_slice + 1) * ReturnXSize() * ReturnYSize());
					for (long counter = 0; counter < number_of_bytes_placed_in_buffer/4; counter ++)
					{
						output_array[output_counter] = buf[counter];
						output_counter ++;
					}
				}
				delete [] buf;
			}
				break;
			default:
				MyPrintfRed("Error. Unsupported float bit depth: %i. Filename = %s, Directory # %i\n",bits_per_sample,filename.GetFullPath(),directory_counter);
				break;
			}
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


