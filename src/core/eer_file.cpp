#include "core_headers.h"
EerFile::EerFile()
{
	tif = NULL;
	logical_dimension_x = 0;
	logical_dimension_y = 0;
	number_of_frames = 1;
}

EerFile::EerFile(std::string wanted_filename, bool overwrite)
{
	OpenFile(wanted_filename, overwrite);
}

EerFile::~EerFile()
{
	CloseFile();
	delete [] frame_starts;
	delete [] frame_sizes;
	delete [] buf;
}


/*
 * By default, this method will check all the images within the file, to make sure
 * they are valid (not corrupted) and of the same dimensions. However, this can take
 * a long time. If you want to save that time, set check_only_the_first_image to true.
 * This is risky because we may not notice that the file is corrupt or has unusual dimensions.
 */
bool EerFile::OpenFile(std::string wanted_filename, bool overwrite, bool wait_for_file_to_exist, bool check_only_the_first_image)
{

	MyDebugAssertFalse(tif != NULL,"File already open: %s",wanted_filename);
	MyDebugAssertFalse(wait_for_file_to_exist,"Waiting for file to exist not implemented for tif files");

	bool file_already_exists = DoesFileExist(wanted_filename);

	bool return_value = true;


	// if overwrite is specified, then we delete the file nomatter what..
	if (overwrite) file_already_exists = false;

	if (file_already_exists)
	{
		// We open to read/write
		// The first dictionary is automatically read in
		tif = TIFFOpen(wanted_filename.c_str(),"rc");
		fh = fopen(wanted_filename.c_str(), "r");
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
	return return_value;
}

void EerFile::CloseFile()
{
	if (tif != NULL) TIFFClose(tif);
	tif = NULL;
}

void EerFile::PrintInfo()
{
	wxPrintf("Dimensions: %i %i %i\n",ReturnXSize(),ReturnYSize(),ReturnZSize());
}

/*
 * By default, this method will check all the images within the file, to make sure
 * they are valid (not corrupted) and of the same dimensions. However, this can take
 * a long time. If you want to save that time, set check_only_the_first_image to true.
 * This is risky because we may not notice that the file is corrupt or has unusual dimensions.
 */
void EerFile::ReadLogicalDimensionsFromDisk()
{
	MyDebugAssertTrue(tif != NULL,"File must be open");
	MyDebugAssertTrue(fh != NULL,"File must be open");
	/*
	 * Since the file was already open, EerOpen has already read in the first dictionary
	 * and it must be valid, else we would have returned an error at open-time already
	 */
	int set_dir_ret = TIFFSetDirectory(tif,0);
	bool return_value = (set_dir_ret == 1);
	int dircount = 1;

	uint32 original_x = 0;
	uint16 compression;

	TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&logical_dimension_x);
	TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&logical_dimension_y);
	frame_size_bits = logical_dimension_x * logical_dimension_y;
	TIFFGetField(tif,TIFFTAG_COMPRESSION,&compression);
	if (compression == 65000) bits_per_rle = 8;
	else if (compression == 65001) bits_per_rle = 7;
	else
	{
		MyPrintWithDetails("Warning:: Unknown Compression in EER tif file, assuming 7-bit (65001)");
		bits_per_rle = 7;
	}
	while (TIFFSetDirectory(tif, number_of_frames) != 0) number_of_frames++;
	fseek(fh, 0, SEEK_END);
	file_size = ftell(fh);
	fseek(fh, 0, SEEK_SET);

	wxPrintf("x_size = %d, y_size = %d, bits = %d, number_of_frames = %d, file_size = %li\n", logical_dimension_x, logical_dimension_y, bits_per_rle, number_of_frames, file_size);
}


void EerFile::ReadSlicesFromDisk()
{
	ReadLogicalDimensionsFromDisk();
	MyDebugAssertTrue(tif != NULL,"File must be open");


	frame_starts = new unsigned long long[number_of_frames];
	frame_sizes = new unsigned long long[number_of_frames];
	buf = new unsigned char[file_size];
	long output_counter;
	unsigned long long pos = 0;

	// Read everything
	for (int frame = 0; frame < number_of_frames; frame++)
	{
		TIFFSetDirectory(tif, frame);
		const int nstrips = TIFFNumberOfStrips(tif);
		frame_starts[frame] = pos;
		frame_sizes[frame] = 0;

		for (int strip = 0; strip < nstrips; strip++)
		{
			unsigned long long strip_size = TIFFRawStripSize(tif, strip);

			TIFFReadRawStrip(tif, strip, buf + pos, strip_size);
			pos += strip_size;
			frame_sizes[frame] += strip_size;
		}
	} // end of loop over slices
}

void EerFile::rleFrames(std::string output_file, int super_res_factor, int temporal_frame_bin_factor, wxString *output_sum_filename)
{
	ReadSlicesFromDisk();
	float temp_float[3];
	Image unaligned_sum;
	Image current_frame_image;

	MRCFile frame_output_file(output_file, true);

	long current_address;

	int frame_binning_counter = 0;
	int output_slice_number = 1;

	if (output_sum_filename != NULL)
	{
		unaligned_sum.Allocate(logical_dimension_x * super_res_factor, logical_dimension_x * super_res_factor,1);
		unaligned_sum.SetToConstant(0.0f);
	}

	current_frame_image.Allocate(logical_dimension_x * super_res_factor, logical_dimension_x * super_res_factor,1);
	current_frame_image.SetToConstant(0.0f);


	for (int iframe = 0; iframe < number_of_frames; iframe++)
	{

		long long pos = frame_starts[iframe];
		unsigned int npixels = 0, nelectrons = 0;
		const int max_electrons = frame_sizes[iframe] * 2; // at 4 bits per electron (very permissive bound!)
		unsigned int * positions = new unsigned int[max_electrons];
		unsigned char * symbols = new unsigned char[max_electrons];

		unsigned int bit_pos = 0; // 4 K * 4 K * 11 bit << 2 ** 32
		unsigned char rle, subpixel;
		long long first_byte;
		unsigned int bit_offset_in_first_byte;
		unsigned int chunk;
		if (bits_per_rle == 7)
		{
			while(true)
			{
				first_byte = pos + (bit_pos >> 3);
				bit_offset_in_first_byte = bit_pos & 7; // 7 = 00000111 (same as % 8)
				chunk = *(unsigned int*)(buf + first_byte);
				rle = (unsigned char)((chunk >> bit_offset_in_first_byte) & 127); // 127 = 01111111
				bit_pos += 7;
				npixels += rle;
				if (npixels == frame_size_bits) break;
				if (rle == 127) continue; // this should be rare.

				first_byte = pos + (bit_pos >> 3);
				bit_offset_in_first_byte = bit_pos & 7;
				chunk = *(unsigned int*)(buf + first_byte);
				subpixel = (unsigned char)((chunk >> bit_offset_in_first_byte) & 15) ^ 0x0A; // 15 = 00001111; 0x0A = 00001010
				bit_pos += 4;
				positions[nelectrons] = npixels;
				symbols[nelectrons] = subpixel;
				nelectrons++;
				npixels++;
			}
		}
		else
		{
			while(true)
			{
				first_byte = pos + (bit_pos >> 3);
				bit_offset_in_first_byte = bit_pos & 7; // 7 = 00000111 (same as % 8)
				chunk = *(unsigned int*)(buf + first_byte);
				rle = (unsigned char)((chunk >> bit_offset_in_first_byte) & 255); // 255 = 11111111
				bit_pos += 8;
				npixels += rle;
				if (npixels == frame_size_bits) break;
				if (rle == 255) continue; // this should be rare.

				first_byte = pos + (bit_pos >> 3);
				bit_offset_in_first_byte = bit_pos & 7;
				chunk = *(unsigned int*)(buf + first_byte);
				subpixel = (unsigned char)((chunk >> bit_offset_in_first_byte) & 15) ^ 0x0A; // 15 = 00001111; 0x0A = 00001010

				bit_pos += 4;
				positions[nelectrons] = npixels;
				symbols[nelectrons] = subpixel;
				nelectrons++;
				npixels++;
			}
		}


		for (int i = 0; i < nelectrons; i++)
		{
			int x,y;
			if (super_res_factor == 1)
			{
				x = positions[i] & 4095; // 4095 = 111111111111b
				y = positions[i] >> 12; //  4096 = 2^12
			}
			else if (super_res_factor == 2)
			{
				x = ((positions[i] & 4095) << 1) | ((symbols[i] & 2) >> 1); //render8K; 4095 = 111111111111b, 2 = 00000010b
				y = ((positions[i] >> 12) << 1) | ((symbols[i] & 8) >> 3); //render8K;  4096 = 2^12, 8 = 00001000b
			}
			else if (super_res_factor == 4)
			{
				x = ((positions[i] & 4095) << 2) | (symbols[i] & 3); //render16K; 4095 = 111111111111b, 3 = 00000011b
				y = ((positions[i] >> 12) << 2) | ((symbols[i] & 12) >> 2); //render16K;  4096 = 2^12, 12 = 00001100b
			}

			current_address = current_frame_image.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);

			if (output_sum_filename != NULL)
			{

				unaligned_sum.real_values[current_address] += 1.0f;
			}

			current_frame_image.real_values[current_address] += 1.0f;
		}

		frame_binning_counter++;
		if (frame_binning_counter == temporal_frame_bin_factor)
		{
			current_frame_image.WriteSlice(&frame_output_file, output_slice_number);
			wxPrintf("Writing Slice %i\n", output_slice_number);
			current_frame_image.SetToConstant(0.0f);
			frame_binning_counter = 0;
			output_slice_number++;
		}
		//wxPrintf("Frame: %d\n",iframe);
	}

	if (output_sum_filename != NULL) unaligned_sum.QuickAndDirtyWriteSlice(output_sum_filename->ToStdString(), 1);
}
