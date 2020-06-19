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
	uint32 current_x = 0;
	uint32 current_y = 0;
	uint32 compression;

	TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&logical_dimension_x);
	TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&logical_dimension_y);
	TIFFGetField(tif,TIFFTAG_COMPRESSION,&compression);
	if (compression == 65000) bits_per_rle = 8;
	else if (compression == 65001) bits_per_rle = 7;
	//logical_dimension_x = current_x;
	//logical_dimension_y = current_y;
	while (TIFFSetDirectory(tif, number_of_frames) != 0) number_of_frames++;
	fseek(fh, 0, SEEK_END);
	file_size = ftell(fh);
	fseek(fh, 0, SEEK_SET);

	wxPrintf("x_size = %d, y_size = %d, bits = %d, nframes = %d, file_size = %li\n", logical_dimension_x, logical_dimension_y, bits_per_rle, number_of_frames, file_size);
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
		wxPrintf("EER in TIFF: Read frame %3d, nstrips = %d, frame size = %lld, current pos in buffer = %9lld / %lld\n", frame, nstrips, frame_sizes[frame], pos, file_size);
	} // end of loop over slices
	//TIFFClose(tif);
}

void EerFile::rleFrames()
{
	ReadSlicesFromDisk();
	//ion_of_each_frame = new unsigned int[number_of_frames];
	for (int iframe = 0; iframe < number_of_frames; iframe++)
	{
		unsigned long long position = frame_starts[iframe];
		unsigned int number_of_bits_left = 0;
		unsigned int bits_left = 0;
		const int max_electrons = frame_sizes[iframe] * 2; // at 4 bits per electron (very permissive bound!)
		unsigned char *rles = new unsigned char[max_electrons];
		unsigned char * subpixels = new unsigned char[max_electrons];
		unsigned int ion_sum = 0;
		unsigned long long total_blocks = frame_sizes[iframe] / 8; // 8 bytes as a block

		for (int block = 0; block < total_blocks; block++)
		{
			unsigned long long block_data = (*(unsigned long long*)(buf + position));
			unsigned long long block_to_use = block_data >> (number_of_bits_left);

			if (number_of_bits_left == 0)
			{
				unsigned long long block_new = block_to_use;
				unsigned long long block_55 = block_new >> 9;
				number_of_bits_left = 9;
				bits_left = block_data & 0x1FF; //0x1FF = 111111111
				for (int i = 0; i < 5; i++)
				{
					int identifier = ion_sum;
					identifier += (4 - i);
					subpixels[identifier] = block_55 & 0xF; //0xF = 1111
					block_55 = block_55 >> 4;
					rles[identifier] = block_55 & 0x7F; //0x7F = 1111111
					block_55 = block_55 >> 7;
				}
				ion_sum += 5;
			}
			else if (number_of_bits_left == 1)
			{
				unsigned long long block_new = bits_left << 63 + block_to_use;
				unsigned long long block_55 = block_new >> 9;
				number_of_bits_left = 10;
				bits_left = block_data & 0x3FF; //0x3FF = 1111111111
				for (int i = 0; i < 5; i++)
				{
					int identifier = ion_sum;
					identifier += (4 - i);
					subpixels[identifier] = block_55 & 0xF; //0xF = 1111
					block_55 = block_55 >> 4;
					rles[identifier] = block_55 & 0x7F; //0x7F = 1111111
					block_55 = block_55 >> 7;
				}
				ion_sum += 5;
			}
			else
			{
				unsigned long long block_new = bits_left << (64 - number_of_bits_left) + block_to_use;
				unsigned long long block_55 = block_new >> 9;
				number_of_bits_left += 9;
				unsigned int tmp1 = pow(2, (number_of_bits_left + 9));
				bits_left = block_data % tmp1;
				for (int i = 0; i < 5; i++)
				{
					int identifier = ion_sum;
					identifier += (4 - i);
					subpixels[identifier] = block_55 & 0xF; //0xF = 1111
					block_55 = block_55 >> 4;
					rles[identifier] = block_55 & 0x7F; //0x7F = 1111111
					block_55 = block_55 >> 7;
				}
				number_of_bits_left -= 11;
				unsigned int block_11 = bits_left >> number_of_bits_left;
				unsigned int tmp2 = pow(2, number_of_bits_left);
				bits_left %= tmp2;
				int identifier = ion_sum + 5;
				subpixels[identifier] = block_11 & 0xF; //0xF = 1111
				block_11 = block_55 >> 4;
				rles[identifier] = block_11 & 0x7F; //0x7F = 1111111
				ion_sum += 6;
			}

			position += 8;

		}
		//ion_of_each_frame[iframe] = ion_sum;
		wxPrintf("Frame %3d, frame size = %lld, ion_sum = %d, ", iframe, frame_sizes[iframe], ion_sum);
		ReadCoordinateFromRle1(ion_sum, rles, subpixels);
		//wxPrintf("\n");
	}

}

void EerFile::ReadCoordinateFromRle1(unsigned int ion_number, unsigned char * rle_in_each_frame, unsigned char * subpixels_in_each_frame)
{
	unsigned int x = 1, y = 1, x_sum = 0;
	unsigned int n_127 = 0;
	for (unsigned int ion = 0; ion < ion_number; ion++)
	{
		unsigned int zeros = rle_in_each_frame[ion];
		//wxPrintf("zeros = %u\n", zeros);

		unsigned int subpixel = subpixels_in_each_frame[ion];
		if (zeros == 127)
		{
			//wxPrintf("zeros = %u, subpixel = %u\n", zeros, subpixel);
			n_127 += 1;
		}
		x += zeros;
		//wxPrintf("zeros = %u, subpixel = %u\n",zeros, subpixel);
		x_sum += zeros;
		if (x > logical_dimension_x)
		{
			x -= logical_dimension_x;
			y += 1;
		}
		if (y > 3000) wxPrintf("x = %4u, y = %4u, subpixel = %4u\t", x, y, subpixel);
		x += 1;
		x_sum += 1;
	}
	wxPrintf("pixel_sum = %u, n_127 = %u\n", x_sum, n_127);
}

void EerFile::ReadCoordinateFromRle2(unsigned int ion_number, unsigned char * rle_in_each_frame, unsigned char * subpixels_in_each_frame)
{
	unsigned int x = 0, y = 0, x_sum = 0;
	unsigned int n_127 = 0;
	for (unsigned int ion = 0; ion < ion_number; ion++)
	{
		unsigned int zeros = rle_in_each_frame[ion];
		//wxPrintf("zeros = %u\n", zeros);

		unsigned int subpixel = subpixels_in_each_frame[ion];

		if (zeros == 127)
		{
			//wxPrintf("zeros = %u, subpixel = %u\n", zeros, subpixel);
			n_127 += 1;
		}

		x_sum += zeros;
		int x = ((x_sum & 4095) << 1) | ((subpixel & 2) >> 1); // 4095 = 111111111111b, 2 = 00000010b
		int y = ((x_sum >> 12) << 1) | ((subpixel & 8) >> 3); //  4096 = 2^12, 8 = 00001000b
		x_sum += 1;
		//wxPrintf("x = %u, y = %u\t", x, y);
	}
	wxPrintf("pixel_sum = %u, n_127 = %u\n", x_sum, n_127);
}
