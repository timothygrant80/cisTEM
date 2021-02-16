#include "core_headers.h"
EerFile::EerFile()
{
	tif = NULL;
	logical_dimension_x = 0;
	logical_dimension_y = 0;
	number_of_images = 0;
	number_of_eer_frames = 0;
	number_of_eer_frames_per_image = 0;
	super_res_factor = 0;
	buf = NULL;
	frame_starts = NULL;
	frame_sizes = NULL;
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


//
bool EerFile::OpenFile(std::string wanted_filename, bool overwrite, bool wait_for_file_to_exist, bool check_only_the_first_image, int eer_super_res_factor, int eer_frames_per_image)
{

	MyDebugAssertFalse(tif != NULL,"File already open: %s",wanted_filename);
	MyDebugAssertFalse(wait_for_file_to_exist,"Waiting for file to exist not implemented for tif files");

	bool file_already_exists = DoesFileExist(wanted_filename);

	bool return_value = true;

	filename = wanted_filename;
	super_res_factor = eer_super_res_factor;
	number_of_eer_frames_per_image = eer_frames_per_image;

	// if overwrite is specified, then we delete the file nomatter what..
	if (overwrite) file_already_exists = false;

	if (file_already_exists)
	{
		// We open to read/write
		// The first dictionary is automatically read in
		tif = TIFFOpen(wanted_filename.c_str(),"rc");
		fh = fopen(wanted_filename.c_str(), "r");
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
		MyDebugAssertTrue(false,"EER file writing is not supported");
	}
	pixel_size = 1.0; //TODO: can we get a useful pixel value from the header of an EER file?
	return return_value;
}

void EerFile::CloseFile()
{
	if (tif != NULL) TIFFClose(tif);
	tif = NULL;
}

void EerFile::PrintInfo()
{
	wxPrintf("Filename  : %s\n",filename.GetFullName());
	wxPrintf("Dimensions: %i %i %i\n",ReturnXSize()*super_res_factor,ReturnYSize()*super_res_factor,ReturnZSize());
	wxPrintf("Number of EER frames: %i\n",number_of_eer_frames);
	wxPrintf("Super resolution factor: %i\n", super_res_factor);
	wxPrintf("Bits per RLE: %i\n",bits_per_rle);
}

/*
 * By default, this method will check all the images within the file, to make sure
 * they are valid (not corrupted) and of the same dimensions. However, this can take
 * a long time. If you want to save that time, set check_only_the_first_image to true.
 * This is risky because we may not notice that the file is corrupt or has unusual dimensions.
 * The file size and the number of images will not be set in that case. 
 */
bool EerFile::ReadLogicalDimensionsFromDisk(bool check_only_the_first_image)
{
	MyDebugAssertTrue(tif != NULL,"File must be open");
	MyDebugAssertTrue(fh != NULL,"File must be open");
	MyDebugAssertFalse(number_of_eer_frames_per_image == 0, "Number of EER frames per image has not yet been set. Cannot work out logical dimensions.");
	/*
	 * Since the file was already open, EerOpen has already read in the first dictionary
	 * and it must be valid, else we would have returned an error at open-time already
	 */
	int set_dir_ret = TIFFSetDirectory(tif,0);
	bool return_value = (set_dir_ret == 1);
	int dircount = 1;

	uint32 original_x = 0;
	uint16 compression;

	// Work out logical dimensions of the frames (x,y)
	TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&logical_dimension_x);
	TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&logical_dimension_y);
	frame_size_bits = logical_dimension_x * logical_dimension_y;

	// Work out RLE compression scheme
	TIFFGetField(tif,TIFFTAG_COMPRESSION,&compression);
	if (compression == 65000) bits_per_rle = 8;
	else if (compression == 65001) bits_per_rle = 7;
	else
	{
		MyPrintWithDetails("Warning:: Unknown Compression in EER tif file, assuming 7-bit (65001)");
		bits_per_rle = 7;
	}

	if (!check_only_the_first_image)
	{
		// Work out the number of frames
		while (TIFFSetDirectory(tif, number_of_eer_frames) != 0) number_of_eer_frames++;

		// Work out the number of images
		number_of_images = number_of_eer_frames / number_of_eer_frames_per_image;

		// Work out the total file size
		fseek(fh, 0, SEEK_END);
		file_size_bytes = ftell(fh);
		fseek(fh, 0, SEEK_SET); //go back to the beginning of the file
	}
	
	return return_value;
}

void EerFile::ReadWholeFileIntoBuffer()
{
	MyDebugAssertTrue(tif != NULL,"File must be open");
	MyDebugAssertTrue(logical_dimension_x > 0, "You must call ReadLogicalDimensionsFromDisk first");
	MyDebugAssertTrue(buf == NULL,"Buffer was already allocated");
	MyDebugAssertTrue(frame_starts == NULL, "frame_starts was already allocated");
	MyDebugAssertTrue(frame_sizes == NULL, "frame_sizes was already allocated");

	frame_starts = new unsigned long long[number_of_eer_frames];
	frame_sizes = new unsigned long long[number_of_eer_frames];
	buf = new unsigned char[file_size_bytes];
	long output_counter;
	unsigned long long pos = 0;

	// Read everything into the buffer
	for (int frame = 0; frame < number_of_eer_frames; frame++)
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

void EerFile::ReadSliceFromDisk(int slice_number, float * output_array)
{
	ReadSlicesFromDisk(slice_number, slice_number, output_array);
}

/*
 * start_slice and end_slice are 1-indexed
 * output_array must be allocated to the correct dimensions (logical_dimension_x * logical_dimension_y * super_res_factor**2 * number_of_frames)
 * and will be zeroed internally
 */
void EerFile::ReadSlicesFromDisk(int start_slice, int end_slice, float *output_array)
{
	MyDebugAssertTrue(tif != NULL,"File must be open");
	MyDebugAssertTrue(start_slice > 0 && end_slice >= start_slice && end_slice <= number_of_images,"Bad start or end slice number");
	MyDebugAssertTrue(number_of_eer_frames_per_image > 0,"You have not yet set the number of EER frames per image");
	MyDebugAssertTrue(super_res_factor > 0,"Super res factor was not set");
	MyDebugAssertTrue(logical_dimension_y > 0,"Logical dimensions were not set");


	// Read the contents of the file into the buffer
	if (buf==NULL) ReadWholeFileIntoBuffer();

	long start_pos_in_output_array = 0;

	// Loop over output slices
	for (int slice_counter = start_slice; slice_counter <= end_slice; slice_counter++)
	{
		

		// Work out start and finish EER frames (0-indexed)
		int start_eer_frame = (slice_counter-1) * number_of_eer_frames_per_image;
		int finish_eer_frame = std::min(start_eer_frame+number_of_eer_frames_per_image-1,number_of_eer_frames);

		start_pos_in_output_array = (slice_counter-start_slice) * (logical_dimension_x*logical_dimension_y*super_res_factor*super_res_factor);

		//MyDebugPrint("Reading slice %i of %i (EER frames %i to %i)",slice_counter,number_of_images, start_eer_frame,finish_eer_frame);
		DecodeToFloatArray(start_eer_frame,finish_eer_frame,&output_array[start_pos_in_output_array]);
	}

	
}

/*
 * Start and finish EER frames should be 0-indexed
 * A single image data array will be returned, summing all the events founds between the start and finish eer_frames
 */
void EerFile::DecodeToFloatArray(int start_eer_frame, int finish_eer_frame, float *output_array)
{
	MyDebugAssertTrue(buf != NULL,"Data from the file has not been read into the buffer yet");
	long current_address;

	// Zeroe the array
	for ( current_address=0; current_address < logical_dimension_x*logical_dimension_y*super_res_factor*super_res_factor; current_address++) { output_array[current_address] = 0.0f; }

	/*
	 * Decode into a list of events
	 */
	for (int iframe = start_eer_frame; iframe <= finish_eer_frame; iframe++)
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

		/*
		 * Insert the events into a pixel array
		 */
		for (int i = 0; i < nelectrons; i++)
		{
			// Work out 0-indexed x,y location of the event
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

			current_address = y * logical_dimension_y * super_res_factor + x;

			output_array[current_address] += 1.0f;
		}
	}
}

void EerFile::WriteSliceToDisk(int slice_number, float * input_array)
{
	WriteSlicesToDisk(slice_number, slice_number, input_array);
}

void EerFile::WriteSlicesToDisk(int start_slice, int end_slice, float * input_array)
{
	MyDebugAssertTrue(false,"Not implemented yet");
}