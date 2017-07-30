#include "core_headers.h"

//!>  \brief  Default contructor - just calls init to setup the pointers.
//

MRCHeader::MRCHeader()
{
	buffer = new unsigned char[1024];
	InitPointers();
	bytes_are_swapped = false;

}

//!>  \brief  Default destructor - deallocate the memory;
//


MRCHeader::~MRCHeader()
{
	delete [] buffer;
}


void MRCHeader::PrintInfo()
{
	char *current_label = new char[81];
	int char_counter;

	// Work out pixel size
	float pixel_size[3];
	if (cell_a_x[0] == 0.0)
	{
		pixel_size[0] = 0.0;
	}
	else
	{
		pixel_size[0] = cell_a_x[0] / mx[0];
	}
	if (cell_a_y[0] == 0.0)
	{
		pixel_size[1] = 0.0;
	}
	else
	{
		pixel_size[1] = cell_a_y[0] / my[0];
	}
	if (cell_a_z[0] == 0.0)
	{
		pixel_size[2] = 0.0;
	}
	else
	{
		pixel_size[2] = cell_a_z[0] / mz[0];
	}

	// Start printing
	wxPrintf("Number of columns, rows, sections: %i, %i, %i\n",nx[0],ny[0],nz[0]);
	wxPrintf("MRC data mode: %i\n",mode[0]);
	wxPrintf("Bit depth: %i\n",bytes_per_pixel * 8);
	wxPrintf("Pixel size: %0.3f %0.3f %0.3f\n",pixel_size[0],pixel_size[1],pixel_size[2]);
	wxPrintf("Bytes in symmetry header: %i\n",symmetry_data_bytes[0]);
	wxPrintf("Bytes are swapped: ");
	if (bytes_are_swapped)
	{
		wxPrintf("yes\n");
	}
	else
	{
		wxPrintf("no\n");
	}
	for (int label_counter=0; label_counter < number_of_labels_used[0]; label_counter++)
	{
		for (char_counter=0;char_counter < 80; char_counter++)
		{
			current_label[char_counter] = labels[label_counter*80+char_counter];
		}
		current_label[80] = 0;
		wxPrintf("Label %i : %s\n",label_counter+1,current_label);
	}

	delete [] current_label;
}


float MRCHeader::ReturnPixelSize()
{
	if (cell_a_x[0] == 0.0)
	{
		return 0.0;
	}
	else
	{
		return cell_a_x[0] / mx[0];
	}
}


// Make sure to call this after the volume / image dimensions have been set
void MRCHeader::SetPixelSize(float wanted_pixel_size)
{
	cell_a_x[0] = wanted_pixel_size * mx[0];
	cell_a_y[0] = wanted_pixel_size * my[0];
	cell_a_z[0] = wanted_pixel_size * mz[0];
}

void MRCHeader::SetDimensionsImage(int wanted_x_dim, int wanted_y_dim)
{
	nx[0] = wanted_x_dim;
	ny[0] = wanted_y_dim;

	nxstart[0] = 0;
	nystart[0] = 0;
	nzstart[0] = 0;

	mx[0] = wanted_x_dim;
	my[0] = wanted_y_dim;


	space_group_number[0] = 1; // Contravening MRC2014 definition, which indicate we should set this to 0 when dealing with (stacks of) 2D images
}

void MRCHeader::SetNumberOfImages(int wanted_number_of_images)
{
	nz[0] = wanted_number_of_images;
	mz[0] = wanted_number_of_images;
}

void MRCHeader::SetDimensionsVolume(int wanted_x_dim, int wanted_y_dim, int wanted_z_dim)
{
	nx[0] = wanted_x_dim;
	ny[0] = wanted_y_dim;
	nz[0] = wanted_z_dim;

	nxstart[0] = 0;
	nystart[0] = 0;
	nzstart[0] = 0;

	mx[0] = wanted_x_dim;
	my[0] = wanted_y_dim;
	mz[0] = wanted_z_dim;

	space_group_number[0] = 1;
}


void MRCHeader::SetDensityStatistics( float wanted_min, float wanted_max, float wanted_mean, float wanted_rms )
{
	dmin[0] = wanted_min;
	dmax[0] = wanted_max;
	dmean[0] = wanted_mean;

	rms[0] = wanted_rms;
}

void MRCHeader::ResetLabels()
{
	labels[0] = '*';
	labels[1] = '*';
	labels[2] = ' ';
	labels[3] = 'G';
	labels[4] = 'u';
	labels[5] = 'i';
	labels[6] = 'X';
	labels[7] = ' ';
	labels[8] = '*';
	labels[9] = '*';

	for (int counter = 10; counter < 800; counter++)
	{
		labels[counter]					= ' ';
	}
	number_of_labels_used[0] = 1;
}

void MRCHeader::ResetOrigin()
{
	origin_x[0] = 0;
	origin_y[0] = 0;
	origin_z[0] = 0;
}

void MRCHeader::SetOrigin(float wanted_x, float wanted_y, float wanted_z)
{
	origin_x[0] = wanted_x;
	origin_y[0] = wanted_y;
	origin_z[0] = wanted_z;
}



//!>  \brief  Setup the pointers so that they point to the correct place in the buffer
//

void MRCHeader::InitPointers()
{
	nx 						= (int*) &buffer[0];
	ny 						= (int*) &buffer[4];
	nz 						= (int*) &buffer[8];
	mode 					= (int*) &buffer[12];
	nxstart 				= (int*) &buffer[16];
	nystart 				= (int*) &buffer[20];
	nzstart 				= (int*) &buffer[24];
	mx						= (int*) &buffer[28];
	my 						= (int*) &buffer[32];
	mz						= (int*) &buffer[36];
	cell_a_x				= (float*) &buffer[40];
	cell_a_y				= (float*) &buffer[44];
	cell_a_z				= (float*) &buffer[48];
	cell_b_x				= (float*) &buffer[52];
	cell_b_y				= (float*) &buffer[56];
	cell_b_z				= (float*) &buffer[60];
	map_c					= (int*) &buffer[64];
	map_r       			= (int*) &buffer[68];
	map_s       			= (int*) &buffer[72];
	dmin					= (float*) &buffer[76];
	dmax       	 			= (float*) &buffer[80];
	dmean       			= (float*) &buffer[84];
	space_group_number 		= (int*) &buffer[88];
	symmetry_data_bytes 	= (int*) &buffer[92];
	extra					= (int*) &buffer[96];
	origin_x				= (float*) &buffer[196];
	origin_y				= (float*) &buffer[200];
	origin_z				= (float*) &buffer[204];
	map						= (char*) &buffer[208];
	machine_stamp			= (int*) &buffer[212];
	rms                 	= (float*) &buffer[216];
	number_of_labels_used 	= (int*) &buffer[220];
	labels					= (char*) &buffer[224];



}

void MRCHeader::ReadHeader(std::fstream *MRCFile)
{
	MyDebugAssertTrue(MRCFile->is_open(), "File not open!");

	char	*temp_signed_buffer = (char *) buffer;

	// Read the first 1024 bytes into buffer, the pointers should then all be set..
    MRCFile->seekg(0);
    MRCFile->read (temp_signed_buffer, 1024);

    // Is this a byte-swapped file?
    bytes_are_swapped = abs(mode[0]) > SWAPTRIG || abs(nx[0]) > SWAPTRIG;

    // We may need to byte-swap the header
    if (bytes_are_swapped)
    {
    	wxPrintf("Swapping byte\n");
    	for (int i = 0; i < 1024 - 800; i ++ )
    	{
    		swapbytes(buffer + i, 4);
    	}
    }

	//MyDebugAssertTrue(ReturnMachineStamp() == ReturnLocalMachineStamp(), "Byteswapping not yet supported");
	wxPrintf("       local mchst: %i\n", ReturnLocalMachineStamp());
	wxPrintf("mchst from headers: %i\n",ReturnMachineStamp());

    // work out some extra details..

	switch ( mode[0] )
	{
		case 0:
			bytes_per_pixel = 1;
			pixel_data_are_signed = true; // Note that MRC mode 0 is sometimes signed, sometimes not signed. TODO: sort this out by checking the imodStamp header
			pixel_data_are_of_type = MRCByte;
			pixel_data_are_complex = false;
		break;

		case 1:
			bytes_per_pixel = 2;
			pixel_data_are_signed = true;
			pixel_data_are_of_type = Float;
			pixel_data_are_complex = false;
		break;

		case 2:
			bytes_per_pixel = 4;
			pixel_data_are_signed = true;
			pixel_data_are_of_type = MRCByte;
			pixel_data_are_complex = false;
		break;

		case 3:
			bytes_per_pixel = 2;
			pixel_data_are_signed = true;
			pixel_data_are_of_type = MRCInteger;
			pixel_data_are_complex = true;
		break;

		case 4:
			bytes_per_pixel = 4;
			pixel_data_are_signed = true;
			pixel_data_are_of_type = MRCFloat;
			pixel_data_are_complex = true;
		break;

		case 5:
			bytes_per_pixel = 1;
			pixel_data_are_signed = false;
			pixel_data_are_of_type = MRCByte;
			pixel_data_are_complex = false;
		break;

		case 6:
			bytes_per_pixel = 2;
			pixel_data_are_signed = false;
			pixel_data_are_of_type = MRCInteger;
			pixel_data_are_complex = false;
		break;


		default:
		{
			MyPrintfRed("Error: mode %i MRC files not currently supported\n",mode[0]);
			abort();
		}
		break;
	}
}

void MRCHeader::WriteHeader(std::fstream *MRCFile)
{
	MyDebugAssertTrue(MRCFile->is_open(), "File not open!");

	char	*temp_signed_buffer = (char *) buffer;

	// Write the first 1024 bytes from buffer.

    MRCFile->seekg(0);
    MRCFile->write (temp_signed_buffer, 1024);

}

void MRCHeader::BlankHeader()
{
	long counter;

	nx[0] 						= 0;
	ny[0] 						= 0;
	nz[0] 						= 0;
	mode[0] 					= 2; // always for now
	nxstart[0] 				= 0;
	nystart[0] 				= 0;
	nzstart[0] 				= 0;
	mx[0]						= 1;
	my[0] 						= 1;
	mz[0]						= 1;
	cell_a_x[0]				= 1.0;
	cell_a_y[0]				= 1.0;
	cell_a_z[0]				= 1.0;
	cell_b_x[0]				= 90.0;
	cell_b_y[0]				= 90.0;
	cell_b_z[0]				= 90.0;
	map_c[0]				= 1;
	map_r[0]       			= 2;
	map_s[0]       			= 3;
	dmin[0]					= 0.0;
	dmax[0]       	 		= 0.0;
	dmean[0]       			= 0.0;
	space_group_number[0] 	= 1; // assume we'll treat stacks of images as volumes
	symmetry_data_bytes[0] 	= 0;

	for (counter = 0; counter < 25; counter++)
	{
		extra[counter] = 0;
	}

	origin_x[0]				= 0;
	origin_y[0]				= 0;
	origin_z[0]				= 0;
	map[0]					= 'M';
	map[1]					= 'A';
	map[2]                  = 'P';
	map[3]                  = ' ';

	SetMachineStampToLocal();

	rms[0]                 	= 0;
	number_of_labels_used[0]= 1;

	ResetLabels();

	bytes_per_pixel = 4;

}



void MRCHeader::SetMachineStampToLocal()
{
	/*
	int i = 858927408;
	char *first_byte  = (char*)&i;

	if (first_byte[0] == '0') // little-endian
	{
		// 0100 0100
		buffer[212] = 68;
		// 0100 0001
		buffer[213] = 65;
		buffer[214] = 0;
		buffer[215] = 0;
	}
	else
	if (first_byte[0] == '3') // big-endian
	{
		// 0001 0001
		buffer[212] = 17;
		// 0001 0001
		buffer[213] = 17;
		buffer[214] = 0;
		buffer[215] = 0;
	}
	else // mixed endianity machine (vax)
	{
		// 0010 0010
		buffer[212] = 34;
		// 0010 0001
		buffer[213] = 33;
		buffer[214] = 0;
		buffer[215] = 0;
	}
	*/
	machine_stamp = (int*) ReturnLocalMachineStamp();
}

int MRCHeader::ReturnMachineStamp()
{
	return *machine_stamp;
}


/*
 * Compute the local machine's "machinestamp", which is endian-specific.
 *
 * Fortran code from jfem (unblur, ctffind4, etc.):
 *
 *          integer(kind=4),  parameter     ::  a0  =   48
            integer(kind=4),  parameter     ::  a1  =   49
            integer(kind=4),  parameter     ::  a2  =   50
            integer(kind=4),  parameter     ::  a3  =   51
 *
 *          i=a0+a1*256+a2*(256**2)+a3*(256**3) !  = 858927408 (decimal)
                                                !  = 0011 0011 0011 0010 0011 0001 0011 0000 (binary, little endian)
                                                !  when this is converted to ASCII characters (1 byte per character, with the most significant bit always 0)
                                                ! this will give different results on little- and big-endian machines
                                                ! For example, '0' in ASCII has decimal value 48 and bit value 011 0000
                                                ! '3' in ASCII has decimal value 51 and bit value 011 0011
                                                ! Therefore the value computed above, when converted to bytes will have the first byte
                                                ! read off as ASCII character '0' on little-endian and '3' on big-endian machines
 *
 *
 */
int MRCHeader::ReturnLocalMachineStamp()
{
	int local_machine_stamp;
	int *machine_stamp_ptr;
	char local_buffer[4];
	machine_stamp_ptr = (int*) local_buffer[0];

	int i = 858927408;
	char *first_byte  = (char*)&i;

	if (first_byte[0] == '0') // little-endian
	{
		// 0100 0100
		local_buffer[0] = 68;
		// 0100 0001
		local_buffer[1] = 65;
		local_buffer[2] = 0;
		local_buffer[3] = 0;
	}
	else
	if (first_byte[0] == '3') // big-endian
	{
		// 0001 0001
		local_buffer[0] = 17;
		// 0001 0001
		local_buffer[1] = 17;
		local_buffer[2] = 0;
		local_buffer[3] = 0;
	}
	else // mixed endianity machine (vax)
	{
		// 0010 0010
		local_buffer[0] = 34;
		// 0010 0001
		local_buffer[1] = 33;
		local_buffer[2] = 0;
		local_buffer[3] = 0;
	}

	//local_machine_stamp = *machine_stamp_ptr;

	return local_machine_stamp;
}

