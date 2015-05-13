#include "core_headers.h"

//!>  \brief  Default contructor - just calls init to setup the pointers.
//

MRCHeader::MRCHeader()
{
	buffer = new char[1024];
	InitPointers();

}

//!>  \brief  Default destructor - deallocate the memory;
//


MRCHeader::~MRCHeader()
{
	delete [] buffer;
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
	origin_x				= (int*) &buffer[196];
	origin_y				= (int*) &buffer[200];
	origin_z				= (int*) &buffer[204];
	map						= (char*) &buffer[208];
	machine_stamp			= (int*) &buffer[212];
	rms                 	= (float*) &buffer[216];
	number_of_labels_used 	= (int*) &buffer[220];
	labels					= (char*) &buffer[224];



}

void MRCHeader::ReadHeader(std::fstream *MRCFile)
{
	MyDebugAssertTrue(MRCFile->is_open(), "File not open!");

	// Read the first 1024 bytes into buffer, the pointers should then all be set..
    MRCFile->seekg(0);
    MRCFile->read (buffer, 1024);

    // work out some extra details..

	switch ( mode[0] )
	{
		case 0:
			bytes_per_pixel = 1;
		break;

		case 1:
			bytes_per_pixel = 2;
		break;

		case 2:
			bytes_per_pixel = 4;
		break;

		default:
		{
			printf("Error: Complex MRC files not currently supported!!\n");
			abort();
		}
		break;
	}
}

void MRCHeader::WriteHeader(std::fstream *MRCFile)
{
	MyDebugAssertTrue(MRCFile->is_open(), "File not open!");

	// Write the first 1024 bytes from buffer.

    MRCFile->seekg(0);
    MRCFile->write (buffer, 1024);

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
	space_group_number[0] 	= 0;
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

	SetLocalMachineStamp();

	rms[0]                 	= 0;
	number_of_labels_used[0]= 1;

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

	for (counter = 10; counter < 800; counter++)
	{
		labels[counter]					= ' ';
	}

}

void MRCHeader::SetLocalMachineStamp()
{
	int i = 858927408;
	char *first_byte  = (char*)&i;

	if (first_byte[0] == '0')
	{
		buffer[212] = 68;
		buffer[213] = 65;
		buffer[214] = 0;
		buffer[215] = 0;
	}
	else
	if (first_byte[0] == '3')
	{
		buffer[212] = 17;
		buffer[213] = 17;
		buffer[214] = 0;
		buffer[215] = 0;
	}
	else
	{
		buffer[212] = 34;
		buffer[213] = 33;
		buffer[214] = 0;
		buffer[215] = 0;
	}
}
