#include "core_headers.h"

//!>  \brief  Default contructor - just calls init to setup the pointers.
//

MRCHeader::MRCHeader( ) {
    buffer = new char[1024];
    InitPointers( );
    this_is_in_mastronarde_4bit_hack_format = false;
    dimensions_set                          = false;
}

//!>  \brief  Default destructor - deallocate the memory;
//

MRCHeader::~MRCHeader( ) {
    delete[] buffer;
}

void MRCHeader::PrintInfo( ) {
    char* current_label = new char[81];
    int   char_counter;

    // Work out pixel size
    float pixel_size[3];
    if ( cell_a_x[0] == 0.0 ) {
        pixel_size[0] = 0.0;
    }
    else {
        pixel_size[0] = cell_a_x[0] / mx[0];
    }
    if ( cell_a_y[0] == 0.0 ) {
        pixel_size[1] = 0.0;
    }
    else {
        pixel_size[1] = cell_a_y[0] / my[0];
    }
    if ( cell_a_z[0] == 0.0 ) {
        pixel_size[2] = 0.0;
    }
    else {
        pixel_size[2] = cell_a_z[0] / mz[0];
    }

    // Start printing
    wxPrintf("Number of columns, rows, sections: %i, %i, %i\n", nx[0], ny[0], nz[0]);
    wxPrintf("MRC data mode: %i", mode[0]);
    if ( mode[0] == 0 && PixelDataAreSigned( ) ) {
        wxPrintf(" (bytes - signed in file)\n");
    }
    else {
        wxPrintf("\n");
    }
    wxPrintf("Bit depth: %i\n", int(bytes_per_pixel * 8));
    wxPrintf("Pixel size: %0.3f %0.3f %0.3f\n", pixel_size[0], pixel_size[1], pixel_size[2]);
    wxPrintf("Column index: %i; Row index: %i; Section index: %i\n", map_c[0], map_r[0], map_s[0]);
    wxPrintf("Bytes in symmetry header: %i\n", symmetry_data_bytes[0]);
    for ( int label_counter = 0; label_counter < number_of_labels_used[0]; label_counter++ ) {
        for ( char_counter = 0; char_counter < 80; char_counter++ ) {
            current_label[char_counter] = labels[label_counter * 80 + char_counter];
        }
        current_label[80] = 0;
        wxPrintf("Label %i : %s\n", label_counter + 1, current_label);
    }

    delete[] current_label;
}

float MRCHeader::ReturnPixelSize( ) {
    MyDebugAssertTrue(dimensions_set, "Dimensions not set yet!");
    if ( cell_a_x[0] == 0.0 ) {
        return 0.0;
    }
    else {
        return cell_a_x[0] / mx[0];
    }
}

void MRCHeader::SetPixelSize(float wanted_pixel_size) {
    MyDebugAssertTrue(dimensions_set, "Dimensions not set yet!");
    cell_a_x[0] = wanted_pixel_size * mx[0];
    cell_a_y[0] = wanted_pixel_size * my[0];
    cell_a_z[0] = wanted_pixel_size * mz[0];
}

void MRCHeader::SetDimensionsImage(int wanted_x_dim, int wanted_y_dim) {
    nx[0] = wanted_x_dim;
    ny[0] = wanted_y_dim;

    nxstart[0] = 0;
    nystart[0] = 0;
    nzstart[0] = 0;

    mx[0] = wanted_x_dim;
    my[0] = wanted_y_dim;

    space_group_number[0] = 1; // Contravening MRC2014 definition, which indicate we should set this to 0 when dealing with (stacks of) 2D images
    dimensions_set        = true;
}

void MRCHeader::SetNumberOfImages(int wanted_number_of_images) {
    nz[0] = wanted_number_of_images;
    mz[0] = wanted_number_of_images;
}

void MRCHeader::SetDimensionsVolume(int wanted_x_dim, int wanted_y_dim, int wanted_z_dim) {
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
    dimensions_set        = true;
}

void MRCHeader::SetDensityStatistics(float wanted_min, float wanted_max, float wanted_mean, float wanted_rms) {
    dmin[0]  = wanted_min;
    dmax[0]  = wanted_max;
    dmean[0] = wanted_mean;

    rms[0] = wanted_rms;
}

void MRCHeader::ResetLabels( ) {
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

    for ( int counter = 10; counter < 800; counter++ ) {
        labels[counter] = ' ';
    }
    number_of_labels_used[0] = 1;
}

void MRCHeader::ResetOrigin( ) {
    origin_x[0] = 0;
    origin_y[0] = 0;
    origin_z[0] = 0;
}

void MRCHeader::SetOrigin(float wanted_x, float wanted_y, float wanted_z) {
    origin_x[0] = wanted_x;
    origin_y[0] = wanted_y;
    origin_z[0] = wanted_z;
}

//!>  \brief  Setup the pointers so that they point to the correct place in the buffer
//

void MRCHeader::InitPointers( ) {
    nx                    = (int*)&buffer[0];
    ny                    = (int*)&buffer[4];
    nz                    = (int*)&buffer[8];
    mode                  = (int*)&buffer[12];
    nxstart               = (int*)&buffer[16];
    nystart               = (int*)&buffer[20];
    nzstart               = (int*)&buffer[24];
    mx                    = (int*)&buffer[28];
    my                    = (int*)&buffer[32];
    mz                    = (int*)&buffer[36];
    cell_a_x              = (float*)&buffer[40];
    cell_a_y              = (float*)&buffer[44];
    cell_a_z              = (float*)&buffer[48];
    cell_b_x              = (float*)&buffer[52];
    cell_b_y              = (float*)&buffer[56];
    cell_b_z              = (float*)&buffer[60];
    map_c                 = (int*)&buffer[64];
    map_r                 = (int*)&buffer[68];
    map_s                 = (int*)&buffer[72];
    dmin                  = (float*)&buffer[76];
    dmax                  = (float*)&buffer[80];
    dmean                 = (float*)&buffer[84];
    space_group_number    = (int*)&buffer[88]; // should be 0 for an image stack, 1 for a volume
    symmetry_data_bytes   = (int*)&buffer[92]; // number of bytes in extended header ("next" in IMOD C)
    extra                 = (int*)&buffer[96];
    imodStamp             = (int*)&buffer[152];
    imodFlags             = (int*)&buffer[156];
    origin_x              = (float*)&buffer[196];
    origin_y              = (float*)&buffer[200];
    origin_z              = (float*)&buffer[204];
    map                   = (char*)&buffer[208];
    machine_stamp         = (int*)&buffer[212];
    rms                   = (float*)&buffer[216];
    number_of_labels_used = (int*)&buffer[220];
    labels                = (char*)&buffer[224];
}

void MRCHeader::ReadHeader(std::fstream* MRCFile) {
    MyDebugAssertTrue(MRCFile->is_open( ), "File not open!");

    // Read the first 1024 bytes into buffer, the pointers should then all be set..
    MRCFile->seekg(0);
    MRCFile->read(buffer, 1024);

    dimensions_set = true;

    // work out some extra details..

    switch ( mode[0] ) {
        case 0:
            bytes_per_pixel = 1;
            if ( imodStamp[0] == 1146047817 ) {
                // This is a file from IMOD. Let's check whether signed or not
                pixel_data_are_signed = imodFlags[0] & 1;
            }
            else {
                pixel_data_are_signed = false; // default is unsigned bytes for mode 0
            }
            pixel_data_are_of_type = MRCByte;
            pixel_data_are_complex = false;

            // need to check for 4-bit..

            if ( sizeCanBe4BitK2SuperRes(nx[0], ny[0]) == 1 ) // this is David Mastronarde's 4-bit hack
            {
                nx[0] *= 2;
                this_is_in_mastronarde_4bit_hack_format = true;
                bytes_per_pixel                         = 0.5;
            }
            else {
                this_is_in_mastronarde_4bit_hack_format = false;
            }

            break;

        case 1:
            bytes_per_pixel        = 2;
            pixel_data_are_signed  = true;
            pixel_data_are_of_type = Float; // FIXME: Why are we using defintions from multiple enums (this from dm.h others from mrc_header.h)
            pixel_data_are_complex = false;
            break;

        case 2:
            bytes_per_pixel        = 4;
            pixel_data_are_signed  = true;
            pixel_data_are_of_type = MRCByte; // FIXME: Why is this not MRCFloat?
            pixel_data_are_complex = false;
            break;

        case 12: {
            bytes_per_pixel        = 2;
            pixel_data_are_signed  = true;
            pixel_data_are_of_type = MRCFloat16;
            pixel_data_are_complex = false;
            break;
        }

        case 3:
            bytes_per_pixel        = 2;
            pixel_data_are_signed  = true;
            pixel_data_are_of_type = MRCInteger;
            pixel_data_are_complex = true;
            break;

        case 4:
            bytes_per_pixel        = 4;
            pixel_data_are_signed  = true;
            pixel_data_are_of_type = MRCFloat;
            pixel_data_are_complex = true;
            break;

        case 5:
            bytes_per_pixel        = 1;
            pixel_data_are_signed  = false;
            pixel_data_are_of_type = MRCByte;
            pixel_data_are_complex = false;
            break;

        case 6:
            bytes_per_pixel        = 2;
            pixel_data_are_signed  = false;
            pixel_data_are_of_type = MRCInteger;
            pixel_data_are_complex = false;
            break;

        case 101:
            bytes_per_pixel        = 0.5;
            pixel_data_are_signed  = false;
            pixel_data_are_of_type = MRC4Bit;
            pixel_data_are_complex = false;
            break;

        default: {
            MyPrintfRed("Error: mode %i MRC files not currently supported\n", mode[0]);
            DEBUG_ABORT;
        } break;
    }
}

void MRCHeader::WriteHeader(std::fstream* MRCFile) {
    MyDebugAssertTrue(MRCFile->is_open( ), "File not open!");

    // Write the first 1024 bytes from buffer.

    MRCFile->seekg(0);
    MRCFile->write(buffer, 1024);
}

void MRCHeader::BlankHeader( ) {
    long counter;

    dimensions_set = false;

    nx[0]                  = 0;
    ny[0]                  = 0;
    nz[0]                  = 0;
    mode[0]                = 2; // always for now
    nxstart[0]             = 0;
    nystart[0]             = 0;
    nzstart[0]             = 0;
    mx[0]                  = 1;
    my[0]                  = 1;
    mz[0]                  = 1;
    cell_a_x[0]            = 1.0;
    cell_a_y[0]            = 1.0;
    cell_a_z[0]            = 1.0;
    cell_b_x[0]            = 90.0;
    cell_b_y[0]            = 90.0;
    cell_b_z[0]            = 90.0;
    map_c[0]               = 1;
    map_r[0]               = 2;
    map_s[0]               = 3;
    dmin[0]                = 0.0;
    dmax[0]                = 0.0;
    dmean[0]               = 0.0;
    space_group_number[0]  = 1; // assume we'll treat stacks of images as volumes
    symmetry_data_bytes[0] = 0;

    for ( counter = 0; counter < 25; counter++ ) {
        extra[counter] = 0;
    }

    origin_x[0] = 0;
    origin_y[0] = 0;
    origin_z[0] = 0;
    map[0]      = 'M';
    map[1]      = 'A';
    map[2]      = 'P';
    map[3]      = ' ';

    SetLocalMachineStamp( );

    rms[0]                   = 0;
    number_of_labels_used[0] = 1;

    ResetLabels( );

    bytes_per_pixel = 4;
}

void MRCHeader::SetLocalMachineStamp( ) {
    int   i          = 858927408;
    char* first_byte = (char*)&i;

    if ( first_byte[0] == '0' ) {
        buffer[212] = 68;
        buffer[213] = 65;
        buffer[214] = 0;
        buffer[215] = 0;
    }
    else if ( first_byte[0] == '3' ) {
        buffer[212] = 17;
        buffer[213] = 17;
        buffer[214] = 0;
        buffer[215] = 0;
    }
    else {
        buffer[212] = 34;
        buffer[213] = 33;
        buffer[214] = 0;
        buffer[215] = 0;
    }
}
