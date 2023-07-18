#include "core_headers.h"

#include "../../include/ieee-754-half/half.hpp"

MRCFile::MRCFile( ) {
    rewrite_header_on_close                         = false;
    max_number_of_seconds_to_wait_for_file_to_exist = 30;
    my_file                                         = new std::fstream;
    do_nothing                                      = false;
}

MRCFile::MRCFile(std::string filename, bool overwrite) {
    rewrite_header_on_close                         = false;
    max_number_of_seconds_to_wait_for_file_to_exist = 30;
    my_file                                         = new std::fstream;
    OpenFile(filename, overwrite);
}

MRCFile::MRCFile(std::string filename, bool overwrite, bool wait_for_file_to_exist) {
    rewrite_header_on_close                         = false;
    max_number_of_seconds_to_wait_for_file_to_exist = 30;
    my_file                                         = new std::fstream;
    OpenFile(filename, overwrite, wait_for_file_to_exist);
}

void MRCFile::SetOutputToFP16( ) {
    MyDebugAssertTrue(my_file->is_open( ), "File not open!");
    my_header.SetMode(12);
}

MRCFile::~MRCFile( ) {
    if ( my_file != NULL ) {
        CloseFile( );
        delete my_file;
    }
    my_file = NULL;
}

void MRCFile::CloseFile( ) {
    if ( my_file->is_open( ) ) {
        if ( rewrite_header_on_close == true )
            WriteHeader( );
        my_file->close( );
    }
    do_nothing = false;
}

void MRCFile::FlushFile( ) {
    if ( my_file->is_open( ) ) {
        my_file->flush( );
    }
}

bool MRCFile::OpenFile(std::string wanted_filename, bool overwrite, bool wait_for_file_to_exist, bool check_only_first_image, int eer_super_res_factor, int eer_frames_per_image) {
    //	MyDebugAssertFalse(my_file->is_open(), "File Already Open: %s",wanted_filename);
    CloseFile( );

    do_nothing = StartsWithDevNull(wanted_filename);
    if ( do_nothing ) {
        filename = wanted_filename;
    }
    else {
        bool file_already_exists;

        if ( wait_for_file_to_exist ) {
            file_already_exists = DoesFileExistWithWait(wanted_filename, max_number_of_seconds_to_wait_for_file_to_exist);
        }
        else {
            file_already_exists = DoesFileExist(wanted_filename);
        }

        //MyDebugPrintWithDetails("%s File size = %li\n",wanted_filename,ReturnFileSizeInBytesAlternative(wanted_filename));

        // if overwrite is specified, then we delete the file nomatter what..
        // if it isn't, then we need to know if the file already exists..

        if ( overwrite == true )
            file_already_exists = false;

        // Now open it, truncating to 0 if it doesn't already exist, or we specified overwrite

        if ( file_already_exists == true ) {
            my_file->open(wanted_filename.c_str( ), std::ios::in | std::ios::out | std::ios::binary);

            if ( my_file->is_open( ) == false ) {
                // Try without connecting the out (i.e. read only)
                my_file->open(wanted_filename.c_str( ), std::ios::in | std::ios::binary);

                // If it still didn't work, we're buggered
                if ( my_file->is_open( ) == false ) {
                    MyPrintWithDetails("Opening of file %s failed!! - Exiting..\n\n", wanted_filename.c_str( ));
                    DEBUG_ABORT;
                }
            }

            // read the header

            if ( file_already_exists == true )
                my_header.ReadHeader(my_file);
        }
        else {
            my_file->open(wanted_filename.c_str( ), std::ios::in | std::ios::out | std::ios::trunc | std::ios::binary);

            if ( my_file->is_open( ) == false ) {
                MyPrintWithDetails("Opening of file %s failed!! - Exiting..\n\n", wanted_filename.c_str( ));
                DEBUG_ABORT;
            }

            // Blank the header, it'll have to be written with the correct values later..

            my_header.BlankHeader( );
        }

        rewrite_header_on_close = false;

        filename = wanted_filename;
    }

    // TODO: return false is something is wrong about this file
    return true;
}

void MRCFile::PrintInfo( ) {
    wxPrintf("\nSummary information for file %s\n", filename);
    my_header.PrintInfo( );
    wxPrintf("\n");
}

float MRCFile::ReturnPixelSize( ) {
    return my_header.ReturnPixelSize( );
}

void MRCFile::SetPixelSize(float wanted_pixel_size) {
    my_header.SetPixelSize(wanted_pixel_size);
}

void MRCFile::ReadSlicesFromDisk(int start_slice, int end_slice, float* output_array) {

    using half = half_float::half;

    if ( ! do_nothing ) {
        MyDebugAssertTrue(my_file->is_open( ), "File not open!");
        MyDebugAssertTrue(start_slice <= ReturnNumberOfSlices( ), "Start slice number larger than total slices!");
        MyDebugAssertTrue(end_slice <= ReturnNumberOfSlices( ), "end slice number larger than total slices!");
        MyDebugAssertTrue(start_slice <= end_slice, "Start slice larger than end slice!");
        //MyDebugAssertTrue(my_header.ReturnMachineStamp() == my_header.ReturnLocalMachineStamp(), "Byteswapping not yet supported");
        //wxPrintf("mchst from headers: %i, local mchst: %i\n",my_header.ReturnMachineStamp(), my_header.ReturnLocalMachineStamp());

        // calculate and seek to the start byte..

        long records_to_read;
        long bytes_per_slice;
        long image_offset;
        long current_position;
        long seek_position;

        if ( my_header.Mode( ) == 101 ) {
            records_to_read = long(my_header.ReturnDimensionX( )) * long(my_header.ReturnDimensionY( )) * (long(end_slice - start_slice) + 1);

            if ( IsOdd(my_header.ReturnDimensionX( )) == true ) {
                bytes_per_slice = ((long(my_header.ReturnDimensionX( )) - 1) / 2) + 1;
                bytes_per_slice *= long(my_header.ReturnDimensionY( ));
            }
            else {
                bytes_per_slice = long(my_header.ReturnDimensionX( )) / 2;
                bytes_per_slice *= long(my_header.ReturnDimensionY( ));
            }

            image_offset     = (start_slice - 1) * bytes_per_slice;
            current_position = my_file->tellg( );
            seek_position    = 1024 + image_offset + my_header.SymmetryDataBytes( );
        }
        else {
            // check for mastronarde 4-bit hack.

            if ( my_header.ReturnIfThisIsInMastronarde4BitHackFormat( ) == true ) {
                records_to_read = long(my_header.ReturnDimensionX( )) * long(my_header.ReturnDimensionY( )) * (long(end_slice - start_slice) + 1);
                records_to_read /= 2;

                bytes_per_slice = long(my_header.ReturnDimensionX( )) * long(my_header.ReturnDimensionY( )) * long(my_header.BytesPerPixel( ));
                image_offset    = long(start_slice - 1) * bytes_per_slice;
            }
            else {
                records_to_read = long(my_header.ReturnDimensionX( )) * long(my_header.ReturnDimensionY( )) * (long(end_slice - start_slice) + 1);
                bytes_per_slice = long(my_header.ReturnDimensionX( )) * long(my_header.ReturnDimensionY( )) * long(my_header.BytesPerPixel( ));
                image_offset    = long(start_slice - 1) * bytes_per_slice;
            }

            current_position = my_file->tellg( );
            seek_position    = 1024 + image_offset + my_header.SymmetryDataBytes( );
        }

        if ( current_position != seek_position )
            my_file->seekg(seek_position);

        // we need a temp array for non float formats..

        //	wxPrintf("seek_position = %li\n", (seek_position - 1024) / 1679616 + 1);
        switch ( my_header.Mode( ) ) {
            // 1-byte integer
            case 0: {
                long  output_counter = 0;
                uint8 low_4bits;
                uint8 hi_4bits;

                char* temp_char_array = new char[records_to_read];
                my_file->read(temp_char_array, records_to_read);
                signed char*   temp_signed_char_array   = reinterpret_cast<signed char*>(temp_char_array);
                unsigned char* temp_unsigned_char_array = reinterpret_cast<unsigned char*>(temp_char_array);

                // Convert to float array
                if ( my_header.ReturnIfThisIsInMastronarde4BitHackFormat( ) ) {
                    for ( long counter = 0; counter < records_to_read; counter++ ) {

                        low_4bits = temp_char_array[counter] & 0x0F;
                        hi_4bits  = (temp_char_array[counter] >> 4) & 0x0F;

                        output_array[output_counter] = float(low_4bits);
                        output_counter++;
                        output_array[output_counter] = float(hi_4bits);
                        output_counter++;
                    }
                }
                else {
                    if ( my_header.PixelDataAreSigned( ) ) {
                        for ( long counter = 0; counter < records_to_read; counter++ ) {
                            output_array[counter] = float(temp_signed_char_array[counter] + 128);
                        }
                    }
                    else {
                        for ( long counter = 0; counter < records_to_read; counter++ ) {
                            output_array[counter] = float(temp_unsigned_char_array[counter]);
                        }
                    }
                }

                delete[] temp_char_array;
            } break;

            // 2-byte integer
            case 1: {
                short* temp_short_array = new short[records_to_read];
                my_file->read((char*)temp_short_array, records_to_read * 2);

                for ( long counter = 0; counter < records_to_read; counter++ ) {
                    output_array[counter] = float(temp_short_array[counter]);
                }

                delete[] temp_short_array;
            } break;

            // 4-byte real
            case 2:
                my_file->read((char*)output_array, records_to_read * 4);
                break;

            // 2-byte real
            case 12: {
                std::vector<half> temp_half_array(records_to_read);
                my_file->read((char*)temp_half_array.data( ), records_to_read * 2);
                for ( long counter = 0; counter < records_to_read; counter++ ) {
                    // float() operator overloaded in half_float namespace
                    output_array[counter] = float(temp_half_array[counter]);
                }
            } break;

            // unsigned 2-byte integers
            case 6: {
                unsigned short int* temp_int_array = new unsigned short int[records_to_read];
                my_file->read((char*)temp_int_array, records_to_read * 2);
                for ( long counter = 0; counter < records_to_read; counter++ ) {
                    output_array[counter] = float(temp_int_array[counter]);
                }
                delete[] temp_int_array;
            } break;

                // 101.. 4-bit integer..

            case 101: {
                // horrible format.. each byte is two pixels, if x is odd then the last one is padded.
                // so recalculate the number of bytes to read

                long input_array_position  = 0;
                long output_array_position = 0;
                int  x_pos                 = 0;

                long actual_bytes_to_read = ((end_slice - start_slice) + 1) * bytes_per_slice;

                uint8 hi_4bits;
                uint8 low_4bits;

                char* temp_char_array = new char[actual_bytes_to_read];
                my_file->read(temp_char_array, actual_bytes_to_read);

                // now we have to convert..

                for ( long counter = 0; counter < actual_bytes_to_read; counter++ ) {

                    low_4bits = temp_char_array[input_array_position] & 0x0F;
                    hi_4bits  = (temp_char_array[input_array_position] >> 4) & 0x0F;

                    //wxPrintf("\n\ninput = %i, low = %i, high = %i\n\n", int(temp_char_array[input_array_position]), int(low_4bits), int(hi_4bits));

                    input_array_position++;

                    x_pos++;

                    if ( x_pos == my_header.ReturnDimensionX( ) && IsOdd(my_header.ReturnDimensionX( )) == true ) {
                        x_pos                               = 0;
                        output_array[output_array_position] = float(low_4bits);
                        output_array_position++;
                    }
                    else {
                        output_array[output_array_position] = float(low_4bits);
                        output_array_position++;
                        output_array[output_array_position] = float(hi_4bits);
                        output_array_position++;
                    }
                }

                delete[] temp_char_array;

            } break;

            default: {
                MyPrintfRed("Error: mode %i MRC files not currently supported\n", my_header.Mode( ));
                DEBUG_ABORT;
            } break;
        }

        {
            /*
			* Deal with the cases where the data are not indexed like this:
			* - fastest = column (map_c = 1)
			* - medium = row (map_r = 2)
			* - slow = section (map_s = 3)
			*/

            long counter;
            long counter_in_file;
            long number_of_voxels = my_header.ReturnDimensionX( ) * my_header.ReturnDimensionY( ) * my_header.ReturnDimensionZ( );

            //
            int col_index;
            int row_index;
            int sec_index;

            if ( my_header.ReturnMapC( ) == 1 && my_header.ReturnMapR( ) == 2 && my_header.ReturnMapS( ) == 3 ) {
                // Nothing to do, this is how cisTEM expects the data to be laid out
            }
            else if ( my_header.ReturnMapS( ) == 1 && my_header.ReturnMapC( ) == 3 ) {

                // Allocate a temp array and copy data over
                float* temp_array;
                temp_array = new float[number_of_voxels];
                for ( counter = 0; counter < number_of_voxels; counter++ ) {
                    temp_array[counter] = output_array[counter];
                }

                // Loop over output array and copy voxel values over one by one
                counter = 0;
                for ( sec_index = 0; sec_index < my_header.ReturnDimensionZ( ); sec_index++ ) //z
                {
                    for ( row_index = 0; row_index < my_header.ReturnDimensionY( ); row_index++ ) //y
                    {
                        for ( col_index = 0; col_index < my_header.ReturnDimensionX( ); col_index++ ) //x
                        {
                            // compute address of voxel in the file
                            counter_in_file = sec_index + my_header.ReturnDimensionZ( ) * row_index + my_header.ReturnDimensionZ( ) * my_header.ReturnDimensionY( ) * col_index;

                            MyDebugAssertTrue(counter_in_file >= 0 && counter_in_file < number_of_voxels, "Oops bad counter_in_file = %li\n", counter_in_file);
                            MyDebugAssertTrue(counter >= 0 && counter < number_of_voxels, "Oops bad counter = %li\n", counter);

                            output_array[counter] = temp_array[counter_in_file];
                            counter++;
                        }
                    }
                }

                // Deallocate temp array
                delete[] temp_array;
            }
            else {
                wxPrintf("Ooops, strange ordering of data in MRC file not yet supported");
                DEBUG_ABORT;
            }
        }
    }
}

void MRCFile::WriteSlicesToDisk(int start_slice, int end_slice, float* input_array) {
    if ( ! do_nothing ) {
        MyDebugAssertTrue(my_file->is_open( ), "File not open!");
        MyDebugAssertTrue(start_slice <= end_slice, "Start slice larger than end slice!");
        MyDebugAssertTrue(start_slice <= ReturnNumberOfSlices( ), "Start slice number larger than total slices!");
        MyDebugAssertTrue(end_slice <= ReturnNumberOfSlices( ), "end slice number larger than total slices!");

        // calculate and seek to the start byte..

        long records_to_read  = long(my_header.ReturnDimensionX( )) * long(my_header.ReturnDimensionY( )) * long((end_slice - start_slice) + 1);
        long bytes_per_slice  = long(my_header.ReturnDimensionX( )) * long(my_header.ReturnDimensionY( )) * long(my_header.BytesPerPixel( ));
        long image_offset     = long(start_slice - 1) * bytes_per_slice;
        long current_position = my_file->tellg( );
        long seek_position    = 1024 + image_offset + long(my_header.SymmetryDataBytes( ));

        if ( current_position != seek_position )
            my_file->seekg(seek_position);

        // we need a temp array for non float formats..

        switch ( my_header.Mode( ) ) {
            case 0: {
                char* temp_char_array = new char[records_to_read];

                for ( long counter = 0; counter < records_to_read; counter++ ) {
                    temp_char_array[counter] = char(input_array[counter]);
                }

                my_file->write(temp_char_array, records_to_read);

                delete[] temp_char_array;
            } break;

            case 1: {
                short* temp_short_array = new short[records_to_read];

                for ( long counter = 0; counter < records_to_read; counter++ ) {
                    temp_short_array[counter] = short(input_array[counter]);
                }

                my_file->write((char*)temp_short_array, records_to_read * 2);

                delete[] temp_short_array;
            } break;

            case 2:
                my_file->write((char*)input_array, records_to_read * 4);
                break;

            case 12: {
                std::vector<half> temp_half_array(records_to_read);

                for ( long counter = 0; counter < records_to_read; counter++ ) {
                    temp_half_array[counter] = half(input_array[counter]);
                }

                my_file->write((char*)temp_half_array.data( ), records_to_read * 2);
                break;
            }

            default: {
                MyPrintfRed("Error: mode %i MRC files not currently supported\n", my_header.Mode( ));
                DEBUG_ABORT;
            } break;
        }
    }
}

MRCFile& MRCFile::operator=(const MRCFile& other_file) {
    *this = &other_file;
    return *this;
}

MRCFile& MRCFile::operator=(const MRCFile* other_file) {
    // Check for self assignment
    if ( this != other_file ) {
        my_file   = other_file->my_file;
        my_header = other_file->my_header;
        filename  = other_file->filename;

        rewrite_header_on_close                         = other_file->rewrite_header_on_close;
        max_number_of_seconds_to_wait_for_file_to_exist = other_file->max_number_of_seconds_to_wait_for_file_to_exist;

        do_nothing = other_file->do_nothing;
    }

    return *this;
}
