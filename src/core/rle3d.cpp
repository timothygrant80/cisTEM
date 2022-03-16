#include "core_headers.h"

rle3d_coord::rle3d_coord( ) {
    x_pos        = 0.;
    y_pos        = 0.;
    z_pos        = 0.;
    length       = 0.;
    group_number = 0.;
}

rle3d::rle3d( ) {
    allocated_coordinates = 0;
    number_of_coordinates = -1;
    number_of_groups      = 0;
}

rle3d::rle3d(Image& input3d) {
    allocated_coordinates = 0;
    number_of_coordinates = -1;
    number_of_groups      = 0;

    EncodeFrom(input3d);
}

rle3d::~rle3d( ) {
    if ( allocated_coordinates > 0 )
        delete[] rle_coordinates;
}

void rle3d::EncodeFrom(Image& input3d) {

    long pixel_counter;
    long current_length;

    int i, j, k;
    int inner_x;
    int inner_y;
    int inner_z;

    long inner_pixel_counter;

    // if we've already allocated stuff then clear everything..

    if ( allocated_coordinates > 0 ) {
        delete[] rle_coordinates;
        number_of_coordinates = -1;
        allocated_coordinates = 0;
    }

    // now check the 3d is binary

    MyDebugAssertTrue(input3d.IsBinary( ) == true, "Can only work on binary images")
            // Size info..

            x_size = input3d.logical_x_dimension;
    y_size         = input3d.logical_y_dimension;
    z_size         = input3d.logical_z_dimension;

    // loop through and encode..

    pixel_counter = 0;

    for ( k = 0; k < input3d.logical_z_dimension; k++ ) {
        for ( j = 0; j < input3d.logical_y_dimension; j++ ) {
            for ( i = 0; i < input3d.logical_x_dimension; i++ ) {
                if ( pixel_counter >= input3d.real_memory_allocated )
                    break;

                if ( input3d.real_values[pixel_counter] == 1 ) {
                    inner_x = i + 1;
                    inner_y = j;
                    inner_z = k;

                    inner_pixel_counter = pixel_counter + 1;
                    current_length      = 1;

                    while ( inner_pixel_counter <= input3d.real_memory_allocated ) {
                        if ( inner_x == input3d.logical_x_dimension ) {
                            inner_pixel_counter--;
                            inner_x--;
                            break;
                        }

                        if ( input3d.real_values[inner_pixel_counter] == 1 ) {
                            current_length++;
                            inner_pixel_counter++;
                            inner_x++;
                        }
                        else
                            break;
                    }

                    AddCoord(i, j, k, current_length);

                    // we can skip to the end now..

                    pixel_counter = inner_pixel_counter;
                    i             = inner_x;
                }

                pixel_counter++;
            }
            pixel_counter += input3d.padding_jump_value;
        }
    }
}

// No decode fo rnow..

/*void rle3d::DecodeTo(Tigris3d *output3d)
{

	long coord_counter;
	long pixel_counter;

	// are we allocated...

	if (allocated_coordinates == 0)
	{
		cout << endl << "Error: rle3d::DecodeTo :-" << endl;
		cout << "Decoding from an unallocated (blank) rle3d" << endl << endl;
		exit(-1);
	}

	// do we have some coordinates?

	if (number_of_coordinates == 0)
	{
		cout << endl << "Warning: rle3d::DecodeTo :-" << endl;
		cout << "Decoding from a blank rle3d, the image will be blank" << endl;
	}

	// check the sizing is correct..

	if (output3d->x_size != x_size || output3d->y_size != y_size || output3d->z_size != z_size)
	{
		cout << endl << "Error: rle3d::DecodeTo :-" << endl;
		cout << "The encoded 3d and the output 3d are different sizes!!" << endl << endl;
		exit(-1);
	}

	// clear the output 3d..

	output3d->SetToConstant(0.);

	// now loop through the co-ordinates and fill up the 3d..

	for (coord_counter = 0; coord_counter <= number_of_coordinates; coord_counter++)
	{
		for (pixel_counter = 0; pixel_counter < rle_coordinates[coord_counter].length; pixel_counter++)
		{
			output3d->SetPixelValue(rle_coordinates[coord_counter].x_pos + pixel_counter, rle_coordinates[coord_counter].y_pos, rle_coordinates[coord_counter].z_pos, 1.);

		}
	}

}*/

void rle3d::AddCoord(long x, long y, long z, long current_length) {

    // Do we have any co-ordinates allocated?

    if ( allocated_coordinates == 0 ) {
        rle_coordinates       = new rle3d_coord[100];
        allocated_coordinates = 100;
    }

    // Do we have enough co-ordinates allocated?

    number_of_coordinates++;

    if ( number_of_coordinates + 1 > allocated_coordinates ) {
        // we need to allocate more memory..

        rle3d_coord* buffer = new rle3d_coord[allocated_coordinates * 2];

        for ( long counter = 0; counter < allocated_coordinates; counter++ ) {
            buffer[counter].x_pos  = rle_coordinates[counter].x_pos;
            buffer[counter].y_pos  = rle_coordinates[counter].y_pos;
            buffer[counter].z_pos  = rle_coordinates[counter].z_pos;
            buffer[counter].length = rle_coordinates[counter].length;
        }

        allocated_coordinates *= 2;

        // ok we've allocated and copied so we just need to delete the old one and switch over

        delete[] rle_coordinates;
        rle_coordinates = buffer;
    }

    // ok so now we just add it..

    rle_coordinates[number_of_coordinates].x_pos  = x;
    rle_coordinates[number_of_coordinates].y_pos  = y;
    rle_coordinates[number_of_coordinates].z_pos  = z;
    rle_coordinates[number_of_coordinates].length = current_length;
}

void rle3d::Write(const char* filename) {
    // just dump it to disk.. i've done this for debugging so it isn't very sophisticated

    FILE* output_file;

    output_file = fopen(filename, "w");
    if ( output_file == 0 ) {
        MyPrintWithDetails("error opening file");
        DEBUG_ABORT;
    }

    for ( long counter = 0; counter <= number_of_coordinates; counter++ ) {
        fprintf(output_file, "%d %d %d %d\n", int(rle_coordinates[counter].x_pos), int(rle_coordinates[counter].y_pos), int(rle_coordinates[counter].z_pos), int(rle_coordinates[counter].length));
    }

    fclose(output_file);
}

// this should group all connected rle's into the same group number..

void rle3d::GroupConnected( ) {
    long coord_counter;
    long reverse_coord_counter;
    long start_x;
    long finish_x;

    // are we allocated...

    MyDebugAssertTrue(allocated_coordinates > 0, "Grouping unallocated rle3d")
            //MyDebugAssertTrue(number_of_coordinates > 0, "Grouping from blank rle3d")

            // ok so lets go..

            for ( coord_counter = 0; coord_counter <= number_of_coordinates; coord_counter++ ) {
        if ( coord_counter == 0 ) {
            number_of_groups                            = 1;
            rle_coordinates[coord_counter].group_number = 1;
        }
        else {
            // what are my current start/finish coords..

            start_x  = rle_coordinates[coord_counter].x_pos;
            finish_x = rle_coordinates[coord_counter].x_pos + (rle_coordinates[coord_counter].length - 1);
            // does this coord, overlap with a previous coord..

            for ( reverse_coord_counter = coord_counter - 1; reverse_coord_counter >= 0; reverse_coord_counter-- ) {
                if ( rle_coordinates[coord_counter].z_pos - rle_coordinates[reverse_coord_counter].z_pos > 1 ) {
                    // if the current coord has an z value of 2 slices before or more we can stop the search..
                    break;
                }
                else if ( rle_coordinates[coord_counter].z_pos - rle_coordinates[reverse_coord_counter].z_pos == 0 ) {
                    // we are in the same slice..

                    if ( rle_coordinates[coord_counter].y_pos - rle_coordinates[reverse_coord_counter].y_pos == 1 ) {
                        // so we are in the same slice, 1 line previous do we overlap?

                        if ( start_x <= rle_coordinates[reverse_coord_counter].x_pos && finish_x >= rle_coordinates[reverse_coord_counter].x_pos ) {
                            // yes we do overlap.. sort out group numbers..
                            SameGroup(coord_counter, reverse_coord_counter);
                        }
                        else if ( start_x >= rle_coordinates[reverse_coord_counter].x_pos && start_x <= rle_coordinates[reverse_coord_counter].x_pos + (rle_coordinates[reverse_coord_counter].length - 1) ) {
                            // yes we do overlap.. sort out group numbers..
                            SameGroup(coord_counter, reverse_coord_counter);
                        }
                    }
                }
                else if ( rle_coordinates[coord_counter].z_pos - rle_coordinates[reverse_coord_counter].z_pos == 1 ) {
                    // we are in the previous slice..

                    if ( rle_coordinates[coord_counter].y_pos - rle_coordinates[reverse_coord_counter].y_pos == 0 ) {
                        // so we are in previous same slice, same line, do we overlap?

                        if ( start_x <= rle_coordinates[reverse_coord_counter].x_pos && finish_x >= rle_coordinates[reverse_coord_counter].x_pos ) {
                            // yes we do overlap.. sort out group numbers..
                            SameGroup(coord_counter, reverse_coord_counter);
                        }
                        else if ( start_x >= rle_coordinates[reverse_coord_counter].x_pos && start_x <= rle_coordinates[reverse_coord_counter].x_pos + (rle_coordinates[reverse_coord_counter].length - 1) ) {
                            // yes we do overlap.. sort out group numbers..
                            SameGroup(coord_counter, reverse_coord_counter);
                        }
                    }
                }
            }
        }

        // if we got here and haven't assigned a group number, we need to create a new group..

        if ( rle_coordinates[coord_counter].group_number == 0 ) {
            number_of_groups++;
            rle_coordinates[coord_counter].group_number = number_of_groups;
        }
    }
}

// This is the whole reason for this rle stuff.. I want to find all the individual connected
// blobs and save them back with the value of their group size.. you can thus threshold small
// or specifically sized things out..

void rle3d::ConnectedSizeDecodeTo(Image& output3d) {

    long group_counter;
    long coord_counter;
    long pixel_counter;
    long inner_pixel_counter;
    long start_address;

    int inner_x;

    // are we allocated...
    MyDebugAssertTrue(allocated_coordinates > 0, "Decoding from  unallocated rle3d")

            // check the sizing is correct..

            MyDebugAssertFalse(output3d.logical_x_dimension != x_size || output3d.logical_y_dimension != y_size || output3d.logical_z_dimension != z_size, "The encoded and output 3ds are different sizes");

    // clear the output 3d..

    output3d.SetToConstant(0.);

    // work out connected groups..

    GroupConnected( );

    long group_size[number_of_groups + 1];

    // clear group sizes..

    for ( group_counter = 0; group_counter <= number_of_groups; group_counter++ ) {
        group_size[group_counter] = 0;
    }

    // first of all - how many pixels are in all the groups group..

    for ( coord_counter = 0; coord_counter <= number_of_coordinates; coord_counter++ ) {
        group_size[rle_coordinates[coord_counter].group_number] += rle_coordinates[coord_counter].length;
    }

    // now decode..

    for ( coord_counter = 0; coord_counter <= number_of_coordinates; coord_counter++ ) {
        start_address       = output3d.ReturnReal1DAddressFromPhysicalCoord(rle_coordinates[coord_counter].x_pos, rle_coordinates[coord_counter].y_pos, rle_coordinates[coord_counter].z_pos);
        inner_pixel_counter = 0;
        inner_x             = rle_coordinates[coord_counter].x_pos;

        for ( pixel_counter = 0; pixel_counter < rle_coordinates[coord_counter].length; pixel_counter++ ) {

            MyDebugAssertTrue(start_address + inner_pixel_counter < output3d.real_memory_allocated, "Invalid array location")
                    output3d.real_values[start_address + inner_pixel_counter] = group_size[rle_coordinates[coord_counter].group_number];
            inner_pixel_counter++;
            inner_x++;

            if ( inner_x == output3d.logical_x_dimension ) {
                inner_x = 0;
                inner_pixel_counter += output3d.padding_jump_value;
            }
        }
    }
}
