#include "core_headers.h"

bool GetMRCDetails(const char *filename, int &x_size, int &y_size, int &number_of_images)
{
	FILE * input;
	input = fopen(filename, "rb");

	long file_byte_size;
	long number_of_pixels;
	long bytes_per_pixel;
	long bytes_per_slice;


	int mode;
	int temp_int;

	int success;
	int bytes_in_extended_header;

	if (input == NULL) return false;
	else
	{
		fseek(input, 0L, SEEK_END);
		file_byte_size = ftell(input);

		if (file_byte_size < 1025)
		{
			fclose(input);
			return false;
		}

		fseek(input, 0L, SEEK_SET);

		// read in the image size and number of slices..

		success = fread(&temp_int, 4, 1, input);
		x_size = long(temp_int);
		success = fread(&temp_int, 4, 1, input);
		y_size = long(temp_int);
		success = fread(&temp_int, 4, 1, input);
		number_of_images = long(temp_int);
		number_of_pixels = x_size * y_size;
		success = fread(&temp_int, 4, 1, input);
		mode = temp_int;

		if (mode == 0) bytes_per_pixel = 1;
		else
		if (mode == 1) bytes_per_pixel = 2;
		else
		if (mode == 2) bytes_per_pixel = 4;
		else
		if (mode == 3) bytes_per_pixel = 4;
		else
		if (mode == 4) bytes_per_pixel = 8;
		else
		{
			fclose(input);
			return false;
		}

		bytes_per_slice = number_of_pixels * bytes_per_pixel;

		// now we need to know the number of bytes in the extended header...

		fseek(input, 92, SEEK_SET);

		success = fread(&temp_int, 4, 1, input);
		bytes_in_extended_header = temp_int;

	//	cout << "file size = " << file_byte_size << endl;
	//	cout << "Should be = " << bytes_per_slice * number_of_images + 1024 << endl;

		if (bytes_per_slice * number_of_images + 1024 + bytes_in_extended_header > file_byte_size)
		{
			fclose(input);
			return false;
		}

		if (x_size < 1 || y_size < 1 || number_of_images < 1)
		{
			fclose(input);
			return false;
		}

	}

	fclose(input);
	return true;

}

// Assumption is that turning off of events etc has been done, and that
// all other considerations have done.

void SendwxStringToSocket(wxString *string_to_send, wxSocketBase *socket)
{
	wxCharBuffer buffer = string_to_send->mb_str();
	int length_of_string = buffer.length();
	unsigned char *char_pointer;

	// send the length of the string, followed by the string

	char_pointer = (unsigned char*)&length_of_string;
	socket->WriteMsg(char_pointer, 4);
	socket->WriteMsg(buffer.data(), length_of_string);
}

wxString ReceivewxStringFromSocket(wxSocketBase *socket)
{

	int length_of_string;
	unsigned char *char_pointer;

	// receive the length of the string, followed by the string

	char_pointer = (unsigned char*)&length_of_string;
	socket->ReadMsg(char_pointer, 4);

	// setup a temp array to receive the string into.

	unsigned char *transfer_buffer = new unsigned char[length_of_string + 1]; // + 1 for the terminating null character;

	socket->ReadMsg(transfer_buffer, length_of_string);

	// add the null

	transfer_buffer[length_of_string] = 0;

	// make a wxstring from this buffer..

	wxString temp_string(transfer_buffer);

	// delete the buffer

	delete [] transfer_buffer;

	return temp_string;

}
