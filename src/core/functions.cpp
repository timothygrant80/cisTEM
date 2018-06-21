#include "core_headers.h"

// for ip address
#include <stdio.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>

/**
@brief 	Swaps bytes.
@param	*v 			a pointer to the bytes.
@param 	n			number of bytes to swap.

	Byte swapping is done in place.

**/
void		swapbytes(unsigned char* v, size_t n)
{
	unsigned char	t;
	size_t	i;

	for ( i=0, n--; i<n; i++, n-- ) {
		t = v[i];
		v[i] = v[n];
		v[n] = t;
	}
}

/**
@brief 	Swaps bytes.
@param 	size		size of the block to be swapped.
@param 	*v 			a pointer to the bytes.
@param 	n			number of bytes to swap.

	Byte swapping is done in place.

**/
void		swapbytes(size_t size, unsigned char* v, size_t n)
{
	if ( n < 2 ) return;

	MyDebugPrintWithDetails("DEBUG swapbytes: size = %i n= %i\n",size,n);

	size_t	i;

	for ( i=0; i<size; i+=n, v+=n ) swapbytes(v, n);
}

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
		if (mode == 6) bytes_per_pixel = 2;
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
	WriteToSocket(socket, char_pointer, 4, true);
	WriteToSocket(socket, buffer.data(), length_of_string, true);
}

// This is here as when the binned resolution is too close to the refinement resolution or reconstruction
// resolution (could be 1 or other or both), then it seems to cause some problems - e.g. in Proteasome.
// I'm going to add a 1.3 factor here, but this can be changed.
//
// My plan is to use this function all the places it is calculated in the code, so that changing it here will change
// it everywhere.

int ReturnSafeBinnedBoxSize(int original_box_size, float bin_factor)
{
	//return (original_box_size / bin_factor) * 1.3f;
	return myroundint(float(original_box_size) / bin_factor);
}

wxString ReceivewxStringFromSocket(wxSocketBase *socket)
{

	int length_of_string;
	unsigned char *char_pointer;

	// receive the length of the string, followed by the string

	char_pointer = (unsigned char*)&length_of_string;
	ReadFromSocket(socket, char_pointer, 4, true);

	// setup a temp array to receive the string into.

	unsigned char *transfer_buffer = new unsigned char[length_of_string + 1]; // + 1 for the terminating null character;

	ReadFromSocket(socket, transfer_buffer, length_of_string, true);

	// add the null

	transfer_buffer[length_of_string] = 0;

	// make a wxstring from this buffer..

	wxString temp_string(transfer_buffer);

	// delete the buffer

	delete [] transfer_buffer;

	return temp_string;

}


wxArrayString ReturnIPAddress()
{

	  	struct ifaddrs * ifAddrStruct=NULL;
	    struct ifaddrs * ifa=NULL;
	    void * tmpAddrPtr=NULL;
        char addressBuffer[INET_ADDRSTRLEN + 1];

        wxString ip_address;
        wxArrayString all_ip_addresses;

        for (int counter = 0; counter <= INET_ADDRSTRLEN; counter++)
        {
        	addressBuffer[counter] = 0;
        }

        getifaddrs(&ifAddrStruct);

	    for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next)
	    {
	        if (!ifa->ifa_addr)
	        {
	            continue;
	        }
	        if (ifa->ifa_addr->sa_family == AF_INET)
	        { // check it is IP4
	            // is a valid IP4 Address
	            tmpAddrPtr=&((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;

	            inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);

	            ip_address = addressBuffer;
	            ip_address.Trim();
	            ip_address.Trim(false);

	            if (ip_address.Find("127.0.0.1") == wxNOT_FOUND) all_ip_addresses.Add(ip_address);

	            //if (memcmp(addressBuffer, "127.0.0.1", INET_ADDRSTRLEN) != 0) ip_address = addressBuffer;
	            //if (memcmp(addressBuffer, "127.0.0.1", INET_ADDRSTRLEN) != 0) all_ip_addresses.Add(addressBuffer);

	        }

	    }

	    all_ip_addresses.Add("127.0.0.1");

/*    wxPrintf("There are %li ip addresses\n", all_ip_addresses.GetCount());

	    for (int counter = 0; counter < all_ip_addresses.GetCount(); counter++)
	    {
	    	wxPrintf("%i = %s\n", counter, all_ip_addresses.Item(counter));
	    }*/

	    if (ifAddrStruct!=NULL) freeifaddrs(ifAddrStruct);


	    return all_ip_addresses;
}

wxString ReturnIPAddressFromSocket(wxSocketBase *socket)
{
	wxString ip_address;
	wxIPV4address my_address;

	socket->GetLocal(my_address);
	ip_address = my_address.IPAddress();

	// is this 127.0.0.1 - in which case it may cause trouble..
/*
	if (ip_address == "127.0.0.1")
	{
		ip_address = "";
		//ip_address = ReturnIPAddress(); // last chance to get a non loopback address
	}*/

	return ip_address;

}

/*
 *
 * String manipulations
 *
 */
std::string FilenameReplaceExtension(std::string filename, std::string new_extension)
{
	return filename.substr(0,filename.find_last_of('.')+1)+new_extension;
}
std::string FilenameAddSuffix(std::string filename, std::string suffix_to_add)
{
	return filename.substr(0,filename.find_last_of('.')) + suffix_to_add + filename.substr(filename.find_last_of('.'), filename.length() - 1);
}

int ReturnClosestFactorizedUpper(int wanted_int, int largest_factor, bool enforce_even)
{
	int number;
	int remainder = wanted_int;
	int factor;
	int temp_int;

	if (enforce_even)
	{
		temp_int = wanted_int;
		if (! IsEven(temp_int)) temp_int++;
		for (number = temp_int; number < 10000 * wanted_int; number += 2)
		{
			remainder = number;
			for (factor = 2; (factor <= largest_factor) && (remainder != 1); factor++)
			{
				temp_int = remainder % factor;
				while (temp_int == 0)
				{
					remainder /= factor;
					temp_int = remainder % factor;
				}
			}
			if (remainder == 1) break;
		}
	}
	else
	{
		for (number = wanted_int; number < 10000 * wanted_int; number++)
		{
			remainder = number;
			for (factor = 2; (factor <= largest_factor) && (remainder != 1); factor++)
			{
				temp_int = remainder % factor;
				while (temp_int == 0)
				{
					remainder /= factor;
					temp_int = remainder % factor;
				}
			}
			if (remainder == 1) break;
		}
	}
	return number;
}

int ReturnClosestFactorizedLower(int wanted_int, int largest_factor, bool enforce_even)
{
	int number;
	int remainder = wanted_int;
	int factor;
	int temp_int;

	if (enforce_even)
	{
		temp_int = wanted_int;
		if (! IsEven(temp_int)) temp_int--;
		for (number = temp_int; number >= 2; number -= 2)
		{
			remainder = number;
			for (factor = 2; (factor <= largest_factor) && (remainder != 1); factor++)
			{
				temp_int = remainder % factor;
				while (temp_int == 0)
				{
					remainder /= factor;
					temp_int = remainder % factor;
				}
			}
			if (remainder == 1) break;
		}
	}
	else
	{
		for (number = wanted_int; number >= 1; number--)
		{
			remainder = number;
			for (factor = 2; (factor <= largest_factor) && (remainder != 1); factor++)
			{
				temp_int = remainder % factor;
				while (temp_int == 0)
				{
					remainder /= factor;
					temp_int = remainder % factor;
				}
			}
			if (remainder == 1) break;
		}
	}
	return number;
}

float CalculateAngularStep(float required_resolution, float radius_in_angstroms)
{
	return 360.0 * required_resolution / PI / radius_in_angstroms;
}

void Allocate2DFloatArray(float **&array, int dim1, int dim2)
{
	array = new float* [dim1];			// dynamic array of pointers to float
	for (int i = 0; i < dim1; ++i)
	{
		array[i] = new float[dim2];		// each i-th pointer is now pointing to dynamic array (size number_of_positions) of actual float values
	}
}

void Deallocate2DFloatArray(float **&array, int dim1)
{
	for (int i = 0; i < dim1; ++i)
	{
		delete [] array[i];				// each i-th pointer must be deleted first
	}
	delete [] array;					// now delete pointer array
}

long ReturnFileSizeInBytes(wxString filename)
{
	long size;
	std::ifstream filesize(filename, std::ifstream::ate | std::ifstream::binary);
	size = filesize.tellg();
	filesize.close();
	return size;
}

void CheckSocketForError(wxSocketBase *socket_to_check)
{
	if (socket_to_check->Error() == true)
	{
		wxPrintf("Socket Error : ");

		switch(socket_to_check->LastError())
		{
		    case wxSOCKET_NOERROR  :
		       wxPrintf("No Error!\n");
		       break;

		    case wxSOCKET_INVOP  :
		       wxPrintf("Invalid Operation.\n");
		       break;

		    case wxSOCKET_IOERR  :
		       wxPrintf("Input/Output error.\n");
		       break;

		    case wxSOCKET_INVADDR  :
		       wxPrintf("Invalid address passed to wxSocket.\n");
		       break;

		    case wxSOCKET_INVSOCK  :
		       wxPrintf("Invalid socket (uninitialized).\n");
		       break;

		    case wxSOCKET_NOHOST  :
				wxPrintf("No corresponding host.\n");
			break;

		    case wxSOCKET_INVPORT  :
				 wxPrintf("Invalid port.\n");
			break;

		    case wxSOCKET_WOULDBLOCK  :
				 wxPrintf("The socket is non-blocking and the operation would block.\n");
			break;

		    case wxSOCKET_TIMEDOUT  :
				 wxPrintf("The timeout for this operation expired.\n");
			break;

		    case wxSOCKET_MEMERR  :
				 wxPrintf("Memory exhausted.\n");
			break;
		    // you can have any number of case statements.
		    default :
		    	wxPrintf("unknown.\n");
		}
	}
}

wxString ReturnSocketErrorText(wxSocketBase *socket_to_check)
{
	if (socket_to_check->Error() == true)
	{
		switch(socket_to_check->LastError())
		{
		    case wxSOCKET_NOERROR  :
		       return "No Error!";
		       break;

		    case wxSOCKET_INVOP  :
		       return "Invalid Operation.";
		       break;

		    case wxSOCKET_IOERR  :
		       return "Input/Output error";
		       break;

		    case wxSOCKET_INVADDR  :
		       return "Invalid address passed to wxSocket.";
		       break;

		    case wxSOCKET_INVSOCK  :
		       return "Invalid socket (uninitialized).";
		       break;

		    case wxSOCKET_NOHOST  :
				return "No corresponding host.";
			break;

		    case wxSOCKET_INVPORT  :
				 return "Invalid port.";
			break;

		    case wxSOCKET_WOULDBLOCK  :
				 return ("The socket is non-blocking and the operation would block");
			break;

		    case wxSOCKET_TIMEDOUT  :
				 return "The timeout for this operation expired.";
			break;

		    case wxSOCKET_MEMERR  :
				 return "Memory exhausted.";
			break;
		    // you can have any number of case statements.
		    default :
		    	return("Unknown error.");
		}
	}
	else return "Error, not set.";

}


float ReturnSumOfLogP(float logp1, float logp2, float log_range = 20.0)
{
	float logp;

	if (logp2 < logp1 - log_range)
	{
		logp = logp1;
	}
	else
	if (logp1 < logp2 - log_range)
	{
		logp = logp2;
	}
	else
	{
		logp = logp1 + log(double(1.0) + exp(double(logp2) - double(logp1)));
	}

	return logp;
}

float ReturnMagDistortionCorrectedPixelSize(float original_pixel_size, float major_axis_scale, float minor_axis_scale)
{
	float average_scale = (major_axis_scale + minor_axis_scale) / 2.0;
	return original_pixel_size / average_scale;
}

void SplitFileIntoDirectoryAndFile(wxString &input_file, wxString &output_directory, wxString &output_file)
{
	wxFileName current_filename = input_file;
	wxArrayString directories = current_filename.GetDirs();
	output_directory = "/";

	for (int directory_counter = 0; directory_counter < directories.GetCount(); directory_counter++)
	{
		output_directory += directories.Item(directory_counter);
		output_directory += "/";
	}

	output_file = current_filename.GetFullName();
}

int ReturnNumberofAsymmetricUnits(wxString symmetry)
{
	wxString current_symmetry_string = symmetry;
	wxChar   symmetry_type;
	long     symmetry_number;

	current_symmetry_string = current_symmetry_string.Trim();
	current_symmetry_string = current_symmetry_string.Trim(false);

	MyDebugAssertTrue(current_symmetry_string.Length() > 0, "symmetry string is blank");
	symmetry_type = current_symmetry_string.Capitalize()[0];

	if (current_symmetry_string.Length() == 1)
	{
		symmetry_number = 0;
	}
	else
	{
		if (! current_symmetry_string.Mid(1).ToLong(&symmetry_number))
		{
			MyPrintWithDetails("Error: Invalid n after symmetry symbol: %s\n", current_symmetry_string.Mid(1));
			DEBUG_ABORT;
		}
	}

	if (symmetry_type == 'C')
	{
		return int(symmetry_number);
	}
	else
	if (symmetry_type == 'D')
	{
		return int(symmetry_number * 2);
	}
	else
	if (symmetry_type == 'T')
	{
		return 12;
	}
	else
	if (symmetry_type == 'O')
	{
		return 24;
	}
	else
	if (symmetry_type == 'I')
	{
		return 60;
	}

	// shouldn't get here

	return 1;

}

// Return a vector with the rank of the elements of the input array
std::vector<size_t> rankSort(const float* v_temp, const size_t size) {
    std::vector<std::pair<float, size_t> > v_sort(size);

    for (size_t i = 0U; i < size; ++i) {
        v_sort[i] = std::make_pair(v_temp[i], i);
    }

    sort(v_sort.begin(), v_sort.end());

    std::pair<double, size_t> rank;
    std::vector<size_t> result(size);

    for (size_t i = 0U; i < size; ++i) {
        if (v_sort[i].first != rank.first) {
            rank = std::make_pair(v_sort[i].first, i);
        }
        result[v_sort[i].second] = rank.second;
    }
    return result;
}
std::vector<size_t> rankSort(const std::vector<float>& v_temp) {
    std::vector<std::pair<float, size_t> > v_sort(v_temp.size());

    for (size_t i = 0U; i < v_sort.size(); ++i) {
        v_sort[i] = std::make_pair(v_temp[i], i);
    }

    sort(v_sort.begin(), v_sort.end());

    std::pair<double, size_t> rank;
    std::vector<size_t> result(v_temp.size());

    for (size_t i = 0U; i < v_sort.size(); ++i) {
        if (v_sort[i].first != rank.first) {
            rank = std::make_pair(v_sort[i].first, i);
        }
        result[v_sort[i].second] = rank.second;
    }
    return result;
}

wxString StringFromSocketCode(unsigned char *socket_input_buffer)
{
	if (memcmp(socket_input_buffer, socket_please_identify, 			SOCKET_CODE_SIZE) == 0) { return "socket_please_identify"; }
	if (memcmp(socket_input_buffer, socket_identification, 				SOCKET_CODE_SIZE) == 0) { return "socket_identification"; }
	if (memcmp(socket_input_buffer, socket_you_are_connected, 			SOCKET_CODE_SIZE) == 0) { return "socket_you_are_connected"; }
	if (memcmp(socket_input_buffer, socket_send_job_details,			SOCKET_CODE_SIZE) == 0) { return "socket_send_job_details"; }
	if (memcmp(socket_input_buffer, socket_ready_to_send_job_package, 	SOCKET_CODE_SIZE) == 0) { return "socket_ready_to_send_job_package"; }
	if (memcmp(socket_input_buffer, socket_send_job_package, 			SOCKET_CODE_SIZE) == 0) { return "socket_send_job_package"; }
	if (memcmp(socket_input_buffer, socket_you_are_the_master, 			SOCKET_CODE_SIZE) == 0) { return "socket_you_are_the_master"; }
	if (memcmp(socket_input_buffer, socket_you_are_a_slave, 			SOCKET_CODE_SIZE) == 0) { return "socket_you_are_a_slave"; }
	if (memcmp(socket_input_buffer, socket_send_next_job, 				SOCKET_CODE_SIZE) == 0) { return "socket_send_next_job"; }
	if (memcmp(socket_input_buffer, socket_time_to_die, 				SOCKET_CODE_SIZE) == 0) { return "socket_time_to_die"; }
	if (memcmp(socket_input_buffer, socket_ready_to_send_single_job, 	SOCKET_CODE_SIZE) == 0) { return "socket_ready_to_send_single_job"; }
	if (memcmp(socket_input_buffer, socket_send_single_job, 			SOCKET_CODE_SIZE) == 0) { return "socket_send_single_job"; }
	if (memcmp(socket_input_buffer, socket_i_have_an_error, 			SOCKET_CODE_SIZE) == 0) { return "socket_i_have_an_error"; }
	if (memcmp(socket_input_buffer, socket_i_have_info, 				SOCKET_CODE_SIZE) == 0) { return "socket_i_have_info"; }
	if (memcmp(socket_input_buffer, socket_job_finished, 				SOCKET_CODE_SIZE) == 0) { return "socket_job_finished"; }
	if (memcmp(socket_input_buffer, socket_number_of_connections,		SOCKET_CODE_SIZE) == 0) { return "socket_number_of_connections"; }
	if (memcmp(socket_input_buffer, socket_all_jobs_finished, 			SOCKET_CODE_SIZE) == 0) { return "socket_all_jobs_finished"; }
	if (memcmp(socket_input_buffer, socket_job_result, 					SOCKET_CODE_SIZE) == 0) { return "socket_job_result"; }
	if (memcmp(socket_input_buffer, socket_job_result_queue, 			SOCKET_CODE_SIZE) == 0) { return "socket_job_result_queue"; }
	if (memcmp(socket_input_buffer, socket_result_with_image_to_write, 	SOCKET_CODE_SIZE) == 0) { return "socket_result_with_image_to_write"; }
	return "socket code not recognized";
}


