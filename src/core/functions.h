#include "defines.h"
#include "../constants/constants.h"

void swapbytes(unsigned char* v, size_t n);
void swapbytes(size_t size, unsigned char* v, size_t n);

bool GetMRCDetails(const char* filename, int& x_size, int& y_size, int& number_of_images);

inline void ZeroBoolArray(bool* array_to_zero, int size_of_array) {
    for ( int counter = 0; counter < size_of_array; counter++ ) {
        array_to_zero[counter] = false;
    }
}

int sizeCanBe4BitK2SuperRes(int nx, int ny); // provided by David Mastronarde

inline void ZeroIntArray(int* array_to_zero, int size_of_array) {
    for ( int counter = 0; counter < size_of_array; counter++ ) {
        array_to_zero[counter] = 0;
    }
}

inline void ZeroLongArray(long* array_to_zero, int size_of_array) {
    for ( int counter = 0; counter < size_of_array; counter++ ) {
        array_to_zero[counter] = 0;
    }
}

inline void ZeroFloatArray(float* array_to_zero, int size_of_array) {
    for ( int counter = 0; counter < size_of_array; counter++ ) {
        array_to_zero[counter] = 0.0;
    }
}

void FirstLastParticleForJob(long& first_particle, long& last_particle, long number_of_particles, int current_job_number, int number_of_jobs);

int ReturnSafeBinnedBoxSize(int original_box_size, float bin_factor);

float ReturnMagDistortionCorrectedPixelSize(float original_pixel_size, float major_axis_scale, float minor_axis_scale);

wxString ReturnSocketErrorText(wxSocketBase* socket_to_check);

bool     SendwxStringToSocket(wxString* string_to_send, wxSocketBase* socket);
wxString ReceivewxStringFromSocket(wxSocketBase* socket, bool& receive_worked);

bool SendTemplateMatchingResultToSocket(wxSocketBase* socket, int& image_number, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes);
bool ReceiveTemplateMatchingResultFromSocket(wxSocketBase* socket, int& image_number, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes);

inline bool WriteToSocket(wxSocketBase* socket, const void* buffer, wxUint32 nbytes, bool die_on_error = false, wxString identification_code = "NO_IDENT", wxString sender_details = "NO_DETAILS") {
    if ( socket != NULL ) {
        if ( socket->IsOk( ) == true && socket->IsConnected( ) == true ) {

#ifdef DEBUG
            if ( socket->GetFlags( ) != (SOCKET_FLAGS) ) {
                MyPrintWithDetails("Wait all / block flag not set!");
                DEBUG_ABORT;
            }
#endif

#ifdef RIGOROUS_SOCKET_CHECK
            // if we are doing intensive socket checking, use the identification code etc

            wxCharBuffer identification_string_buffer = identification_code.mb_str( );
            int          length_of_string             = identification_string_buffer.length( );

            // send the length of the string, followed by the string

            if ( socket->IsData( ) == false )
                socket->WaitForWrite( );
            socket->Write(&length_of_string, sizeof(int));

            if ( socket->IsData( ) == false )
                socket->WaitForWrite( );
            socket->Write(identification_string_buffer.data( ), length_of_string);

            // send caller details..

            wxCharBuffer sender_details_string_buffer = sender_details.mb_str( );
            length_of_string                          = sender_details_string_buffer.length( );

            if ( socket->IsData( ) == false )
                socket->WaitForWrite( );
            socket->Write(&length_of_string, sizeof(int));

            if ( socket->IsData( ) == false )
                socket->WaitForWrite( );
            socket->Write(sender_details_string_buffer.data( ), length_of_string);

#endif

            if ( socket->IsData( ) == false )
                socket->WaitForWrite( );
            socket->Write(buffer, nbytes);

            if ( socket->LastWriteCount( ) != nbytes ) {
                MyDebugPrintWithDetails("Socket didn't write all bytes! (%u / %u)", socket->LastWriteCount( ), nbytes);
                return false;
            }
            if ( socket->Error( ) == true ) {
                MyDebugPrintWithDetails("Socket has an error (%s) ", ReturnSocketErrorText(socket));
                return false;
            }

            return true; // if we got here, should be ok.
        }
        else {
            return false;
        }
    }

    return false;
}

inline bool ReadFromSocket(wxSocketBase* socket, void* buffer, wxUint32 nbytes, bool die_on_error = false, wxString identification_code = "NO_IDENT", wxString receiver_details = "NO_DETAILS") {

    if ( socket != NULL ) {
        if ( socket->IsOk( ) == true && socket->IsConnected( ) == true ) {
#ifdef DEBUG
            if ( socket->GetFlags( ) != (SOCKET_FLAGS) ) {
                MyPrintWithDetails("Wait all / block flag not set!");
                DEBUG_ABORT
            }
#endif

#ifdef RIGOROUS_SOCKET_CHECK
            // if we are intensive checking use the identification code etc.

            int length_of_string;

            // receive the length of the string, followed by the string

            if ( socket->IsData( ) == false )
                socket->WaitForRead( );
            socket->Read(&length_of_string, sizeof(int));

            unsigned char* transfer_buffer = new unsigned char[length_of_string + 1]; // + 1 for the terminating null character;

            // setup a temp array to receive the string into.

            if ( socket->IsData( ) == false )
                socket->WaitForRead( );
            socket->Read(transfer_buffer, length_of_string);

            // add the null

            transfer_buffer[length_of_string] = 0;

            // make a wxstring from this buffer..

            wxString sent_identification_code(transfer_buffer);

            delete[] transfer_buffer;

            // get sender details..

            if ( socket->IsData( ) == false )
                socket->WaitForRead( );
            socket->Read(&length_of_string, sizeof(int));
            transfer_buffer = new unsigned char[length_of_string + 1];
            if ( socket->IsData( ) == false )
                socket->WaitForRead( );
            socket->Read(transfer_buffer, length_of_string);
            transfer_buffer[length_of_string] = 0;
            wxString sender_details(transfer_buffer);

            delete[] transfer_buffer;

            if ( sent_identification_code != identification_code ) {
                MyDebugPrint("\n\nERROR : Mismatched socket identification codes\nSender: %s, Expected: %s\n", sent_identification_code, identification_code);
                MyDebugPrint("Receiver at %s\n", receiver_details);
                MyDebugPrintWithDetails("Sender at %s\n\n", sender_details);
            }
            //	else wxPrintf("Ident ok - Sender: %s, Expected: %s\n", sent_identification_code, identification_code);
#endif

            //socket->SetTimeout(60);
            if ( socket->IsData( ) == false )
                socket->WaitForRead( );
            socket->Read(buffer, nbytes);

            if ( socket->LastReadCount( ) != nbytes ) {
                MyDebugPrintWithDetails("Socket didn't read all bytes! (%u / %u)", socket->LastReadCount( ), nbytes);
                return false;
            }
            if ( socket->Error( ) == true ) {
                MyDebugPrintWithDetails("Socket has an error (%s) ", ReturnSocketErrorText(socket));
                return false;
            }
        }

        return true;
    }
    else {
        return false;
    }

    return false;
}

/*
inline void WriteToSocket	(	wxSocketBase *socket, const void * 	buffer, wxUint32 nbytes, bool die_on_error = false,  wxString identification_code = "NO_IDENT", wxString sender_details = "NO_DETAILS" )
{

	bool should_abort = false;
	if (socket != NULL)
	{
		if (socket->IsOk() == true && socket->IsConnected() == true)
		{

#ifdef DEBUG
			//	socket->SetFlags(wxSOCKET_WAITALL | wxSOCKET_BLOCK);
			//if (socket->GetFlags() != (wxSOCKET_WAITALL) && socket->GetFlags() != (wxSOCKET_WAITALL | wxSOCKET_BLOCK)) 	{MyPrintWithDetails("Wait all flag not set!"); should_abort = true;}
			if (socket->GetFlags() != (SOCKET_FLAGS)) 	{MyPrintWithDetails("Wait all / block flag not set!"); should_abort = true;}
#endif


#ifdef RIGOROUS_SOCKET_CHECK
			// if we are doing intensive socket checking, use the identification code etc

			wxCharBuffer identification_string_buffer = identification_code.mb_str();
			int length_of_string = identification_string_buffer.length();

			// send the length of the string, followed by the string

			if (socket->IsData() == false) socket->WaitForWrite();
			socket->Write(&length_of_string, sizeof(int));

			if (socket->IsData() == false) socket->WaitForWrite();
			socket->Write(identification_string_buffer.data(), length_of_string);

			// send caller details..

			wxCharBuffer sender_details_string_buffer = sender_details.mb_str();
			length_of_string = sender_details_string_buffer.length();

			if (socket->IsData() == false) socket->WaitForWrite();
			socket->Write(&length_of_string, sizeof(int));

			if (socket->IsData() == false) socket->WaitForWrite();
			socket->Write(sender_details_string_buffer.data(), length_of_string);

#endif

			//socket->SetTimeout(60);
			if (socket->IsData() == false) socket->WaitForWrite();
			socket->Write(buffer, nbytes);

			int number_of_retries = 100;
			while (socket->LastWriteCount() == 0 && number_of_retries < 10)
			{
				wxMilliSleep(100);
				socket->WaitForWrite();
				socket->Write(buffer, nbytes);
				number_of_retries++;
			}

			if (socket->LastWriteCount() != nbytes) {MyDebugPrintWithDetails("Socket didn't write all bytes! (%u / %u) with %i retries ", socket->LastWriteCount(), nbytes, number_of_retries); should_abort = true;}
			if (socket->Error() == true) {MyDebugPrintWithDetails("Socket has an error (%s) ", ReturnSocketErrorText(socket)); should_abort = true;}

			if (should_abort == true)
			{
#ifdef DEBUG
				wxIPV4address peer_address;
				socket->GetPeer(peer_address);

				wxPrintf("Failed socket is connected to : %s, (%s)\n", peer_address.Hostname(), peer_address.IPAddress());
				socket->Destroy();
				socket = NULL;
				wxFAIL;
				DEBUG_ABORT;
#else
				exit(-1);
#endif
			}
		}
		else
		{
			if (die_on_error == true) DEBUG_ABORT;
		}
	}
}
*/

/*
inline void ReadFromSocket	(	wxSocketBase *socket, void * 	buffer, wxUint32 nbytes,  bool die_on_error = false, wxString identification_code = "NO_IDENT", wxString receiver_details = "NO_DETAILS" )
{
	bool should_abort = false;
	if (socket != NULL)
	{
		if (socket->IsOk() == true && socket->IsConnected() == true)
		{
#ifdef DEBUG
			//	socket->SetFlags(wxSOCKET_WAITALL | wxSOCKET_BLOCK);
			//if (socket->GetFlags() != (wxSOCKET_WAITALL) && socket->GetFlags() != (wxSOCKET_WAITALL | wxSOCKET_BLOCK)) 	{MyPrintWithDetails("Wait all flag not set!"); should_abort = true;}
			if (socket->GetFlags() != (SOCKET_FLAGS)) 	{MyPrintWithDetails("Wait all / block flag not set!"); should_abort = true;}
#endif


#ifdef RIGOROUS_SOCKET_CHECK
			// if we are intensive checking use the identification code etc.

			int length_of_string;

			// receive the length of the string, followed by the string

			if (socket->IsData() == false) socket->WaitForRead();
			socket->Read(&length_of_string, sizeof(int));

			unsigned char *transfer_buffer = new unsigned char[length_of_string + 1]; // + 1 for the terminating null character;

			// setup a temp array to receive the string into.

			if (socket->IsData() == false) socket->WaitForRead();
			socket->Read(transfer_buffer,length_of_string);

			// add the null

			transfer_buffer[length_of_string] = 0;

			// make a wxstring from this buffer..

			wxString sent_identification_code(transfer_buffer);

			delete [] transfer_buffer;

			// get sender details..

			if (socket->IsData() == false) socket->WaitForRead();
			socket->Read(&length_of_string, sizeof(int));
			transfer_buffer = new unsigned char[length_of_string + 1];
			if (socket->IsData() == false) socket->WaitForRead();
			socket->Read(transfer_buffer,length_of_string);
			transfer_buffer[length_of_string] = 0;
			wxString sender_details(transfer_buffer);

			delete [] transfer_buffer;

			if (sent_identification_code != identification_code)
			{
				wxPrintf("\n\nERROR : Mismatched socket identification codes\nSender: %s, Expected: %s\n", sent_identification_code, identification_code);
				wxPrintf("Receiver at %s\n", receiver_details);
				wxPrintf("Sender at %s\n\n", sender_details);
				DEBUG_ABORT;
			}
		//	else wxPrintf("Ident ok - Sender: %s, Expected: %s\n", sent_identification_code, identification_code);
#endif

			//socket->SetTimeout(60);
			if (socket->IsData() == false) socket->WaitForRead();
			socket->Read(buffer, nbytes);

			int number_of_retries = 100;
			while (socket->LastReadCount() == 0 && number_of_retries < 10)
			{
				wxMilliSleep(100);
				socket->WaitForRead();
				socket->Read(buffer, nbytes);
				number_of_retries++;
			}

			if (socket->LastReadCount() != nbytes) {MyDebugPrintWithDetails("Socket didn't read all bytes! (%u / %u) with %i retries ", socket->LastReadCount(), nbytes, number_of_retries); should_abort = true;}
			if (socket->Error() == true) {MyDebugPrintWithDetails("Socket has an error (%s) ", ReturnSocketErrorText(socket)); should_abort = true;}
			if (should_abort == true)
			{
#ifdef DEBUG
				wxIPV4address peer_address;
				socket->GetPeer(peer_address);

				wxPrintf("Failed socket is connected to : %s, (%s)\n", peer_address.Hostname(), peer_address.IPAddress());
				socket->Destroy();
				socket = NULL;
				wxFAIL;
				DEBUG_ABORT;
#else
				exit(-1);
#endif

			}
		}
		else
		{
			if (die_on_error == true) DEBUG_ABORT;
		}
	}
}
*/

inline void ZeroDoubleArray(double* array_to_zero, int size_of_array) {
    for ( int counter = 0; counter < size_of_array; counter++ ) {
        array_to_zero[counter] = 0.0;
    }
}

inline bool IsEven(int number_to_check) {
    if ( number_to_check % 2 == 0 )
        return true;
    else
        return false;
}

inline bool DoesFileExist(wxString filename) {
    std::ifstream file_to_check(filename.c_str( ));

    if ( file_to_check.is_open( ) )
        return true;
    return false;
}

inline bool DoesFileExistWithWait(wxString filename, int max_wait_time_in_seconds) {

    if ( ! DoesFileExist(filename) ) {
        for ( int wait_counter = 0; wait_counter < max_wait_time_in_seconds; wait_counter++ ) {
            wxSleep(1);
            if ( DoesFileExist(filename) )
                break;
        }
    }

    return DoesFileExist(filename);
}

// Function to check if x is power of 2
inline bool is_power_of_two(int n) {
    if ( n == 0 )
        return false;
    return (ceil(log2((float)n)) == floor(log2((float)n)));
}

inline float rad_2_deg(float radians) {
    return radians / (PIf / 180.);
}

inline float deg_2_rad(float degrees) {
    return degrees * PIf / 180.;
}

inline float clamp_angular_range_0_to_2pi(float angle, bool units_are_degrees = false) {
    // Clamps the angle to be in the range ( 0,+360 ] { exclusive, inclusive }
    if ( units_are_degrees ) {
        angle = fmodf(angle, 360.0f);
    }
    else {
        angle = fmodf(angle, 2.0f * PIf);
    }
    return angle;
}

inline float clamp_angular_range_negative_pi_to_pi(float angle, bool units_are_degrees = false) {
    // Clamps the angle to be in the range ( -180,+180 ] { exclusive, inclusive }
    if ( units_are_degrees ) {
        angle = fmodf(angle, 360.0f);
        if ( angle > 180.0f )
            angle -= 360.0f;
        if ( angle <= -180.0f )
            angle += 360.0f;
        ;
    }
    else {
        angle = fmodf(angle, 2.0f * PIf);
        if ( angle > PIf )
            angle -= 2.0f * PIf;
        if ( angle <= -PIf )
            angle += 2.0f * PIf;
    }
    return angle;
}

inline float sinc(float radians) {
    if ( radians < 0.00001 )
        return 1.0;
    if ( radians >= 0.01 )
        return sinf(radians) / radians;
    float temp_float = radians * radians;
    return 1.0 - temp_float / 6.0 + temp_float * temp_float / 120.0;
}

inline double myround(double a) {
    if ( a > 0 )
        return double(long(a + 0.5));
    else
        return double(long(a - 0.5));
}

inline float myround(float a) {
    if ( a > 0 )
        return float(int(a + 0.5));
    else
        return float(int(a - 0.5));
}

inline int myroundint(double a) {
    if ( a > 0 )
        return int(a + 0.5);
    else
        return int(a - 0.5);
}

inline int myroundint(float a) {
    if ( a > 0 )
        return int(a + 0.5);
    else
        return int(a - 0.5);
}

inline bool IsOdd(int number) {
    if ( (number & 1) == 0 )
        return false;
    else
        return true;
}

wxArrayString ReturnIPAddress( );
wxString      ReturnIPAddressFromSocket(wxSocketBase* socket);

inline float ReturnPhaseFromShift(float real_space_shift, float distance_from_origin, float dimension_size) {
    return real_space_shift * distance_from_origin * 2.0 * PI / dimension_size;
}

inline std::complex<float> Return3DPhaseFromIndividualDimensions(float phase_x, float phase_y, float phase_z) {
    float temp_phase = -phase_x - phase_y - phase_z;

    return cosf(temp_phase) + sinf(temp_phase) * I;
}

inline bool DoublesAreAlmostTheSame(double a, double b) {
    return (fabs(a - b) < 0.000001);
}

inline bool FloatsAreAlmostTheSame(float a, float b) {
    return (fabs(a - b) < 0.0001);
}

template <typename T>
inline bool RelativeErrorIsLessThanEpsilon(T reference, T test_value, T epsilon = 0.0001) {
    return (std::abs((reference - test_value) / reference) < epsilon);
}

inline bool InputIsATerminal( ) {
    return isatty(fileno(stdin));
};

inline bool OutputIsAtTerminal( ) {
    return isatty(fileno(stdout));
}

float CalculateAngularStep(float required_resolution, float radius_in_angstroms);

int ReturnClosestFactorizedUpper(int wanted_int, int largest_factor, bool enforce_even = false, int enforce_factor = 0);
int ReturnClosestFactorizedLower(int wanted_int, int largest_factor, bool enforce_even = false, int enforce_factor = 0);

bool FilenameExtensionMatches(std::string filename, std::string extension);

/*
 *
 * String manipulations
 *
 */
std::string FilenameReplaceExtension(std::string filename, std::string new_extension);
std::string FilenameAddSuffix(std::string filename, std::string suffix_to_add);
void        SplitFileIntoDirectoryAndFile(wxString& input_file, wxString& output_directory, wxString& output_file);

void Allocate2DFloatArray(float**& array, int dim1, int dim2);
void Deallocate2DFloatArray(float**& array, int dim1);

void CheckSocketForError(wxSocketBase* socket_to_check);

inline wxString BoolToYesNo(bool b) {
    return b ? "Yes" : "No";
}

long ReturnFileSizeInBytes(wxString filename);

inline float kDa_to_Angstrom3(float kilo_daltons) {
    return kilo_daltons * 1000.0 / 0.81;
}

inline bool IsAValidSymmetry(wxString* string_to_check) {
    long     junk;
    wxString buffer_string = *string_to_check;
    buffer_string.Trim( );
    buffer_string.Trim(false);

    if ( string_to_check->StartsWith("C") == false && string_to_check->StartsWith("c") == false && string_to_check->StartsWith("D") == false && string_to_check->StartsWith("d") == false && string_to_check->StartsWith("I") == false && string_to_check->StartsWith("i") == false && string_to_check->StartsWith("O") == false && string_to_check->StartsWith("o") == false && string_to_check->StartsWith("T") == false && string_to_check->StartsWith("t") == false )
        return false;
    if ( buffer_string.Mid(1).IsEmpty( ) == true && (buffer_string.StartsWith("I") == true || buffer_string.StartsWith("i") == true || buffer_string.StartsWith("T") == true || buffer_string.StartsWith("t") == true || buffer_string.StartsWith("O") == true || buffer_string.StartsWith("o") == true) )
        return true;

    return string_to_check->Mid(1).ToLong(&junk);
}

float ReturnSumOfLogP(float logp1, float logp2, float log_range);

int ReturnNumberofAsymmetricUnits(wxString symmetry);

inline float ConvertProjectionXYToThetaInDegrees(float x, float y) // assumes that max x,y is 1
{
    return rad_2_deg(asin(sqrtf(powf(x, 2) + powf(y, 2))));
}

inline float ConvertXYToPhiInDegrees(float x, float y) {
    if ( x == 0 && y == 0 )
        return 0;
    else
        return rad_2_deg(atan2f(y, x));
}

std::vector<size_t> rankSort(const float* v_temp, const size_t size);
std::vector<size_t> rankSort(const std::vector<float>& v_temp);

wxString StringFromSocketCode(unsigned char* socket_input_buffer);

int CheckNumberOfThreads(int number_of_threads);

// From David Mastronarde
int ReturnAppropriateNumberOfThreads(int optimalThreads);
int ReturnThreadNumberOfCurrentThread( );

double cisTEM_erfinv(double x);
double cisTEM_erfcinv(double x);

bool StripEnclosingSingleQuotesFromString(wxString& string_to_strip); // returns true if it was done, false if first and last characters are not '

void ActivateMKLDebugForNonIntelCPU( ); // will activate MKL debug environment variable if running on an AMD that supports high level features.  This works on my version on intel MKL - it is disabled in the released MKL (although setting it should not break anything)
