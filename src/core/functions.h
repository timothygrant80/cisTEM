#include "defines.h"
#include "non_wx_functions.h"
#include "socket_communication_utils/tcp_socket.h"

void swapbytes(unsigned char* v, size_t n);
void swapbytes(size_t size, unsigned char* v, size_t n);

bool GetMRCDetails(const char* filename, int& x_size, int& y_size, int& number_of_images);

int  sizeCanBe4BitK2SuperRes(int nx, int ny); // provided by David Mastronarde
void FirstLastParticleForJob(long& first_particle, long& last_particle, long number_of_particles, int current_job_number, int number_of_jobs);

int ReturnSafeBinnedBoxSize(int original_box_size, float bin_factor);

float ReturnMagDistortionCorrectedPixelSize(float original_pixel_size, float major_axis_scale, float minor_axis_scale);

std::string ReturnSocketErrorText(TcpSocket* socket_to_check);

// A string on the wire is a 4-byte length followed by the bytes (no terminator).
bool        SendStringToSocket(const std::string& string_to_send, TcpSocket* socket);
std::string ReceiveStringFromSocket(TcpSocket* socket, bool& receive_worked);

bool SendTemplateMatchingResultToSocket(TcpSocket* socket, int& image_number, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes);
bool ReceiveTemplateMatchingResultFromSocket(TcpSocket* socket, int& image_number, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes);

/**
 * Write the whole buffer to a connected socket. Returns false (after a debug message) if the socket
 * is null, not connected, or the transfer fails; the socket is then marked disconnected and the
 * monitor thread reports it through HandleSocketDisconnect. With RIGOROUS_SOCKET_CHECK defined the
 * identification code and sender details are transmitted first and checked by ReadFromSocket.
 */
inline bool WriteToSocket(TcpSocket* socket, const void* buffer, size_t nbytes, bool die_on_error = false, std::string identification_code = "NO_IDENT", std::string sender_details = "NO_DETAILS") {
    if ( socket == NULL || ! socket->IsConnected( ) )
        return false;

#ifdef RIGOROUS_SOCKET_CHECK
    std::string identification_string = identification_code;
    std::string sender_details_string = sender_details;
    int         length_of_string      = int(identification_string.size( ));
    if ( ! socket->Write(&length_of_string, sizeof(int)) || ! socket->Write(identification_string.data( ), length_of_string) )
        return false;
    length_of_string = int(sender_details_string.size( ));
    if ( ! socket->Write(&length_of_string, sizeof(int)) || ! socket->Write(sender_details_string.data( ), length_of_string) )
        return false;
#endif

    if ( ! socket->Write(buffer, nbytes) ) {
        MyDebugPrintWithDetails("Socket write of %zu bytes failed (%s) ", nbytes, ReturnSocketErrorText(socket));
        return false;
    }
    return true;
}

/**
 * Read exactly nbytes from a connected socket. Returns false if the socket is null, not connected,
 * the peer closed, or the transfer fails.
 */
inline bool ReadFromSocket(TcpSocket* socket, void* buffer, size_t nbytes, bool die_on_error = false, std::string identification_code = "NO_IDENT", std::string receiver_details = "NO_DETAILS") {
    if ( socket == NULL || ! socket->IsConnected( ) )
        return false;

#ifdef RIGOROUS_SOCKET_CHECK
    int length_of_string;
    if ( ! socket->Read(&length_of_string, sizeof(int)) )
        return false;
    std::string sent_identification_code(size_t(length_of_string), '\0');
    if ( length_of_string > 0 && ! socket->Read(&sent_identification_code[0], length_of_string) )
        return false;
    if ( ! socket->Read(&length_of_string, sizeof(int)) )
        return false;
    std::string sender_details(size_t(length_of_string), '\0');
    if ( length_of_string > 0 && ! socket->Read(&sender_details[0], length_of_string) )
        return false;

    if ( sent_identification_code != identification_code ) {
        MyDebugPrint("\n\nERROR : Mismatched socket identification codes\nSender: %s, Expected: %s\n", sent_identification_code, identification_code);
        MyDebugPrint("Receiver at %s\n", receiver_details);
        MyDebugPrintWithDetails("Sender at %s\n\n", sender_details);
    }
#endif

    if ( ! socket->Read(buffer, nbytes) ) {
        MyDebugPrintWithDetails("Socket read of %zu bytes failed (%s) ", nbytes, ReturnSocketErrorText(socket));
        return false;
    }
    return true;
}

inline bool DoesFileExist(std::string filename) {
    std::ifstream file_to_check(filename.c_str( ));

    if ( file_to_check.is_open( ) )
        return true;
    return false;
}

inline bool DoesFileExistWithWait(std::string filename, int max_wait_time_in_seconds) {

    if ( ! DoesFileExist(filename) ) {
        for ( int wait_counter = 0; wait_counter < max_wait_time_in_seconds; wait_counter++ ) {
            SleepForSeconds(1);
            if ( DoesFileExist(filename) )
                break;
        }
    }

    return DoesFileExist(filename);
}

std::vector<std::string> ReturnIPAddress( );
std::string      ReturnIPAddressFromSocket(TcpSocket* socket);

float CalculateAngularStep(float required_resolution, float radius_in_angstroms);

int  ReturnClosestFactorizedUpper(int wanted_int, int largest_factor, bool enforce_even = false, int enforce_factor = 0);
int  ReturnClosestFactorizedLower(int wanted_int, int largest_factor, bool enforce_even = false, int enforce_factor = 0);
void ReturnBestFourierBinnedSize(float& output_binning_factor, int& dx, int& dy, const int input_x_size, const int input_y_size);

bool FilenameExtensionMatches(std::string filename, std::string extension);

/*
 *
 * String manipulations
 *
 */
std::string FilenameReplaceExtension(std::string filename, std::string new_extension);
std::string FilenameAddSuffix(std::string filename, std::string suffix_to_add);
void        SplitFileIntoDirectoryAndFile(std::string& input_file, std::string& output_directory, std::string& output_file);

void Allocate2DFloatArray(float**& array, int dim1, int dim2);
void Deallocate2DFloatArray(float**& array, int dim1);

void CheckSocketForError(TcpSocket* socket_to_check);

inline std::string BoolToYesNo(bool b) {
    return b ? "Yes" : "No";
}

long ReturnFileSizeInBytes(std::string filename);

inline bool IsAValidSymmetry(std::string* string_to_check) {
    long        junk;
    std::string buffer_string = Trimmed(*string_to_check);

    if ( buffer_string.empty( ) )
        return false;

    char first_character = char(toupper(static_cast<unsigned char>(buffer_string[0])));
    if ( first_character != 'C' && first_character != 'D' && first_character != 'I' && first_character != 'O' && first_character != 'T' )
        return false;
    if ( buffer_string.length( ) == 1 && (first_character == 'I' || first_character == 'T' || first_character == 'O') )
        return true;

    return StringToLong(buffer_string.substr(1), junk);
}

float ReturnSumOfLogP(float logp1, float logp2, float log_range);

int ReturnNumberofAsymmetricUnits(std::string symmetry);

std::vector<size_t> rankSort(const float* v_temp, const size_t size);
std::vector<size_t> rankSort(const std::vector<float>& v_temp);

std::string StringFromSocketCode(unsigned char* socket_input_buffer);

int CheckNumberOfThreads(int number_of_threads);

// From David Mastronarde
int ReturnAppropriateNumberOfThreads(int optimalThreads);
int ReturnThreadNumberOfCurrentThread( );

double cisTEM_erfinv(double x);
double cisTEM_erfcinv(double x);

bool StripEnclosingSingleQuotesFromString(std::string& string_to_strip); // returns true if it was done, false if first and last characters are not '

void ActivateMKLDebugForNonIntelCPU( ); // will activate MKL debug environment variable if running on an AMD that supports high level features.  This works on my version on intel MKL - it is disabled in the released MKL (although setting it should not break anything)

inline bool InputIsATerminal( ) {
    return isatty(fileno(stdin));
};

inline bool OutputIsAtTerminal( ) {
    return isatty(fileno(stdout));
};