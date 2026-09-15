#include "core_headers.h"

/**
 * @brief Construct a new Numeric Text File:: Numeric Text File object
 * Disallowed before Apr-2024
 */
NumericTextFile::NumericTextFile( ) {
    file_is_not_dev_null = false;
}

/**
 * @brief Construct a new Numeric Text File:: Numeric Text File object and open the file
 * 
 * @param Filename may be /dev/null to avoid file operations (noop)
 * @param wanted_access_type OPEN_TO_READ, OPEN_TO_WRITE, OPEN_TO_APPEND
 * @param wanted_records_per_line expected to be equal for all lines, defaults to 1, ignored when reading and determined from file.
 */
NumericTextFile::NumericTextFile(std::string Filename, long wanted_access_type, long wanted_records_per_line) {
    Open(Filename, wanted_access_type, wanted_records_per_line);
}

NumericTextFile::~NumericTextFile( ) {
    Close( );
}

/**
 * @brief Open a file for reading, writing, or appending
 * 
 * @param Filename may be /dev/null to avoid file operations (noop)
 * @param wanted_access_type OPEN_TO_READ, OPEN_TO_WRITE, OPEN_TO_APPEND
 * @param wanted_records_per_line expected to be equal for all lines, defaults to 1, ignored when reading and determined from file.
 */
void NumericTextFile::Open(std::string Filename, long wanted_access_type, long wanted_records_per_line) {
    MyDebugAssertTrue(wanted_access_type == OPEN_TO_READ || wanted_access_type == OPEN_TO_WRITE || wanted_access_type == OPEN_TO_APPEND, "Invalid access type");

    access_type      = wanted_access_type;
    records_per_line = wanted_records_per_line;
    text_filename    = Filename;

    file_is_not_dev_null = ! StartsWithDevNull(text_filename);
    if ( file_is_not_dev_null ) {

        switch ( access_type ) {
            case OPEN_TO_READ: {
                if ( input_file_stream ) {
                    if ( input_file_stream->is_open( ) ) {
                        MyPrintWithDetails("File already Open\n");
                        DEBUG_ABORT;
                    }
                }
                break;
            }
            case OPEN_TO_WRITE: {
                if ( records_per_line <= 0 ) {
                    MyPrintWithDetails("NumericTextFile asked to OPEN_TO_WRITE, but with erroneous records per line\n");
                    DEBUG_ABORT;
                }

                if ( output_file_stream ) {
                    if ( output_file_stream->is_open( ) ) {
                        MyPrintWithDetails("File already Open\n");
                        DEBUG_ABORT;
                    }
                }
                break;
            }
            case OPEN_TO_APPEND: {
                // FIXME: I don't think that open to append is handled.
                MyDebugAssertTrue(false, "OPEN_TO_APPEND not implemented");
                records_per_line = wanted_records_per_line;
                break;
            }
            default: {
                // This should probably be a run-time assert
                MyPrintWithDetails("Unknown access type!\n");
                DEBUG_ABORT;
                break;
            }
        }

        Init( );
    }
}

void NumericTextFile::Close( ) {
    if ( output_file_stream ) {
        if ( output_file_stream->is_open( ) )
            output_file_stream->close( );
        delete output_file_stream;
    }

    if ( input_file_stream ) {
        if ( input_file_stream->is_open( ) )
            input_file_stream->close( );
        delete input_file_stream;
    }

    input_file_stream  = nullptr;
    output_file_stream = nullptr;
}

// private, only called form Open which has asserts there.
void NumericTextFile::Init( ) {
    if ( file_is_not_dev_null ) {
        if ( access_type == OPEN_TO_READ ) {
            std::string current_line;
            std::string token;
            double   temp_double;
            int      current_records_per_line;
            // When reading, we ignore the records per line and get this info from the file.
            records_per_line          = -1;
            bool records_per_line_set = false;

            input_file_stream = new std::ifstream(text_filename);

            if ( ! input_file_stream->good( ) ) {
                MyPrintWithDetails("Attempt to access %s for reading failed\n", text_filename);
                DEBUG_ABORT;
            }

            // work out the records per line and how many lines

            number_of_lines = 0;

            while ( std::getline(*input_file_stream, current_line) ) {
                if ( ! current_line.empty( ) && current_line.back( ) == '\r' )
                    current_line.pop_back( );
                TrimLeft(current_line);

                if ( ! LineIsACommentOrZeroLength(current_line) ) {
                    number_of_lines++;
                    std::vector<std::string> tokenizer = SplitString(current_line);

                    current_records_per_line = 0;

                    for ( size_t token_counter = 0; token_counter < tokenizer.size( ); token_counter++ ) {
                        token = tokenizer[token_counter];

                        if ( StringToDouble(token, temp_double) ) {
                            current_records_per_line++;
                        }
                        else {
                            MyPrintWithDetails("Failed on the following record : %s\n", token);
                            DEBUG_ABORT;
                        }
                    }

                    // we want to check records_per_line for consistency..

                    if ( records_per_line_set ) {
                        if ( records_per_line != current_records_per_line ) {
                            MyPrintWithDetails("Different records per line found");
                            DEBUG_ABORT;
                        }
                    }
                    else {
                        records_per_line     = current_records_per_line;
                        records_per_line_set = true;
                    }
                }
            }

            // rewind the file..
            Rewind( );
        }
        else if ( access_type == OPEN_TO_WRITE ) {
            // check if the file exists..

            if ( DoesFileExist(text_filename) ) {
                if ( RemoveFile(text_filename) == false ) {
                    MyDebugPrintWithDetails("Cannot remove already existing text file");
                }
            }

            output_file_stream = new std::ofstream(text_filename);
        }
        else {
            MyPrintWithDetails("Unknown access type!\n");
            DEBUG_ABORT;
        }
    }
}

/**
 * @brief Reset the file pointer to the beginning of the file
 * 
 */
void NumericTextFile::Rewind( ) {
    if ( file_is_not_dev_null ) {
        MyDebugAssertTrue(access_type == OPEN_TO_READ ? input_file_stream != nullptr : output_file_stream != nullptr, "Rewind called on a file that is not open");
        if ( access_type == OPEN_TO_READ ) {
            delete input_file_stream;

            input_file_stream = new std::ifstream(text_filename);
        }
        else
            output_file_stream->seekp(0);
    }
}

void NumericTextFile::Flush( ) {
    if ( file_is_not_dev_null ) {
        if ( access_type == OPEN_TO_READ )
            input_file_stream->sync( );
        else
            output_file_stream->flush( );
    }
}

void NumericTextFile::ReadLine(float* data_array) {
    if ( file_is_not_dev_null ) {
        if ( access_type != OPEN_TO_READ ) {
            MyPrintWithDetails("Attempt to read from %s however access type is not READ\n", text_filename);
            DEBUG_ABORT;
        }

        std::string current_line;
        std::string token;
        double   temp_double;

        while ( std::getline(*input_file_stream, current_line) ) {
            if ( ! current_line.empty( ) && current_line.back( ) == '\r' )
                current_line.pop_back( );
            TrimLeft(current_line);

            if ( ! LineIsACommentOrZeroLength(current_line) )
                break;
        }

        std::vector<std::string> tokenizer = SplitString(current_line);

        for ( int counter = 0; counter < records_per_line; counter++ ) {
            token = size_t(counter) < tokenizer.size( ) ? tokenizer[counter] : std::string( );

            if ( StringToDouble(token, temp_double) == false ) {
                MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", token.c_str(), current_line.c_str());
                DEBUG_ABORT;
            }
            else {
                data_array[counter] = temp_double;
            }
        }
    }
}

template <bool flag = false>
inline void static_WriteLine_type_not_allowed( ) { static_assert(flag, "no NumericTextFile::WriteLine is only valid for float and double type!"); }

template <typename T>
void NumericTextFile::WriteLine(T* data_array) {

    if ( file_is_not_dev_null ) {
        if ( access_type != OPEN_TO_WRITE ) {
            MyPrintWithDetails("Attempt to read from %s however access type is not WRITE\n", text_filename);
            DEBUG_ABORT;
        }

        for ( int counter = 0; counter < records_per_line; counter++ ) {
            if constexpr ( std::is_same_v<T, float> )
                *output_file_stream << Format("%14.5f", data_array[counter]);
            else if constexpr ( std::is_same_v<T, double> )
                *output_file_stream << Format("%f", data_array[counter]);
            else
                static_WriteLine_type_not_allowed( );

            if ( counter != records_per_line - 1 )
                *output_file_stream << " ";
        }

        *output_file_stream << "\n";
    }
}

template void NumericTextFile::WriteLine<float>(float* data_array);
template void NumericTextFile::WriteLine<double>(double* data_array);

void NumericTextFile::WriteCommentLine(const char* format, ...) {
    if ( file_is_not_dev_null ) {
        va_list args;
        va_start(args, format);

        std::string comment_string;
        std::string buffer;

        comment_string = cistem::detail::VFormat(format, args);

        buffer = comment_string;
        TrimLeft(buffer);

        if ( StartsWith(buffer, "#") == false && StartsWith(buffer, "C") == false ) {
            comment_string = "# " + comment_string;
        }

        *output_file_stream << comment_string;

        if ( EndsWith(comment_string, "\n") == false )
            *output_file_stream << "\n";

        va_end(args);
    }
}
