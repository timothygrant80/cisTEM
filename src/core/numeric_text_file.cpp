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
NumericTextFile::NumericTextFile(wxString Filename, long wanted_access_type, long wanted_records_per_line) {
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
void NumericTextFile::Open(wxString Filename, long wanted_access_type, long wanted_records_per_line) {
    MyDebugAssertTrue(wanted_access_type == OPEN_TO_READ || wanted_access_type == OPEN_TO_WRITE || wanted_access_type == OPEN_TO_APPEND, "Invalid access type");

    access_type      = wanted_access_type;
    records_per_line = wanted_records_per_line;
    text_filename    = Filename;

    file_is_not_dev_null = ! StartsWithDevNull(text_filename.ToStdString( ));
    if ( file_is_not_dev_null ) {

        switch ( access_type ) {
            case OPEN_TO_READ: {
                if ( input_file_stream ) {
                    if ( input_file_stream->GetFile( )->IsOpened( ) ) {
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
                    if ( output_file_stream->GetFile( )->IsOpened( ) ) {
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
    if ( input_text_stream )
        delete input_text_stream;
    if ( output_text_stream )
        delete output_text_stream;

    if ( output_file_stream ) {
        if ( output_file_stream->GetFile( )->IsOpened( ) )
            output_file_stream->GetFile( )->Close( );
        delete output_file_stream;
    }

    if ( input_file_stream ) {
        if ( input_file_stream->GetFile( )->IsOpened( ) )
            input_file_stream->GetFile( )->Close( );
        delete input_file_stream;
    }

    input_file_stream  = nullptr;
    input_text_stream  = nullptr;
    output_file_stream = nullptr;
    output_text_stream = nullptr;
}

// private, only called form Open which has asserts there.
void NumericTextFile::Init( ) {
    if ( file_is_not_dev_null ) {
        if ( access_type == OPEN_TO_READ ) {
            wxString current_line;
            wxString token;
            double   temp_double;
            int      current_records_per_line;
            // When reading, we ignore the records per line and get this info from the file.
            records_per_line          = -1;
            bool records_per_line_set = false;

            input_file_stream = new wxFileInputStream(text_filename);
            input_text_stream = new wxTextInputStream(*input_file_stream);

            if ( ! input_file_stream->IsOk( ) ) {
                MyPrintWithDetails("Attempt to access %s for reading failed\n", text_filename);
                DEBUG_ABORT;
            }

            // work out the records per line and how many lines

            number_of_lines = 0;

            while ( ! input_file_stream->Eof( ) ) {
                current_line = input_text_stream->ReadLine( );
                current_line.Trim(false);

                if ( ! LineIsACommentOrZeroLength(current_line) ) {
                    number_of_lines++;
                    wxStringTokenizer tokenizer(current_line);

                    current_records_per_line = 0;

                    while ( tokenizer.HasMoreTokens( ) ) {
                        token = tokenizer.GetNextToken( );

                        if ( token.ToDouble(&temp_double) ) {
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
                if ( wxRemoveFile(text_filename) == false ) {
                    MyDebugPrintWithDetails("Cannot remove already existing text file");
                }
            }

            output_file_stream = new wxFileOutputStream(text_filename);
            output_text_stream = new wxTextOutputStream(*output_file_stream);
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
        MyDebugAssertTrue(access_type == OPEN_TO_READ ? (input_file_stream && input_text_stream) : output_file_stream != nullptr, "Rewind called on a file that is not open");
        if ( access_type == OPEN_TO_READ ) {
            delete input_file_stream;
            delete input_text_stream;

            input_file_stream = new wxFileInputStream(text_filename);
            input_text_stream = new wxTextInputStream(*input_file_stream);
        }
        else
            output_file_stream->GetFile( )->Seek(0);
    }
}

void NumericTextFile::Flush( ) {
    if ( file_is_not_dev_null ) {
        if ( access_type == OPEN_TO_READ )
            input_file_stream->GetFile( )->Flush( );
        else
            output_file_stream->GetFile( )->Flush( );
    }
}

void NumericTextFile::ReadLine(float* data_array) {
    if ( file_is_not_dev_null ) {
        if ( access_type != OPEN_TO_READ ) {
            MyPrintWithDetails("Attempt to read from %s however access type is not READ\n", text_filename);
            DEBUG_ABORT;
        }

        wxString current_line;
        wxString token;
        double   temp_double;

        while ( ! input_file_stream->Eof( ) ) {
            current_line = input_text_stream->ReadLine( );
            current_line.Trim(false);

            if ( ! LineIsACommentOrZeroLength(current_line) )
                break;
        }

        wxStringTokenizer tokenizer(current_line);

        for ( int counter = 0; counter < records_per_line; counter++ ) {
            token = tokenizer.GetNextToken( );

            if ( token.ToDouble(&temp_double) == false ) {
                MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", token.ToUTF8( ).data( ), current_line.ToUTF8( ).data( ));
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
                output_text_stream->WriteString(wxString::Format("%14.5f", data_array[counter]));
            else if constexpr ( std::is_same_v<T, double> )
                output_text_stream->WriteDouble(data_array[counter]);
            else
                static_WriteLine_type_not_allowed( );

            if ( counter != records_per_line - 1 )
                output_text_stream->WriteString(" ");
        }

        output_text_stream->WriteString("\n");
    }
}

template void NumericTextFile::WriteLine<float>(float* data_array);
template void NumericTextFile::WriteLine<double>(double* data_array);

void NumericTextFile::WriteCommentLine(const char* format, ...) {
    if ( file_is_not_dev_null ) {
        va_list args;
        va_start(args, format);

        wxString comment_string;
        wxString buffer;

        comment_string.PrintfV(format, args);

        buffer = comment_string;
        buffer.Trim(false);

        if ( buffer.StartsWith("#") == false && buffer.StartsWith("C") == false ) {
            comment_string = "# " + comment_string;
        }

        output_text_stream->WriteString(comment_string);

        if ( comment_string.EndsWith("\n") == false )
            output_text_stream->WriteString("\n");

        va_end(args);
    }
}
