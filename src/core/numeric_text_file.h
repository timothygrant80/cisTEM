#ifndef __SRC_CORE_NUMERIC_TEXT_FILE_H__
#define __SRC_CORE_NUMERIC_TEXT_FILE_H__

#include "../constants/constants.h"

class NumericTextFile {

    long access_type;

    wxString            text_filename;
    wxFileInputStream*  input_file_stream{ };
    wxTextInputStream*  input_text_stream{ };
    wxFileOutputStream* output_file_stream{ };
    wxTextOutputStream* output_text_stream{ };
    bool                default_constructed{ };
    // In special cases (e.g. if the filename is /dev/null), we don't do anything
    bool file_is_not_dev_null{ };
    void Init( );

    inline bool LineIsACommentOrZeroLength(const wxString& current_line) const {
        return (current_line.StartsWith("#") || current_line.StartsWith("C") || current_line.Length( ) == 0);
    }

  public:
    NumericTextFile( );
    NumericTextFile(wxString Filename, long wanted_access_type, long wanted_records_per_line = 1);
    ~NumericTextFile( );

    // We don't want to allow copying or moving of this class
    NumericTextFile(const NumericTextFile&)            = delete;
    NumericTextFile& operator=(const NumericTextFile&) = delete;
    NumericTextFile(NumericTextFile&&)                 = delete;
    NumericTextFile& operator=(NumericTextFile&&)      = delete;

    // data

    int number_of_lines;
    int records_per_line;

    // Methods

    void Open(wxString Filename, long wanted_access_type, long wanted_records_per_line = 1);
    void Close( );
    void Rewind( );
    void Flush( );

    wxString ReturnFilename( ) { return text_filename; }

    void ReadLine(float* data_array);

    template <typename T>
    void WriteLine(T* data_array);

    void WriteCommentLine(const char* format, ...);
};

#endif // __SRC_CORE_NUMERIC_TEXT_FILE_H__