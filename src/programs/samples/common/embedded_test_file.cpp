#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#include "../../../core/core_headers.h"
#endif

#include "helper_functions.h"
#include "embedded_test_file.h"

EmbeddedTestFile::EmbeddedTestFile(wxString path, const unsigned char* dataArray, long length) {

    try {
        //wxPrintf("Size of embbeded file: %s\n", std::to_string(length));
        WriteEmbeddedArray(path, dataArray, length);
        filePath = path;
    } catch ( ... ) {
        wxPrintf("Failed writing embbeded file: %s\n", path);
    }
}

void EmbeddedTestFile::WriteEmbeddedArray(const char*          filename,
                                          const unsigned char* array,
                                          long                 length) {

    FILE* output_file = NULL;
    wxPrintf("  %s\n", filename);
    output_file = fopen(filename, "wb+");

    if ( output_file == NULL ) {
        wxPrintf(ANSI_COLOR_RED "\n\nError: Can't open output file %s.\n",
                 filename);
        wxPrintf(ANSI_COLOR_RESET "\n\nError: Can't open output file %s.\n",
                 filename);
        DEBUG_ABORT;
    }

    fwrite(array, sizeof(unsigned char), length, output_file);

    fclose(output_file);
}
