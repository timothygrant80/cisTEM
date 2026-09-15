#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#include "../../../core/core_headers.h"
#endif

#include "helper_functions.h"
#include "embedded_test_file.h"

EmbeddedTestFile::EmbeddedTestFile(std::string path, const unsigned char* dataArray, long length) {

    try {
        //Printf("Size of embbeded file: %s\n", std::to_string(length));
        WriteEmbeddedArray(path.c_str( ), dataArray, length);
        filePath = path;
    } catch ( ... ) {
        Printf("Failed writing embbeded file: %s\n", path);
    }
}

void EmbeddedTestFile::WriteEmbeddedArray(const char*          filename,
                                          const unsigned char* array,
                                          long                 length) {

    FILE* output_file = NULL;
    Printf("  %s\n", filename);
    output_file = fopen(filename, "wb+");

    if ( output_file == NULL ) {
        Printf(ANSI_COLOR_RED "\n\nError: Can't open output file %s.\n",
                 filename);
        Printf(ANSI_COLOR_RESET "\n\nError: Can't open output file %s.\n",
                 filename);
        DEBUG_ABORT;
    }

    fwrite(array, sizeof(unsigned char), length, output_file);

    fclose(output_file);
}
