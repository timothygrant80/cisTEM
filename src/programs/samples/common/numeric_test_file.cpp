
#ifdef ENABLEGPU
  #include "../../../gpu/gpu_core_headers.h"
#else
  #include "../../../core/core_headers.h"
#endif

#include "helper_functions.h"
#include "numeric_test_file.h"

NumericTestFile::NumericTestFile(wxString path) 
{
  
  const char *filename = path.mb_str();
  wxPrintf("  %s\n",filename);
  FILE *output_file = NULL;
  output_file = fopen(filename, "wb+");

  if (output_file == NULL) {
    wxPrintf(ANSI_COLOR_RED "\n\nError: Can't open output file %s.\n",
             filename);
    wxPrintf(ANSI_COLOR_RESET "\n\nError: Can't open output file %s.\n",
             filename);
    DEBUG_ABORT;
  }

  fprintf(output_file, "# This is comment, starting with #\n");
  fprintf(output_file, "C This is comment, starting with C\n");
  fprintf(output_file, "%f %f %f %f %f\n%f %f %f %f %f\n", 1.0, 2.0, 3.0, 4.0,
          5.0, 6.0, 7.1, 8.3, 9.4, 10.5);
  fprintf(output_file,
          "# The next line will be blank, but contain 5 spaces\n     \n");
  fprintf(output_file, "%f %f %f %f %f\n", 11.2, 12.7, 13.2, 14.1, 15.8);
  fprintf(
      output_file,
      "   # This comment line starts with #, but not at the first character\n");
  fprintf(
      output_file,
      "   C This comment line starts with C, but not at the first character\n");
  fprintf(output_file,
          "C The next line will have varying spaces between the datapoints\n");
  fprintf(output_file, "   %f %f   %f       %f          %f\n", 16.1245,
          17.81003, 18.5467, 19.7621, 20.11111);

  fclose(output_file);
  filePath = path;
}
