#ifndef SRC_PROGRAMS_SAMPLES_COMMON_EMBEDDED_TEST_FILE_H_
#define SRC_PROGRAMS_SAMPLES_COMMON_EMBEDDED_TEST_FILE_H_

class EmbeddedTestFile : public TestFile 
{
  public:

    EmbeddedTestFile(wxString path, const unsigned char *dataArray, long length);

  private:

    void WriteEmbeddedArray(const char *filename, const unsigned char *array, long length);
};

#endif
