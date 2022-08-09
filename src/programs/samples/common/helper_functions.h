
#ifndef SRC_PROGRAMS_SAMPLES_COMMON_HELPER_FUNCTIONS_H_
#define SRC_PROGRAMS_SAMPLES_COMMON_HELPER_FUNCTIONS_H_

#define SamplesTestResult(result) SamplesPrintResult(result, __LINE__);

void print2DArray(Image& image);

void PrintArray(float* p, int maxLoops = 10);

bool ProperCompareRealValues(Image& first_image, Image& second_image, float epsilon = 1e-5);

void SamplesPrintTestStartMessage(wxString message, bool bold = false);
void SamplesPrintUnderlined(wxString message);
void SamplesPrintBold(wxString message);

void SamplesPrintResult(bool result, int line);

void SamplesBeginPrint(const char* test_name);

void SamplesBeginTest(const char* test_name, bool& test_has_passed);

class TestFile {

  public:
    // default constructor
    virtual ~TestFile(void) {
        wxString tempString;
        // There is nothing to remove
        if ( filePath.IsNull( ) || filePath.IsEmpty( ) || ! filePath || filePath.Len( ) == 0 ) {
            return;
        }

        if ( ! filePath.IsNull( ) && ! filePath.IsEmpty( ) ) {

            tempString = "Deleting file " + filePath;
            SamplesBeginPrint(tempString.ToUTF8( ));
            const int result = remove(filePath.mb_str( ));

            if ( result == 0 )
                SamplesPrintResult(true, 1);
            else
                SamplesPrintResult(false, 1);
        }
    };

    wxString filePath;
};

class FileTracker {
  public:
    ~FileTracker( );
    void                   Cleanup( );
    std::vector<TestFile*> testFiles;
};

#endif
