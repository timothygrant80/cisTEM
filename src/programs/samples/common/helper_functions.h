
#ifndef SRC_PROGRAMS_SAMPLES_COMMON_HELPER_FUNCTIONS_H_
#define SRC_PROGRAMS_SAMPLES_COMMON_HELPER_FUNCTIONS_H_

#define SamplesTestResult(result) SamplesPrintResult(result, __LINE__);
#define SamplesTestResultCanFail(result) SamplesPrintResultCanFail(result, __LINE__);

class Image;

void print2DArray(Image& image);

void PrintArray(float* p, int maxLoops = 10);

bool CompareRealValues(Image& first_image, Image& second_image, float minimum_ccc = 0.999f, float mask_radius = 0.f);
bool CompareComplexValues(Image& first_image, Image& second_image, float minimum_ccc = 0.999f, float mask_radius = 0.f);

Image GetAbsOfFourierTransformAsRealImage(Image& input_image);

void SamplesPrintTestStartMessage(std::string message, bool bold = false);

inline void SamplesPrintEndMessage( ) {
    Printf("\n");
}

void SamplesPrintUnderlined(std::string message);
void SamplesPrintBold(std::string message);

void SamplesPrintResult(bool result, int line);
void SamplesPrintResultCanFail(bool passed, int line);

void SamplesBeginPrint(const char* test_name);

void SamplesBeginTest(const char* test_name, bool& test_has_passed);

class TestFile {

  public:
    // default constructor
    virtual ~TestFile(void) {
        std::string tempString;
        // There is nothing to remove
        if ( filePath.empty( ) ) {
            return;
        }

        if ( ! filePath.empty( ) ) {

            tempString = "\nDeleting file " + filePath;
            SamplesBeginPrint(tempString.c_str());
            const int result = remove(filePath.c_str());

            if ( result == 0 )
                SamplesPrintResult(true, 1);
            else
                SamplesPrintResult(false, 1);
        }
    };

    std::string filePath;
};

class FileTracker {
  public:
    ~FileTracker( );
    void                   Cleanup( );
    std::vector<TestFile*> testFiles;
};

#endif
