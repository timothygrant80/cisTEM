#ifndef SRC_PROGRAMS_SAMPLES_SAMPLES_FUNCTIONAL_TESTING_H_
#define SRC_PROGRAMS_SAMPLES_SAMPLES_FUNCTIONAL_TESTING_H_

#define FailTest                                \
    {                                           \
        if ( test_has_passed == true )          \
            PrintResultWorker(false, __LINE__); \
        test_has_passed = false;                \
    }

class
        SamplesTestingApp : public MyApp {

  public:
    // SamplesTestingApp();
    // ~SamplesTestingApp();
    void WriteFiles( );
    void TestResultWorker(bool passed, int line);

    bool DoCalculation( );
    void DoInteractiveUserInput( );
    void ProgramSpecificInit( );

    void MyInteractiveProgramCleanup( ) { file_tracker.Cleanup( ); };

    wxString temp_directory;
    wxString hiv_image_80x80x1_filename;
    wxString hiv_images_80x80x10_filename;
    wxString sine_wave_128x128x1_filename;
    wxString numeric_text_filename;

    FileTracker file_tracker;
};

#endif
