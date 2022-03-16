class AbstractImageFile {

  public:
    wxFileName filename;

    AbstractImageFile( );
    AbstractImageFile(std::string filename, bool overwrite = false);
    ~AbstractImageFile( );

    virtual int   ReturnXSize( )          = 0;
    virtual int   ReturnYSize( )          = 0;
    virtual int   ReturnZSize( )          = 0;
    virtual int   ReturnNumberOfSlices( ) = 0;
    virtual float ReturnPixelSize( )      = 0;

    virtual bool IsOpen( ) = 0;

    virtual bool OpenFile(std::string filename, bool overwrite = false, bool wait_for_file_to_exist = false, bool check_only_the_first_image = false, int eer_super_res_factor = 1, int eer_frames_per_image = 0) = 0; // Return true if everything about the file looks OK
    virtual void CloseFile( )                                                                                                                                                                                     = 0;

    virtual void ReadSliceFromDisk(int slice_number, float* output_array)                = 0;
    virtual void ReadSlicesFromDisk(int start_slice, int end_slice, float* output_array) = 0;

    virtual void WriteSliceToDisk(int slice_number, float* input_array)                = 0;
    virtual void WriteSlicesToDisk(int start_slice, int end_slice, float* input_array) = 0;

    virtual void PrintInfo( ) = 0;
};
