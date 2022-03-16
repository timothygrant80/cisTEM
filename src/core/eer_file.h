class EerFile : public AbstractImageFile {
  private:
    TIFF*               tif;
    FILE*               fh;
    int                 logical_dimension_x;
    int                 logical_dimension_y;
    int                 number_of_images;
    int                 bits_per_rle;
    int                 number_of_eer_frames;
    int                 number_of_eer_frames_per_image;
    int                 super_res_factor;
    unsigned long long* frame_starts;
    unsigned long long* frame_sizes;
    unsigned char*      buf;
    //unsigned int * ion_of_each_frame;
    unsigned long long file_size_bytes;
    unsigned long long frame_size_bits;

    float pixel_size;

    void ReadWholeFileIntoBuffer( );
    void DecodeToFloatArray(int start_eer_frame, int finish_eer_frame, float* output_array);
    bool ReadLogicalDimensionsFromDisk(bool check_only_the_first_image = false);

  public:
    EerFile( );
    EerFile(std::string wanted_filename, bool overwrite = false);
    ~EerFile( );

    inline int ReturnXSize( ) { return logical_dimension_x * super_res_factor; };

    inline int ReturnYSize( ) { return logical_dimension_y * super_res_factor; };

    inline int ReturnZSize( ) { return number_of_images; };

    inline int ReturnNumberOfSlices( ) { return number_of_images; };

    inline float ReturnPixelSize( ) { return pixel_size; };

    inline bool IsOpen( ) {
        if ( tif ) {
            return true;
        }
        else {
            return false;
        }
    };

    bool OpenFile(std::string filename, bool overwrite = false, bool wait_for_file_to_exist = false, bool check_only_the_first_image = false, int eer_super_res_factor = 1, int eer_frames_per_image = 0); // we default eer_frames_per_image to zero so that it'll be obvious if the user forgot to set it
    void CloseFile( );

    void ReadSliceFromDisk(int slice_number, float* output_array);
    void ReadSlicesFromDisk(int start_slice, int end_slice, float* output_array);

    void WriteSliceToDisk(int slice_number, float* input_array);
    void WriteSlicesToDisk(int start_slice, int end_slice, float* input_array);

    void PrintInfo( );
};
