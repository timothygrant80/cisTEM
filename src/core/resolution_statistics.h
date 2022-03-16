/*  \brief  ResolutionStatistics class */

class NumericTextFile;

class ResolutionStatistics {

  public:
    Curve FSC;
    Curve part_FSC;
    Curve part_SSNR;
    Curve rec_SSNR;

    float pixel_size;
    int   number_of_bins;
    int   number_of_bins_extended; // Extend table to include corners in 3D Fourier space

    ResolutionStatistics( );
    ResolutionStatistics(float wanted_pixel_size, int box_size = 0);
    ResolutionStatistics(const ResolutionStatistics& other_statistics); // copy constructor
    //	~ResolutionStatistics();

    ResolutionStatistics& operator=(const ResolutionStatistics& t);
    ResolutionStatistics& operator=(const ResolutionStatistics* t);

    void ResampleFrom(ResolutionStatistics& other_statistics, int wanted_number_of_bins = 0);
    void CopyFrom(ResolutionStatistics& other_statistics, int wanted_number_of_bins = 0);
    void CopyParticleSSNR(ResolutionStatistics& other_statistics, int wanted_number_of_bins = 0);
    void ResampleParticleSSNR(ResolutionStatistics& other_statistics, int wanted_number_of_bins = 0);
    void Init(float wanted_pixel_size, int box_size = 0);
    void NormalizeVolumeWithParticleSSNR(Image& reconstructed_volume);
    void CalculateFSC(Image& reconstructed_volume_1, Image& reconstructed_volume_2, bool smooth_curve = false);
    void CalculateParticleFSCandSSNR(float mask_volume_in_voxels, float molecular_mass_in_kDa);
    void CalculateParticleSSNR(Image& image_reconstruction, float* ctf_reconstruction, float wanted_mask_volume_fraction = 1.0f);
    void RestrainParticleSSNR(float low_resolution_limit = FLT_MAX);
    void ZeroToResolution(float resolution_limit);
    void PrintStatistics( );
    void WriteStatisticsToFile(NumericTextFile& output_statistics_file, float pssnr_division_factor = 1.0f);
    void WriteStatisticsToFloatArray(float* float_array, int wanted_class);
    void ReadStatisticsFromFile(wxString input_file);
    void GenerateDefaultStatistics(float molecular_mass_in_kDa);

    int   ReturnResolutionShellNumber(float wanted_resolution);
    float ReturnResolutionNShellsBefore(float wanted_resolution, int number_of_shells);
    float ReturnResolutionNShellsAfter(float wanted_resolution, int number_of_shells);
    float ReturnEstimatedResolution(bool use_part_fsc = true);
    float Return0p8Resolution(bool use_part_fsc = true);
    float Return0p5Resolution(bool use_part_fsc = true);

    inline float kDa_to_area_in_pixel(float molecular_mass_in_kDa) {
        return PI * powf(3.0f * (kDa_to_Angstrom3(molecular_mass_in_kDa) / powf(pixel_size, 3)) / 4.0f / PI, 2.0f / 3.0f);
    };
};
