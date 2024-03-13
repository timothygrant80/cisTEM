#ifndef _SRC_PROGRAMS_SIMULATE_COORDS_H_
#define _SRC_PROGRAMS_SIMULATE_COORDS_H_

// Use to keep track of what the specimen looks like.
enum PaddingStatus : int { none    = 0,
                           fft     = 1,
                           solvent = 2 };

// FIXME this should just be given to Cosine Rectangular method in the image class
constexpr int TAPERWIDTH = 29; // TODO this should be set to 12 with zeros padded out by size neighborhood and calculated by taper = 0.5+0.5.*cos((((1:pixelFallOff)).*pi)./(length((1:pixelFallOff+1))));
constexpr int N_TAPERS   = 3; // for trimming final image

#ifndef ENABLEGPU

typedef struct _int3 {
    int x;
    int y;
    int z;
} int3;

typedef struct _float3 {
    float x;
    float y;
    float z;
} float3;

#endif

class Coords {

  public:
    Coords( );
    ~Coords( );

    inline void CheckVectorIsSet(bool input) {
        if ( ! input ) {
            wxPrintf("Trying to use a coord vector that is not yet set\n");
            exit(-1);
        }
    };

    void SetLargestSpecimenVolume(int nx, int ny, int nz);
    int3 GetLargestSpecimenVolume( );
    void SetSpecimenVolume(int nx, int ny, int nz);
    int3 GetSpecimenVolume( );

    void SetSolventPadding(int nx, int ny, int nz);
    int3 GetSolventPadding( );

    void SetFFTPadding(int max_factor, int pad_by);
    int3 GetFFTPadding( );

    float3 ReturnOrigin(PaddingStatus status);
    int    ReturnLargestDimension(int dimension);

    void Allocate(Image* image_to_allocate, PaddingStatus status, bool should_be_in_real_space, bool only_2d);
    void PadSolventToFFT(Image* image_to_resize);

    void PadFFTToSolvent(Image* image_to_resize);
    void PadFFTToSpecimen(Image* image_to_resize);
    void PadToLargestSpecimen(Image* image_to_resize, bool should_be_square);

    void PadToWantedSize(Image* image_to_resize, int wanted_size);
    bool IsAllocationNecessary(Image* image_to_allocate, int3 size);

    void   SetSolventPadding_Z(int wanted_nz);
    int3   make_int3(int x, int y, int z);
    float3 make_float3(float x, float y, float z);

  private:
    bool is_set_specimen;
    bool is_set_fft_padding;
    bool is_set_solvent_padding;

    int3   pixel;
    float3 fractional;

    // These are the dimensions of the final specimen.
    int3 specimen;
    int3 largest_specimen;

    // This is the padding needed to take the specimen size to a good FFT size
    int3 fft_padding;

    // This is the padding for the water so that under rotation, the projected density remains constant.
    int3 solvent_padding;
};

#endif