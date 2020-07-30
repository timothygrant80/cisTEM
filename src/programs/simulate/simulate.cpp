#include "../../core/core_headers.h"
#include "wave_function_propagator.h"
#include "scattering_potential.h"

#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!


//#define DO_BEAM_TILT_FULL true

// From Shang and Sigworth, average of polar and non-polar from table 1 (with correction to polar radius 3, 1.7-->3.0);
// terms 3-5 have the average atomic radius of C,N,O,H added to shift the curve to be relative to atomic center.
const float xtra_shift = -0.5f;
const float HYDRATION_VALS[8] = {0.1750f,   -0.1350f,    2.23f,    3.43f,    4.78f,    1.0000f,    1.7700f,    0.9550f};
const float DISTANCE_INIT = 100000.0f; // Set the distance slab to a large value

const int N_WATER_TERMS = 40;

const float MIN_BFACTOR = 30.0f;
const float inelastic_scalar = 0.0725f ; // this value is for 1 Apix. When not taking the sqrt (original but I think wrong approach) the value 0.75 worked

// Parameters for calculating water. Because all waters are the same, except a sub-pixel origin offset. A SUB_PIXEL_NeL stack of projected potentials is pre-calculated with given offsets.
const AtomType SOLVENT_TYPE = oxygen; // 2 N, 3 O 15, fit water
const float water_oxygen_ratio = 1.0f;//0.8775;

const int SUB_PIXEL_NEIGHBORHOOD = 2;
const int SUB_PIXEL_NeL = pow((SUB_PIXEL_NEIGHBORHOOD*2+1),3);

// FIXME this should just be given to Cosine Rectangular method in the image class
const int TAPERWIDTH = 29; // TODO this should be set to 12 with zeros padded out by size neighborhood and calculated by taper = 0.5+0.5.*cos((((1:pixelFallOff)).*pi)./(length((1:pixelFallOff+1))));
const int N_TAPERS = 3; // for trimming final image
const float TAPER[29] = {0,0,0,0,0,
						 0.003943, 0.015708, 0.035112, 0.061847, 0.095492, 0.135516, 0.181288, 0.232087,
						 0.287110, 0.345492, 0.406309, 0.468605, 0.531395, 0.593691, 0.654508, 0.712890,
						 0.767913, 0.818712, 0.864484, 0.904508, 0.938153, 0.964888, 0.984292, 0.996057};

const int IMAGEPADDING =  0;//512; // Padding applied for the Fourier operations in the propagation steps. Removed prior to writing out.
const int IMAGETRIMVAL = 0;//IMAGEPADDING + 2*TAPERWIDTH;

const float MAX_PIXEL_SIZE = 4.0f;



//beam_tilt_xform



//const int n_tilt_angles = 101;
//const float SET_TILT_ANGLES[n_tilt_angles] = {-70.000, -68.600, -67.200, -65.800, -64.400, -63.000, -61.600, -60.200, -58.800, -57.400, -56.000, -54.600, -53.200, -51.800, -50.400, -49.000, -47.600, -46.200, -44.800, -43.400, -42.000, -40.600, -39.200, -37.800, -36.400, -35.000, -33.600, -32.200, -30.800, -29.400, -28.000, -26.600, -25.200, -23.800, -22.400, -21.000, -19.600, -18.200, -16.800, -15.400, -14.000, -12.600, -11.200, -9.800, -8.400, -7.000, -5.600, -4.200, -2.800, -1.400, 0.000, 1.400, 2.800, 4.200, 5.600, 7.000, 8.400, 9.800, 11.200, 12.600, 14.000, 15.400, 16.800, 18.200, 19.600, 21.000, 22.400, 23.800, 25.200, 26.600, 28.000, 29.400, 30.800, 32.200, 33.600, 35.000, 36.400, 37.800, 39.200, 40.600, 42.000, 43.400, 44.800, 46.200, 47.600, 49.000, 50.400, 51.800, 53.200, 54.600, 56.000, 57.400, 58.800, 60.200, 61.600, 63.000, 64.400, 65.800, 67.200, 68.600, 70.000};
//const float SET_TILT_ANGLES[n_tilt_angles] = {-15.400, -14.000, -12.600, -11.200, -9.800, -8.400, -7.000, -5.600, -4.200, -2.800, -1.400, -0.000, 1.400, 2.800, 4.200, 5.600, 7.000, 8.400, 9.800, 11.200, 12.600, 14.000, 15.400, 16.800, 18.200, 19.600, 21.000, 22.400, 23.800, 25.200, 26.600, 28.000, 29.400, 30.800, 32.200, 33.600, 35.000, 36.400, 37.800, 39.200, 40.600, 42.000, 43.400, 44.800, 46.200, 47.600, 49.000, 50.400, 51.800, 53.200, 54.600, 56.000, 57.400, 58.800, 60.200, 61.600, 63.000, 64.400, 65.800, 67.200, 68.600, 70.000, -16.800, -18.200, -19.600, -21.000, -22.400, -23.800, -25.200, -26.600, -28.000, -29.400, -30.800, -32.200, -33.600, -35.000, -36.400, -37.800, -39.200, -40.600, -42.000, -43.400, -44.800, -46.200, -47.600, -49.000, -50.400, -51.800, -53.200, -54.600, -56.000, -57.400, -58.800, -60.200, -61.600, -63.000, -64.400, -65.800, -67.200, -68.600, -70.000};

//const int n_tilt_angK2Count-300kv-50e-3eps.txtles = 50;
//const float   SET_TILT_ANGLES[n_tilt_angles] = {-15.400, -12.600, -9.800, -7.000, -4.200, -1.400, 1.400, 4.200, 7.000, 9.800, 12.600, 15.400, 18.200, 21.000, 23.800, 26.600, 29.400, 32.200, 35.000, 37.800, 40.600, 43.400, 46.200, 49.000, 51.800, 54.600, 57.400, 60.200, 63.000, 65.800, 68.600, -18.200, -21.000, -23.800, -26.600, -29.400, -32.200, -35.000, -37.800, -40.600, -43.400, -46.200, -49.000, -51.800, -54.600, -57.400, -60.200, -63.000, -65.800, -68.600};

//const int n_tilt_angles = 51;
//const float SET_TILT_ANGLES[n_tilt_angles] = {0.00, 2.800, -2.800, 5.600, -5.600, 8.400, -8.400, 11.200, -11.200, 14.000, -14.000, 16.800, -16.800, 19.600, -19.600, 22.400, -22.400, 25.200, -25.200, 28.000, -28.000, 30.800, -30.800, 33.600, -33.600, 36.400, -36.400, 39.200, -39.200, 42.000, -42.000, 44.800, -44.800, 47.600, -47.600, 50.400, -50.400, 53.200, -53.200, 56.000, -56.000, 58.800, -58.800, 61.600, -61.600, 64.400, -64.400, 67.200, -67.200, 70.000, -70.000};
// Some of the more common elements, should add to this later. These are from Peng et al. 1996.
//const int n_tilt_angles = 41;
//const float SET_TILT_ANGLES[n_tilt_angles] = {0, 3, -3, 6, -6, 9, -9, 12, -12, 15, -15, 18, -18, 21, -21, 24, -24, 27, -27, 30, -30, 33, -33, 36, -36, 39, -39, 42, -42, 45, -45, 48, -48, 51, -51, 54, -54, 57, -57, 60, -60};
const int n_tilt_angles = 27;
const float SET_TILT_ANGLES[n_tilt_angles] = {0, 3, -3, 6, -6, 9, -9, 12, -12, 15, -15, 18, -18, 21, -21, 24, -24, 27, -27, 30, -30, 33, 36, 39, 42, 45, 48};
//const int n_tilt_angles = 2;
//const float SET_TILT_ANGLES[n_tilt_angles] = {60,55,50,40,45,40,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35,-40,-45,-50,-55,-60};
//const int n_tilt_angles = 1;
//const float SET_TILT_ANGLES[n_tilt_angles] = {55};



// Assuming a perfect counting - so no read noise, and no coincednce loss. Then we have a flat NPS and the DQE is just DQE(0)*MTF^2
// The parameters below are for a 5 gaussian fit to 300 KeV , 2.5 EPS from Ruskin et al. with DQE(0) = 0.791
const float DQE_PARAMETERS_A[1][5] = {

		{-0.01516,-0.5662,-0.09731,-0.01551,21.47}


};

const float DQE_PARAMETERS_B[1][5] = {

		{0.02671,-0.02504, 0.162,0.2831,-2.28}

};

const float DQE_PARAMETERS_C[1][5] = {

		{0.01774,0.1441,0.1082,0.07916,1.372}

};


//// Rough fit for ultrascan4000
//const float DQE_PARAMETERS_A[1][5] = {
//
//		{-1.356, 59.87, -0.2552, -0.00679, -0.02245 }
//
//
//};
//
//const float DQE_PARAMETERS_B[1][5] = {
//
//		{-0.1878, -3.064, 0.0995, 0.2449, 0.4422}
//
//};
//
//const float DQE_PARAMETERS_C[1][5] = {
//
//		{0.1207,1.457,0.264,0.006423,0.06323}
//
//};

struct corners {

	float x1; float x2;
	float y1; float y2;
	float z1; float z2;
};

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

// Use to keep track of what the specimen looks like.
enum PaddingStatus : int { none = 0, fft = 1, solvent = 2};

class Coords
{

public:


	Coords() {
		is_set_specimen = false;
		is_set_fft_padding = false;
		is_set_solvent_padding = false;
		largest_specimen = make_int3(0,0,0);
	}
	~Coords() {
		// DO NOTHING

	}


	void CheckVectorIsSet(bool input)
	{
		if( ! input )
		{
			wxPrintf("Trying to use a coord vector that is not yet set\n") ;
			exit(-1);
		}
	}

	int3 make_int3(int x, int y, int z) { int3 retVal; retVal.x = x; retVal.y = y; retVal.z = z; return retVal; }
	float3 make_float3(float x, float y, float z) { float3 retVal; retVal.x = x; retVal.y = y; retVal.z = z; return retVal; }

	void SetSpecimenVolume(int nx, int ny, int nz)
	{
		// Update with the current specimen dimensions
		specimen = make_int3(nx, ny, nz);
		if (! is_set_specimen)
		{
			is_set_specimen = true;
		}
		// Keep a running record of the largest specimen dimensions encountered yet
		SetLargestSpecimenVolume(nx, ny, nz);
	}
	int3 GetSpecimenVolume() { CheckVectorIsSet( is_set_specimen) ; return specimen ; }

	// The minimum specimen dimensions can vary depending on the current orientation. We need to track the largest dimension for output
	void SetLargestSpecimenVolume(int nx, int ny, int nz)
	{
		CheckVectorIsSet(is_set_specimen);
		largest_specimen = make_int3(std::max(specimen.x, nx),
								     std::max(specimen.y, ny),
									 std::max(specimen.z, nz));
	}
	int3 GetLargestSpecimenVolume()  { CheckVectorIsSet(is_set_specimen) ; return largest_specimen ; }


	void SetSolventPadding(int nx, int ny, int nz)
	{
		solvent_padding = make_int3(nx, ny, nz);
		is_set_solvent_padding = true;
	}

	int3 GetSolventPadding() { CheckVectorIsSet(is_set_solvent_padding) ; return solvent_padding; }

	void SetFFTPadding(int max_factor, int pad_by)
	{

		CheckVectorIsSet(is_set_solvent_padding);

		fft_padding = make_int3(ReturnClosestFactorizedUpper(solvent_padding.x * pad_by, max_factor, true),
								ReturnClosestFactorizedUpper(solvent_padding.y * pad_by, max_factor, true),
								(int)0);
		is_set_fft_padding = true;

	}
	int3 GetFFTPadding() { CheckVectorIsSet(is_set_fft_padding) ; return fft_padding; }

	float3 ReturnOrigin(PaddingStatus status)
	{
		float3 output;
		switch (status)
		{
		// TODO think about why the 0.5 is needed. It doesn't seem like it should be, but when checking the crosshair pdb it seems to be the best solution to keep everything centered.
			case none:	output = make_float3(specimen.x/2 + 0.5f, specimen.y/2 + 0.5f, specimen.z/2 + 0.5f);
					break;

			case fft:	output = make_float3(fft_padding.x/2 + 0.5f, fft_padding.y/2 + 0.5f, fft_padding.z/2 + 0.5f);
					break;

			case solvent:	output = make_float3(solvent_padding.x/2 + 0.5f, solvent_padding.y/2 + 0.5f, solvent_padding.z/2 + 0.5f);
					break;
		}
		return output;
	}

	int ReturnLargestDimension(int dimension)
	{
		int output;
		switch (dimension)
		{
			case 0:	output = largest_specimen.x - N_TAPERS*TAPERWIDTH;
					break;

			case 1:	output = largest_specimen.y - N_TAPERS*TAPERWIDTH;
					break;

			case 2:	output = largest_specimen.z;
					break;
		}
		if ( IsOdd(output) ) output += 1;
		return output;
	}


	void Allocate(Image* image_to_allocate, PaddingStatus status, bool should_be_in_real_space, bool only_2d)
	{
		int3 size;
		switch (status)
		{
			case none :
				size = GetSpecimenVolume(); if (only_2d) { size.z = 1; }
				// Sets to zero and returns fals if no allocation needed.
				if  (IsAllocationNecessary(image_to_allocate, size))
				{
					image_to_allocate->Allocate(size.x, size.y, size.z, should_be_in_real_space);

				}
				break;

			case fft :
				size = GetFFTPadding(); if (only_2d) { size.z = 1; }
				// Sets to zero and returns fals if no allocation needed.
				if  (IsAllocationNecessary(image_to_allocate, size))
				{
					image_to_allocate->Allocate(size.x, size.y, size.z, should_be_in_real_space);
				}
				break;

			case solvent :
				size = GetSolventPadding(); if (only_2d) { size.z = 1; }

				// Sets to zero and returns fals if no allocation needed.
				if  (IsAllocationNecessary(image_to_allocate, size))
				{
					image_to_allocate->Allocate(size.x, size.y, size.z, should_be_in_real_space);
				}
				break;
		}



	}

	void PadSolventToFFT(Image* image_to_resize)
	{
		// I expect the edges to already be tapered to 0;
		// The assumption is that we are working in 2d at this point.

		CheckVectorIsSet(is_set_fft_padding);
		image_to_resize->Resize(fft_padding.x, fft_padding.y, 1,0.0f);
	}

	void PadFFTToSolvent(Image* image_to_resize)
	{
		// The assumption is that we are working in 2d at this point.
		CheckVectorIsSet(is_set_solvent_padding);
		image_to_resize->Resize(solvent_padding.x, solvent_padding.y, 1,0.0f);
	}

	void PadFFTToSpecimen(Image* image_to_resize)
	{
		CheckVectorIsSet(is_set_specimen);
		image_to_resize->Resize(specimen.x, specimen.y, 1,0.0f);
	}

	void PadToLargestSpecimen(Image* image_to_resize, bool should_be_square)
	{

		CheckVectorIsSet(is_set_specimen);

		if (should_be_square)
		{
			int sq_dim = std::max(ReturnLargestDimension(0), ReturnLargestDimension(1));
			image_to_resize->Resize(sq_dim, sq_dim, 1);

		}

		else
		{

			image_to_resize->Resize(ReturnLargestDimension(0), ReturnLargestDimension(1), 1);

		}
	}

	void PadToWantedSize(Image* image_to_resize, int wanted_size)
	{

		if (wanted_size < 0)
		{
			PadToLargestSpecimen(image_to_resize, true);
		}
		else
		{
			image_to_resize->Resize(wanted_size, wanted_size, 1);
		}
	}

	bool IsAllocationNecessary(Image* image_to_allocate, int3 size)
	{

		bool allocation_required = true;

		if (image_to_allocate->is_in_memory == true &&
			size.x == image_to_allocate->logical_x_dimension &&
			size.y == image_to_allocate->logical_y_dimension &&
			size.z == image_to_allocate->logical_z_dimension)
		{
			allocation_required = false;
		}



		return allocation_required;

	}


private:

	bool is_set_specimen;
	bool is_set_fft_padding;
	bool is_set_solvent_padding;

	int3 pixel;
	float3 fractional;

	// These are the dimensions of the final specimen.
	int3 specimen;
	int3 largest_specimen;

	// This is the padding needed to take the specimen size to a good FFT size
	int3 fft_padding;


	// This is the padding for the water so that under rotation, the projected density remains constant.
	int3 solvent_padding;



};






class SimulateApp : public MyApp
{

	public:


    bool 		doExpertOptions = false;
	bool 	 	tilt_series = false;
	bool 		need_to_allocate_projected_water = true;
	bool  		do_beam_tilt = false;
	bool 		use_existing_params = false;




	ScatteringPotential sp;

	Image *projected_water; // waters calculated over the specified subpixel shifts and projected.
	Image *projected_cation; //
	Image *projected_anion;


	bool DoCalculation();
	void DoInteractiveUserInput();
	std::string output_filename;

	float	    defocus = 0.0f;
    float 		kV = 300.0f;
    float 		spherical_aberration = 2.7f;
    float 		objective_aperture = 100.0f;
    float 		wavelength = 1.968697e-2; // Angstrom, default for 300kV
	float    	wanted_pixel_size = 0.0f;
	float		wanted_pixel_size_sq = 0.0f;
	float 		unscaled_pixel_size = 0.0f; // When there is a mag change
    float		lead_term = 0.0f;
    float 		phase_plate_thickness = 276.0; //TODO CITE Angstrom, default for pi/2 phase shift, from 2.0 g/cm^3 amorphous carbon, 300 kV and 10.7V mean inner potential as determined by electron holography (carbon atoms are ~ 8)

    float       astigmatism_scaling = 0.0f;

	long 	 	number_of_non_water_atoms = 0; // a 32 bit int would be limited to ~ 400 nm^3
	float 	 	*image_mean;
	float		*inelastic_mean;
	float		current_total_exposure = 0;
	float 		pre_exposure = 0;

	int 	 	size_neighborhood = 0;
	int 		size_neighborhood_water = 0;

	long 		doParticleStack = 0;
	int 	 	number_of_threads = 1;
    float 	 	do3d = 0.0f;
    int			bin3d = 1;





	float 		dose_per_frame = 1;
	float 		dose_rate = 3.0; // e/physical pixel/s this should be set by the user. Changes the illumination aperature and DQE curve
	float 		number_of_frames = 1;
	double		total_waters_incorporated = 0;
	float 		average_at_cutoff[N_WATER_TERMS]; // This assumes a fixed 1/2 angstrom sampling of the hydration shell curve
	float		water_weight[N_WATER_TERMS];

	float       set_real_part_wave_function_in = 1.0f;
	int 		minimum_padding_x_and_y = 32; // This will be changed to be the larger of this value plus the required taper region
	float  		propagation_distance = 5.0f;
	float       minimum_thickness_z = 20.0f;


	bool ONLY_SAVE_SUMS = true;
	bool DO_NOT_RANDMOIZE_ANGLES = false;
	bool MODIFY_ONLY_SIGNAL = true;
	// To add error to the global alignment
	float 	tilt_axis = 0; // degrees from Y-axis FIXME thickness calc, water padding, a few others are only valid for 0* tilt axis.
	float 	in_plane_sigma = 2; // spread in-plane angles based on neighbors
	float 	tilt_angle_sigma = 0.1; //;
	float 	magnification_sigma = 0.0001;//;
    float	stdErr = 0.0f;
    float	extra_phase_shift = 0.0f;

	RotationMatrix beam_tilt_xform;
	float beam_tilt_x = 0.0f;//0.6f;
	float beam_tilt_y = 0.0f;//-0.2f;
	float beam_tilt_z_X_component;
	float beam_tilt_z_Y_component;
	float beam_tilt = 0.0f;
	float beam_tilt_azimuth = 0.0f;
	float particle_shift_x = 0.0f;
	float particle_shift_y =0.0f;
	float DO_BEAM_TILT_FULL = true;
	float amplitude_contrast;


    float bFactor_scaling;
    float min_bFactor;

    FrealignParameterFile  parameter_file;
    cisTEMParameters parameter_star;
    cisTEMParameterLine parameters;
    std::string			   parameter_file_name;
    std::string	parameter_star_file_name;
    long number_preexisting_particles;
    wxString preexisting_particle_file_name;

    float parameter_vect[17] = {0.0f};
    float water_scaling = 1.0f;

    Coords coords;
    StopWatch timer;

    // Intermediate images that may be useful for diagnostics.
    void AddCommandLineOptions();
    bool SAVE_WATER_AND_OTHER = false;
    bool SAVE_PROJECTED_WATER = false;
    bool SAVE_PHASE_GRATING = false;
    bool SAVE_PHASE_GRATING_DOSE_FILTERED = false;
    bool SAVE_PHASE_GRATING_PLUS_WATER = false;
    bool SAVE_PROBABILITY_WAVE = false;
    bool SAVE_TO_COMPARE_JPR = false;
    int  JPR_SIZE = 514;
    bool SAVE_WITH_DQE = false;
    bool SAVE_WITH_NORMALIZED_DOSE = false;
    bool SAVE_POISSON_PRE_NTF = false;
    bool SAVE_POISSON_WITH_NTF = false;
    // Add these from the command line with long option
	int max_number_of_noise_particles = 0;
	float noise_particle_radius_as_mutliple_of_particle_radius = 1.8;
	float noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius = -0.05;
	float noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius = 0.10;
	float emulate_tilt_angle = 0.0f;

	void probability_density_2d(PDB *pdb_ensemble, int time_step);
	// Note the water does not take the dose as an argument.
	void  calc_scattering_potential(const PDB * current_specimen, Image *scattering_slab, Image *inelastic_slab, Image *distance_slab, RotationMatrix rotate_waters,
			                        float rotated_oZ, int *slabIDX_start, int *slabIDX_end, int iSlab);

	void  calc_water_potential(Image *projected_water, AtomType atom_id);
	void  fill_water_potential(const PDB * current_specimen,Image *scattering_slab, Image *scattering_potential,
							   Image *inelastic_potential, Image *distance_slab, Water *water_box,RotationMatrix rotate_waters,
														   float rotated_oZ, int *slabIDX_start, int *slabIDX_end, int iSlab);


	void  project(Image *image_to_project, Image *image_to_project_into,  int iSlab);
	void  taper_edges(Image *image_to_taper,  int iSlab, bool inelastic_img);
	void  apply_sqrt_DQE_or_NTF(Image *image_in, int iTilt_IDX, bool do_root_DQE);


	inline float return_bfactor(float pdb_bfactor)
	{
		return 	 0.25f*(this->min_bFactor + pdb_bfactor*this->bFactor_scaling);
	}


	inline float return_scattering_potential(corners &R, float* bPlusB, AtomType &atom_id)
	{

		float temp_potential = 0.0f;
		float t0;
		bool t1,t2,t3;

		// if product < 0, we need to sum the two independent terms, otherwise we want the difference.
		t1 = R.x1 * R.x2 < 0  ?  false : true;
		t2 = R.y1 * R.y2 < 0  ?  false : true;
		t3 = R.z1 * R.z2 < 0  ?  false : true;

		for (int iGaussian = 0; iGaussian < 5; iGaussian++)
		{

			t0  = (t1) ? erff(bPlusB[iGaussian]*R.x2) - erff(bPlusB[iGaussian]*R.x1) : fabsf(erff(bPlusB[iGaussian]*R.x2)) + fabsf(erff(bPlusB[iGaussian]*R.x1));
			t0 *= (t2) ? erff(bPlusB[iGaussian]*R.y2) - erff(bPlusB[iGaussian]*R.y1) : fabsf(erff(bPlusB[iGaussian]*R.y2)) + fabsf(erff(bPlusB[iGaussian]*R.y1));
			t0 *= (t3) ? erff(bPlusB[iGaussian]*R.z2) - erff(bPlusB[iGaussian]*R.z1) : fabsf(erff(bPlusB[iGaussian]*R.z2)) + fabsf(erff(bPlusB[iGaussian]*R.z1));

			temp_potential += sp.ReturnScatteringParamtersA(atom_id,iGaussian) * fabsf(t0) ;

		} // loop over gaussian fits

		return temp_potential *= this->lead_term;

	};



	// Shift the curves to the right as the values from Shang/Sigworth are distance to VDW radius (avg C/O/N/H = 1.48 A)
	// FIXME now that you are saving distances, you can also consider polar/non-polar residues separately for an "effective" distance since the curves have the same shape with a linear offset.
	const float PUSH_BACK_BY = -1.48;
	inline float return_hydration_weight(float &radius)
	{
		 return  0.5f + 0.5f *  std::erff( (radius + PUSH_BACK_BY)-(HYDRATION_VALS[2]+xtra_shift*this->wanted_pixel_size) / (sqrtf(2)*HYDRATION_VALS[5])) +
									   HYDRATION_VALS[0] * expf(-powf((radius + PUSH_BACK_BY)-(HYDRATION_VALS[3]+xtra_shift*this->wanted_pixel_size),2)/(2*powf(HYDRATION_VALS[6],2))) +
									   HYDRATION_VALS[1] * expf(-powf((radius + PUSH_BACK_BY)-(HYDRATION_VALS[4]+xtra_shift*this->wanted_pixel_size),2)/(2*powf(HYDRATION_VALS[7],2)));
	}
	// Same as above but taper to zero from 3 - 7 Ang
	inline float return_hydration_weight_tapered(float taper_from, float &radius)
	{

		 return  (0.5f + 0.5f *  std::erff( (radius + PUSH_BACK_BY)-(HYDRATION_VALS[2]+xtra_shift*this->wanted_pixel_size) / (sqrtf(2)*HYDRATION_VALS[5])) +
									   HYDRATION_VALS[0] * expf(-powf((radius + PUSH_BACK_BY)-(HYDRATION_VALS[3]+xtra_shift*this->wanted_pixel_size),2)/(2*powf(HYDRATION_VALS[6],2))) +
									   HYDRATION_VALS[1] * expf(-powf((radius + PUSH_BACK_BY)-(HYDRATION_VALS[4]+xtra_shift*this->wanted_pixel_size),2)/(2*powf(HYDRATION_VALS[7],2))) ) *
				 (0.5f + 0.5f * cosf(radius - taper_from));
	}





	//////////////////////////////////////////
	////////////
	// For development
	//
	//const float  MEAN_FREE_PATH = 4000;// Angstrom, newer paper by bridgett (2018) and a couple older TODO add ref. Use to reduce total probability (https://doi.org/10.1016/j.jsb.2018.06.007)


	// Note that the first two errors here are just used in matching amorphous carbon for validation. The third is used in simulations.
	// The surface phase error (Wanner & Tesche 2005) quantified by holography accounts for a bias due to surface effects not included in the simple model here
	float	SURFACE_PHASE_ERROR = 0.497;
	// The bond phase error is used to account for the remaining phase shift that is missing, due to all remaining scattering. The assumption is that amorphous water has >= the scattering due to delocalized electrons
	float	BOND_PHASE_ERROR = 0.0;
	// To account for the bond phase error in practice, a small scaling factor is applied to the atomic potentials
	float	BOND_SCALING_FACTOR = 1.043; //1.0475;
	float	MEAN_INNER_POTENTIAL = 9.09; // for 1.75 g/cm^3 amorphous carbon as in Wanner & Tesche
	bool	DO_PHASE_PLATE = false;

	// CONTROL FOR DEBUGGING AND VALIDATION
	bool 	add_mean_water_potential  = false; // For comparing to published results - only applies to a 3d potential

	bool 	DO_SINC_BLUR = false; // TODO add me back in. This is to blur the image due to intra-frame motion. Apply to projected specimen not waters
	bool 	DO_PRINT = false;
	//
	bool 	DO_SOLVENT = true; // False to skip adding any solvent
	bool 	CALC_WATER_NO_HOLE = false; // Makes the solvent cutoff so large that water is added on top of the protein. This is to mimick Rullgard/Vulovic
	bool	CALC_HOLES_ONLY = false; // This should calculate the atoms so the water is properly excluded, but not include these in the propagation. TODO checkme
	bool 	DEBUG_POISSON = false; // Skip the poisson draw - must be true (this is gets over-ridden) if DO_PHASE_PLATE is true


	bool 	DO_COHERENCE_ENVELOPE = true;

	bool 	WHITEN_IMG = false; // if SAVE_REF then it will also be whitened when this option is enabled TODO checkme
	bool 	SAVE_REF   = false;
	bool 	CORRECT_CTF= false; // only affects the image if Whiten_img is also true
	bool 	EXPOSURE_FILTER_REF = false;
	bool 	DO_EXPOSURE_FILTER_FINAL_IMG = false;
	int 	DO_EXPOSURE_FILTER = 2;  ///// In 2d or 3d - or 0 to turn of. 2d does not appear to produce the expected results.
	bool 	DO_EXPOSURE_COMPLEMENT_PHASE_RANDOMIZE = false; // maintain the Energy of the protein (since no mass is actually lost) by randomizing the phases with weights that complement the exposure filter
	bool 	DO_APPLY_DQE = true;
	int 	CAMERA_MODEL=0;

	int wanted_output_size = -1;

	float wgt = 0.0f;
	float bf = 0.0f;
	///////////
	/////////////////////////////////////////

	private:

};

// Optional command-line stuff
void SimulateApp::AddCommandLineOptions()
{


	// TODO consider short vs long switches.

	// Options for saving diagnostic images
	command_line_parser.AddLongSwitch("save-water-and-other","Save image of water and scattering.");
	command_line_parser.AddLongSwitch("save-projected-water","Save image projected water atoms with sub-pixel offsets.");
	command_line_parser.AddLongSwitch("save-phase-grating","Save phase grating of non-solvent atoms.");
	command_line_parser.AddLongSwitch("save-phase-grating-dose-filtered","Save phase grating of non-solvent atoms + exposure filter.");
	command_line_parser.AddLongSwitch("save-phase-grating-plus-water","Save phase grating of non-solvent atoms plus water atoms.");
	command_line_parser.AddLongSwitch("save-to-compare-jpr","Save images with the same size to compare to JPR reference images.");
	command_line_parser.AddLongSwitch("save-with-dqe","Save |wavefunction|^2 + DQE effects.");
	// These shouldn't be needed anymore, no NTF applied.
	command_line_parser.AddLongSwitch("save-poisson-pre-ntf","Save image of water and scattering.");
	command_line_parser.AddLongSwitch("save-poisson-post-ntf","Save image of water and scattering.");
	command_line_parser.AddLongSwitch("save-detector-wavefunction", "Save the detector wave function directly? Skip Poisson draw, default is no"); // Skip the poisson draw - must be true (this is gets over-ridden) if DO_PHASE_PLATE is true


	// Options for computation (TODO add gpu flag here)
	command_line_parser.AddOption("j","","Desired number of threads. Overrides interactive user input. Is overriden by env var OMP_NUM_THREADS",wxCMD_LINE_VAL_NUMBER);

	// Options related to calibration
	command_line_parser.AddOption("","surface-phase-error","Surface phase error (radian) From Wanner & Tesche, default is 0.497",wxCMD_LINE_VAL_DOUBLE);
	command_line_parser.AddOption("","bond-phase-error","Bond phase error (radian) Calibrated from experiment, may need to be updated, default is 0.09",wxCMD_LINE_VAL_DOUBLE);
	// Note that the first two errors here are just used in matching amorphous carbon for validation. The third is used in simulations.
	// The surface phase error (Wanner & Tesche 2005) quantified by holography accounts for a bias due to surface effects not included in the simple model here
	// The bond phase error is used to account for the remaining phase shift that is missing, due to all remaining scattering. The assumption is that amorphous water has >= the scattering due to delocalized electrons
	// To account for the bond phase error in practice, a small scaling factor is applied to the atomic potentials
	command_line_parser.AddOption("","bond-scaling","Compensate for bond phase error by scaling the interaction potential. Calibrated from experiment, may need to be updated. default is 1.065",wxCMD_LINE_VAL_DOUBLE);
	command_line_parser.AddOption("","carbon-mip","Amorphous carbon MIP From Wanner & Tesche for 1.75 g/cm^3 carbon. default is 9.09",wxCMD_LINE_VAL_DOUBLE); // for 1.75 g/cm^3 amorphous carbon as in Wanner & Tesche
	command_line_parser.AddLongSwitch("do-phase-plate","do a phase plate simulation? default is no");

	// CONTROL FOR DEBUGGING AND VALIDATION
	command_line_parser.AddLongSwitch("add-constant-background","Add a constant potential for the mean water? default is no"); // For comparing to published results - only applies to a 3d potential
	command_line_parser.AddLongSwitch("do-sinc-blur", "Intra-frame sinc blur on protein? Currently this does nothing, default is no"); // TODO add me back in. This is to blur the image due to intra-frame motion. Apply to projected specimen not waters
	command_line_parser.AddLongSwitch("print-extra-info", "Print out a bunch of extra diagnostic information? default is no");
	command_line_parser.AddLongSwitch("skip-solvent", "Skip adding potential, constant or atom-wise for solvent? default is no"); // False to skip adding any solvent
	command_line_parser.AddLongSwitch("calc-water-no-hole", "Ignore protein envelope and add solvent everywhere? default is no"); // Makes the solvent cutoff so large that water is added on top of the protein. This is to mimick Rullgard/Vulovic
	command_line_parser.AddLongSwitch("calc-holes-only", "Use protein envelope, but don't add it in (holes only)? default is no");// This should calculate the atoms so the water is properly excluded, but not include these in the propagation. TODO checkme
//	command_line_parser.AddLongSwitch("save-detector-wavefunction", "Save the detector wave function directly? Skip Poisson draw, default is no"); // Skip the poisson draw - must be true (this is gets over-ridden) if DO_PHASE_PLATE is true
	command_line_parser.AddLongSwitch("skip-random-angles", "Skip randomizing angles in a particle stack? default is no"); //
	command_line_parser.AddLongSwitch("only-modify-signal-3d", "When applying the cummulative exposure filter to the 3d reference, only modify the signal (reduce exposure filter by 1 - (1-x)/(1+x)");


	command_line_parser.AddLongSwitch("skip-coherence-envelope","Apply spatial coherence envelope? default is no"); // These do nothing~! FIXME
	command_line_parser.AddOption("","max_number_of_noise_particles","Maximum number of neighboring noise particles when simulating an image stack. Default is 0",wxCMD_LINE_VAL_NUMBER);
	command_line_parser.AddOption("","noise_particle_radius_as_mutliple_of_particle_radius","default is 1.8",wxCMD_LINE_VAL_DOUBLE);
	command_line_parser.AddOption("","noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius","default is -0.05",wxCMD_LINE_VAL_DOUBLE);
	command_line_parser.AddOption("","noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius","default is  0.10",wxCMD_LINE_VAL_DOUBLE);
	command_line_parser.AddOption("","emulate_tilt_angle","default is 0.0 degrees around Y axis", wxCMD_LINE_VAL_DOUBLE);


	command_line_parser.AddLongSwitch("whiten-output", "Whiten the image? default is no"); // if SAVE_REF then it will also be whitened when this option is enabled TODO checkme
	command_line_parser.AddLongSwitch("do-perfect-reference","Save a perfect reference image with no noise in addition to the image. default is no");
	command_line_parser.AddLongSwitch("apply-ctf", "Multiply by average CTF? default is no");// only affects the image if Whiten_img is also true
	command_line_parser.AddLongSwitch("apply-exposure-filter-ref","Exposure filter the perfect reference? default is no");
	command_line_parser.AddLongSwitch("apply-exposure-filter-img","Exposure filter the final movie/image? default is no");
	command_line_parser.AddLongSwitch("skip-radiation-damage","Do NOT apply the 2d exposure filter to the scattering potential prior to adding the solvent. default is no");  ///// In 2d or 3d - or 0 to turn of. 2d does not appear to produce the expected results.
	command_line_parser.AddLongSwitch("maintain-power", "Maintain power with random phases in exposure filter? default is no");// maintain the Energy of the protein (since no mass is actually lost) by randomizing the phases with weights that complement the exposure filter
	command_line_parser.AddLongSwitch("skip-dqe","Do NOT apply the DQE? depends on camera model default is no and applies K2 fixme");
	command_line_parser.AddLongSwitch("skip-tilted-propagation","Apply the phase shift due to beam tilt, but don't actually do the inclined wave propagation. default no");
	command_line_parser.AddLongSwitch("save-frames","The default is to save the integrated frames. This option overrides this behavior.");


	command_line_parser.AddOption("","wgt","Maximum number of neighboring noise particles when simulating an image stack. Default is 0",wxCMD_LINE_VAL_DOUBLE);
	command_line_parser.AddOption("","bf","Maximum number of neighboring noise particles when simulating an image stack. Default is 0",wxCMD_LINE_VAL_DOUBLE);

//	command_line_parser.AddOption("j","","Desired number of threads. Overrides interactive user input. Is overriden by env var OMP_NUM_THREADS",wxCMD_LINE_VAL_NUMBER);
}

IMPLEMENT_APP(SimulateApp);

// override the DoInteractiveUserInput

void SimulateApp::DoInteractiveUserInput()
{


	 bool add_more_pdbs = true;
	 bool supply_origin = false;
	 int iPDB = 0;
	 int iOrigin;
	 int iParticleCopy;
	 double temp_double;
	 long temp_long;

	 //////////
	/////////// Check the command line options, could this be done at the top?
	SAVE_WATER_AND_OTHER = command_line_parser.FoundSwitch("save-water-and-other");
	SAVE_PROJECTED_WATER = command_line_parser.FoundSwitch("save-projected-water");
	SAVE_PHASE_GRATING = command_line_parser.FoundSwitch("save-phase-grating");
	SAVE_PHASE_GRATING_DOSE_FILTERED = command_line_parser.FoundSwitch("save-phase-grating-dose-filtered");
	SAVE_PHASE_GRATING_PLUS_WATER = command_line_parser.FoundSwitch("save-phase-grating-plus-water");
	SAVE_TO_COMPARE_JPR = command_line_parser.FoundSwitch("save-to-compare-jpr");
	SAVE_WITH_DQE = command_line_parser.FoundSwitch("save-with-dqe");
	SAVE_POISSON_PRE_NTF = command_line_parser.FoundSwitch("save-poisson-pre-ntf");
	SAVE_POISSON_WITH_NTF = command_line_parser.FoundSwitch("save-poisson-post-ntf");
	DEBUG_POISSON = command_line_parser.Found("save-detector-wavefunction");


	if (command_line_parser.Found("j", &temp_long)) { number_of_threads = (int)temp_long;}
	if (command_line_parser.Found("surface-phase-error", &temp_double)) { SURFACE_PHASE_ERROR = (float)temp_double;}
	if (command_line_parser.Found("bond-phase-error", &temp_double)) { BOND_PHASE_ERROR = (float)temp_double;}

	// Note that the first two errors here are just used in matching amorphous carbon for validation. The third is used in simulations.
	// The surface phase error (Wanner & Tesche 2005) quantified by holography accounts for a bias due to surface effects not included in the simple model here
	// The bond phase error is used to account for the remaining phase shift that is missing, due to all remaining scattering. The assumption is that amorphous water has >= the scattering due to delocalized electrons
	// To account for the bond phase error in practice, a small scaling factor is applied to the atomic potentials
	if (command_line_parser.Found("bond-scaling", &temp_double)) { BOND_SCALING_FACTOR = (float)temp_double;}
	if (command_line_parser.Found("carbon-mip",  &temp_double)) { MEAN_INNER_POTENTIAL = (float)temp_double;}
	if (command_line_parser.FoundSwitch("do-phase-plate")) DO_PHASE_PLATE = true;

	// CONTROL FOR DEBUGGING AND VALIDATION
	add_mean_water_potential = command_line_parser.Found("add-constant-background");
	DO_SINC_BLUR = command_line_parser.Found("do-sinc-blur");
	DO_PRINT = command_line_parser.Found("print-extra-info");
	if (command_line_parser.Found("skip-solvent")) DO_SOLVENT = false;
	CALC_WATER_NO_HOLE = command_line_parser.Found("calc-water-no-hole");
	CALC_HOLES_ONLY = command_line_parser.Found("calc-holes-only");
	if (command_line_parser.Found("skip-coherence-envelope")) DO_COHERENCE_ENVELOPE = false;
	WHITEN_IMG = command_line_parser.Found("whiten-output");
	SAVE_REF = command_line_parser.Found("do-perfect-reference");
	CORRECT_CTF = command_line_parser.Found("apply-ctf");
	EXPOSURE_FILTER_REF = command_line_parser.Found("apply-exposure-filter-ref");
	DO_EXPOSURE_FILTER_FINAL_IMG = command_line_parser.Found("apply-exposure-filter-img");
	if (command_line_parser.Found("skip-radiation-damage")) DO_EXPOSURE_FILTER = false;
	DO_EXPOSURE_COMPLEMENT_PHASE_RANDOMIZE = command_line_parser.Found("maintain-power");
	if (command_line_parser.Found("skip-dqe")) DO_APPLY_DQE = false;
	if (command_line_parser.Found("skip-tilted-propagation")) DO_BEAM_TILT_FULL = false;
	if (command_line_parser.Found("save-frames")) ONLY_SAVE_SUMS = false;
	if (command_line_parser.Found("skip-random-angles")) DO_NOT_RANDMOIZE_ANGLES = true;
	if (command_line_parser.Found("only-modify-signal-3d")) MODIFY_ONLY_SIGNAL = true;

	if (command_line_parser.Found("max_number_of_noise_particles", &temp_long)) { max_number_of_noise_particles = (int)temp_long;}
	if (command_line_parser.Found("noise_particle_radius_as_mutliple_of_particle_radius", &temp_double)) { noise_particle_radius_as_mutliple_of_particle_radius = (float)temp_double;}
	if (command_line_parser.Found("noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius", &temp_double)) { noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius = (float)temp_double;}
	if (command_line_parser.Found("noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius", &temp_double)) { noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius = (float)temp_double;}
	if (command_line_parser.Found("emulate_tilt_angle",&temp_double)) { emulate_tilt_angle = (float)temp_double ; }


	if (command_line_parser.Found("wgt", &temp_double)) { wgt = (float)temp_double;}
	if (command_line_parser.Found("bf", &temp_double)) { bf = (float)temp_double;}



//	command_line_parser.AddLongSwitch(, "Save the detector wave function directly? Skip Poisson draw, default is no"); // Skip the poisson draw - must be true (this is gets over-ridden) if DO_PHASE_PLATE is true



	///////////////////
	//////////////////
	 UserInput *my_input = new UserInput("Simulator", 0.25);

	 this->output_filename = my_input->GetFilenameFromUser("Output filename","just the base, no extension, will be mrc","test_tilt.mrc",false);
	 // FIXME the range is way too big for 3d but needed for the fixed size hack.
	 this->do3d = my_input->GetFloatFromUser("Make a 3d scattering potential?","just potential if > 0, input is the wanted cubic size","0.0",-4096.0f,4096.0f);


 	 this->number_of_threads = my_input->GetIntFromUser("Number of threads", "Max is number of tilts", "1", 1);

	 while (sp.number_of_pdbs < MAX_NUMBER_PDBS && add_more_pdbs)
	 {
		 sp.pdb_file_names[sp.number_of_pdbs-1] = my_input->GetFilenameFromUser("PDB file name", "an input PDB", "my_atom_coords.pdb", true );
		 // This is now coming directly from the PDB
		 //particle_copy_number[number_of_pdbs-1] = my_input->GetIntFromUser("Copy number of this particle", "To be inserted into the ensemble", "1", 1);


		 add_more_pdbs = my_input->GetYesNoFromUser("Add another type of particle?", "Add another pdb to create additional features in the ensemble", "no");
		 if (add_more_pdbs) {sp.number_of_pdbs++;}
	 }



	 this->wanted_pixel_size 		= my_input->GetFloatFromUser("Output pixel size (Angstroms)","Output size for the final projections","1.0",0.01,MAX_PIXEL_SIZE);
	 this->bFactor_scaling		 = my_input->GetFloatFromUser("Linear scaling of per atom bFactor","0 off, 1 use as is","0",0,10000);
	 this->min_bFactor    		 = my_input->GetFloatFromUser("Per atom (xtal) bFactor added to all atoms","0 off, 1 use as is","10.0",0.0f,10000);



	 // Hack to set the wanted output size for a 2d. Build this in as a real option FIXME
	 if (this->do3d < 0)
	 {
		 wanted_output_size = fabsf(this->do3d);
		 this->do3d = 0;
	 }
	 wxPrintf("\n\n\t\twanted size is %d\n\n",wanted_output_size);
	 // end hack

	 if (this->do3d)
	 {
//		 // Check to make sure the sampling is sufficient, if not, oversample and bin at the end.
//		 if (this->wanted_pixel_size > 0.8 && this->wanted_pixel_size <= 1.5)
//		 {
//			 wxPrintf("\nOversampling your 3d by a factor of 2 for calculation.\n");
//			 this->wanted_pixel_size /= 2.0f;
//			 this->bin3d = 2;
//		 }
//		 else if (this->wanted_pixel_size > 1.5 && this->wanted_pixel_size < 3.0)
//		 {
//			 wxPrintf("\nOversampling your 3d by a factor of 4 for calculation.\n");
//
//			 this->wanted_pixel_size /= 4.0f;
//			 this->bin3d = 4;
//		 }
//		 else
//		 {
//			 //do nothing
//		 }

	 }
	 else
	 {
		 this->tilt_series = my_input->GetYesNoFromUser("Create a tilt-series as well?","Should make 0 degree with full dose, then tilt","no");
		 if ( this->tilt_series == true )
		 {
			 // not doing anything for now, fixed range and increment.test_multi.mrc
			 // TODO read in star file name
			 (long)my_input->GetFloatFromUser("Dummy particle stack question, not valid for tilt-series","Number of particles at random orientations, 0 for just an image","1",0,1e7);
		 }
		 if (this->tilt_series == false)
		 {
			 this->doParticleStack = (long)my_input->GetFloatFromUser("Create a particle stack?","Number of particles at random orientations, 0 for just an image","1",0,1e7);
			 wxPrintf("Making a particle stack with %ld images\n",this->doParticleStack);
		 }

		 this->defocus                  = my_input->GetFloatFromUser("wanted defocus (Angstroms)","Out","700",0,120000);
		 this->extra_phase_shift        = my_input->GetFloatFromUser("wanted additional phase shift x * PI radians, i.e. 0.5 for PI/2 shift.","","0.0",-2.0,2.0);
		 this->dose_per_frame			= my_input->GetFloatFromUser("electrons/Ang^2 in a frame at the specimen","","1.0",0.05,20.0);
		 this->dose_rate			    = my_input->GetFloatFromUser("electrons/Pixel/sec","Affects coherence but not coincidence loss","3.0",0.001,200.0);

		 this->number_of_frames			= my_input->GetFloatFromUser("number of frames per movie (micrograph or tilt)","","30",1.0,1000.0);
	 }

	 this->doExpertOptions			= my_input->GetYesNoFromUser("Set expert options?","","no");
	 this->wanted_pixel_size_sq 	= powf(this->wanted_pixel_size,2);


	 if (this->doExpertOptions)
	 {

		 this->use_existing_params   = my_input->GetYesNoFromUser("Use an existing set of orientations","yes no","no");
			if (use_existing_params)
			{
				// Check to see if the paramter file is valid
				preexisting_particle_file_name = my_input->GetFilenameFromUser("cisTEM star file name", "an input star file to match reconstruction", "myparams.star", true );
				if (! DoesFileExist(preexisting_particle_file_name))
				{
					SendError(wxString::Format("Error: Input star file %s not found\n", preexisting_particle_file_name));
					exit(-1);
				}

			    number_preexisting_particles = my_input->GetIntFromUser("Number of particles in the param file", "I don't know of a better way to get this info jsut yet", "1", 1);
//				this->parameter_file.Open(parameter_file_name, access_type,17);
//				wxPrintf("%d access_type\n", this->parameter_file.access_type);
//				this->parameter_file.ReadFile(true, number_of_particles);
//				wxPrintf("\nRecreating %ld particles from the supplied parameter file\n", number_of_particles);

			}
		 this->water_scaling 		 = my_input->GetFloatFromUser("Linear scaling water intensity","0 off, 1 use as is","1",0,1);
		 this->astigmatism_scaling	 = my_input->GetFloatFromUser("fraction of the defocus to add as astigmatism","0 off, 1 use as is","0.0",0,0.5);
		 this->kV 					 = my_input->GetFloatFromUser("Accelrating volatage (kV)","Default is 300","300.0",80.0,1000.0f); // Calculations are not valid outside this range
		 this->objective_aperture 	 = my_input->GetFloatFromUser("Objective aperture diameter in micron","","100.0",0.0,1000.0);
		 this->spherical_aberration	 = my_input->GetFloatFromUser("Spherical aberration constant in millimeters","","2.7",0.0,5.0);
		 this->stdErr = my_input->GetFloatFromUser("Std deviation of error to use in shifts, astigmatism, rotations etc.","","0.0",0.0,100.0);
		 this->pre_exposure = my_input->GetFloatFromUser("Pre-exposure in electron/A^2","use for testing exposure filter","0",0.0);

		 // Since kV is not default, need to calculate these parameters
//		 const float WAVELENGTH = pow(1.968697e-2,1); // Angstrom
		 this->wavelength 		  = 1226.39 / sqrtf(this->kV*1000 + 0.97845e-6*powf(this->kV*1000,2)) * 1e-2; // Angstrom

			// This minimum padding should probably depend on the defocus. FIXME
			this->minimum_padding_x_and_y = my_input->GetIntFromUser("minimum padding of images with solvent", "", "32", 0,4096);
			this->minimum_thickness_z = my_input->GetIntFromUser("minimum thickness in Z", "", "10",2,10000);
			this->propagation_distance = my_input->GetFloatFromUser("Propagation distance in angstrom","Also used as minimum thickness","5",-1e6,1e6);

			// propagation_distance can be negative to set nSlabs = 1;
			if (fabsf(propagation_distance) > minimum_thickness_z)
			{
				minimum_thickness_z = fabsf(propagation_distance);
				wxPrintf("min thickness was less than propagation distance, so setting it there\n");
			}

			// To add error to the global alignment
			this->tilt_axis = my_input->GetFloatFromUser("Rotation of tilt-axis from Y (degrees)","","0.0",0,180); // TODO does this apply everywhere it should
			this->in_plane_sigma = my_input->GetFloatFromUser("Standard deviation on angles in plane (degrees)","","2",0,100); // spread in-plane angles based on neighbors
			this->tilt_angle_sigma = my_input->GetFloatFromUser("Standard deviation on tilt-angles (degrees)","","0.1",0,10);; //;
			this->magnification_sigma = my_input->GetFloatFromUser("Standard deviation on magnification (fraction)","","0.0001",0,1);//;
			this->beam_tilt_x = my_input->GetFloatFromUser("Beam-tilt in X (milli radian)","","0.0",-300,300);//0.6f;
			this->beam_tilt_y = my_input->GetFloatFromUser("Beam-tilt in Y (milli radian)","","0.0",-300,300);//-0.2f;
			this->particle_shift_x =  my_input->GetFloatFromUser("Beam-tilt particle shift in X (Angstrom)","","0.0",-100,100);
			this->particle_shift_y =  my_input->GetFloatFromUser("Beam-tilt particle shift in Y (Angstrom)","","0.0",-100,100);



			set_real_part_wave_function_in = sqrt(this->dose_per_frame * this->wanted_pixel_size_sq);


	 }
	 else
	 {
		 this->wavelength 		  = 1226.39 / sqrtf(this->kV*1000 + 0.97845e-6*powf(this->kV*1000,2)) * 1e-2; // Angstrom
		 this->water_scaling=1.0f;
		 this->astigmatism_scaling=0.0f;
		 this->objective_aperture = 100.0f;
		 this->spherical_aberration = 2.7f;
		 this->stdErr = 0.0f;
	 }



	parameter_star_file_name = output_filename + ".star";

//	parameter_file_name = output_filename + ".par";
//	wxString parameter_header = "C           PSI   THETA     PHI       SHX       SHY     MAG  INCLUDE   DF1      DF2  ANGAST  PSHIFT     OCC      LogP      SIGMA   SCORE  CHANGE";
//	this->parameter_file.Open(parameter_file_name,1,17);
//	this->parameter_file.WriteCommentLine(parameter_header);


	 if (DO_PHASE_PLATE)
	 {

		 if (SURFACE_PHASE_ERROR < 0)
		 {
			 wxPrintf("SURFACE_PHASE_ERROR < 0, subbing min thickness for phase plate thickness\n");
			 this->phase_plate_thickness = this->minimum_thickness_z;
		 }
		 else
		 {
			 this->phase_plate_thickness = (PIf/2.0f + SURFACE_PHASE_ERROR + BOND_PHASE_ERROR) / ( MEAN_INNER_POTENTIAL/(kV*1000) * (511+kV)/(2*511+kV) * (2*PIf / (wavelength*1e-10)) )*1e10;
		 }

		 wxPrintf("With a mean inner potential of %2.2fV a thickness of %2.2f ang is needed for a pi/2 phase shift \n",MEAN_INNER_POTENTIAL,this->phase_plate_thickness);
		 wxPrintf("Phase shift params %f %f %f\n",BOND_PHASE_ERROR,BOND_SCALING_FACTOR,SURFACE_PHASE_ERROR);

	 }

	 //	 this->lead_term = BOND_SCALING_FACTOR * this->wavelength / this->wanted_pixel_size_sq / 8.0f;
	 	 // the 1/8 just comes from the integration of the gaussian which is too large by a factor of 2 in each dimension
//	     BOND_SCALING_FACTOR = (BOND_SCALING_FACTOR - 1.0f) / wanted_pixel_size + 1.0f;
	 	 this->lead_term = BOND_SCALING_FACTOR * this->wavelength  / 8.0f / wanted_pixel_size_sq;

	 	 // The third term is a rough estiamte to ensure any delocalized info from particles is retained. It should probably also consider the stdErr
	 	 minimum_padding_x_and_y =  myroundint(N_TAPERS*TAPERWIDTH*wanted_pixel_size + (float)minimum_padding_x_and_y + defocus/100.0f);

	delete my_input;


}

// overide the do calculation method which will be what is actually run..

bool SimulateApp::DoCalculation()
{




//	if (use_existing_params) wxPrintf("\nRecreating %d particles from the supplied parameter file\n", this->parameter_file.number_of_lines);


	// Profiling

	// get the arguments for this job..



	// TODO is this the best place to put this?
	this->current_total_exposure = pre_exposure;

	if (CORRECT_CTF && use_existing_params)
	{
		wxPrintf("I did not set up ctf correction and the use of existing parameters. FIXME\n");
		throw;
	}


	if (doParticleStack > 0)
	{

		sp.InitPdbEnsemble( wanted_pixel_size, do3d, minimum_padding_x_and_y, minimum_thickness_z,
				max_number_of_noise_particles,
				noise_particle_radius_as_mutliple_of_particle_radius,
				noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
				noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
				emulate_tilt_angle
				);

	}
	else
	{
		// Over-ride the max number of noise particles
		max_number_of_noise_particles = 0;
		sp.InitPdbEnsemble( wanted_pixel_size, do3d, minimum_padding_x_and_y, minimum_thickness_z,
				max_number_of_noise_particles,
				noise_particle_radius_as_mutliple_of_particle_radius,
				noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
				noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
				emulate_tilt_angle
				);
	}

	number_of_non_water_atoms = sp.ReturnTotalNumberOfNonWaterAtoms();



	wxPrintf("\nThere are %ld non-water atoms in the specimen.\n",this->number_of_non_water_atoms);
	wxPrintf("\nCurrent number of PDBs %d\n",sp.number_of_pdbs);

	// FIXME add time steps.
	int time_step = 0 ;

    this->probability_density_2d(sp.pdb_ensemble, time_step);


	wxPrintf("\nFinished pre seg fault\n");


	// It gives a segfault at the end either way.
   // pdb_ensemble[0].Close();




	return true;
}


/*
I've moved the wanted originz and euler angles as REMARK 351 in the PDB so that environments may be "easily" created in chimera.
It makes more sense then to intialize the trajectories in the call to PDB::init
Leave this in until convinced it works ok.
*/




void SimulateApp::probability_density_2d(PDB *pdb_ensemble, int time_step)
{

	bool SCALE_DEFOCUS_TO_MATCH_300 = true;
	float scale_defocus = 1.0f;

	// TODO Set even range in z to avoid large zero areas
	// TODO Set a check on the solvent fraction and scaling and report if it is unreasonable. Define reasonable
	// TODO Set a check on the range of values, report if defocus tolerance is too small (should all be positive)
//	Image img;
//	img.QuickAndDirtyReadSlice("/groups/grigorieff/home/himesb/cisTEM_2/cisTEM/trunk/gpu/include/oval_full.mrc",1);
//	img.QuickAndDirtyWriteSlice("/groups/grigorieff/home/himesb/tmp/noshift.mrc",1,1,false);
//	img.PhaseShift(1.5,3.5,0.0);
//	img.QuickAndDirtyWriteSlice("/groups/grigorieff/home/himesb/tmp/withShift.mrc",1,1,false);
//	exit(-1);

	long current_atom;
	long nOutOfBounds = 0;
	long iTilt_IDX;
	int iSlab = 0;
	int current_3D_slice_to_save = 0;
	float *shift_x, *shift_y, *shift_z;
	float *mag_diff;
	float euler1(0), euler2(0), euler3(0);

	// CTF parameters:  There should be an associated variablility with tilt angle TODO and get rid of extra parameters
	float wanted_acceleration_voltage = this->kV; // keV
	float wanted_spherical_aberration = this->spherical_aberration; // mm
	float wanted_defocus_1_in_angstroms = this->defocus; // A
	float wanted_defocus_2_in_angstroms = this->defocus; //A
	float wanted_astigmatism_azimuth = 0.0; // degrees
	float astigmatism_angle_randomizer = 0.0; //
	float defocus_randomizer = 0.0;
	float wanted_additional_phase_shift_in_radians  = this->extra_phase_shift*PI;
    float *propagator_distance; // in Angstom, <= defocus tolerance.
    float defocus_offset = 0;



    wxPrintf("Using extra phase shift of %f radians\n",wanted_additional_phase_shift_in_radians);

	int frame_lines;
	if (ONLY_SAVE_SUMS) { frame_lines = 1; }
	else { frame_lines = number_of_frames; }
	parameter_vect[0]  = 1; // idx
	parameter_vect[1]  = 0; // psi
	parameter_vect[2]  = 0; // theta
	parameter_vect[3]  = 0; // phi
	parameter_vect[4]  = 0; // shx
	parameter_vect[5]  = 0; // shy
	parameter_vect[6]  = 0; // mag
	parameter_vect[7]  = 1; // include
	parameter_vect[8]  = wanted_defocus_1_in_angstroms; //
	parameter_vect[9]  = wanted_defocus_2_in_angstroms; //
	parameter_vect[10] = wanted_astigmatism_azimuth; //
	parameter_vect[11] = wanted_additional_phase_shift_in_radians;
	parameter_vect[12] = 100; // Occupancy
	parameter_vect[13] = -1000; // LogP
	parameter_vect[14] = 10; //Sigma
	parameter_vect[15] = 10; //Score
	parameter_vect[16] = 0; // Change
	// Keep a copy of the unscaled pixel size to handle magnification changes.
	this->unscaled_pixel_size = this->wanted_pixel_size;

	// Keep a copy of the unscaled pixel size to handle magnification changes.
	this->unscaled_pixel_size = this->wanted_pixel_size;
	if (use_existing_params)
	{
		// For now, we are only reading in from the par file FIXME

//		this->parameter_file.Open(preexisting_particle_file_name, 0,17);
//		this->parameter_file.ReadFile(true, number_preexisting_particles);
//
		//		wxPrintf("\nRecreating %ld particles from the supplied parameter file\n", number_preexisting_particles);
//		// Read the first line so that all of the values are initialized in parameter_vect
//		this->parameter_file.ReadLine(this->parameter_vect);
//		// Reset the counter to the first line
//		this->parameter_file.current_line--;

		parameter_star.ReadFromcisTEMStarFile(preexisting_particle_file_name);
		wxPrintf("\nRecreating %ld particles from the supplied star file\n", parameter_star.ReturnNumberofLines());

		parameter_star.parameters_to_write.SetAllToTrue();
		parameter_star.parameters_to_write.image_is_active = false;
		parameter_star.parameters_to_write.original_image_filename = false;
		parameter_star.parameters_to_write.reference_3d_filename = false;
		parameter_star.parameters_to_write.stack_filename = false;

		// Need a check on number of particles TODO
		//parameter_star.ReturnNumberofLines()
	}
	else
	{


		if (this->tilt_series)
		{
			parameter_star.PreallocateMemoryAndBlank(n_tilt_angles * frame_lines);
		}
	    else if ( this->doParticleStack > 0)
	    {
	    	parameter_star.PreallocateMemoryAndBlank(doParticleStack * frame_lines);
	    }
	    else
	    {
	    	parameter_star.PreallocateMemoryAndBlank(frame_lines);
	    }

		parameter_star.parameters_to_write.SetAllToTrue();
		parameter_star.parameters_to_write.image_is_active = false;
		parameter_star.parameters_to_write.original_image_filename = false;
		parameter_star.parameters_to_write.reference_3d_filename = false;
		parameter_star.parameters_to_write.stack_filename = false;



		parameters.position_in_stack = 0;
		parameters.image_is_active = 0;
		parameters.psi = 0.0f;
		parameters.theta = 0.0f;
		parameters.phi = 0.0f;
		parameters.x_shift = 0.0f;
		parameters.y_shift = 0.0f;
		parameters.defocus_1 = wanted_defocus_1_in_angstroms;
		parameters.defocus_2 = wanted_defocus_2_in_angstroms;
		parameters.defocus_angle = wanted_astigmatism_azimuth;
		parameters.phase_shift = wanted_additional_phase_shift_in_radians;
		parameters.occupancy = 100.0f;
		parameters.logp = -1000.0f;
		parameters.sigma = 10.0f;
		parameters.score = 10.0f;
		parameters.score_change = 0.0f;
		parameters.pixel_size = wanted_pixel_size;
		parameters.microscope_voltage_kv = wanted_acceleration_voltage;
		parameters.microscope_spherical_aberration_mm = wanted_spherical_aberration;
		parameters.amplitude_contrast = 0.0f;
		parameters.beam_tilt_x = beam_tilt_x;
		parameters.beam_tilt_y = beam_tilt_y;
		parameters.image_shift_x = particle_shift_x;
		parameters.image_shift_y = particle_shift_y;
		parameters.stack_filename = output_filename;
		parameters.original_image_filename = wxEmptyString;
		parameters.reference_3d_filename = wxEmptyString;
		parameters.best_2d_class = 0;
		parameters.beam_tilt_group = 0;
		parameters.particle_group = 0;
		parameters.pre_exposure = 0.0f;
		parameters.total_exposure = 0.0f;

	}


	// Do this after intializing the parameters which should be stored in millirad
    if (beam_tilt_x != 0.0 || beam_tilt_y != 0.0 )
    {

    	do_beam_tilt = true;
    	beam_tilt_azimuth = atan2f(beam_tilt_y,beam_tilt_x);
    	beam_tilt = sqrtf(powf(beam_tilt_x, 2) + powf(beam_tilt_y, 2));

    	beam_tilt   /= 1000.0f;
    	beam_tilt_x /= 1000.0f;
    	beam_tilt_y /= 1000.0f;

    	beam_tilt_z_X_component = tanf(beam_tilt)*cosf(beam_tilt_azimuth);
    	beam_tilt_z_Y_component = tanf(beam_tilt)*sinf(beam_tilt_azimuth);


    }

    // TODO fix periodicity in Z on slab


    int iTilt;
    int nTilts;
    float max_tilt = 0;
    float * tilt_psi;
    float * tilt_theta;
    float * tilt_phi;

	// FIXME use new header in random
	std::default_random_engine generator;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> uniform_dist(0.000000001f, 1.0f);


    if ( this-> tilt_series )
    {

    	nTilts = n_tilt_angles;
		if (this->use_existing_params)
		{
			wxPrintf("\n\nUsing an existing parameter file only supported on a particle stack\n\n");
			throw;
		}
//    	max_tilt  = 60.0f;
//    	float tilt_range = max_tilt;
//    	float tilt_inc = 3.0f;
    	max_tilt  = 0.0f;
    	for (int iTilt = 0; iTilt < nTilts; iTilt++)
    	{
    		if (fabsf(SET_TILT_ANGLES[iTilt]) > fabsf(max_tilt)) { max_tilt = SET_TILT_ANGLES[iTilt] ; }
    	}
//    	float tilt_range = max_tilt;
    	float tilt_inc = 1.4;
//    	nTilts = ceil(tilt_range/tilt_inc)*2 +1;
    	tilt_psi   = new float[nTilts];
    	tilt_theta = new float[nTilts];
    	tilt_phi   = new float[nTilts];
    	shift_x    = new float[nTilts];
    	shift_y    = new float[nTilts];
    	shift_z    = new float[nTilts];
    	mag_diff   = new float[nTilts];



    	std::normal_distribution<float>  norm_dist_inplane(0.0f,in_plane_sigma);
    	std::normal_distribution<float>  norm_dist_tiltangle(0.0f,tilt_angle_sigma);
    	std::normal_distribution<float>  norm_dist_mag(0.0f,magnification_sigma);

    	for (int iTilt=0; iTilt < nTilts; iTilt++)
    	{

    		// to create a conical tilt (1.8*iTilt)+
    		tilt_psi[iTilt] = tilt_axis + this->stdErr * norm_dist_inplane(gen); // *(2*PI);
//    		tilt_theta[iTilt] = -((tilt_range - (float)iTilt*tilt_inc) + this->stdErr * norm_dist_tiltangle(gen));
    		tilt_theta[iTilt] = SET_TILT_ANGLES[iTilt];
    		wxPrintf("%f\n",SET_TILT_ANGLES[iTilt]);
    		tilt_phi[iTilt] = 0.0f;
    		shift_x[iTilt] = this->stdErr * 8*uniform_dist(gen) ;
    		shift_y[iTilt] = this->stdErr * 8*uniform_dist(gen);
    		shift_z[iTilt] = 0.0f;
    		mag_diff[iTilt] = 1.0f + (this->stdErr*norm_dist_mag(gen));



    	}

    }
    else if ( this->doParticleStack > 0)
    {
    	max_tilt = 0.0f;
    	nTilts = this->doParticleStack;
    	tilt_psi   = new float[nTilts];
    	tilt_theta = new float[nTilts];
    	tilt_phi   = new float[nTilts];
    	shift_x    = new float[nTilts];
    	shift_y    = new float[nTilts];
    	shift_z    = new float[nTilts];
    	mag_diff   = new float[nTilts];

    	std::normal_distribution<float> normal_dist(0.0f,1.0f);

    	for (int iTilt=0; iTilt < nTilts; iTilt++)
    	{
    		if (this->use_existing_params)
    		{
//    			this->parameter_file.ReadLine(this->parameter_vect);

        		tilt_psi[iTilt]   = parameter_star.ReturnPsi(iTilt); //parameter_vect[1];
        		tilt_theta[iTilt] = parameter_star.ReturnTheta(iTilt); //parameter_vect[2];
        		tilt_phi[iTilt]   = parameter_star.ReturnPhi(iTilt); //parameter_vect[3];

        		shift_x[iTilt]    = parameter_star.ReturnXShift(iTilt); //parameter_vect[4];
        		shift_y[iTilt]    = parameter_star.ReturnYShift(iTilt); //parameter_vect[5];
        		shift_z[iTilt]  = 0;
        		mag_diff[iTilt] =   parameter_star.ReturnPixelSize(iTilt) / this->unscaled_pixel_size; //parameter_vect[6];
    		}
    		else
    		{
        		tilt_psi[iTilt] = uniform_dist(gen)*360.0f; // *(2*PI);
        		tilt_theta[iTilt] = std::acos(2.0f*uniform_dist(gen)-1.0f) * 180.0f/(float)PI;
        		tilt_phi[iTilt] = -1*tilt_psi[iTilt] + uniform_dist(gen)*360.0f; //*(2*PI);

        		shift_x[iTilt]  = this->stdErr * normal_dist(gen) * 3.0f; // should be in the low tens of Angstroms
        		shift_y[iTilt]  = this->stdErr * normal_dist(gen) * 3.0f;
        		shift_z[iTilt]  = this->stdErr * normal_dist(gen) * 1000; // should be in the low tens of Nanometers

        		mag_diff[iTilt] = 1.0f;
    		}


    	}
    }
    else
    {
		if (this->use_existing_params)
		{
			wxPrintf("\n\nUsing an existing parameter file only supported on a particle stack\n\n");
			throw;
		}

    	max_tilt = 0.0;
    	tilt_theta = new float[1];
    	tilt_theta[0] = 0;
    	nTilts = 1;
    	tilt_psi   = new float[nTilts];
    	tilt_theta = new float[nTilts];
    	tilt_phi   = new float[nTilts];
    	shift_x    = new float[nTilts];
    	shift_y    = new float[nTilts];
    	shift_z    = new float[nTilts];
    	mag_diff   = new float[nTilts];
    	tilt_psi[0]=0; tilt_theta[0]=0;tilt_phi[0]=0;shift_x[0]=0;shift_y[0]=0;shift_z[0]=0;mag_diff[0]=1.0f;
    }


    // I don't think the frames need to be saved in these first two stacks. Rather they are accumulated
	Image *img_frame_stack;
	Image *ref_frame_stack;
	Image *output_image_stack;
	Image *output_reference_stack;
	RotationMatrix particle_rot;


	// Whether we save frames or only sums, make the array of image pointers full (frames.)
	// Output will be a stack of particles (not frames)
	img_frame_stack = new Image[1];
	output_image_stack = new Image[nTilts*(int)this->number_of_frames];

	if (SAVE_REF)
	{
		ref_frame_stack = new Image[1] ;
		output_reference_stack = new Image[nTilts*(int)this->number_of_frames];
	}



	timer.start("Init H20 & Spec");

	// We only want one water box for a tilt series. For a particle stack, re-initialize for each particle.
	Water water_box(DO_PHASE_PLATE);

	// Create a new PDB object that represents the current state of the specimen, with each local motion applied. // TODO mag changes and pixel size (here to determine water box)?
	// For odd sizes there will be an offset of 0.5 pix if simply padded by 2 and cropped by 2 so make it even to start then trim at the end.

	wxPrintf("Got here emulate is %f\n",emulate_tilt_angle);
	// FIXME I think this should somehow be derived from the information already stored in the scattering potential  object
	PDB current_specimen(this->number_of_non_water_atoms, bin3d*(do3d-IsOdd(do3d)), wanted_pixel_size, minimum_padding_x_and_y, minimum_thickness_z,
							max_number_of_noise_particles,
							noise_particle_radius_as_mutliple_of_particle_radius,
							noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
							noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
							emulate_tilt_angle
							);

	timer.lap("Init H20 & Spec");



	wxPrintf("\nThere are %d tilts\n",nTilts);
    for ( iTilt = 0 ; iTilt < nTilts ; iTilt++)
    {

    	this->wanted_pixel_size = this->unscaled_pixel_size * mag_diff[iTilt];

    	this->wanted_pixel_size_sq = this->wanted_pixel_size * this->wanted_pixel_size;

    	wxPrintf("for Tilt %d, scaling the pixel size from %3.3f to %3.3f\n",iTilt,this->unscaled_pixel_size,this->wanted_pixel_size);

    	float total_drift = 0.0f;
		RotationMatrix rotate_waters;
		RotationMatrix max_rotation;
		if (this->tilt_series)
		{
//			rotate_waters.SetToRotation(-tilt_phi[iTilt],-tilt_theta[iTilt],-tilt_psi[iTilt]);
			rotate_waters.SetToEulerRotation(-tilt_psi[iTilt],-tilt_theta[iTilt],-tilt_phi[iTilt]);
			max_rotation.SetToEulerRotation(-tilt_psi[iTilt],-max_tilt,0.0f);

		}
		else
		{
			rotate_waters.SetToEulerRotation(euler1,euler2,euler3);
			max_rotation.SetToEulerRotation(-tilt_psi[iTilt],-tilt_theta[iTilt],-tilt_phi[iTilt]);

		}
		if (this->doParticleStack > 0)
		{


			float phiOUT = 0;
			float psiOUT = 0;
			float thetaOUT = 0;

			wxPrintf("\n\nWorking on iParticle %d/ %d\n\n",iTilt,nTilts);

//			particle_rot.SetToEulerRotation(tilt_phi[iTilt],tilt_theta[iTilt],tilt_psi[iTilt]);
			particle_rot.SetToEulerRotation(-tilt_psi[iTilt],-tilt_theta[iTilt],-tilt_phi[iTilt]);


		    // For particle stack, use the fixed supplied defocus, and apply a fixed amount of astigmatism at random angle to make sure everything is filled in

			if (this->use_existing_params)
			{

				wanted_defocus_1_in_angstroms = parameter_star.ReturnDefocus1(iTilt); //parameter_vect[8];
				wanted_defocus_2_in_angstroms = parameter_star.ReturnDefocus2(iTilt); //parameter_vect[9];
				wanted_astigmatism_azimuth    = parameter_star.ReturnDefocusAngle(iTilt); //parameter_vect[10];
				wanted_additional_phase_shift_in_radians = parameter_star.ReturnPhaseShift(iTilt); //parameter_vect[10];


			}
			else
			{
			    defocus_randomizer = uniform_dist(gen)*this->astigmatism_scaling*this->stdErr;
		    	wxPrintf("For the particle stack, stretching the defocus by %3.2f percent and randmozing the astigmatism angle -90,90",100*defocus_randomizer);
		    	wanted_defocus_1_in_angstroms = this->defocus*(1+defocus_randomizer) + shift_z[iTilt]; // A
		    	wanted_defocus_2_in_angstroms = this->defocus*(1-defocus_randomizer) + shift_z[iTilt]; //A
				wanted_astigmatism_azimuth = (uniform_dist(gen)-0.5f)*179.99f;
			}


		}
		else
		{
			particle_rot.SetToIdentity();

		}


		if (SCALE_DEFOCUS_TO_MATCH_300)
		{
			scale_defocus = (0.0196869700756145f / this->wavelength);
			wanted_defocus_1_in_angstroms *= scale_defocus;
			wanted_defocus_2_in_angstroms *= scale_defocus;
			wxPrintf("Scaling the defocus by %6.6f to match the def at 300 KeV\n", scale_defocus);

		}
		// Scale the defocus so that it is equivalent to 300KeV for experiment


	// Override any rotations when making a 3d reference
	if (this->do3d > 1 || DO_PHASE_PLATE || DO_NOT_RANDMOIZE_ANGLES)
	{
		particle_rot.SetToIdentity();
		rotate_waters.SetToIdentity();
		max_rotation.SetToIdentity();

	}
	if (DO_PHASE_PLATE)
	{
		max_rotation.m[0][0] =  this->phase_plate_thickness;
	}


	for ( int iFrame = 0; iFrame < this->number_of_frames; iFrame ++)
	{



    	int slab_nZ;
    	int rotated_Z; // full range in Z to cover the rotated specimen
    	float rotated_oZ;
    	float slab_oZ;
	    int nSlabs;
		int nS;
		float full_tilt_radians = 0;


		Image jpr_sum_phase;
		Image jpr_sum_detector;


		// Include the max rand shift in z for thickness

		timer.start("Xform Local");
		current_specimen.TransformLocalAndCombine(pdb_ensemble,sp.number_of_pdbs, iFrame, particle_rot, 0.0f); // Shift just defocus shift_z[iTilt]);
		timer.lap("Xform Local");

		// FIXME method for defining the size (in pixels) needed for incorporating the atoms density. The formulat used below is based on including the strongest likely scatterer (Phosphorous) given the bfactor.
		// FIXME
		float BF;
		if (DO_PHASE_PLATE) { BF = MIN_BFACTOR ;} else { BF = return_bfactor(current_specimen.average_bFactor);}

		this->size_neighborhood 	  =  1 + myroundint( (0.4f *sqrtf(0.6f*BF) + 0.2f) / this->wanted_pixel_size);
		wxPrintf("\n\n\tfor frame %d the size neigborhood is %d\n\n", iFrame, this->size_neighborhood);


		if (DO_PHASE_PLATE)
		{
			this->size_neighborhood_water = this->size_neighborhood ;

		}
		else
		{
			this->size_neighborhood_water = 1+myroundint(ceilf(1.0f / this->wanted_pixel_size));

		}

	    wxPrintf("using neighborhood of %2.2f vox^3 for waters and %2.2f vox^3 for non-waters\n",powf(this->size_neighborhood_water*2+1,3),powf(this->size_neighborhood*2+1,3));
	    timer.start("Calc H20 Atoms");
	    ////////////////////////////////////////////////////////////////////////////////
	    ////////// PRE_CALCULATE WATERS
	    // Because the waters are all identical, we create an array of projected water atoms with the wanted sub-pixel offsets. When water is added, the pre-calculated atom with the closest sub-pixel origin is used, weighted depending on
	    // it's proximity to any non-water atoms, and added to the 2d.

	    if ( DO_SOLVENT && this->need_to_allocate_projected_water)
	    {

	    	projected_water= new Image[SUB_PIXEL_NeL];
	    	projected_cation= new Image[SUB_PIXEL_NeL];
	    	projected_anion= new Image[SUB_PIXEL_NeL];


	        for (int iWater = 0 ; iWater < SUB_PIXEL_NeL; iWater++)
	        {
	            projected_water[iWater].Allocate(this->size_neighborhood_water*2+1,this->size_neighborhood_water*2+1,true);
	            projected_water[iWater].SetToConstant(0.0f);

	            projected_cation[iWater].Allocate(this->size_neighborhood_water*2+1,this->size_neighborhood_water*2+1,true);
	            projected_cation[iWater].SetToConstant(0.0f);

	            projected_anion[iWater].Allocate(this->size_neighborhood_water*2+1,this->size_neighborhood_water*2+1,true);
	            projected_anion[iWater].SetToConstant(0.0f);
	        }

			wxPrintf("Starting projected water calc with sizeN %d, %d\n",this->size_neighborhood_water*2+1,this->size_neighborhood_water*2+1);
			this->calc_water_potential(projected_water, water);

			wxPrintf("Finishing projected water calc\n");


			this->need_to_allocate_projected_water = false;
	    }
	    timer.lap("Calc H20 Atoms");
	    //////////
	    ///////////////////////////////////////////////////////////////////////////////

	    // If negative ignore, otherwise make sure the solvent is at least this dimension. The actual padding including rotations are returned in these variables after initializeation.
	    int padSpecimenX = wanted_output_size;
	    int padSpecimenY = wanted_output_size;

		if (padSpecimenX > 0)
		{
			int x_diff = current_specimen.vol_nX- padSpecimenX;
			if (x_diff < 0)
			{
				padSpecimenX= -x_diff;
			}

		}
		else padSpecimenX = 0;
		if  (padSpecimenY > 0)
		{
			int y_diff = current_specimen.vol_nY - padSpecimenY;
			if (y_diff < 0)
			{
				padSpecimenY = -y_diff;
			}
		}
		else padSpecimenY = 0;

		padSpecimenX += N_TAPERS*TAPERWIDTH;
		padSpecimenY += N_TAPERS*TAPERWIDTH;

		// Set the minimum specimen volume to allow trimming of the tapered region.
		coords.SetSpecimenVolume(current_specimen.vol_nX, current_specimen.vol_nY, current_specimen.vol_nZ);
		coords.SetSolventPadding(current_specimen.vol_nX + padSpecimenX, current_specimen.vol_nY + padSpecimenY, current_specimen.vol_nZ);
		 wxPrintf("\n\n\t\twanted size is %d\n\n",padSpecimenX);



		 // TODO put me at the top
		const int fft_max_factor = 5;
		const int fft_max_padding_factor = 1;
	    timer.start("Calc H20 Box");
	    if (iTilt == 0 && iFrame == 0)
	    {
	    	bool is_single_particle;
	    	if (this->tilt_series || this->doParticleStack == 0) is_single_particle = false;
	    	else is_single_particle = true;
	    	wxPrintf("Is single particle is %d\n", is_single_particle);
		    water_box.Init( &current_specimen,this->size_neighborhood_water, this->wanted_pixel_size, this->dose_per_frame, max_rotation, tilt_axis, &padSpecimenX, &padSpecimenY, number_of_threads, is_single_particle);
	    }
	    timer.lap("Calc H20 Box");

	    // Previously I was padding the specimen by the padding needed for in plane rotation. With all rotations in the water padding, this shouldn't be needed.

//		coords.SetSolventPadding(water_box.vol_nX, water_box.vol_nY, water_box.vol_nZ);
		coords.SetFFTPadding(fft_max_factor, fft_max_padding_factor);



		if (SAVE_TO_COMPARE_JPR || DO_PHASE_PLATE)
		{
			if (DO_PHASE_PLATE)
			{
				wxPrintf("\n\nSimulating a phase plate for validation, this overrides SAVE_TO_COMPARE_JPR\n\n");


				coords.Allocate(&jpr_sum_phase, (PaddingStatus)solvent, true, true);
				jpr_sum_phase.SetToConstant(0.0f);

				coords.Allocate(&jpr_sum_detector, (PaddingStatus)solvent, true, true);
				jpr_sum_detector.SetToConstant(0.0f);

			}
			else
			{
				jpr_sum_phase.Allocate(JPR_SIZE,JPR_SIZE,true);
				jpr_sum_phase.SetToConstant(0.0f);

				jpr_sum_detector.Allocate(JPR_SIZE,JPR_SIZE,true);
				jpr_sum_detector.SetToConstant(0.0f);
			}

		}

		// For the tilt pair experiment, add additional drift to the specimen. This is a rough fit to the mean abs shifts measured over 0.33 elec/A^2 frames on my ribo data
		// The tilted is roughly 4x worse than the untilted.
		float iDrift = 1.75f * (4.684f * expf(-1.0f*powf((iFrame*dose_per_frame-0.2842f)/0.994f,2)) +
					            0.514f * expf(-1.0f*powf((iFrame*dose_per_frame-3.21f  )/7.214f,2))) * (0.25f + 0.75f*iTilt);

		total_drift += 0.0f;//iDrift/sqrt(2);
		wxPrintf("\n\tDrift for iTilt %d, iFrame %d is %4.4f Ang\n",iTilt,iFrame,total_drift);
//		current_specimen.TransformGlobalAndSortOnZ(number_of_non_water_atoms, total_drift, total_drift, 0.0f, rotate_waters);
		// TODO incororate the drift;

		// Apply acurrent_specimen.vol_nY global shifts and rotations
//		current_specimen.TransformGlobalAndSortOnZ(number_of_non_water_atoms, shift_x[iTilt], shift_y[iTilt], shift_z[iTilt], rotate_waters);

		timer.start("Xform Global");
		current_specimen.TransformGlobalAndSortOnZ(number_of_non_water_atoms, shift_x[iTilt] + total_drift, shift_y[iTilt] + total_drift, 0.0f, rotate_waters);
		timer.lap("Xform Global");

		// Compute the solvent fraction, with ratio of protein/ water density.
		// Assuming an average 2.2Ang vanderwaal radius ~50 cubic ang, 33.33 waters / cubic nanometer.

		if ( DO_SOLVENT  && water_box.number_of_waters == 0 && this->do3d < 1 )
		{
			// Waters are member variables of the scatteringPotential app - currentSpecimen is passed for size information.


			timer.start("Seed H20");
			water_box.SeedWaters3d();
			timer.lap("Seed H20");


			timer.start("Shake H20");
			water_box.ShakeWaters3d(this->number_of_threads);
			timer.lap("Shake H20");


		}
		else if ( DO_SOLVENT && this->do3d < 1 && ! DO_PHASE_PLATE)
		{


			timer.start("Shake H20");
			water_box.ShakeWaters3d(this->number_of_threads);
			timer.lap("Shake H20");

		}

		timer.start("Allocate Sum");

		// TODO with new solvent add, the edges should not need to be tapered or padded
		coords.Allocate(&img_frame_stack[0], (PaddingStatus)fft, true, true);
		img_frame_stack[0].SetToConstant(0.0f);
		timer.lap("Allocate Sum");

		// Output allocations
		// TODO should these allocations be made for the trimmed stack to save some memory?
//		if (iFrame == 0)
//		{
//			coords.Allocate(&output_image_stack[iTilt*(int)number_of_frames],(PaddingStatus)fft, true, true);
//			if (SAVE_REF)
//			{
//				coords.Allocate(&output_reference_stack[iTilt*(int)number_of_frames],(PaddingStatus)fft, true, true);
//			}
//		}
//		else if ( ! ONLY_SAVE_SUMS )
//		{
			// Only allocate the other images if saving frames
			coords.Allocate(&output_image_stack[iTilt*(int)number_of_frames + iFrame],(PaddingStatus)fft, true, true);
			if (SAVE_REF)
			{
				coords.Allocate(&output_reference_stack[iTilt*(int)number_of_frames + iFrame],(PaddingStatus)fft, true, true);
			}
//
//		}

		if (this->tilt_series)
		{
			full_tilt_radians = PI/180.0f*(tilt_theta[iTilt]);
		}
		else
		{
			full_tilt_radians = PI/180.0f*(euler2);
		}

		if (DO_PRINT) {wxPrintf("tilt angle in radians/deg %2.2e/%2.2e iFrame %d/%f\n",full_tilt_radians,tilt_theta[iTilt],iFrame,this->number_of_frames);}


		rotated_Z = myroundint((float)water_box.vol_nX*fabsf(std::sin(full_tilt_radians)) + (float)water_box.vol_nZ*std::cos(full_tilt_radians));

		wxPrintf("wZ %d csZ %d,rotZ %d\n",water_box.vol_nZ,current_specimen.vol_nZ, rotated_Z);

		//rotated_oZ = ceilf((rotated_Z+1)/2);
		if (DO_PRINT) {wxPrintf("\nflat thicknes, %d and rotated_Z %d\n", current_specimen.vol_nZ, rotated_Z);}
		wxPrintf("\nWorking on iTilt %d at %f degrees for frame %d\n",iTilt,tilt_theta[iTilt],iFrame);

		//  TODO Should separate the mimimal slab thickness, which is a smaller to preserve memory from the minimal prop distance (ie. project sub regions of a slab)
		if ( this->propagation_distance < 0 ) {nSlabs  = 1; this->propagation_distance = fabsf(this->propagation_distance);}
		else { nSlabs = ceilf( (float)rotated_Z * this->wanted_pixel_size/  this->propagation_distance);}

		timer.start("Allocate potentials");
		Image *scattering_potential = new Image[nSlabs];
		Image *inelastic_potential = new Image[nSlabs];
		Image  sampled_potential;

		Image *ref_potential;
		if (SAVE_REF) { ref_potential = new Image[nSlabs] ;}


		nS = ceil((float)rotated_Z / (float)nSlabs);
		wxPrintf("rotated_Z %d nSlabs %d\n",rotated_Z,nSlabs);

		int slabIDX_start[nSlabs];
		int slabIDX_end[nSlabs];
		this->image_mean 	  = new float[nSlabs];
		this->inelastic_mean  = new float[nSlabs];

		// Set up slabs, padded by neighborhood for working
		for (iSlab = 0; iSlab < nSlabs; iSlab++)
		{
			slabIDX_start[iSlab] = iSlab*nS;
			slabIDX_end[iSlab]   = (iSlab+1)*nS - 1;
//			if (iSlab < nSlabs - 1) if (DO_PRINT) {wxPrintf("%d %d\n",slabIDX_start[iSlab],slabIDX_end[iSlab]);}
		}
		// The last slab may be a bit bigger, so make sure you don't miss acurrent_specimen.vol_nYthing.
		slabIDX_end[nSlabs-1] = rotated_Z - 1;
		if (slabIDX_end[nSlabs-1] - slabIDX_start[nSlabs-1] + 1 < 1)
		{
			nSlabs -= 1;
		}
		if (nSlabs > 2)
		{
			wxPrintf("\n\nnSlabs %d\niSlab %d\nlSlab %d\n",nSlabs,slabIDX_end[nSlabs-2]-slabIDX_end[nSlabs-3]+1,slabIDX_end[nSlabs-1]-slabIDX_end[nSlabs-2]+1);
		}
		else
		{
			wxPrintf("\n\nnSlabs %d\niSlab %d\nlSlab %d\n",nSlabs,slabIDX_end[0]+1,slabIDX_start[nSlabs-1]+1);
		}

		propagator_distance = new float[nSlabs];
		bool no_material_encountered = true;


		Image Potential_3d;
		float* scattering_total_shift = new float[nSlabs];
		float* scattering_mass = new float[nSlabs];
		float scattering_center_of_mass = 0.0f;
		timer.lap("Allocate potentials");

		coords.Allocate(&sampled_potential, (PaddingStatus)solvent, true, true);
		sampled_potential.SetToConstant(0.0f); // We'll binarise this then make a mask.

		// Track the elastic scattering potential, smallest distance to scattering center, inelastic scattering potential
		// Moved this from inside slab loop. Could it even go outside the frame loop?
		Image scattering_slab;
		Image distance_slab;
		Image inelastic_slab;

		for (iSlab = 0; iSlab < nSlabs; iSlab++)
		{

			scattering_total_shift[iSlab] = 0.0f;
			propagator_distance[iSlab] =  -1.0f*( this->wanted_pixel_size * (slabIDX_end[iSlab] - slabIDX_start[iSlab] + 1) );
//			propagator_distance[iSlab] =  ( this->wanted_pixel_size * (slabIDX_end[iSlab] - slabIDX_start[iSlab] + 0) );

			coords.Allocate(&scattering_potential[iSlab], (PaddingStatus)solvent, true, true);
//			scattering_potential[iSlab].Allocate(current_specimen.vol_nX, current_specimen.vol_nY,1);
			scattering_potential[iSlab].SetToConstant(0.0f);
			coords.Allocate(&inelastic_potential[iSlab], (PaddingStatus)solvent, true, true);
//			inelastic_potential[iSlab].Allocate(current_specimen.vol_nX, current_specimen.vol_nY,1);
			inelastic_potential[iSlab].SetToConstant(0.0f);


			if (SAVE_REF)
			{
				coords.Allocate(&ref_potential[iSlab], (PaddingStatus)solvent, true, true);
//				ref_potential[iSlab].Allocate(current_specimen.vol_nX, current_specimen.vol_nY,1);
			}

			slab_nZ = slabIDX_end[iSlab] - slabIDX_start[iSlab] + 1;// + 2*this->size_neighborhood;
			slab_oZ = floorf(slab_nZ/2); // origin in the volume containing the rotated slab
			rotated_oZ = floorf(rotated_Z/2);

			// Because we will project along Z, we could put Z on the rows
			if (DO_PRINT) {wxPrintf("iSlab %d %d %d\n",iSlab, slabIDX_start[iSlab],slabIDX_end[iSlab] );}
			if (DO_PRINT) {wxPrintf("slab_oZ %f slab_nZ %d rotated_oZ %f\n",slab_oZ,slab_nZ,rotated_oZ);}


			timer.start("Allocate 3d slabs");
			coords.Allocate(&scattering_slab, (PaddingStatus)solvent, true, false);
			coords.Allocate(&distance_slab, (PaddingStatus)solvent, true, false);
			coords.Allocate(&inelastic_slab, (PaddingStatus)solvent, true, false);
			#pragma omp parallel for num_threads(this->number_of_threads)
			for (long pixel_counter = 0; pixel_counter < scattering_slab.real_memory_allocated; pixel_counter++)
			{
				// Somehow multiplication by zero is about 70% runtime relative to seting zero with image method which in turn is 87 % runtime compared to std::fill or std::memset
				scattering_slab.real_values[pixel_counter] *= 0.0f;
			}

			#pragma omp parallel for num_threads(this->number_of_threads)
			for (long pixel_counter = 0; pixel_counter < inelastic_slab.real_memory_allocated; pixel_counter++)
			{
				// Somehow multiplication by zero is about 70% runtime relative to seting zero with image method which in turn is 87 % runtime compared to std::fill or std::memset
				inelastic_slab.real_values[pixel_counter] *= 0.0f;
			}

			#pragma omp parallel for num_threads(this->number_of_threads)
			for (long pixel_counter = 0; pixel_counter < distance_slab.real_memory_allocated; pixel_counter++)
			{
				distance_slab.real_values[pixel_counter] = DISTANCE_INIT;
			}


			timer.lap("Allocate 3d slabs");

			timer.start("Calc Atoms");
			if (! DO_PHASE_PLATE)
			{
				this->calc_scattering_potential(&current_specimen, &scattering_slab, &inelastic_slab, &distance_slab, rotate_waters, rotated_oZ, slabIDX_start, slabIDX_end, iSlab);
			}
			timer.lap("Calc Atoms");



			////////////////////
			if (this->do3d)
			{

				int iPot;
				long current_pixel;
				bool testHoles = false;

				// Test to look at "holes"
				Image buffer;
				if (testHoles)
				{
					buffer.CopyFrom(&scattering_slab);
					buffer.SetToConstant(0.0);
				}

				if (iSlab == 0)
				{
					coords.Allocate(&Potential_3d, (PaddingStatus)solvent,true, false);
					Potential_3d.SetToConstant(0.0);
				}

				if (add_mean_water_potential)
				{


					float scattering_per_water = 0.0f;
					float avg_scattering_per_voxel;
					float current_weight = 1.0f;
					for (int iWater=0; iWater < projected_water[0].real_memory_allocated; iWater++)
					{
						scattering_per_water += projected_water[0].real_values[iWater];
					}
					float waters_per_ang_cubed = 0.94 * 0.6022140857 / 18.01528; // from water.cpp where 0.94 is a define for water density and 18 is mw

					avg_scattering_per_voxel = scattering_per_water * waters_per_ang_cubed * (this->wanted_pixel_size * this->wanted_pixel_size_sq);

					bool only_use_water_shell = true;
					float taper_from = 4.0; // distance form which to taper off the constant water shell
					for (long current_pixel = 0; current_pixel < distance_slab.real_memory_allocated; current_pixel++)
					{
						if (distance_slab.real_values[current_pixel] < DISTANCE_INIT)
						{
							current_weight = sqrtf(distance_slab.real_values[current_pixel]);
							if (only_use_water_shell && current_weight > taper_from)
							{
								distance_slab.real_values[current_pixel]  = return_hydration_weight_tapered(taper_from, current_weight) * avg_scattering_per_voxel;
							}
							else
							{
								distance_slab.real_values[current_pixel]  = return_hydration_weight(current_weight) * avg_scattering_per_voxel;
							}

						}
						else
						{
							if (only_use_water_shell)
							{
								distance_slab.real_values[current_pixel]  = 0.0f;
							}
							else
							{
								distance_slab.real_values[current_pixel]  = avg_scattering_per_voxel;
							}

						}

//						scattering_slab.real_values[current_pixel] += current_weight;
//						if (testHoles) {buffer.real_values[current_pixel] += current_weight;}

					}

//					distance_slab.QuickAndDirtyWriteSlices("distance_slab.mrc",1,distance_slab.logical_z_dimension,false,1.5);
					distance_slab.ForwardFFT(true);
					distance_slab.ApplyBFactor(this->bf / wanted_pixel_size_sq);
					distance_slab.BackwardFFT();
					distance_slab.MultiplyByConstant(this->wgt);
					scattering_slab.AddImage(&distance_slab);

				}



				if (testHoles) {scattering_slab = buffer;}

				 //for trouble shooting, save each 3d slab

                int offset_slab = scattering_slab.physical_address_of_box_center_z - Potential_3d.physical_address_of_box_center_z + slabIDX_start[iSlab];
				wxPrintf("Inserting slab %d at position %d\n",iSlab,offset_slab);
				Potential_3d.InsertOtherImageAtSpecifiedPosition(&scattering_slab,0,0,offset_slab);

				if (iSlab == nSlabs - 1)
				{


					std::string fileNameOUT = "tmpSlab" + std::to_string(iSlab) + ".mrc";
					MRCFile mrc_out(this->output_filename,true);
					int cubic_size = std::max(std::max(Potential_3d.logical_x_dimension,Potential_3d.logical_y_dimension),Potential_3d.logical_z_dimension);
					// To get the pixel size exact, ensure the volume is a factor of the binning size. If this differs from the wanted size, address after Fourier cropping.
					int exact_cropping_size = cubic_size +  ((int)do3d - IsOdd(do3d)) - ( cubic_size / bin3d );
					wxPrintf("Found a max cubic dimension of %d\nFound an exact cropping size of %d\n",cubic_size,exact_cropping_size);
					Potential_3d.Resize(exact_cropping_size,exact_cropping_size,exact_cropping_size, Potential_3d.ReturnAverageOfRealValuesOnEdges());
//					Potential_3d.QuickAndDirtyWriteSlices("tmpNotCropped.mrc",1,cubic_size);


					Potential_3d.ForwardFFT(true);


					// Apply the pre-exposure
					ElectronDose my_electron_dose(wanted_acceleration_voltage, this->wanted_pixel_size);
					float *dose_filter = new float[Potential_3d.real_memory_allocated/2];
					ZeroFloatArray(dose_filter, Potential_3d.real_memory_allocated/2);


					// Normally the pre-exposure is added to each frame. Here it is taken to be the total exposure.
					my_electron_dose.CalculateCummulativeDoseFilterAs1DArray(&Potential_3d, dose_filter,std::max(this->pre_exposure,1.0f));


					if (MODIFY_ONLY_SIGNAL)
					{
						for (long pixel_counter = 0; pixel_counter < Potential_3d.real_memory_allocated / 2; pixel_counter++)
						{

								Potential_3d.complex_values[pixel_counter] *= 	(1.0f -
																				(1.0f - dose_filter[pixel_counter]) / (1.0f + dose_filter[pixel_counter])
																				);

						}
					}
					else
					{
						for (long pixel_counter = 0; pixel_counter < Potential_3d.real_memory_allocated / 2; pixel_counter++)
						{

								Potential_3d.complex_values[pixel_counter] *= 	dose_filter[pixel_counter];

						}
					}

					if (DO_APPLY_DQE )
					{
						apply_sqrt_DQE_or_NTF(&Potential_3d,0,true);
					}


					delete [] dose_filter;


					if (this->bin3d > 1)
					{
						wxPrintf("\nFourier cropping your 3d by a factor of %d\n",this->bin3d );
						Potential_3d.Resize(exact_cropping_size/this->bin3d,exact_cropping_size/this->bin3d,exact_cropping_size/this->bin3d);

					}

					Potential_3d.BackwardFFT();


					// Now make sure we come out at the correct 3d size.
					Potential_3d.Resize((int)do3d,(int)do3d,(int)do3d,0.0f);



					wxPrintf("Writing out your 3d slices %d --> %d\n",1,(int)do3d);
					Potential_3d.WriteSlices(&mrc_out,1,(int)do3d);
					mrc_out.SetPixelSize(this->wanted_pixel_size * this->bin3d);
					mrc_out.CloseFile();
					// Exit after writing the final slice for the reference. Is this the best way to do this? FIXME
					exit(0);
				}
				continue;

			}
			////////////////////


//			if (DO_EXPOSURE_FILTER == 3 && CALC_HOLES_ONLY == false && CALC_WATER_NO_HOLE == false)
			if (DO_EXPOSURE_FILTER == 3  && CALC_WATER_NO_HOLE == false)
			{
			// add in the exposure filter
				timer.start("ExpFilt 3d");
				scattering_slab.ForwardFFT(true);

				ElectronDose my_electron_dose(wanted_acceleration_voltage, this->wanted_pixel_size);
				float *dose_filter = new float[scattering_slab.real_memory_allocated/2];
				ZeroFloatArray(dose_filter, scattering_slab.real_memory_allocated/2);


				// Normally the pre-exposure is added to each frame. Here it is taken to be the total exposure.
				my_electron_dose.CalculateDoseFilterAs1DArray(&scattering_slab, dose_filter, current_total_exposure, current_total_exposure + dose_per_frame);

				for (long pixel_counter = 0; pixel_counter < scattering_slab.real_memory_allocated/2; pixel_counter++)
				{
					scattering_slab.complex_values[pixel_counter] *= dose_filter[pixel_counter];
				}

				delete [] dose_filter;

				scattering_slab.BackwardFFT();
				timer.lap("ExpFilt 3d");

			}


			if ( CALC_HOLES_ONLY == false )
			{

				timer.start("Project 3d");
				this->project(&scattering_slab,scattering_potential,iSlab);
				this->project(&inelastic_slab,inelastic_potential,iSlab);
				timer.lap("Project 3d");

			}

			// We need to check to make sure there is any material in this slab. Otherwise the wave function should not include any additional phase shifts.


			// Keep a clean copy of the ref without any dose filtering or water (alternate condition below)
			if (SAVE_REF && EXPOSURE_FILTER_REF == false)
			{
				ref_potential[iSlab].CopyFrom(&scattering_potential[iSlab]);
			}

			// TODO the edges should be a problem here, but maybe it is good to subtract the mean, exposure filter, then add back the mean around the edges? Can test with solvent off.

			if (SAVE_PHASE_GRATING)
			{
				std::string fileNameOUT = "with_phaseGrating_" + std::to_string(iSlab) + this->output_filename;
					MRCFile mrc_out(fileNameOUT,true);
					scattering_potential[iSlab].WriteSlices(&mrc_out,1,1);
					mrc_out.SetPixelSize(this->wanted_pixel_size);
					mrc_out.CloseFile();
			}

			if (SAVE_TO_COMPARE_JPR && ! DO_PHASE_PLATE)
			{
				// For comparing to JPR @ 0.965
				Image binImage;
				binImage.CopyFrom(&scattering_potential[iSlab]);
				binImage.Resize(JPR_SIZE,JPR_SIZE,1);
				jpr_sum_phase.AddImage(&binImage);

			}


			if (DO_EXPOSURE_FILTER == 2 && CALC_WATER_NO_HOLE == false)
			{

				timer.start("ExpFilt 2d");
			// add in the exposure filter

				scattering_potential[iSlab].ForwardFFT(true);

				ElectronDose my_electron_dose(wanted_acceleration_voltage, this->wanted_pixel_size);
				float *dose_filter = new float[scattering_potential[iSlab].real_memory_allocated/2];
				ZeroFloatArray(dose_filter, scattering_potential[iSlab].real_memory_allocated/2);


				// Normally the pre-exposure is added to each frame. Here it is taken to be the total exposure.
				my_electron_dose.CalculateDoseFilterAs1DArray(&scattering_potential[iSlab], dose_filter, current_total_exposure, current_total_exposure + dose_per_frame);

				for (long pixel_counter = 0; pixel_counter < scattering_potential[iSlab].real_memory_allocated/2; pixel_counter++)
				{
					scattering_potential[iSlab].complex_values[pixel_counter] *= dose_filter[pixel_counter];
				}

				delete [] dose_filter;

				scattering_potential[iSlab].BackwardFFT();

				if (SAVE_PHASE_GRATING_DOSE_FILTERED)
				{

					std::string fileNameOUT;
					if (iSlab < 9)
					{
						fileNameOUT = "withDoseFilter_phaseGrating_0" + std::to_string(iSlab) + this->output_filename;
					}
					else
					{
						fileNameOUT = "withDoseFilter_phaseGrating_" + std::to_string(iSlab) + this->output_filename;
					}

						MRCFile mrc_out(fileNameOUT,true);
						scattering_potential[iSlab].WriteSlices(&mrc_out,1,1);
						mrc_out.SetPixelSize(this->wanted_pixel_size);
						mrc_out.CloseFile();
				}
				timer.lap("ExpFilt 2d");
			}


			// Keep a clean copy of the ref with dose filtering
			if (SAVE_REF && EXPOSURE_FILTER_REF == true)
			{
				ref_potential[iSlab].CopyFrom(&scattering_potential[iSlab]);

			}



			if ( DO_SOLVENT && this->do3d < 1)
			{


				timer.start("Fill H20");
				// Now loop back over adding waters where appropriate
				if (DO_PRINT) {wxPrintf("Working on waters, slab %d\n",iSlab);}


				this->fill_water_potential(&current_specimen, &scattering_slab,
										   scattering_potential,inelastic_potential, &distance_slab, &water_box,rotate_waters,
						   	   	   	   	   rotated_oZ, slabIDX_start, slabIDX_end, iSlab);

				timer.lap("Fill H20");

				// Keep the running average to use as a mask
				sampled_potential.AddImage(&scattering_potential[iSlab]);

				timer.start("Taper Edges");
				this->taper_edges(scattering_potential, iSlab, false);
				this->taper_edges(inelastic_potential, iSlab, true);
				timer.lap("Taper Edges");


				if (SAVE_PHASE_GRATING_PLUS_WATER)
				{


					std::string fileNameOUT;
					if (iSlab < 9)
					{
						fileNameOUT = "withWater_phaseGrating_0" + std::to_string(iSlab) + this->output_filename;
					}
					else
					{
						fileNameOUT = "withWater_phaseGrating_" + std::to_string(iSlab) + this->output_filename;
					}

						MRCFile mrc_out(fileNameOUT,true);
						scattering_potential[iSlab].WriteSlices(&mrc_out,1,1);
						mrc_out.SetPixelSize(this->wanted_pixel_size);
						mrc_out.CloseFile();
				}


			} // end if on adding solvent
			else
			{
				// Keep the running average to use as a mask
				sampled_potential.AddImage(&scattering_potential[iSlab]);
			}




			timer.start("Deallocate Slabs");
//			scattering_slab.Deallocate();
//			inelastic_slab.Deallocate();
			timer.lap("Deallocate Slabs");


			if (DO_PHASE_PLATE)
			{

				jpr_sum_phase.AddImage(&scattering_potential[iSlab]);

			}



//			// Now apply the CTF - check that slab_oZ is doing what you intend it to TODO
//			float defocus_offset = ((slabIDX_end[iSlab]-slabIDX_start[iSlab])/2 - rotated_oZ + slabIDX_start[iSlab] + 1) * this->wanted_pixel_size;

			scattering_mass[iSlab] = scattering_potential[iSlab].ReturnSumOfRealValues(); //slab_mass * ((slabIDX_end[iSlab]-slabIDX_start[iSlab])/2 - rotated_oZ + slabIDX_start[iSlab] + 1) * this->wanted_pixel_size;
			for (int iTot = iSlab; iTot >= 0; iTot--)
			{
				scattering_total_shift[iTot] += propagator_distance[iSlab];
			}
			if (scattering_mass[iSlab] < 1e-1)
			{
				propagator_distance[iSlab] = 0.0f;
			}



		} // end loop nSlabs

		// Make the sampling mask
		Image float_img;
		Image complemenatry_mask;

		sampled_potential.Binarise(1e-3f);

		sampled_potential.ErodeBinarizedMask(7.0f);

		sampled_potential.ForwardFFT(true);
		sampled_potential.GaussianLowPassFilter(0.05f);
		sampled_potential.BackwardFFT();

		complemenatry_mask.CopyFrom(&sampled_potential);
		complemenatry_mask.MultiplyAddConstant(-1.0,1.0);


		// Apply the sampling mask
		for (int iTot = 0; iTot < nSlabs; iTot++)
		{

			float_img.CopyFrom(&complemenatry_mask);
			float_img.MultiplyByConstant(image_mean[iTot]);
			scattering_potential[iTot].MultiplyPixelWise(sampled_potential);
			scattering_potential[iTot].AddImage(&float_img);


			float_img.CopyFrom(&complemenatry_mask);
			float_img.MultiplyByConstant(inelastic_mean[iTot]);
			inelastic_potential[iTot].MultiplyPixelWise(sampled_potential);
			inelastic_potential[iTot].AddImage(&float_img);
			if (SAVE_REF)
			{
				ref_potential[iTot].MultiplyPixelWise(sampled_potential);
			}
		}

		float total_mass = 0.0f;
		float total_prod = 0.0f;
//		float fractional_surface_error = 0.02f/(float)nSlabs;
		for (int iTot = 0; iTot < nSlabs; iTot++)
		{
			total_mass += scattering_mass[iTot];
			total_prod += scattering_mass[iTot] * scattering_total_shift[iTot];
			if (DO_PRINT) wxPrintf("Mass, prj %3.3e %3.3e\n", scattering_mass[iTot], scattering_total_shift[iTot]);

//			inelastic_potential[iTot].AddConstant(fractional_surface_error);
		}

		scattering_center_of_mass = total_prod / total_mass;

		wxPrintf("\n\nFound a scattering cetner of mass at %3.3f Ang\n\n", scattering_center_of_mass);

		this->current_total_exposure += this->dose_per_frame; // increment the dose
		wxPrintf("Exposure is %3.3f for frame\n",this->current_total_exposure,iFrame+1);

		if (DO_PRINT) {wxPrintf("\n\t%ld out of bounds of %ld = percent\n\n", nOutOfBounds,number_of_non_water_atoms);}



//		#pragma omp parallel num_threads(4)
		// TODO make propagtor class
		int propagate_threads_4;
		int propagate_threads_2;

		if (this->number_of_threads > 4)
		{
		  propagate_threads_4 = 4;
		  propagate_threads_2 = 2;
		}
		else if (this->number_of_threads > 1)
		{
                  propagate_threads_4 = this->number_of_threads;
                  propagate_threads_2 = 2;
		}
		else
		{
		  propagate_threads_4 = 1;
                  propagate_threads_2 = 1;
		}


		for (int defOffset = 0; defOffset < nSlabs; defOffset++)
		{
			defocus_offset += propagator_distance[defOffset];
		}

//		float average_prop = defocus_offset / (float)nSlabs;
//		defocus_offset +=  propagator_distance[nSlabs];
//		defocus_offset /= 2.0f;
//		defocus_offset -= propagator_distance[0];
//		defocus_offset = average_prop;
//		defocus_offset = ((float)rotated_Z*this->wanted_pixel_size +  this->propagation_distance)/2;

//			inelastic_potential[iTot].AddConstant(fractional_surface_error);


		delete [] scattering_mass;
		delete [] scattering_total_shift;

		wxPrintf("%f %f %f\n",scattering_center_of_mass , propagator_distance[0] ,defocus_offset/(2.0f*(float)nSlabs));


		// todo it still isn't immediately clear which approach is correct. I would think the center of mass is what will be measured by ctffind
//		defocus_offset = defocus_offset/2.0f + propagator_distance[0];
		defocus_offset = scattering_center_of_mass - propagator_distance[0]/2.0f;

		wxPrintf("Propagator distance is %3.3e Angstroms, with offset for CTF of %3.3e Angstroms for the specimen.\n",propagator_distance[0],defocus_offset);

		wxPrintf("\n\t%ld out of bounds of %ld = percent\n\n", nOutOfBounds,number_of_non_water_atoms);

//		#pragma omp parallel num_threads(4)
//		{

		int nLoops = 1;
		if (SAVE_REF) { nLoops = 2; }
		 WaveFunctionPropagator wave_function(this->set_real_part_wave_function_in, objective_aperture, wanted_pixel_size, number_of_threads, beam_tilt_x, beam_tilt_y, DO_BEAM_TILT_FULL, propagator_distance);

		 // TODO redundancies with SetCTF and fit params, fit params etc.
		 wave_function.SetFitParams(wanted_pixel_size, wanted_acceleration_voltage, wanted_spherical_aberration, 0.0f,
				 	 	 	 	 	std::max(768,ReturnClosestFactorizedUpper(std::max(coords.GetLargestSpecimenVolume().x,coords.GetLargestSpecimenVolume().y),5,true)),
									20.0f, wanted_pixel_size*2.5, wanted_defocus_1_in_angstroms - 1000.0f, wanted_defocus_1_in_angstroms + 1000.0f, number_of_threads, 1.0f);

//		WaveFunctionPropagator wave_function(this->set_real_part_wave_function_in, wanted_amplitude_contrast, wanted_pixel_size, number_of_threads, beam_tilt_x, beam_tilt_y, DO_BEAM_TILT_FULL);

		 wxPrintf("\n\n%5.5e %5.5e\n\n", wanted_defocus_1_in_angstroms, wanted_defocus_2_in_angstroms);
		if (DO_COHERENCE_ENVELOPE)
		{
			wave_function.SetCTF(wanted_acceleration_voltage,
								  wanted_spherical_aberration,
								  wanted_defocus_1_in_angstroms,
								  wanted_defocus_2_in_angstroms,
								  wanted_astigmatism_azimuth,
								  wanted_additional_phase_shift_in_radians,
								  defocus_offset,
								  dose_rate);
		}
		else
		{

			wave_function.SetCTF(wanted_acceleration_voltage,
								  wanted_spherical_aberration,
								  wanted_defocus_1_in_angstroms,
								  wanted_defocus_2_in_angstroms,
								  wanted_astigmatism_azimuth,
								  wanted_additional_phase_shift_in_radians,
								  defocus_offset);
		}


//		wave_function.do_beam_tilt_full = true;

		// Expand the range searched to correct any defocus errors.
		float tilt_to_scale_search_range;
		if (tilt_series) tilt_to_scale_search_range = 0.0f;//fabsf(tilt_theta[iTilt]);
		else tilt_to_scale_search_range = 0.0f;

		for (int iLoop = 0; iLoop < nLoops; iLoop ++)
		{

			timer.start("Propagate WaveFunc");

			if (iFrame == 0 && iLoop == 0)
			{
				amplitude_contrast = wave_function.DoPropagation(img_frame_stack, scattering_potential,inelastic_potential, 0, nSlabs, image_mean, inelastic_mean, propagator_distance, true, tilt_to_scale_search_range);
				wxPrintf("\nFound an amplitude contrast of %3.6f\n\n", amplitude_contrast);
			}
			else
			{
				 wave_function.DoPropagation(img_frame_stack, scattering_potential,inelastic_potential, 0, nSlabs, image_mean, inelastic_mean, propagator_distance, false, tilt_to_scale_search_range);
			}

			timer.lap("Propagate WaveFunc");
			if (SAVE_PROBABILITY_WAVE && iLoop < 1)
			{
				std::string fileNameOUT = "withProbabilityWave_" + std::to_string(iFrame) + this->output_filename;
					MRCFile mrc_out(fileNameOUT,true);
					img_frame_stack[0].WriteSlices(&mrc_out,1,1);
					mrc_out.SetPixelSize(this->wanted_pixel_size);
					mrc_out.CloseFile();

			}


			if (SAVE_TO_COMPARE_JPR || DO_PHASE_PLATE && iLoop < 1)
			{
				// For comparing to JPR @ 0.965
				Image binImage;
				binImage.CopyFrom(&img_frame_stack[0]);
				if (! DO_PHASE_PLATE)
				{
					binImage.Resize(JPR_SIZE,JPR_SIZE,1);
				}
				else
				{
					coords.PadFFTToSolvent(&binImage);
				}
				jpr_sum_detector.AddImage(&binImage);

			}

			if (DO_APPLY_DQE && iLoop < 1)
			{
				if (SAVE_WITH_DQE)
				{
					std::string fileNameOUT = "withOUT_DQE_" + std::to_string(iFrame) + this->output_filename;
						MRCFile mrc_out(fileNameOUT,true);
						img_frame_stack[0].WriteSlices(&mrc_out,1,1);
						mrc_out.SetPixelSize(this->wanted_pixel_size);
						mrc_out.CloseFile();
				}

				timer.start("DQE");
				// FIXME am I only applying the DQE to the sum and not the frames? What if I'm only saving the frames?
				this->apply_sqrt_DQE_or_NTF(img_frame_stack,  0, true);
				timer.lap("DQE");
				if (SAVE_WITH_DQE)
				{
					std::string fileNameOUT = "withDQE_" + std::to_string(iFrame) + this->output_filename;
						MRCFile mrc_out(fileNameOUT,true);
						img_frame_stack[0].WriteSlices(&mrc_out,1,1);
						mrc_out.SetPixelSize(this->wanted_pixel_size);
						mrc_out.CloseFile();
				}
			}


			if (DEBUG_POISSON == false && iLoop < 1 && DO_PHASE_PLATE == false)
			{

				timer.start("Poisson Noise");
				// Next we draw from a poisson distribution and then finally apply the NTF
				Image cpp_poisson;
				cpp_poisson.Allocate(img_frame_stack[0].logical_x_dimension,img_frame_stack[0].logical_y_dimension,1,true);
				cpp_poisson.SetToConstant(0.0);

				RandomNumberGenerator my_rand(PIf);


				for (long iPixel = 0; iPixel < img_frame_stack[0].real_memory_allocated; iPixel++ )
				{
//					std::poisson_distribution<int> distribution(img_frame_stack[iFrame].real_values[iPixel]);
					cpp_poisson.real_values[iPixel] += my_rand.GetPoissonRandomSTD(img_frame_stack[0].real_values[iPixel]); //distribution(gen);

				}

				img_frame_stack[0].CopyFrom(&cpp_poisson);
				timer.lap("Poisson Noise");
			}



			if (SAVE_POISSON_PRE_NTF && iLoop < 1)
			{

				std::string fileNameOUT = "withPoisson_noNTF_" + std::to_string(iFrame) + this->output_filename;
					MRCFile mrc_out(fileNameOUT,true);
					img_frame_stack[0].WriteSlices(&mrc_out,1,1);
					mrc_out.SetPixelSize(this->wanted_pixel_size);
					mrc_out.CloseFile();

			}


			if (SAVE_REF)
			{
				if (iLoop == 0)
				{
					// Copy the img_frame_stack into the reference for storage
					ref_frame_stack[0].CopyFrom(&img_frame_stack[0]);
					for (int iSlab = 0; iSlab < nSlabs; iSlab ++)
					{
						// Copy the ref potential into scattering potential
						scattering_potential[iSlab].CopyFrom(&ref_potential[iSlab]);
					}
				}
				else
				{
					Image bufferImg;
					// Second loop so sum image is actually the ref image
					// This is stupid.
					bufferImg.CopyFrom(&img_frame_stack[0]);
					img_frame_stack[0].CopyFrom(&ref_frame_stack[0]);
					ref_frame_stack[0].CopyFrom(&bufferImg);

				}
			}


		} // loop over perfect reference


	    wxPrintf("before the destructor there are %ld non-water-atoms\n",this->number_of_non_water_atoms);
		if (SAVE_TO_COMPARE_JPR || DO_PHASE_PLATE)
		{

			std::string fileNameOUT = "compareJPR_phaseGrating_" + this->output_filename;
			MRCFile mrc_out(fileNameOUT,true);
			jpr_sum_phase.WriteSlices(&mrc_out,1,1);
			mrc_out.SetPixelSize(this->wanted_pixel_size);
			mrc_out.CloseFile();

			std::string fileNameOUT2 = "compareJPR_detector_" + this->output_filename;
			MRCFile mrc_out2(fileNameOUT2,true);
			jpr_sum_detector.WriteSlices(&mrc_out2,1,1);
			mrc_out2.SetPixelSize(this->wanted_pixel_size);
			mrc_out2.CloseFile();

		}

		timer.start("Delete potential");
		delete [] scattering_potential;
		if (SAVE_REF)
		{
			delete [] ref_potential;
		}

		delete [] propagator_distance;
		timer.lap("Delete potential");
		defocus_offset = 0;

		timer.start("Update parameters");
		if (ONLY_SAVE_SUMS) parameters.position_in_stack = iTilt + 1;
		else parameters.position_in_stack = (iTilt*(int)number_of_frames) + iFrame + 1;
		parameters.psi = tilt_psi[iTilt];
		parameters.theta = tilt_theta[iTilt];
		parameters.phi = tilt_phi[iTilt];
//		if (this->stdErr != 0)
//		{
			parameters.x_shift = shift_x[iTilt]; // shx
			parameters.y_shift = shift_y[iTilt]; // shy
//		}
		// TODO make sure this includes any random changes that might gave
		parameters.defocus_1 = wanted_defocus_1_in_angstroms;
		parameters.defocus_2 = wanted_defocus_2_in_angstroms;
		parameters.defocus_angle = wanted_astigmatism_azimuth;
		parameters.phase_shift = wanted_additional_phase_shift_in_radians;
		parameters.pixel_size = wanted_pixel_size; // This includes any scaling due to mag changes.
		parameters.microscope_voltage_kv = wanted_acceleration_voltage;
		parameters.microscope_spherical_aberration_mm = wanted_spherical_aberration;
		parameters.amplitude_contrast = amplitude_contrast;
		parameters.particle_group = iTilt + 1; // FIXME, this is not the intended behavior after switching from tracking frame_number -> particle group
		parameters.pre_exposure =  current_total_exposure - dose_per_frame;
		parameters.total_exposure = current_total_exposure;

		if ( ( ONLY_SAVE_SUMS && iFrame < 1 ) || ( ! ONLY_SAVE_SUMS ) )
		{
			parameter_star.all_parameters.Add(parameters);
		}


		output_image_stack[iTilt*(int)number_of_frames + iFrame].CopyFrom(&img_frame_stack[0]);

		if (SAVE_REF)
		{
			output_reference_stack[iTilt*(int)number_of_frames + iFrame].CopyFrom(&ref_frame_stack[0]);
		}


		timer.lap("Update parameters");
    } // end of loop over frames

	timer.start("Final mods and save");



	// If we aren't simulating a tilt-series, the exposure should be reset, and a new distribution of waters created.
	if ( ! tilt_series )
	{
		water_box.number_of_waters = 0;
		this->current_total_exposure = this->pre_exposure;
	}


	if (DO_EXPOSURE_FILTER_FINAL_IMG )
	{
		// sum the frames
		float final_img_exposure;
		float *dose_filter;
		float *dose_filter_sum_of_squares = new float[output_image_stack[0].real_memory_allocated / 2];
		ZeroFloatArray(dose_filter_sum_of_squares, output_image_stack[0].real_memory_allocated/2);
		ElectronDose my_electron_dose(wanted_acceleration_voltage, this->wanted_pixel_size);

		float local_pre_exposure = pre_exposure;


		for (int iFrame = 0; iFrame < this->number_of_frames; iFrame++)
		{

			dose_filter = new float[output_image_stack[iTilt*(int)number_of_frames + iFrame].real_memory_allocated / 2];
			for (long pixel_counter = 0; pixel_counter < output_image_stack[iTilt*(int)number_of_frames + iFrame].real_memory_allocated / 2; pixel_counter++)
			{
				dose_filter[pixel_counter] = 0.0;
			}

			output_image_stack[iTilt*(int)number_of_frames + iFrame].ForwardFFT(true);
			if (EXPOSURE_FILTER_REF && SAVE_REF)
			{
				output_reference_stack[iTilt*(int)number_of_frames + iFrame].ForwardFFT(true);
			}

			my_electron_dose.CalculateDoseFilterAs1DArray(&output_image_stack[iTilt*(int)number_of_frames + iFrame], dose_filter, local_pre_exposure, local_pre_exposure + this->dose_per_frame );
			local_pre_exposure += this->dose_per_frame;


			for (long pixel_counter = 0; pixel_counter < output_image_stack[iTilt*(int)number_of_frames + iFrame].real_memory_allocated / 2; pixel_counter++)
			{

				output_image_stack[iTilt*(int)number_of_frames + iFrame].complex_values[pixel_counter] *= dose_filter[pixel_counter];
				if (EXPOSURE_FILTER_REF && SAVE_REF)
				{
					output_reference_stack[iTilt*(int)number_of_frames + iFrame].complex_values[pixel_counter] *=  dose_filter[pixel_counter];
				}

				dose_filter_sum_of_squares[pixel_counter] += powf(dose_filter[pixel_counter],2);
			}



			delete [] dose_filter;

			if (ONLY_SAVE_SUMS)
			{
				output_image_stack[iTilt*(int)number_of_frames].AddImage(&output_image_stack[iTilt*(int)number_of_frames + iFrame]);
				if (SAVE_REF)
				{
				  output_reference_stack[iTilt*(int)number_of_frames].AddImage(&output_reference_stack[iTilt*(int)number_of_frames + iFrame]);
				}
			}




		}

		int exposure_filter_range;
		if (ONLY_SAVE_SUMS) exposure_filter_range = 1;
		else exposure_filter_range = (int)number_of_frames;

		for (int iFrame = 0; iFrame < exposure_filter_range; iFrame ++)
		{
			for (long pixel_counter = 0; pixel_counter < output_image_stack[iTilt*(int)number_of_frames + iFrame].real_memory_allocated / 2; pixel_counter++)
			{
				output_image_stack[iTilt*(int)number_of_frames + iFrame].complex_values[pixel_counter] /= sqrtf(dose_filter_sum_of_squares[pixel_counter]);
				if (SAVE_REF)
				{
					output_reference_stack[iTilt*(int)number_of_frames + iFrame].complex_values[pixel_counter] /= sqrtf(dose_filter_sum_of_squares[pixel_counter]);
				}

			}
		}



		delete [] dose_filter_sum_of_squares;

	}
	else if (ONLY_SAVE_SUMS)
	{
		for (int iFrame = 1; iFrame < this->number_of_frames; iFrame++)
		{
			output_image_stack[iTilt*(int)number_of_frames].AddImage(&output_image_stack[iTilt*(int)number_of_frames + iFrame]);
			if (SAVE_REF)
			{
			  output_reference_stack[iTilt*(int)number_of_frames].AddImage(&output_reference_stack[iTilt*(int)number_of_frames + iFrame]);
			}
		}

	} // if exposure filter




    } // end of loop over tilts

    // Write out the star file, TODO should the ranges be explicitly defined?
	parameter_star.WriteTocisTEMStarFile(parameter_star_file_name);


    if (DO_PRINT) {wxPrintf("%s\n",this->output_filename);}

	bool over_write = true;
	MRCFile mrc_out_final(this->output_filename,over_write);

    std::string fileNameRefSum = "perfRef_" + this->output_filename;
	MRCFile mrc_ref_final;
	if (SAVE_REF)
	{
		mrc_ref_final.OpenFile(fileNameRefSum,over_write);
	}

	std::string fileNameREF = "ref_" + this->output_filename;

    std::string fileNameTiltSum = "tiltSum_" + this->output_filename;
    MRCFile mrc_tlt_final;

    if (this->doParticleStack <= 0)
    {
    	mrc_tlt_final.OpenFile(fileNameTiltSum,over_write);
	}

	if (DO_PRINT) {wxPrintf("\n\nnTilts %d N_FRAMES %d\n\n",nTilts,myroundint(this->number_of_frames));}

	Curve whitening_filter;
	Curve number_of_terms;
	CTF my_ctf;

	/////////// START NEW COMBINED
	int current_tilt_sub_frame = 1;
	int current_tilt_sum_saved = 1;
	int3 outputSize;
	Image tilt_sum;
	Image ref_sum;



	// Make the particle Square
	if (wanted_output_size > 0)
	{
		outputSize.x = wanted_output_size;
		outputSize.y = wanted_output_size;
	}
	else
	{
		outputSize.x = coords.ReturnLargestDimension(0);
		outputSize.y = coords.ReturnLargestDimension(1);
//			outputSize = std::max(coords.ReturnLargestDimension(0), coords.ReturnLargestDimension(1));
	}

	// This assumes all tilts have been made the same size (which they should be.)
//		tilt_sum.Allocate(xDIM,yDIM, 1);
	wxPrintf("outputSize %d %d\n",outputSize.x, outputSize.y);
	tilt_sum.Allocate(outputSize.x,outputSize.y, 1);
	tilt_sum.SetToConstant(0.0);

	if (SAVE_REF)
	{
//			ref_sum.Allocate(xDIM,yDIM, 1);
		ref_sum.Allocate(outputSize.x,outputSize.y, 1);
		ref_sum.SetToConstant(0.0);
	}



	if (WHITEN_IMG)
	{
		whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((outputSize.x / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
		number_of_terms.SetupXAxis( 0.0, 0.5 * sqrtf(2.0), int((outputSize.x / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
	}


//	int final_tilt_range;
//	if (ONLY_SAVE_SUMS) final_tilt_range = nTilts;
//
//	else final_tilt_range = nTilts * (int)this->number_of_frames;
	int final_tilt_inc = 1;
	int n_tilts_saved = 0;
	if (ONLY_SAVE_SUMS) final_tilt_inc = this->number_of_frames;


	for (iTilt=0; iTilt < nTilts*(int)this->number_of_frames; iTilt+=final_tilt_inc)
	{

		wxPrintf("save sums inc saved i %d %d %d %d\n", ONLY_SAVE_SUMS, final_tilt_inc, n_tilts_saved, iTilt);
		// I am getting some nans on occassion, but so far they only show up in big expensive calcs, so add a nan check and print info to see if
		// the problem can be isolated.
		if (output_image_stack[iTilt].HasNan() == true)
		{
			wxPrintf("Frame %d / %d has NaN values, trashing it\n",iTilt,nTilts*(int)this->number_of_frames);
			output_image_stack[iTilt].SetToConstant(0.0f);
			continue;
		}


		if (CORRECT_CTF)
		{

			my_ctf.Init(parameter_star.ReturnMicroscopekV(n_tilts_saved),
						parameter_star.ReturnMicroscopeCs(n_tilts_saved),
						parameter_star.ReturnAmplitudeContrast(n_tilts_saved),
						parameter_star.ReturnDefocus1(n_tilts_saved),
						parameter_star.ReturnDefocus2(n_tilts_saved),
						parameter_star.ReturnDefocusAngle(n_tilts_saved),
						parameter_star.ReturnPhaseShift(n_tilts_saved),
						parameter_star.ReturnPixelSize(n_tilts_saved));

		}

		coords.PadToWantedSize(&output_image_stack[iTilt], wanted_output_size);
		if (SAVE_REF) {	coords.PadToWantedSize(&output_reference_stack[iTilt],wanted_output_size); }



		if (WHITEN_IMG)
		{

			output_image_stack[iTilt].ForwardFFT(true);
			output_image_stack[iTilt].ZeroCentralPixel();
			output_image_stack[iTilt].Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
			whitening_filter.SquareRoot();
			whitening_filter.Reciprocal();
			whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue());

			//whitening_filter.WriteToFile("/tmp/filter.txt");
			output_image_stack[iTilt].ApplyCurveFilter(&whitening_filter);
			output_image_stack[iTilt].ZeroCentralPixel();


			output_image_stack[iTilt].DivideByConstant(sqrtf(output_image_stack[iTilt].ReturnSumOfSquares()));
			output_image_stack[iTilt].BackwardFFT();

			if (SAVE_REF)
			{

				output_reference_stack[iTilt].ForwardFFT(true);
				output_reference_stack[iTilt].ZeroCentralPixel();
				output_reference_stack[iTilt].ApplyCurveFilter(&whitening_filter);
				output_reference_stack[iTilt].ZeroCentralPixel();
				if (CORRECT_CTF) {output_reference_stack[iTilt].ApplyCTF(my_ctf,false,false) ;} // TODO is this the right spot to put this?
				output_reference_stack[iTilt].DivideByConstant(sqrt(output_reference_stack[iTilt].ReturnSumOfSquares()));
				output_reference_stack[iTilt].BackwardFFT();
			}

		}
		else
		{

			if (CORRECT_CTF)
			{
				output_image_stack[iTilt].ForwardFFT(true);
				output_image_stack[iTilt].ApplyCTF(my_ctf,false,false) ;
				output_image_stack[iTilt].BackwardFFT();

			} // TODO is this the right spot to put this?

		}

		output_image_stack[iTilt].WriteSlices(&mrc_out_final,1+n_tilts_saved,1+n_tilts_saved);
		if (SAVE_REF)
		{
			output_reference_stack[iTilt].WriteSlices(&mrc_ref_final,1+n_tilts_saved,1+n_tilts_saved);
		}

		n_tilts_saved++;

	}



   ////////////// START FORMER TILT

	// TODO what about different pixel sizes?
	mrc_out_final.SetPixelSize(this->wanted_pixel_size);
	mrc_out_final.CloseFile();

	mrc_tlt_final.SetPixelSize(this->wanted_pixel_size);
	mrc_tlt_final.CloseFile();

//	this->parameter_file.Close();


	delete [] tilt_psi;
	delete [] tilt_theta;
	delete [] tilt_phi;
	delete [] shift_x;
	delete [] shift_y;
	delete [] shift_z;


    delete [] img_frame_stack;

 //   delete noise_dist;
    timer.lap("Final mods and save");
    timer.print_times();

}

void SimulateApp::calc_scattering_potential(const PDB * current_specimen,
											Image *scattering_slab,
											Image *inelastic_slab,
											Image *distance_slab,
											RotationMatrix rotate_waters,
											float rotated_oZ,
											int *slabIDX_start,
											int *slabIDX_end,
											int iSlab)

// The if conditions needed to have water and protein in the same function
// make it too complicated and about 10x less parallel friendly.
{


	int  z_low = slabIDX_start[iSlab] - size_neighborhood;
	int  z_top = slabIDX_end[iSlab] + size_neighborhood;

	float slab_half_thickness_angstrom = (slabIDX_end[iSlab] - slabIDX_start[iSlab] + 1)*this->wanted_pixel_size/2.0f;

	// Private
	AtomType atom_id;
	float element_inelastic_ratio;
	float bFactor;
	float bPlusB[5];
	float radius;
	float ix(0), iy(0), iz(0);
	float dx(0), dy(0), dz (0);
	float x1(0.0f), y1(0.0f), z1(0.0f);
	float x2(0.0f), y2(0.0f), z2(0.0f);
	int indX(0), indY(0), indZ(0);
	float sx(0), sy(0), sz(0);
	float xDistSq(0),zDistSq(0),yDistSq(0);
	int iLim, jLim, kLim;
	int iGaussian;
	float water_offset;
	int cubic_vol = (int)powf(size_neighborhood*2+1,3);
	long atoms_added_idx[cubic_vol];
	float atoms_values_tmp[cubic_vol];
	float atoms_distances_tmp[cubic_vol];

	int n_atoms_added;
	float pixel_offset = 0.5f;
	float bfX(0), bfY(0), bfZ(0);





	// TODO experiment with the scheduling. Until the specimen is consistently full, many consecutive slabs may have very little work for the assigned threads to handle.
	#pragma omp parallel for num_threads(this->number_of_threads) private(atom_id, bFactor, bPlusB, radius, ix,iy,iz,x1,x2,y1,y2,z1,z2,indX,indY,indZ,sx,sy,sz,xDistSq,yDistSq,zDistSq,iLim,jLim,kLim, iGaussian,water_offset,atoms_values_tmp,atoms_added_idx,atoms_distances_tmp,n_atoms_added,pixel_offset,bfX,bfY,bfZ)
	for (long current_atom = 0; current_atom < this->number_of_non_water_atoms; current_atom++)
	{

		n_atoms_added = 0;

		atom_id = current_specimen->my_atoms.Item(current_atom).atom_type;
		element_inelastic_ratio = sqrtf(inelastic_scalar / sp.ReturnAtomicNumber(atom_id)); // Reimer/Ross_Messemer 1989
		bFactor = return_bfactor(current_specimen->my_atoms.Item(current_atom).bfactor);




		corners R;
//		Coords coords;

		float3 origin = coords.ReturnOrigin((PaddingStatus)solvent);
		int3 size  	= coords.GetSolventPadding();

		if (DO_BEAM_TILT_FULL)
		{
			// Shift atoms positions in X/Y so that they end up being projected at the correct position at the BOTTOM of the slab
			// TODO save some comp and pre calc the factors on the right


			x1 = current_specimen->my_atoms.Item(current_atom).x_coordinate;
			y1 = current_specimen->my_atoms.Item(current_atom).y_coordinate;
			z1 = current_specimen->my_atoms.Item(current_atom).z_coordinate + (rotated_oZ - slabIDX_start[iSlab])*wanted_pixel_size;



//			z1 = current_specimen->my_atoms.Item(current_atom).z_coordinate - slabIDX_start[iSlab]*wanted_pixel_size;

			x1 += (z1)*beam_tilt_z_X_component;
			y1 += (z1)*beam_tilt_z_Y_component;

			// Convert atom origin to pixels and shift by volume origin to get pixel coordinates. Add 0.5 to place origin at "center" of voxel
			dx = modff(origin.x + (x1 / wanted_pixel_size + pixel_offset), &ix);
			dy = modff(origin.y + (y1 / wanted_pixel_size + pixel_offset), &iy);
			// Notes this is the unmodified dz (not using z1)

			dz = modff(rotated_oZ 			    + ((float)current_specimen->my_atoms.Item(current_atom).z_coordinate / wanted_pixel_size + pixel_offset), &iz);

		}
		else
		{

			// Convert atom origin to pixels and shift by volume origin to get pixel coordinates. Add 0.5 to place origin at "center" of voxel
			dx = modff(origin.x + (current_specimen->my_atoms.Item(current_atom).x_coordinate / wanted_pixel_size + pixel_offset), &ix);
			dy = modff(origin.y + (current_specimen->my_atoms.Item(current_atom).y_coordinate / wanted_pixel_size + pixel_offset), &iy);
			dz = modff(rotated_oZ 			    + (current_specimen->my_atoms.Item(current_atom).z_coordinate / wanted_pixel_size + pixel_offset), &iz);

		}


		// With the correct pixel indices in ix,iy,iz now subtract off the 0.5
		dx -= pixel_offset;
		dy -= pixel_offset;
		dz -= pixel_offset;

		#pragma omp simd
		for (iGaussian = 0; iGaussian < 5 ; iGaussian++)
		{
			bPlusB[iGaussian] = 2*PIf/sqrt(bFactor+sp.ReturnScatteringParamtersB(atom_id,iGaussian));
		}

		// For accurate calculations, a thin slab is used, s.t. those atoms outside are the majority. Check this first, but account for the size of the atom, as it may reside in more than one slab.
//		if (iz <= slabIDX_end[iSlab]  && iz >= slabIDX_start[iSlab])
		if (iz <= z_top && iz >= z_low)
		{


		for (sx = -size_neighborhood; sx <= size_neighborhood ; sx++)
		{
			indX = ix + sx;
			R.x1 = (sx - pixel_offset - dx) * this->wanted_pixel_size;
			R.x2 = (R.x1 + this->wanted_pixel_size);
			xDistSq = (sx - dx) * wanted_pixel_size;
			xDistSq *= xDistSq;

			for (sy = -size_neighborhood; sy <= size_neighborhood; sy++)
			{
				indY = iy + sy ;
				R.y1 = (sy - pixel_offset - dy) * this->wanted_pixel_size;
				R.y2 = (R.y1 + this->wanted_pixel_size);
				yDistSq = (sy - dy) * wanted_pixel_size;
				yDistSq *= yDistSq;

				for (sz = -size_neighborhood; sz <= size_neighborhood ; sz++)
				{
					indZ = iz  + sz;
					R.z1 = (sz - pixel_offset - dz) * this->wanted_pixel_size;
					R.z2 = (R.z1 + this->wanted_pixel_size);
					zDistSq = (sz - dz) * wanted_pixel_size;
					zDistSq *= zDistSq;
					// Put Z condition first since it should fail most often (does c++ fall out?)

					if (indZ <= slabIDX_end[iSlab]  && indZ >= slabIDX_start[iSlab] && indX > 0 && indY > 0 && indX < size.x && indY < size.y)
					{
						// Calculate the scattering potential



							atoms_added_idx[n_atoms_added] = scattering_slab->ReturnReal1DAddressFromPhysicalCoord(indX,indY,indZ - slabIDX_start[iSlab]);
							atoms_values_tmp[n_atoms_added] = return_scattering_potential(R, bPlusB, atom_id);
							atoms_distances_tmp[n_atoms_added] = xDistSq + yDistSq + zDistSq;

//							scattering_slab->real_values[atoms_added_idx[n_atoms_added]] += temp_potential;
							n_atoms_added++;


					}

				} // end of loop over the neighborhood Z
			} // end of loop over the neighborhood Y
		} // end of loop over the neighborhood X

//		wxPrintf("Possible positions added %3.3e %\n", 100.0f* (float)n_atoms_added/(float)cubic_vol);

			for (int iIDX = 0; iIDX < n_atoms_added-1; iIDX++)
			{
				#pragma omp atomic update
				scattering_slab->real_values[atoms_added_idx[iIDX]] += (atoms_values_tmp[iIDX]);
				// This is the value for 100 KeV --> scale (if needed) the final projected density
				inelastic_slab->real_values[atoms_added_idx[iIDX]]  += element_inelastic_ratio * atoms_values_tmp[iIDX];
				distance_slab->real_values[atoms_added_idx[iIDX]] = std::min(distance_slab->real_values[atoms_added_idx[iIDX]],atoms_distances_tmp[iIDX]);
			}

		}// if statment into neigh

	} // end loop over atoms



}

void SimulateApp::calc_water_potential(Image *projected_water, AtomType wanted_atom_id)

// The if conditions needed to have water and protein in the same function
// make it too complicated and about 10x less parallel friendly.
{

	long current_atom;
	float bFactor;
	float radius;
	float water_lead_term;



	// Private variables:
	AtomType atom_id;
	if (DO_PHASE_PLATE)
	{
		atom_id = carbon;
		bFactor = 0.25f * MIN_BFACTOR;
		water_lead_term = this->lead_term;
	}
	else
	{
		bFactor = 0.25f * MIN_BFACTOR;
		water_lead_term = this->lead_term;

		atom_id = SOLVENT_TYPE;

		if (atom_id == oxygen)
		{
			water_lead_term *= water_oxygen_ratio;
		}
	}

	if (DO_PRINT) 	 {wxPrintf("\nbFactor and leadterm and wavelength %f %4.4e %4.4e %4.4e\n",bFactor,water_lead_term,lead_term,wavelength);}


	float bPlusB[5];
	float ix(0), iy(0), iz(0);
	float dx(0), dy(0), dz (0);
	float indX(0), indY(0), indZ(0);
	float sx(0), sy(0), sz(0);
	float xLow(0),xTop(0),yLow(0),yTop(0),zLow(0),zTop(0);
    int iLim, jLim, kLim;
	int iGaussian;
	long real_address;

	float center_offset = 0.0f;
	float  pixel_offset = 0.5f;


	 // for ions this should be a bigger window but that complicaes things for now FIXME
	for (iGaussian = 0; iGaussian < 5; iGaussian++)
	{
		bPlusB[iGaussian] = 2*PIf/sqrtf(bFactor+sp.ReturnScatteringParamtersB(atom_id,iGaussian));
	}


	int nSubPixCenter = 0;

	for (int iSubPixZ = -SUB_PIXEL_NEIGHBORHOOD; iSubPixZ <= SUB_PIXEL_NEIGHBORHOOD ; iSubPixZ++)
	{
		dz = ((float)iSubPixZ / (float)(SUB_PIXEL_NEIGHBORHOOD*2+2) + center_offset);

	for (int iSubPixY = -SUB_PIXEL_NEIGHBORHOOD; iSubPixY <= SUB_PIXEL_NEIGHBORHOOD ; iSubPixY++)
	{
		dy =  ((float)iSubPixY / (float)(SUB_PIXEL_NEIGHBORHOOD*2+2) + center_offset);

		for (int iSubPixX = -SUB_PIXEL_NEIGHBORHOOD; iSubPixX <= SUB_PIXEL_NEIGHBORHOOD ; iSubPixX++)
		{
			dx =  ((float)iSubPixX / (float)(SUB_PIXEL_NEIGHBORHOOD*2+2) + center_offset);


			int n_atoms_added = 0;
			double temp_potential_sum = 0.0;

			corners R;


			for (sx = -size_neighborhood_water ; sx <= size_neighborhood_water ; sx++)
			{
				R.x1 = (sx - pixel_offset - dx) * this->wanted_pixel_size;
				R.x2 = (R.x1 + this->wanted_pixel_size);

				for (sy = -size_neighborhood_water ; sy <=  size_neighborhood_water; sy++)
				{

					R.y1 = (sy - pixel_offset - dy) * this->wanted_pixel_size;
					R.y2 = R.y1 + this->wanted_pixel_size;

					// We want the projected potential so zero here
					temp_potential_sum = 0.0;

					for (sz = -size_neighborhood_water  ; sz <= size_neighborhood_water  ; sz++)
					{
						R.z1 = (sz - pixel_offset - dz) * this->wanted_pixel_size;
					    R.z2 = R.z1 + this->wanted_pixel_size;


							temp_potential_sum += (double)return_scattering_potential(R, bPlusB, atom_id);

					} // end of loop over Z

					projected_water[nSubPixCenter].real_values[projected_water[nSubPixCenter].ReturnReal1DAddressFromPhysicalCoord(sx+size_neighborhood_water,sy+size_neighborhood_water,0)] = (float)temp_potential_sum * water_scaling ;


			} // end of loop Y
		} // end loop X Neighborhood




		nSubPixCenter++;

			} // inner SubPixX
		}  // mid SubPiY
	} // outer SubPixZ


	if (SAVE_PROJECTED_WATER)
	{
		std::string fileNameOUT = "projected_water.mrc";
		MRCFile mrc_out(fileNameOUT,true);
		for (int iWater = 0; iWater < nSubPixCenter  ; iWater++)
		{
			projected_water[iWater].WriteSlices(&mrc_out,iWater+1,iWater+1);
		}

		mrc_out.SetPixelSize(this->wanted_pixel_size);
		mrc_out.CloseFile();
	}


}

void SimulateApp::fill_water_potential(const PDB * current_specimen,Image *scattering_slab, Image *scattering_potential,
												  Image *inelastic_potential, Image *distance_slab, Water *water_box, RotationMatrix rotate_waters,
													   float rotated_oZ, int *slabIDX_start, int *slabIDX_end, int iSlab)
{

	long current_atom;
	long nWatersAdded = 0;

	float radius;


	float bPlusB[5];
	float ix(0), iy(0), iz(0);
	float dx(0), dy(0), dz (0);
	float x1(0.0f), y1(0.0f), z1(0.0f);
	int indX(0), indY(0), indZ(0);
	int sx(0), sy(0);
	int iSubPixX;
	int iSubPixY;
	int iSubPixZ;
	int iSubPixLinearIndex;
	float avg_cutoff;
	float current_weight = 0.0f;
	float current_potential = 0.0f;
	float current_distance = 0.0f;
	int iPot;
	const float pixel_offset = 0.5f;

	timer.start("water_pre");

	if( CALC_WATER_NO_HOLE || DO_PHASE_PLATE)
	{
		avg_cutoff = 10000; // TODO check this can't break, I think the potential should always be < 1
	}
	else
	{
		avg_cutoff = this->average_at_cutoff[0];
	}
	const int upper_bound = (size_neighborhood_water*2+1);
	const int numel_water = upper_bound*upper_bound;


//	float3 origin = coords.ReturnOrigin((PaddingStatus)solvent);
//	int3 size  	= coords.GetSolventPadding();
	Image projected_water_atoms;
	projected_water_atoms.Allocate(scattering_slab->logical_x_dimension,scattering_slab->logical_y_dimension,1);
	projected_water_atoms.SetToConstant(0.0f);

	Image water_mask;
	if (add_mean_water_potential)
	{
		water_mask.Allocate(scattering_slab->logical_x_dimension,scattering_slab->logical_y_dimension,scattering_slab->logical_z_dimension);
		water_mask.SetToConstant(1.0f);

		for (long iVoxel = 0; iVoxel < scattering_slab->real_memory_allocated; iVoxel++)
		{
			current_potential = scattering_slab->real_values[iVoxel];


			if (DO_PHASE_PLATE)
			{
				current_weight = 1;
			}
			else
			{

				for (iPot = N_WATER_TERMS - 1; iPot >=0; iPot--)
				{
					if (current_potential < this->average_at_cutoff[iPot])
					{

						current_weight = this->water_weight[iPot];
						break;
					}
				}
			}

			water_mask.real_values[iVoxel] *= current_weight;
		}

	}

	// To compare the thread block ordering, undo with schedule(dynamic,1)
	// schedule(static, water_box->number_of_waters /number_of_threads )

//	timer.lap("water_pre");

	long n_waters_ignored = 0;
	#pragma omp parallel for num_threads(this->number_of_threads) schedule(static,water_box->number_of_waters/number_of_threads) private(radius,ix,iy,iz,dx,dy,dz,x1,y1,z1,indX,indY,indZ,sx,sy,iSubPixX,iSubPixY,iSubPixLinearIndex,n_waters_ignored, current_weight, current_distance, current_potential)
	for (int current_atom = 0; current_atom < water_box->number_of_waters; current_atom++)
	{

		int int_x,int_y,int_z;
		double temp_potential_sum = 0;
		double norm_value = 0;


		water_box->ReturnCenteredCoordinates(current_atom,dx,dy,dz);

		rotate_waters.RotateCoords(dx, dy, dz, ix, iy, iz);

		// The broadest contition for exclusion is being outside the slab, so calculate that first to avoid redundant calcs.


		if (DO_BEAM_TILT_FULL)
		{
			// Shift atoms positions in X/Y so that they end up being projected at the correct position at the BOTTOM of the slab
			// TODO save some comp and pre calc the factors on the right

			z1 = iz + (rotated_oZ - slabIDX_start[iSlab]);

			ix += z1*beam_tilt_z_X_component;
			iy += z1*beam_tilt_z_Y_component;

		}

		dx = modff(ix + scattering_slab->logical_x_dimension/2 + pixel_offset, &ix) - pixel_offset;
		dy = modff(iy + scattering_slab->logical_y_dimension/2 + pixel_offset, &iy) - pixel_offset;
		dz = modff(iz + rotated_oZ + pixel_offset, &iz) - pixel_offset; // Why am I subtracting here? Should it be an add? TODO

		// Convert these once to avoid type conversion in every loop
		int_x = myroundint(ix);
		int_y = myroundint(iy);
		int_z = myroundint(iz);



		// FIXME confirm the +1 on the sub pix makes sense
		// FIXME printing out an error I'm getting iSubPixLinearIndex = -38 w/ iSubpixXYZ = 2,2,4 (should be 62) suspect water_edge having been declared as float. If this fixes, just make two vars.
		float water_edge = ((float)SUB_PIXEL_NEIGHBORHOOD*2) + 1.0f;
		iSubPixX = (int)trunc(dx * water_edge) + SUB_PIXEL_NEIGHBORHOOD;
		iSubPixY = (int)trunc(dy * water_edge) + SUB_PIXEL_NEIGHBORHOOD;
		iSubPixZ = (int)trunc(dz * water_edge) + SUB_PIXEL_NEIGHBORHOOD;

		iSubPixLinearIndex = int(water_edge * water_edge * iSubPixZ) + int(water_edge * iSubPixY) + (int)iSubPixX;

//
//if (ReturnThreadNumberOfCurrentThread()==0) timer.lap("w_center");

		if ( int_z >= slabIDX_start[iSlab] && int_z  <= slabIDX_end[iSlab] && int_x-1 > 0 && int_y-1 > 0 && int_x-1 < scattering_slab->logical_x_dimension && int_y-1 < scattering_slab->logical_y_dimension)
		{

			if (ReturnThreadNumberOfCurrentThread()==0) timer.start("w_weight");

			current_potential = scattering_slab->ReturnRealPixelFromPhysicalCoord(int_x-1,int_y-1,int_z - slabIDX_start[iSlab]);
			current_distance  = distance_slab->ReturnRealPixelFromPhysicalCoord(int_x-1,int_y-1,int_z - slabIDX_start[iSlab]);

			// FIXME
			if (DO_PHASE_PLATE)
			{
				current_weight = 1;
			}
			else
			{
				if (current_distance < DISTANCE_INIT)
				{

					current_distance = sqrtf(current_distance);
					current_weight = return_hydration_weight(current_distance);
//					wxPrintf("Hydration weight at distance r is %3.3e %3.3f\n",current_weight,current_distance);

				}
				else
				{
					current_weight = 1.0f;
				}

			}


//			wxPrintf("This water scale is %3.3f\n",current_weight);
			if (ReturnThreadNumberOfCurrentThread()==0) timer.start("w_neigh");

			for (sy = 0; sy <  upper_bound ; sy++ )
			{
				indY = int_y - upper_bound + sy + size_neighborhood_water + 1;
				for (sx = 0;  sx < upper_bound ; sx++ )
				{
					indX = int_x -upper_bound + sx + size_neighborhood_water + 1;
					// Even with the periodic boundaries checked in shake, the rotation may place waters out of bounds. TODO this is true for non-waters as well.
					if (indX >= 0 && indX < projected_water_atoms.logical_x_dimension && indY >= 0 && indY < projected_water_atoms.logical_y_dimension)
					{

//						wxPrintf("%d %d ,%d,  %d %d %d sx sy idx\n", sx, sy, iSubPixLinearIndex, iSubPixX, iSubPixY, iSubPixZ);
//						if (iSubPixLinearIndex < 0 || iSubPixLinearIndex > SUB_PIXEL_NeL -1)
//						{
//							wxPrintf("%d %d ,%d,  %d %d %d sx sy idx\n", sx, sy, iSubPixLinearIndex, iSubPixX, iSubPixY, iSubPixZ);
//							continue;
//						}
						if (iSubPixLinearIndex >= 0 && iSubPixLinearIndex <= SUB_PIXEL_NeL -1)
						{
							if (ReturnThreadNumberOfCurrentThread()==0) timer.start("omp");

							#pragma omp atomic update
							projected_water_atoms.real_values[projected_water_atoms.ReturnReal1DAddressFromPhysicalCoord(indX,indY,0)] +=
								(current_weight*this->projected_water[iSubPixLinearIndex].ReturnRealPixelFromPhysicalCoord(sx,sy,0)); // TODO could I land out of bounds?] += projected_water_atoms[iSubPixLinearIndex].real_values[iWater];
							if (ReturnThreadNumberOfCurrentThread()==0) timer.lap("omp");

						}

//						wxPrintf("Current Water %3.3e\n",current_weight*this->projected_water[iSubPixLinearIndex].real_values[this->projected_water[iSubPixLinearIndex].ReturnReal1DAddressFromPhysicalCoord(sx,sy,0)]);

					}

				}

			}



		}



	} // end loop over atoms


//	this->project(&volume_water,projected_water,0);

//	if (DO_PRINT) {wxPrintf("\nnWaters %ld added (%2.2f%%) of total on slab %d\n",nWatersAdded,100.0f*(float)nWatersAdded/(float)water_box->number_of_waters, iSlab);}
	if (DO_PRINT)  {
		this->total_waters_incorporated += nWatersAdded;
		wxPrintf("Water occupies %2.2f percent of the 3d, total added = %2.0f of %ld (%2.2f)\n",
			100*nWatersAdded/((double)water_box->number_of_waters),this->total_waters_incorporated,water_box->number_of_waters,100*this->total_waters_incorporated/(double)(water_box->number_of_waters));
	}

	if (ReturnThreadNumberOfCurrentThread()==0) timer.start("w_after");

	MRCFile mrc_out;

	if (SAVE_WATER_AND_OTHER)
	{
		std::string fileNameOUT = "tmpWat_prj_comb" + std::to_string(iSlab) + ".mrc";
		// Only open the file if we are going to use it.
		mrc_out.OpenFile(fileNameOUT,true);
		projected_water_atoms.WriteSlices(&mrc_out,1,1);
		scattering_potential[iSlab].WriteSlices(&mrc_out,2,2);

	}



	if (add_mean_water_potential)
	{
		Image *tmpPrj;
		tmpPrj = new Image[1];

		// Rather than using the mean inner potential, which we don't really have, project the water slab and normalize based on
		// the number of sample points in Z. Then calc a 2d average of the projected potential
		tmpPrj[0].Allocate(scattering_potential[iSlab].logical_x_dimension,scattering_potential[iSlab].logical_y_dimension,1);
		tmpPrj[0].SetToConstant(0.0f);
		this->project(&water_mask,tmpPrj,0);
		tmpPrj->DivideByConstant((float)water_mask.logical_z_dimension);
		float mean_water_value = projected_water_atoms.ReturnAverageOfRealValues(-1.0,false);// / water_mask.logical_z_dimension;
		projected_water_atoms.CopyFrom(tmpPrj);
		projected_water_atoms.MultiplyByConstant(mean_water_value);



//		tmpPrj[0].QuickAndDirtyWriteSlice("checkWater.mrc",1,true);
//		projected_water_atoms.QuickAndDirtyWriteSlice("checkPrjWater.mrc",1,true);

//		scattering_potential[iSlab].QuickAndDirtyWriteSlice("atoms.mrc",1,true);
		scattering_potential[iSlab].AddImage(&projected_water_atoms);
//		scattering_potential[iSlab].QuickAndDirtyWriteSlice("atoms_water.mrc",1,true);


		delete [] tmpPrj;
//		exit(-1);

	}
	else
	{
		float oxygen_inelastic_to_elastic_ratio;

		// The inelastic/elastic ratio for water gives a Zeff of carbon
		if (DO_PHASE_PLATE)
		{
			oxygen_inelastic_to_elastic_ratio= sqrtf(( inelastic_scalar / sp.ReturnAtomicNumber(carbon)));

		}
		else
		{
			if (SOLVENT_TYPE == oxygen)
			{
				// The total elastic cross section of water is nearly equal to water but the inelastic is higher than expected (via 22.2/Z Reimer) using values from Wanner et al. 2006
				oxygen_inelastic_to_elastic_ratio = sqrtf(( inelastic_scalar / sp.ReturnAtomicNumber(plasmon)));
			}
			else
			{
				 oxygen_inelastic_to_elastic_ratio = sqrtf(( inelastic_scalar / sp.ReturnAtomicNumber(SOLVENT_TYPE)));

			}
		}


		scattering_potential[iSlab].AddImage(&projected_water_atoms);
		projected_water_atoms.MultiplyByConstant(oxygen_inelastic_to_elastic_ratio); // Just assuming oxygen for now
//		this->inelastic_mean[iSlab] *= oxygen_inelastic_to_elastic_ratio;
		inelastic_potential[iSlab].AddImage(&projected_water_atoms);
	}


	if (SAVE_WATER_AND_OTHER)
	{
		scattering_potential[iSlab].WriteSlices(&mrc_out,3,3);
		mrc_out.SetPixelSize(this->wanted_pixel_size);
		mrc_out.CloseFile();
	}


	if (ReturnThreadNumberOfCurrentThread()==0) timer.lap("w_after");


} // End of fill water func
void SimulateApp::project(Image *image_to_project, Image *image_to_project_into,  int iSlab)
{

	/* Image.AddSlices accumulates in float. Maybe just add an option there to add in double.
	 *
	 */
	// Project the slab into the two

	double pixel_accumulator;
	int prjX, prjY, prjZ;
	int edgeX = 0;
	int edgeY = 0;
	int column_length = image_to_project_into[iSlab].logical_x_dimension + image_to_project_into[iSlab].padding_jump_value;
	long pixel_counter;


//	// Have each thread work on a column so sequential addresses may be coalesced
	#pragma omp parallel for num_threads(this->number_of_threads)
	for (int iCol = 0 ; iCol < image_to_project->logical_y_dimension; iCol++)
	{
		int image_plane = 0;
		int skip_col = iCol*column_length;

		for (prjZ = 0; prjZ < image_to_project->logical_z_dimension; prjZ++)
		{
			for (int iPixel = 0; iPixel < image_to_project->logical_x_dimension; iPixel++)
			{
				image_to_project_into[iSlab].real_values[iPixel + skip_col ] += image_to_project->real_values[iPixel + skip_col + image_plane];
			}

			image_plane += image_to_project_into[iSlab].real_memory_allocated;

		}
	}
	//

//
//	long image_plane = 0;
//	for (prjZ = 0; prjZ < image_to_project->logical_z_dimension; prjZ++)
//	{
//
//		for (int iPixel = 0; iPixel < image_to_project_into[iSlab].real_memory_allocated; iPixel++)
//		{
//			image_to_project_into[iSlab].real_values[iPixel] += image_to_project->real_values[iPixel + image_plane];
//		}
//
//		image_plane += image_to_project_into[iSlab].real_memory_allocated;
//
//	}




}

void SimulateApp::taper_edges(Image *image_to_taper,  int iSlab, bool inelastic_img)
{
	// Taper edges to the mean TODO see if this can be removed
	// Update with the current mean. Why am I saving this? Is it just for SOLVENT ==1 ? Then probably can kill TODO
	int prjX, prjY, prjZ;
	int edgeX;
	int edgeY;
	long slab_address;
	float taper_val;

//	for (int iSlab = 0; iSlab < nSlabs; iSlab++)
//	{


		edgeX = 0;
		edgeY = 0;

		float avgRadius = std::min(image_to_taper[iSlab].logical_x_dimension/2 - TAPERWIDTH,image_to_taper[iSlab].logical_y_dimension/2 - TAPERWIDTH);
		image_to_taper[iSlab].CosineRectangularMask(image_to_taper[iSlab].logical_x_dimension/2 - TAPERWIDTH,
													image_to_taper[iSlab].logical_y_dimension/2 - TAPERWIDTH,
													0.0, TAPERWIDTH/2, false, true, image_to_taper[iSlab].ReturnAverageOfRealValuesAtRadius(avgRadius));
		if (inelastic_img)
		{
			this->inelastic_mean[iSlab] =  image_to_taper[iSlab].ReturnAverageOfRealValues(0.0);

		}
		else
		{
			this->image_mean[iSlab] =  image_to_taper[iSlab].ReturnAverageOfRealValues(0.0);

		}
		if (DO_PRINT) {wxPrintf("%d image mean for taper %f\n",iSlab, this->image_mean[iSlab]);}




}

void SimulateApp::apply_sqrt_DQE_or_NTF(Image *image_in, int iTilt_IDX, bool do_root_DQE)
{


	bool do_backward_fft = false;

	if (image_in[iTilt_IDX].is_in_real_space)
	{
		image_in[iTilt_IDX].ForwardFFT(true);
		do_backward_fft = true;
	}

	float x_coord_sq, y_coord_sq, z_coord_sq, spatial_frequency;
	float weight;
	long pixel_counter = 0;

	for (int k = 0; k <= image_in[iTilt_IDX].physical_upper_bound_complex_z; k++)
	{
		z_coord_sq = powf(image_in[iTilt_IDX].ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * image_in[iTilt_IDX].fourier_voxel_size_z, 2);


		for (int j = 0; j <= image_in[iTilt_IDX].physical_upper_bound_complex_y; j++)
		{
			//
			y_coord_sq = powf( image_in[iTilt_IDX].ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * image_in[iTilt_IDX].fourier_voxel_size_y , 2);

			for (int i = 0; i <= image_in[iTilt_IDX].physical_upper_bound_complex_x; i++)
			{
				weight = 0.0f;

				//
				x_coord_sq = powf(  i * image_in[iTilt_IDX].fourier_voxel_size_x , 2);

				// compute squared radius, in units of reciprocal pixels

				spatial_frequency = sqrtf(x_coord_sq + y_coord_sq + z_coord_sq);

				if (do_root_DQE)
				{
					// Sum of three gaussians
					for (int iGaussian = 0; iGaussian < 5; iGaussian++)
					{
						weight += ( DQE_PARAMETERS_A[CAMERA_MODEL][iGaussian] * expf(-1.0f*
																			  powf( (spatial_frequency-DQE_PARAMETERS_B[CAMERA_MODEL][iGaussian]) /
																					 DQE_PARAMETERS_C[CAMERA_MODEL][iGaussian],2)) );
					}
				}
	//			else
	//			{
	//				// NTF (NPS/ConversionFactor^2*Neletrons Ttotal) sum of 2 gaussians
	//				for (int iGaussian = 0; iGaussian < 2; iGaussian++)
	//				{
	//					weight += ( NTF_PARAMETERS_A[CAMERA_MODEL][iGaussian] * expf(-1.0f*
	//																		  powf( (spatial_frequency-NTF_PARAMETERS_B[CAMERA_MODEL][iGaussian]) /
	//																				 NTF_PARAMETERS_C[CAMERA_MODEL][iGaussian],2)) );
	//				}
	//			}
				image_in[iTilt_IDX].complex_values[pixel_counter] *= weight;

				pixel_counter++;

			}
		}

	}

	if (do_backward_fft)
	{
		image_in[iTilt_IDX].BackwardFFT();
	}



}







