#include "../../core/core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
#include <ctime>


//// Required to include CUDA vector types
//#include <cuda_runtime.h>
//#include <vector_types.h>
//#include <helper_cuda.h>

#define DO_BEAM_TILT_FULL false
#define NARGSEXPECTED 4 // number of arguments passed to the app.
#define MAX_NUMBER_PDBS 13 // This probably doesn't need to be limited - check use TODO.
typedef float WANTED_PRECISION;

// From Shang and Sigworth, average of polar and non-polar from table 1 (with correction to polar radius 3, 1.7-->3.0);
// terms 3-5 have the average atomic radius of C,N,O,H added to shift the curve to be relative to atomic center.
const float xtra_shift = -0.5f;
const float HYDRATION_VALS[8] = {0.1750f,   -0.1350f,    2.23f,    3.43f,    4.78f,    1.0000f,    1.7700f,    0.9550f};

const int N_WATER_TERMS = 40;

// These could probably all just be MIN_BFACTOR
const float SOLVENT_BFACTOR = 10.0f; // times dose 8pi^2*msd(Ang) (defined) and sqrtf(dose) ~ rmsd(Ang) , note no sqrtf in the def for bfactor. latter is from Henderson, thon rings amorphous ice
const float CARBON_BFACTOR = 10.0f;
const float MIN_BFACTOR = 10.0f;// Used if supplied is -1 (for random perturb) //16*0.965*0.965/2/PI; // 85 matches the Phase grating to JPR pretty close

// Parameters for calculating water. Because all waters are the same, except a sub-pixel origin offset. A SUB_PIXEL_NeL stack of projected potentials is pre-calculated with given offsets.
const int SOLVENT_TYPE = 3; // 2 N, 3 O 15, fit water
const float water_oxygen_ratio = 0.8775;
const int SUB_PIXEL_NEIGHBORHOOD = 2;
const int SUB_PIXEL_NeL = (SUB_PIXEL_NEIGHBORHOOD*2+1)*(SUB_PIXEL_NEIGHBORHOOD*2+1)*(SUB_PIXEL_NEIGHBORHOOD*2+1);


const std::complex<float>  i2pi2(0.0f,19.7392088021787f);


const int TAPERWIDTH = 29; // TODO this should be set to 12 with zeros padded out by size neighborhood and calculated by taper = 0.5+0.5.*cos((((1:pixelFallOff)).*pi)./(length((1:pixelFallOff+1))));
const float TAPER[29] = {0,0,0,0,0,
						 0.003943, 0.015708, 0.035112, 0.061847, 0.095492, 0.135516, 0.181288, 0.232087,
						 0.287110, 0.345492, 0.406309, 0.468605, 0.531395, 0.593691, 0.654508, 0.712890,
						 0.767913, 0.818712, 0.864484, 0.904508, 0.938153, 0.964888, 0.984292, 0.996057};

const int IMAGEPADDING =  256; // Padding applied for the Fourier operations in the propagation steps. Removed prior to writing out.
const int IMAGETRIMVAL = IMAGEPADDING + 2*TAPERWIDTH;

const float MAX_PIXEL_SIZE = 3.0f;


// Intermediate images that may be useful for diagnostics.
#define SAVE_WATER_AND_OTHER false
#define SAVE_PROJECTED_WATER false
#define SAVE_PHASE_GRATING false
#define SAVE_PHASE_GRATING_DOSE_FILTERED false
#define SAVE_PHASE_GRATING_PLUS_WATER false
#define SAVE_PROBABILITY_WAVE false
#define SAVE_TO_COMPARE_JPR false
#define JPR_SIZE 514
#define SAVE_WITH_DQE false
#define SAVE_WITH_NORMALIZED_DOSE false
#define SAVE_POISSON_PRE_NTF false
#define SAVE_POISSON_WITH_NTF false
//beam_tilt_xform



//const int n_tilt_angles = 101;
//const float SET_TILT_ANGLES[n_tilt_angles] = {-70.000, -68.600, -67.200, -65.800, -64.400, -63.000, -61.600, -60.200, -58.800, -57.400, -56.000, -54.600, -53.200, -51.800, -50.400, -49.000, -47.600, -46.200, -44.800, -43.400, -42.000, -40.600, -39.200, -37.800, -36.400, -35.000, -33.600, -32.200, -30.800, -29.400, -28.000, -26.600, -25.200, -23.800, -22.400, -21.000, -19.600, -18.200, -16.800, -15.400, -14.000, -12.600, -11.200, -9.800, -8.400, -7.000, -5.600, -4.200, -2.800, -1.400, 0.000, 1.400, 2.800, 4.200, 5.600, 7.000, 8.400, 9.800, 11.200, 12.600, 14.000, 15.400, 16.800, 18.200, 19.600, 21.000, 22.400, 23.800, 25.200, 26.600, 28.000, 29.400, 30.800, 32.200, 33.600, 35.000, 36.400, 37.800, 39.200, 40.600, 42.000, 43.400, 44.800, 46.200, 47.600, 49.000, 50.400, 51.800, 53.200, 54.600, 56.000, 57.400, 58.800, 60.200, 61.600, 63.000, 64.400, 65.800, 67.200, 68.600, 70.000};
//const float SET_TILT_ANGLES[n_tilt_angles] = {-15.400, -14.000, -12.600, -11.200, -9.800, -8.400, -7.000, -5.600, -4.200, -2.800, -1.400, -0.000, 1.400, 2.800, 4.200, 5.600, 7.000, 8.400, 9.800, 11.200, 12.600, 14.000, 15.400, 16.800, 18.200, 19.600, 21.000, 22.400, 23.800, 25.200, 26.600, 28.000, 29.400, 30.800, 32.200, 33.600, 35.000, 36.400, 37.800, 39.200, 40.600, 42.000, 43.400, 44.800, 46.200, 47.600, 49.000, 50.400, 51.800, 53.200, 54.600, 56.000, 57.400, 58.800, 60.200, 61.600, 63.000, 64.400, 65.800, 67.200, 68.600, 70.000, -16.800, -18.200, -19.600, -21.000, -22.400, -23.800, -25.200, -26.600, -28.000, -29.400, -30.800, -32.200, -33.600, -35.000, -36.400, -37.800, -39.200, -40.600, -42.000, -43.400, -44.800, -46.200, -47.600, -49.000, -50.400, -51.800, -53.200, -54.600, -56.000, -57.400, -58.800, -60.200, -61.600, -63.000, -64.400, -65.800, -67.200, -68.600, -70.000};

//const int n_tilt_angK2Count-300kv-50e-3eps.txtles = 50;
//const float   SET_TILT_ANGLES[n_tilt_angles] = {-15.400, -12.600, -9.800, -7.000, -4.200, -1.400, 1.400, 4.200, 7.000, 9.800, 12.600, 15.400, 18.200, 21.000, 23.800, 26.600, 29.400, 32.200, 35.000, 37.800, 40.600, 43.400, 46.200, 49.000, 51.800, 54.600, 57.400, 60.200, 63.000, 65.800, 68.600, -18.200, -21.000, -23.800, -26.600, -29.400, -32.200, -35.000, -37.800, -40.600, -43.400, -46.200, -49.000, -51.800, -54.600, -57.400, -60.200, -63.000, -65.800, -68.600};

//const int n_tilt_angles = 51;
//const float SET_TILT_ANGLES[n_tilt_angles] = {0.00, 2.800, -2.800, 5.600, -5.600, 8.400, -8.400, 11.200, -11.200, 14.000, -14.000, 16.800, -16.800, 19.600, -19.600, 22.400, -22.400, 25.200, -25.200, 28.000, -28.000, 30.800, -30.800, 33.600, -33.600, 36.400, -36.400, 39.200, -39.200, 42.000, -42.000, 44.800, -44.800, 47.600, -47.600, 50.400, -50.400, 53.200, -53.200, 56.000, -56.000, 58.800, -58.800, 61.600, -61.600, 64.400, -64.400, 67.200, -67.200, 70.000, -70.000};
// Some of the more common elements, should add to this later. These are from Peng et al. 1996.
const int n_tilt_angles = 41;
const float SET_TILT_ANGLES[n_tilt_angles] = {0, 3, -3, 6, -6, 9, -9, 12, -12, 15, -15, 18, -18, 21, -21, 24, -24, 27, -27, 30, -30, 33, -33, 36, -36, 39, -39, 42, -42, 45, -45, 48, -48, 51, -51, 54, -54, 57, -57, 60, -60};
//const int n_tilt_angles = 2;
//const float SET_TILT_ANGLES[2] = {0,25};

// The name is to an index matching here in the PDB class. If you change this, you MUST change that. This is probably a bad idea.
// H(0),C(1),N(2),O(3),F(4),Na(5),Mg(6),P(7),S(8),Cl(9),K(10),Ca(11),Mn(12),Fe(13),Zn(14),H20(15),0-(16)
const float  ATOMIC_NUMBER[17] = {1.0f, 6.0f, 7.0f, 8.0f, 9.0f, 11.0f, 12.0f, 15.0f, 16.0f, 17.0f, 19.0f, 20.0f, 25.0f, 26.0f, 30.0f, 10.0f, 8.0f};
const WANTED_PRECISION WN = 0.8045*0.79; // sum netOxy A / sum water (A) = 0.8045 and ratio of total elastic cross section water/oxygen 0.67-0.92 Using average 0.79 (there is no fixed estimate)
const WANTED_PRECISION   SCATTERING_PARAMETERS_A[17][5] = {
	{ 0.0349,  0.1201, 0.1970, 0.0573, 0.1195},
	{ 0.0893,  0.2563, 0.7570, 1.0487, 0.3575},
	{ 0.1022,  0.3219, 0.7982, 0.8197, 0.1715},
	{ 0.0974,  0.2921, 0.6910, 0.6990, 0.2039},
	{ 0.1083,  0.3175, 0.6487, 0.5846, 0.1421},
	{ 0.2142,  0.6853, 0.7692, 1.6589, 1.4482},
	{ 0.2314,  0.6866, 0.9677, 2.1882, 1.1339},
	{ 0.2548,  0.6106, 1.4541, 2.3204, 0.8477},
	{ 0.2497,  0.5628, 1.3899, 2.1865, 0.7715},
	{ 0.2443,  0.5397, 1.3919, 2.0197, 0.6621},
	{ 0.4115, -1.4031, 2.2784, 2.6742, 2.2162},
	{ 0.4054,  1.3880, 2.1602, 3.7532, 2.2063},
	{ 0.3796,  1.2094, 1.7815, 2.5420, 1.5937},
	{ 0.3946,  1.2725, 1.7031, 2.3140, 1.4795},
	{ 0.4288,  1.2646, 1.4472, 1.8294, 1.0934},
	{WN*0.07967, WN*0.1053, WN* 0.2933, WN*0.6831, WN*1.304},
	{ 0.2050,  0.6280, 1.1700, 1.0300, 0.290 }, // Peng 1998
};

// 12.5664 ~ (4*pi)
// -39.47841760685077 ~ -4*pi^2
const WANTED_PRECISION SCATTERING_PARAMETERS_B[17][5] = {
		{0.5347, 3.5867, 12.347, 18.9525, 38.6269},
		{0.2465, 1.7100, 6.4094, 18.6113, 50.2523},
		{0.2451, 1.7481, 6.1925, 17.3894, 48.1431},
		{0.2067, 1.3815, 4.6943, 12.7105, 32.4726},
		{0.2057, 1.3439, 4.2788, 11.3932, 28.7881},
		{0.3334, 2.3446, 10.083, 48.3037, 138.270},
		{0.3278, 2.2720, 10.924, 39.2898, 101.9748},
		{0.2908, 1.8740, 8.5176, 24.3434, 63.2996},
		{0.2681, 1.6711, 7.0267, 19.5377, 50.3888},
		{0.2468, 1.5242, 6.1537, 16.6687, 42.3086},
		{0.3703, 3.3874, 13.1029, 68.9592, 194.4329},
		{0.3499, 3.0991, 11.9608, 53.9353, 142.3892},
		{0.2699, 2.0455, 7.4726, 31.0604, 91.5622},
		{0.2717, 2.0443, 7.6007, 29.9714, 86.2265},
		{0.2593, 1.7998, 6.7500, 25.5860, 73.5284},
		{WN*4.718 , WN* 16.75,WN* 0.4524,  WN* 13.43, WN* 4.4480},
		{ 0.397, 2.6400, 8.8000, 27.1, 91.8}, //Peng 98

};


// Assuming a perfect counting - so no read noise, and no coincednce loss. Then we have a flat NPS and the DQE is just DQE(0)*MTF^2
// The parameters below are for a 5 gaussian fit to 300 KeV , 2.5 EPS from Ruskin et al. with DQE(0) = 0.791
const WANTED_PRECISION DQE_PARAMETERS_A[1][5] = {

		{-0.01516,-0.5662,-0.09731,-0.01551,21.47}


};

const WANTED_PRECISION DQE_PARAMETERS_B[1][5] = {

		{0.02671,-0.02504, 0.162,0.2831,-2.28}

};

const WANTED_PRECISION DQE_PARAMETERS_C[1][5] = {

		{0.01774,0.1441,0.1082,0.07916,1.372}

};

// Assume a more practical dose rate for the envelope function FIXME make this variable
const float DOSE_RATE_FOR_ENVELOPE = 8.0f; // EPS, converted to e/A**@/sec on use


//const WANTED_PRECISION NTF_PARAMETERS_A[3][2] = {
//		{3.0,1.09},
//		{1.206,3.0},
//		{0.0101,2.824},


struct corners {

	float x1; float x2;
	float y1; float y2;
	float z1; float z2;
};



//extern "C" bool runTest(const int argc, const char **argv,
//                        char *data, int2 *data_int2, unsigned int len);

class ScatteringPotentialApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();
	wxString 	pdb_file_names[MAX_NUMBER_PDBS];
	std::string output_filename;
	int 	 	number_of_pdbs = 1;
	float	    defocus = 0.0f;
	//int 	 	particle_copy_number[MAX_NUMBER_PDBS];
	float 	 	particle_origin[MAX_NUMBER_PDBS][3];
	float 	 	particle_eulers[MAX_NUMBER_PDBS][3];
	long 	 	number_of_non_water_atoms = 0; // could this overffow??
	float 	 	*image_mean;
	float		*inelastic_mean;
	float		current_total_exposure = 0;
	float 		pre_exposure = 0;
	float    	wanted_pixel_size = 0;
	float		wanted_pixel_size_sq = 0;
	float 		unscaled_pixel_size = 0; // When there is a mag change
	int 	 	size_neighborhood = 0;
	int 		size_neighborhood_water = 0;
	bool 	 	tilt_series = false;
	long 		doParticleStack = 0;
	int 	 	number_of_threads = 1;
    float 	 	do3d = 0.0f;
    int			bin3d = 1;
    bool 		doExpertOptions = false;
    bool 		get_devel_options = false;
    float		stdErr = 0.0f;
    float		extra_phase_shift = 0.0f;
    float 		kV = 300;
    float 		wavelength = 1.968697e-2; // Angstrom, default for 300kV
    float		lead_term = 0.0f;
    float 		phase_plate_thickness = 276.0; //TODO CITE Angstrom, default for pi/2 phase shift, from 2.0 g/cm^3 amorphous carbon, 300 kV and 10.7V mean inner potential as determined by electron holography (carbon atoms are ~ 8)
    float 		spherical_aberration = 2.7f;
    float 		amplitude_contrast = 0.07f;
    float       astigmatism_scaling = 0.0f;
	float 		dose_per_frame = 1;
	float 		dose_rate = 3.0; // e/physical pixel/s this should be set by the user. Changes the illumination aperature and DQE curve
	float 		number_of_frames = 1;
	double		total_waters_incorporated = 0;
	float 		average_at_cutoff[N_WATER_TERMS]; // This assumes a fixed 1/2 angstrom sampling of the hydration shell curve
	float		water_weight[N_WATER_TERMS];
	bool 		need_to_allocate_projected_water = true;
	float 		chromatic_aberration = 3.49;
	float       set_real_part_wave_function_in = 1.0f;
	int 		minimum_paddeding_x_and_y = 64;
	float  		propagation_distance = 20.0f;
	float       minimum_thickness_z = propagation_distance = 20.0f;


	bool ONLY_SAVE_SUMS = true;
	// To add error to the global alignment
	float tilt_axis = 0; // degrees from Y-axis FIXME thickness calc, water padding, a few others are only valid for 0* tilt axis.
	float in_plane_sigma = 2; // spread in-plane angles based on neighbors
	float tilt_angle_sigma = 0.1; //;
	float magnification_sigma = 0.0001;//;
	float beam_tilt_x = 0.0f;//0.6f;
	float beam_tilt_y = 0.0f;//-0.2f;
	float beam_tilt = 0.0f;
	float beam_tilt_azimuth = 0.0f;
	RotationMatrix beam_tilt_xform;
	bool  DO_BEAM_TILT = false;
	float particle_shift_x = 0.0f;
	float particle_shift_y =0.0f;

	Image number_of_pixels_averaged;
	Image number_of_non_waters_averaged;


//    double   	average_at_cutoff[4] = {0,0,0,0};
	int padded_x_dim;
	int padded_y_dim;

    float bFactor_scaling;
    float min_bFactor;

    FrealignParameterFile  parameter_file;
    std::string			   parameter_file_name;
    bool use_existing_params = false;
    long number_preexisting_particles;
    wxString preexisting_particle_file_name;

    float parameter_vect[17] = {0};



    float water_scaling;

	Image *projected_water; // waters calculated over the specified subpixel shifts and projected.


	Image *correlation_check;


	void probability_density_2d(PDB *pdb_ensemble, int time_step);
	// Note the water does not take the dose as an argument.
	void  calc_scattering_potential(const PDB * current_specimen,Image *scattering_slab, Image *inelastic_slab,  RotationMatrix rotate_waters,
			                        float rotated_oZ, int *slabIDX_start, int *slabIDX_end, int iSlab);

	void  calc_water_potential(Image *projected_water);
	void  fill_water_potential(const PDB * current_specimen,Image *scattering_slab, Image *scattering_potential,
							   Image *inelastic_potential, Water *water_box,RotationMatrix rotate_waters,
														   float rotated_oZ, int *slabIDX_start, int *slabIDX_end, int iSlab);


	void  project(Image *image_to_project, Image *image_to_project_into,  int iSlab);
	void  taper_edges(Image *image_to_taper,  int iSlab, bool inelastic_img);
	void  apply_sqrt_DQE_or_NTF(Image *image_in, int iTilt_IDX, bool do_root_DQE);
	void  normalize_set_dose_expectation(Image *sum_image, int iTilt_IDX, float current_thickness);

	void calc_average_intensity_at_solvent_cutoff(float average_bfactor);
	float return_bfactor_given_dose(float relative_bfactor);




	// Profiling
	wxDateTime  timer_start;
	wxDateTime	overall_start;
	wxTimeSpan	span_seed;
	wxTimeSpan	span_atoms;
	wxTimeSpan  span_waters;
	wxTimeSpan  span_shake;
	wxTimeSpan	span_propagate;
	wxDateTime 	overall_finish;


	//////////////////////////////////////////
	////////////
	// For development
	//
	//const float  MEAN_FREE_PATH = 4000;// Angstrom, newer paper by bridgett (2018) and a couple older TODO add ref. Use to reduce total probability (https://doi.org/10.1016/j.jsb.2018.06.007)


	// Note that the first two errors here are just used in matching amorphous carbon for validation. The third is used in simulations.
	// The surface phase error (Wanner & Tesche 2005) quantified by holography accounts for a bias due to surface effects not included in the simple model here
	float  SURFACE_PHASE_ERROR = 0.3; //0.497;
	// The bond phase error is used to account for the remaining phase shift that is missing, due to all remaining scattering. The assumption is that amorphous water has >= the scattering due to delocalized electrons
	float  BOND_PHASE_ERROR = 0.0;
	// To account for the bond phase error in practice, a small scaling factor is applied to the atomic potentials
	float  BOND_SCALING_FACTOR = 1.02; //1.0475;
	float MEAN_INNER_POTENTIAL = 9.09; // for 1.75 g/cm^3 amorphous carbon as in Wanner & Tesche
	bool  DO_PHASE_PLATE = false;

	// CONTROL FOR DEBUGGING AND VALIDATION
	bool add_mean_water_potential  = false; // For comparing to published results - only applies to a 3d potential

	bool DO_SINC_BLUR = false; // TODO add me back in. This is to blur the image due to intra-frame motion. Apply to projected specimen not waters
	bool DO_PRINT = false;
	//
	bool DO_SOLVENT = true; // False to skip adding any solvent
	bool CALC_WATER_NO_HOLE = false; // Makes the solvent cutoff so large that water is added on top of the protein. This is to mimick Rullgard/Vulovic
	bool CALC_HOLES_ONLY = false; // This should calculate the atoms so the water is properly excluded, but not include these in the propagation. TODO checkme
	bool DEBUG_POISSON = false; // Skip the poisson draw - must be true (this is gets over-ridden) if DO_PHASE_PLATE is true


	bool DO_COHERENCE_ENVELOPE = true;

	bool WHITEN_IMG = false; // if SAVE_REF then it will also be whitened when this option is enabled TODO checkme
	bool SAVE_REF   = false;
	bool CORRECT_CTF= false; // only affects the image if Whiten_img is also true
	bool EXPOSURE_FILTER_REF = false;
	bool DO_EXPOSURE_FILTER_FINAL_IMG = false;
	int DO_EXPOSURE_FILTER = 3;  ///// In 2d or 3d - or 0 to turn of. 2d does not appear to produce the expected results.
	bool DO_EXPOSURE_COMPLEMENT_PHASE_RANDOMIZE = false; // maintain the Energy of the protein (since no mass is actually lost) by randomizing the phases with weights that complement the exposure filter
	bool DO_COMPEX_AMPLITUDE_TERM = true;
	bool DO_APPLY_DQE = true;
	bool DO_APPLY_NTF = false;
	int CAMERA_MODEL=0;
	///////////
	/////////////////////////////////////////

	private:

};



IMPLEMENT_APP(ScatteringPotentialApp)

// override the DoInteractiveUserInput

void ScatteringPotentialApp::DoInteractiveUserInput()
{



	 bool add_more_pdbs = true;
	 bool supply_origin = false;
	 int iPDB = 0;
	 int iOrigin;
	 int iParticleCopy;
	 this->number_of_pdbs = 1;

	 UserInput *my_input = new UserInput("ScatteringPotential", 0.1);
	 this->output_filename = my_input->GetFilenameFromUser("Output filename","just the base, no extension, will be mrc","test_tilt.mrc",false);
	 this->do3d = my_input->GetFloatFromUser("Make a 3d scattering potential?","just potential if > 0, input is the wanted cubic size","0.0",0.0f,2048.0f);


 	 this->number_of_threads = my_input->GetIntFromUser("Number of threads", "Max is number of tilts", "1", 1);

	 while (number_of_pdbs < MAX_NUMBER_PDBS && add_more_pdbs)
	 {
		 pdb_file_names[number_of_pdbs-1] = my_input->GetFilenameFromUser("PDB file name", "an input PDB", "my_atom_coords.pdb", true );
		 // This is now coming directly from the PDB
		 //particle_copy_number[number_of_pdbs-1] = my_input->GetIntFromUser("Copy number of this particle", "To be inserted into the ensemble", "1", 1);




		 add_more_pdbs = my_input->GetYesNoFromUser("Add another type of particle?", "Add another pdb to create additional features in the ensemble", "no");



		 if (add_more_pdbs) {number_of_pdbs++;}
	 }



	 this->wanted_pixel_size 		= my_input->GetFloatFromUser("Output pixel size (Angstroms)","Output size for the final projections","1.0",0.01,MAX_PIXEL_SIZE);
	 this->bFactor_scaling		 = my_input->GetFloatFromUser("Linear scaling of per atom bFactor","0 off, 1 use as is","0",0,10000);
	 this->min_bFactor    		 = my_input->GetFloatFromUser("Per atom (xtal) bFactor added to all atoms","0 off, 1 use as is","-1",-1,10000);

	 if (this->do3d)
	 {
		 // Check to make sure the sampling is sufficient, if not, oversample and bin at the end.
		 if (this->wanted_pixel_size > 0.8 && this->wanted_pixel_size < 1.2)
		 {
			 wxPrintf("\nOversampling your 3d by a factor of 2 for calculation.\n");
			 this->wanted_pixel_size /= 2.0f;
			 this->bin3d = 2;
		 }
		 else if (this->wanted_pixel_size >= 1.2 && this->wanted_pixel_size < 2.4)
		 {
			 wxPrintf("\nOversampling your 3d by a factor of 4 for calculation.\n");

			 this->wanted_pixel_size /= 4.0f;
			 this->bin3d = 4;
		 }
		 else
		 {
			 //do nothing
		 }

		 this->do3d *= this->bin3d;
	 }
	 else
	 {
		 this->tilt_series = my_input->GetYesNoFromUser("Create a tilt-series as well?","Should make 0 degree with full dose, then tilt","no");
		 if ( this->tilt_series == true )
		 {
			 // not doing anything for now, fixed range and increment.test_multi.mrc
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
				preexisting_particle_file_name = my_input->GetFilenameFromUser("Freealign parameter file name", "an input parameter file to match reconstruction", "myparams.par", true );
				if (! DoesFileExist(preexisting_particle_file_name))
				{
					SendError(wxString::Format("Error: Input parameter file %s not found\n", preexisting_particle_file_name));
					exit(-1);
				}

			    number_preexisting_particles = my_input->GetIntFromUser("Number of particles in the param file", "I don't know of a better way to get this info jsut yet", "1", 1);
//				this->parameter_file.Open(parameter_file_name, access_type,17);
//				wxPrintf("%d access_type\n", this->parameter_file.access_type);
//				this->parameter_file.ReadFile(true, number_of_particles);
//				wxPrintf("\nRecreating %ld particles from the supplied parameter file\n", number_of_particles);

			}
		 this->water_scaling 		 = my_input->GetFloatFromUser("Linear scaling water intensity","0 off, 1 use as is","1",0,10);
		 this->astigmatism_scaling	 = my_input->GetFloatFromUser("fraction of the defocus to add as astigmatism","0 off, 1 use as is","0.0",0,0.5);
		 this->kV 					 = my_input->GetFloatFromUser("Accelrating volatage (kV)","Default is 300","300.0",80.0,1000.0f); // Calculations are not valid outside this range
		 this->amplitude_contrast 	 = my_input->GetFloatFromUser("Amplitude contrast ratio (as complex potential. 0 to turn off","","0.07",0.0,1.0);
		 this->spherical_aberration	 = my_input->GetFloatFromUser("Spherical aberration constant in millimeters","","2.7",0.0,5.0);
		 this->stdErr = my_input->GetFloatFromUser("Std deviation of error to use in shifts, astigmatism, rotations etc.","","0.0",0.0,100.0);
		 this->pre_exposure = my_input->GetFloatFromUser("Pre-exposure in electron/A^2","use for testing exposure filter","0",0.0);

		 // Since kV is not default, need to calculate these parameters
//		 const float WAVELENGTH = pow(1.968697e-2,1); // Angstrom
		 this->wavelength 		  = 1226.39 / sqrtf(this->kV*1000 + 0.97845e-6*powf(this->kV*1000,2)) * 1e-2; // Angstrom
		 this->phase_plate_thickness = (PI/2.0f + SURFACE_PHASE_ERROR + BOND_PHASE_ERROR) / ( MEAN_INNER_POTENTIAL/(kV*1000) * (511+kV)/(2*511+kV) * (2*PI / (wavelength*1e-10)) )*1e10;


	 }
	 else
	 {
		 this->wavelength 		  = 1226.39 / sqrtf(this->kV*1000 + 0.97845e-6*powf(this->kV*1000,2)) * 1e-2; // Angstrom
		 this->water_scaling=1;
		 this->astigmatism_scaling=0.0;
		 this->amplitude_contrast = 0.07;
		 this->spherical_aberration = 2.7;
		 this->stdErr = 0.0;
	 }


	if (doParticleStack > 0 || this->tilt_series)
	{
		parameter_file_name = output_filename + ".par";
		wxString parameter_header = "C           PSI   THETA     PHI       SHX       SHY     MAG  INCLUDE   DF1      DF2  ANGAST  PSHIFT     OCC      LogP      SIGMA   SCORE  CHANGE";
		this->parameter_file.Open(parameter_file_name,1,17);
		this->parameter_file.WriteCommentLine(parameter_header);

	}

	this->get_devel_options			= my_input->GetYesNoFromUser("Set development options?","","no");

	if (this->get_devel_options)
	{
		//////////////////////////////////////////
		////////////
		// For development
		//

		// Note that the first two errors here are just used in matching amorphous carbon for validation. The third is used in simulations.
		// The surface phase error (Wanner & Tesche 2005) quantified by holography accounts for a bias due to surface effects not included in the simple model here
		this->SURFACE_PHASE_ERROR = my_input->GetFloatFromUser("Surface phase error (radian)","From Wanner & Tesche","0.497",0,3.121593);
		// The bond phase error is used to account for the remaining phase shift that is missing, due to all remaining scattering. The assumption is that amorphous water has >= the scattering due to delocalized electrons
		this->BOND_PHASE_ERROR = my_input->GetFloatFromUser("Bond phase error (radian)","Calibrated from experiment, may need to be updated","0.098",0,3.121593);
		// To account for the bond phase error in practice, a small scaling factor is applied to the atomic potentials
		this->BOND_SCALING_FACTOR = my_input->GetFloatFromUser("Compensate for bond phase error by scaling the interaction potential","Calibrated from experiment, may need to be updated","1.0475",0,10);
		this->MEAN_INNER_POTENTIAL = my_input->GetFloatFromUser("Amorphous carbon MIP","From Wanner & Tesche for 1.75 g/cm^3 carbon","9.09",0,100); // for 1.75 g/cm^3 amorphous carbon as in Wanner & Tesche
		this->DO_PHASE_PLATE = my_input->GetYesNoFromUser("So a phase plate simulation?","","no");

		// CONTROL FOR DEBUGGING AND VALIDATION
		this->add_mean_water_potential  = my_input->GetYesNoFromUser("Add a constant potential for the mean water?","","no"); // For comparing to published results - only applies to a 3d potential


		this->DO_SINC_BLUR = my_input->GetYesNoFromUser("Intra-frame sinc blur on protein?","Currently this does nothing","no"); // TODO add me back in. This is to blur the image due to intra-frame motion. Apply to projected specimen not waters
		this->DO_PRINT = my_input->GetYesNoFromUser("Print out a bunch of extra diagnostic information?","","no");
		//
		this->DO_SOLVENT = my_input->GetYesNoFromUser("Add potential, constant or atom-wise for solvent?","","yes"); // False to skip adding any solvent
		this->CALC_WATER_NO_HOLE = my_input->GetYesNoFromUser("Ignore protein envelope and add solvent everywhere?","","no"); // Makes the solvent cutoff so large that water is added on top of the protein. This is to mimick Rullgard/Vulovic
		this->CALC_HOLES_ONLY = my_input->GetYesNoFromUser("Use protein envelope, but don't add it in (holes only)?","","no");// This should calculate the atoms so the water is properly excluded, but not include these in the propagation. TODO checkme
		this->DEBUG_POISSON = my_input->GetYesNoFromUser("Save the detector wave function directly?","Skip Poisson draw","no"); // Skip the poisson draw - must be true (this is gets over-ridden) if DO_PHASE_PLATE is true


		this->DO_COHERENCE_ENVELOPE = my_input->GetYesNoFromUser("Apply spatial coherence envelope?","This currently doesn't do anything","yes");// These do nothing~! FIXME

		this->WHITEN_IMG = my_input->GetYesNoFromUser("Whiten the image?","","no"); // if SAVE_REF then it will also be whitened when this option is enabled TODO checkme
		this->SAVE_REF   = my_input->GetYesNoFromUser("Save a perfect reference","","no");
		this->CORRECT_CTF= my_input->GetYesNoFromUser("Multiply by average CTF?","","no");// only affects the image if Whiten_img is also true
		this->EXPOSURE_FILTER_REF = my_input->GetYesNoFromUser("Exposure filter the perfect reference?","","no");
		this->DO_EXPOSURE_FILTER_FINAL_IMG = my_input->GetYesNoFromUser("Exposure filter the final movie/image?","","no");
		this->DO_EXPOSURE_FILTER = my_input->GetIntFromUser("Do exposure filter", "0 off, 2 2d (broken), 3 3d", "3", 3);  ///// In 2d or 3d - or 0 to turn of. 2d does not appear to produce the expected results.
		this->DO_EXPOSURE_COMPLEMENT_PHASE_RANDOMIZE =my_input->GetYesNoFromUser("Maintain power with random phases in exposure filter?","","no");// maintain the Energy of the protein (since no mass is actually lost) by randomizing the phases with weights that complement the exposure filter
		this->DO_COMPEX_AMPLITUDE_TERM = my_input->GetYesNoFromUser("Use the complex amplitude term?","(optical potential)","yes");
		this->DO_APPLY_DQE = my_input->GetYesNoFromUser("Apply the DQE?","depends on camera model","yes");

		this->DO_APPLY_NTF = my_input->GetYesNoFromUser("Apply NTF?","depends on camera model","yes");
		if (DO_APPLY_NTF)
		{
			wxPrintf("Currently a perfect counting detector is modelled. The noise spectrum is flat.\n\n");
			exit(-1);
		}



		this->minimum_paddeding_x_and_y = my_input->GetIntFromUser("minimum padding of images with solvent", "", "64", 32,4096);
		this->minimum_thickness_z = my_input->GetIntFromUser("minimum thickness in Z", "", "20",2,10000);
		this->propagation_distance = my_input->GetFloatFromUser("Propagation distance in angstrom","Also used as minimum thickness","20",-1e6,1e6);

		// propagation_distance can be negative to set nSlabs = 1;
		if (fabsf(propagation_distance) > minimum_thickness_z)
		{
			minimum_thickness_z = fabsf(propagation_distance);
			wxPrintf("min thickness was less than propagation distance, so setting it there\n");
		}

		this->ONLY_SAVE_SUMS = my_input->GetYesNoFromUser("Only save sums","don't save movies for particle stacks","yes"); // TODO check me
		// To add error to the global alignment
		this->tilt_axis = my_input->GetFloatFromUser("Rotation of tilt-axis from Y (degrees)","","0.0",0,180); // TODO does this apply everywhere it should
		this->in_plane_sigma = my_input->GetFloatFromUser("Standard deviation on angles in plane (degrees)","","2",0,100); // spread in-plane angles based on neighbors
		this->tilt_angle_sigma = my_input->GetFloatFromUser("Standard deviation on tilt-angles (degrees)","","0.1",0,10);; //;
		this->magnification_sigma = my_input->GetFloatFromUser("Standard deviation on magnification (fraction)","","0.0001",0,1);//;
		this->beam_tilt_x = my_input->GetFloatFromUser("Beam-tilt in X (milli radian)","","0.0",-300,300);//0.6f;
		this->beam_tilt_y = my_input->GetFloatFromUser("Beam-tilt in Y (milli radian)","","0.0",-300,300);//-0.2f;
		this->particle_shift_x =  my_input->GetFloatFromUser("Beam-tilt particle shift in X (Angstrom)","","0.0",-100,100);
		this->particle_shift_y =  my_input->GetFloatFromUser("Beam-tilt particle shift in Y (Angstrom)","","0.0",-100,100);;


		// Set the expected dose by setting the amplitude of the incoming plane wave
		set_real_part_wave_function_in = sqrtf(this->dose_per_frame);
		wxPrintf("Setting the real part of the object wavefunction to %f to start\n",set_real_part_wave_function_in);

		///////////
		/////////////////////////////////////////
	}

	 if (DO_PHASE_PLATE)
	 {
		 this->phase_plate_thickness = (PI/2.0f + SURFACE_PHASE_ERROR + BOND_PHASE_ERROR) / ( MEAN_INNER_POTENTIAL/(kV*1000) * (511+kV)/(2*511+kV) * (2*PI / (wavelength*1e-10)) )*1e10;

		 wxPrintf("With a mean inner potential of %2.2fV a thickness of %2.2f ang is needed for a pi/2 phase shift \n",MEAN_INNER_POTENTIAL,this->phase_plate_thickness);
		 wxPrintf("Phase shift params %f %f %f\n",BOND_PHASE_ERROR,BOND_SCALING_FACTOR,SURFACE_PHASE_ERROR);

	 }

	 //	 this->lead_term = BOND_SCALING_FACTOR * this->wavelength / this->wanted_pixel_size_sq / 8.0f;
	 	 // the 1/8 just comes from the integration of the gaussian which is too large by a factor of 2 in each dimension
	 	 this->lead_term = BOND_SCALING_FACTOR * this->wavelength  / 8.0f;

	delete my_input;



/*	my_current_job.Reset(NARGSEXPECTED);
	my_current_job.ManualSetArguments("ffff",	wanted_volume_size_X,
												wanted_volume_size_Y,
												wanted_volume_size_Z,
												wanted_output_pixel_size);
*/


}

// overide the do calculation method which will be what is actually run..

bool ScatteringPotentialApp::DoCalculation()
{

	wxPrintf("\nRecreating %d particles from the supplied parameter file\n", this->parameter_file.number_of_lines);

//


	// Profiling
	wxDateTime	overall_start = wxDateTime::Now();

	// get the arguments for this job..

	// backwards compatible with tigress where everything is double (ints would make more sense here.)
	long access_type_read = 0;
	long records_per_line = 1;
	int iLine;

	// set other vars
	long iPDB;
	long iParticle;

	// TODO is this the best place to put this?
	this->current_total_exposure = pre_exposure;


	if (CORRECT_CTF && use_existing_params)
	{
		wxPrintf("I did not set up ctf correction and the use of existing parameters. FIXME\n");
		throw;
	}


	PDB  *pdb_ensemble = new PDB[number_of_pdbs] ;

	// For Tim and Peter
	if (number_of_pdbs == 1) {wxPrintf("\nThere is %d pdb\n",number_of_pdbs);}
	else { wxPrintf("\nThere are %d pdbs\n",number_of_pdbs);}

	// Initialize each of the PDB objects, this reads in and centers each PDB, but does not make any copies (instances) of the trajectories.

	for (int iPDB = 0; iPDB < number_of_pdbs ; iPDB++)
	{

		pdb_ensemble[iPDB] = PDB(pdb_file_names[iPDB],access_type_read,records_per_line, this->minimum_paddeding_x_and_y, this->minimum_thickness_z);
		if (this->do3d > 0 && iPDB > 0)
		{
			pdb_ensemble[0] = PDB(pdb_file_names[iPDB],access_type_read,records_per_line, this->minimum_paddeding_x_and_y, this->minimum_thickness_z, pdb_ensemble[0].center_of_mass);
			this->number_of_pdbs = 1;
		}


	}







    // Get a count of the total non water atoms
	for (int iPDB = 0; iPDB < number_of_pdbs; iPDB++)
	{
		//this->number_of_non_water_atoms += (pdb_ensemble[iPDB].number_of_atoms * this->particle_copy_number[iPDB]);
		this->number_of_non_water_atoms += (pdb_ensemble[iPDB].number_of_atoms * pdb_ensemble[iPDB].number_of_particles_initialized);
//		wxPrintf("%ld %d\n",pdb_ensemble[iPDB].number_of_atoms , pdb_ensemble[iPDB].number_of_particles_initialized);
		// These sizes will need to be determined by the min and max dimensions of the base shifted ensemble and removed from user input TODO

	}

	wxPrintf("\nThere are %ld non-water atoms in the specimen.\n",this->number_of_non_water_atoms);
	wxPrintf("\nCurrent number of PDBs %d\n",number_of_pdbs);


	// Set-up the ensemble
//	this->set_initial_trajectories(pdb_ensemble);



	int time_step = 0 ;


    	this->probability_density_2d(pdb_ensemble, time_step);



	wxPrintf("\nFinished pre seg fault");

	overall_finish = wxDateTime::Now();


	wxPrintf("Timings: seed_waters: %s\n",(this->span_seed).Format());
	wxPrintf("Timings: shake_waters: %s\n",(this->span_shake).Format());
	wxPrintf("Timings: calc_atoms: %s\n",(this->span_atoms).Format());
	wxPrintf("Timings: calc_waters: %s\n",(this->span_waters).Format());
	wxPrintf("Timings: propagate_wave_function: %s\n",(this->span_propagate).Format());
	wxPrintf("Timings: unaccounted: %s\n",((overall_finish-overall_start)-(this->span_atoms+this->span_propagate+this->span_seed+this->span_shake+this->span_waters)).Format());

	// It gives a segfault at the end either way.
   // pdb_ensemble[0].Close();

	//overall_finish = wxDateTime::Now();



	return true;
}


/*
I've moved the wanted originz and euler angles as REMARK 351 in the PDB so that environments may be "easily" created in chimera.
It makes more sense then to intialize the trajectories in the call to PDB::init
Leave this in until convinced it works ok.
*/




void ScatteringPotentialApp::probability_density_2d(PDB *pdb_ensemble, int time_step)
{

	const bool do_exp_timings = false;
	bool SCALE_DEFOCUS_TO_MATCH_300 = true;
	float scale_defocus = 1.0f;
// scale propagator
	if (do_exp_timings)
	{
		///////////////////////////////////////////////////////////////////////////
		/*
		 * A simple block to use for testing things without having a sep. "experimental" program
		 */
		MRCFile input_file;
		MRCFile output_file;
		Image test_image;

//		input_file.OpenFile("testmask.mrc", false);
//		test_image.ReadSlices(&input_file, 1, input_file.ReturnNumberOfSlices());
//		input_file.CloseFile();

		const int n_sizes = 1;
		const int test_size[n_sizes] = {1024};
		long long n_iters = 1e7;
		int n_inner_loops = 1;

		clock_t exp_timer;
		double *exp_span;
		exp_span = new double[n_sizes];
		float iSin;
		float iCos;
		float vectSin[2] = {0.0f,0.0f};
	/////////////////////////////////////
	///	GPU GPU GPU
//
//	    // input data
//	    int len = 16;
//	    // the data has some zero padding at the end so that the size is a multiple of
//	    // four, this simplifies the processing as each thread can process four
//	    // elements (which is necessary to avoid bank conflicts) but no branching is
//	    // necessary to avoid out of bounds reads
//	    char str[] = { 82, 111, 118, 118, 121, 42, 97, 121, 124, 118, 110, 56,
//	                   10, 10, 10, 10
//	                 };
//
//	    // Use int2 showing that CUDA vector types can be used in cpp code
//	    int2 i2[16];
//
//	    for (int i = 0; i < len; i++)
//	    {
//	        i2[i].x = str[i];
//	        i2[i].y = 10;
//	    }
//
//	    bool bTestResult;
//
//	    // run the device part of the program
//	    const int nGPUs = 1;
//	    const char **useGPU = 0;
//	    bTestResult = runTest(nGPUs, useGPU, str, i2, len);
//
//	    std::cout << str << std::endl;
//
//	    char str_device[16];
//
//	    for (int i = 0; i < len; i++)
//	    {
//	        str_device[i] = (char)(i2[i].x);
//	    }
//
//	    std::cout << str_device << std::endl;
//
//	    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);

    /// GPU GPU GPU
    //////////////////////////////////////



//		Image test_image;
//		std::complex<float> check_val; // Each loop check a random value to ensure the compiler isn't being sneaky.
//		CTF my_ctf(300,2.7,0.07,8000,7000,29,1.35,0.0);
//		std::complex<float> *aC;;
//		float a = 0.0f;
//		for (long long iImg = 0; iImg < n_sizes; iImg ++)
//		{
//			exp_span[iImg] = 0.0f;
//			test_image.Allocate(test_size[iImg],test_size[iImg],false);
//			for (int iIter = 0; iIter < n_iters; iIter++)
//			{
//				for (int iLoop = 0 ; iLoop < n_inner_loops; iLoop++)
//				{
////					test_image.SetToConstant(1.0f);
//					float thisNumber = (float)iIter;
//					vectSin[0] = thisNumber; vectSin[1] = thisNumber + 90.0f;
//					exp_timer = clock();
////					iSin = sinf(thisNumber);
////					iCos = cosf(thisNumber);
//			aC = ReturnComplexExp((float)iIter, &aC);
////					test_image.AutoMask(1.023f,180.0,true);
////					test_image.ApplyCTF(my_ctf,false);
//					exp_span[iImg] += (clock() - exp_timer);
//					// Record the time before the check. Assuming that the image is at least 101 elements
////					check_val = test_image.complex_values[myroundint(50*(1+global_random_number_generator.GetUniformRandom()))];
//				}
//			}
//		}
//
////		aC = abs(aC);
//		iSin += 1;
//		iCos += 1;
		wxPrintf("Timings: Experimental loop: \n");
		for (int iImg = 0; iImg < n_sizes; iImg++)
		{
//			test_size[iImg]
			wxPrintf("Size %d: %3.3e\n",1,exp_span[iImg]/CLOCKS_PER_SEC /(n_inner_loops*n_iters));
		}
		delete [] exp_span;

		output_file.OpenFile("testWithMask.mrc", true);
		test_image.WriteSlices(&output_file, 1, test_image.logical_z_dimension);
		output_file.CloseFile();

		exit(-1);
	}



	// TODO Set even range in z to avoid large zero areas
	// TODO Set a check on the solvent fraction and scaling and report if it is unreasonable. Define reasonable
	// TODO Set a check on the range of values, report if defocus tolerance is too small (should all be positive)

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
	float wanted_amplitude_contrast = this->amplitude_contrast;
	float wanted_defocus_1_in_angstroms = this->defocus; // A
	float wanted_defocus_2_in_angstroms = this->defocus; //A
	float wanted_astigmatism_azimuth = 0.0; // degrees
	float astigmatism_angle_randomizer = 0.0; //
	float defocus_randomizer = 0.0;
	float wanted_additional_phase_shift_in_radians  = this->extra_phase_shift*PI;
    float *propagator_distance; // in Angstom, <= defocus tolerance.
    float defocus_offset = 0;



    if (beam_tilt_x != 0.0 || beam_tilt_y != 0.0 )
    {

    	wxPrintf("\n\nWhy am I here? %f %f\n\n",beam_tilt_x,beam_tilt_y);
    	DO_BEAM_TILT = true;
    	beam_tilt_azimuth = atan2f(beam_tilt_y,beam_tilt_x);
    	beam_tilt = sqrtf(powf(beam_tilt_x, 2) + powf(beam_tilt_y, 2));

    	beam_tilt   /= 1000.0f;
    	beam_tilt_x /= 1000.0f;
    	beam_tilt_y /= 1000.0f;

    	// These are needed in the propagation step and the final applicatino of beam tilt.
//    	beam_tilt_xform.SetToEulerRotation( rad_2_deg(beam_tilt_azimuth) , rad_2_deg(-beam_tilt), rad_2_deg(-beam_tilt_azimuth) );
    	// I want the transpose of the above (the order makes sense compared to emClarity)
    	beam_tilt_xform.SetToEulerRotation( rad_2_deg(-beam_tilt_azimuth) , rad_2_deg(beam_tilt), rad_2_deg(beam_tilt_azimuth) );
    	wxPrintf("%3.3e %3.3e %3.3e\n%3.3e %3.3e %3.3e\n %3.3e %3.3e %3.3e\n\n",beam_tilt_xform.m[0][0],beam_tilt_xform.m[0][1],beam_tilt_xform.m[0][2],
    											  							  beam_tilt_xform.m[1][0],beam_tilt_xform.m[1][1],beam_tilt_xform.m[1][2],
																			  beam_tilt_xform.m[2][0],beam_tilt_xform.m[2][1],beam_tilt_xform.m[2][2]);
    	float ix(-3.0f), iy(0.0f), iz(10.0f);
    	float dx(0.0f), dy(0.0f), dz(0.0f);

    	beam_tilt_xform.RotateCoords(ix,iy,iz,dx,dy,dz);
    	wxPrintf("%e %e %e\n",dx,dy,dz);

    }


    wxPrintf("Using extra phase shift of %f radians\n",wanted_additional_phase_shift_in_radians);

//    Image testFFT;
//    int testFFT_size = 5308;
//    testFFT_size = ReturnClosestFactorizedUpper(testFFT_size,3,true);
//    testFFT.Allocate(testFFT_size,testFFT_size,1,true);
//    testFFT.SetToConstant(0.0);
//    testFFT.AddGaussianNoise(1.0);
//
//    wxDateTime startFFT = wxDateTime::Now();
//    for (int iTime = 0; iTime < 10; iTime++)
//    {
//    	testFFT.ForwardFFT(true);
//		testFFT.BackwardFFT();
//    }
//    wxPrintf("Timings: FFT: %s\n",(wxDateTime::Now()-startFFT).Format());
//    throw;






	if (use_existing_params)
	{
		this->parameter_file.Open(preexisting_particle_file_name, 0,17);
		this->parameter_file.ReadFile(true, number_preexisting_particles);

		wxPrintf("\nRecreating %ld particles from the supplied parameter file\n", number_preexisting_particles);
		// Read the first line so that all of the values are initialized in parameter_vect
		this->parameter_file.ReadLine(this->parameter_vect);
		// Reset the counter to the first line
		this->parameter_file.current_line--;
	}
	else
	{
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
	}




    // TODO fix periodicity in Z on slab





    int iTilt;
    int nTilts;
    float max_tilt = 0;
    float * tilt_psi;
    float * tilt_theta;
    float * tilt_phi;

	// TODO either put into the class or better just update the global_random to use this.
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
    		if (fabsf(SET_TILT_ANGLES[iTilt]) > max_tilt) { max_tilt = fabsf(SET_TILT_ANGLES[iTilt]) ; }
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
    			this->parameter_file.ReadLine(this->parameter_vect);

        		tilt_psi[iTilt]   = parameter_vect[1];
        		tilt_theta[iTilt] = parameter_vect[2];
        		tilt_phi[iTilt]   = parameter_vect[3];

        		shift_x[iTilt]    = parameter_vect[4];
        		shift_y[iTilt]    = parameter_vect[5];
        		shift_z[iTilt]  = 0;
        		mag_diff[iTilt] =   parameter_vect[6];
    		}
    		else
    		{
        		tilt_psi[iTilt] = uniform_dist(gen)*360.0f; // *(2*PI);
        		tilt_theta[iTilt] = std::acos(2.0f*uniform_dist(gen)-1.0f) * 180.0f/(float)PI;
        		tilt_phi[iTilt] = -1*tilt_psi[iTilt] + uniform_dist(gen)*360.0f; //*(2*PI);

        		shift_x[iTilt]  = this->stdErr * normal_dist(gen); // should be in the low tens of Angstroms
        		shift_y[iTilt]  = this->stdErr * normal_dist(gen);
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



	// Not sure it makes sense to put this here.
    // Could I save memory by waiting and going slice by slice?
	Image *sum_image;
	Image *ref_image;
	Image *particle_stack;
	Image *ref_stack;
	RotationMatrix particle_rot;
	Image *reference_stack;


	// TODO Set allocation to save all frames _ edit needed spots down the road
	if (this->doParticleStack > 0 && ONLY_SAVE_SUMS)
	{
		// Output will be a stack of particles (not frames)
		sum_image = new Image[(int)(this->number_of_frames)];
		particle_stack = new Image[nTilts];

		if (SAVE_REF)
		{
			ref_image = new Image[(int)(this->number_of_frames)] ;
			ref_stack = new Image[nTilts];
		}

	}
	else
	{
		sum_image = new Image[(int)(nTilts*this->number_of_frames)];
		if (SAVE_REF)
		{
			ref_image = new Image[(int)(nTilts*this->number_of_frames)] ;
		}

	}

	// We only want one water box for a tilt series. For a particle stack, re-initialize for each particle.
	Water water_box(DO_PHASE_PLATE);
	// Create a new PDB object that represents the current state of the specimen, with each local motion applied.
	PDB current_specimen(this->number_of_non_water_atoms, this->do3d, this->minimum_paddeding_x_and_y, this->minimum_thickness_z);

	// Keep a copy of the unscaled pixel size to handle magnification changes.
	this->unscaled_pixel_size = this->wanted_pixel_size;

	wxPrintf("\nThere are %d tilts\n",nTilts);
    for ( iTilt = 0 ; iTilt < nTilts ; iTilt++)
    {

    	this->wanted_pixel_size = this->unscaled_pixel_size * mag_diff[iTilt];

    	this->wanted_pixel_size_sq = this->wanted_pixel_size * this->wanted_pixel_size;

    	wxPrintf("for Tilt %d, scaling the pixel size from %3.3f to %3.3f\n",this->unscaled_pixel_size,this->wanted_pixel_size);

    	float total_drift = 0.0f;
		RotationMatrix rotate_waters;
		if (this->tilt_series)
		{
//			rotate_waters.SetToRotation(-tilt_phi[iTilt],-tilt_theta[iTilt],-tilt_psi[iTilt]);
			rotate_waters.SetToEulerRotation(-tilt_psi[iTilt],-tilt_theta[iTilt],-tilt_phi[iTilt]);

		}
		else
		{
			rotate_waters.SetToRotation(euler1,euler2,euler3);

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
		    	wanted_defocus_1_in_angstroms = parameter_vect[8];
		    	wanted_defocus_2_in_angstroms = parameter_vect[9];
				wanted_astigmatism_azimuth    = parameter_vect[10];

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
	if (this->do3d > 1 || DO_PHASE_PLATE)
	{
		particle_rot.SetToIdentity();
		rotate_waters.SetToIdentity();

	}
	// pickup here TODO for checking amplitude
	if (DO_PHASE_PLATE)
	{
		max_tilt =  this->phase_plate_thickness;
	}


	for ( int iFrame = 0; iFrame < this->number_of_frames; iFrame ++)
	{



    	long iTilt_IDX;
    	if (this->doParticleStack > 0)
		{
    		iTilt_IDX = iFrame;
		}
    	else
    	{
        	iTilt_IDX = (long)((iTilt*this->number_of_frames)+iFrame);
    	}

    	int slab_nZ;
    	int rotated_Z; // full range in Z to cover the rotated specimen
    	float rotated_oZ;
    	float slab_oZ;
	    int nSlabs;
		int nS;
		float full_tilt_radians = 0;

		// Exposure filter TODO add check that it is wanted
		float *dose_filter;
		ElectronDose my_electron_dose(wanted_acceleration_voltage, this->wanted_pixel_size);


		Image jpr_sum_phase;
		Image jpr_sum_detector;


		// Create a new PDB object that represents the current state of the specimen, with each local motion applied.
//		PDB current_specimen(this->number_of_non_water_atoms, this->do3d, this->minimum_paddeding_x_and_y, this->minimum_thickness_z);


		// Include the max rand shift in z for thickness
		current_specimen.TransformLocalAndCombine(pdb_ensemble,this->number_of_pdbs,this->number_of_non_water_atoms,this->wanted_pixel_size, time_step, particle_rot, 0.0f); // Shift just defocus shift_z[iTilt]);

		// Calculated cutoffs to include > 99.5% of the potential for water at the given bFactor
		if (fabsf(this->min_bFactor + 1.0f) < 1e-3)
		{
			float BF = MIN_BFACTOR;
		    this->size_neighborhood 	  =  1 + myroundint( (0.4f *sqrtf(0.6f*BF) + 0.2f) / this->wanted_pixel_size);
		    wxPrintf("\n\n\tfor frame %d the size neigborhood is %d\n\n", iFrame, this->size_neighborhood);
		}
		else
		{
			float BF;
			if (DO_PHASE_PLATE) { BF = CARBON_BFACTOR ;} else { BF = return_bfactor_given_dose(current_specimen.average_bFactor);}

		    this->size_neighborhood 	  =  1 + myroundint( (0.4f *sqrtf(0.6f*BF) + 0.2f) / this->wanted_pixel_size);
		    wxPrintf("\n\n\tfor frame %d the size neigborhood is %d\n\n", iFrame, this->size_neighborhood);
		}



		if (DO_PHASE_PLATE)
		{
			this->size_neighborhood_water = this->size_neighborhood ;

		}
		else
		{
			this->size_neighborhood_water = myroundint(ceilf(1.0f / this->wanted_pixel_size));

		}

//	    this->size_neighborhood 	  = myroundint(ceilf(powf(CALC_DIST_OTHER,0.5)/ this->wanted_pixel_size));
//	    this->size_neighborhood_water = myroundint(ceilf(powf(CALC_DIST_WATER,0.5)/ this->wanted_pixel_size));
	    wxPrintf("using neighboorhood of %2.2f vox^3 for waters and %2.2f vox^3 for non-waters\n",powf(this->size_neighborhood_water*2+1,3),powf(this->size_neighborhood*2+1,3));

	    if ( DO_SOLVENT && this->need_to_allocate_projected_water)
	    {

	    	projected_water= new Image[SUB_PIXEL_NeL];

	        for (int iWater = 0 ; iWater < SUB_PIXEL_NeL; iWater++)
	        {
	            projected_water[iWater].Allocate(this->size_neighborhood_water*2+1,this->size_neighborhood_water*2+1,true);
	            projected_water[iWater].SetToConstant(0.0f);
	        }

			wxPrintf("Starting projected water calc with sizeN %d, %d\n",this->size_neighborhood_water*2+1,this->size_neighborhood_water*2+1);
			this->calc_water_potential(projected_water);
			wxPrintf("Finishing projected water calc\n");


			this->need_to_allocate_projected_water = false;
	    }

	    //Calculate 3d atomic potentials

	    // Use this value to determine if a water is too close to a non-water atom
	    this->calc_average_intensity_at_solvent_cutoff(current_specimen.average_bFactor);

//		Water water_box( &current_specimen,this->size_neighborhood_water, this->wanted_pixel_size, DOSE_PER_FRAME, max_tilt);
	    if (iTilt == 0 && iFrame == 0)
	    {
		    water_box.Init( &current_specimen,this->size_neighborhood_water, this->wanted_pixel_size, this->dose_per_frame, max_tilt);
	    }



		if (SAVE_TO_COMPARE_JPR || DO_PHASE_PLATE)
		{
			if (DO_PHASE_PLATE)
			{
				wxPrintf("\n\nSimulating a phase plate for validation, this overrides SAVE_TO_COMPARE_JPR\n\n");

				// Override the size of the specimen
				current_specimen.vol_angX = water_box.vol_angX; current_specimen.vol_nX = water_box.vol_nX; current_specimen.vol_oX = water_box.vol_oX;
				current_specimen.vol_angY = water_box.vol_angY; current_specimen.vol_nY = water_box.vol_nY; current_specimen.vol_oY = water_box.vol_oY;
				current_specimen.vol_angZ = water_box.vol_angZ; current_specimen.vol_nZ = water_box.vol_nZ; current_specimen.vol_oZ = water_box.vol_oZ;


				jpr_sum_phase.Allocate(water_box.vol_nX,water_box.vol_nY,true);
				jpr_sum_phase.SetToConstant(0.0f);

				jpr_sum_detector.Allocate(water_box.vol_nX,water_box.vol_nY,true);
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
				current_specimen.TransformGlobalAndSortOnZ(number_of_non_water_atoms, total_drift, total_drift, 0.0f, rotate_waters);


		// Compute the solvent fraction, with ratio of protein/ water density.
		// Assuming an average 2.2Ang vanderwaal radius ~50 cubic ang, 33.33 waters / cubic nanometer.

		if ( DO_SOLVENT  && water_box.number_of_waters == 0 && this->do3d < 1 )
		{
			// Waters are member variables of the scatteringPotential app - currentSpecimen is passed for size information.
			this->timer_start = wxDateTime::Now();


			if (DO_PRINT) {wxPrintf("n_waters added %ld\n", water_box.number_of_waters);}

			water_box.SeedWaters3d();

			if (DO_PRINT) {wxPrintf("n_waters added %ld\n", water_box.number_of_waters);}


//			water_seed_3d(&current_specimen);

			this->span_seed += wxDateTime::Now()-this->timer_start;

			if (DO_PRINT) {wxPrintf("Timings: seed_waters: %s\n",(this->span_seed).Format());}


			this->timer_start = wxDateTime::Now();

			water_box.ShakeWaters3d(this->number_of_threads);
//			water_shake_3d();

			this->span_shake += wxDateTime::Now()-this->timer_start;



		}
		else if ( DO_SOLVENT && this->do3d < 1 && ! DO_PHASE_PLATE)
		{

			this->timer_start = wxDateTime::Now();

			water_box.ShakeWaters3d(this->number_of_threads);

//			water_shake_3d();

			this->span_shake += wxDateTime::Now()-this->timer_start;
		}



		if (DO_PHASE_PLATE)
		{

			padded_x_dim = current_specimen.vol_nX;
			padded_y_dim = current_specimen.vol_nY;

		}
		else
		{
			padded_x_dim = ReturnClosestFactorizedUpper(IMAGEPADDING+current_specimen.vol_nX,7,true);
			padded_y_dim = ReturnClosestFactorizedUpper(IMAGEPADDING+current_specimen.vol_nY,7,true);

		}


		// TODO with new solvent add, the edges should not need to be tapered or padded
		sum_image[iTilt_IDX].Allocate(padded_x_dim,padded_y_dim,true);
		sum_image[iTilt_IDX].SetToConstant(0.0f);


		number_of_pixels_averaged.Allocate(current_specimen.vol_nX,current_specimen.vol_nY,true);
		number_of_pixels_averaged.SetToConstant(0.0f);
		number_of_non_waters_averaged.Allocate(current_specimen.vol_nX,current_specimen.vol_nY,true);
		number_of_non_waters_averaged.SetToConstant(0.0f);


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

		wxPrintf("wZ %ld csZ %ld,rotZ %d\n",water_box.vol_nZ,current_specimen.vol_nZ, rotated_Z);

		//rotated_oZ = ceilf((rotated_Z+1)/2);
		if (DO_PRINT) {wxPrintf("\nflat thicknes, %ld and rotated_Z %d\n", current_specimen.vol_nZ, rotated_Z);}
		wxPrintf("\nWorking on iTilt %d at %f degrees for frame %d\n",iTilt,tilt_theta[iTilt],iFrame);

		//  TODO Should separate the mimimal slab thickness, which is a smaller to preserve memory from the minimal prop distance (ie. project sub regions of a slab)
		if ( this->propagation_distance < 0 ) {nSlabs  = 1; this->propagation_distance = fabsf(this->propagation_distance);}
		else { nSlabs = ceilf( (float)rotated_Z * this->wanted_pixel_size/  this->propagation_distance);}

		Image *scattering_potential = new Image[nSlabs];
		Image *inelastic_potential = new Image[nSlabs];

		Image *ref_potential;
		if (SAVE_REF) { ref_potential = new Image[nSlabs] ;}


		nS = ceil((float)rotated_Z / (float)nSlabs);
		wxPrintf("rotated_Z %d nSlabs %d nS %d\n",rotated_Z,nSlabs,nS);

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
			wxPrintf("\n\nnSlabs %d\niSlab %d\nlSlab %d\n",nSlabs,slabIDX_end[0]+1,slabIDX_start[nSlabs]+1);
		}

		propagator_distance = new float[nSlabs];


		Image Potential_3d;

		for (iSlab = 0; iSlab < nSlabs; iSlab++)
		{

			propagator_distance[iSlab] =  ( this->wanted_pixel_size * (slabIDX_end[iSlab] - slabIDX_start[iSlab] + 1) );
			scattering_potential[iSlab].Allocate(current_specimen.vol_nX, current_specimen.vol_nY,1);
			scattering_potential[iSlab].SetToConstant(0.0);
			inelastic_potential[iSlab].Allocate(current_specimen.vol_nX, current_specimen.vol_nY,1);
			inelastic_potential[iSlab].SetToConstant(0.0);

			if (SAVE_REF)
			{
				ref_potential[iSlab].Allocate(current_specimen.vol_nX, current_specimen.vol_nY,1);
			}

			slab_nZ = slabIDX_end[iSlab] - slabIDX_start[iSlab] + 1;// + 2*this->size_neighborhood;
			slab_oZ = floorf(slab_nZ/2); // origin in the volume containing the rotated slab
			rotated_oZ = floorf(rotated_Z/2);
			// Because we will project along Z, we could put Z on the rows
			if (DO_PRINT) {wxPrintf("iSlab %d %d %d\n",iSlab, slabIDX_start[iSlab],slabIDX_end[iSlab] );}
			if (DO_PRINT) {wxPrintf("slab_oZ %f slab_nZ %d rotated_oZ %f\n",slab_oZ,slab_nZ,rotated_oZ);}
			Image scattering_slab;
			scattering_slab.Allocate(current_specimen.vol_nX,current_specimen.vol_nY,slab_nZ);
			scattering_slab.SetToConstant(0.0);

			Image inelastic_slab;
			inelastic_slab.Allocate(current_specimen.vol_nX,current_specimen.vol_nY,slab_nZ);
			inelastic_slab.SetToConstant(0.0);



			this->timer_start = wxDateTime::Now();

			if (! DO_PHASE_PLATE)
			{
				this->calc_scattering_potential(&current_specimen, &scattering_slab,&inelastic_slab,rotate_waters, rotated_oZ, slabIDX_start, slabIDX_end, iSlab);
			}

			this->span_atoms += (wxDateTime::Now()-this->timer_start);

			if (DO_PRINT) {wxPrintf("Span: calc_atoms: %s\n",(span_atoms).Format());}



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
					Potential_3d.Allocate(current_specimen.vol_nX,current_specimen.vol_nY,slabIDX_end[nSlabs-1] - slabIDX_start[0] + 1);
//					if (add_mean_water_potential)
//					{
//						wxPrintf("Adding a constant background potential of %f \n",this->lead_term * 4.871);
//						Potential_3d.SetToConstant(this->lead_term * 4.871);
//					}
//					else
//					{
						Potential_3d.SetToConstant(0.0);
//					}
				}


				if (add_mean_water_potential)
				{
					for (long current_pixel = 0; current_pixel < scattering_slab.real_memory_allocated; current_pixel++)
					{
						for (iPot = N_WATER_TERMS - 1; iPot >=0; iPot--)
						{
							if (scattering_slab.real_values[current_pixel] < this->average_at_cutoff[iPot])
							{

								scattering_slab.real_values[current_pixel] += this->water_weight[iPot];
								if (testHoles) {buffer.real_values[current_pixel] += this->water_weight[iPot];}
								break;
							}


						}
					}
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
					Potential_3d.Resize(cubic_size,cubic_size,cubic_size);
//					Potential_3d.QuickAndDirtyWriteSlices("tmpNotCropped.mrc",1,cubic_size);
					if (this->bin3d > 1)
					{
						wxPrintf("\nFourier cropping your 3d by a factor of %d\n",this->bin3d );
						Potential_3d.ForwardFFT(true);
						Potential_3d.Resize(cubic_size/this->bin3d,cubic_size/this->bin3d,cubic_size/this->bin3d);
						Potential_3d.BackwardFFT();
					}
					wxPrintf("Writing out your 3d slices %d --> %d\n",1,slabIDX_end[nSlabs-1] - slabIDX_start[0] + 1);
					Potential_3d.WriteSlices(&mrc_out,1,cubic_size/this->bin3d);
					mrc_out.SetPixelSize(this->wanted_pixel_size/this->bin3d);
					mrc_out.CloseFile();
					// Exit after writing the final slice for the reference. Is this the best way to do this? FIXME
					throw;
				}
				continue;

			}
			////////////////////


//			if (DO_EXPOSURE_FILTER == 3 && CALC_HOLES_ONLY == false && CALC_WATER_NO_HOLE == false)
			if (DO_EXPOSURE_FILTER == 3  && CALC_WATER_NO_HOLE == false)

			{
			// add in the exposure filter

				scattering_slab.ForwardFFT(true);

				ElectronDose my_electron_dose(wanted_acceleration_voltage, this->wanted_pixel_size);

				int j;
				int i;
				int k;

				long pixel_counter = 0;

				float y_coord_sq;
				float x_coord_sq;
				float z_coord_sq;

				float y_coord;
				float x_coord;
				float z_coord;

				float frequency_squared;
				float current_critical_dose;

				for (k = 0; k <= scattering_slab.physical_upper_bound_complex_z; k++)
				{
					z_coord = scattering_slab.ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * scattering_slab.fourier_voxel_size_z;
					z_coord_sq = powf(z_coord, 2.0);

					for (j = 0; j <= scattering_slab.physical_upper_bound_complex_y; j++)
					{
						y_coord = scattering_slab.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * scattering_slab.fourier_voxel_size_y;
						y_coord_sq = powf(y_coord, 2.0);

						for (i = 0; i <= scattering_slab.physical_upper_bound_complex_x; i++)
						{
							x_coord = i * scattering_slab.fourier_voxel_size_x;
							x_coord_sq = powf(x_coord, 2.0);


							// Compute the square of the frequency
							frequency_squared = x_coord_sq + y_coord_sq + z_coord_sq;

							current_critical_dose = my_electron_dose.ReturnCriticalDose(sqrtf(frequency_squared) / this->wanted_pixel_size);

							scattering_slab.complex_values[pixel_counter] *= my_electron_dose.ReturnDoseFilter((this->current_total_exposure+this->dose_per_frame), current_critical_dose);

							pixel_counter++;
						}
					}
				}

				scattering_slab.BackwardFFT();


			}



			if ( CALC_HOLES_ONLY == false )
			{
				this->project(&scattering_slab,scattering_potential,iSlab);
				this->project(&inelastic_slab,inelastic_potential,iSlab);
			}

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

//			if (DO_EXPOSURE_FILTER == 2 && CALC_HOLES_ONLY == false && CALC_WATER_NO_HOLE == false)
			if (DO_EXPOSURE_FILTER == 2 && CALC_WATER_NO_HOLE == false)

			{
			// add in the exposure filter
//				std::string fileNameOUT = "withOUT_DoseFilter_phaseGrating_" + std::to_string(iSlab) + this->output_filename;
//				scattering_potential[iSlab].QuickAndDirtyWriteSlice(fileNameOUT,1);
				scattering_potential[iSlab].ForwardFFT(true);
//				dose_filter = new float[scattering_potential[0].real_memory_allocated / 2];

				ElectronDose my_electron_dose(wanted_acceleration_voltage, this->wanted_pixel_size);

				int j;
				int i;

				long pixel_counter = 0;

				float y_coord_sq;
				float x_coord_sq;

				float y_coord;
				float x_coord;

				float frequency_squared;
				float current_critical_dose;


				for (j = 0; j <= scattering_slab.physical_upper_bound_complex_y; j++)
				{
					y_coord = scattering_slab.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * scattering_slab.fourier_voxel_size_y;
					y_coord_sq = powf(y_coord, 2.0);

					for (i = 0; i <= scattering_slab.physical_upper_bound_complex_x; i++)
					{
						x_coord = i * scattering_slab.fourier_voxel_size_x;
						x_coord_sq = powf(x_coord, 2.0);


						// Compute the square of the frequency
						frequency_squared = x_coord_sq + y_coord_sq;

						current_critical_dose = my_electron_dose.ReturnCriticalDose(sqrtf(frequency_squared) / this->wanted_pixel_size);

						scattering_slab.complex_values[pixel_counter] *= my_electron_dose.ReturnDoseFilter((this->current_total_exposure+this->dose_per_frame), current_critical_dose);

						pixel_counter++;
					}
				}

//				for (long pixel_counter = 0; pixel_counter < scattering_potential[0].real_memory_allocated / 2; pixel_counter++)
//				{
//					dose_filter[pixel_counter] = 0.0;
//				}
//				my_electron_dose.CalculateDoseFilterAs1DArray(&scattering_potential[iSlab], dose_filter, this->current_total_exposure, this->current_total_exposure+this->dose_per_frame );
//
//
//				for (long pixel_counter = 0; pixel_counter < scattering_potential[0].real_memory_allocated / 2; pixel_counter++)
//				{
//
//					if (DO_EXPOSURE_COMPLEMENT_PHASE_RANDOMIZE)
//					{
//						scattering_potential[iSlab].complex_values[pixel_counter] =
//								scattering_potential[iSlab].complex_values[pixel_counter] * (dose_filter[pixel_counter] + ( (1-dose_filter[pixel_counter]) *  exp(i2pi2 *uniform_dist(gen)) ));
//					}
//					else
//					{
//						scattering_potential[iSlab].complex_values[pixel_counter] =
//								scattering_potential[iSlab].complex_values[pixel_counter] * (dose_filter[pixel_counter]);
//					}
//

//				}
				scattering_potential[iSlab].BackwardFFT();
//				delete [] dose_filter;
				//////////
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
			}


			// Keep a clean copy of the ref with dose filtering
			if (SAVE_REF && EXPOSURE_FILTER_REF == true)
			{
				ref_potential[iSlab].CopyFrom(&scattering_potential[iSlab]);

			}


			if ( DO_SOLVENT && this->do3d < 1)
			{


				this->timer_start = wxDateTime::Now();

				// Now loop back over adding waters where appropriate
				if (DO_PRINT) {wxPrintf("Working on waters, slab %d\n",iSlab);}


				this->fill_water_potential(&current_specimen,&scattering_slab,
										   scattering_potential,inelastic_potential,&water_box,rotate_waters,
						   	   	   	   	   rotated_oZ, slabIDX_start, slabIDX_end, iSlab);

				this->span_waters += wxDateTime::Now() - this->timer_start;



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
			}



			scattering_slab.Deallocate();
			inelastic_slab.Deallocate();

			if (DO_PHASE_PLATE)
			{

				jpr_sum_phase.AddImage(&scattering_potential[iSlab]);

			}



			this->timer_start = wxDateTime::Now();


//			// Now apply the CTF - check that slab_oZ is doing what you intend it to TODO
//			float defocus_offset = ((slabIDX_end[iSlab]-slabIDX_start[iSlab])/2 - rotated_oZ + slabIDX_start[iSlab] + 1) * this->wanted_pixel_size;





		} // end loop nSlabs


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


		float average_propagator = 0;
		for (int defOffset = 0; defOffset < nSlabs; defOffset++)
		{
			average_propagator += propagator_distance[defOffset];
		}
		average_propagator /= nSlabs;
		defocus_offset -= average_propagator/2.0f;
		defocus_offset = ((float)rotated_Z*this->wanted_pixel_size +  this->propagation_distance)/2;

		wxPrintf("Propagator distance is %3.3e Angstroms, with offset for CTF of %3.3e Angstroms for the specimen.\n",propagator_distance[0],defocus_offset);


		wxPrintf("\n\t%ld out of bounds of %ld = percent\n\n", nOutOfBounds,number_of_non_water_atoms);

//		#pragma omp parallel num_threads(4)
//		{

		int nLoops = 1;
		if (SAVE_REF) { nLoops = 2; }

		for (int iLoop = 0; iLoop < nLoops; iLoop ++)
		{


		int iPar;
		int iSeq;

		Image *temp_img = new Image[4];
		Image *t_N = new Image[4];
		Image *wave_function = new Image[2];
		Image *phase_grating = new Image[2];
		Image *amplitude_grating;
		CTF *ctf = new CTF[2];
		CTF *propagator = new CTF[2];



		// Values to use in parallel sections
		const float set_wave_func[2] = {this->set_real_part_wave_function_in,0.0f};
		const int	copy_from_1[4] 	 = {0,1,1,0};
		const int	copy_from_2[4]	 = {0,1,1,0};
		const int  	mult_by[4]		 = {0,1,0,1};
		const int 	prop_apply_real[4] = {0,0,1,1};
		const int 	prop_apply_imag[4] = {1,1,0,0};
		const int  	ctf_apply[4]	   = {0,1,0,1};
		float total_shift_x = 0.0f;
		float total_shift_y = 0.0f;

		wxPrintf("%f %f %f %f %f %f\n",wanted_additional_phase_shift_in_radians,wanted_defocus_1_in_angstroms,wanted_defocus_2_in_angstroms,wanted_astigmatism_azimuth, wanted_spherical_aberration,propagator_distance[iSlab]);
		// get values for just sin or cos of the ctf by manipulating the amplitude term
		ctf[0].Init(wanted_acceleration_voltage,
					  wanted_spherical_aberration,
					  1.0,
					  wanted_defocus_1_in_angstroms+defocus_offset,
					  wanted_defocus_2_in_angstroms+defocus_offset,
					  wanted_astigmatism_azimuth,
					  this->wanted_pixel_size,
					  wanted_additional_phase_shift_in_radians+PI);
		ctf[1].Init(wanted_acceleration_voltage,
					  wanted_spherical_aberration,
					  0.0,
					  wanted_defocus_1_in_angstroms+defocus_offset,
					  wanted_defocus_2_in_angstroms+defocus_offset,
					  wanted_astigmatism_azimuth,
					  this->wanted_pixel_size,
					  wanted_additional_phase_shift_in_radians+PI);

		if (DO_COHERENCE_ENVELOPE )
		{
			ctf[0].SetEnvelope(wanted_acceleration_voltage,this->wanted_pixel_size,this->dose_rate / this->wanted_pixel_size_sq);
			ctf[1].SetEnvelope(wanted_acceleration_voltage,this->wanted_pixel_size,this->dose_rate / this->wanted_pixel_size_sq);
		}



		if (DO_PRINT) {wxPrintf("Propdist %f\n",propagator_distance[iSlab]);}
		// get values for just sin or cos of the ctf by manipulating the amplitude term
		// For the fresnel prop, set Cs = 0 and def = dz

		propagator[0].Init(wanted_acceleration_voltage,
						   0.0,
						   1.0,
						   - this->propagation_distance,
						   - this->propagation_distance,
						   0.0,
						   this->wanted_pixel_size,
						   0.0 + PI);
						// Shift of PI puts the image farther from focus, shift of zero puts it closer
		propagator[1].Init(wanted_acceleration_voltage,
						   0.0,
						   0.0,
						   - this->propagation_distance,
						   - this->propagation_distance,
						   0.0,
						   this->wanted_pixel_size,
						   0.0+PI);




		#pragma omp parallel for num_threads(propagate_threads_2)
		for (iPar = 0; iPar < 2; iPar++)
		{
			wave_function[iPar].Allocate(padded_x_dim,padded_y_dim,1);
			wave_function[iPar].SetToConstant(set_wave_func[iPar]);
			phase_grating[iPar].Allocate(padded_x_dim,padded_y_dim,1);
		}

		#pragma omp parallel for num_threads(propagate_threads_4)
		for (iPar = 0; iPar < 4; iPar++)
		{
			t_N[iPar].Allocate(padded_x_dim,padded_y_dim,1);
			if (DO_PRINT) {wxPrintf("Allocating t_N %d\n",iPar);}
			t_N[iPar].SetToConstant(0.0);
			temp_img[iPar].Allocate(padded_x_dim,padded_y_dim,1);
		}


		for (int iSlab = 0; iSlab < nSlabs; iSlab++)
		{

			// Taper here?
			this->taper_edges(scattering_potential, iSlab, false);
			this->taper_edges(inelastic_potential, iSlab, true);



			if (DO_BEAM_TILT)
			{
				// For tilted illumination, the scattering plane sees only the z-component of the wave-vector = lower energy = stronger interaction
				// so the interaction constant must be scaled. (Ishizuka 1982 eq 12) cos(B) = K/Kz K = 1/Lambda pointing to the displaced origin of the Ewald Sphere
				scattering_potential[iSlab].DivideByConstant(cosf(beam_tilt));
			}

			phase_grating[0].SetToConstant(0.0);
			phase_grating[1].SetToConstant(0.0);
			scattering_potential[iSlab].ClipInto(&phase_grating[0],this->image_mean[iSlab]);
			scattering_potential[iSlab].ClipInto(&phase_grating[1],this->image_mean[iSlab]);


			amplitude_grating = new Image[2];

			#pragma omp parallel for num_threads(propagate_threads_2)
			for (int iPG = 0; iPG < 2 ; iPG++)
			{

				amplitude_grating[iPG].Allocate(padded_x_dim,padded_y_dim,1);
				amplitude_grating[iPG].SetToConstant(0.0);
				inelastic_potential[iSlab].ClipInto(&amplitude_grating[iPG],this->inelastic_mean[iSlab]);

				float starting_sum = amplitude_grating[iPG].ReturnSumOfSquares(0.0f,0.0f,0.0f,0.0f);

//					float starting_sum = amplitude_grating[iPG].ReturnSumOfRealValues();
//					wxPrintf("\nn\The starting amplitude grating %d for slab %d is %e\n\n",iPG,iSlab,starting_sum);
				if (starting_sum > 1e-6)
				{

				// FIXME this is just multipied by the average value of C+C+N+0+C -> a projected mass density would be more accurate (26/Z*1.27) - this is also only valid at 300 KeV
//				amplitude_grating[iPG].MultiplyByConstant(3.2);
					if (wanted_acceleration_voltage < 301 && wanted_acceleration_voltage > 299)
					{
						amplitude_grating[iPG].MultiplyByConstant(1.27f);
					}
					else if(wanted_acceleration_voltage < 201 && wanted_acceleration_voltage > 199)
					{
						amplitude_grating[iPG].MultiplyByConstant(1.135f);
					}
					else
					{
						// Nothing to rescale at 100 KeV, this is where the inelastic/elastic ratios were empircally determined.
					}
				amplitude_grating[iPG].ForwardFFT(true);
				int i;
				int j;

				float x;
				float y;

				long pixel_counter = 0;
				float frequency_squared;

				for (j = 0; j <= amplitude_grating[iPG].physical_upper_bound_complex_y; j++)
				{
					y = powf(amplitude_grating[iPG].ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * amplitude_grating[iPG].fourier_voxel_size_y, 2);
					//#pragma omp simd simdlen(16)
					for (i = 0; i <= amplitude_grating[iPG].physical_upper_bound_complex_x; i++)
					{
						x = powf(i * amplitude_grating[iPG].fourier_voxel_size_x, 2);

						// compute squared radius, in units of reciprocal pixels angstroms
						frequency_squared = (x + y) / this->wanted_pixel_size_sq ;
						amplitude_grating[iPG].complex_values[pixel_counter] *= expf(-2500*powf(frequency_squared,2)); // FIXME only ~right for 300
						pixel_counter++;

					}
				}
				amplitude_grating[iPG].BackwardFFT();
				}


//					starting_sum = amplitude_grating[iPG].ReturnSumOfRealValues();
//					wxPrintf("\nn\The finishing amplitude grating %d for slab %d is %3.3e\n\n",iPG,iSlab,starting_sum);

			}



			  MyDebugAssertFalse(scattering_potential[iSlab].HasNan(),"There is a NAN 1");
			  MyDebugAssertFalse(phase_grating[0].HasNan(),"There is a NAN 2");
			  MyDebugAssertFalse(phase_grating[1].HasNan(),"There is a NAN 3");



			if (DO_COMPEX_AMPLITUDE_TERM)
			{


				// The amplitude contrast ratio entered is in reference to the image (squared modulus) which carries implicitly a factor of 2 for the amplitude term.
				float amplitude_term = -0.5f*wanted_amplitude_contrast;

				#pragma omp simd
				for ( long iPixel = 0; iPixel < phase_grating[0].real_memory_allocated; iPixel++)
				{
					phase_grating[0].real_values[iPixel] = expf(amplitude_term*amplitude_grating[0].real_values[iPixel]) * std::cos(phase_grating[0].real_values[iPixel]);
				}

				#pragma omp simd
				for ( long iPixel = 0; iPixel < phase_grating[1].real_memory_allocated; iPixel++)
				{
					phase_grating[1].real_values[iPixel] = expf(amplitude_term*amplitude_grating[1].real_values[iPixel]) * std::sin(phase_grating[1].real_values[iPixel]);
				}



			}
			else
			{
				#pragma omp simd
				// Could make this a sub-routine and use c++ threads or a logical arg to handle both cases (sin/cos)
				for ( long iPixel = 0; iPixel < phase_grating[0].real_memory_allocated; iPixel++)
				{
					phase_grating[0].real_values[iPixel] = std::cos(phase_grating[0].real_values[iPixel]);
				}
				#pragma omp simd
				for ( long iPixel = 0; iPixel < phase_grating[1].real_memory_allocated; iPixel++)
				{
					phase_grating[1].real_values[iPixel] = std::sin(phase_grating[1].real_values[iPixel]);
				}
			}


			  MyDebugAssertFalse(phase_grating[0].HasNan(),"There is a NAN 2a");
			  MyDebugAssertFalse(phase_grating[1].HasNan(),"There is a NAN 3a");

			#pragma omp parallel for num_threads(propagate_threads_4)
			for (iPar = 0; iPar < 4; iPar++)
			{
				t_N[iPar].CopyFrom(&wave_function[copy_from_1[iPar]]);
				t_N[iPar].MultiplyPixelWise(phase_grating[mult_by[iPar]]);

				  MyDebugAssertFalse(t_N[iPar].HasNan(),"There is a NAN t11");
				t_N[iPar].ForwardFFT(true);
				  MyDebugAssertFalse(t_N[iPar].HasNan(),"There is a NAN t11F");

			}

			// Reset the wave function to zero to store the update results
			wave_function[0].SetToConstant(0.0);
			wave_function[1].SetToConstant(0.0);

			#pragma omp parallel for num_threads(propagate_threads_4)
			for (iPar = 0; iPar < 4; iPar++)
			{


				// Get the real part of the new exit wave
				temp_img[iPar].CopyFrom(&t_N[iPar]);
				  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1");

					if (DO_BEAM_TILT)
					{
						// The propagtor must be normalized by Kz/K, and the current wave-front needs to be shifted BACK opposite the inclined
						// direction of propagtion, so that the next slice is relatively shifted forward along the direction of propagation.

						// Phase shifts here are in pixels & tan(B) ~ B

						// TODO make sure the sign is correct here
						if (DO_BEAM_TILT_FULL)
						{
							temp_img[iPar].PhaseShift(1.0f * propagator_distance[iSlab]*beam_tilt_x/this->wanted_pixel_size,1.0f* propagator_distance[iSlab]*beam_tilt_y/wanted_pixel_size,0.0f);
						}
						if (iPar == 0)
						{
							total_shift_x += cosf(beam_tilt)*1.0f*propagator_distance[iSlab]*beam_tilt_x/wanted_pixel_size;
							total_shift_y += cosf(beam_tilt)*1.0f*propagator_distance[iSlab]*beam_tilt_y/wanted_pixel_size;
							wxPrintf("%f, %f\n",total_shift_x,total_shift_y);
						}

	//					wxPrintf("Shifting b %f slab %d\n",-1.0f* propagator_distance[iSlab]*beam_tilt_y/wanted_pixel_size,iSlab);
					    temp_img[iPar].MultiplyByConstant(cosf(beam_tilt));

					}

				temp_img[iPar].ApplyCTF(propagator[prop_apply_real[iPar]],false);

				  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1CTF");
				temp_img[iPar].BackwardFFT();
				  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1BFFT");

			}

			for (iSeq = 0; iSeq < 4; iSeq++)
			{
				if (iSeq == 0)
				{
					wave_function[0].AddImage(&temp_img[iSeq]);
				}
				else
				{
					wave_function[0].SubtractImage(&temp_img[iSeq]);
				}
			}


			#pragma omp parallel for num_threads(propagate_threads_4)
			for (iPar = 0; iPar < 4; iPar++)
			{


				// Get the real part of the new exit wave
				temp_img[iPar].CopyFrom(&t_N[iPar]);
				  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1");


				temp_img[iPar].ApplyCTF(propagator[prop_apply_imag[iPar]],false);
				MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1CTF");
				temp_img[iPar].BackwardFFT();
				  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1BFFT");

			}

			for (iSeq = 0; iSeq < 4; iSeq++)
			{
				if (iSeq == 1)
				{
					wave_function[1].SubtractImage(&temp_img[iSeq]);
				}
				else
				{
					wave_function[1].AddImage(&temp_img[iSeq]);
				}
			}



		} // end of loop over slabs


			// Now apply the CTF for the center of mass, which also includes an envelopes (but not beam tilt as this depends has an axial dependence);
			#pragma omp parallel for num_threads(propagate_threads_4)
			for (iPar = 0; iPar < 4; iPar++)
			{

				// Re-use t_N[0] through t_N[3]
				t_N[iPar].CopyFrom(&wave_function[copy_from_2[iPar]]);
				t_N[iPar].ForwardFFT(true);

				// FIXME combine booleans and just pass without a need to check
				if (DO_COHERENCE_ENVELOPE )
				{
					t_N[iPar].ApplyCTF(ctf[ctf_apply[iPar]],false,false,true);
				}
				else
				{
					t_N[iPar].ApplyCTF(ctf[ctf_apply[iPar]],false);
				}

				 // FIXME this should be run through CTF
				if (DO_BEAM_TILT)
				{



					float spherical_aberration_in_angstrom = spherical_aberration * 1e7;
					float y_fourier_voxel_size_angstrom =  t_N[iPar].fourier_voxel_size_y / wanted_pixel_size;
					float x_fourier_voxel_size_angstrom =  t_N[iPar].fourier_voxel_size_x / wanted_pixel_size;
					float wavelength_squared = powf(wavelength,2);

					float mean_defocus = 0.5f*(wanted_defocus_1_in_angstroms+wanted_defocus_2_in_angstroms);

					float major_axis_scale = wanted_defocus_1_in_angstroms / mean_defocus;
					float minor_axis_scale = mean_defocus / wanted_defocus_2_in_angstroms;


					// real, imag, full
					float phase_shift = 0.0f;

					int j;
					int i;

					long pixel_counter = 0;

					float y_coord_sq;
					float x_coord_sq;

					float y_coord;
					float x_coord;

					float frequency_squared;
					float frequency;
					float azimuth;
					float phase_shift_major = 0.0f;
					float phase_shift_minor = 0.0f;


					for (j = 0; j <= t_N[iPar].physical_upper_bound_complex_y; j++)
					{
						y_coord = t_N[iPar].ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * y_fourier_voxel_size_angstrom;
						y_coord_sq = powf(y_coord, 2.0);


						for (i = 0; i <= t_N[iPar].physical_upper_bound_complex_x; i++)
						{
							x_coord = i * x_fourier_voxel_size_angstrom;
							x_coord_sq = powf(x_coord, 2);

							// Compute the azimuth
							if ( i == 0 && j == 0 ) {
								azimuth = 0.0;
							} else {
								azimuth = atan2f(y_coord,x_coord);
							}


							// Compute the square of the frequency
							frequency_squared = x_coord_sq + y_coord_sq;
							frequency = sqrtf(frequency_squared);


							// First term
							phase_shift_major =  2.0f * PIf * wavelength_squared  *
									             spherical_aberration_in_angstrom * frequency * frequency_squared * beam_tilt * cosf( azimuth - beam_tilt_azimuth );


							t_N[iPar].complex_values[pixel_counter] *= (cosf(phase_shift_major) - I*sinf(phase_shift_major));


							pixel_counter++;
						}
					}


				}

				if (DO_BEAM_TILT_FULL)
				{
					t_N[iPar].PhaseShift(-0.5f*total_shift_x,-0.5f*total_shift_y,0.0f);
				}

				t_N[iPar].BackwardFFT();

			}

			wave_function[0].SetToConstant(0.0);
			wave_function[1].SetToConstant(0.0);

			wave_function[0].AddImage(&t_N[0]);
			wave_function[0].SubtractImage(&t_N[1]);
			wave_function[1].AddImage(&t_N[2]);
			wave_function[1].AddImage(&t_N[3]);

			 MyDebugAssertFalse(wave_function[0].HasNan(),"There is a NAN 6");
			  MyDebugAssertFalse(wave_function[1].HasNan(),"There is a NAN 7");


			#pragma omp simd
			// Now get the square modulus of the wavefunction
			for (long iPixel = 0; iPixel < sum_image[iTilt_IDX].real_memory_allocated; iPixel++)
			{
				 sum_image[iTilt_IDX].real_values[iPixel] = (powf(wave_function[0].real_values[iPixel],2) + powf(wave_function[1].real_values[iPixel],2));
			}


		if (SAVE_PROBABILITY_WAVE && iLoop < 1)
		{
			std::string fileNameOUT = "withProbabilityWave_" + std::to_string(iTilt_IDX) + this->output_filename;
				MRCFile mrc_out(fileNameOUT,true);
				sum_image[iTilt_IDX].WriteSlices(&mrc_out,1,1);
				mrc_out.SetPixelSize(this->wanted_pixel_size);
				mrc_out.CloseFile();



		}





		if (SAVE_TO_COMPARE_JPR || DO_PHASE_PLATE && iLoop < 1)
		{
			// For comparing to JPR @ 0.965
			Image binImage;
			binImage.CopyFrom(&sum_image[iTilt_IDX]);
			if (! DO_PHASE_PLATE)
			{
				binImage.Resize(JPR_SIZE,JPR_SIZE,1);
			}
			jpr_sum_detector.AddImage(&binImage);

		}



		if (DO_APPLY_DQE && iLoop < 1)
		{
			if (SAVE_WITH_DQE)
			{
				std::string fileNameOUT = "withOUT_DQE_" + std::to_string(iTilt_IDX) + this->output_filename;
					MRCFile mrc_out(fileNameOUT,true);
					sum_image[iTilt_IDX].WriteSlices(&mrc_out,1,1);
					mrc_out.SetPixelSize(this->wanted_pixel_size);
					mrc_out.CloseFile();
			}

			this->apply_sqrt_DQE_or_NTF(sum_image,  iTilt_IDX, true);
			if (SAVE_WITH_DQE)
			{
				std::string fileNameOUT = "withDQE_" + std::to_string(iTilt_IDX) + this->output_filename;
					MRCFile mrc_out(fileNameOUT,true);
					sum_image[iTilt_IDX].WriteSlices(&mrc_out,1,1);
					mrc_out.SetPixelSize(this->wanted_pixel_size);
					mrc_out.CloseFile();
			}
		}




		if (DEBUG_POISSON == false && iLoop < 1 && DO_PHASE_PLATE == false)
		{


			// Next we draw from a poisson distribution and then finally apply the NTF
			Image cpp_poisson;
			cpp_poisson.Allocate(sum_image[iTilt_IDX].logical_x_dimension,sum_image[iTilt_IDX].logical_y_dimension,1,true);
			cpp_poisson.SetToConstant(0.0);


			std::default_random_engine generator;

			std::random_device rd;  //Will be used to obtain a seed for the random number engine
			std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
			std::uniform_real_distribution<> dis(0.000000001, 1.0);



			for (long iPixel = 0; iPixel < sum_image[iTilt_IDX].real_memory_allocated; iPixel++ )
			{
				std::poisson_distribution<int> distribution(sum_image[iTilt_IDX].real_values[iPixel]);
//					sum_image[iTilt_IDX].real_values[iPixel] = global_random_number_generator.GetPoissonRandom(sum_image[iTilt_IDX].real_values[iPixel]);
				//sum_image[iTilt_IDX].real_values[iPixel] = distribution(generator);
//						pre_poisson.real_values[iPixel] += global_random_number_generator.GetPoissonRandom(sum_image[iTilt_IDX].real_values[iPixel],dis(gen));
				// Only observed in large images (4-6K) so far, the frames from these movies show strong ripples in X in CTFFIND spectrum but not unBlur.
				// Using the probdist to generate in matlab seems to largely alleviate this, as does using the cpp generator. This should be looked at, to see if it is
				// a problem with my poisson rng, or with the underlying uniform_distribution. For progress sake, just using CPP right now. TODO
				cpp_poisson.real_values[iPixel] += distribution(gen);

			}


			sum_image[iTilt_IDX].CopyFrom(&cpp_poisson);

			// Trouble shooting poisson vs ripples
//				cpp_poisson.QuickAndDirtyWriteSlice("banding_cppPoisson.mrc",1);
//				pre_poisson.QuickAndDirtyWriteSlice("banding_bahPoisson.mrc",1);

		}



		if (SAVE_POISSON_PRE_NTF && iLoop < 1)
		{

			std::string fileNameOUT = "withPoisson_noNTF_" + std::to_string(iTilt_IDX) + this->output_filename;
				MRCFile mrc_out(fileNameOUT,true);
				sum_image[iTilt_IDX].WriteSlices(&mrc_out,1,1);
				mrc_out.SetPixelSize(this->wanted_pixel_size);
				mrc_out.CloseFile();

		}


		if (DO_APPLY_NTF && DEBUG_POISSON == false && iLoop < 1)
		{

			if (SAVE_POISSON_WITH_NTF)
			{

				std::string fileNameOUT = "withOUT_Poisson_withNTF_" + std::to_string(iTilt_IDX) + this->output_filename;
					MRCFile mrc_out(fileNameOUT,true);
					sum_image[iTilt_IDX].WriteSlices(&mrc_out,1,1);
					mrc_out.SetPixelSize(this->wanted_pixel_size);
					mrc_out.CloseFile();

			}

			// Now apply NTF fit with Fourier series, 1 term. Values now for 300kV @ 3e-/physical pixel*s TODO consider other rates, and actually convert physical pixel size etc.
			// ONLY VALID UP TO PHYSICAL NYQUIST WHICH SHOULD BE SET TO 1.0
			this->apply_sqrt_DQE_or_NTF(sum_image, iTilt_IDX,false);

//				std::string fileNameOUT = "with_DQE_BTF_poisson_" + std::to_string(iTilt_IDX) + this->output_filename;
//				sum_image[iTilt_IDX].QuickAndDirtyWriteSlice("withDQE_NTF_poisson.mrc",1);

			// Now round as threshold for counting mode (since NTF spreads poisson counts a bit)
			for (long iPixel = 0; iPixel < sum_image[iTilt_IDX].real_memory_allocated; iPixel++ )
			{
				sum_image[iTilt_IDX].real_values[iPixel] = myround(sum_image[iTilt_IDX].real_values[iPixel]);
			}//

			if (SAVE_POISSON_WITH_NTF)
			{

				std::string fileNameOUT = "withPoisson_withNTF_" + std::to_string(iTilt_IDX) + this->output_filename;
					MRCFile mrc_out(fileNameOUT,true);
					sum_image[iTilt_IDX].WriteSlices(&mrc_out,1,1);
					mrc_out.SetPixelSize(this->wanted_pixel_size);
					mrc_out.CloseFile();

			}
		}

		delete [] temp_img;
		delete [] t_N;
		delete [] wave_function;
		delete [] phase_grating;
		delete [] ctf;
		delete [] propagator;


		if (SAVE_REF)
		{
			if (iLoop == 0)
			{
				// Copy the sum_image into the reference for storage
				ref_image[iTilt_IDX].CopyFrom(&sum_image[iTilt_IDX]);
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
				bufferImg.CopyFrom(&sum_image[iTilt_IDX]);
				sum_image[iTilt_IDX].CopyFrom(&ref_image[iTilt_IDX]);
				ref_image[iTilt_IDX].CopyFrom(&bufferImg);

			}
		}


		} // loop over perfect reference




//		} // end fft push omp block



//	    delete [] slabIDX_start;
//	    delete [] slabIDX_end;
//	    if (SOLVENT != 0) delete [] this->image_mean;

	    this->span_propagate += wxDateTime::Now()-this->timer_start;
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

		delete [] scattering_potential;
		if (SAVE_REF)
		{
			delete [] ref_potential;
		}

		delete [] propagator_distance;
		defocus_offset = 0;


    } // end of loop over frames




		if (this->doParticleStack > 0)
		{

			// Reset the water count so that a new water_box is initialized for the next particle
			water_box.number_of_waters = 0;
//			// Reset the dose
			this->current_total_exposure = this->pre_exposure;

			if (! this->use_existing_params)
			{
				parameter_vect[0] = iTilt + 1;
				parameter_vect[1] = tilt_psi[iTilt];
				parameter_vect[2] = tilt_theta[iTilt];
				parameter_vect[3] = tilt_phi[iTilt];
				if (this->stdErr != 0)
				{
					parameter_vect[4]  = shift_x[iTilt]; // shx
					parameter_vect[5]  = shift_y[iTilt]; // shy
				}
				parameter_vect[6] = mag_diff[iTilt];
				parameter_vect[8] = wanted_defocus_1_in_angstroms;
				parameter_vect[9] = wanted_defocus_2_in_angstroms;
				parameter_vect[10]= wanted_astigmatism_azimuth;

				parameter_file.WriteLine(parameter_vect, false);
			}


			if ( ! ONLY_SAVE_SUMS)
			{
//				particle_stack.CopyFrom(&sum_image);
			}
			else
			{

				particle_stack[iTilt].Allocate(sum_image[0].logical_x_dimension,sum_image[0].logical_y_dimension,true);
				particle_stack[iTilt].CopyFrom(&sum_image[0]);
				if (SAVE_REF)
				{
					ref_stack[iTilt].Allocate(ref_image[0].logical_x_dimension,ref_image[0].logical_y_dimension,true);
					ref_stack[iTilt].CopyFrom(&ref_image[0]);
				}


				if (DO_EXPOSURE_FILTER_FINAL_IMG )
				{

					// sum the frames
					float final_img_exposure;
					float *dose_filter;
					float *dose_filter_sum_of_squares = new float[sum_image[0].real_memory_allocated / 2];
					ZeroFloatArray(dose_filter_sum_of_squares, sum_image[0].real_memory_allocated/2);
					ElectronDose my_electron_dose(wanted_acceleration_voltage, this->wanted_pixel_size);

					particle_stack[iTilt].ForwardFFT(true);
					if (EXPOSURE_FILTER_REF && SAVE_REF)
					{
						ref_stack[iTilt].ForwardFFT(true);
					}

					for (int iFrame = 1; iFrame < this->number_of_frames; iFrame++)
					{

						dose_filter = new float[sum_image[iFrame].real_memory_allocated / 2];
						for (long pixel_counter = 0; pixel_counter < sum_image[iFrame].real_memory_allocated / 2; pixel_counter++)
						{
							dose_filter[pixel_counter] = 0.0;
						}

						sum_image[iFrame].ForwardFFT(true);
						if (EXPOSURE_FILTER_REF && SAVE_REF)
						{
							ref_image[iFrame].ForwardFFT(true);
						}
						my_electron_dose.CalculateDoseFilterAs1DArray(&sum_image[iFrame], dose_filter, (iFrame-1)*this->dose_per_frame, iFrame*this->dose_per_frame );



						for (long pixel_counter = 0; pixel_counter < sum_image[iFrame].real_memory_allocated / 2; pixel_counter++)
						{

							sum_image[iFrame].complex_values[pixel_counter] *= dose_filter[pixel_counter];
							if (EXPOSURE_FILTER_REF && SAVE_REF)
							{
								ref_image[iFrame].complex_values[pixel_counter] *=  dose_filter[pixel_counter];
							}

							dose_filter_sum_of_squares[pixel_counter] += powf(dose_filter[pixel_counter],2);
						}

						particle_stack[iTilt].AddImage(&sum_image[iFrame]);


						delete [] dose_filter;
						if (SAVE_REF)
						{
						  ref_stack[iTilt].AddImage(&ref_image[iFrame]);
						}


					}

					for (long pixel_counter = 0; pixel_counter < particle_stack[iTilt].real_memory_allocated / 2; pixel_counter++)
					{
						particle_stack[iTilt].complex_values[pixel_counter] /= sqrtf(dose_filter_sum_of_squares[pixel_counter]);
						if (EXPOSURE_FILTER_REF && SAVE_REF)
						{
							ref_stack[iTilt].complex_values[pixel_counter] /= sqrtf(dose_filter_sum_of_squares[pixel_counter]);
						}

					}

					if (EXPOSURE_FILTER_REF && SAVE_REF)
					{
					  ref_stack[iTilt].BackwardFFT();
					}
					particle_stack[iTilt].BackwardFFT();

					delete [] dose_filter_sum_of_squares;

				}
				else
				{
					for (int iFrame = 1; iFrame < this->number_of_frames; iFrame++)
					{
						particle_stack[iTilt].AddImage(&sum_image[iFrame]);
						if (SAVE_REF)
						{
						  ref_stack[iTilt].AddImage(&ref_image[iFrame]);
						}
					}

				}


			}


			// Normalize to 1 sigma, at a radius 2/3 the box size
			/////particle_stack[iTilt].ZeroFloatAndNormalize(1.0,0.33*particle_stack[0].logical_x_dimension,true);
		}
		else if (this->tilt_series && ! this->use_existing_params)
		{

			parameter_vect[0] = iTilt + 1;
			parameter_vect[1] = tilt_psi[iTilt];
			parameter_vect[2] = tilt_theta[iTilt];
			parameter_vect[3] = tilt_phi[iTilt];
			if (this->stdErr != 0)
			{
				parameter_vect[4]  = shift_x[iTilt]; // shx
				parameter_vect[5]  = shift_y[iTilt]; // shy
			}
		    parameter_vect[6]  = mag_diff[iTilt]; // mag

		    parameter_file.WriteLine(parameter_vect, false);


		}


    } // end of loop over tilts


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

	if (this->doParticleStack > 0)
	{

		CTF my_ctf;

		// FIXME
		// Pick the largest size and clip all to that
		int maxX = 0;
		int maxY = 0;
		int maxSize;

		for (iTilt=0; iTilt < nTilts; iTilt++)
		{
			if (particle_stack[iTilt].logical_x_dimension - IMAGETRIMVAL  > maxX) {maxX = particle_stack[iTilt].logical_x_dimension - IMAGETRIMVAL;}
			if (particle_stack[iTilt].logical_y_dimension - IMAGETRIMVAL  > maxY) {maxY = particle_stack[iTilt].logical_y_dimension - IMAGETRIMVAL;}

		}

		if (maxX == 0 || maxY == 0)
		{
			wxPrintf("Something went quite wrong in determining the max image dimenstions for the particle stack maxX %d, maxY %d",maxX,maxY);
			throw;
		}
		else
		{
			// Make the particle Square
		    maxSize = std::min(maxX,maxY);

		}

		if (WHITEN_IMG)
		{
			whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((maxX / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
			number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((maxX / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
		}


		if (CORRECT_CTF)
		{

			// Is the close and open really necessary? FIXME
			int nLines = this->parameter_file.number_of_lines;
			this->parameter_file.Close();
			this->parameter_file.Open(this->parameter_file_name,0,17);
			this->parameter_file.ReadFile(false, nLines);
//			this->parameter_file.ReadLine(this->parameter_vect); // Read in the comment line

		}

		for (iTilt=0; iTilt < nTilts; iTilt++)
		{


			// I am getting some nans on occassion, but so far they only show up in big expensive calcs, so add a nan check and print info to see if
			// the problem can be isolated.
			if (particle_stack[iTilt].HasNan() == true)
			{
				wxPrintf("Frame %d / %d has NaN values, trashing it\n",iTilt,nTilts*(int)this->number_of_frames);
			}
			else
			{

				if (CORRECT_CTF)
				{
					this->parameter_file.ReadLine(this->parameter_vect);
					wxPrintf("%3.3e %3.3e %3.3e %3.3e %3.3e %3.3e %3.3e %3.3e\n",this->kV,this->spherical_aberration,this->amplitude_contrast,
							    parameter_vect[8],parameter_vect[9],parameter_vect[10],this->wanted_pixel_size,parameter_vect[11]);
					my_ctf.Init(this->kV,this->spherical_aberration,this->amplitude_contrast,
							    parameter_vect[8],parameter_vect[9],parameter_vect[10],this->wanted_pixel_size,parameter_vect[11]);

				}
				particle_stack[iTilt].Resize(maxSize,maxSize,1);
				if (SAVE_REF) {	ref_stack[iTilt].Resize(maxSize,maxSize,1); }


//				// temp to test
//				Image testImg;
//				testImg.Allocate(512,512,false);
//				testImg.SetToConstant(1.0);
//				testImg.ApplyCTF(my_ctf,true);
//				testImg.QuickAndDirtyWriteSlice("ctf_noEnvelop.mrc",1);
//				testImg.SetToConstant(1.0);
//				my_ctf.SetEnvelope(wanted_acceleration_voltage, this->dose_rate * this->wanted_pixel_size_sq);
//				testImg.ApplyCTF(my_ctf,true, false, true);
//				testImg.QuickAndDirtyWriteSlice("ctf_withEnvelope.mrc",1);
//
//				exit(-1);

				if (WHITEN_IMG)
				{

					particle_stack[iTilt].ForwardFFT(true);

					particle_stack[iTilt].ZeroCentralPixel();
					particle_stack[iTilt].Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
					whitening_filter.SquareRoot();
					whitening_filter.Reciprocal();
					whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue());

					//whitening_filter.WriteToFile("/tmp/filter.txt");
					particle_stack[iTilt].ApplyCurveFilter(&whitening_filter);
					particle_stack[iTilt].ZeroCentralPixel();

					if (CORRECT_CTF) { particle_stack[iTilt].ApplyCTF(my_ctf,false,false) ;} // TODO is this the right spot to put this?

					particle_stack[iTilt].DivideByConstant(sqrtf(particle_stack[iTilt].ReturnSumOfSquares()));
					particle_stack[iTilt].BackwardFFT();

					if (SAVE_REF)
					{

						ref_stack[iTilt].ForwardFFT(true);
					    ref_stack[iTilt].ZeroCentralPixel();
						ref_stack[iTilt].ApplyCurveFilter(&whitening_filter);
						ref_stack[iTilt].ZeroCentralPixel();
						if (CORRECT_CTF) {ref_stack[iTilt].ApplyCTF(my_ctf,false,false) ;} // TODO is this the right spot to put this?
						ref_stack[iTilt].DivideByConstant(sqrt(ref_stack[iTilt].ReturnSumOfSquares()));
						ref_stack[iTilt].BackwardFFT();
					}

				}

				particle_stack[iTilt].WriteSlices(&mrc_out_final,1+iTilt,1+iTilt);
				if (SAVE_REF)
				{
					ref_stack[iTilt].WriteSlices(&mrc_ref_final,1+iTilt,1+iTilt);
				}


			}
		}
	}
	else
	{


		int current_tilt_sub_frame = 1;
		int current_tilt_sum_saved = 1;
		int xDIM = sum_image[0].logical_x_dimension - IMAGETRIMVAL;
		int yDIM = sum_image[0].logical_y_dimension - IMAGETRIMVAL;

		// Make the particle Square
	    int maxSize = std::max(xDIM,yDIM);

		Image tilt_sum;
		Image ref_sum;


		// This assumes all tilts have been made the same size (which they should be.)
//		tilt_sum.Allocate(xDIM,yDIM, 1);
		tilt_sum.Allocate(maxSize,maxSize, 1);
		tilt_sum.SetToConstant(0.0);

		if (SAVE_REF)
		{
//			ref_sum.Allocate(xDIM,yDIM, 1);
			ref_sum.Allocate(maxSize,maxSize, 1);
			ref_sum.SetToConstant(0.0);
		}

		if (WHITEN_IMG)
		{
			whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((maxSize / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
			number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((maxSize / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
		}

		for (iTilt=0; iTilt < nTilts*this->number_of_frames; iTilt++)
		{
			// I am getting some nans on occassion, but so far they only show up in big expensive calcs, so add a nan check and print info to see if
			// the problem can be isolated.
			if (sum_image[iTilt].HasNan() == true)
			{
				wxPrintf("Frame %d / %d has NaN values, trashing it\n",iTilt,nTilts*(int)this->number_of_frames);
			}
			else
			{


				sum_image[iTilt].Resize(maxSize,maxSize, 1);
				if (SAVE_REF)
				{
					ref_image[iTilt].Resize(maxSize,maxSize, 1);
				}

				sum_image[iTilt].WriteSlices(&mrc_out_final,1+iTilt,1+iTilt);

				if (current_tilt_sub_frame < this->number_of_frames)
				{
					tilt_sum.AddImage(&sum_image[iTilt]);
					if (SAVE_REF)
					{
						ref_sum.AddImage(&ref_image[iTilt]);
					}

					current_tilt_sub_frame += 1;
				}
				else if (current_tilt_sub_frame == this->number_of_frames)
				{
					tilt_sum.AddImage(&sum_image[iTilt]);
					if (SAVE_REF) { ref_sum.AddImage(&ref_image[iTilt]); }

					if (WHITEN_IMG)
					{
						tilt_sum.ForwardFFT(true);
						tilt_sum.ZeroCentralPixel();
						tilt_sum.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
						whitening_filter.SquareRoot();
						whitening_filter.Reciprocal();
						whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue());

						//whitening_filter.WriteToFile("/tmp/filter.txt");
						tilt_sum.ApplyCurveFilter(&whitening_filter);
						tilt_sum.ZeroCentralPixel();
						tilt_sum.DivideByConstant(sqrt(tilt_sum.ReturnSumOfSquares()));
						tilt_sum.BackwardFFT();


						if (SAVE_REF)
						{

							ref_sum.ForwardFFT(true);
						    ref_sum.ZeroCentralPixel();
							ref_sum.ApplyCurveFilter(&whitening_filter);
							ref_sum.ZeroCentralPixel();
							ref_sum.DivideByConstant(sqrt(ref_sum.ReturnSumOfSquares()));
							ref_sum.BackwardFFT();
						}

					}

					tilt_sum.WriteSlices(&mrc_tlt_final,current_tilt_sum_saved,current_tilt_sum_saved);
					tilt_sum.SetToConstant(0.0);

					if (SAVE_REF)
					{
						ref_sum.WriteSlices(&mrc_ref_final,current_tilt_sum_saved,current_tilt_sum_saved);
						ref_sum.SetToConstant(0.0);
					}

					current_tilt_sum_saved += 1;
					current_tilt_sub_frame = 1;
				}

			}
		}
	}


	mrc_out_final.SetPixelSize(this->wanted_pixel_size);
	mrc_out_final.CloseFile();

	mrc_tlt_final.SetPixelSize(this->wanted_pixel_size);
	mrc_tlt_final.CloseFile();

	this->parameter_file.Close();

	delete [] tilt_psi;
	delete [] tilt_theta;
	delete [] tilt_phi;
	delete [] shift_x;
	delete [] shift_y;
	delete [] shift_z;


    delete [] sum_image;

 //   delete noise_dist;



}

void ScatteringPotentialApp::calc_scattering_potential(const PDB * current_specimen,Image *scattering_slab,Image *inelastic_slab,
		                                               RotationMatrix rotate_waters,
													   float rotated_oZ, int *slabIDX_start, int *slabIDX_end, int iSlab)

// The if conditions needed to have water and protein in the same function
// make it too complicated and about 10x less parallel friendly.
{

	long nAtoms;
	long current_atom;
	long normalization_factor[this->number_of_threads];
	int  z_low = slabIDX_start[iSlab] - size_neighborhood;
	int  z_top = slabIDX_end[iSlab] + size_neighborhood;

	float slab_half_thickness_angstrom = (slabIDX_end[iSlab] - slabIDX_start[iSlab] + 1)*this->wanted_pixel_size/2.0f;
//	wxPrintf("z_low %d and z_top %d\n",z_low,z_top);
	double avg_scale;
	double nAvg;


	nAtoms = this->number_of_non_water_atoms;



	// TODO experiment with the scheduling. Until the specimen is consistently full, many consecutive slabs may have very little work for the assigned threads to handle.
	#pragma omp parallel for  num_threads(this->number_of_threads)

	for (long current_atom = 0; current_atom < nAtoms; current_atom++)
	{

		int element_id;
		float element_inelastic_ratio;
		float bFactor;
		float bPlusB[5];
		float bPlusB_outside[5];
		float radius;
		float ix(0), iy(0), iz(0);
		float dx(0), dy(0), dz (0);
		float x1(0.0f), y1(0.0f), z1(0.0f);
		float x2(0.0f), y2(0.0f), z2(0.0f);
		int indX(0), indY(0), indZ(0);
		float sx(0), sy(0), sz(0);
		float xLow(0),xTop(0),yLow(0),yTop(0),zLow(0),zTop(0);
		int iLim, jLim, kLim;
		int iGaussian;
		float water_offset;
		long atoms_added_idx[(int)powf(size_neighborhood*2+1,3)];
		float atoms_values_tmp[(int)powf(size_neighborhood*2+1,3)];
		float atoms_distances_tmp[(int)powf(size_neighborhood*2+1,3)];

		int n_atoms_added =0;
		float temp_potential = 0.0f;
		double norm_value = 0;
		float pixel_offset = 0.5f;
//		int threadIDX = omp_get_thread_num();
		float bfX(0), bfY(0), bfZ(0);


		element_id = current_specimen->my_atoms.Item(current_atom).element_name;
		element_inelastic_ratio = 20.2f / ATOMIC_NUMBER[element_id]; // Reimer/Ross_Messemer 1989


		bFactor = return_bfactor_given_dose(current_specimen->my_atoms.Item(current_atom).relative_bfactor);
		if (this->min_bFactor == -1)
		{

		    std::random_device rd;  //Will be used to obtain a seed for the random number engine
		    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

		    // Divide the crystallographic bFactor by 8pi^2 to get the mean squared displacement
			bFactor = sqrt(bFactor/(78.95680235571201)); // 8*pi**2

			std::uniform_real_distribution<float> norm_dist(0.0, bFactor*(1.5)); // The 1.5 is included so the RMSD in 3D is 1*bFactor			bfX = norm_dist(gen);
			bfY = norm_dist(gen);
			bfZ = norm_dist(gen);
			bFactor     =  MIN_BFACTOR;//current_specimen->my_atoms.Item(current_atom).bfactor*this->bFactor_scaling
//			wxPrintf("%f %f %f\n",bfX,bfY,bfZ);

		}

		corners R;


		// Convert atom origin to pixels and shift by volume origin to get pixel coordinates. Add 0.5 to place origin at "center" of voxel
		dx = modff(current_specimen->vol_oX + (current_specimen->my_atoms.Item(current_atom).x_coordinate / this->wanted_pixel_size + 0.5f), &ix);
		dy = modff(current_specimen->vol_oY + (current_specimen->my_atoms.Item(current_atom).y_coordinate / this->wanted_pixel_size + 0.5f), &iy);
		dz = modff(rotated_oZ 			    + (current_specimen->my_atoms.Item(current_atom).z_coordinate / this->wanted_pixel_size + 0.5f), &iz);



		// With the corect pixel indices in ix,iy,iz now subtract off the 0.5
		dx -= pixel_offset;
		dy -= pixel_offset;
		dz -= pixel_offset;

		#pragma omp simd
		for (iGaussian = 0; iGaussian < 5 ; iGaussian++)
		{
			bPlusB[iGaussian] = 2*PI/sqrt(bFactor+SCATTERING_PARAMETERS_B[element_id][iGaussian]);
		}

		// For accurate calculations, a thin slab is used, s.t. those atoms outside are the majority. Check this first, but account for the size of the atom, as it may reside in more than one slab.
//		if (iz <= slabIDX_end[iSlab]  && iz >= slabIDX_start[iSlab])
		if (iz <= z_top && iz >= z_low)
		{

			if (DO_BEAM_TILT)
			{

//				wxPrintf("\nStarting vals %3.3e %3.3e %3.3e\n",dx+ix,dy+iy,dz+iz);
				x1 = current_specimen->my_atoms.Item(current_atom).x_coordinate;
				y1 = current_specimen->my_atoms.Item(current_atom).y_coordinate;
				z1 = rotated_oZ 			    + (current_specimen->my_atoms.Item(current_atom).z_coordinate / this->wanted_pixel_size) - slabIDX_start[iSlab];

				// adjust Z to centered position in slab
			    z1 = (z1 * this->wanted_pixel_size) - slab_half_thickness_angstrom;

			    // now get the adjusted positions to project along the beam direction
			    beam_tilt_xform.RotateCoords(x1,y1,z1,x2,y2,z2);
			    // now apply the shifts to the atom coords
			    dx = modff((dx+ix)+(x2-x1)/this->wanted_pixel_size,&ix);
			    dy = modff((dy+iy)+(y2-y1)/this->wanted_pixel_size,&iy);
			    dz = modff((dz+iz)+(z2-z1)/this->wanted_pixel_size,&iz);
//				wxPrintf("Ending vals %3.3e %3.3e %3.3e\n",dx+ix,dy+iy,dz+iz);


			}

		for (sx = -size_neighborhood; sx <= size_neighborhood ; sx++)
		{
			indX = ix + sx;
			R.x1 = (sx - pixel_offset - dx) * this->wanted_pixel_size;
			R.x2 = (R.x1 + this->wanted_pixel_size);

			for (sy = -size_neighborhood; sy <= size_neighborhood; sy++)
			{
				indY = iy + sy ;
				R.y1 = (sy - pixel_offset - dy) * this->wanted_pixel_size;
				R.y2 = (R.y1 + this->wanted_pixel_size);

				for (sz = -size_neighborhood; sz <= size_neighborhood ; sz++)
				{
					indZ = iz  + sz;
					R.z1 = (sz - pixel_offset - dz) * this->wanted_pixel_size;
					R.z2 = (R.z1 + this->wanted_pixel_size);
					// Put Z condition first since it should fail most often (does c++ fall out?)
					if (indZ <= slabIDX_end[iSlab]  && indZ >= slabIDX_start[iSlab] && indX > 0 && indY > 0 && indX < current_specimen->vol_nX && indY < current_specimen->vol_nY)
					{
						// Calculate the scattering potential

						// The case of the central voxel is special
						if (sx == 0 && sy == 0 && sz == 0)
						{
							for (kLim = 0; kLim < 2; kLim++)
							{
								for (jLim = 0; jLim < 2; jLim++)
								{
									for (iLim = 0; iLim < 2; iLim++)
									{


										// Vector to lower left of given voxel, with centered spatial coords (pixel 0 is from -0.5 --> 0.5)
													xTop = R.x1 * (iLim);
													yTop = R.y1 * (jLim);
													zTop = R.z1 * (kLim);

													xLow = R.x2 *(1-iLim);
													yLow = R.y2 *(1-jLim);
													zLow = R.z2 *(1-kLim);

//													#pragma omp simd
													for (iGaussian = 0; iGaussian < 5; iGaussian++)
													{

														temp_potential += (SCATTERING_PARAMETERS_A[element_id][iGaussian] *
																   fabsf((erff(bPlusB[iGaussian]*xTop)-erff(bPlusB[iGaussian]*xLow)) *
																		(erff(bPlusB[iGaussian]*yTop)-erff(bPlusB[iGaussian]*yLow)) *
																		(erff(bPlusB[iGaussian]*zTop)-erff(bPlusB[iGaussian]*zLow))));

													} // loop over gaussian fits




									}
								}
							}


						}
						else
						{


							// General case
//							#pragma omp simd
							for (iGaussian = 0; iGaussian < 5; iGaussian++)
							{

								temp_potential += (SCATTERING_PARAMETERS_A[element_id][iGaussian] *
										   fabsf((erff(bPlusB[iGaussian]*R.x2)-erff(bPlusB[iGaussian]*R.x1)) *
												 (erff(bPlusB[iGaussian]*R.y2)-erff(bPlusB[iGaussian]*R.y1)) *
												 (erff(bPlusB[iGaussian]*R.z2)-erff(bPlusB[iGaussian]*R.z1))));


							} // loop over gaussian fits




						}



//						for (iGaussian = 1; iGaussian < 5; iGaussian++)
//						{
//							temp_potential[0] += temp_potential[iGaussian];
//							temp_potential[iGaussian] = 0;
//						}
//						temp_potential[0] *= this->lead_term;
						temp_potential *= this->lead_term;


							atoms_added_idx[n_atoms_added] = scattering_slab->ReturnReal1DAddressFromPhysicalCoord(indX,indY,indZ - slabIDX_start[iSlab]);
							atoms_values_tmp[n_atoms_added] = temp_potential;
							temp_potential = 0.0f;

//							scattering_slab->real_values[atoms_added_idx[n_atoms_added]] += temp_potential;
							n_atoms_added++;


					}

				}
			}
		} // end of loop over the neighborhood



			for (int iIDX = 0; iIDX < n_atoms_added-1; iIDX++)
			{
				#pragma omp atomic update
				scattering_slab->real_values[atoms_added_idx[iIDX]] += (atoms_values_tmp[iIDX]);
				// This is the value for 100 KeV --> scale (if needed) the final projected density
				inelastic_slab->real_values[atoms_added_idx[iIDX]]  += element_inelastic_ratio * atoms_values_tmp[iIDX];
			}

		}// if statment into neigh

	} // end loop over atoms





}

void ScatteringPotentialApp::calc_water_potential(Image *projected_water)

// The if conditions needed to have water and protein in the same function
// make it too complicated and about 10x less parallel friendly.
{

	long current_atom;
	float bFactor;
	float radius;
	float water_lead_term;


	// Private variables:
	int element_id;
	if (DO_PHASE_PLATE)
	{
		element_id = 1; // Carbon
		bFactor = 0.25 * (CARBON_BFACTOR + this->min_bFactor); // FIXME the min bfactor is just to make testing convenient - prob should remove it.
		water_lead_term = this->lead_term;
	}
	else
	{
		element_id = SOLVENT_TYPE; // use oxygen as proxy for water
		bFactor = 0.25 * (SOLVENT_BFACTOR + this->min_bFactor);
		water_lead_term = this->lead_term;
		if (element_id == 3)
		{
			water_lead_term *= water_oxygen_ratio;
		}
	}

	wxPrintf("\nbFactor and leadterm %f %f\n",bFactor,water_lead_term);

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
	float  pixel_offset2 = 0.5f;






	for (iGaussian = 0; iGaussian < 5; iGaussian++)
	{
		bPlusB[iGaussian] = 2*PI/sqrtf(bFactor+SCATTERING_PARAMETERS_B[element_id][iGaussian]);
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
				double temp_potential = 0;
				double temp_potential_sum = 0;

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


							// Calculate the scattering potential
							temp_potential = 0.0f;


							// The case of the central voxel is special
							if (sx == 0 && sy == 0 && sz == 0)
							{
								for (kLim = 0; kLim < 2; kLim++)
								{
									for (jLim = 0; jLim < 2; jLim++)
									{
										for (iLim = 0; iLim < 2; iLim++)
										{
											// Vector to lower left of given voxel, with centered spatial coords (pixel 0 is from -0.5 --> 0.5)
											xTop = R.x1 * (iLim);
											yTop = R.y1 * (jLim);
											zTop = R.z1 * (kLim);

											xLow = R.x2 *(1-iLim);
											yLow = R.y2 *(1-jLim);
											zLow = R.z2 *(1-kLim);

											for (iGaussian = 0; iGaussian < 5; iGaussian++)
											{

												temp_potential += (SCATTERING_PARAMETERS_A[element_id][iGaussian] *
														   fabsf((erff(bPlusB[iGaussian]*xTop)-erff(bPlusB[iGaussian]*xLow)) *
																(erff(bPlusB[iGaussian]*yTop)-erff(bPlusB[iGaussian]*yLow)) *
																(erff(bPlusB[iGaussian]*zTop)-erff(bPlusB[iGaussian]*zLow))));



											} // loop over gaussian fits

										}
									}
								}


							}
							else
							{


								// General case
								for (iGaussian = 0; iGaussian < 5; iGaussian++)
								{

									temp_potential += (SCATTERING_PARAMETERS_A[element_id][iGaussian] *
											   fabsf((erff(bPlusB[iGaussian]*R.x2)-erff(bPlusB[iGaussian]*R.x1)) *
													 (erff(bPlusB[iGaussian]*R.y2)-erff(bPlusB[iGaussian]*R.y1)) *
													 (erff(bPlusB[iGaussian]*R.z2)-erff(bPlusB[iGaussian]*R.z1))));


								} // loop over gaussian fits


							}
//	//							temp_potential *= water_lead_term;

							temp_potential_sum += (double)(temp_potential*water_lead_term);

//							wxPrintf("nSubPix, ir,dr,cr, %d  %d, %d, %d  %3.3f,%3.3f,%3.3f  %3.3f,%3.3f,%3.3f\n ",nSubPixCenter,iSubPixX,iSubPixY,iSubPixZ,sx,sy,sz,dx,dy,dz);

					} // end of loop over Z

					projected_water[nSubPixCenter].real_values[projected_water[nSubPixCenter].ReturnReal1DAddressFromPhysicalCoord(sx+size_neighborhood_water,sy+size_neighborhood_water,0)] = (float)temp_potential_sum;


			} // end of loop Y
		} // end loop X Neighborhood




		nSubPixCenter++;

			} // inner SubPixZ
		}  // mid SubPiX
	} // outer SubPixY


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

void ScatteringPotentialApp::fill_water_potential(const PDB * current_specimen,Image *scattering_slab, Image *scattering_potential,
												  Image *inelastic_potential, Water *water_box, RotationMatrix rotate_waters,
													   float rotated_oZ, int *slabIDX_start, int *slabIDX_end, int iSlab)

// The if conditions needed to have water and protein in the same function
// make it too complicated and about 10x less parallel friendly.
{

	long current_atom;
	long nWatersAdded = 0;

	float radius;


	float bPlusB[5];
	float ix(0), iy(0), iz(0);
	float dx(0), dy(0), dz (0);
	int indX(0), indY(0), indZ(0);
	int sx(0), sy(0);
	int iSubPixX;
	int iSubPixY;
	int iSubPixZ;
	int iSubPixLinearIndex;
	float avg_cutoff;
	float current_weight = 0.0;
	float current_potential = 0.0;
	int iPot;

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


	// Change previous projected_water to projected_water_atoms for later confusion prevention TODO
	Image projected_water_atoms;
	projected_water_atoms.Allocate(scattering_slab->logical_x_dimension,scattering_slab->logical_y_dimension,1);
	projected_water_atoms.SetToConstant(0.0);

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

						current_weight = this-> water_weight[iPot];
						break;
					}
				}
			}

			water_mask.real_values[iVoxel] *= current_weight;
		}

	}
	// TODO experiment with the scheduling. Until the specimen is consistently full, many consecutive slabs may have very little work for the assigned threads to handle.
// With this new insertion approach, threading only slows the overall process. where on small images, 1,4,40 threads are about, 1,1.5,2x and large (6kx6k) 1,2,3x
	long n_waters_ignored = 0;
//	wxPrintf("\n\n%d %d %d\n\n",iSlab, slabIDX_start[iSlab] ,  slabIDX_end[iSlab] );
	#pragma omp parallel for num_threads(this->number_of_threads) private(radius,ix,iy,iz,dx,dy,dz,indX,indY,indZ,sx,sy,iSubPixX,iSubPixY,iSubPixLinearIndex,n_waters_ignored)
	for (long current_atom = 0; current_atom < water_box->number_of_waters; current_atom++)
	{


		float temp_potential = 0;
		double temp_potential_sum = 0;
		double norm_value = 0;

		// TODO put other water manipulation methods inside the water class.
		water_box->ReturnCenteredCoordinates(current_atom,dx,dy,dz);

		// Rotate to the slab frame
		rotate_waters.RotateCoords(dx, dy, dz, ix, iy, iz);
		// Shift back to lower left origin, now with the slab oZ origin, need to use the dimensions of the scattering slab not the water box here
//		dx = modff(ix + ((float)water_box->vol_oX), &ix);
//		dy = modff(iy + ((float)water_box->vol_oY), &iy);
		dx = modff(ix + scattering_slab->logical_x_dimension/2, &ix);
		dy = modff(iy + scattering_slab->logical_y_dimension/2, &iy);
		dz = modff(iz + rotated_oZ, &iz); // Why am I subtracting here? Should it be an add? TODO



		float water_edge = (float)(SUB_PIXEL_NEIGHBORHOOD*2) + 1.0f;
		iSubPixX = trunc((dx-0.5f) * water_edge) + SUB_PIXEL_NEIGHBORHOOD;
		iSubPixY = trunc((dy-0.5f) * water_edge) + SUB_PIXEL_NEIGHBORHOOD;
		iSubPixZ = trunc((dz-0.5f) * water_edge) + SUB_PIXEL_NEIGHBORHOOD;

		iSubPixLinearIndex = int(water_edge * water_edge * iSubPixZ) + (water_edge * iSubPixY) + iSubPixX;



//		iSubPixLinearIndex = iSubPixY*(float)(SUB_PIXEL_NEIGHBORHOOD*2+1) + iSubPixX;




//		if ( iz >= slabIDX_start[iSlab] && iz  <= slabIDX_end[iSlab] && myroundint(ix)-1 > 0 && myroundint(iy)-1 > 0 && myroundint(ix)-1 < scattering_slab->logical_x_dimension && myroundint(iy)-1 < scattering_slab->logical_y_dimension &&
//			 current_potential < avg_cutoff)
		if ( iz >= slabIDX_start[iSlab] && iz  <= slabIDX_end[iSlab] && myroundint(ix)-1 > 0 && myroundint(iy)-1 > 0 && myroundint(ix)-1 < scattering_slab->logical_x_dimension && myroundint(iy)-1 < scattering_slab->logical_y_dimension)
		{


			current_potential = scattering_slab->ReturnRealPixelFromPhysicalCoord(myroundint(ix)-1,myroundint(iy)-1,myroundint(iz) - slabIDX_start[iSlab]);

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



			if (DO_PRINT)
			{
				#pragma omp atomic update
				nWatersAdded++;
			}
			int iWater = 0;

			for (sy = 0; sy <  upper_bound ; sy++ )
			{
				indY = myroundint(iy) - upper_bound + sy + size_neighborhood_water + 1;
				for (sx = 0;  sx < upper_bound ; sx++ )
				{
					indX = myroundint(ix) -upper_bound + sx + size_neighborhood_water + 1;
					// Even with the periodic boundaries checked in shake, the rotation may place waters out of bounds. TODO this is true for non-waters as well.
					if (indX >= 0 && indX < projected_water_atoms.logical_x_dimension && indY >= 0 && indY < projected_water_atoms.logical_y_dimension)
					{
						#pragma omp atomic update
						projected_water_atoms.real_values[projected_water_atoms.ReturnReal1DAddressFromPhysicalCoord(indX,indY,0)] +=
								(current_weight*this->projected_water[iSubPixLinearIndex].real_values[this->projected_water[iSubPixLinearIndex].ReturnReal1DAddressFromPhysicalCoord(sx,sy,0)]); // TODO could I land out of bounds?] += projected_water_atoms[iSubPixLinearIndex].real_values[iWater];
//						wxPrintf("Current Water %3.3e\n",current_weight*this->projected_water[iSubPixLinearIndex].real_values[this->projected_water[iSubPixLinearIndex].ReturnReal1DAddressFromPhysicalCoord(sx,sy,0)]);

					}
					else
					{

						continue;

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

		tmpPrj[0].Allocate(scattering_potential[iSlab].logical_x_dimension,scattering_potential[iSlab].logical_y_dimension,1);
		tmpPrj[0].SetToConstant(0.0f);
		this->project(&water_mask,tmpPrj,0);
		tmpPrj->DivideByConstant((float)water_mask.logical_z_dimension);
		float mean_water_value = projected_water_atoms.ReturnAverageOfRealValues(-1.0,false);// / water_mask.logical_z_dimension;
		projected_water_atoms.CopyFrom(tmpPrj);
		projected_water_atoms.MultiplyByConstant(mean_water_value);
		tmpPrj[0].QuickAndDirtyWriteSlice("checkWater.mrc",1,true);
		projected_water_atoms.QuickAndDirtyWriteSlice("checkPrjWater.mrc",1,true);

		scattering_potential[iSlab].QuickAndDirtyWriteSlice("atoms.mrc",1,true);
		scattering_potential[iSlab].AddImage(&projected_water_atoms);
		scattering_potential[iSlab].QuickAndDirtyWriteSlice("atoms_water.mrc",1,true);


		delete [] tmpPrj;
//		exit(-1);

	}
	else
	{
		float oxygen_inelastic_to_elastic_ratio = ( 20.2f / ATOMIC_NUMBER[SOLVENT_TYPE] );
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






//  delete 	 [] threaded_water;

} // End of fill water func
void ScatteringPotentialApp::project(Image *image_to_project, Image *image_to_project_into,  int iSlab)
{

	/* Image.AddSlices accumulates in float. Maybe just add an option there to add in double.
	 *
	 */
	// Project the slab into the two

	double pixel_accumulator;
	int prjX, prjY, prjZ;
	int edgeX = 0;
	int edgeY = 0;
	long slab_address;
	// TODO add check that these are the same size.

	// Tracking total number added
	number_of_pixels_averaged.AddConstant(image_to_project->logical_z_dimension);


	for (prjX = 0; prjX < image_to_project->logical_x_dimension; prjX++)
	{

		for (prjY = 0; prjY < image_to_project->logical_y_dimension; prjY++)
		{
			slab_address = image_to_project_into[iSlab].ReturnReal1DAddressFromPhysicalCoord(prjX,prjY,0);
			pixel_accumulator  = 0;
			for (prjZ = 0; prjZ < image_to_project->logical_z_dimension; prjZ++)
			{

				pixel_accumulator +=  image_to_project->ReturnRealPixelFromPhysicalCoord(prjX,prjY,prjZ);
				if (image_to_project->ReturnRealPixelFromPhysicalCoord(prjX,prjY,prjZ) > 1e-6)
				{
					number_of_non_waters_averaged.real_values[slab_address] += 1;
				}


			}
			 image_to_project_into[iSlab].real_values[slab_address] += (float)pixel_accumulator;

		}
	}

}

void ScatteringPotentialApp::taper_edges(Image *image_to_taper,  int iSlab, bool inelastic_img)
{
	// Taper edges to the mean TODO see if this can be removed
	// Update with the current mean. Why am I saving this? Is it just for SOLVENT ==1 ? Then probably can kill TODO
	int prjX, prjY, prjZ;
	int edgeX = 0;
	int edgeY = 0;
	long slab_address;
	float taper_val;

	if (inelastic_img)
	{
		this->inelastic_mean[iSlab] =  image_to_taper[iSlab].ReturnAverageOfRealValues(0.0);

	}
	else
	{
		this->image_mean[iSlab] =  image_to_taper[iSlab].ReturnAverageOfRealValues(0.0);

	}
	if (DO_PRINT) {wxPrintf("%d image mean for taper %f\n",iSlab, this->image_mean[iSlab]);}

	for (prjX = 0  ; prjX < image_to_taper[iSlab].logical_x_dimension ; prjX++)
	{
		if (prjX < TAPERWIDTH) edgeX = prjX;
		else edgeX = image_to_taper[iSlab].logical_x_dimension - prjX - 1 ;

		for (prjY = 0 ; prjY < image_to_taper[iSlab].logical_y_dimension  ; prjY++)
		{
			if (prjY < TAPERWIDTH) edgeY = prjY  ;
			else edgeY = image_to_taper[iSlab].logical_y_dimension  - prjY -1 ;

			slab_address = image_to_taper[iSlab].ReturnReal1DAddressFromPhysicalCoord(prjX,prjY,0);

			// Taper the edges
			if (edgeX < TAPERWIDTH && edgeX <= edgeY)
			{
				taper_val = TAPER[edgeX];
			}
			else if (edgeY < TAPERWIDTH && edgeY <= edgeX)
			{
				taper_val = TAPER[edgeY];
			}
			else
			{
				taper_val = 1;
			}

			if (taper_val != 1)
			{
				image_to_taper[iSlab].real_values[slab_address] *= taper_val;
				if (inelastic_img)
				{
					image_to_taper[iSlab].real_values[slab_address] += this->inelastic_mean[iSlab]*(1-taper_val);

				}
				else
				{
					image_to_taper[iSlab].real_values[slab_address] += this->image_mean[iSlab]*(1-taper_val);

				}
			}


		}
	}

}

void ScatteringPotentialApp::apply_sqrt_DQE_or_NTF(Image *image_in, int iTilt_IDX, bool do_root_DQE)
{


	image_in[iTilt_IDX].ForwardFFT(true);
	float x_coord_sq, y_coord_sq, spatial_frequency;
	float weight;
	long pixel_counter = 0;

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

			spatial_frequency = sqrtf(x_coord_sq + y_coord_sq);

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

	image_in[iTilt_IDX].BackwardFFT();



}

void ScatteringPotentialApp::normalize_set_dose_expectation(Image *sum_image, int iTilt_IDX, float current_thickness)
{

	float expected_electrons;

	expected_electrons = this->dose_per_frame;

	float wavelength_scaling;
	if (this->kV < 150)
	{
		wavelength_scaling = -1.5f;
	}
	else
	if (this->kV < 250)
	{
		wavelength_scaling = -1.25f;
	}
	else
	{
		wavelength_scaling = -1.0f;
	}

    float avg_radius; // If there are strong unique features in the corners, this might prove to be a problem. Could
    float loss_due_to_thickness;
    avg_radius = std::min((sum_image[iTilt_IDX].logical_x_dimension-IMAGEPADDING)/2.0f,(sum_image[iTilt_IDX].logical_y_dimension-IMAGEPADDING)/2.0f);
    if (DO_PRINT) {wxPrintf("wanted radius is %f, and size is %d %d\n",avg_radius,sum_image[iTilt_IDX].logical_x_dimension,sum_image[iTilt_IDX].logical_y_dimension);}
    // For positive control: assuming dose fractionation theorem, a "perfect" 3d should match a 2d



	//if (DO_PRINT)
	{wxPrintf("\nWith a rotated thickness of %3.4e Angstrom calculated a probability of NOT scattering inelastically of %3.3f, total scaling %3.3f\n",current_thickness,loss_due_to_thickness,
			loss_due_to_thickness * this->wanted_pixel_size_sq * expected_electrons / sum_image[iTilt_IDX].ReturnAverageOfRealValues(0.95*avg_radius));}

	sum_image[iTilt_IDX].MultiplyByConstant(loss_due_to_thickness * this->wanted_pixel_size_sq * expected_electrons / sum_image[iTilt_IDX].ReturnAverageOfRealValues(0.95*avg_radius));






}

void ScatteringPotentialApp::calc_average_intensity_at_solvent_cutoff(float average_bfactor)
{


	int iCutoff;
	int n_atoms;
	float bFactor = 0;
	float bPlusB = 0;
	float element_vect[6] = {1,2,3,4,15};
	float water_total = 0;

	// Shift the curves to the right as the values from Shang/Sigworth are distance to VDW radius (avg C/O/N/H = 1.48 A)
	const float PUSH_BACK_BY = -1.48;

	if (DO_PHASE_PLATE)
	{
		n_atoms = 2;
	}
	else
	{
		if (SOLVENT_TYPE == 15)
		{
			// TODO determine if you want to include the water line in the average.
			n_atoms = 3;
		}
		else
		{
			n_atoms = 3;
		}

	}

	for (int iGaussian = 0; iGaussian < 5; iGaussian++)
	{
		water_total += SCATTERING_PARAMETERS_A[SOLVENT_TYPE][iGaussian];
	}
	// Total potential times the average number of waters/voxel FIXME water/angCubed is calculated in water class.
//	water_total *= this->lead_term * (0.0314 * this->wanted_pixel_size * this->wanted_pixel_size_sq);
	water_total *= 8.0 * this->lead_term * ( 0.0314 * this->wanted_pixel_size * this->wanted_pixel_size_sq);

	// 0 is used for individual water molecules
	// 1,2,and 2 are the range over which the bulk solvent is tapered for the 3d ground truth calculation
	for (iCutoff = 0; iCutoff < N_WATER_TERMS ; iCutoff++)
	{

		int number_at_cutoff = 0;
		float radius = iCutoff * (4.0 / (float)N_WATER_TERMS);
		float radius_sq = radius * radius;


		if (this->min_bFactor != -1)
		{
//			bFactor = 0.25 * (this->min_bFactor + average_bfactor*this->bFactor_scaling) ;
			bFactor = 0.25 * (this->min_bFactor + average_bfactor) ;
		}
		else
		{
			bFactor = 0.25 * (average_bfactor + MIN_BFACTOR); // If using physical displacement, just use a zero bfactor
		}


		// TODO add an explicit weigting by element type actually included in a given PDB
		// If you change to include hydrogen, the phase plate option needs to be considered (here 1 is carbon)

		for (int iElement  = 0; iElement  < n_atoms; iElement++)
		{
			int element_id = element_vect[iElement];
			double temp_potential = 0;
			for (int iGaussian = 0; iGaussian < 5; iGaussian++)
				// Vector to lower left of given voxel

				{
	//				bPlusB = 2*PI/sqrt(bFactor+SCATTERING_PARAMETERS_B[element_id][iGaussian]);
	//				temp_potential += (SCATTERING_PARAMETERS_A[element_id][iGaussian] *
	//								  powf((erff(bPlusB*xTop)-erff(bPlusB*xLow)),3));
				bPlusB = bFactor+SCATTERING_PARAMETERS_B[element_id][iGaussian];


				// This is a 1D curve so the normalization term from integrating the gaussians should be 1/2 not 3/2
				temp_potential += (SCATTERING_PARAMETERS_A[element_id][iGaussian] *
								   powf(4*PI/bPlusB,1/2) *
								   expf(-4*PI*PI*radius_sq/bPlusB));



				} // loop over gaussian fits


			// Since this is not the integrated form of the potential, the factor of 1/8 must be cancelled from the "lead term"
			this->average_at_cutoff[iCutoff] += (this->lead_term*temp_potential* 8.0f);


			number_at_cutoff++;


		}

		this->average_at_cutoff[iCutoff] /= number_at_cutoff;


		this->water_weight[iCutoff] =  0.5 + 0.5 *    std::erff(      (radius + PUSH_BACK_BY)-(HYDRATION_VALS[2]+xtra_shift*this->wanted_pixel_size) / (sqrtf(2)*HYDRATION_VALS[5])) +
									   HYDRATION_VALS[0] * expf(-powf((radius + PUSH_BACK_BY)-(HYDRATION_VALS[3]+xtra_shift*this->wanted_pixel_size),2)/(2*powf(HYDRATION_VALS[6],2))) +
									   HYDRATION_VALS[1] * expf(-powf((radius + PUSH_BACK_BY)-(HYDRATION_VALS[4]+xtra_shift*this->wanted_pixel_size),2)/(2*powf(HYDRATION_VALS[7],2)));

		if (add_mean_water_potential && this->do3d)
		{
			// This will be used for a reference, so rather than the weight, we want the weighted MIP
			// Since this is not the integrated form of the potential, the factor of 1/8 must be cancelled from the "lead term"
			this->water_weight[iCutoff] *=  water_total; // MIP for water
		}

		if (DO_PRINT) {wxPrintf("Average potential/ water weight at cutoff %d is %3.3e | %3.3e\n",iCutoff,this->average_at_cutoff[iCutoff],this->water_weight[iCutoff]);}
//		 {wxPrintf("Average potential/ water weight at cutoff %d is %3.3e | %3.3e\n",iCutoff,this->average_at_cutoff[iCutoff],this->water_weight[iCutoff]);}

	}



}


float ScatteringPotentialApp::return_bfactor_given_dose(float relative_bfactor)
{
	float bFactor;
	// The most simple model. Maybe as a function of exposure squared? who knows
//	bFactor     =  0.25 * ((std::max(0.0f,this->current_total_exposure-5)/2 + logf(this->current_total_exposure+1 + std::max(0.0f,this->current_total_exposure-5)))* this->bFactor_scaling + (this->min_bFactor ));
	float termA = 0.245;
	if (this->min_bFactor != -1)
	{
		if (DO_EXPOSURE_FILTER == 2 || DO_EXPOSURE_FILTER == 3)
		{
			bFactor = 0.25*this->min_bFactor;
		}
		else
		{
	//		bFactor     =  0.25 * (std::max(0.0,(termA*powf(this->current_total_exposure, 1.665) - sqrt(this->current_total_exposure) )* this->bFactor_scaling) + this->min_bFactor);
	//		bFactor     =  0.25 * ((std::max(0.0f,this->current_total_exposure-2) + std::max(0.0f,this->current_total_exposure-12))*this->bFactor_scaling + this->min_bFactor);
	//		bFactor     =  0.25 * ((this->current_total_exposure)*this->bFactor_scaling + this->min_bFactor);
			bFactor     =  0.25 * (10*this->current_total_exposure + powf(this->current_total_exposure, 1.685) + 15);

		}
	}
	else
	{
		if  (DO_EXPOSURE_FILTER == 2 || DO_EXPOSURE_FILTER == 3)
		{
			bFactor = 0.25*this->min_bFactor;
		}
		else
		{
		bFactor     =  0.25 * (std::max(0.0f,(termA*powf(this->current_total_exposure, 1.665) - sqrtf(this->current_total_exposure) )* this->bFactor_scaling) + MIN_BFACTOR);
		}
	}


	return bFactor;
}


