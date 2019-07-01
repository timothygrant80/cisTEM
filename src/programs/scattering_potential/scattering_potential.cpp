#include "../../core/core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!

#define NARGSEXPECTED 4 // number of arguments passed to the app.
#define MAX_NUMBER_PDBS 10
typedef float WANTED_PRECISION;

const float WAVELENGTH = pow(1.968697e-2,1); // Angstrom
const float SOLVENT_CUTOFF[4] = {2.8,0.5,2.0,3.3};//powf(3.2,2); // Angstrom from a non-water scattering center (squared radius)
const float SOLVENT_BFACTOR = 0; // times dose 8pi^2*msd(Ang) (defined) and sqrt(dose) ~ rmsd(Ang) , note no sqrt in the def for bfactor. latter is from Henderson, thon rings amorphous ice
//const float  SOLVENT_RATIO = 0.85 ; // ratio to convert total cross-section of oxygen --> water Ianik Plante1,2 and Francis A Cucinotta1 2009const int SUB_PIXEL_NeL = (SUB_PIXEL_NEIGHBORHOOD*2+1)*(SUB_PIXEL_NEIGHBORHOOD*2+1);
const int SOLVENT_TYPE = 3; // 2 N, 3 )
const int SUB_PIXEL_NEIGHBORHOOD = 11;
const int SUB_PIXEL_NeL = (SUB_PIXEL_NEIGHBORHOOD*2+1)*(SUB_PIXEL_NEIGHBORHOOD*2+1);
const float  MEAN_FREE_PATH = 4000;// Angstrom, newer paper by bridgett (2018) and a couple older TODO add ref. Use to reduce total probability


const int TAPERWIDTH = 29; // TODO this should be set to 12 with zeros padded out by size neighborhood and calculated by taper = 0.5+0.5.*cos((((1:pixelFallOff)).*pi)./(length((1:pixelFallOff+1))));
const float TAPER[29] = {0,0,0,0,0,
						 0.003943, 0.015708, 0.035112, 0.061847, 0.095492, 0.135516, 0.181288, 0.232087,
						 0.287110, 0.345492, 0.406309, 0.468605, 0.531395, 0.593691, 0.654508, 0.712890,
						 0.767913, 0.818712, 0.864484, 0.904508, 0.938153, 0.964888, 0.984292, 0.996057};

const int IMAGEPADDING = 128; // Padding applied for the Fourier operations in the propagation steps
const int IMAGETRIMVAL = IMAGEPADDING + 2*TAPERWIDTH;

const float N_FRAMES = 12;
const float N_SUB_FRAMES = 1; // I think I'm convinced the sum of poisson dist with params a, b, c is (a+b+c)
const double DOSE_PER_FRAME = 15/N_FRAMES;//1.184;

const float CALC_DIST_WATER = 1.5;//3.0; // With bfactor 0, then apply SOLVENT_BFACTOR to all water at the same time.
const float CALC_DIST_OTHER = 2.0;//5.0; // Now using actual displacements on the atoms // With bfactor 60 this should be ok
const float OFF_BY_LINEAR_FACTOR = 8.0; // my formula includes a 1/8 in front of the integral. Removing this gives a mean phase shift of
										// pi/2 rad for 31.5 nm of carbon atoms at 2.0 g/cm^3 the value NIST gives for amorphous carbon.
										// The 31.5 nm comes from Rados measurements. Those measurements of course include scattering from the bonds too.

// CONTROL FOR DEBUGGING AND VALIDATION
const bool do_complexCTF = true;
const bool do_realCTF = false;

// Control for trouble shooting, propagation. Might be nice to have something similar on a global context
const bool DO_PARALLEL = true;
const bool DO_SINC_BLUR = false;
const bool DO_PRINT = false;
//
const bool DO_SOLVENT = true; // 0 none, 2 add 2d layer, 3 add in 3d rand gaussian no correlation, ?
const bool CALC_WATER_NO_HOLE = false;
const bool CALC_HOLES_ONLY = false;


#define DEBUG_NAN true
#define DEBUG_MSG false
#define DEBUG_POISSON false // Skip the poisson draw

#define SAVE_PERFECT_REFERENCE false // Save the projected potential for each tilt + ctf
#define SAVE_3D_SLAB false
#define SAVE_WATER_AND_OTHER false
#define SAVE_PROJECTED_WATER false
#define SAVE_PHASE_GRATING false
#define SAVE_PHASE_GRATING_DOSE_FILTERED false
#define SAVE_PHASE_GRATING_PLUS_WATER false
#define SAVE_PROBABILITY_WAVE false
#define SAVE_TO_COMPARE_JPR false
#define JPR_SIZE 350
#define SAVE_WITH_DQE false
#define SAVE_WITH_NORMALIZED_DOSE false
#define SAVE_POISSON_PRE_NTF false
#define SAVE_POISSON_WITH_NTF false
#define DO_EXPOSURE_FILTER true // gave seg fault TODO
#define DO_COMPEX_AMPLITUDE_TERM true
#define DO_APPLY_DQE false // This is broken - curves are ~ correct for 1 apix, but are fit using sin/cos. TODO
#define DO_NORMALIZE_SET_DOSE true
#define DO_APPLY_NTF false
#define DO_FLIP_PROPAGATOR true

// Some of the more common elements, should add to this later. These are from Peng et al. 1996.
// The name is to an index matching here in the PDB class. If you change this, you MUST change that. This is probably a bad idea.
// H(0),C(1),N(2),O(3),F(4),Na(5),Mg(6),P(7),S(8),Cl(9),K(10),Ca(11),Mn(12),Fe(13),Zn(14)scattering_potential.cpp.20160630_preOpt

const WANTED_PRECISION   SCATTERING_PARAMETERS_A[15][5] = {
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
};

// 12.5664 ~ (4*pi)
// -39.47841760685077 ~ -4*pi^2
const WANTED_PRECISION SCATTERING_PARAMETERS_B[15][5] = {
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

};



class ScatteringPotentialApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();
	wxString 	pdb_file_names[MAX_NUMBER_PDBS];
	std::string output_filename;
	int 	 	number_of_pdbs;
	float	    defocus;
	//int 	 	particle_copy_number[MAX_NUMBER_PDBS];
	float 	 	particle_origin[MAX_NUMBER_PDBS][3];
	float 	 	particle_eulers[MAX_NUMBER_PDBS][3];
	long 	 	number_of_non_water_atoms; // could this overflow??
	float 	 	*image_mean;
	float		current_total_exposure = 0;
	float    	wanted_pixel_size;
	float		wanted_pixel_size_sq;
	int 	 	size_neighborhood;
	int 		size_neighborhood_water;
	bool 	 	tilt_series;
	long 		particle_stack;
	int 	 	number_of_threads;
    bool 	 	do3d;
    double   	average_at_cutoff[4] = {0,0,0,0};
	int padded_x_dim;
	int padded_y_dim;

    float bFactor_scaling;
    float min_bFactor;

    FrealignParameterFile  parameter_file;
    float parameter_vect[17] = {0};



    float water_scaling;

	long n_waters_added = 0;
	Image *projected_water; // waters calculated over the specified subpixel shifts and projected.


	Image *correlation_check;


	void probability_density_2d(PDB *pdb_ensemble, int time_step);
	// Note the water does not take the dose as an argument.
	void  calc_scattering_potential(const PDB * current_specimen,Image *scattering_slab,  RotationMatrix rotate_waters,
			                        float rotated_oZ, int *slabIDX_start, int *slabIDX_end, int iSlab);

	void  calc_water_potential(Image *projected_water);
	void  fill_water_potential(const PDB * current_specimen,Image *scattering_slab, Image *scattering_potential, Water *water_box,RotationMatrix rotate_waters,
														   float rotated_oZ, int *slabIDX_start, int *slabIDX_end, int iSlab);

	void  fill_water_potential_flat(Image *potential3D);


	void  project(Image *image_to_project, Image *image_to_project_into,  int iSlab);
	void  taper_edges(Image *image_to_taper,  int iSlab);
	void  apply_sqrt_DQE_or_NTF(Image *image_in, int iTilt_IDX, const float a0, const float a1, const float a2, const float w);
	void  normalize_set_dose_expectation(Image *sum_image, int iTilt_IDX, float current_thickness);

	void calc_average_intensity_at_solvent_cutoff(float average_bfactor);




	// Profiling
	wxDateTime  timer_start;
	wxDateTime	overall_start;
	wxTimeSpan	span_seed;
	wxTimeSpan	span_atoms;
	wxTimeSpan  span_waters;
	wxTimeSpan  span_shake;
	wxTimeSpan	span_propagate;
	wxDateTime 	overall_finish;


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
	 wxPrintf("%s",this->output_filename);
	// this->do3d = my_input->GetYesNoFromUser("do3d?","just potential if true","no");
	 this->bFactor_scaling = my_input->GetFloatFromUser("Linear scaling of per atom bFactor","0 off, 1 use as is","1",0,10000);
	 this->min_bFactor     = my_input->GetFloatFromUser("Per atom bFactor added to all atoms","0 off, 1 use as is","5",-1,10000);

	 this->water_scaling = my_input->GetFloatFromUser("Linear scaling water intensity","0 off, 1 use as is","1",0,1);
 	 this->number_of_threads = my_input->GetIntFromUser("Number of threads", "Max is number of tilts", "1", 1);
	 this->tilt_series = my_input->GetYesNoFromUser("Create a tilt-series as well?","Should make 0 degree with full dose, then tilt","no");
	 if ( this->tilt_series == true )
	 {
		 // not doing anything for now, fixed range and increment.test_multi.mrc
	 }
	 if (this->tilt_series == false)
	 {
		 this->particle_stack = (long)my_input->GetFloatFromUser("Create a particle stack?","Number of particles at random orientations, 0 for just an image","1",0,1e7);
		 wxPrintf("Making a particle stack with %ld images\n",this->particle_stack);
	 }
	 while (number_of_pdbs < MAX_NUMBER_PDBS && add_more_pdbs)
	 {
		 pdb_file_names[number_of_pdbs-1] = my_input->GetFilenameFromUser("PDB file name", "an input PDB", "my_atom_coords.pdb", true );
		 // This is now coming directly from the PDB
		 //particle_copy_number[number_of_pdbs-1] = my_input->GetIntFromUser("Copy number of this particle", "To be inserted into the ensemble", "1", 1);




		 add_more_pdbs = my_input->GetYesNoFromUser("Add another type of particle?", "Add another pdb to create additional features in the ensemble", "yes");



		 if (add_more_pdbs) {number_of_pdbs++;}
	 }




	 this->wanted_pixel_size 		= my_input->GetFloatFromUser("Output pixel size (Angstroms)","Output size for the final projections","1.0",0.01,2.0);
	 this->defocus                  = my_input->GetFloatFromUser("wanted defocus (Angstroms)","Out","700",0,120000);
	 this->wanted_pixel_size_sq 	= powf(this->wanted_pixel_size,2);



	if (particle_stack > 0 || this->tilt_series)
	{
		std::string parameter_file_name = output_filename + ".par";
		wxString parameter_header = "C           PSI   THETA     PHI       SHX       SHY     MAG  INCLUDE   DF1      DF2  ANGAST  PSHIFT     OCC      LogP      SIGMA   SCORE  CHANGE";
		this->parameter_file.Open(parameter_file_name,1,17);
		this->parameter_file.WriteCommentLine(parameter_header);

	}

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

	PDB  *pdb_ensemble = new PDB[number_of_pdbs] ;

	// For Tim and Peter
	if (number_of_pdbs == 1) {wxPrintf("There is %d pdb\n",number_of_pdbs);}
	else { wxPrintf("There are %d pdbs\n",number_of_pdbs);}

	// Initialize each of the PDB objects, this reads in and centers each PDB, but does not make any copies (instances) of the trajectories.

	for (iPDB = 0; iPDB < number_of_pdbs ; iPDB++)
	{
		pdb_ensemble[iPDB] = PDB(pdb_file_names[iPDB],access_type_read,records_per_line);
	}



    // Get a count of the total non water atoms
	for (iPDB = 0; iPDB < number_of_pdbs; iPDB++)
	{
		wxPrintf("THIS PDB %d\n",iPDB);
		//this->number_of_non_water_atoms += (pdb_ensemble[iPDB].number_of_atoms * this->particle_copy_number[iPDB]);
		this->number_of_non_water_atoms += (pdb_ensemble[iPDB].number_of_atoms * pdb_ensemble[iPDB].number_of_particles_initialized);
		wxPrintf("%ld %d\n",pdb_ensemble[iPDB].number_of_atoms , pdb_ensemble[iPDB].number_of_particles_initialized);
		// These sizes will need to be determined by the min and max dimensions of the base shifted ensemble and removed from user input TODO
	}

	wxPrintf("\nThere are %ld non-water atoms in the specimen.\n",this->number_of_non_water_atoms);
	wxPrintf("\nCurrent number of PDBs %d\n",number_of_pdbs);


	// Set-up the ensemble
//	this->set_initial_trajectories(pdb_ensemble);

    this->size_neighborhood 	  = myroundint(ceilf(powf(CALC_DIST_OTHER,0.5)/ this->wanted_pixel_size));
    this->size_neighborhood_water = myroundint(ceilf(powf(CALC_DIST_WATER,0.5)/ this->wanted_pixel_size));

    if ( DO_SOLVENT )
    {

    	projected_water= new Image[SUB_PIXEL_NeL];

        for (int iWater = 0 ; iWater < SUB_PIXEL_NeL; iWater++)
        {
            projected_water[iWater].Allocate(this->size_neighborhood_water*2+1,this->size_neighborhood_water*2+1,true);
            projected_water[iWater].SetToConstant(0.0);
        }

		wxPrintf("Starting projected water calc with sizeN %d, %d\n",this->size_neighborhood_water*2+1,this->size_neighborhood_water*2+1);
		this->calc_water_potential(projected_water);
		wxPrintf("Finishing projected water calc\n");
    }


	int time_step = 0 ;
    if (this->do3d == true)
    {
    	//TODO add option to generate and save 3d using the 2d function
    }
    else
    {
    	this->probability_density_2d(pdb_ensemble, time_step);
    }


	wxPrintf("\nFinished pre seg fault");

	overall_finish = wxDateTime::Now();


	wxPrintf("Timings: Overall: %s\n",(overall_finish-overall_start).Format());
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

	// CTF parameters:  There should be an associated variablility with tilt angle TODO
	float wanted_acceleration_voltage = 300; // keV
	float wanted_spherical_aberration = 2.7; // mm
	float wanted_amplitude_contrast = 0.07;
	float wanted_defocus_1_in_angstroms = this->defocus; // A
	float wanted_defocus_2_in_angstroms = this->defocus; //A
	float wanted_astigmatism_azimuth = 0.0; // degrees
	float astigmatism_angle_randomizer = 0.0; //
	float percent_stretch = 0.1;
	float defocus_randomizer = 0.0;
	float wanted_additional_phase_shift_in_radians  = 0.0;// rad
    float defocus_tolerance =  20; // in Angstrom, allowed thickness per slab
    float propagator_distance = 0; // in Angstom, <= defocus tolerance.




	float  use_error = 1.0;
	// To add error to the global alignment
	float tilt_axis = 10; // degrees from Y-axis
	float in_plane_sigma = 2; // spread in-plane angles based on neighbors
	float tilt_angle_sigma = 0.1; //;
	float magnification_sigma = 0.0001;//;

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




    // TODO fix periodicity in Z on slab





    int iTilt;
    int nTilts;
    float max_tilt;
    float * tilt_psi;
    float * tilt_theta;
    float * tilt_phi;

	// TODO either put into the class or better just update the global_random to use this.
	std::default_random_engine generator;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> uniform_dist(0.000000001, 1.0);


    //Calculate 3d atomic potentials
    wxPrintf("using neighboorhood of %2.2f vox^3 for waters and %2.2f vox^3 for non-waters\n",powf(this->size_neighborhood_water*2+1,3),powf(this->size_neighborhood*2+1,3));

    if ( this-> tilt_series )
    {
    	max_tilt  = 60.0;
    	float tilt_range = max_tilt;
    	float tilt_inc = 3.0;
    	nTilts = ceil(tilt_range/tilt_inc)*2 +1;
    	tilt_psi   = new float[nTilts];
    	tilt_theta = new float[nTilts];
    	tilt_phi   = new float[nTilts];
    	shift_x    = new float[nTilts];
    	shift_y    = new float[nTilts];
    	shift_z    = new float[nTilts];
    	mag_diff   = new float[nTilts];




    	std::normal_distribution<float>  norm_dist_inplane(0.0,in_plane_sigma);
    	std::normal_distribution<float>  norm_dist_tiltangle(0.0,tilt_angle_sigma);

    	for (iTilt=0; iTilt < nTilts; iTilt++)
    	{


    		tilt_psi[iTilt] = tilt_axis + use_error * norm_dist_inplane(gen); // *(2*PI);
    		tilt_theta[iTilt] = -((tilt_range - iTilt*tilt_inc) + use_error * norm_dist_tiltangle(gen));
    		tilt_phi[iTilt] = 0;
    		shift_x[iTilt] = use_error * 8*uniform_dist(gen) ;
    		shift_y[iTilt] = use_error * 8*uniform_dist(gen);
    		shift_z[iTilt] = 0;
    		mag_diff[iTilt] = 0;


    	}

    }
    else if ( this->particle_stack > 0)
    {
    	max_tilt = 0.0;
    	nTilts = this->particle_stack;
    	tilt_psi   = new float[nTilts];
    	tilt_theta = new float[nTilts];
    	tilt_phi   = new float[nTilts];
    	shift_x    = new float[nTilts];
    	shift_y    = new float[nTilts];
    	shift_z    = new float[nTilts];
    	mag_diff   = new float[nTilts];

    	std::normal_distribution<float> normal_dist(0.0,1.0);

    	for (iTilt=0; iTilt < nTilts; iTilt++)
    	{
    		tilt_psi[iTilt] = uniform_dist(gen)*360.0f; // *(2*PI);
    		tilt_theta[iTilt] = std::acos(2*uniform_dist(gen)-1) * 180.0f/(float)PI;
    		tilt_phi[iTilt] = -1*tilt_psi[iTilt] + uniform_dist(gen)*360.0f; //*(2*PI);

    		shift_x[iTilt]  = use_error * normal_dist(gen) ;
    		shift_y[iTilt]  = use_error * normal_dist(gen);
    		shift_z[iTilt]  = 0;
    		mag_diff[iTilt] = 0;

    	}
    }
    else
    {
    	max_tilt = 0.0;
    	wxPrintf("\n\nAM I HERE\n");
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
    	tilt_psi[0]=0; tilt_theta[0]=0;tilt_phi[0]=0;shift_x[0]=0;shift_y[0]=0;shift_z[0]=0;mag_diff[0]=0;
    }



	// Not sure it makes sense to put this here.
    // Could I save memory by waiting and going slice by slice?
	Image *sum_image;
	Image *particle_stack;
	RotationMatrix particle_rot;
	Image *reference_stack;

	if (SAVE_PERFECT_REFERENCE)
	{
		reference_stack = new Image[nTilts];
	}

	if (this->particle_stack > 0)
	{
		// Output will be a stack of particles (not frames)
		sum_image = new Image[(int)(N_FRAMES)];
		particle_stack = new Image[nTilts];
	}
	else
	{
		sum_image = new Image[(int)(nTilts*N_FRAMES)];

	}

	// We only want one water box for a tilt series. For a particle stack, re-initialize for each particle.
	Water water_box;
	// Create a new PDB object that represents the current state of the specimen, with each local motion applied.
	PDB current_specimen(this->number_of_non_water_atoms);



    for ( iTilt = 0 ; iTilt < nTilts ; iTilt++)
    {

		RotationMatrix rotate_waters;
		if (this->tilt_series)
		{
//			rotate_waters.SetToRotation(-tilt_phi[iTilt],-tilt_theta[iTilt],-tilt_psi[iTilt]);
			rotate_waters.SetToEulerRotation(tilt_psi[iTilt],tilt_theta[iTilt],tilt_phi[iTilt]);

		}
		else
		{
			rotate_waters.SetToRotation(euler1,euler2,euler3);

		}
		if (this->particle_stack > 0)
		{
			float phiOUT = 0;
			float psiOUT = 0;
			float thetaOUT = 0;

			wxPrintf("\n\nWorking on iParticle %d/ %d\n\n",iTilt,nTilts);

//			particle_rot.SetToEulerRotation(tilt_phi[iTilt],tilt_theta[iTilt],tilt_psi[iTilt]);
			particle_rot.SetToEulerRotation(-tilt_psi[iTilt],-tilt_theta[iTilt],-tilt_phi[iTilt]);


		    // For particle stack, use the fixed supplied defocus, and apply a fixed amount of astigmatism at random angle to make sure everything is filled in

			    defocus_randomizer = uniform_dist(gen)*percent_stretch;
		    	wxPrintf("For the particle stack, stretching the defocus by %3.2f percent and randmozing the astigmatism angle -90,90",100*defocus_randomizer);
		    	wanted_defocus_1_in_angstroms = this->defocus*(1+defocus_randomizer); // A
		    	wanted_defocus_2_in_angstroms = this->defocus*(1-defocus_randomizer); //A
				wanted_astigmatism_azimuth = (uniform_dist(gen)-0.5)*179.99;

		}
		else
		{
			particle_rot.SetToIdentity();

		}

	for ( int iFrame = 0; iFrame < N_FRAMES; iFrame ++)
	{


    	long iTilt_IDX;
    	if (this->particle_stack > 0)
		{
    		iTilt_IDX = iFrame;
		}
    	else
    	{
        	iTilt_IDX = (long)((iTilt*N_FRAMES)+iFrame);
    	}

    	int slab_nZ;
    	int rotated_Z; // full range in Z to cover the rotated specimen
    	float rotated_oZ;
    	float slab_oZ;
	    int nSlabs;
		int nS;
		double full_tilt_radians = 0;

		// Exposure filter TODO add check that it is wanted
		float *dose_filter;
		ElectronDose my_electron_dose(wanted_acceleration_voltage, this->wanted_pixel_size);


		Image jpr_sum_phase;
		Image jpr_sum_detector;
		if (SAVE_TO_COMPARE_JPR)
		{
			jpr_sum_phase.Allocate(JPR_SIZE,JPR_SIZE,true);
			jpr_sum_phase.SetToConstant(0.0);

			jpr_sum_detector.Allocate(JPR_SIZE,JPR_SIZE,true);
			jpr_sum_detector.SetToConstant(0.0);

		}


		current_specimen.TransformLocalAndCombine(pdb_ensemble,this->number_of_pdbs,this->number_of_non_water_atoms,this->wanted_pixel_size, time_step, particle_rot);

	    // Use this value to determine if a water is too close to a non-water atom
	    this->calc_average_intensity_at_solvent_cutoff(current_specimen.average_bFactor);

//		Water water_box( &current_specimen,this->size_neighborhood_water, this->wanted_pixel_size, DOSE_PER_FRAME, max_tilt);
	    if (iTilt == 0 && iFrame == 0)
	    {
		    water_box.Init( &current_specimen,this->size_neighborhood_water, this->wanted_pixel_size, DOSE_PER_FRAME, max_tilt);

	    }
		// Compute the solvent fraction, with ratio of protein/ water density.
		// Assuming an average 2.2Ang vanderwaal radius ~50 cubic ang, 33.33 waters / cubic nanometer.

		if ( DO_SOLVENT  && water_box.number_of_waters == 0 )
		{
			// Waters are member variables of the scatteringPotential app - currentSpecimen is passed for size information.
			this->timer_start = wxDateTime::Now();


			if (DO_PRINT) {wxPrintf("n_waters added %ld\n", water_box.number_of_waters);}

			water_box.SeedWaters3d();

//			water_seed_3d(&current_specimen);

			this->span_seed += wxDateTime::Now()-this->timer_start;

			if (DO_PRINT) {wxPrintf("Timings: seed_waters: %s\n",(this->span_seed).Format());}


			this->timer_start = wxDateTime::Now();

			water_box.ShakeWaters3d(this->number_of_threads);
//			water_shake_3d();

			this->span_shake += wxDateTime::Now()-this->timer_start;



		}
		else if ( DO_SOLVENT )
		{

			this->timer_start = wxDateTime::Now();

			water_box.ShakeWaters3d(this->number_of_threads);

//			water_shake_3d();

			this->span_shake += wxDateTime::Now()-this->timer_start;
		}


		padded_x_dim = ReturnClosestFactorizedUpper(IMAGEPADDING+current_specimen.vol_nX,7,true);
		padded_y_dim = ReturnClosestFactorizedUpper(IMAGEPADDING+current_specimen.vol_nY,7,true);




		// TODO with new solvent add, the edges should not need to be tapered or padded
		sum_image[iTilt_IDX].Allocate(padded_x_dim,padded_y_dim,true);
		sum_image[iTilt_IDX].SetToConstant(0.0);
		if (SAVE_PERFECT_REFERENCE && iFrame == 0)
		{
			// Only save the reference for the first frame from each tilt.
			reference_stack[iTilt].Allocate(padded_x_dim,padded_y_dim,true);
			reference_stack[iTilt].SetToConstant(0.0);
		}


		// Apply acurrent_specimen.vol_nY global shifts and rotations
		float tilt_specimen = 0; // TODO Set this at the outset.
		current_specimen.TransformGlobalAndSortOnZ(number_of_non_water_atoms, shift_x[iTilt], shift_y[iTilt], shift_z[iTilt], rotate_waters);
		if (this->tilt_series)
		{
			full_tilt_radians = PI/180*(tilt_theta[iTilt]);
		}
		else
		{
			full_tilt_radians = PI/180*(euler2);
		}
		if (DO_PRINT) {wxPrintf("tilt angle in radians/deg %2.2e/%2.2e iFrame %d/%f\n",full_tilt_radians,tilt_theta[iTilt],iFrame,N_FRAMES);}




		rotated_Z =  myroundint((float)current_specimen.vol_nZ+((double)water_box.vol_nX) * fabs(std::sin(full_tilt_radians)));
		if (DO_PRINT) {wxPrintf("%f %f %f\n",((double)current_specimen.vol_nX),((double)water_box.vol_nX)/2,fabs(std::sin(full_tilt_radians)));}


		//rotated_oZ = ceilf((rotated_Z+1)/2);
		if (DO_PRINT) {wxPrintf("\nflat thicknes, %ld and rotated_Z %d\n", current_specimen.vol_nZ, rotated_Z);}
		wxPrintf("\nWorking on iTilt %d at %f degrees for frame %d\n",iTilt,tilt_theta[iTilt],iFrame);

		//  TODO Should separate the mimimal slab thickness, which is a smaller to preserve memory from the minimal prop distance (ie. project sub regions of a slab)
		if (defocus_tolerance < 0 ) {nSlabs  = 1;}  else { nSlabs = ceilf( (float)rotated_Z * this->wanted_pixel_size/ defocus_tolerance);}
		if (DEBUG_MSG) {wxPrintf("Calc N Slabs %d nSlabs %d rotZ %f \n",nSlabs,rotated_Z,this->wanted_pixel_size/ defocus_tolerance);}
		Image *scattering_potential = new Image[nSlabs];
		Image *reference_potential;
		if (SAVE_PERFECT_REFERENCE && iFrame == 0)
		{
			reference_potential = new Image[nSlabs];
		}
		if (DEBUG_MSG) {wxPrintf("Declare scattering potential\n");}
		nS = floor(rotated_Z / nSlabs);
		if (DEBUG_MSG) {wxPrintf("Calc NS\n");}
		propagator_distance = (float)nS * this->wanted_pixel_size;

		if (DO_PRINT) {wxPrintf("Propagator distance is %f Angstroms\n",propagator_distance);}


//		slabIDX_start = new int[nSlabs];
//		slabIDX_end   = new int[nSlabs];
		int slabIDX_start[nSlabs];
		int slabIDX_end[nSlabs];
		this->image_mean 	  = new float[nSlabs];

		// Set up slabs, padded by neighborhood for working
		for (iSlab = 0; iSlab < nSlabs; iSlab++)
		{
			slabIDX_start[iSlab] = iSlab*nS;
			slabIDX_end[iSlab]   = (iSlab+1)*nS - 1;
			if (iSlab < nSlabs - 1) wxPrintf("%d %d\n",slabIDX_start[iSlab],slabIDX_end[iSlab]);
		}
		// The last slab may be a bit bigger, so make sure you don't miss acurrent_specimen.vol_nYthing.
		slabIDX_end[nSlabs-1] = rotated_Z - 1;
		if (DO_PRINT) {wxPrintf("%d %d\n",slabIDX_start[nSlabs-1],slabIDX_end[nSlabs-1]);}

		Image Potential_3d;

		for (iSlab = 0; iSlab < nSlabs; iSlab++)
		{


			scattering_potential[iSlab].Allocate(current_specimen.vol_nX, current_specimen.vol_nY,1);
			scattering_potential[iSlab].SetToConstant(0.0);


			slab_nZ = slabIDX_end[iSlab] - slabIDX_start[iSlab] + 1;// + 2*this->size_neighborhood;
			slab_oZ = floorf(slab_nZ/2); // origin in the volume containing the rotated slab
			rotated_oZ = floorf(rotated_Z/2);
			// Because we will project along Z, we could put Z on the rows
			if (DO_PRINT) {wxPrintf("iSlab %d %d %d\n",iSlab, slabIDX_start[iSlab],slabIDX_end[iSlab] );}
			if (DO_PRINT) {wxPrintf("slab_oZ %f slab_nZ %d rotated_oZ %f\n",slab_oZ,slab_nZ,rotated_oZ);}
			Image scattering_slab;
			scattering_slab.Allocate(current_specimen.vol_nX,current_specimen.vol_nY,slab_nZ);
			scattering_slab.SetToConstant(0.0);




			this->timer_start = wxDateTime::Now();

			this->calc_scattering_potential(&current_specimen, &scattering_slab,rotate_waters, rotated_oZ, slabIDX_start, slabIDX_end, iSlab);

			this->span_atoms += (wxDateTime::Now()-this->timer_start);

			if (DO_PRINT) {wxPrintf("Span: calc_atoms: %s\n",(span_atoms).Format());}



			////////////////////
			if (SAVE_3D_SLAB)
			{

				if (iFrame > 0)
				{
					wxPrintf("Only one frame needed for 3d calc.\n");
					throw;
				}
				if (iSlab == 0)
				{
					Potential_3d.Allocate(current_specimen.vol_nX,current_specimen.vol_nY,slabIDX_end[nSlabs-1] - slabIDX_start[0] + 1);
					Potential_3d.SetToConstant(0.0);
				}
				 //for trouble shooting, save each 3d slab
                int offset_slab = scattering_potential->physical_address_of_box_center_z - Potential_3d.physical_address_of_box_center_z + slabIDX_start[iSlab];
				wxPrintf("Inserting slab %d at position %d\n",iSlab,offset_slab);
				Potential_3d.InsertOtherImageAtSpecifiedPosition(&scattering_slab,0,0,offset_slab);

				if (iSlab == nSlabs - 1)
				{

					this->fill_water_potential_flat(&Potential_3d);

					std::string fileNameOUT = "tmpSlab" + std::to_string(iSlab) + ".mrc";
					MRCFile mrc_out(this->output_filename,true);
					wxPrintf("Writing out your 3d slices %d --> %d\n",1,slabIDX_end[nSlabs-1] - slabIDX_start[0] + 1);
					Potential_3d.WriteSlices(&mrc_out,1,slabIDX_end[nSlabs-1] - slabIDX_start[0] + 1);
					mrc_out.SetPixelSize(this->wanted_pixel_size);
					mrc_out.CloseFile();
				}
				continue;

			}
			////////////////////


			if ( CALC_HOLES_ONLY == false && CALC_WATER_NO_HOLE == false )
			{
				this->project(&scattering_slab,scattering_potential,iSlab);
				if (SAVE_PERFECT_REFERENCE && iFrame == 0)
				{
					reference_potential[iSlab].Allocate(current_specimen.vol_nX, current_specimen.vol_nY,1);
					reference_potential[iSlab].CopyFrom(&scattering_potential[iSlab]);
				}
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

			if (SAVE_TO_COMPARE_JPR)
			{
				// For comparing to JPR @ 0.965
				Image binImage;
				binImage.CopyFrom(&scattering_potential[iSlab]);

				binImage.Resize(JPR_SIZE,JPR_SIZE,1);
				jpr_sum_phase.AddImage(&binImage);
//				std::string fileNameOUT = "combareJPR_phaseGrating_" + this->output_filename;
//				MRCFile mrc_out(fileNameOUT,true);
//				binImage.WriteSlices(&mrc_out,1,1);
//				mrc_out.SetPixelSize(0.965);
//				mrc_out.CloseFile();
			}

			if (DO_EXPOSURE_FILTER && CALC_HOLES_ONLY == false && CALC_WATER_NO_HOLE == false)
			{
			// add in the exposure filter
//				std::string fileNameOUT = "withOUT_DoseFilter_phaseGrating_" + std::to_string(iSlab) + this->output_filename;
//				scattering_potential[iSlab].QuickAndDirtyWriteSlice(fileNameOUT,1);
				scattering_potential[iSlab].ForwardFFT(true);
				dose_filter = new float[scattering_potential[0].real_memory_allocated / 2];
				for (long pixel_counter = 0; pixel_counter < scattering_potential[0].real_memory_allocated / 2; pixel_counter++)
				{
					dose_filter[pixel_counter] = 0.0;
				}
				// TODO switch cummulative dose to be calc over begining and end of the frames.
				my_electron_dose.CalculateDoseFilterAs1DArray(&scattering_potential[iSlab], dose_filter, this->current_total_exposure, this->current_total_exposure+DOSE_PER_FRAME );


				for (long pixel_counter = 0; pixel_counter < scattering_potential[0].real_memory_allocated / 2; pixel_counter++)
				{
					scattering_potential[iSlab].complex_values[pixel_counter] =
							scattering_potential[iSlab].complex_values[pixel_counter] *
							(dose_filter[pixel_counter]);// + (1-dose_filter[pixel_counter]) *  exp(I * PI * global_random_number_generator.GetUniformRandom())) ;

				}
				scattering_potential[iSlab].BackwardFFT();
				delete [] dose_filter;
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

			if ( DO_SOLVENT )
			{


				this->timer_start = wxDateTime::Now();

				// Now loop back over adding waters where appropriate
				if (DO_PRINT) {wxPrintf("Working on waters, slab %d\n",iSlab);}


				this->fill_water_potential(&current_specimen,&scattering_slab,scattering_potential,&water_box,rotate_waters,
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





			this->timer_start = wxDateTime::Now();


			// Now apply the CTF - check that slab_oZ is doing what you intend it to TODO
			float defocus_offset = ((slabIDX_end[iSlab]-slabIDX_start[iSlab])/2 - rotated_oZ + slabIDX_start[iSlab] + 1) * this->wanted_pixel_size;

			if (DO_PRINT) {wxPrintf("using defocus offset %f for slab %d/%d, %d %d %f\n",defocus_offset,iSlab,nSlabs,slabIDX_end[iSlab],slabIDX_start[iSlab],rotated_oZ);}
			CTF ctf_value(wanted_acceleration_voltage,
						  wanted_spherical_aberration,
						  wanted_amplitude_contrast,
						  wanted_defocus_1_in_angstroms + defocus_offset,
						  wanted_defocus_2_in_angstroms + defocus_offset,
						  wanted_astigmatism_azimuth,
						  this->wanted_pixel_size,
						  wanted_additional_phase_shift_in_radians);



			if (do_realCTF == true)
			{
				Image padded_image;
				padded_image.Allocate(IMAGEPADDING+current_specimen.vol_nX,IMAGEPADDING+current_specimen.vol_nY,1); // Set at top and whatev.
				scattering_potential[iSlab].ClipInto(&padded_image,this->image_mean[iSlab]);
				padded_image.ForwardFFT(true);

				padded_image.ApplyCTF(ctf_value,false);
				padded_image.BackwardFFT();
				scattering_potential[iSlab] = padded_image;
			}



		} // end loop nSlabs


		this->current_total_exposure += DOSE_PER_FRAME; // increment the dose

		if (DO_PRINT) {wxPrintf("\n\t%ld out of bounds of %ld = percent\n\n", nOutOfBounds,number_of_non_water_atoms);}

//		#pragma omp parallel num_threads(4)
		// TODO make propagtor class
		{
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

		int nLoops;
		if (do_complexCTF == true) // this should be do_multi_slice

			if (SAVE_PERFECT_REFERENCE && iFrame == 0)
			{
				nLoops = 2;
			}
			else
			{
				nLoops = 1;

			}
		for (int iRefLoop = 0; iRefLoop < nLoops ; iRefLoop++)
		{
		{
			int iPar;
			int iSeq;

			Image *temp_img = new Image[4];
			Image *t_N = new Image[4];
			Image *wave_function = new Image[2];
			Image *phase_grating = new Image[2];
			CTF *ctf = new CTF[2];
			CTF *propagator = new CTF[2];



			// Values to use in parallel sections
			const float set_wave_func[2] = {1.0,0.0};
			const int	copy_from_1[4] 	 = {0,1,0,1};
			const int	copy_from_2[4]	 = {0,0,1,1};
			const int  	mult_by[4]		 = {0,1,1,0};
			const int 	prop_apply_real[4] = {0,0,1,1};
			const int 	prop_apply_imag[4] = {1,1,0,0};
			const int  	ctf_apply[4]	   = {0,1,1,0};


			// get values for just sin or cos of the ctf by manipulating the amplitude term
			ctf[0].Init(wanted_acceleration_voltage,
						  wanted_spherical_aberration,
						  1.0,
						  wanted_defocus_1_in_angstroms,
						  wanted_defocus_2_in_angstroms,
						  wanted_astigmatism_azimuth,
						  this->wanted_pixel_size,
						  wanted_additional_phase_shift_in_radians);
			ctf[1].Init(wanted_acceleration_voltage,
						  wanted_spherical_aberration,
						  0.0,
						  wanted_defocus_1_in_angstroms,
						  wanted_defocus_2_in_angstroms,
						  wanted_astigmatism_azimuth,
						  this->wanted_pixel_size,
						  wanted_additional_phase_shift_in_radians);

			if (DO_PRINT) {wxPrintf("Propdist %f\n",propagator_distance);}
			// get values for just sin or cos of the ctf by manipulating the amplitude term
			// For the fresnel prop, set Cs = 0 and def = dz
			propagator[0].Init(wanted_acceleration_voltage,
							   0.0,
							   1.0,
							   propagator_distance,
							   propagator_distance,
							   0.0,
							   this->wanted_pixel_size,
							   0.0);
			propagator[1].Init(wanted_acceleration_voltage,
							   0.0,
							   0.0,
							   propagator_distance,
							   propagator_distance,
							   0.0,
							   this->wanted_pixel_size,
							   0.0);





			#pragma omp parallel for num_threads(propagate_threads_2) if (DO_PARALLEL)
			for (iPar = 0; iPar < 2; iPar++)
			{
				wave_function[iPar].Allocate(padded_x_dim,padded_y_dim,1);
				wave_function[iPar].SetToConstant(set_wave_func[iPar]);
				phase_grating[iPar].Allocate(padded_x_dim,padded_y_dim,1);
			}

			#pragma omp parallel for num_threads(propagate_threads_4) if (DO_PARALLEL)
			for (iPar = 0; iPar < 4; iPar++)
			{
				t_N[iPar].Allocate(padded_x_dim,padded_y_dim,1);
				if (DO_PRINT) {wxPrintf("Allocating t_N %d\n",iPar);}
				t_N[iPar].SetToConstant(0.0);
				temp_img[iPar].Allocate(padded_x_dim,padded_y_dim,1);
			}




			for (iSlab = 0; iSlab < nSlabs; iSlab++)
			{

				if (SAVE_PERFECT_REFERENCE && iFrame == 0 && iRefLoop > 0)
				{
					wxPrintf("iFrame %d and iRefLoop %d", iFrame, iRefLoop);
	  				scattering_potential[iSlab].CopyFrom(&reference_potential[iSlab]);
				}

				// Taper here?
				this->taper_edges(scattering_potential, iSlab);
				this->image_mean[iSlab] = scattering_potential[iSlab].ReturnAverageOfRealValues(0.0);

//				for (long iPixel = 0; iPixel < scattering_potential[iSlab].real_memory_allocated; iPixel++)
//				{
//					if (scattering_potential[iSlab].real_values[iPixel] == 0)
//					{
//						scattering_potential[iSlab].real_values[iPixel] = this->image_mean[iSlab];
//					}
//				}
				// Convert the potential to the arguments of the complex phase grating (exp(-1i*interactionConst*ProjectedPotential))
				phase_grating[0].SetToConstant(0.0);
				phase_grating[1].SetToConstant(0.0);
				scattering_potential[iSlab].ClipInto(&phase_grating[0],this->image_mean[iSlab]);
				scattering_potential[iSlab].ClipInto(&phase_grating[1],this->image_mean[iSlab]);

				if (DEBUG_NAN == true)  MyDebugAssertFalse(scattering_potential[iSlab].HasNan(),"There is a NAN 1");
				if (DEBUG_NAN == true)  MyDebugAssertFalse(phase_grating[0].HasNan(),"There is a NAN 2");
				if (DEBUG_NAN == true)  MyDebugAssertFalse(phase_grating[1].HasNan(),"There is a NAN 3");



				if (DO_COMPEX_AMPLITUDE_TERM)
				{
					#pragma omp simd
					for ( long iPixel = 0; iPixel < phase_grating[0].real_memory_allocated; iPixel++)
					{
						phase_grating[0].real_values[iPixel] = exp(-1*wanted_amplitude_contrast*phase_grating[0].real_values[iPixel]) * std::cos(phase_grating[0].real_values[iPixel]);
					}

					#pragma omp simd
					for ( long iPixel = 0; iPixel < phase_grating[1].real_memory_allocated; iPixel++)
					{
						phase_grating[1].real_values[iPixel] = exp(-1*wanted_amplitude_contrast*phase_grating[1].real_values[iPixel]) * std::sin(phase_grating[1].real_values[iPixel]);
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


				if (DEBUG_NAN == true)  MyDebugAssertFalse(phase_grating[0].HasNan(),"There is a NAN 2a");
				if (DEBUG_NAN == true)  MyDebugAssertFalse(phase_grating[1].HasNan(),"There is a NAN 3a");

				#pragma omp parallel for num_threads(propagate_threads_4) if (DO_PARALLEL)
				for (iPar = 0; iPar < 4; iPar++)
				{
					t_N[iPar].CopyFrom(&wave_function[copy_from_1[iPar]]);
					t_N[iPar].MultiplyPixelWise(phase_grating[mult_by[iPar]]);

					if (DEBUG_NAN == true)  MyDebugAssertFalse(t_N[iPar].HasNan(),"There is a NAN t11");
					t_N[iPar].ForwardFFT(true);
					if (DEBUG_NAN == true)  MyDebugAssertFalse(t_N[iPar].HasNan(),"There is a NAN t11F");

				}

				// Reset the wave function to zero to store the update results
				wave_function[0].SetToConstant(0.0);
				wave_function[1].SetToConstant(0.0);

				#pragma omp parallel for num_threads(propagate_threads_4) if (DO_PARALLEL)
				for (iPar = 0; iPar < 4; iPar++)
				{


					// Get the real part of the new exit wave
					temp_img[iPar].CopyFrom(&t_N[iPar]);
					if (DEBUG_NAN == true)  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1");


					temp_img[iPar].ApplyCTF(propagator[prop_apply_real[iPar]],false);
					if (DEBUG_NAN == true)  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1CTF");
					temp_img[iPar].BackwardFFT();
					if (DEBUG_NAN == true)  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1BFFT");

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


				#pragma omp parallel for num_threads(propagate_threads_4) if (DO_PARALLEL)
				for (iPar = 0; iPar < 4; iPar++)
				{


					// Get the real part of the new exit wave
					temp_img[iPar].CopyFrom(&t_N[iPar]);
					if (DEBUG_NAN == true)  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1");


					temp_img[iPar].ApplyCTF(propagator[prop_apply_imag[iPar]],false);
					if (DEBUG_NAN == true)  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1CTF");
					temp_img[iPar].BackwardFFT();
					if (DEBUG_NAN == true)  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1BFFT");

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



				}


				#pragma omp parallel for num_threads(propagate_threads_2) if (DO_PARALLEL)
				for (iPar = 0; iPar < 2; iPar++)
				{
					// Now we need to add the CTF (F(w_Real +i*w_Image)*(exp(-1i*X))
					wave_function[iPar].ForwardFFT(true);

				}

				#pragma omp parallel for num_threads(propagate_threads_4) if (DO_PARALLEL)
				for (iPar = 0; iPar < 4; iPar++)
				{

					// Re-use t_N[0] through t_N[3]
					t_N[iPar].CopyFrom(&wave_function[copy_from_2[iPar]]);
					t_N[iPar].ApplyCTF(ctf[ctf_apply[iPar]],false);
					t_N[iPar].BackwardFFT();

				}

				wave_function[0].SetToConstant(0.0);
				wave_function[1].SetToConstant(0.0);

				wave_function[0].AddImage(&t_N[0]);
				wave_function[0].SubtractImage(&t_N[2]);
				wave_function[1].AddImage(&t_N[1]);
				wave_function[1].AddImage(&t_N[3]);

				if (DEBUG_NAN == true) MyDebugAssertFalse(wave_function[0].HasNan(),"There is a NAN 6");
				if (DEBUG_NAN == true)  MyDebugAssertFalse(wave_function[1].HasNan(),"There is a NAN 7");


				// Now get the square modulus of the wavefunction
				if (iRefLoop < 1)
				{
					#pragma omp simd
					for (long iPixel = 0; iPixel < sum_image[iTilt_IDX].real_memory_allocated; iPixel++)
					{
						 sum_image[iTilt_IDX].real_values[iPixel] = (powf(wave_function[0].real_values[iPixel],2) + powf(wave_function[1].real_values[iPixel],2));
					}
				}
				else
				{
					#pragma omp simd
					for (long iPixel = 0; iPixel < sum_image[iTilt_IDX].real_memory_allocated; iPixel++)
					{
						 reference_stack[iTilt].real_values[iPixel] = (powf(wave_function[0].real_values[iPixel],2) + powf(wave_function[1].real_values[iPixel],2));
					}
				}

			if (SAVE_PROBABILITY_WAVE && iRefLoop < 1)
			{
				std::string fileNameOUT = "withProbabilityWave_" + std::to_string(iSlab) + this->output_filename;
					MRCFile mrc_out(fileNameOUT,true);
					sum_image[iTilt_IDX].WriteSlices(&mrc_out,1,1);
					mrc_out.SetPixelSize(this->wanted_pixel_size);
					mrc_out.CloseFile();

			}





			if (SAVE_TO_COMPARE_JPR && iRefLoop < 1)
			{
				// For comparing to JPR @ 0.965
				Image binImage;
				binImage.CopyFrom(&sum_image[iTilt_IDX]);

				binImage.Resize(JPR_SIZE,JPR_SIZE,1);
				jpr_sum_detector.AddImage(&binImage);

			}



			if (DO_APPLY_DQE)
			{
//				// Now apply Square root of the DQE fit with Fourier series, 1 term. Values now for 300kV @ 3e-/physical pixel*s TODO consider other rates, and actually convert physical pixel size etc.
//				// ONLY VALID UP TO PHYSICAL NYQUIST WHICH SHOULD BE SET TO 1.0
//				this->apply_sqrt_DQE_or_NTF(sum_image,  iTilt_IDX, 0.5305, 0.3549, -0.025555, 1.784);
//				if (SAVE_WITH_DQE)
//				{
//					std::string fileNameOUT = "withDQE_" + std::to_string(iSlab) + this->output_filename;
//						MRCFile mrc_out(fileNameOUT,true);
//						sum_image[iTilt_IDX].WriteSlices(&mrc_out,1,1);
//						mrc_out.SetPixelSize(this->wanted_pixel_size);
//						mrc_out.CloseFile();
//				}
			}



			if (DO_NORMALIZE_SET_DOSE )
			{
				///sum_image[iTilt_IDX].MultiplyByConstant(DOSE_PER_FRAME);
				this->normalize_set_dose_expectation(sum_image, iTilt_IDX, rotated_Z*this->wanted_pixel_size);
				if (SAVE_WITH_NORMALIZED_DOSE)
				{
					std::string fileNameOUT = "withNORMALIZED_DOSE_" + std::to_string(iSlab) + this->output_filename;
						MRCFile mrc_out(fileNameOUT,true);
						sum_image[iTilt_IDX].WriteSlices(&mrc_out,1,1);
						mrc_out.SetPixelSize(this->wanted_pixel_size);
						mrc_out.CloseFile();
				}
			}



			if (DEBUG_POISSON == false)
			{


				// Next we draw from a poisson distribution and then finally apply the NTF
				Image cpp_poisson;
				cpp_poisson.Allocate(sum_image[iTilt_IDX].logical_x_dimension,sum_image[iTilt_IDX].logical_y_dimension,1,true);
				cpp_poisson.SetToConstant(0.0);


				std::default_random_engine generator;

			    std::random_device rd;  //Will be used to obtain a seed for the random number engine
			    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
			    std::uniform_real_distribution<> dis(0.000000001, 1.0);


				for (int iSubFrame = 0; iSubFrame < N_SUB_FRAMES ; iSubFrame++)
				{
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
				}

				sum_image[iTilt_IDX].CopyFrom(&cpp_poisson);

				// Trouble shooting poisson vs ripples
//				cpp_poisson.QuickAndDirtyWriteSlice("banding_cppPoisson.mrc",1);
//				pre_poisson.QuickAndDirtyWriteSlice("banding_bahPoisson.mrc",1);

			}



			if (SAVE_POISSON_PRE_NTF && iRefLoop < 1)
			{

				std::string fileNameOUT = "withPoisson_noNTF_" + std::to_string(iSlab) + this->output_filename;
					MRCFile mrc_out(fileNameOUT,true);
					sum_image[iTilt_IDX].WriteSlices(&mrc_out,1,1);
					mrc_out.SetPixelSize(this->wanted_pixel_size);
					mrc_out.CloseFile();

			}

			// Ineffecient but very simple way to add a intra-frame motional blur.

			if (DO_SINC_BLUR && iRefLoop < 1 )
			{
				float sinc_shift_x = 1.0;
				float sinc_shift_y = 2.0;
				int nSegments = 5;
				Image tmpShift;
				Image tmpAccum;


				sum_image[iTilt_IDX].ForwardFFT(true);
				tmpAccum.CopyFrom(&sum_image[iTilt_IDX]);

				for (int iShift = 1 ; iShift < nSegments; iShift++)
				{
					tmpShift.CopyFrom(&sum_image[iTilt_IDX]);
					tmpShift.PhaseShift(sinc_shift_x/(float)nSegments*iShift,sinc_shift_y/(float)nSegments*iShift,0);
					tmpAccum.AddImage(&tmpShift);
				}

				tmpAccum.MultiplyByConstant(1/(float(nSegments)));
				sum_image[iTilt_IDX].CopyFrom(&tmpAccum);
				sum_image[iTilt_IDX].BackwardFFT();

			}

			if (DO_APPLY_NTF && DEBUG_POISSON == false && iRefLoop < 1)
			{
				// Now apply NTF fit with Fourier series, 1 term. Values now for 300kV @ 3e-/physical pixel*s TODO consider other rates, and actually convert physical pixel size etc.
				// ONLY VALID UP TO PHYSICAL NYQUIST WHICH SHOULD BE SET TO 1.0
				this->apply_sqrt_DQE_or_NTF(sum_image,  iTilt_IDX, 1.075, -0.01443, -0.01034,  4.369);

				std::string fileNameOUT = "with_DQE_BTF_poisson_" + std::to_string(iSlab) + this->output_filename;
				sum_image[iTilt_IDX].QuickAndDirtyWriteSlice("withDQE_NTF_poisson.mrc",1);

				// Now round as threshold for counting mode (since NTF spreads poisson counts a bit)
				for (long iPixel = 0; iPixel < sum_image[iTilt_IDX].real_memory_allocated; iPixel++ )
				{
					sum_image[iTilt_IDX].real_values[iPixel] = myround(sum_image[iTilt_IDX].real_values[iPixel]);
				}//

				if (SAVE_POISSON_WITH_NTF)
				{

					std::string fileNameOUT = "withPoisson_withNTF_" + std::to_string(iSlab) + this->output_filename;
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

		}
		} // End of loop to optionally propagate the perfect reference


		} // end fft push omp block



//	    delete [] slabIDX_start;
//	    delete [] slabIDX_end;
//	    if (SOLVENT != 0) delete [] this->image_mean;

	    this->span_propagate += wxDateTime::Now()-this->timer_start;
	    wxPrintf("before the destructor there are %ld non-water-atoms\n",this->number_of_non_water_atoms);
		if (SAVE_TO_COMPARE_JPR)
		{

		std::string fileNameOUT = "combareJPR_phaseGrating_" + this->output_filename;
		MRCFile mrc_out(fileNameOUT,true);
		jpr_sum_phase.WriteSlices(&mrc_out,1,1);
		mrc_out.SetPixelSize(this->wanted_pixel_size);
		mrc_out.CloseFile();

		std::string fileNameOUT2 = "combareJPR_detector_" + this->output_filename;
		MRCFile mrc_out2(fileNameOUT2,true);
		jpr_sum_detector.WriteSlices(&mrc_out2,1,1);
		mrc_out2.SetPixelSize(this->wanted_pixel_size);
		mrc_out2.CloseFile();

		}

		delete [] scattering_potential;
	    if (SAVE_PERFECT_REFERENCE && iFrame == 0 )
	    {
	    	delete [] reference_potential;
	    }
    } // end of loop over frames



		if (this->particle_stack > 0)
		{

			// Reset the water count so that a new water_box is initialized for the next particle
			water_box.number_of_waters = 0;
			// Reset the dose
			this->current_total_exposure = 0;


			parameter_vect[0] = iTilt + 1;
			parameter_vect[1] = tilt_psi[iTilt];
			parameter_vect[2] = tilt_theta[iTilt];
			parameter_vect[3] = tilt_phi[iTilt];
			if (use_error != 0)
			{
				parameter_vect[4]  = shift_x[iTilt]; // shx
				parameter_vect[5]  = shift_y[iTilt]; // shy
			}
			parameter_vect[8] = wanted_defocus_1_in_angstroms;
			parameter_vect[9] = wanted_defocus_2_in_angstroms;
			parameter_vect[10]= wanted_astigmatism_azimuth;

			parameter_file.WriteLine(parameter_vect, false);


			particle_stack[iTilt].Allocate(sum_image[0].logical_x_dimension,sum_image[0].logical_y_dimension,true);
			particle_stack[iTilt].CopyFrom(&sum_image[0]);

			// sum the frames
			for (int iFrame = 1; iFrame < N_FRAMES; iFrame++)
			{
				particle_stack[iTilt].AddImage(&sum_image[iFrame]);
			}

			// Normalize to 1 sigma, at a radius 2/3 the box size
			/////particle_stack[iTilt].ZeroFloatAndNormalize(1.0,0.33*particle_stack[0].logical_x_dimension,true);
		}
		else if (this->tilt_series)
		{
			parameter_vect[0] = iTilt + 1;
			parameter_vect[1] = tilt_psi[iTilt];
			parameter_vect[2] = tilt_theta[iTilt];
			parameter_vect[3] = tilt_phi[iTilt];
			if (use_error != 0)
			{
				parameter_vect[4]  = shift_x[iTilt]; // shx
				parameter_vect[5]  = shift_y[iTilt]; // shy
			}
		    parameter_vect[6]  = 0; // mag

		    parameter_file.WriteLine(parameter_vect, false);


		}



    } // end of loop over tilts


    if (DO_PRINT) {wxPrintf("%s\n",this->output_filename);}

	bool over_write = true;
	MRCFile mrc_out_final(this->output_filename,over_write);

	std::string fileNameREF = "ref_" + this->output_filename;
    MRCFile mrc_ref_final(fileNameREF,over_write);

    std::string fileNameTiltSum = "tiltSum_" + this->output_filename;
    MRCFile mrc_tlt_final(fileNameTiltSum,over_write);

	if (DO_PRINT) {wxPrintf("\n\nnTilts %d N_FRAMES %d\n\n",nTilts,myroundint(N_FRAMES));}

	if (this->particle_stack > 0)
	{


		// FIXME
		// Pick the smallest size and clip all to that
		int minX = 1e6;
		int minY = 1e6;
		int minSize;

		for (iTilt=0; iTilt < nTilts; iTilt++)
		{
			if (particle_stack[iTilt].logical_x_dimension - IMAGETRIMVAL  < minX) {minX = particle_stack[iTilt].logical_x_dimension - IMAGETRIMVAL;}
			if (particle_stack[iTilt].logical_y_dimension - IMAGETRIMVAL  < minY) {minY = particle_stack[iTilt].logical_y_dimension - IMAGETRIMVAL;}

		}

		if (minX == 1e6 || minY == 1e6)
		{
			wxPrintf("Something went quite wrong in determining the min image dimenstions for the particle stack minX %d, minY %d",minX,minY);
			throw;
		}
		else
		{
			// Make the particle Square
		    minSize = std::min(minX,minY);
		}

		for (iTilt=0; iTilt < nTilts; iTilt++)
		{


			// I am getting some nans on occassion, but so far they only show up in big expensive calcs, so add a nan check and print info to see if
			// the problem can be isolated.
			if (particle_stack[iTilt].HasNan() == true)
			{
				wxPrintf("Frame %d / %d has NaN values, trashing it\n",iTilt,nTilts*(int)N_FRAMES);
			}
			else
			{
				particle_stack[iTilt].Resize(minSize,minSize,1);
				particle_stack[iTilt].WriteSlices(&mrc_out_final,1+iTilt,1+iTilt);
				if (SAVE_PERFECT_REFERENCE)
				{
					reference_stack[iTilt].Resize(minSize,minSize,1);
					reference_stack[iTilt].WriteSlices(&mrc_ref_final,1+iTilt,1+iTilt);
				}
			}
		}
	}
	else
	{
		int current_tilt_sub_frame = 1;
		int current_tilt_sum_saved = 1;
		Image tilt_sum;
		tilt_sum.Allocate(sum_image[iTilt].logical_x_dimension - IMAGETRIMVAL,
				 	 	  sum_image[iTilt].logical_y_dimension - IMAGETRIMVAL, 1);
		tilt_sum.SetToConstant(0.0);
		for (iTilt=0; iTilt < nTilts*N_FRAMES; iTilt++)
		{
			// I am getting some nans on occassion, but so far they only show up in big expensive calcs, so add a nan check and print info to see if
			// the problem can be isolated.
			if (sum_image[iTilt].HasNan() == true)
			{
				wxPrintf("Frame %d / %d has NaN values, trashing it\n",iTilt,nTilts*(int)N_FRAMES);
			}
			else
			{
				if (SAVE_PERFECT_REFERENCE && iTilt < nTilts)
				{
					// Do this first, or else the size from sum_image will already be trimmed
					reference_stack[iTilt].Resize(sum_image[iTilt].logical_x_dimension - IMAGETRIMVAL,
			                							 sum_image[iTilt].logical_y_dimension - IMAGETRIMVAL, 1);
					reference_stack[iTilt].WriteSlices(&mrc_ref_final,1+iTilt,1+iTilt);
				}

				sum_image[iTilt].Resize(sum_image[iTilt].logical_x_dimension - IMAGETRIMVAL,
						                sum_image[iTilt].logical_y_dimension - IMAGETRIMVAL, 1);
				sum_image[iTilt].WriteSlices(&mrc_out_final,1+iTilt,1+iTilt);
				if (current_tilt_sub_frame <= N_FRAMES)
				{
					tilt_sum.AddImage(&sum_image[iTilt]);
					current_tilt_sub_frame += 1;
				}
				else
				{
					tilt_sum.WriteSlices(&mrc_tlt_final,current_tilt_sum_saved,current_tilt_sum_saved);
					tilt_sum.SetToConstant(0.0);
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

	mrc_ref_final.SetPixelSize(this->wanted_pixel_size);
	mrc_ref_final.CloseFile();

	delete [] tilt_psi;
	delete [] tilt_theta;
	delete [] tilt_phi;
	delete [] shift_x;
	delete [] shift_y;
	delete [] shift_z;


    delete [] sum_image;
    if (SAVE_PERFECT_REFERENCE)
    {
    	delete [] reference_stack;
    }
 //   delete noise_dist;
}




void ScatteringPotentialApp::calc_scattering_potential(const PDB * current_specimen,Image *scattering_slab,RotationMatrix rotate_waters,
													   float rotated_oZ, int *slabIDX_start, int *slabIDX_end, int iSlab)

// The if conditions needed to have water and protein in the same function
// make it too complicated and about 10x less parallel friendly.
{

	long nAtoms;
	long current_atom;
	long normalization_factor[this->number_of_threads];




	nAtoms = this->number_of_non_water_atoms;



	// TODO experiment with the scheduling. Until the specimen is consistently full, many consecutive slabs may have very little work for the assigned threads to handle.
	#pragma omp parallel for  num_threads(this->number_of_threads)

	for (current_atom = 0; current_atom < nAtoms; current_atom++)
	{

		int element_id;
		float bFactor;
		float bPlusB[5];
		float radius;
		float ix(0), iy(0), iz(0);
		float dx(0), dy(0), dz (0);
		int indX(0), indY(0), indZ(0);
		float sx(0), sy(0), sz(0);
		float xLow(0),xTop(0),yLow(0),yTop(0),zLow(0),zTop(0);
		int iLim, jLim, kLim;
		int iGaussian;
		float water_offset;
		long atoms_added_idx[(int)powf(size_neighborhood*2+1,3)];
		float atoms_values_tmp[(int)powf(size_neighborhood*2+1,3)];
		int n_atoms_added =0;
		float temp_potential[5] = {0,0,0,0,0};
		double temp_potential_sum = 0;
		double norm_value = 0;
//		int threadIDX = omp_get_thread_num();
		float bfX(0), bfY(0), bfZ(0);


		element_id = current_specimen->my_atoms.Item(current_atom).element_name;
		if (this->min_bFactor != -1)
		{
			bFactor     =  current_specimen->my_atoms.Item(current_atom).bfactor*this->bFactor_scaling + (this->min_bFactor );
//			wxPrintf("%f\n",bFactor);
		}
		else
		{

		    std::random_device rd;  //Will be used to obtain a seed for the random number engine
		    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()


			bFactor = sqrt((current_specimen->my_atoms.Item(current_atom).bfactor*this->bFactor_scaling)/(78.95680235571201)); // 8*pi**2
			std::uniform_real_distribution<float> norm_dist(0.0, bFactor*(1.5)); // The 1.5 is included so the RMSD in 3D is 1*bFactor
			bfX = norm_dist(gen);
			bfY = norm_dist(gen);
			bfZ = norm_dist(gen);
			bFactor     =  0;//current_specimen->my_atoms.Item(current_atom).bfactor*this->bFactor_scaling
//			wxPrintf("%f %f %f\n",bfX,bfY,bfZ);

		}

		// Convert atom origin to pixels and shift by volume origin to get pixel coordinates. Subtract 0.5 to place origin at "center" of voxel
		dx = modff(current_specimen->vol_oX + (current_specimen->my_atoms.Item(current_atom).x_coordinate+0.5 + bfX) / this->wanted_pixel_size, &ix);
		dy = modff(current_specimen->vol_oY + (current_specimen->my_atoms.Item(current_atom).y_coordinate+0.5 + bfY) / this->wanted_pixel_size, &iy);
		dz = modff(rotated_oZ 			    + (current_specimen->my_atoms.Item(current_atom).z_coordinate+0.5 + bfZ) / this->wanted_pixel_size, &iz);





		#pragma omp simd
		for (iGaussian = 0; iGaussian < 5 ; iGaussian++)
		{
			bPlusB[iGaussian] = 2*PI/sqrt(bFactor+SCATTERING_PARAMETERS_B[element_id][iGaussian]);

		}

		if (iz <= slabIDX_end[iSlab]  && iz >= slabIDX_start[iSlab])
		{


		for (sx = -size_neighborhood; sx <= size_neighborhood ; sx++)
		{
			indX = ix + sx;

			for (sy = -size_neighborhood; sy <= size_neighborhood; sy++)
			{
				indY = iy + sy ;
				for (sz = -size_neighborhood; sz <= size_neighborhood ; sz++)
				{
					indZ = iz  + sz;
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
										// Vector to lower left of given voxel
										xLow = this->wanted_pixel_size * (iLim*(dx));
										yLow = this->wanted_pixel_size * (jLim*(dy));
										zLow = this->wanted_pixel_size * (kLim*(dz));

										xTop = this->wanted_pixel_size * ((1-iLim)*(dx) + iLim);
										yTop = this->wanted_pixel_size * ((1-jLim)*(dy) + jLim);
										zTop = this->wanted_pixel_size * ((1-kLim)*(dz) + kLim);

										#pragma omp simd
										for (iGaussian = 0; iGaussian < 5; iGaussian++)
										{
//											bPlusB = 2*PI/sqrt(bFactor+SCATTERING_PARAMETERS_B[element_id][iGaussian]);
											temp_potential[iGaussian]  = (SCATTERING_PARAMETERS_A[element_id][iGaussian] *
															   (erff(bPlusB[iGaussian]*xTop)-erff(bPlusB[iGaussian]*xLow)) *
															   (erff(bPlusB[iGaussian]*yTop)-erff(bPlusB[iGaussian]*yLow)) *
															   (erff(bPlusB[iGaussian]*zTop)-erff(bPlusB[iGaussian]*zLow)));



										} // loop over gaussian fits


									}
								}
							}


						}
						else
						{

							// Vector to lower left of given voxel
							xLow = (sx - dx) * this->wanted_pixel_size;
							yLow = (sy - dy) * this->wanted_pixel_size;
							zLow = (sz - dz) * this->wanted_pixel_size;

							xTop = xLow + this->wanted_pixel_size;
							yTop = yLow + this->wanted_pixel_size;
							zTop = zLow + this->wanted_pixel_size;

							// General case
							#pragma omp simd
							for (iGaussian = 0; iGaussian < 5; iGaussian++)
							{
//								bPlusB = 2*PI/sqrt(bFactor+SCATTERING_PARAMETERS_B[element_id][iGaussian]);
								temp_potential[iGaussian] = (SCATTERING_PARAMETERS_A[element_id][iGaussian] *
												   (erff(bPlusB[iGaussian]*xTop)-erff(bPlusB[iGaussian]*xLow)) *
												   (erff(bPlusB[iGaussian]*yTop)-erff(bPlusB[iGaussian]*yLow)) *
												   (erff(bPlusB[iGaussian]*zTop)-erff(bPlusB[iGaussian]*zLow)));



							} // loop over gaussian fits


						}
						// multiply the outer most term
						#pragma omp simd
						for (iGaussian = 1; iGaussian < 5; iGaussian++)
						{
							temp_potential[0] += temp_potential[iGaussian];
						}

						temp_potential[0] *= OFF_BY_LINEAR_FACTOR*(WAVELENGTH/8.0/this->wanted_pixel_size_sq);
//						radius = this->wanted_pixel_size * sqrt((powf(dx-size_neighborhood+sx,2) + powf(dy-size_neighborhood+sy,2) + powf(dz-size_neighborhood+sz,2)));
//
//						if (radius < CALC_DIST_OTHER)
//						{
//
//
							atoms_added_idx[n_atoms_added] = scattering_slab->ReturnReal1DAddressFromPhysicalCoord(indX,indY,indZ - slabIDX_start[iSlab]);
							atoms_values_tmp[n_atoms_added] = temp_potential[0];
//							scattering_slab->real_values[atoms_added_idx[n_atoms_added]] += temp_potential;
							temp_potential_sum += temp_potential[0];
							n_atoms_added++;
//
//
//						}

					}

				}
			}
		} // end of loop over the neighborhood
				// Loop over it again to normalize

			for (int iNorm = 0; iNorm < 5; iNorm++)
			{
				norm_value += SCATTERING_PARAMETERS_A[element_id][iNorm];
			}
			norm_value = OFF_BY_LINEAR_FACTOR * WAVELENGTH / 8.0 / this->wanted_pixel_size_sq * norm_value / temp_potential_sum;
			for (int iIDX = 0; iIDX < n_atoms_added-1; iIDX++)
			{

				#pragma omp atomic update
				scattering_slab->real_values[atoms_added_idx[iIDX]] += (atoms_values_tmp[iIDX] * norm_value);
			}

		}// if statment into neigh

	} // end loop over atoms





}

void ScatteringPotentialApp::calc_water_potential(Image *projected_water)

// The if conditions needed to have water and protein in the same function
// make it too complicated and about 10x less parallel friendly.
{

	long current_atom;

	float radius;


	// Private variables:

	const int element_id = SOLVENT_TYPE; // use oxygen as proxy for water
	float bFactor = SOLVENT_BFACTOR ;
	float bPlusB[5];
	float ix(0), iy(0), iz(0);
	float dx(0), dy(0), dz (0);
	int indX(0), indY(0), indZ(0);
	int sx(0), sy(0), sz(0);
	float xLow(0),xTop(0),yLow(0),yTop(0),zLow(0),zTop(0);
	int iLim, jLim, kLim;
	int iGaussian;


	float preFactor = OFF_BY_LINEAR_FACTOR*WAVELENGTH/8.0/this->wanted_pixel_size_sq;

	for (int iGaussian = 0; iGaussian < 5; iGaussian++)
	{
		bPlusB[iGaussian] = 2*PI/sqrt(bFactor+SCATTERING_PARAMETERS_B[element_id][iGaussian]);
	}


	int nSubPixCenter = 0;

	for (int iSubPixY = -SUB_PIXEL_NEIGHBORHOOD; iSubPixY <= SUB_PIXEL_NEIGHBORHOOD ; iSubPixY++)
	{
		for (int iSubPixX = -SUB_PIXEL_NEIGHBORHOOD; iSubPixX <= SUB_PIXEL_NEIGHBORHOOD ; iSubPixX++)
		{


		int n_atoms_added = 0;
		float temp_potential = 0;
		double temp_potential_sum = 0;
		double norm_value = 0;
		double raw_sum = 0;


		dx = (float)iSubPixX / (float)(SUB_PIXEL_NEIGHBORHOOD*2+1)+0.5;
		dy = (float)iSubPixY / (float)(SUB_PIXEL_NEIGHBORHOOD*2+1)+0.5;
		dz = (float)0;



		for (sx = -size_neighborhood_water; sx <= size_neighborhood_water ; sx++)
		{

			for (sy = -size_neighborhood_water; sy <=  size_neighborhood_water; sy++)
			{

				for (sz = -size_neighborhood_water; sz <= size_neighborhood_water  ; sz++)
				{

						// Calculate the scattering potential
						temp_potential = 0;
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
											// Vector to lower left of given voxel
											xLow = this->wanted_pixel_size * (iLim*(dx));
											yLow = this->wanted_pixel_size * (jLim*(dy));
											zLow = this->wanted_pixel_size * (kLim*(dz));

											xTop = this->wanted_pixel_size * ((1-iLim)*(dx) + iLim);
											yTop = this->wanted_pixel_size * ((1-jLim)*(dy) + jLim);
											zTop = this->wanted_pixel_size * ((1-kLim)*(dz) + kLim);

											for (iGaussian = 0; iGaussian < 5; iGaussian++)
											{
//												bPlusB = 2*PI/sqrt(bFactor+SCATTERING_PARAMETERS_B[element_id][iGaussian]);
												temp_potential += (SCATTERING_PARAMETERS_A[element_id][iGaussian] *
																   (erf(bPlusB[iGaussian]*xTop)-erf(bPlusB[iGaussian]*xLow)) *
																   (erf(bPlusB[iGaussian]*yTop)-erf(bPlusB[iGaussian]*yLow)) *
																   (erf(bPlusB[iGaussian]*zTop)-erf(bPlusB[iGaussian]*zLow)));



											} // loop over gaussian fits

										}
									}
								}


							}
							else
							{

								// Vector to lower left of given voxel
								xLow = (sx - dx) * this->wanted_pixel_size;
								yLow = (sy - dy) * this->wanted_pixel_size;
								zLow = (sz - dz) * this->wanted_pixel_size;

								xTop = xLow + this->wanted_pixel_size;
								yTop = yLow + this->wanted_pixel_size;
								zTop = zLow + this->wanted_pixel_size;

								// General case
								for (iGaussian = 0; iGaussian < 5; iGaussian++)
								{
//									bPlusB = 2*PI/sqrt(bFactor+SCATTERING_PARAMETERS_B[element_id][iGaussian]);
									temp_potential += (SCATTERING_PARAMETERS_A[element_id][iGaussian] *
													   (erf(bPlusB[iGaussian]*xTop)-erf(bPlusB[iGaussian]*xLow)) *
													   (erf(bPlusB[iGaussian]*yTop)-erf(bPlusB[iGaussian]*yLow)) *
													   (erf(bPlusB[iGaussian]*zTop)-erf(bPlusB[iGaussian]*zLow)));



								} // loop over gaussian fits


							}


							temp_potential *= preFactor;
							projected_water[nSubPixCenter].real_values[projected_water[nSubPixCenter].ReturnReal1DAddressFromPhysicalCoord(sx+size_neighborhood_water,sy+size_neighborhood_water,0)] += temp_potential;
							temp_potential_sum += temp_potential;

				} // end of loop over Z

			} // end of loop Y
		} // end loop X Neighborhood




		// Could do this once since all the waters should be the same.
		for (int iNorm = 0; iNorm < 5; iNorm++)
		{
			norm_value += SCATTERING_PARAMETERS_A[element_id][iNorm];
		}
		norm_value = this->water_scaling * preFactor * norm_value / temp_potential_sum;
		double testSum = 0;
		double testSumAfter = 0;

		for (sx = -size_neighborhood_water; sx <= size_neighborhood_water ; sx++)
		{
			for (sy = -size_neighborhood_water; sy <= size_neighborhood_water ; sy++)
			{
				testSum += projected_water[nSubPixCenter].real_values[projected_water[nSubPixCenter].ReturnReal1DAddressFromPhysicalCoord(sx+size_neighborhood_water,sy+size_neighborhood_water,0)] ;
				projected_water[nSubPixCenter].real_values[projected_water[nSubPixCenter].ReturnReal1DAddressFromPhysicalCoord(sx+size_neighborhood_water,sy+size_neighborhood_water,0)] *= norm_value;
				testSumAfter += projected_water[nSubPixCenter].real_values[projected_water[nSubPixCenter].ReturnReal1DAddressFromPhysicalCoord(sx+size_neighborhood_water,sy+size_neighborhood_water,0)] ;
			}
		}
		nSubPixCenter++;

		}  // inner SubPix

	} // outer SubPix


	if (SAVE_PROJECTED_WATER)
	{
		std::string fileNameOUT = "projected_water.mrc";
		MRCFile mrc_out(fileNameOUT,true);
		for (int iWater = 0; iWater < nSubPixCenter -1 ; iWater++)
		{
			projected_water[iWater].WriteSlices(&mrc_out,iWater+1,iWater+1);
		}

		mrc_out.SetPixelSize(this->wanted_pixel_size);
		mrc_out.CloseFile();
	}

}

void ScatteringPotentialApp::fill_water_potential(const PDB * current_specimen,Image *scattering_slab, Image *scattering_potential, Water *water_box, RotationMatrix rotate_waters,
													   float rotated_oZ, int *slabIDX_start, int *slabIDX_end, int iSlab)

// The if conditions needed to have water and protein in the same function
// make it too complicated and about 10x less parallel friendly.
{

	long current_atom;
	long nWatersAdded = 0;

	float radius;


	// Private variables:
	const int element_id = 3; // use oxygen as proxy for water
	float bFactor = SOLVENT_BFACTOR ;
	float bPlusB[5];
	float ix(0), iy(0), iz(0);
	float dx(0), dy(0), dz (0);
	int indX(0), indY(0), indZ(0);
	int sx(0), sy(0);
	int iSubPixX;
	int iSubPixY;
	int iSubPixLinearIndex;


	float avg_cutoff;
	if( CALC_WATER_NO_HOLE )
	{
		avg_cutoff = 1000; // TODO check this can't break, I think the potential should always be < 1
	}
	else
	{
		avg_cutoff = this->average_at_cutoff[0];
	}
	const int upper_bound = (size_neighborhood_water*2+1);
	const int numel_water = upper_bound*upper_bound;


	// Change previous projected_water to projected_water_atoms for later confusion prevention TODO
	Image projected_water;
	projected_water.Allocate(scattering_slab->logical_x_dimension,scattering_slab->logical_y_dimension,1);
	projected_water.SetToConstant(0.0);



	// TODO experiment with the scheduling. Until the specimen is consistently full, many consecutive slabs may have very little work for the assigned threads to handle.
// With this new insertion approach, threading only slows the overall process. where on small images, 1,4,40 threads are about, 1,1.5,2x and large (6kx6k) 1,2,3x
	long n_waters_ignored = 0;
	#pragma omp parallel for num_threads(this->number_of_threads) private(radius,ix,iy,iz,dx,dy,dz,indX,indY,indZ,sx,sy,iSubPixX,iSubPixY,iSubPixLinearIndex,n_waters_ignored)
	for (current_atom = 0; current_atom < water_box->number_of_waters; current_atom++)
	{


		float temp_potential = 0;
		double temp_potential_sum = 0;
		double norm_value = 0;

		// TODO put other water manipulation methods inside the water class.
		water_box->ReturnCenteredCoordinates(current_atom,dx,dy,dz);

		// Rotate to the slab frame
		rotate_waters.RotateCoords(dx, dy, dz, ix, iy, iz);
		// Shift back to lower left origin, now with the slab oZ origin
		dx = modff(ix + ((float)water_box->vol_oX), &ix);
		dy = modff(iy + ((float)water_box->vol_oY), &iy);
		dz = modff(iz + rotated_oZ, &iz); // Why am I subtracting here? Should it be an add? TODO

		iSubPixX = myroundint((dx) * (float)(SUB_PIXEL_NEIGHBORHOOD*2));
		iSubPixY = myroundint((dy) * (float)(SUB_PIXEL_NEIGHBORHOOD*2)); // Do not add 1 so it comes out indexed from zero.
		iSubPixLinearIndex = iSubPixY*(float)(SUB_PIXEL_NEIGHBORHOOD*2+1) + iSubPixX;



		if ( iz >= slabIDX_start[iSlab] && iz  <= slabIDX_end[iSlab] && myroundint(ix)-1 > 0 && myroundint(iy)-1 > 0 && myroundint(ix)-1 < scattering_slab->logical_x_dimension && myroundint(iy)-1 < scattering_slab->logical_y_dimension &&
			 scattering_slab->ReturnRealPixelFromPhysicalCoord(myroundint(ix)-1,myroundint(iy)-1,myroundint(iz) - slabIDX_start[iSlab]) < avg_cutoff)
		{
			nWatersAdded++;
			int iWater = 0;

			for (sy = 0; sy <  upper_bound ; sy++ )
			{
				indY = myroundint(iy) - upper_bound + sy + size_neighborhood_water; // last term centers water, but I'm not sure why I need it. This all needs to be simplf
				for (sx = 0;  sx < upper_bound ; sx++ )
				{
					indX = myroundint(ix) -upper_bound + sx + size_neighborhood_water;
					// Even with the periodic boundaries checked in shake, the rotation may place waters out of bounds. TODO this is true for non-waters as well.
					if (indX >= 0 && indX < projected_water.logical_x_dimension && indY >= 0 && indY < projected_water.logical_y_dimension)
					{
						#pragma omp atomic update
						projected_water.real_values[projected_water.ReturnReal1DAddressFromPhysicalCoord(indX,indY,0)] +=
								this->projected_water[iSubPixLinearIndex].real_values[this->projected_water[iSubPixLinearIndex].ReturnReal1DAddressFromPhysicalCoord(sx,sy,0)]; // TODO could I land out of bounds?] += projected_water_atoms[iSubPixLinearIndex].real_values[iWater];

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

	if (DO_PRINT) {wxPrintf("\nnWaters %ld added (%2.2f%%) of total on slab %d\n",nWatersAdded,100.0f*(float)nWatersAdded/(float)water_box->number_of_waters, iSlab);}

	std::string fileNameOUT = "tmpWat_prj_comb" + std::to_string(iSlab) + ".mrc";
	MRCFile mrc_out;

	if (SAVE_WATER_AND_OTHER)
	{
		// Only open the file if we are going to use it.
		mrc_out.OpenFile(fileNameOUT,true);
		projected_water.WriteSlices(&mrc_out,1,1);
		scattering_potential[iSlab].WriteSlices(&mrc_out,2,2);

	}


	scattering_potential[iSlab].AddImage(&projected_water);

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

	for (prjX = 0; prjX < image_to_project->logical_x_dimension; prjX++)
	{

		for (prjY = 0; prjY < image_to_project->logical_y_dimension; prjY++)
		{
			slab_address = image_to_project_into[iSlab].ReturnReal1DAddressFromPhysicalCoord(prjX,prjY,0);
			pixel_accumulator  = 0;
			for (prjZ = 0; prjZ < image_to_project->logical_z_dimension; prjZ++)
			{

				pixel_accumulator +=  image_to_project->ReturnRealPixelFromPhysicalCoord(prjX,prjY,prjZ);


			}
			 image_to_project_into[iSlab].real_values[slab_address] += (float)pixel_accumulator;

		}
	}

}

void ScatteringPotentialApp::taper_edges(Image *image_to_taper,  int iSlab)
{
	// Taper edges to the mean TODO see if this can be removed
	// Update with the current mean. Why am I saving this? Is it just for SOLVENT ==1 ? Then probably can kill TODO
	int prjX, prjY, prjZ;
	int edgeX = 0;
	int edgeY = 0;
	long slab_address;
	float taper_val;

	this->image_mean[iSlab] =  image_to_taper[iSlab].ReturnAverageOfRealValues(0.0);
	if (DO_PRINT) {wxPrintf("image mean for taper %f\n",this->image_mean[iSlab]);}

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
				image_to_taper[iSlab].real_values[slab_address] += this->image_mean[iSlab]*(1-taper_val);
			}


		}
	}

}

void ScatteringPotentialApp::apply_sqrt_DQE_or_NTF(Image *image_in, int iTilt_IDX, const float a0, const float a1, const float a2, const float w)
{


	image_in[iTilt_IDX].ForwardFFT(true);
	float x_coord_sq, y_coord_sq, spatial_frequency;
	long pixel_counter = 0;
	for (int j = 0; j <= image_in[iTilt_IDX].physical_upper_bound_complex_y; j++)
	{
		// the two is because the curve is fit to have the physical nyquist
		y_coord_sq = powf( 2 *image_in[iTilt_IDX].ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * image_in[iTilt_IDX].fourier_voxel_size_y , 2);

		for (int i = 0; i <= image_in[iTilt_IDX].physical_upper_bound_complex_x; i++)
		{
			// the two is because the curve is fit to have the physical nyquist
			x_coord_sq = powf( 2 * i * image_in[iTilt_IDX].fourier_voxel_size_x , 2);

			// compute squared radius, in units of reciprocal pixels

			spatial_frequency = sqrt(x_coord_sq + y_coord_sq);

			image_in[iTilt_IDX].complex_values[pixel_counter] *= a0 + a1*std::cos(spatial_frequency*w) + a2*std::sin(spatial_frequency*w);

			pixel_counter++;

		}
	}

	image_in[iTilt_IDX].BackwardFFT();



}

void ScatteringPotentialApp::normalize_set_dose_expectation(Image *sum_image, int iTilt_IDX, float current_thickness)
{


    float avg_radius; // If there are strong unique features in the corners, this might prove to be a problem. Could
    float loss_due_to_thickness;
    avg_radius = std::min((sum_image[iTilt_IDX].logical_x_dimension-IMAGEPADDING)/2.0f,(sum_image[iTilt_IDX].logical_y_dimension-IMAGEPADDING)/2.0f);
    if (DO_PRINT) {wxPrintf("wanted radius is %f, and size is %d %d\n",avg_radius,sum_image[iTilt_IDX].logical_x_dimension,sum_image[iTilt_IDX].logical_y_dimension);}
	loss_due_to_thickness = expf(-current_thickness/MEAN_FREE_PATH);

	if (DO_PRINT) {wxPrintf("\nWith a rotated thickness of %3.4e Angstrom calculated a probability of NOT scattering inelastically of %3.3f, total scaling %3.3f\n",current_thickness,loss_due_to_thickness,
			1/N_SUB_FRAMES * loss_due_to_thickness * this->wanted_pixel_size_sq * DOSE_PER_FRAME / sum_image[iTilt_IDX].ReturnAverageOfRealValues(0.95*avg_radius));}

	//angstrom_sq = sum_image[iTilt_IDX].number_of_real_space_pixels * this->wanted_pixel_size_sq;
	//total_density = sum_image[iTilt_IDX].ReturnAverageOfRealValues(0.0) * sum_image[iTilt_IDX].number_of_real_space_pixels;

	sum_image[iTilt_IDX].MultiplyByConstant(1/N_SUB_FRAMES * loss_due_to_thickness * this->wanted_pixel_size_sq * DOSE_PER_FRAME / sum_image[iTilt_IDX].ReturnAverageOfRealValues(0.95*avg_radius));
}

void ScatteringPotentialApp::calc_average_intensity_at_solvent_cutoff(float average_bfactor)
{


	int iCutoff;
	float bFactor = 0;
	float bPlusB = 0;
	float radius_squared;

	// 0 is used for individual water molecules
	// 1,2,and 2 are the range over which the bulk solvent is tapered for the 3d ground truth calculation
	for (iCutoff = 0; iCutoff < 4 ; iCutoff++)
	{

		int number_at_cutoff = 0;

		float xLow = SOLVENT_CUTOFF[iCutoff] - (this->wanted_pixel_size * 0.5);
		float xTop = SOLVENT_CUTOFF[iCutoff] + (this->wanted_pixel_size * 0.5);

		if (this->min_bFactor != -1)
		{
			bFactor = this->min_bFactor + average_bfactor*this->bFactor_scaling ;
		}
		else
		{
			bFactor = 0; // If using physical displacement, just use a zero bfactor
		}


		// TODO add an explicit weigting by element type actually included in a given PDB
		for (int element_id  = 2; element_id  < 5; element_id++)
		{
			double temp_potential = 0;
			for (int iGaussian = 0; iGaussian < 5; iGaussian++)
				// Vector to lower left of given voxel

				{
	//				bPlusB = 2*PI/sqrt(bFactor+SCATTERING_PARAMETERS_B[element_id][iGaussian]);
	//				temp_potential += (SCATTERING_PARAMETERS_A[element_id][iGaussian] *
	//								  powf((erf(bPlusB*xTop)-erf(bPlusB*xLow)),3));
				bPlusB = bFactor+SCATTERING_PARAMETERS_B[element_id][iGaussian];
				temp_potential += (SCATTERING_PARAMETERS_A[element_id][iGaussian] *
								   powf(4*PI/bPlusB,3/2) *
								   exp(-4*PI*PI*SOLVENT_CUTOFF[iCutoff]/bPlusB));


				} // loop over gaussian fits
			this->average_at_cutoff[iCutoff] += OFF_BY_LINEAR_FACTOR*WAVELENGTH/8.0/this->wanted_pixel_size_sq*temp_potential;
			number_at_cutoff++;
		}

		this->average_at_cutoff[iCutoff] /= number_at_cutoff;

		if (DO_PRINT) {wxPrintf("Average at cutoff %d is %3.3e\n",iCutoff,this->average_at_cutoff[iCutoff]);}
	}

}

void ScatteringPotentialApp::fill_water_potential_flat(Image *potential3D)
{

	long current_atom;
	double nWatersAdded = 0;

	float radius;


	// Private variables:
	const int taper_length = 12;
	const int element_id = 3; // use oxygen as proxy for water
	float bFactor = SOLVENT_BFACTOR ;
//	double water_scale[taper_length] = {0,0.0438,0.1684,0.3550,0.5750,0.7950,0.9816,1.1062,1.15,1.1125,1.0375,1.0};
	double water_scale[taper_length] = {0,0.0438,0.1684,0.3550,0.5750,0.7950,0.9816,1.0,1.0,1.0,1.0,1.0};

	double water_thresh[taper_length]= {0};
	float norm_value = 0;
	float bPlusB[5];
	float ix(0), iy(0), iz(0);
	float dx(0), dy(0), dz (0);
	int indX(0), indY(0), indZ(0);
	int sx(0), sy(0);
	int iSubPixX;
	int iSubPixY;
	int iSubPixLinearIndex;
	long n_voxels = potential3D->logical_x_dimension*potential3D->logical_y_dimension*potential3D->logical_z_dimension;



	// Get the total specimen potential for a single water molecule
	for (int iNorm = 0; iNorm < 5; iNorm++)
	{
		norm_value += SCATTERING_PARAMETERS_A[element_id][iNorm];
	}
	norm_value =  OFF_BY_LINEAR_FACTOR*WAVELENGTH/8.0/this->wanted_pixel_size_sq * norm_value;

	// Get the max volume occupied by water
	current_atom = 0;
	for (int k = 0 ; k < potential3D->logical_z_dimension; k++)
	{
		for (int j = 0; j < potential3D->logical_y_dimension; j++)
		{
			for (int i = 0; i < potential3D->logical_x_dimension; i++)
			{
				if (potential3D->real_values[current_atom] < this->average_at_cutoff[2])
				{

					nWatersAdded++;
				}

				current_atom++;
			}
			current_atom += potential3D->padding_jump_value;
		}
	}



	wxPrintf("Water occupies %2.2f percent of the 3d\n",100*nWatersAdded/((double)n_voxels));



	// Set the average water potential per voxel (TODO these should come from the already generated water object)
	double waters_per_angstrom_cubed = 0.94 * 602.2140857 / (18.01528 * 1000);
	double mean_water_potential = norm_value*waters_per_angstrom_cubed*pow((double)this->wanted_pixel_size,3);
	wxPrintf("%2.2e %2.2e\n",norm_value,mean_water_potential);
	// For the taper, should probably use Shang/Sigworths hydration shell model that spikes at 1.15 x the mean potential ~ 2 Ang and then drops to the mean potential around 3
	// For initial testing just use a cosine edge from 0.5 --> 2, then 2--0.3
    // 1.1062    0.9816    0.7950    0.5750    0.3550    0.1684    0.0438 0
	// 1.1125    1.0375 1.0
	float protein_to_water_max_inc = (this->average_at_cutoff[2] - this->average_at_cutoff[1]) / 8.0f;
	float water_max_to_avg_inc     = (this->average_at_cutoff[3] - this->average_at_cutoff[2]) / 3.0f;
	for (int iCut = 0; iCut < 9; iCut++)
	{

		water_thresh[iCut] = this->average_at_cutoff[1] + iCut*protein_to_water_max_inc;
		wxPrintf("water threshold at %d is %2.2e\n",iCut,water_thresh[iCut]);

	}
	for (int iCut = 0; iCut < 3 ; iCut++)
	{
		water_thresh[iCut+9] = this->average_at_cutoff[2] + (iCut+1)*water_max_to_avg_inc;
		wxPrintf("water threshold at %d is %2.2e\n",iCut+9,water_thresh[iCut+9]);
	}


	// fill the volume
	current_atom = 0;
	for (int k = 0 ; k < potential3D->logical_z_dimension; k++)
	{
		for (int j = 0; j < potential3D->logical_y_dimension; j++)
		{
			for (int i = 0; i < potential3D->logical_x_dimension; i++)
			{
				for (int iTaper = taper_length - 1; iTaper >= 0; iTaper-- )
				{

					if(potential3D->real_values[current_atom] >= water_thresh[0])
					{
						potential3D->real_values[current_atom] = 0;
						break;
					}
					else if (potential3D->real_values[current_atom] < water_thresh[iTaper])
					{
//						potential3D->real_values[current_atom] += water_scale[iTaper]*mean_water_potential;
						potential3D->real_values[current_atom] = water_scale[iTaper]*mean_water_potential;

						break;
					}

				}
				current_atom++;
			}
			current_atom += potential3D->padding_jump_value;
		}
	}


}


