#define OPEN_TO_READ 0
#define OPEN_TO_WRITE 1
#define OPEN_TO_APPEND 2
#define MAX_NUMBER_OF_TIMESTEPS 200
#define NUMBER_OF_ATOM_TYPES 16

class Atom {
	// We don't know how many atoms there will be at the outset so use this wxArrayObj
	private:

	public:

		wxString name;
		wxString atom_type;
		float x_coordinate;  // Angstrom
		float y_coordinate;  // Angstrom
		float z_coordinate;  // Angstrom
		float occupancy;
		float bfactor;
		float charge;
		float relative_bfactor; // Meik et al.
		int element_name;


};

WX_DECLARE_OBJARRAY(Atom, ArrayOfAtoms);

class ParticleTrajectory {

	// We don't know how many time steps there will be at the outset so use this wxArrayObj
	private:

	public:


		// I am assuming less than 200 time steps, but maybe too conservative. Cost is pretty small
		float current_orientation[MAX_NUMBER_OF_TIMESTEPS][12]; // Total xyz shift in ang then listed in column major order (surprised that marix.cpp does this. m00,m10,m20...m22
		float current_update[MAX_NUMBER_OF_TIMESTEPS][6]; // Update to get here. the first entries in these two should be the same.


};

WX_DECLARE_OBJARRAY(ParticleTrajectory, ArrayOfParticleTrajectories);



class PDB {

	private:

		void Init();
		wxString text_filename;
		long access_type;
		wxFileInputStream *input_file_stream;
		wxTextInputStream *input_text_stream;
		wxFileOutputStream *output_file_stream;
		wxTextOutputStream *output_text_stream;


	public:

		ArrayOfAtoms my_atoms;
		ArrayOfParticleTrajectories my_trajectory;
		//ArrayOfParticleInstances my_particle;

		// Constructors
		PDB();
		PDB(long number_of_non_water_atoms, float cubic_size);
		PDB(wxString Filename, long wanted_access_type, long wanted_records_per_line = 1);
		~PDB();

		// data

		long number_of_lines;
		long number_of_atoms;
		int records_per_line;
		double center_of_mass[3];
		int number_of_particles_initialized;
		long number_of_each_atom[NUMBER_OF_ATOM_TYPES];
		float atomic_volume;
		float average_bFactor;
		float vol_angX, vol_angY, vol_angZ;
		long vol_nX, vol_nY, vol_nZ;
		long vol_oX, vol_oY, vol_oZ;
		float cubic_size;
		float offset_z;
		float min_z;
		float max_z;



		// Methods

        void Open(wxString Filename, long wanted_access_type, long wanted_records_per_line = 1);
        void Close();
        void Rewind();
        void Flush();
        wxString ReturnFilename();
        RotationMatrix defaultRot;

		void ReadLine(float *data_array);
        void WriteLine(float *data_array);
        void WriteLine(double *data_array);
        void WriteCommentLine(const char * format, ...);
        void TransformBaseCoordinates(float wanted_origin_x,float wanted_origin_y,float wanted_origin_z, float euler1, float euler2, float euler3);
        void TransformLocalAndCombine(PDB *pdb_ensemble, int number_of_pdbs, long number_of_non_water_atoms,float wanted_pixel_size, int time_step, RotationMatrix particle_rot, float shift_z);
        void TransformGlobalAndSortOnZ(long number_of_non_water_atoms,float shift_x, float shift_y, float shift_z,  RotationMatrix rotate_waters);
        float ReturnRelativeBFactor(wxString residue_name);

        inline bool IsNonAminoAcid(wxString atom_name)
        {
        	// TODO make sure this is a valid way to check
        	bool isNonAminoAcid = true;
        	if (atom_name.length() == 3) { isNonAminoAcid = false ;}
        	return isNonAminoAcid;
        }
        inline bool IsBackbone(wxString atom_name)
        {
        	// Assuming I'm not going to run it any parsing problems with case sensitivity. This might be a bad assumption FIXME
        	bool isBackbone = false;
        	if (atom_name == "C" || atom_name == "CA" || atom_name == "N" || atom_name == "O") { isBackbone = true; }
        	return isBackbone;

        }
        inline bool  IsAcidicOxygen(wxString atom_name)
        {
        	// Assuming I'm not going to run it any parsing problems with case sensitivity. This might be a bad assumption FIXME
        	bool isAcidicOxygen = false; //Asp Glu C-term
        	if (atom_name == "OD2" || atom_name == "OE2" || atom_name == "OXT" ) { isAcidicOxygen = true; }
        	return isAcidicOxygen;
        }


};
