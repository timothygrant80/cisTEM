#define OPEN_TO_READ 0
#define OPEN_TO_WRITE 1
#define OPEN_TO_APPEND 2
#define MAX_NUMBER_OF_TIMESTEPS 2000

#define MAX_NUMBER_OF_NOISE_PARTICLES 6


enum  AtomType : int  { hydrogen = 0,     carbon = 1,     nitrogen = 2,     oxygen = 3,         fluorine = 4,
                        sodium = 5,     magnesium = 6,  silicon = 17,     phosphorus = 7,     sulfur = 8,
                        chlorine = 9,    potassium = 10, calcium = 11,     manganese = 12,     iron = 13,
                        cobalt = 18,     zinc = 14,         selenium = 19,     gold = 20,             water = 15,
                        oxygen_anion_1 = 16, plasmon = 21,
                        };


class Atom {
    // We don't know how many atoms there will be at the outset so use this wxArrayObj
    private:

    public:

        // If any fields are added or modified here, be sure to update the CopyAtom method in the PDB class.
        wxString name;
        AtomType atom_type;
        bool is_real_particle;
        float x_coordinate;  // Angstrom
        float y_coordinate;  // Angstrom
        float z_coordinate;  // Angstrom
        float occupancy;
        float bfactor;
        float charge;


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

class InitialOrientations {

public:
    float euler1;
    float euler2;
    float euler3;
    float ox;
    float oy;
    float oz;

    InitialOrientations(float phi, float theta, float psi, float x, float y, float z) :  euler1(phi),  euler2(theta), euler3(psi), ox(x), oy(y), oz(z) {}
};

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
    // This should be used to replace my_trajectory. For now just using for the noise atoms.
    AnglesAndShifts my_angles_and_shifts[MAX_NUMBER_OF_NOISE_PARTICLES];
    std::vector< InitialOrientations > initial_values;

    // If generating a particle stack, this may be set to true after PDB initialization. It wouldn't make sense to enable this for a micrograph or tilt-series
    bool generate_noise_atoms = false;
    int number_of_noise_particles = 0;
    int max_number_of_noise_particles = 0;
    float noise_particle_radius_as_mutliple_of_particle_radius;
    float noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius;
    float noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius;
    float emulate_tilt_angle;
    bool shift_by_center_of_mass;
    //ArrayOfParticleInstances my_particle;
    // Constructors
    PDB();
    PDB(long number_of_non_water_atoms, float cubic_size, float wanted_pixel_size, int minimum_paddeding_x_and_y, double minimum_thickness_z,
            int max_number_of_noise_particles,
            float wanted_noise_particle_radius_as_mutliple_of_particle_radius,
            float wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
            float wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
            float emulate_tilt_angle,
            bool shift_by_center_of_mass, 
            bool is_alpha_fold_prediction,
            bool use_star_file);

    PDB(wxString Filename, long wanted_access_type, float wanted_pixel_size, long wanted_records_per_line, int minimum_paddeding_x_and_y, double minimum_thickness_z,
        int max_number_of_noise_particles,
        float wanted_noise_particle_radius_as_mutliple_of_particle_radius,
        float wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
        float wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
        float emulate_tilt_angle,
        bool shift_by_center_of_mass, 
        bool is_alpha_fold_prediction,
        cisTEMParameters& wanted_star_file, bool use_star_file);

    PDB(wxString Filename, long wanted_access_type, float wanted_pixel_size, long wanted_records_per_line, int minimum_paddeding_x_and_y, double minimum_thickness_z, double *center_of_mass, bool is_alpha_fold_prediction);

    ~PDB();

    // data
    long number_of_lines;
    long number_of_atoms; // total atoms to simulate, active in this particle
    long number_of_real_and_noise_atoms; // total atoms in the PDB object
    long number_of_real_atoms; // atoms not in the noise particles

    int records_per_line;
    double center_of_mass[3];
    bool use_provided_com;
    bool is_alpha_fold_prediction;

    int number_of_particles_initialized;
    long number_of_each_atom[NUMBER_OF_ATOM_TYPES];
    float atomic_volume;
    float average_bFactor;
    float pixel_size;
    float vol_angX, vol_angY, vol_angZ;
    int vol_nX, vol_nY, vol_nZ;
    int vol_oX, vol_oY, vol_oZ;
    float cubic_size;
    float offset_z;
    float min_z;
    float max_z;
    float max_radius = 0;

    int MIN_PADDING_XY;
    double MIN_THICKNESS;

    cisTEMParameters star_file_parameters;
    bool use_star_file;

    // H(0),C(1),N(2),O(3),F(4),Na(5),Mg(6),P(7),S(8),Cl(9),K(10),Ca(11),Mn(12),Fe(13),Zn(14),H20(15),0-(16)

    // This should probably be a copy constructor in the Atom class FIXME
    Atom CopyAtom(Atom &atom_to_copy) {
        Atom atom_out;
        atom_out.atom_type = atom_to_copy.atom_type;
        atom_out.is_real_particle = atom_to_copy.is_real_particle;
        atom_out.name = atom_to_copy.name;
        atom_out.x_coordinate = atom_to_copy.x_coordinate;  // Angstrom
        atom_out.y_coordinate = atom_to_copy.y_coordinate;  // Angstrom
        atom_out.z_coordinate = atom_to_copy.z_coordinate;  // Angstrom
        atom_out.occupancy = atom_to_copy.occupancy;
        atom_out.bfactor = atom_to_copy.bfactor;
        atom_out.charge = atom_to_copy.charge;
        return atom_out;
    };


    // Methods

    void Open(wxString Filename, long wanted_access_type, long wanted_records_per_line = 1);
    void SetEmpty();

    void Close();
    void Rewind();
    void Flush();
    wxString ReturnFilename();
    RotationMatrix defaultRot;

    void ReadLine(float *data_array);
    void WriteLine(float *data_array);
    void WriteLine(double *data_array);
    void WriteCommentLine(const char * format, ...);
    void TransformBaseCoordinates(float wanted_origin_x,float wanted_origin_y,float wanted_origin_z, float euler1, float euler2, float euler3, int particle_idx, int frame_number);
    void TransformLocalAndCombine(PDB *pdb_ensemble, int number_of_pdbs, int frame_number, RotationMatrix particle_rot, float shift_z, bool is_single_particle = false);
    void TransformGlobalAndSortOnZ(long number_of_non_water_atoms,float shift_x, float shift_y, float shift_z,  RotationMatrix rotate_waters);

    inline bool IsNonAminoAcid(wxString atom_name) {
        // TODO make sure this is a valid way to check
        bool isNonAminoAcid = true;
        if (atom_name.length() == 3) { isNonAminoAcid = false ;}
        return isNonAminoAcid;
    }
    inline bool IsBackbone(wxString atom_name) {
        // Assuming I'm not going to run it any parsing problems with case sensitivity. This might be a bad assumption FIXME
        bool isBackbone = false;
        if (atom_name == "C" || atom_name == "CA" || atom_name == "N" || atom_name == "O") { isBackbone = true; }
        return isBackbone;
    }
    inline bool  IsAcidicOxygen(wxString atom_name) {
        // Assuming I'm not going to run it any parsing problems with case sensitivity. This might be a bad assumption FIXME
        bool isAcidicOxygen = false; //Asp Glu C-term
        if (atom_name == "OD2" || atom_name == "OE2" || atom_name == "OXT" ) { isAcidicOxygen = true; }
        return isAcidicOxygen;
    }

};
