#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfParticleTrajectories);

#include "../../include/gemmi/model.hpp"
#include "../../include/gemmi/mmread.hpp"
#include "../../include/gemmi/gz.hpp"

Atom::Atom( ) {
    name             = "";
    atom_type        = hydrogen;
    is_real_particle = true;
    x_coordinate     = 0.0;
    y_coordinate     = 0.0;
    z_coordinate     = 0.0;
    occupancy        = 0.0;
    bfactor          = 0.0;
    charge           = 0.0;
}

Atom::Atom(const wxString& name, const bool& is_real_particle, const AtomType& atom_type, const float& x, const float& y, const float& z, const float& occ, const float& bfactor, const float& charge) {
    this->name             = name;
    this->is_real_particle = is_real_particle;
    this->atom_type        = atom_type;
    this->is_real_particle = true;
    this->x_coordinate     = x;
    this->y_coordinate     = y;
    this->z_coordinate     = z;
    this->occupancy        = occ;
    this->bfactor          = bfactor;
    this->charge           = charge;
}

Atom::~Atom( ) {}

Atom::Atom(const Atom& other_atom) {
    *this = other_atom;
}

Atom& Atom::operator=(const Atom& other_atom) {
    *this = &other_atom;
    return *this;
}

Atom& Atom::operator=(const Atom* other_atom) {
    if ( this != other_atom ) {
        this->atom_type        = other_atom->atom_type;
        this->is_real_particle = other_atom->is_real_particle;
        this->name             = other_atom->name;
        this->x_coordinate     = other_atom->x_coordinate; // Angstrom
        this->y_coordinate     = other_atom->y_coordinate; // Angstrom
        this->z_coordinate     = other_atom->z_coordinate; // Angstrom
        this->occupancy        = other_atom->occupancy;
        this->bfactor          = other_atom->bfactor;
        this->charge           = other_atom->charge;
    }
    return *this;
}

const double MIN_PADDING_Z    = 4;
const int    MAX_XY_DIMENSION = 4096 * 2;

// Define fixed width limits for PDB reading
#define NAMESTART 12 //Char
#define NAMELENGTH 4
#define RESIDUESTART 16
#define RESIDUELENGTH 5
#define XSTART 30 // Float
#define YSTART 38
#define ZSTART 46
#define XYZLENGTH 8
#define OCCUPANCYSTART 54 //Float
#define OCCUPANCYLENGTH 6
#define BFACTORSTART 60 //Float
#define BFACTORLENGTH 6
#define WATERSTART 72 // Char, where VMD put WTAA for a TIP3 Water. This is a deprecated segment id spot also used still by Chimera for pdbSegment attribute
#define WATERLENGTH 4
#define ELEMENTSTART 76 // Char, sometimes length = 3, e.g. N1+ or O1- :: I expect not to see O1+ s.t. the charge may be inferred from the extra "1" and element name
#define ELEMENTLENGTH 2
#define CHARGESTART 78 //Char, not enabled, set to 0.0
#define CHARGELENGTH 2
#define REMARKSTART 7 // Remark ID 351 used to hold particle instances.
#define REMARKLENGTH 4
#define XPARTICLE 11
#define YPARTICLE 19
#define ZPARTICLE 27
#define E1PARTICLE 35
#define E2PARTICLE 43
#define E3PARTICLE 51
#define PARTICLELENGTH 8

PDB::PDB( ) {
    SetDefaultValues( );
}

void PDB::SetDefaultValues( ) {
    input_file_stream  = NULL;
    input_text_stream  = NULL;
    output_file_stream = NULL;
    output_text_stream = NULL;

    SetEmpty( );
}

PDB::PDB(const PDB& other_pdb) {
    *this = other_pdb;
}

PDB& PDB::operator=(const PDB& other_pdb) {

    *this = &other_pdb;
    return *this;
}

PDB& PDB::operator=(const PDB* other_pdb) {

    if ( this != other_pdb ) {
        SetDefaultValues( );
        this->generate_noise_atoms          = other_pdb->generate_noise_atoms;
        this->number_of_noise_particles     = other_pdb->number_of_noise_particles;
        this->max_number_of_noise_particles = other_pdb->max_number_of_noise_particles;

        this->noise_particle_radius_as_mutliple_of_particle_radius                        = other_pdb->noise_particle_radius_as_mutliple_of_particle_radius;
        this->noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius = other_pdb->noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius;
        this->noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius = other_pdb->noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius;

        this->emulate_tilt_angle             = other_pdb->emulate_tilt_angle;
        this->shift_by_center_of_mass        = other_pdb->shift_by_center_of_mass;
        this->number_of_lines                = other_pdb->number_of_lines;
        this->number_of_atoms                = other_pdb->number_of_atoms;
        this->number_of_real_and_noise_atoms = other_pdb->number_of_real_and_noise_atoms;
        this->number_of_real_atoms           = other_pdb->number_of_real_atoms;
        this->records_per_line               = other_pdb->records_per_line;
        this->center_of_mass[0]              = other_pdb->center_of_mass[0];
        this->center_of_mass[1]              = other_pdb->center_of_mass[1];
        this->center_of_mass[2]              = other_pdb->center_of_mass[2];
        this->use_provided_com               = other_pdb->use_provided_com;
        this->is_alpha_fold_prediction       = other_pdb->is_alpha_fold_prediction;
        this->pixel_size                     = other_pdb->pixel_size;
        this->average_bFactor                = other_pdb->average_bFactor;
        // This will be set in the loop below over Transform Coordintaes
        // this->number_of_particles_initialized = other_pdb->number_of_particles_initialized;

        for ( int iAtom = 0; iAtom < cistem::number_of_atom_types; iAtom++ ) {
            this->number_of_each_atom[iAtom] = other_pdb->number_of_each_atom[iAtom];
        }

        this->atomic_volume        = other_pdb->atomic_volume;
        this->vol_angX             = other_pdb->vol_angX;
        this->vol_angY             = other_pdb->vol_angY;
        this->vol_angZ             = other_pdb->vol_angZ;
        this->vol_nX               = other_pdb->vol_nX;
        this->vol_nY               = other_pdb->vol_nY;
        this->vol_nZ               = other_pdb->vol_nZ;
        this->vol_oX               = other_pdb->vol_oX;
        this->vol_oY               = other_pdb->vol_oY;
        this->vol_oZ               = other_pdb->vol_oZ;
        this->cubic_size           = other_pdb->cubic_size;
        this->offset_z             = other_pdb->offset_z;
        this->min_z                = other_pdb->min_z;
        this->max_z                = other_pdb->max_z;
        this->max_radius           = other_pdb->max_radius;
        this->MIN_PADDING_XY       = other_pdb->MIN_PADDING_XY;
        this->MIN_THICKNESS        = other_pdb->MIN_THICKNESS;
        this->star_file_parameters = other_pdb->star_file_parameters;
        this->use_star_file        = other_pdb->use_star_file;

        this->atoms.reserve(other_pdb->atoms.size( ));
        for ( long current_atom = 0; current_atom < other_pdb->atoms.size( ); current_atom++ ) {
            this->atoms.push_back(other_pdb->atoms[current_atom]);
            // std::cerr << "Atom is " << atoms.back( ).name << " " << current_atom << " " << atoms.back( ).atom_type << " " << atoms.back( ).bfactor << '\n';
            // if ( isnan(atoms.back( ).bfactor) )
            //     std::cerr << "Atom is " << atoms.back( ).name << " " << current_atom << " " << atoms.back( ).atom_type << " " << atoms.back( ).bfactor << '\n';
        }

        for ( int i = 0; i < other_pdb->initial_values.size( ); i++ ) {
            this->TransformBaseCoordinates(other_pdb->initial_values[i].ox, other_pdb->initial_values[i].oy, other_pdb->initial_values[i].oz,
                                           other_pdb->initial_values[i].euler1, other_pdb->initial_values[i].euler2, other_pdb->initial_values[i].euler3, i, 0);
        }
    }
    return *this;
}

PDB::PDB(long   number_of_non_water_atoms,
         float  cubic_size,
         float  wanted_pixel_size,
         int    minimum_padding_x_and_y,
         double minimum_thickness_z,
         int    max_number_of_noise_particles,
         float  wanted_noise_particle_radius_as_mutliple_of_particle_radius,
         float  wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
         float  wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
         float  wanted_tilt_angle_to_emulate,
         bool   shift_by_center_of_mass,
         bool   is_alpha_fold_prediction,
         bool   use_star_file) {

    this->use_star_file = use_star_file;
    wxPrintf("IN constructor 1 and use_star_file = %d\n", use_star_file);

    input_file_stream  = NULL;
    input_text_stream  = NULL;
    output_file_stream = NULL;
    output_text_stream = NULL;
    // Create a total PDB object to hold all the atoms in a specimen at a given time in the trajectory
    atoms.reserve(number_of_non_water_atoms);

    this->cubic_size = cubic_size;

    this->use_provided_com         = false;
    this->is_alpha_fold_prediction = is_alpha_fold_prediction;

    this->MIN_PADDING_XY = minimum_padding_x_and_y;
    this->MIN_THICKNESS  = minimum_thickness_z;

    SetEmpty( );
    this->pixel_size = wanted_pixel_size;

    // The default is to not generate neighboring noise particles. This should probably be switched.
    if ( max_number_of_noise_particles > 0 ) {
        this->generate_noise_atoms = true;
    }

    this->max_number_of_noise_particles                                               = max_number_of_noise_particles;
    this->noise_particle_radius_as_mutliple_of_particle_radius                        = wanted_noise_particle_radius_as_mutliple_of_particle_radius;
    this->noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius = wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius;
    this->noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius = wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius;
    this->emulate_tilt_angle                                                          = wanted_tilt_angle_to_emulate;
    this->shift_by_center_of_mass                                                     = shift_by_center_of_mass;
}

PDB::PDB(wxString          Filename,
         long              wanted_access_type,
         float             wanted_pixel_size,
         long              wanted_records_per_line,
         int               minimum_padding_x_and_y,
         double            minimum_thickness_z,
         int               max_number_of_noise_particles,
         float             wanted_noise_particle_radius_as_mutliple_of_particle_radius,
         float             wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
         float             wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
         float             wanted_tilt_angle_to_emulate,
         bool              shift_by_center_of_mass,
         bool              is_alpha_fold_prediction,
         cisTEMParameters& wanted_star_file,
         bool              use_star_file) {

    star_file_parameters = wanted_star_file;
    this->use_star_file  = use_star_file;

    wxPrintf("IN constructor 2 and use_star_file = %d\n", use_star_file);

    input_file_stream  = NULL;
    input_text_stream  = NULL;
    output_file_stream = NULL;
    output_text_stream = NULL;

    this->use_provided_com         = false;
    this->is_alpha_fold_prediction = is_alpha_fold_prediction;

    this->MIN_PADDING_XY = minimum_padding_x_and_y;
    this->MIN_THICKNESS  = minimum_thickness_z;

    SetEmpty( );
    this->pixel_size = wanted_pixel_size;
    // The default is to not generate neighboring noise particles. This should probably be switched.
    if ( max_number_of_noise_particles > 0 ) {
        this->generate_noise_atoms = true;
    }

    this->max_number_of_noise_particles                                               = max_number_of_noise_particles;
    this->noise_particle_radius_as_mutliple_of_particle_radius                        = wanted_noise_particle_radius_as_mutliple_of_particle_radius;
    this->noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius = wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius;
    this->noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius = wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius;
    this->emulate_tilt_angle                                                          = wanted_tilt_angle_to_emulate;
    this->shift_by_center_of_mass                                                     = shift_by_center_of_mass;

    Open(Filename, wanted_access_type, wanted_records_per_line);
}

PDB::PDB(wxString Filename,
         long     wanted_access_type,
         float    wanted_pixel_size,
         long     wanted_records_per_line,
         int      minimum_padding_x_and_y,
         double   minimum_thickness_z,
         bool     is_alpha_fold_prediction,
         double*  COM) {
    input_file_stream  = NULL;
    input_text_stream  = NULL;
    output_file_stream = NULL;
    output_text_stream = NULL;

    use_star_file = false;

    MIN_PADDING_XY = minimum_padding_x_and_y;
    MIN_THICKNESS  = minimum_thickness_z;

    use_provided_com               = true;
    this->is_alpha_fold_prediction = is_alpha_fold_prediction;

    if ( COM ) {
        for ( int iCOM = 0; iCOM < 3; iCOM++ ) {
            center_of_mass[iCOM] = COM[iCOM];
            wxPrintf("Using provided center of mass %d %3.3f\n", iCOM, this->center_of_mass[iCOM]);
        }
    }
    else {

        use_provided_com = false;
    }

    shift_by_center_of_mass = true;

    SetEmpty( );
    this->pixel_size = wanted_pixel_size;
    Open(Filename, wanted_access_type, wanted_records_per_line);
}

PDB::~PDB( ) {
    Close( );
}

void PDB::Open(wxString Filename, long wanted_access_type, long wanted_records_per_line) {
    access_type      = wanted_access_type;
    records_per_line = wanted_records_per_line;
    text_filename    = Filename;

    if ( access_type == OPEN_TO_READ ) {
        if ( input_file_stream != NULL ) {
            if ( input_file_stream->GetFile( )->IsOpened( ) == true ) {
                MyPrintWithDetails("File already Open\n");
                DEBUG_ABORT;
            }
        }
    }
    else {
        if ( access_type == OPEN_TO_WRITE ) {
            records_per_line = wanted_records_per_line;
            if ( records_per_line <= 0 ) {
                MyPrintWithDetails("PDB asked to OPEN_TO_WRITE, but with erroneous records per line\n");
                DEBUG_ABORT;
            }

            if ( output_file_stream != NULL ) {
                if ( output_file_stream->GetFile( )->IsOpened( ) == true ) {
                    MyPrintWithDetails("File already Open\n");
                    DEBUG_ABORT;
                }
            }
        }
        else {
            MyPrintWithDetails("Unknown access type!\n");
            DEBUG_ABORT;
        }
    }

    SetEmpty( );
    Init( );
}

void PDB::Close( ) {
    if ( input_text_stream != NULL )
        delete input_text_stream;
    if ( output_text_stream != NULL )
        delete output_text_stream;

    if ( output_file_stream != NULL ) {
        if ( output_file_stream->GetFile( )->IsOpened( ) == true )
            output_file_stream->GetFile( )->Close( );
        delete output_file_stream;
    }

    if ( input_file_stream != NULL ) {
        if ( input_file_stream->GetFile( )->IsOpened( ) == true )
            input_file_stream->GetFile( )->Close( );
        delete input_file_stream;
    }

    input_file_stream  = NULL;
    input_text_stream  = NULL;
    output_file_stream = NULL;
    output_text_stream = NULL;
}

void PDB::SetEmpty( ) {
    this->vol_nX                          = 0;
    this->vol_nY                          = 0;
    this->vol_nZ                          = 0;
    this->vol_oX                          = 0;
    this->vol_oY                          = 0;
    this->vol_oZ                          = 0;
    this->vol_angX                        = 0;
    this->vol_angY                        = 0;
    this->vol_angZ                        = 0;
    this->atomic_volume                   = 0;
    this->pixel_size                      = 0;
    this->number_of_particles_initialized = 0;
    if ( ! use_provided_com ) {
        this->center_of_mass[0] = 0;
        this->center_of_mass[1] = 0;
        this->center_of_mass[2] = 0;
    }
}

void PDB::Init( ) {

    wxString current_line;
    wxString token;
    double   temp_double;
    int      current_records_per_line;
    int      old_records_per_line                      = -1;
    long     current_atom_number                       = 0;
    long     n_atoms_in_single_molecule_from_star_file = 0;
    float    wanted_origin_x;
    float    wanted_origin_y;
    float    wanted_origin_z;
    float    euler1;
    float    euler2;
    float    euler3; // from pdb

    // work out the records per line and how many lines

    this->number_of_lines = 0;
    this->number_of_atoms = 0;

    // After a phenix ADP refinement + Chimera selection and split, all ATOM --> HETATM. Quick hack to set this until I can figure out why. Generally speaking, this should be left as ATOM
    wxString pdb_atom = "ATOM";

    gemmi::Structure st;
    try {
        st = gemmi::read_structure(gemmi::MaybeGzipped(text_filename.ToStdString( )));
        // I'm sure there is already something in GEMMI to do an iteration like this.
        for ( gemmi::Model& model : st.models ) {
            for ( gemmi::Chain& chain : model.chains ) {
                // wxPrintf("Working on chain %s\n",chain.name);
                for ( gemmi::Residue& res : chain.residues ) {
                    // wxPrintf("Residue Name, Segment, Entity type, %s %s\n",res.name,res.segment)
                    for ( gemmi::Atom& atom : res.atoms ) {
                        // For now, we only want ATOM
                        if ( res.het_flag == 'A' ) {
                            number_of_atoms++;
                        }
                    }
                }
            }
        }
    } catch ( std::runtime_error& e ) {
        // It may be nice if this returned and printed in the error dialog invoked when is_valid is false, rather than
        // printing to stdout.
        MyPrintWithDetails("\n\nGEMMI threw an error reading this file:\n %s\n", e.what( ));
        exit(-1);
    }

    // Only those atoms that are part of the target molecule - TODO change the name ... they are all real
    number_of_real_atoms = number_of_atoms;

    // Copy every real atom into a noise atom
    if ( generate_noise_atoms ) {
        number_of_atoms += (number_of_atoms * this->max_number_of_noise_particles);
    }

    // wxPrintf("Max particles is %d here\n", max_number_of_noise_particles);
    // If noie noise atoms, this will always equal number of atoms. Otherwise, number of atoms is at most this, and changes from particle to particle in the stack
    number_of_real_and_noise_atoms = number_of_atoms;

    // wxPrintf("\nIn constructor real total current %ld %ld %ld\n", number_of_real_atoms, number_of_real_and_noise_atoms, number_of_atoms);
    // Create the atom array, then loop back over the pdb to get the desired info.
    atoms.reserve(number_of_real_and_noise_atoms);

    // I'm sure there is already something in GEMMI to do an iteration like this.
    float    i_bfactor;
    AtomType i_atom_type;
    for ( gemmi::Model& model : st.models ) {
        for ( gemmi::Chain& chain : model.chains ) {
            for ( gemmi::Residue& res : chain.residues ) {
                // For now, we only want ATOM
                if ( res.het_flag == 'A' ) { // 'A' = ATOM, 'H' = HETATM, 0 = unspecified
                    for ( gemmi::Atom& atom : res.atoms ) {

                        if ( is_alpha_fold_prediction ) {
                            std::cerr << "Alpha fold prediction not implemented yet" << std::endl;
                            // Convert the confidence score to a bfactor. The formula is adhoc.
                            // Confidence score is 0->100 with higher being more confident.
                            i_bfactor = 4.f + powf(100.f - atom.b_iso, 1.314159f);

                            // atoms[current_atom_number].bfactor = 4.f + powf(100.f - atom.b_iso, 1.314159f);
                            // wxPrintf("Confidence score %f, bfactor %f\n",atom.b_iso,atoms[current_atom_number].bfactor);
                        }
                        else {
                            i_bfactor = atom.b_iso;

                            // atoms[current_atom_number].bfactor = atom.b_iso;
                        }

                        switch ( atom.element.ordinal( ) ) {

                            case 0:
                                MyDebugPrintWithDetails("Error, non-element type, %s\n", atoms[current_atom_number].name);
                                exit(-1);
                                break;
                            case 1:
                                i_atom_type = hydrogen;
                                break;
                            case 6:
                                i_atom_type = carbon;
                                break;
                            case 7:
                                i_atom_type = nitrogen;
                                break;
                            case 8:
                                i_atom_type = oxygen;
                                break;
                            case 9:
                                i_atom_type = fluorine;
                                break;
                            case 11:
                                i_atom_type = sodium;
                                break;
                            case 12:
                                i_atom_type = magnesium;
                                break;
                            case 14:
                                i_atom_type = silicon;
                                break;
                            case 15:
                                i_atom_type = phosphorus;
                                break;
                            case 16:
                                i_atom_type = sulfur;
                                break;
                            case 17:
                                i_atom_type = chlorine;
                                break;
                            case 19:
                                i_atom_type = potassium;
                                break;
                            case 20:
                                i_atom_type = calcium;
                                break;
                            case 25:
                                i_atom_type = manganese;
                                break;
                            case 27:
                                i_atom_type = iron;
                                break;
                            case 28:
                                i_atom_type = cobalt;
                                break;
                            case 30:
                                i_atom_type = zinc;
                                break;
                            case 34:
                                i_atom_type = selenium;
                                break;
                            case 79:
                                i_atom_type = gold;
                                break;
                            default:
                                wxPrintf("Un-coded conversion from gemmi::el to Atom::atom_type\n");
                                std::cerr << "Element is " << atom.element.name( ) << "and el" << atom.element.ordinal( ) << '\n';
                                exit(-1);
                                break;
                        }

                        atoms.emplace_back(wxString(atom.name), true, i_atom_type, float(atom.pos.x), float(atom.pos.y), float(atom.pos.z), atom.occ, i_bfactor, float(atom.charge));
                        current_atom_number++;
                        n_atoms_in_single_molecule_from_star_file++;

                        if ( isnan(atoms.back( ).bfactor) ) {
                            std::cerr << "Atom is " << atoms.back( ).name << " " << current_atom_number << " " << atoms.back( ).x_coordinate << atoms.back( ).atom_type << " " << atoms.back( ).bfactor << '\n';
                            exit(1);
                        }
                    }
                }
            }
        }
    }

    if ( use_star_file && star_file_parameters.ReturnNumberofLines( ) > 1 ) {

        // First we need the average defocus to determine offsets in Z
        // (for underfocus, the focal plane is above the specimen in the scope and the undefocus magnitude increases as we move away (down) the column.)
        star_file_parameters.CalculateDefocusDependence( );
        int current_frame_number = 1;

        // FIXME: there are multiple frames, calling this iParticle doesn't make sense.
        for ( int iParticle = 0; iParticle < star_file_parameters.ReturnNumberofLines( ); iParticle++ ) {
            // If particle group is 0 it is written improperly, but safe to assume there is only one frame per particle.
            // Otherwise, we increment the frame numbers until the number reduces again.
            if ( iParticle > 0 ) {
                if ( star_file_parameters.ReturnParticleGroup(iParticle - 1) != 0 && star_file_parameters.ReturnParticleGroup(iParticle - 1) < current_frame_number ) {
                    current_frame_number++;
                }
                else
                    current_frame_number = 1;
            }
            TransformBaseCoordinates(star_file_parameters.ReturnXShift(iParticle),
                                     star_file_parameters.ReturnYShift(iParticle),
                                     (0.5f * (star_file_parameters.ReturnDefocus1(iParticle) + star_file_parameters.ReturnDefocus2(iParticle))) - star_file_parameters.average_defocus,
                                     -star_file_parameters.ReturnPsi(iParticle),
                                     -star_file_parameters.ReturnTheta(iParticle),
                                     -star_file_parameters.ReturnPhi(iParticle),
                                     iParticle,
                                     current_frame_number - 1);
            // wxPrintf("x,y,z (%f,%f,%f) psi/theta/phi (%f,%f,%f), ipart/frame %d %d\n", star_file_parameters.ReturnXShift(iParticle),
            //                              star_file_parameters.ReturnYShift(iParticle),
            //                              (0.5f * (star_file_parameters.ReturnDefocus1(iParticle)+star_file_parameters.ReturnDefocus2(iParticle))) - star_file_parameters.average_defocus,
            //                              star_file_parameters.ReturnPsi(iParticle),
            //                              star_file_parameters.ReturnTheta(iParticle),
            //                              star_file_parameters.ReturnPhi(iParticle),
            //                              iParticle,
            //                              current_frame_number);
        }
    }
    else {

        // Set up the initial trajectory for this particle instance.
        this->TransformBaseCoordinates(0, 0, 0, 0, 0, 0, 0, 0);
    }
    // Finally, calculate the center of mass of the PDB object if it is not provided and is to be applied.
    if ( ! use_provided_com && shift_by_center_of_mass ) {
        for ( current_atom_number = 0; current_atom_number < number_of_real_atoms; current_atom_number++ ) {
            center_of_mass[0] += atoms[current_atom_number].x_coordinate;
            center_of_mass[1] += atoms[current_atom_number].y_coordinate;
            center_of_mass[2] += atoms[current_atom_number].z_coordinate;
        }

        for ( current_atom_number = 0; current_atom_number < 3; current_atom_number++ ) {
            center_of_mass[current_atom_number] /= number_of_real_atoms;
            if ( std::isnan(center_of_mass[current_atom_number]) ) {
                wxPrintf("NaN in center of mass calc from PDB for coordinate %ld, 0=x,1=y,2=z", current_atom_number);
                throw;
            }
        }
    }

    if ( shift_by_center_of_mass ) {
        // if ( use_provided_com ) {
        //     wxPrintf("\n\nSetting PDB center of mass to that provided %f %f %f (x,y,z Angstrom)\n\n", center_of_mass[0], center_of_mass[1], center_of_mass[2]);
        // }
        // else {
        //     wxPrintf("\n\nPDB center of mass at %f %f %f (x,y,z Angstrom)\n\nSetting origin there.\n\n", center_of_mass[0], center_of_mass[1], center_of_mass[2]);
        // }

        // Set the coordinate origin to the calculated or provided center of mass
        for ( current_atom_number = 0; current_atom_number < number_of_real_atoms; current_atom_number++ ) {
            atoms[current_atom_number].x_coordinate -= center_of_mass[0];
            atoms[current_atom_number].y_coordinate -= center_of_mass[1];
            atoms[current_atom_number].z_coordinate -= center_of_mass[2];
        }
    }

    if ( generate_noise_atoms ) {
        // First get the largest molecular radius of the target molecule (centered)
        for ( current_atom_number = 0; current_atom_number < number_of_real_atoms; current_atom_number++ ) {
            // FIXME number of noise molecules!!
            if ( fabsf(atoms[current_atom_number].x_coordinate) > max_radius )
                max_radius = fabsf(atoms[current_atom_number].x_coordinate);
            if ( fabsf(atoms[current_atom_number].y_coordinate) > max_radius )
                max_radius = fabsf(atoms[current_atom_number].y_coordinate);
            if ( fabsf(atoms[current_atom_number].z_coordinate) > max_radius )
                max_radius = fabsf(atoms[current_atom_number].z_coordinate);
        }
        wxPrintf("Max particles is %d in spot 2\n", max_number_of_noise_particles);

        for ( int iPart = 0; iPart < this->max_number_of_noise_particles; iPart++ ) {
            for ( current_atom_number = 0; current_atom_number < number_of_real_atoms; current_atom_number++ ) {
                atoms[current_atom_number + number_of_real_atoms * (1 + iPart)]                  = atoms[current_atom_number];
                atoms[current_atom_number + number_of_real_atoms * (1 + iPart)].is_real_particle = false;
            }
        }
    }
}

void PDB::Rewind( ) {

    if ( access_type == OPEN_TO_READ ) {
        delete input_file_stream;
        delete input_text_stream;

        input_file_stream = new wxFileInputStream(text_filename);
        input_text_stream = new wxTextInputStream(*input_file_stream);
    }
    else
        output_file_stream->GetFile( )->Seek(0);
}

void PDB::Flush( ) {
    if ( access_type == OPEN_TO_READ )
        input_file_stream->GetFile( )->Flush( );
    else
        output_file_stream->GetFile( )->Flush( );
}

void PDB::ReadLine(float* data_array) {
    if ( access_type != OPEN_TO_READ ) {
        MyPrintWithDetails("Attempt to read from %s however access type is not READ\n", text_filename);
        DEBUG_ABORT;
    }

    wxString current_line;
    wxString token;
    double   temp_double;

    while ( input_file_stream->Eof( ) == false ) {
        current_line = input_text_stream->ReadLine( );
        current_line.Trim(false);

        if ( current_line.StartsWith("C") == false && current_line.StartsWith("#") == false && current_line.Length( ) != 0 )
            break;
    }

    wxStringTokenizer tokenizer(current_line);

    for ( int counter = 0; counter < records_per_line; counter++ ) {
        token = tokenizer.GetNextToken( );
        if ( token.ToDouble(&temp_double) == false ) {
            MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", token.ToUTF8( ).data( ), current_line.ToUTF8( ).data( ));
            DEBUG_ABORT;
        }
        else {
            data_array[counter] = temp_double;
        }
    }
}

void PDB::WriteLine(float* data_array) {
    if ( access_type != OPEN_TO_WRITE ) {
        MyPrintWithDetails("Attempt to read from %s however access type is not WRITE\n", text_filename);
        DEBUG_ABORT;
    }

    for ( int counter = 0; counter < records_per_line; counter++ ) {
        output_text_stream->WriteString(wxString::Format("%14.5f", data_array[counter]));
        if ( counter != records_per_line - 1 )
            output_text_stream->WriteString(" ");
    }

    output_text_stream->WriteString("\n");
}

void PDB::WriteLine(double* data_array) {
    if ( access_type != OPEN_TO_WRITE ) {
        MyPrintWithDetails("Attempt to read from %s however access type is not WRITE\n", text_filename);
        DEBUG_ABORT;
    }

    for ( int counter = 0; counter < records_per_line; counter++ ) {
        output_text_stream->WriteDouble(data_array[counter]);
        if ( counter != records_per_line - 1 )
            output_text_stream->WriteString(" ");
    }

    output_text_stream->WriteString("\n");
}

void PDB::WriteCommentLine(const char* format, ...) {
    va_list args;
    va_start(args, format);

    wxString comment_string;
    wxString buffer;

    comment_string.PrintfV(format, args);

    buffer = comment_string;
    buffer.Trim(false);

    if ( buffer.StartsWith("#") == false && buffer.StartsWith("C") == false ) {
        comment_string = "# " + comment_string;
    }

    output_text_stream->WriteString(comment_string);

    if ( comment_string.EndsWith("\n") == false )
        output_text_stream->WriteString("\n");

    va_end(args);
}

wxString PDB::ReturnFilename( ) {
    return text_filename;
}

void PDB::TransformBaseCoordinates(float wanted_origin_x, float wanted_origin_y, float wanted_origin_z, float euler1, float euler2, float euler3, int particle_idx, int frame_number) {
    // Sets the initial position and orientation of the particle (my_ensemble.my_trajectories.Item(0)...) {

    // Initialize a new trajectory which represents an individual instance of a particle
    ParticleTrajectory dummy_trajectory;
    my_trajectory.Add(dummy_trajectory, 1);

    initial_values.emplace_back(euler1, euler2, euler3, wanted_origin_x, wanted_origin_y, wanted_origin_z);

    RotationMatrix rotmat;

    rotmat.SetToRotation(euler1, euler2, euler3);

    // Is it safe to increment like this?
    my_trajectory.Item(particle_idx).current_orientation[frame_number][0]  = wanted_origin_x;
    my_trajectory.Item(particle_idx).current_orientation[frame_number][1]  = wanted_origin_y;
    my_trajectory.Item(particle_idx).current_orientation[frame_number][2]  = wanted_origin_z;
    my_trajectory.Item(particle_idx).current_orientation[frame_number][3]  = rotmat.m[0][0];
    my_trajectory.Item(particle_idx).current_orientation[frame_number][4]  = rotmat.m[1][0];
    my_trajectory.Item(particle_idx).current_orientation[frame_number][5]  = rotmat.m[2][0];
    my_trajectory.Item(particle_idx).current_orientation[frame_number][6]  = rotmat.m[0][1];
    my_trajectory.Item(particle_idx).current_orientation[frame_number][7]  = rotmat.m[1][1];
    my_trajectory.Item(particle_idx).current_orientation[frame_number][8]  = rotmat.m[2][1];
    my_trajectory.Item(particle_idx).current_orientation[frame_number][9]  = rotmat.m[0][2];
    my_trajectory.Item(particle_idx).current_orientation[frame_number][10] = rotmat.m[1][2];
    my_trajectory.Item(particle_idx).current_orientation[frame_number][11] = rotmat.m[2][2];

    // Storing the update for reference. On the intial round this matches the orientation.
    my_trajectory.Item(particle_idx).current_update[frame_number][0] = wanted_origin_x;
    my_trajectory.Item(particle_idx).current_update[frame_number][1] = wanted_origin_y;
    my_trajectory.Item(particle_idx).current_update[frame_number][2] = wanted_origin_z;
    my_trajectory.Item(particle_idx).current_update[frame_number][3] = euler1;
    my_trajectory.Item(particle_idx).current_update[frame_number][4] = euler2;
    my_trajectory.Item(particle_idx).current_update[frame_number][5] = euler3;

    // Now that a new member is added to the ensemble, increment the counter
    this->number_of_particles_initialized++;
    // wxPrintf("\n\nNumber of particles initialized %d\n", this->number_of_particles_initialized);
}

void PDB::TransformLocalAndCombine(PDB& clean_copy, int number_of_pdbs, int frame_number, RotationMatrix particle_rot, float shift_z, bool is_single_particle) {
    clean_copy.TransformLocalAndCombine(this, number_of_pdbs, frame_number, particle_rot, shift_z, is_single_particle);
    return;
}

void PDB::TransformLocalAndCombine(PDB* pdb_ensemble, int number_of_pdbs, int frame_number, RotationMatrix particle_rot, float shift_z, bool is_single_particle) {
    /*
     * Take an array of PDB objects and create a single array of atoms transformed according to the timestep
    */
    // wxPrintf("\n\nTransforming local and combining\n");
    // std::cerr << " My size their size " << atoms.size( ) << " dd " << pdb_ensemble->atoms.size( ) << std::endl;

    int   current_pdb        = 0;
    int   current_particle   = 0;
    int   current_atom       = 0;
    long  current_total_atom = 0;
    float ox, oy, oz; // origin for the current particle
    float ix, iy, iz; // input coords for current atom
    float tx, ty, tz; // transformed coords for current atom

    // Assuming some a distributino around 0,0,0
    this->min_z           = 0;
    this->max_z           = 0;
    float min_x           = 0; // These are then constant, only the z-dimension will change on tilting, so keep them local.
    float min_y           = 0;
    float max_x           = 0;
    float max_y           = 0;
    this->average_bFactor = 0;
    RotationMatrix rotmat;

    for ( int iAtom = 0; iAtom < cistem::number_of_atom_types; iAtom++ ) {
        number_of_each_atom[iAtom] = 0;
    }

    for ( current_pdb = 0; current_pdb < number_of_pdbs; current_pdb++ ) {
        if ( this->atoms.capacity( ) < pdb_ensemble[current_pdb].number_of_atoms ) {
            this->atoms.reserve(pdb_ensemble[current_pdb].number_of_atoms);
        }
        // wxPrintf("Checking %ld %ld\n", pdb_ensemble[current_pdb].atoms.size( ), atoms.size( ));
        for ( current_particle = 0; current_particle < pdb_ensemble[current_pdb].number_of_particles_initialized; current_particle++ ) {
            ox = pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][0];
            oy = pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][1];
            oz = pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][2];
            rotmat.SetToValues(pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][3],
                               pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][4],
                               pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][5],
                               pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][6],
                               pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][7],
                               pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][8],
                               pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][9],
                               pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][10],
                               pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][11]);

            rotmat *= particle_rot;
            // Randomly set noise atoms before copying in. Initially, there is a copy of every real atom, at the same x,y,z coords. For frame zero we
            // want to randomly rotate each noise particle, and offset each particle by a random amount that ensures no overlap with the real particle.
            // The first two noise particles are set to not overlap. Additional particles are able to overlap, which I think is more likely with high crowding (multiple layers of particles.)
            // It might be that local motion should also be applied, but I think it is fair for now that the neighbors move identically frame to frame as the target particles. They should
            // at the minimum be highly correlated based on empirical observation.

            if ( pdb_ensemble[current_pdb].use_star_file ) {
                // Fetch a clean copy of the atomic coordinates for this molecule
                long current_atom = 0;
                for ( long intra_mol_current_atom = 0; intra_mol_current_atom < pdb_ensemble[current_pdb].number_of_atoms; intra_mol_current_atom++ ) {
                    current_atom              = intra_mol_current_atom + pdb_ensemble[current_pdb].number_of_atoms * current_particle;
                    this->atoms[current_atom] = pdb_ensemble[current_pdb].atoms[intra_mol_current_atom];

                    ix = atoms[current_atom].x_coordinate;
                    iy = atoms[current_atom].y_coordinate;
                    iz = atoms[current_atom].z_coordinate;

                    // Transform the coordinates, but FIXME hard coding zero is shifty
                    AnglesAndShifts my_angles(pdb_ensemble[current_pdb].initial_values[current_particle].euler1,
                                              pdb_ensemble[current_pdb].initial_values[current_particle].euler2,
                                              pdb_ensemble[current_pdb].initial_values[current_particle].euler3, 0.f, 0.f);
                    my_angles.euler_matrix.RotateCoords(ix, iy, iz, tx, ty, tz);

                    this->atoms[current_atom].x_coordinate = tx + pdb_ensemble[current_pdb].initial_values[current_particle].ox;
                    this->atoms[current_atom].y_coordinate = ty + pdb_ensemble[current_pdb].initial_values[current_particle].oy;
                    this->atoms[current_atom].z_coordinate = tz + pdb_ensemble[current_pdb].initial_values[current_particle].oz;

                    // minMax X
                    if ( this->atoms[current_atom].x_coordinate < min_x ) {
                        min_x = this->atoms[current_atom].x_coordinate;
                    }
                    if ( this->atoms[current_atom].x_coordinate > max_x ) {
                        max_x = this->atoms[current_atom].x_coordinate;
                    }
                    // minMax Y
                    if ( this->atoms[current_atom].y_coordinate < min_y ) {
                        min_y = this->atoms[current_atom].y_coordinate;
                    }

                    if ( this->atoms[current_atom].y_coordinate > max_y ) {
                        max_y = this->atoms[current_atom].y_coordinate;
                    }
                    // minMax Z
                    if ( this->atoms[current_atom].z_coordinate < this->min_z ) {
                        this->min_z = this->atoms[current_atom].z_coordinate;
                    }

                    if ( this->atoms[current_atom].z_coordinate > this->max_z ) {
                        this->max_z = this->atoms[current_atom].z_coordinate;
                    }

                    this->number_of_each_atom[atoms[current_atom].atom_type]++;
                    average_bFactor += atoms[current_atom].bfactor;
                    current_total_atom++;
                }
            }
            else {
                // TODO: consider using the assignment instead
                // this->atoms = pdb_ensemble[current_pdb].atoms;
                this->atoms.clear( );
                for ( current_atom = 0; current_atom < pdb_ensemble[current_pdb].number_of_atoms; current_atom++ ) {
                    this->atoms.push_back(pdb_ensemble[current_pdb].atoms[current_atom]);
                }

                if ( pdb_ensemble[current_pdb].generate_noise_atoms && frame_number == 0 ) {

                    RandomNumberGenerator my_rand(pi_v<float>);
                    // Set the number of noise particles for this given particle in the stack
                    pdb_ensemble[current_pdb].number_of_noise_particles = my_rand.GetUniformRandomSTD(std::max(0, this->max_number_of_noise_particles - 2), this->max_number_of_noise_particles);
                    wxPrintf("\n\n\tSetting pdb %d to %d noise particles of max %d\n\n", current_pdb, pdb_ensemble[current_pdb].number_of_noise_particles, max_number_of_noise_particles);

                    // Angular sector size such that noise particles do not overlap. This could be a method.
                    float sector_size = 1.1f; //2.0*pi_v<float> / pdb_ensemble[current_pdb].number_of_noise_particles;
                    float occupied_sectors[this->max_number_of_noise_particles];
                    int   current_sector = 0;
                    bool  non_overlaping_particle_found;

                    for ( int iPart = 0; iPart < this->max_number_of_noise_particles; iPart++ ) {

                        non_overlaping_particle_found = false;
                        float offset_radius;
                        // Rather than mess with the allocations, simply send ignored particles off into space.
                        if ( iPart >= pdb_ensemble[current_pdb].number_of_noise_particles ) {
                            offset_radius = 1e8;
                        }
                        else {
                            //                        offset_radius = my_rand.GetNormalRandomSTD(0.0f, 0.1*pdb_ensemble[current_pdb].max_radius);
                            // These numbers (-0.5, 1.8, 0.1 are not at all thought out - please FIXME)
                            offset_radius = my_rand.GetUniformRandomSTD(this->noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius * pdb_ensemble[current_pdb].max_radius,
                                                                        this->noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius * pdb_ensemble[current_pdb].max_radius);
                            offset_radius += this->noise_particle_radius_as_mutliple_of_particle_radius * pdb_ensemble[current_pdb].max_radius;
                        }

                        float offset_angle;
                        // If we have more than one noise particle, enforce non-overlap
                        if ( iPart == 0 ) {
                            offset_angle                  = clamp_angular_range_negative_pi_to_pi(my_rand.GetUniformRandomSTD(-pi_v<float>, pi_v<float>));
                            non_overlaping_particle_found = true;
                            occupied_sectors[0]           = offset_angle;
                        }
                        else {
                            int  max_tries    = 5000;
                            int  iTry         = 0;
                            bool is_too_close = true;
                            while ( is_too_close && iTry < max_tries ) {
                                iTry += 1;
                                offset_angle = clamp_angular_range_negative_pi_to_pi(my_rand.GetUniformRandomSTD(-pi_v<float>, pi_v<float>));
                                float dx     = cosf(offset_angle);
                                float dy     = sinf(offset_angle);
                                float ang_diff;
                                int   jPart;
                                //                            wxPrintf("offset angle is %3.3f\n", rad_2_deg(offset_angle));

                                // now check the angle against each previous
                                for ( jPart = 0; jPart < iPart; jPart++ ) {
                                    ang_diff = cosf(occupied_sectors[jPart]) * dx + sinf(occupied_sectors[jPart]) * dy;
                                    ang_diff = acosf(ang_diff);
                                    if ( fabsf(ang_diff) >= sector_size ) {
                                        // We need to keep track of whether or not a goo dmatch was found
                                        is_too_close = false;
                                    }
                                    else {
                                        // if any are too close, want to break out
                                        is_too_close = true;
                                        break;
                                    }
                                }
                            }
                            if ( is_too_close ) {
                                wxPrintf("Error, did not find a well separated noise particle\n");
                                //                            exit(-1);
                            }
                            else {
                                non_overlaping_particle_found = true;
                                occupied_sectors[iPart]       = offset_angle;
                            }
                        }

                        if ( ! non_overlaping_particle_found ) {
                            // fix double negative
                            offset_radius = 1e8;
                        }

                        float offset_X = offset_radius * cosf(offset_angle) * cosf(deg_2_rad(emulate_tilt_angle));
                        float offset_Y = offset_radius * sinf(offset_angle);
                        //                    RotationMatrix randmat;

                        pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].Init(my_rand.GetUniformRandomSTD(0, 360), my_rand.GetUniformRandomSTD(0, 360), my_rand.GetUniformRandomSTD(0, 360), offset_X, offset_Y);

                        wxPrintf("\nreal total current %ld %ld %ld\n", pdb_ensemble[current_pdb].number_of_real_atoms, pdb_ensemble[current_pdb].number_of_real_and_noise_atoms, pdb_ensemble[current_pdb].number_of_atoms);

                        for ( int current_atom_number = 0; current_atom_number < pdb_ensemble[current_pdb].number_of_real_atoms; current_atom_number++ ) {

                            pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].euler_matrix.RotateCoords(
                                    atoms[current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms * (1 + iPart)].x_coordinate,
                                    atoms[current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms * (1 + iPart)].y_coordinate,
                                    atoms[current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms * (1 + iPart)].z_coordinate,
                                    tx, ty, tz);
                            atoms[current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms * (1 + iPart)].x_coordinate = tx + pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnShiftX( );
                            atoms[current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms * (1 + iPart)].y_coordinate = ty + pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnShiftY( );
                            atoms[current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms * (1 + iPart)].z_coordinate = tz;
                        }

                    } // end of loop over noise particles
                } // if condition on noise particles and frame 0
                else if ( pdb_ensemble[current_pdb].generate_noise_atoms ) {
                    for ( int iPart = 0; iPart < this->max_number_of_noise_particles; iPart++ ) {
                        for ( int current_atom_number = 0; current_atom_number < pdb_ensemble[current_pdb].number_of_real_atoms; current_atom_number++ ) {

                            pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].euler_matrix.RotateCoords(
                                    atoms[current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms * (1 + iPart)].x_coordinate,
                                    atoms[current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms * (1 + iPart)].y_coordinate,
                                    atoms[current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms * (1 + iPart)].z_coordinate,
                                    tx, ty, tz);

                            atoms[current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms * (1 + iPart)].x_coordinate = tx + pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnShiftX( );
                            atoms[current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms * (1 + iPart)].y_coordinate = ty + pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnShiftY( );
                            atoms[current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms * (1 + iPart)].z_coordinate = tz;
                        }
                    }
                }

                for ( current_atom = 0; current_atom < pdb_ensemble[current_pdb].number_of_atoms; current_atom++ ) {
                    if ( atoms[current_atom].is_real_particle ) {

                        ix = atoms[current_atom].x_coordinate;
                        iy = atoms[current_atom].y_coordinate;
                        iz = atoms[current_atom].z_coordinate;

                        rotmat.RotateCoords(ix, iy, iz, tx, ty, tz); // Why can't I just put the shift operation above inline to the function?
                        // Update the specimen with the transformed coords

                        this->atoms[current_atom].x_coordinate = tx + ox;
                        this->atoms[current_atom].y_coordinate = ty + oy;
                        this->atoms[current_atom].z_coordinate = tz + oz;

                        // minMax X
                        if ( this->atoms[current_atom].x_coordinate < min_x ) {
                            min_x = this->atoms[current_atom].x_coordinate;
                        }
                        if ( this->atoms[current_atom].x_coordinate > max_x ) {
                            max_x = this->atoms[current_atom].x_coordinate;
                        }
                        // minMax Y
                        if ( this->atoms[current_atom].y_coordinate < min_y ) {
                            min_y = this->atoms[current_atom].y_coordinate;
                        }

                        if ( this->atoms[current_atom].y_coordinate > max_y ) {
                            max_y = this->atoms[current_atom].y_coordinate;
                        }
                        // minMax Z
                        if ( this->atoms[current_atom].z_coordinate < this->min_z ) {
                            this->min_z = this->atoms[current_atom].z_coordinate;
                        }

                        if ( this->atoms[current_atom].z_coordinate > this->max_z ) {
                            this->max_z = this->atoms[current_atom].z_coordinate;
                        }

                        this->number_of_each_atom[pdb_ensemble[current_pdb].atoms[current_atom].atom_type]++;
                        average_bFactor += atoms[current_atom].bfactor;

                        current_total_atom++;
                    } // if on real particles
                }
            }
        } // end of the loop on particles
    }

    // This is used in the simulator to determine how large a window should be used for the calculation of the atoms.
    if ( current_total_atom > 0 ) {
        average_bFactor /= current_total_atom;
    }
    if ( isnan(average_bFactor) ) {
        wxPrintf("\n\n\t\tWARNING: average_bFactor is nan setting to zero, this should be fixed, not ignored!\n");
        average_bFactor = 0;
    }
    // wxPrintf("\t\t\n\nAVG BFACTOR FROM PDB IS %f, current_total_atom %ld\n\n", average_bFactor, current_total_atom);

    if ( current_total_atom > 2 ) { // for single atom test
        // Again, need a check to make sure all sizes are consistent
        if ( max_x - min_x <= 0 ) {
            MyPrintWithDetails("The measured X dimension is invalid max - min = X, %f - %f = %f\n", max_x, min_x, max_x - min_x);
            DEBUG_ABORT;
        }
        if ( max_y - min_y <= 0 ) {
            MyPrintWithDetails("The measured Y dimension is invalid max - min = Y, %f - %f = %f\n", max_y, min_y, max_y - min_y);
            DEBUG_ABORT;
        }
        // Allow for a perfectly planar layer of atoms, for testing images. Notice == not a more typical diff < epsilon to make it more likely this only is allowed when intended (dev).
        if ( max_z - min_z == 0 ) {
            max_z += 1;
        }
        if ( this->max_z - this->min_z <= 0 ) {
            MyPrintWithDetails("The measured Z dimension is invalid max - min = Z, %f - %f = %f\n", this->max_z, this->min_z, this->max_z - this->min_z);
            DEBUG_ABORT;
        }
    }

    this->vol_angX = max_x - min_x + MIN_PADDING_XY;
    this->vol_angY = max_y - min_y + MIN_PADDING_XY;

    float max_depth = 0.0f;
    if ( is_single_particle ) {
        // Keep the thickness of the ice mostly constant, which may not happen if the particle is non globular and randomly oriented.
        max_depth = std::max(max_x - min_x, std::max(max_y - min_y, fabsf(max_z - min_z)));
    }
    else {
        max_depth = max_z - min_z;
    }
    this->vol_angZ = std::max(MIN_THICKNESS, (2 * (MIN_PADDING_Z + fabsf(shift_z)) + (MIN_PADDING_Z + fabsf(max_depth)))); // take the larger of 20 nm + range and 1.5x the specimen diameter. Look closer at Nobles paper.
    this->vol_angZ /= cosf(deg_2_rad(emulate_tilt_angle));

    if ( this->cubic_size > 1 ) {
        // Override the dimensions
        // wxPrintf("Cubic size is %d\n", this->cubic_size);
        this->vol_nX = cubic_size;
        this->vol_nY = cubic_size;
        this->vol_nZ = cubic_size;
    }
    else {
        // wxPrintf("Vol ang z is %f\n", this->vol_angZ);
        this->vol_nX = myroundint(this->vol_angX / pixel_size);
        this->vol_nY = myroundint(this->vol_angY / pixel_size);
        this->vol_nZ = myroundint(this->vol_angZ / pixel_size);
        if ( IsEven(this->vol_nZ) == false )
            this->vol_nZ += 1;
    }
}

void PDB::TransformGlobalAndSortOnZ(long number_of_non_water_atoms, float shift_x, float shift_y, float shift_z, RotationMatrix rotmat) {

    long  current_atom;
    float tx, ty, tz; // transformed coords for current atom

    std::cerr << "Transforming and sorting on Z\n";
    std::cerr << "Shift x, y, z: " << shift_x << ", " << shift_y << ", " << shift_z << "\n";
    for ( current_atom = 0; current_atom < number_of_non_water_atoms; current_atom++ ) {

        rotmat.RotateCoords(atoms[current_atom].x_coordinate, atoms[current_atom].y_coordinate, atoms[current_atom].z_coordinate, tx, ty, tz);

        atoms[current_atom].x_coordinate = tx + shift_x;
        atoms[current_atom].y_coordinate = ty + shift_y;
        atoms[current_atom].z_coordinate = tz + shift_z;
    }

    // I am not sure if putting the conditionals in the last loop would prevent vectorization, so this would be good to look at.

    for ( current_atom = 0; current_atom < number_of_non_water_atoms; current_atom++ ) {
        if ( atoms[current_atom].z_coordinate < this->min_z ) {
            min_z = atoms[current_atom].z_coordinate;
        }

        if ( atoms[current_atom].z_coordinate > this->max_z ) {
            max_z = atoms[current_atom].z_coordinate;
        }
    }

    return;
}
