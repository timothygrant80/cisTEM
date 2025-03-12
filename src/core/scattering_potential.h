/*
 * scattering_potential.h
 *
 *  Created on: Oct 3, 2019
 *      Author: himesb
 */

#ifndef _SRC_PROGRAMS_SIMULATE_SCATTERING_POTENTIAL_H_
#define _SRC_PROGRAMS_SIMULATE_SCATTERING_POTENTIAL_H_

#include "../constants/constants.h"

// TODO: x2 = x1 + pixel size, so it might make more sense to limit memory and just store x1,y1,z1 and pixel size.
typedef struct _corners {
    float x1;
    float x2;
    float y1;
    float y2;
    float z1;
    float z2;
} corners;

class ScatteringPotential {

  public:
    ScatteringPotential( );
    ScatteringPotential(const wxString& filename, int wanted_cubic_size);
    virtual ~ScatteringPotential( );

    void SetDefaultValues( );

    std::vector<PDB>      pdb_ensemble;
    std::vector<wxString> pdb_file_names;

    // When simulating a simple 3d density, we only need a single pdb object and not a full ensemble
    void InitPdbObject(bool is_alpha_fold_prediction, bool use_hetatms, double* center_of_mass = nullptr);
    void InitPdbObject(wxString const& filename, int wanted_cubic_size, bool is_alpha_fold_prediction, bool use_hetatms, double* center_of_mass = nullptr);

    void InitPdbEnsemble(bool              shift_by_center_of_mass,
                         int               minimum_padding_x_and_y,
                         int               minimum_thickness_z,
                         int               max_number_of_noise_particles,
                         float             wanted_noise_particle_radius_as_mutliple_of_particle_radius,
                         float             wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
                         float             wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
                         float             wanted_tilt_angle_to_emulat,
                         bool              is_alpha_fold_prediction,
                         bool              allow_hetatms,
                         cisTEMParameters& wanted_star_file,
                         bool              use_star_file);

    long ReturnTotalNumberOfNonSolventAtoms( );

    static inline float ReturnScatteringParamtersA(AtomType id, int term_number) { return SCATTERING_PARAMETERS_A[id][term_number]; };

    static inline float ReturnScatteringParamtersB(AtomType id, int term_number) { return SCATTERING_PARAMETERS_B[id][term_number]; };

    static inline float ReturnAtomicNumber(AtomType id) { return ATOMIC_NUMBER[id]; };

    void SetImagingParameters(float wanted_pixel_size, float wanted_kV, float wanted_scaling = cistem::bond_scaling) {
        _pixel_size = wanted_pixel_size;
        _wavelength = ReturnWavelenthInAngstroms(wanted_kV);
        _lead_term  = wanted_scaling * _wavelength / 8.0f / _pixel_size / _pixel_size;
    };

    void SetMinimumBfactorAppliedToAllAtoms(float wanted_minimum_bfactor) { _minimum_bfactor_applied_to_all_atoms = wanted_minimum_bfactor; };

    void SetBfactorScaling(float wanted_bfactor_scaling) { _bfactor_scaling = wanted_bfactor_scaling; };

    inline float GetCompleteBfactor(float pdb_bfactor) {
        return 0.25f * (pdb_bfactor * _bfactor_scaling + _minimum_bfactor_applied_to_all_atoms);
    }

    int GetNeighborhoodSize( );
    int GetNeighborhoodSize(float wanted_bfactor);

    inline void SetUseHydrogens(float use_hydrogens) { _use_hydrogens = use_hydrogens; };

    inline float lead_term( ) const { return _lead_term; };

    inline float ReturnScatteringPotentialOfAVoxel(corners& R, float* bPlusB, const AtomType atom_id) {

        MyDebugAssertTrue(_wavelength > 0.0, "Wavelength not set");
        MyDebugAssertTrue(_lead_term > 0.0, "Wavelength not set");
        float temp_potential = 0.0f;
        float t0;
        bool  t1, t2, t3;

        // if product < 0, we need to sum the two independent terms, otherwise we want the difference.
        t1 = R.x1 * R.x2 < 0 ? false : true;
        t2 = R.y1 * R.y2 < 0 ? false : true;
        t3 = R.z1 * R.z2 < 0 ? false : true;

        for ( int iGaussian = 0; iGaussian < 5; iGaussian++ ) {

            t0 = (t1) ? erff(bPlusB[iGaussian] * R.x2) - erff(bPlusB[iGaussian] * R.x1) : fabsf(erff(bPlusB[iGaussian] * R.x2)) + fabsf(erff(bPlusB[iGaussian] * R.x1));
            t0 *= (t2) ? erff(bPlusB[iGaussian] * R.y2) - erff(bPlusB[iGaussian] * R.y1) : fabsf(erff(bPlusB[iGaussian] * R.y2)) + fabsf(erff(bPlusB[iGaussian] * R.y1));
            t0 *= (t3) ? erff(bPlusB[iGaussian] * R.z2) - erff(bPlusB[iGaussian] * R.z1) : fabsf(erff(bPlusB[iGaussian] * R.z2)) + fabsf(erff(bPlusB[iGaussian] * R.z1));

            temp_potential += ReturnScatteringParamtersA(atom_id, iGaussian) * fabsf(t0);

        } // loop over gaussian fits
        return temp_potential *= _lead_term;
    };

    void SetVolumeSize(int wanted_volume_dimension) {
        _cubic_size = wanted_volume_dimension;
    };

    void SetVolumePadding(int wanted_padding) {
        _padding = wanted_padding;
    };

    // Called for standalon 3d density simulations
    void calc_scattering_potential(Image&         image_vol,
                                   RotationMatrix rotate_waters,
                                   int            number_of_threads,
                                   float dx = 0.f, float dy = 0.f, float dz = 0.f);
    // Called for more complicated full simulations
    void calc_scattering_potential(const PDB* current_specimen,
                                   Coords&    coords,
                                   Image*     scattering_slab,
                                   Image*     inelastic_slab,
                                   Image*     distance_slab,
                                   float      rotated_oZ,
                                   int*       slabIDX_start,
                                   int*       slabIDX_end,
                                   int        iSlab,
                                   int        size_neighborhood,
                                   int        wanted_number_of_threads,
                                   float      non_water_inelastic_scaling,
                                   bool       tilted_scattering_potential_for_full_beam_tilt,
                                   float      beam_tilt_z_X_component,
                                   float      beam_tilt_z_Y_component,
                                   float dx = 0.f, float dy = 0.f, float dz = 0.f);

  private:
    float _lead_term;
    float _wavelength;
    float _pixel_size;
    float _minimum_bfactor_applied_to_all_atoms;
    float _bfactor_scaling;
    int   _cubic_size;
    float _minimum_thickness_z;
    long  _number_of_non_solvent_atoms;
    int   _padding;
    float _use_hydrogens;
    PDB*  _current_specimen;
};

#endif /* PROGRAMS_SIMULATE_SCATTERING_POTENTIAL_H_ */
