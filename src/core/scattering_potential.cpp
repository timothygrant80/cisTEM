/*
 * scattering_potential.cpp

 *
 *  Created on: Oct 3, 2019
 *      Author: himesb
 */
#include "core_headers.h"
#include "scattering_potential.h"
#include "water.h"

ScatteringPotential::ScatteringPotential( ) {
    SetDefaultValues( );
}

ScatteringPotential::~ScatteringPotential( ) {
}

ScatteringPotential::ScatteringPotential(const wxString& filename, int wanted_cubic_size) {
    SetDefaultValues( );
    _cubic_size = wanted_cubic_size;
    pdb_file_names.push_back(filename);
}

void ScatteringPotential::SetDefaultValues( ) {
    pdb_file_names.reserve(16);
    pdb_ensemble.reserve(16);
    _wavelength                           = 0.0;
    _lead_term                            = 0.0;
    _pixel_size                           = 0.0;
    _minimum_bfactor_applied_to_all_atoms = 0.0;
    _cubic_size                           = 0;
    _minimum_thickness_z                  = 0;
    _bfactor_scaling                      = 1.f;
    _number_of_non_water_atoms            = 0;
    _padding                              = 1;
}

void ScatteringPotential::InitPdbObject(const wxString& filename, int wanted_cubic_size, bool is_alpha_fold_prediction, double* center_of_mass) {
    MyDebugAssertFalse(pdb_file_names.size( ) > 0, "You can only call this function once");
    _cubic_size = wanted_cubic_size;
    pdb_file_names.push_back(filename);
    InitPdbObject(is_alpha_fold_prediction, center_of_mass);
}

void ScatteringPotential::InitPdbObject(bool is_alpha_fold_prediction, double* center_of_mass) {

    MyDebugAssertTrue(_cubic_size > 0, "You must set the cubic size before calling this function");
    MyDebugAssertTrue(_pixel_size > 0.0, "Pixel size not set");

    // backwards compatible with tigress where everything is double (ints would make more sense here.)
    long access_type_read = 0;
    long records_per_line = 1;

    float minimum_thickness_z = _cubic_size * _pixel_size;

    // Initialize each of the PDB objects, this reads in and centers each PDB, but does not make any copies (instances) of the trajectories.
    constexpr int minimum_padding_x_and_y = 0;

    pdb_ensemble.emplace_back(pdb_file_names[0],
                              access_type_read,
                              _pixel_size,
                              records_per_line,
                              minimum_padding_x_and_y,
                              minimum_thickness_z,
                              is_alpha_fold_prediction,
                              center_of_mass);
}

void ScatteringPotential::InitPdbEnsemble(bool              shift_by_center_of_mass,
                                          int               minimum_padding_x_and_y,
                                          int               minimum_thickness_z,
                                          int               max_number_of_noise_particles,
                                          float             wanted_noise_particle_radius_as_mutliple_of_particle_radius,
                                          float             wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
                                          float             wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
                                          float             wanted_tilt_angle_to_emulate,
                                          bool              is_alpha_fold_prediction,
                                          cisTEMParameters& wanted_star_file,
                                          bool              use_star_file) {

    // backwards compatible with tigress where everything is double (ints would make more sense here.)
    long access_type_read = 0;
    long records_per_line = 1;

    MyDebugAssertTrue(_pixel_size > 0.0, "Pixel size not set");

    // Initialize each of the PDB objects, this reads in and centers each PDB, but does not make any copies (instances) of the trajectories.

    for ( int iPDB = 0; iPDB < pdb_file_names.size( ); iPDB++ ) {

        pdb_ensemble.emplace_back(pdb_file_names[iPDB],
                                  access_type_read,
                                  _pixel_size,
                                  records_per_line,
                                  minimum_padding_x_and_y,
                                  minimum_thickness_z,
                                  max_number_of_noise_particles,
                                  wanted_noise_particle_radius_as_mutliple_of_particle_radius,
                                  wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
                                  wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
                                  wanted_tilt_angle_to_emulate,
                                  shift_by_center_of_mass,
                                  is_alpha_fold_prediction,
                                  wanted_star_file,
                                  use_star_file);
    }
}

long ScatteringPotential::ReturnTotalNumberOfNonWaterAtoms( ) {

    if ( _number_of_non_water_atoms == 0 ) {
        // Get a count of the total non water atoms
        for ( int iPDB = 0; iPDB < pdb_ensemble.size( ); iPDB++ ) {
            _number_of_non_water_atoms += (pdb_ensemble[iPDB].number_of_real_and_noise_atoms * pdb_ensemble[iPDB].number_of_particles_initialized);
        }
    }

    return _number_of_non_water_atoms;
}

void ScatteringPotential::calc_scattering_potential(Image&         image_vol,
                                                    RotationMatrix rotate_waters,
                                                    int            number_of_threads,
                                                    float dx, float dy, float dz) {

    MyDebugAssertFalse(pdb_ensemble.size( ) == 0, "You must call InitPdbObject before calling this function");
    MyDebugAssertTrue(image_vol.is_in_memory, "Image must be in memory");
    MyDebugAssertTrue(image_vol.IsCubic( ), "Image must be cubic");
    MyDebugAssertTrue(image_vol.logical_x_dimension == _cubic_size, "Image must be cubic and match the set dimension from the constructor");

    constexpr float non_water_inelastic_scaling                    = 1.0;
    constexpr bool  tilted_scattering_potential_for_full_beam_tilt = false;
    constexpr float beam_tilt_z_X_component                        = 0.0;

    // The book-keeping in the simulator is quite complicated, but we use the coords object here for parity.
    Coords coords;
    coords.SetSpecimenVolume(_cubic_size, _cubic_size, _cubic_size);
    coords.SetSolventPadding(_cubic_size, _cubic_size, _cubic_size);
    // FIXME: these (and equivalent in simulate should be set in constants.h)
    coords.SetFFTPadding(5, 1);

    coords.SetSolventPadding_Z(_cubic_size);

    image_vol.SetToConstant(0.f);

    int slabIDX_start = 0;
    int slabIDX_end   = _cubic_size - 1;

    PDB clean_copy = pdb_ensemble[0];

    pdb_ensemble[0].TransformLocalAndCombine(clean_copy, 1, 0, rotate_waters, 0.f, true);

    calc_scattering_potential(&clean_copy,
                              coords,
                              &image_vol,
                              nullptr,
                              nullptr,
                              floorf(_cubic_size / 2),
                              &slabIDX_start,
                              &slabIDX_end,
                              0,
                              GetNeighborhoodSize( ),
                              number_of_threads,
                              non_water_inelastic_scaling,
                              tilted_scattering_potential_for_full_beam_tilt,
                              beam_tilt_z_X_component,
                              beam_tilt_z_X_component,
                              dx, dy, dz);
}

// Called from simulate.cpp
void ScatteringPotential::calc_scattering_potential(const PDB* current_specimen,
                                                    Coords&    coords,
                                                    Image*     scattering_slab,
                                                    Image*     inelastic_slab,
                                                    Image*     distance_slab,
                                                    float      rotated_oZ,
                                                    int*       slabIDX_start,
                                                    int*       slabIDX_end,
                                                    int        iSlab,
                                                    int        size_neighborhood,
                                                    int        number_of_threads,
                                                    float      non_water_inelastic_scaling,
                                                    bool       tilted_scattering_potential_for_full_beam_tilt,
                                                    float      beam_tilt_z_X_component,
                                                    float      beam_tilt_z_Y_component,
                                                    float ddx, float ddy, float ddz) {
    MyDebugAssertTrue(_pixel_size > 0.0, "Pixel size not set");

    int z_low = slabIDX_start[iSlab] - size_neighborhood;
    int z_top = slabIDX_end[iSlab] + size_neighborhood;

    float slab_half_thickness_angstrom = (slabIDX_end[iSlab] - slabIDX_start[iSlab] + 1) * _pixel_size / 2.0f;

    // Private
    AtomType atom_id;
    float    element_inelastic_ratio;
    float    bFactor;
    float    radius;
    float    ix(0), iy(0), iz(0);
    float    dx(0), dy(0), dz(0);
    float    x1(0.0f), y1(0.0f), z1(0.0f);
    float    x2(0.0f), y2(0.0f), z2(0.0f);
    int      indX(0), indY(0), indZ(0);
    float    sx(0), sy(0), sz(0);
    float    xDistSq(0), zDistSq(0), yDistSq(0);
    int      iLim, jLim, kLim;
    int      iGaussian;
    float    water_offset;
    int      cubic_vol = (int)powf(size_neighborhood * 2 + 1, 3);
    long     atoms_added_idx[cubic_vol];
    float    atoms_values_tmp[cubic_vol];
    float    atoms_distances_tmp[cubic_vol];

    int   n_atoms_added;
    float bfX(0), bfY(0), bfZ(0);

    float bPlusB[5];
    // TODO experiment with the scheduling. Until the specimen is consistently full, many consecutive slabs may have very little work for the assigned threads to handle.

#pragma omp parallel for num_threads(number_of_threads) private(                                                       \
        atom_id, bFactor, bPlusB, radius, ix, iy, iz, x1, x2, y1, y2, z1, z2, indX, indY,                              \
        indZ, sx, sy, sz, dx, dy, dz, xDistSq, yDistSq, zDistSq, iLim, jLim, kLim, iGaussian, element_inelastic_ratio, \
        water_offset, atoms_values_tmp, atoms_added_idx, atoms_distances_tmp, n_atoms_added, bfX, bfY, bfZ)
    for ( long current_atom = 0; current_atom < ReturnTotalNumberOfNonWaterAtoms( ); current_atom++ ) {
        n_atoms_added = 0;

        atom_id = current_specimen->atoms.at(current_atom).atom_type;
        if ( atom_id == hydrogen )
            continue;

        element_inelastic_ratio = sqrtf(non_water_inelastic_scaling / ReturnAtomicNumber(atom_id)); // Reimer/Ross_Messemer 1989
        bFactor                 = GetCompleteBfactor(current_specimen->atoms.at(current_atom).bfactor);

        corners R;
        //        Coords coords;

        float3 origin = coords.ReturnOrigin((PaddingStatus)solvent);
        int3   size   = coords.GetSolventPadding( );

        if ( tilted_scattering_potential_for_full_beam_tilt ) {
            // Shift atoms positions in X/Y so that they end up being projected at the correct position at the BOTTOM of the slab
            // TODO save some comp and pre calc the factors on the right

            x1 = current_specimen->atoms.at(current_atom).x_coordinate;
            y1 = current_specimen->atoms.at(current_atom).y_coordinate;
            z1 = current_specimen->atoms.at(current_atom).z_coordinate + ((rotated_oZ)-slabIDX_start[iSlab]) * _pixel_size;

            x1 += (z1)*beam_tilt_z_X_component;
            y1 += (z1)*beam_tilt_z_Y_component;

            // Convert atom origin to pixels and shift by volume origin to get pixel coordinates. Add 0.5 to place origin at "center" of voxel
            dx = modff(origin.x + ddx + (x1 / _pixel_size) + cistem::atomic_to_pixel_offset, &ix);
            dy = modff(origin.y + ddy + (y1 / _pixel_size) + cistem::atomic_to_pixel_offset, &iy);
            // Notes this is the unmodified dz (not using z1)

            dz = modff((rotated_oZ) + ddz + ((float)current_specimen->atoms.at(current_atom).z_coordinate / _pixel_size) + cistem::atomic_to_pixel_offset, &iz);
        }
        else {

            // Convert atom origin to pixels and shift by volume origin to get pixel coordinates. Add 0.5 to place origin at "center" of voxel
            dx = modff(origin.x + ddx + (current_specimen->atoms.at(current_atom).x_coordinate / _pixel_size) + cistem::atomic_to_pixel_offset, &ix);
            dy = modff(origin.y + ddy + (current_specimen->atoms.at(current_atom).y_coordinate / _pixel_size) + cistem::atomic_to_pixel_offset, &iy);
            dz = modff((rotated_oZ) + ddz + (current_specimen->atoms.at(current_atom).z_coordinate / _pixel_size) + cistem::atomic_to_pixel_offset, &iz);
        }

        // With the correct pixel indices in ix,iy,iz now subtract off the 0.5
        dx -= cistem::atomic_to_pixel_offset;
        dy -= cistem::atomic_to_pixel_offset;
        dz -= cistem::atomic_to_pixel_offset;

#pragma omp simd
        for ( iGaussian = 0; iGaussian < 5; iGaussian++ ) {
            bPlusB[iGaussian] = 2 * pi_v<float> / sqrt(bFactor + ReturnScatteringParamtersB(atom_id, iGaussian));
        }

        // For accurate calculations, a thin slab is used, s.t. those atoms outside are the majority. Check this first, but account for the size of the atom, as it may reside in more than one slab.
        //        if (iz <= slabIDX_end[iSlab]  && iz >= slabIDX_start[iSlab])
        if ( iz <= z_top && iz >= z_low ) {

            for ( sx = -size_neighborhood; sx <= size_neighborhood; sx++ ) {
                indX    = ix + sx;
                R.x1    = (sx - cistem::atomic_to_pixel_offset - dx) * this->_pixel_size;
                R.x2    = (R.x1 + this->_pixel_size);
                xDistSq = (sx - dx) * _pixel_size;
                xDistSq *= xDistSq;

                for ( sy = -size_neighborhood; sy <= size_neighborhood; sy++ ) {
                    indY    = iy + sy;
                    R.y1    = (sy - cistem::atomic_to_pixel_offset - dy) * this->_pixel_size;
                    R.y2    = (R.y1 + this->_pixel_size);
                    yDistSq = (sy - dy) * _pixel_size;
                    yDistSq *= yDistSq;

                    for ( sz = -size_neighborhood; sz <= size_neighborhood; sz++ ) {
                        indZ    = iz + sz;
                        R.z1    = (sz - cistem::atomic_to_pixel_offset - dz) * this->_pixel_size;
                        R.z2    = (R.z1 + this->_pixel_size);
                        zDistSq = (sz - dz) * _pixel_size;
                        zDistSq *= zDistSq;
                        // Put Z condition first since it should fail most often (does c++ fall out?)

                        if ( indZ <= slabIDX_end[iSlab] && indZ >= slabIDX_start[iSlab] && indX > 0 && indY > 0 && indX < size.x && indY < size.y ) {
                            // Calculate the scattering potential

                            atoms_added_idx[n_atoms_added]     = scattering_slab->ReturnReal1DAddressFromPhysicalCoord(indX, indY, indZ - slabIDX_start[iSlab]);
                            atoms_values_tmp[n_atoms_added]    = ReturnScatteringPotentialOfAVoxel(R, bPlusB, atom_id);
                            atoms_distances_tmp[n_atoms_added] = xDistSq + yDistSq + zDistSq;

                            //                            scattering_slab->real_values[atoms_added_idx[n_atoms_added]] += temp_potential;
                            n_atoms_added++;
                        }

                    } // end of loop over the neighborhood Z
                } // end of loop over the neighborhood Y
            } // end of loop over the neighborhood X

            //        wxPrintf("Possible positions added %3.3e %\n", 100.0f* (float)n_atoms_added/(float)cubic_vol);
#pragma omp critical
            for ( int iIDX = 0; iIDX < n_atoms_added - 1; iIDX++ ) {
                //                #pragma omp atomic update
                scattering_slab->real_values[atoms_added_idx[iIDX]] += (atoms_values_tmp[iIDX]);

                if ( inelastic_slab )
                    inelastic_slab->real_values[atoms_added_idx[iIDX]] += element_inelastic_ratio * atoms_values_tmp[iIDX];

                if ( distance_slab )
                    distance_slab->real_values[atoms_added_idx[iIDX]] = std::min(distance_slab->real_values[atoms_added_idx[iIDX]], atoms_distances_tmp[iIDX]);
            }

        } // if statment into neigh

    } // end loop over atoms
}

int ScatteringPotential::GetNeighborhoodSize( ) {
    MyDebugAssertTrue(pdb_ensemble.size( ) == 1, "This function is only valid for a single specimen");
    return GetNeighborhoodSize(_minimum_bfactor_applied_to_all_atoms + pdb_ensemble[0].average_bFactor);
}

int ScatteringPotential::GetNeighborhoodSize(float wanted_bfactor) {
    MyDebugAssertTrue(_pixel_size > 0, "Pixel size is not set");

    return 1 + myroundint((0.4f * sqrtf(0.6f * wanted_bfactor) + 0.2f) / _pixel_size);
}