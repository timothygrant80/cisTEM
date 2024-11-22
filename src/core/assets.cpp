#include "core_headers.h"

#include "../../include/gemmi/model.hpp"
#include "../../include/gemmi/mmread.hpp"
#include "../../include/gemmi/gz.hpp"
#include "../../include/gemmi/calculate.hpp"

AssetList::AssetList( ) {
}

AssetList::~AssetList( ) {
}

Asset::Asset( ) {
}

Asset::~Asset( ) {
}

wxString Asset::ReturnFullPathString( ) {
    return filename.GetFullPath( );
}

wxString Asset::ReturnShortNameString( ) {
    return filename.GetFullName( );
}

MovieAsset::MovieAsset( ) {
    asset_id             = -1;
    parent_id            = -1;
    position_in_stack    = 1;
    number_of_frames     = 0;
    eer_frames_per_image = 0;
    eer_super_res_factor = 1;
    x_size               = 0;
    y_size               = 0;
    pixel_size           = 0;
    microscope_voltage   = 0;
    spherical_aberration = 0;
    dose_per_frame       = 0;
    total_dose           = 0;

    filename   = wxEmptyString;
    asset_name = wxEmptyString;

    gain_filename = wxEmptyString;
    dark_filename = wxEmptyString;

    output_binning_factor = 1;

    is_valid = false;

    correct_mag_distortion     = false;
    mag_distortion_angle       = 0.0;
    mag_distortion_major_scale = 1.0;
    mag_distortion_minor_scale = 1.0;

    protein_is_white = false;
}

MovieAsset::~MovieAsset( ) {
    //Don't have to do anything for now
}

MovieAsset::MovieAsset(wxString wanted_filename) {
    filename          = wanted_filename;
    asset_name        = wanted_filename;
    asset_id          = -1;
    position_in_stack = 1;

    number_of_frames     = 0;
    eer_frames_per_image = 0;
    eer_super_res_factor = 1;
    x_size               = 0;
    y_size               = 0;
    pixel_size           = 0;
    microscope_voltage   = 0;
    dose_per_frame       = 0;
    spherical_aberration = 0;
    total_dose           = 0;

    gain_filename = wxEmptyString;
    dark_filename = wxEmptyString;

    output_binning_factor = 1;

    correct_mag_distortion     = false;
    mag_distortion_angle       = 0.0;
    mag_distortion_major_scale = 1.0;
    mag_distortion_minor_scale = 1.0;

    protein_is_white = false;

    Update(wanted_filename); // this checks filename is OK, reads dimensions from headers
}

/*
void MovieAsset::Recheck_if_valid()
{	
	if (filename.IsOk() == true && filename.FileExists() == true)
	{
		movie_is_valid = GetMRCDetails(filename.GetFullPath().fn_str(), x_size, y_size, number_of_frames);
	}

}*/

/*
 * If you think you already know the number of frames, give it as assume_number_of_images.
 * In that case, we will only check the first frame, and assume all the others are there
 * and of the correct size. This will save a lot of time, but it is risky because
 * we may not notice that the file is corrupt or has unusual dimensions.
 * If not, just don't give assume_number_of_images, or set it to 0.
 */
void MovieAsset::Update(wxString wanted_filename, int assume_number_of_frames) {
    filename = wanted_filename;
    is_valid = false;

    if ( filename.IsOk( ) == true && filename.FileExists( ) == true ) {
        if ( filename.GetExt( ).IsSameAs("mrc", false) || filename.GetExt( ).IsSameAs("mrcs", false) ) {
            is_valid = GetMRCDetails(filename.GetFullPath( ).fn_str( ), x_size, y_size, number_of_frames);
        }
        else if ( filename.GetExt( ).IsSameAs("tif", false) || filename.GetExt( ).IsSameAs("tiff", false) ) {
            TiffFile temp_tif;
            is_valid         = temp_tif.OpenFile(filename.GetFullPath( ).ToStdString( ), false, false, assume_number_of_frames);
            x_size           = temp_tif.ReturnXSize( );
            y_size           = temp_tif.ReturnYSize( );
            number_of_frames = temp_tif.ReturnNumberOfSlices( );
            temp_tif.CloseFile( );
        }
        else if ( filename.GetExt( ).IsSameAs("eer", false) ) {
            EerFile temp_eer;
            is_valid         = temp_eer.OpenFile(filename.GetFullPath( ).ToStdString( ), false, false, assume_number_of_frames, eer_super_res_factor, eer_frames_per_image);
            x_size           = temp_eer.ReturnXSize( );
            y_size           = temp_eer.ReturnYSize( );
            number_of_frames = temp_eer.ReturnNumberOfSlices( );
            temp_eer.CloseFile( );
        }
        else {
            is_valid = false;
        }
    }
}

void MovieAsset::CopyFrom(Asset* other_asset) {
    MovieAsset* casted_asset   = reinterpret_cast<MovieAsset*>(other_asset);
    asset_id                   = casted_asset->asset_id;
    position_in_stack          = casted_asset->position_in_stack;
    x_size                     = casted_asset->x_size;
    y_size                     = casted_asset->y_size;
    number_of_frames           = casted_asset->number_of_frames;
    eer_frames_per_image       = casted_asset->eer_frames_per_image;
    eer_super_res_factor       = casted_asset->eer_super_res_factor;
    filename                   = casted_asset->filename;
    pixel_size                 = casted_asset->pixel_size;
    microscope_voltage         = casted_asset->microscope_voltage;
    spherical_aberration       = casted_asset->spherical_aberration;
    dose_per_frame             = casted_asset->dose_per_frame;
    is_valid                   = casted_asset->is_valid;
    total_dose                 = casted_asset->total_dose;
    asset_name                 = casted_asset->asset_name;
    gain_filename              = casted_asset->gain_filename;
    dark_filename              = casted_asset->dark_filename;
    output_binning_factor      = casted_asset->output_binning_factor;
    correct_mag_distortion     = casted_asset->correct_mag_distortion;
    mag_distortion_angle       = casted_asset->mag_distortion_angle;
    mag_distortion_major_scale = casted_asset->mag_distortion_major_scale;
    mag_distortion_minor_scale = casted_asset->mag_distortion_minor_scale;
    protein_is_white           = casted_asset->protein_is_white;
}

// Movie Metadata asset //

MovieMetadataAsset::MovieMetadataAsset( ) {
    movie_asset_id   = -1;
    metadata_source  = wxEmptyString;
    content_json     = wxEmptyString;
    tilt_angle       = NAN;
    stage_position_x = NAN;
    stage_position_y = NAN;
    stage_position_z = NAN;
    image_shift_x    = NAN;
    image_shift_y    = NAN;
    exposure_dose    = NAN;
    acquisition_time = wxInvalidDateTime;
}

MovieMetadataAsset::~MovieMetadataAsset( ) {
    // Do nothing
}

// Image asset///

ImageAsset::ImageAsset( ) {
    asset_id             = -1;
    parent_id            = -1;
    alignment_id         = -1;
    ctf_estimation_id    = -1;
    position_in_stack    = 1;
    x_size               = 0;
    y_size               = 0;
    pixel_size           = 0;
    microscope_voltage   = 0;
    spherical_aberration = 0;
    is_valid             = false;
    protein_is_white     = false;

    filename   = wxEmptyString;
    asset_name = wxEmptyString;
}

ImageAsset::~ImageAsset( ) {
    //Don't have to do anything for now
}

ImageAsset::ImageAsset(wxString wanted_filename) {
    filename          = wanted_filename;
    asset_name        = wanted_filename;
    asset_id          = -1;
    position_in_stack = 1;
    parent_id         = -1;
    alignment_id      = -1;
    ctf_estimation_id = -1;

    x_size               = 0;
    y_size               = 0;
    pixel_size           = 0;
    microscope_voltage   = 0;
    spherical_aberration = 0;
    is_valid             = false;

    protein_is_white = false;

    int number_in_stack;

    if ( filename.IsOk( ) == true && filename.FileExists( ) == true ) {
        is_valid = GetMRCDetails(filename.GetFullPath( ).fn_str( ), x_size, y_size, number_in_stack);
    }
}

void ImageAsset::Update(wxString wanted_filename) {
    filename = wanted_filename;
    is_valid = false;
    int number_in_stack;

    if ( filename.IsOk( ) == true && filename.FileExists( ) == true ) {
        is_valid = GetMRCDetails(filename.GetFullPath( ).fn_str( ), x_size, y_size, number_in_stack);
    }
}

void ImageAsset::CopyFrom(Asset* other_asset) {
    ImageAsset* casted_asset = reinterpret_cast<ImageAsset*>(other_asset);
    asset_id                 = casted_asset->asset_id;
    position_in_stack        = casted_asset->position_in_stack;
    parent_id                = casted_asset->parent_id;
    x_size                   = casted_asset->x_size;
    y_size                   = casted_asset->y_size;
    alignment_id             = casted_asset->alignment_id;
    ctf_estimation_id        = casted_asset->ctf_estimation_id;

    filename             = casted_asset->filename;
    pixel_size           = casted_asset->pixel_size;
    microscope_voltage   = casted_asset->microscope_voltage;
    spherical_aberration = casted_asset->spherical_aberration;
    is_valid             = casted_asset->is_valid;
    asset_name           = casted_asset->asset_name;

    protein_is_white = casted_asset->protein_is_white;
}

// Particle Position Asset

ParticlePositionAsset::ParticlePositionAsset( ) {
    Reset( );
}

ParticlePositionAsset::ParticlePositionAsset(const float& wanted_x_in_angstroms, const float& wanted_y_in_angstroms) {
    Reset( );
    x_position = wanted_x_in_angstroms;
    y_position = wanted_y_in_angstroms;
}

ParticlePositionAsset::~ParticlePositionAsset( ) {
    //Don't have to do anything for now
}

void ParticlePositionAsset::Reset( ) {
    asset_id           = -1;
    parent_id          = -1;
    picking_id         = -1;
    pick_job_id        = -1;
    parent_template_id = -1;
    x_position         = 0.0;
    y_position         = 0.0;
    peak_height        = 0.0;
    template_phi       = 0.0;
    template_theta     = 0.0;
    template_psi       = 0.0;
    asset_name         = wxEmptyString;
    filename           = wxEmptyString;
}

void ParticlePositionAsset::CopyFrom(Asset* other_asset) {
    ParticlePositionAsset* casted_asset = reinterpret_cast<ParticlePositionAsset*>(other_asset);
    asset_id                            = casted_asset->asset_id;
    parent_id                           = casted_asset->parent_id;
    picking_id                          = casted_asset->picking_id;
    pick_job_id                         = casted_asset->pick_job_id;
    parent_template_id                  = casted_asset->parent_template_id;
    x_position                          = casted_asset->x_position;
    y_position                          = casted_asset->y_position;
    peak_height                         = casted_asset->peak_height;
    template_phi                        = casted_asset->template_phi;
    template_theta                      = casted_asset->template_theta;
    template_psi                        = casted_asset->template_psi;
    asset_name                          = casted_asset->asset_name;
}

#include <wx/arrimpl.cpp>
WX_DEFINE_OBJARRAY(ArrayOfParticlePositionAssets);

// Volume asset///

VolumeAsset::VolumeAsset( ) {
    asset_id              = -1;
    parent_id             = -1;
    reconstruction_job_id = -1;
    x_size                = 0;
    y_size                = 0;
    z_size                = 0;
    pixel_size            = 0;

    is_valid   = false;
    filename   = wxEmptyString;
    asset_name = wxEmptyString;
}

VolumeAsset::~VolumeAsset( ) {
    //Don't have to do anything for now
}

VolumeAsset::VolumeAsset(wxString wanted_filename) {
    filename              = wanted_filename;
    asset_name            = wanted_filename;
    asset_id              = -1;
    parent_id             = -1;
    reconstruction_job_id = -1;

    x_size     = 0;
    y_size     = 0;
    z_size     = 0;
    pixel_size = 0;
    is_valid   = false;

    int number_in_stack;

    if ( filename.IsOk( ) == true && filename.FileExists( ) == true ) {
        is_valid = GetMRCDetails(filename.GetFullPath( ).fn_str( ), x_size, y_size, z_size);
    }
}

void VolumeAsset::Update(wxString wanted_filename) {
    filename = wanted_filename;
    is_valid = false;

    if ( filename.IsOk( ) == true && filename.FileExists( ) == true ) {
        is_valid = GetMRCDetails(filename.GetFullPath( ).fn_str( ), x_size, y_size, z_size);
    }
}

void VolumeAsset::CopyFrom(Asset* other_asset) {
    VolumeAsset* casted_asset = reinterpret_cast<VolumeAsset*>(other_asset);
    asset_id                  = casted_asset->asset_id;
    parent_id                 = casted_asset->parent_id;
    reconstruction_job_id     = casted_asset->reconstruction_job_id;

    x_size = casted_asset->x_size;
    y_size = casted_asset->y_size;
    z_size = casted_asset->z_size;

    filename   = casted_asset->filename;
    pixel_size = casted_asset->pixel_size;
    is_valid   = casted_asset->is_valid;
    asset_name = casted_asset->asset_name;
}

// Return Pointers

// AtomicCoordinates asset///

AtomicCoordinatesAsset::AtomicCoordinatesAsset( ) {
    asset_id             = -1;
    parent_id            = -1;
    simulation_3d_job_id = -1;
    x_size               = 0;
    y_size               = 0;
    z_size               = 0;

    is_valid   = false;
    filename   = wxEmptyString;
    asset_name = wxEmptyString;

    pdb_id           = wxEmptyString;
    pdb_avg_bfactor  = 0.0f;
    pdb_std_bfactor  = 0.0f;
    effective_weight = 0.0f;
}

AtomicCoordinatesAsset::~AtomicCoordinatesAsset( ) {
    //Don't have to do anything for now
}

AtomicCoordinatesAsset::AtomicCoordinatesAsset(wxString wanted_filename) {
    filename             = wanted_filename;
    asset_name           = wanted_filename;
    asset_id             = -1;
    parent_id            = -1;
    simulation_3d_job_id = -1;

    x_size   = 0;
    y_size   = 0;
    z_size   = 0;
    is_valid = false;

    pdb_id           = wxEmptyString;
    pdb_avg_bfactor  = 0.0f;
    pdb_std_bfactor  = 0.0f;
    effective_weight = 0.0f;

    int number_in_stack;

    if ( filename.IsOk( ) == true && filename.FileExists( ) == true ) {
        Update(wanted_filename);
    }
}

void AtomicCoordinatesAsset::Update(wxString wanted_filename) {
    filename = wanted_filename;
    is_valid = false;

    // Using the try/catch incase the user provides a bad file.
    try {
        // Note that gemmi will fail for "bad" extension names, which I should check into. It wouldn't be so hard to add a check on file names (pdb1 for example)
        // but this would be more appropriate to patch and apply upstream in their repo. TODO
        auto st = gemmi::read_structure(gemmi::MaybeGzipped(filename.GetFullPath( ).ToStdString( )));
        pdb_id  = st.name;

        // This loop should be part of some atomic coordinates utiltity class later on. For now, we want to get the x,y,z extents
        // which will minimally be useful in determining the box size for simulating templates.
        // TOOD: Check on NCS and option to expand?

        float  x_min = std::numeric_limits<float>::max( );
        float  x_max = std::numeric_limits<float>::min( );
        float  y_min = std::numeric_limits<float>::max( );
        float  y_max = std::numeric_limits<float>::min( );
        float  z_min = std::numeric_limits<float>::max( );
        float  z_max = std::numeric_limits<float>::min( );
        long   n     = 0;
        double b = 0, bb = 0;
        double weight    = 0;
        long   n_atoms   = 0;
        long   n_hetatms = 0;
        // I'm sure there is already something in GEMMI to do an iteration like this.
        for ( gemmi::Model& model : st.models ) {
            // mass += gemmi::calculate_mass(model);
            for ( gemmi::Chain& chain : model.chains ) {
                for ( gemmi::Residue& res : chain.residues ) {

                    // For now, we only want ATOM
                    if ( res.het_flag == 'A' ) // 'A' = ATOM, 'H' = HETATM, 0 = unspecified
                    {
                        for ( gemmi::Atom& atom : res.atoms ) {
                            n_atoms++;
                            weight += (atom.occ * atom.element.weight( ));
                            b += atom.b_iso;
                            bb += (atom.b_iso * atom.b_iso);
                            if ( atom.pos.x < x_min )
                                x_min = atom.pos.x;
                            if ( atom.pos.x > x_max )
                                x_max = atom.pos.x;
                            if ( atom.pos.y < y_min )
                                y_min = atom.pos.y;
                            if ( atom.pos.y > y_max )
                                y_max = atom.pos.y;
                            if ( atom.pos.z < z_min )
                                z_min = atom.pos.z;
                            if ( atom.pos.z > z_max )
                                z_max = atom.pos.z;
                            n++;
                        }
                    }
                    else if ( res.het_flag == 'H' ) {
                        for ( gemmi::Atom& atom : res.atoms ) {
                            n_hetatms++;
                        }
                    }
                }
            }
        }

        size_t n_hydrogens = gemmi::count_hydrogen_sites(st.models[0]);
        wxPrintf("There are N: \n");
        wxPrintf("Hydrogens : %ld\n", long(n_hydrogens));
        wxPrintf("Atoms     : %ld\n", long(n_atoms));
        wxPrintf("HetAtoms  : %ld\n", long(n_hetatms));

        b /= n;
        bb              = sqrt(bb / n - b * b);
        pdb_avg_bfactor = b;
        pdb_std_bfactor = bb;

        x_size = x_max - x_min;
        y_size = y_max - y_min;
        z_size = z_max - z_min;

        effective_weight = float(weight / 1000.); // kDa

        is_valid = true;
    } catch ( std::runtime_error& e ) {
        // It may be nice if this returned and printed in the error dialog invoked when is_valid is false, rather than
        // printing to stdout.
        MyPrintWithDetails("\n\nGEMMI threw an error reading this file:\n %s\n", e.what( ));
    }
}

void AtomicCoordinatesAsset::CopyFrom(Asset* other_asset) {
    AtomicCoordinatesAsset* casted_asset = reinterpret_cast<AtomicCoordinatesAsset*>(other_asset);
    asset_id                             = casted_asset->asset_id;
    parent_id                            = casted_asset->parent_id;
    simulation_3d_job_id                 = casted_asset->simulation_3d_job_id;

    x_size = casted_asset->x_size;
    y_size = casted_asset->y_size;
    z_size = casted_asset->z_size;

    filename   = casted_asset->filename;
    is_valid   = casted_asset->is_valid;
    asset_name = casted_asset->asset_name;

    pdb_id           = casted_asset->pdb_id;
    pdb_avg_bfactor  = casted_asset->pdb_avg_bfactor;
    pdb_std_bfactor  = casted_asset->pdb_std_bfactor;
    effective_weight = casted_asset->effective_weight;
}

MovieAsset* AssetList::ReturnMovieAssetPointer(long wanted_asset) {
    MyPrintWithDetails("This should never be called!!");
    DEBUG_ABORT;
}

ImageAsset* AssetList::ReturnImageAssetPointer(long wanted_asset) {
    MyPrintWithDetails("This should never be called!!");
    DEBUG_ABORT;
}

ParticlePositionAsset* AssetList::ReturnParticlePositionAssetPointer(long wanted_asset) {
    MyPrintWithDetails("This should never be called!!");
    DEBUG_ABORT;
}

VolumeAsset* AssetList::ReturnVolumeAssetPointer(long wanted_asset) {
    MyPrintWithDetails("This should never be called!!");
    DEBUG_ABORT;
}

AtomicCoordinatesAsset* AssetList::ReturnAtomicCoordinatesAssetPointer(long wanted_asset) {
    MyPrintWithDetails("This should never be called!!");
    DEBUG_ABORT;
}

////////////////////////Movie Asset List//////////////////

MovieAssetList::MovieAssetList( ) {
    number_of_assets = 0;
    number_allocated = 15;
    assets           = new MovieAsset[15];
}

MovieAssetList::~MovieAssetList( ) {
    delete[] reinterpret_cast<MovieAsset*>(assets);
}

void MovieAssetList::CheckMemory( ) {
    MovieAsset* buffer;

    // check we have enough memory

    if ( number_of_assets >= number_allocated ) {
        // reallocate..

        if ( number_of_assets < 10000 )
            number_allocated *= 2;
        else
            number_allocated += 10000;

        buffer = new MovieAsset[number_allocated];

        for ( long counter = 0; counter < number_of_assets; counter++ ) {
            buffer[counter].CopyFrom(&reinterpret_cast<MovieAsset*>(assets)[counter]);
        }

        delete[] reinterpret_cast<MovieAsset*>(assets);
        assets = buffer;
    }
}

long MovieAssetList::FindFile(wxFileName file_to_find, bool also_check_vs_shortname, long max_asset_number_to_check) {
    long found_position = -1;

    if ( max_asset_number_to_check == -1 )
        max_asset_number_to_check = number_of_assets;

    for ( long counter = 0; counter < max_asset_number_to_check; counter++ ) {
        if ( reinterpret_cast<MovieAsset*>(assets)[counter].filename == file_to_find ) {
            found_position = counter;
            break;
        }

        if ( also_check_vs_shortname == true ) {
            if ( reinterpret_cast<MovieAsset*>(assets)[counter].filename.GetFullName( ) == file_to_find.GetFullName( ) ) {
                found_position = counter;
                break;
            }
        }
    }

    return found_position;
}

Asset* MovieAssetList::ReturnAssetPointer(long wanted_asset) {
    MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
    return &reinterpret_cast<MovieAsset*>(assets)[wanted_asset];
}

MovieAsset* MovieAssetList::ReturnMovieAssetPointer(long wanted_asset) {
    MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
    return &reinterpret_cast<MovieAsset*>(assets)[wanted_asset];
}

int MovieAssetList::ReturnAssetID(long wanted_asset) {
    return reinterpret_cast<MovieAsset*>(assets)[wanted_asset].asset_id;
}

long MovieAssetList::ReturnParentAssetID(long wanted_asset) {
    return reinterpret_cast<MovieAsset*>(assets)[wanted_asset].parent_id;
}

wxString MovieAssetList::ReturnAssetName(long wanted_asset) {
    return reinterpret_cast<MovieAsset*>(assets)[wanted_asset].asset_name;
}

wxString MovieAssetList::ReturnAssetFullFilename(long wanted_asset) {
    return reinterpret_cast<MovieAsset*>(assets)[wanted_asset].filename.GetFullPath( );
}

int MovieAssetList::ReturnArrayPositionFromID(int wanted_id, int last_found_position) {
    MyDebugAssertTrue(last_found_position < number_of_assets, "Bad last found position: %i >= %i\n", last_found_position, number_of_assets);

    for ( int counter = last_found_position; counter < number_of_assets; counter++ ) {
        if ( reinterpret_cast<MovieAsset*>(assets)[counter].asset_id == wanted_id )
            return counter;
    }

    for ( int counter = 0; counter < last_found_position; counter++ ) {
        if ( reinterpret_cast<MovieAsset*>(assets)[counter].asset_id == wanted_id )
            return counter;
    }

    return -1;
}

int MovieAssetList::ReturnArrayPositionFromParentID(int wanted_id) {
    for ( int counter = 0; counter < number_of_assets; counter++ ) {
        if ( reinterpret_cast<MovieAsset*>(assets)[counter].parent_id == wanted_id )
            return counter;
    }

    return -1;
}

void MovieAssetList::AddAsset(Asset* asset_to_add) {
    MovieAsset* buffer;

    CheckMemory( );

    // Should be fine for memory, so just add one.

    reinterpret_cast<MovieAsset*>(assets)[number_of_assets].CopyFrom(asset_to_add);
    number_of_assets++;
}

void MovieAssetList::RemoveAsset(long number_to_remove) {
    if ( number_to_remove < 0 || number_to_remove >= number_of_assets ) {
        wxPrintf("Error! Trying to remove a movie that does not exist\n\n");
        exit(-1);
    }

    for ( long counter = number_to_remove; counter < number_of_assets - 1; counter++ ) {
        reinterpret_cast<MovieAsset*>(assets)[counter].CopyFrom(&reinterpret_cast<MovieAsset*>(assets)[counter + 1]);
    }

    number_of_assets--;
}

void MovieAssetList::RemoveAll( ) {
    number_of_assets = 0;

    if ( number_allocated > 100 ) {
        delete[] reinterpret_cast<MovieAsset*>(assets);
        number_allocated = 100;
        assets           = new MovieAsset[number_allocated];
    }
}

////////////////////////Image Asset List//////////////////

ImageAssetList::ImageAssetList( ) {
    number_of_assets = 0;
    number_allocated = 15;
    assets           = new ImageAsset[15];
}

ImageAssetList::~ImageAssetList( ) {
    delete[] reinterpret_cast<ImageAsset*>(assets);
}

void ImageAssetList::CheckMemory( ) {
    ImageAsset* buffer;

    // check we have enough memory

    if ( number_of_assets >= number_allocated ) {
        // reallocate..

        if ( number_of_assets < 10000 )
            number_allocated *= 2;
        else
            number_allocated += 10000;

        buffer = new ImageAsset[number_allocated];

        for ( long counter = 0; counter < number_of_assets; counter++ ) {
            buffer[counter].CopyFrom(&reinterpret_cast<ImageAsset*>(assets)[counter]);
        }

        delete[] reinterpret_cast<ImageAsset*>(assets);
        assets = buffer;
    }
}

long ImageAssetList::FindFile(wxFileName file_to_find, bool also_check_vs_shortname, long max_asset_number_to_check) {
    long found_position = -1;

    if ( max_asset_number_to_check == -1 )
        max_asset_number_to_check = number_of_assets;

    for ( long counter = 0; counter < max_asset_number_to_check; counter++ ) {
        if ( reinterpret_cast<ImageAsset*>(assets)[counter].filename == file_to_find ) {
            found_position = counter;
            break;
        }

        if ( also_check_vs_shortname == true ) {
            if ( reinterpret_cast<ImageAsset*>(assets)[counter].filename.GetFullName( ) == file_to_find.GetFullName( ) ) {
                found_position = counter;
                break;
            }
        }
    }

    return found_position;
}

Asset* ImageAssetList::ReturnAssetPointer(long wanted_asset) {
    MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
    return &reinterpret_cast<ImageAsset*>(assets)[wanted_asset];
}

ImageAsset* ImageAssetList::ReturnImageAssetPointer(long wanted_asset) {
    if ( wanted_asset >= 0 && wanted_asset < number_of_assets )
        return &reinterpret_cast<ImageAsset*>(assets)[wanted_asset];
    else {
        MyDebugPrintWithDetails("Requesting an asset (%li) that doesn't exist!", wanted_asset);
        return NULL;
    }
}

int ImageAssetList::ReturnAssetID(long wanted_asset) {
    return reinterpret_cast<ImageAsset*>(assets)[wanted_asset].asset_id;
}

long ImageAssetList::ReturnParentAssetID(long wanted_asset) {
    return reinterpret_cast<ImageAsset*>(assets)[wanted_asset].parent_id;
}

wxString ImageAssetList::ReturnAssetName(long wanted_asset) {
    return reinterpret_cast<ImageAsset*>(assets)[wanted_asset].asset_name;
}

wxString ImageAssetList::ReturnAssetFullFilename(long wanted_asset) {
    return reinterpret_cast<ImageAsset*>(assets)[wanted_asset].filename.GetFullPath( );
}

int ImageAssetList::ReturnArrayPositionFromID(int wanted_id, int last_found_position) {
    MyDebugAssertTrue(last_found_position < number_of_assets, "Bad last found position: %i >= %i\n", last_found_position, number_of_assets);

    for ( int counter = last_found_position; counter < number_of_assets; counter++ ) {
        if ( reinterpret_cast<ImageAsset*>(assets)[counter].asset_id == wanted_id )
            return counter;
    }

    for ( int counter = 0; counter < last_found_position; counter++ ) {
        if ( reinterpret_cast<ImageAsset*>(assets)[counter].asset_id == wanted_id )
            return counter;
    }

    return -1;
}

int ImageAssetList::ReturnArrayPositionFromParentID(int wanted_id) {
    for ( int counter = 0; counter < number_of_assets; counter++ ) {
        if ( reinterpret_cast<ImageAsset*>(assets)[counter].parent_id == wanted_id )
            return counter;
    }

    return -1;
}

void ImageAssetList::AddAsset(Asset* asset_to_add) {
    CheckMemory( );

    // Should be fine for memory, so just add one.

    reinterpret_cast<ImageAsset*>(assets)[number_of_assets].CopyFrom(asset_to_add);
    number_of_assets++;
}

void ImageAssetList::RemoveAsset(long number_to_remove) {
    if ( number_to_remove < 0 || number_to_remove >= number_of_assets ) {
        wxPrintf("Error! Trying to remove a movie that does not exist\n\n");
        exit(-1);
    }

    for ( long counter = number_to_remove; counter < number_of_assets - 1; counter++ ) {
        reinterpret_cast<ImageAsset*>(assets)[counter].CopyFrom(&reinterpret_cast<ImageAsset*>(assets)[counter + 1]);
    }

    number_of_assets--;
}

void ImageAssetList::RemoveAll( ) {
    number_of_assets = 0;

    if ( number_allocated > 100 ) {
        delete[] reinterpret_cast<ImageAsset*>(assets);
        number_allocated = 100;
        assets           = new ImageAsset[number_allocated];
    }
}

// Particle Asset List

ParticlePositionAssetList::ParticlePositionAssetList( ) {
    number_of_assets = 0;
    number_allocated = 15;
    assets           = new ParticlePositionAsset[15];
}

ParticlePositionAssetList::~ParticlePositionAssetList( ) {
    delete[] reinterpret_cast<ParticlePositionAsset*>(assets);
}

void ParticlePositionAssetList::CheckMemory( ) {
    ParticlePositionAsset* buffer;

    const int chunk_size = 2500000;

    // check we have enough memory

    if ( number_of_assets >= number_allocated ) {
        // reallocate..

        if ( number_of_assets < chunk_size )
            number_allocated *= 2;
        else
            number_allocated += chunk_size;

        buffer = new ParticlePositionAsset[number_allocated];

        for ( long counter = 0; counter < number_of_assets; counter++ ) {
            buffer[counter].CopyFrom(&reinterpret_cast<ParticlePositionAsset*>(assets)[counter]);
        }

        delete[] reinterpret_cast<ParticlePositionAsset*>(assets);
        assets = buffer;
    }
}

Asset* ParticlePositionAssetList::ReturnAssetPointer(long wanted_asset) {
    MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
    return &reinterpret_cast<ParticlePositionAsset*>(assets)[wanted_asset];
}

ParticlePositionAsset* ParticlePositionAssetList::ReturnParticlePositionAssetPointer(long wanted_asset) {
    MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
    return &reinterpret_cast<ParticlePositionAsset*>(assets)[wanted_asset];
}

int ParticlePositionAssetList::ReturnAssetID(long wanted_asset) {
    return reinterpret_cast<ParticlePositionAsset*>(assets)[wanted_asset].asset_id;
}

long ParticlePositionAssetList::ReturnParentAssetID(long wanted_asset) {
    return reinterpret_cast<ParticlePositionAsset*>(assets)[wanted_asset].parent_id;
}

int ParticlePositionAssetList::ReturnArrayPositionFromID(int wanted_id, int last_found_position) {
    MyDebugAssertTrue(last_found_position < number_of_assets, "Bad last found position: %i >= %i\n", last_found_position, number_of_assets);
    MyDebugAssertTrue(last_found_position >= 0, "Bad last found position: %i < 0\n", last_found_position);

    for ( int counter = last_found_position; counter < number_of_assets; counter++ ) {
        if ( reinterpret_cast<ParticlePositionAsset*>(assets)[counter].asset_id == wanted_id )
            return counter;
    }

    for ( int counter = 0; counter < last_found_position; counter++ ) {
        if ( reinterpret_cast<ParticlePositionAsset*>(assets)[counter].asset_id == wanted_id )
            return counter;
    }

    return -1;
}

int ParticlePositionAssetList::ReturnArrayPositionFromParentID(int wanted_id) {
    for ( int counter = 0; counter < number_of_assets; counter++ ) {
        if ( reinterpret_cast<ParticlePositionAsset*>(assets)[counter].parent_id == wanted_id )
            return counter;
    }

    return -1;
}

void ParticlePositionAssetList::AddAsset(Asset* asset_to_add) {
    CheckMemory( );

    // Should be fine for memory, so just add one.

    reinterpret_cast<ParticlePositionAsset*>(assets)[number_of_assets].CopyFrom(asset_to_add);
    number_of_assets++;
}

void ParticlePositionAssetList::RemoveAssetsWithGivenParentImageID(long parent_image_id) {
    long copy_from                  = -1;
    long copy_to                    = 0;
    long number_of_remaining_assets = number_of_assets;

    ParticlePositionAsset* current_asset;

    while ( copy_to < number_of_remaining_assets ) {
        copy_from++;
        MyDebugAssertTrue(copy_from < number_of_assets, "Can't copy from %li, because it's beyond %li\n", copy_from, number_of_assets - 1);
        current_asset = &reinterpret_cast<ParticlePositionAsset*>(assets)[copy_from];

        if ( current_asset->parent_id == parent_image_id ) {
            // We're not keeping this one
            number_of_remaining_assets--;
        }
        else {
            // We're keeping this one
            reinterpret_cast<ParticlePositionAsset*>(assets)[copy_to].CopyFrom(current_asset);
            MyDebugAssertTrue(copy_to <= copy_from, "Can't copy from %li to %li\n", copy_from, copy_to);
            copy_to++;
        }
    }

    MyDebugAssertTrue(number_of_remaining_assets >= 0, "Bad number of remaining assets: %li\n", number_of_remaining_assets);
    number_of_assets = number_of_remaining_assets;
}

void ParticlePositionAssetList::RemoveAsset(long number_to_remove) {
    if ( number_to_remove < 0 || number_to_remove >= number_of_assets ) {
        wxPrintf("Error! Trying to remove a particle position that does not exist\n\n");
        exit(-1);
    }

    for ( long counter = number_to_remove; counter < number_of_assets - 1; counter++ ) {
        reinterpret_cast<ParticlePositionAsset*>(assets)[counter].CopyFrom(&reinterpret_cast<ParticlePositionAsset*>(assets)[counter + 1]);
    }

    number_of_assets--;
}

void ParticlePositionAssetList::RemoveAll( ) {
    number_of_assets = 0;

    if ( number_allocated > 100 ) {
        delete[] reinterpret_cast<ParticlePositionAsset*>(assets);
        number_allocated = 100;
        assets           = new ParticlePositionAsset[number_allocated];
    }
}

// Volume Asset List

VolumeAssetList::VolumeAssetList( ) {
    number_of_assets = 0;
    number_allocated = 15;
    assets           = new VolumeAsset[15];
}

VolumeAssetList::~VolumeAssetList( ) {
    delete[] reinterpret_cast<VolumeAsset*>(assets);
}

void VolumeAssetList::CheckMemory( ) {
    VolumeAsset* buffer;

    // check we have enough memory

    if ( number_of_assets >= number_allocated ) {
        // reallocate..

        if ( number_of_assets < 10000 )
            number_allocated *= 2;
        else
            number_allocated += 10000;

        buffer = new VolumeAsset[number_allocated];

        for ( long counter = 0; counter < number_of_assets; counter++ ) {
            buffer[counter].CopyFrom(&reinterpret_cast<VolumeAsset*>(assets)[counter]);
        }

        delete[] reinterpret_cast<VolumeAsset*>(assets);
        assets = buffer;
    }
}

long VolumeAssetList::FindFile(wxFileName file_to_find, bool also_check_vs_shortname, long max_asset_number_to_check) {
    long found_position = -1;

    if ( max_asset_number_to_check == -1 )
        max_asset_number_to_check = number_of_assets;

    for ( long counter = 0; counter < max_asset_number_to_check; counter++ ) {
        if ( reinterpret_cast<VolumeAsset*>(assets)[counter].filename == file_to_find ) {
            found_position = counter;
            break;
        }

        if ( also_check_vs_shortname == true ) {
            if ( reinterpret_cast<VolumeAsset*>(assets)[counter].filename.GetFullName( ) == file_to_find.GetFullName( ) ) {
                found_position = counter;
                break;
            }
        }
    }

    return found_position;
}

Asset* VolumeAssetList::ReturnAssetPointer(long wanted_asset) {
    MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
    return &reinterpret_cast<VolumeAsset*>(assets)[wanted_asset];
}

long VolumeAssetList::ReturnParentAssetID(long wanted_asset) {
    return reinterpret_cast<VolumeAsset*>(assets)[wanted_asset].parent_id;
}

VolumeAsset* VolumeAssetList::ReturnVolumeAssetPointer(long wanted_asset) {
    MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
    return &reinterpret_cast<VolumeAsset*>(assets)[wanted_asset];
}

int VolumeAssetList::ReturnAssetID(long wanted_asset) {
    return reinterpret_cast<VolumeAsset*>(assets)[wanted_asset].asset_id;
}

wxString VolumeAssetList::ReturnAssetName(long wanted_asset) {
    return reinterpret_cast<VolumeAsset*>(assets)[wanted_asset].asset_name;
}

wxString VolumeAssetList::ReturnAssetFullFilename(long wanted_asset) {
    return reinterpret_cast<VolumeAsset*>(assets)[wanted_asset].filename.GetFullPath( );
}

int VolumeAssetList::ReturnArrayPositionFromID(int wanted_id, int last_found_position) {
    MyDebugAssertTrue(last_found_position < number_of_assets || number_of_assets == 0, "Bad last found position: %i >= %li\n", last_found_position, number_of_assets);

    for ( int counter = last_found_position; counter < number_of_assets; counter++ ) {
        if ( reinterpret_cast<VolumeAsset*>(assets)[counter].asset_id == wanted_id )
            return counter;
    }

    for ( int counter = 0; counter < last_found_position; counter++ ) {
        if ( reinterpret_cast<VolumeAsset*>(assets)[counter].asset_id == wanted_id )
            return counter;
    }

    return -1;
}

int VolumeAssetList::ReturnArrayPositionFromParentID(int wanted_id) {
    for ( int counter = 0; counter < number_of_assets; counter++ ) {
        if ( reinterpret_cast<VolumeAsset*>(assets)[counter].parent_id == wanted_id )
            return counter;
    }

    return -1;
}

void VolumeAssetList::AddAsset(Asset* asset_to_add) {
    CheckMemory( );

    // Should be fine for memory, so just add one.

    reinterpret_cast<VolumeAsset*>(assets)[number_of_assets].CopyFrom(asset_to_add);
    number_of_assets++;
}

void VolumeAssetList::RemoveAsset(long number_to_remove) {
    if ( number_to_remove < 0 || number_to_remove >= number_of_assets ) {
        wxPrintf("Error! Trying to remove a movie that does not exist\n\n");
        exit(-1);
    }

    for ( long counter = number_to_remove; counter < number_of_assets - 1; counter++ ) {
        reinterpret_cast<VolumeAsset*>(assets)[counter].CopyFrom(&reinterpret_cast<VolumeAsset*>(assets)[counter + 1]);
    }

    number_of_assets--;
}

void VolumeAssetList::RemoveAll( ) {
    number_of_assets = 0;

    if ( number_allocated > 100 ) {
        delete[] reinterpret_cast<VolumeAsset*>(assets);
        number_allocated = 100;
        assets           = new VolumeAsset[number_allocated];
    }
}

// AtomicCoordinates Asset List

AtomicCoordinatesAssetList::AtomicCoordinatesAssetList( ) {
    // TODO: This is likely too few pre allocated memory.
    number_of_assets = 0;
    number_allocated = 15;
    assets           = new AtomicCoordinatesAsset[15];
}

AtomicCoordinatesAssetList::~AtomicCoordinatesAssetList( ) {
    delete[] reinterpret_cast<AtomicCoordinatesAsset*>(assets);
}

void AtomicCoordinatesAssetList::CheckMemory( ) {
    AtomicCoordinatesAsset* buffer;

    // check we have enough memory

    if ( number_of_assets >= number_allocated ) {
        // reallocate..

        if ( number_of_assets < 10000 )
            number_allocated *= 2;
        else
            number_allocated += 10000;

        buffer = new AtomicCoordinatesAsset[number_allocated];

        for ( long counter = 0; counter < number_of_assets; counter++ ) {
            buffer[counter].CopyFrom(&reinterpret_cast<AtomicCoordinatesAsset*>(assets)[counter]);
        }

        delete[] reinterpret_cast<AtomicCoordinatesAsset*>(assets);
        assets = buffer;
    }
}

long AtomicCoordinatesAssetList::FindFile(wxFileName file_to_find, bool also_check_vs_shortname, long max_asset_number_to_check) {
    long found_position = -1;

    if ( max_asset_number_to_check == -1 )
        max_asset_number_to_check = number_of_assets;

    for ( long counter = 0; counter < max_asset_number_to_check; counter++ ) {
        if ( reinterpret_cast<AtomicCoordinatesAsset*>(assets)[counter].filename == file_to_find ) {
            found_position = counter;
            break;
        }

        if ( also_check_vs_shortname == true ) {
            if ( reinterpret_cast<AtomicCoordinatesAsset*>(assets)[counter].filename.GetFullName( ) == file_to_find.GetFullName( ) ) {
                found_position = counter;
                break;
            }
        }
    }

    return found_position;
}

Asset* AtomicCoordinatesAssetList::ReturnAssetPointer(long wanted_asset) {
    MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
    return &reinterpret_cast<AtomicCoordinatesAsset*>(assets)[wanted_asset];
}

long AtomicCoordinatesAssetList::ReturnParentAssetID(long wanted_asset) {
    return reinterpret_cast<AtomicCoordinatesAsset*>(assets)[wanted_asset].parent_id;
}

AtomicCoordinatesAsset* AtomicCoordinatesAssetList::ReturnAtomicCoordinatesAssetPointer(long wanted_asset) {
    MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
    return &reinterpret_cast<AtomicCoordinatesAsset*>(assets)[wanted_asset];
}

int AtomicCoordinatesAssetList::ReturnAssetID(long wanted_asset) {
    return reinterpret_cast<AtomicCoordinatesAsset*>(assets)[wanted_asset].asset_id;
}

wxString AtomicCoordinatesAssetList::ReturnAssetName(long wanted_asset) {
    return reinterpret_cast<AtomicCoordinatesAsset*>(assets)[wanted_asset].asset_name;
}

wxString AtomicCoordinatesAssetList::ReturnAssetFullFilename(long wanted_asset) {
    return reinterpret_cast<AtomicCoordinatesAsset*>(assets)[wanted_asset].filename.GetFullPath( );
}

int AtomicCoordinatesAssetList::ReturnArrayPositionFromID(int wanted_id, int last_found_position) {
    MyDebugAssertTrue(last_found_position < number_of_assets || number_of_assets == 0, "Bad last found position: %i >= %li\n", last_found_position, number_of_assets);

    for ( int counter = last_found_position; counter < number_of_assets; counter++ ) {
        if ( reinterpret_cast<AtomicCoordinatesAsset*>(assets)[counter].asset_id == wanted_id )
            return counter;
    }

    for ( int counter = 0; counter < last_found_position; counter++ ) {
        if ( reinterpret_cast<AtomicCoordinatesAsset*>(assets)[counter].asset_id == wanted_id )
            return counter;
    }

    return -1;
}

int AtomicCoordinatesAssetList::ReturnArrayPositionFromParentID(int wanted_id) {
    for ( int counter = 0; counter < number_of_assets; counter++ ) {
        if ( reinterpret_cast<AtomicCoordinatesAsset*>(assets)[counter].parent_id == wanted_id )
            return counter;
    }

    return -1;
}

void AtomicCoordinatesAssetList::AddAsset(Asset* asset_to_add) {
    CheckMemory( );

    // Should be fine for memory, so just add one.

    reinterpret_cast<AtomicCoordinatesAsset*>(assets)[number_of_assets].CopyFrom(asset_to_add);
    number_of_assets++;
}

void AtomicCoordinatesAssetList::RemoveAsset(long number_to_remove) {
    if ( number_to_remove < 0 || number_to_remove >= number_of_assets ) {
        wxPrintf("Error! Trying to remove a movie that does not exist\n\n");
        exit(-1);
    }

    for ( long counter = number_to_remove; counter < number_of_assets - 1; counter++ ) {
        reinterpret_cast<AtomicCoordinatesAsset*>(assets)[counter].CopyFrom(&reinterpret_cast<AtomicCoordinatesAsset*>(assets)[counter + 1]);
    }

    number_of_assets--;
}

void AtomicCoordinatesAssetList::RemoveAll( ) {
    number_of_assets = 0;

    if ( number_allocated > 100 ) {
        delete[] reinterpret_cast<AtomicCoordinatesAsset*>(assets);
        number_allocated = 100;
        assets           = new AtomicCoordinatesAsset[number_allocated];
    }
}
