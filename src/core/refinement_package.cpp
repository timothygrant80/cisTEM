#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfRefinmentPackageParticleInfos);
WX_DEFINE_OBJARRAY(ArrayOfRefinementPackages);

//WX_DEFINE_OBJARRAY(ArrayofSingleRefinementResults);
//WX_DEFINE_OBJARRAY(ArrayofMultiClassRefinementResults);
//WX_DEFINE_OBJARRAY(ArrayofWholeRefinementResults);

RefinementPackageParticleInfo::RefinementPackageParticleInfo( ) {
    parent_image_id                     = -1;
    original_particle_position_asset_id = -1;
    position_in_stack                   = -1;
    x_pos                               = -1;
    y_pos                               = -1;
    pixel_size                          = -1;
    defocus_1                           = 0;
    defocus_2                           = 0;
    defocus_angle                       = 0;
    phase_shift                         = 0;
    spherical_aberration                = 0;
    microscope_voltage                  = 0;
    amplitude_contrast                  = 0.07;
    assigned_subset                     = -1;

    // Multi-view fields
    particle_group = 1; // Default: all particles in same group
    pre_exposure   = 0.0f; // Default: no pre-exposure
    total_exposure = 0.1f; // Default: minimal exposure
}

RefinementPackageParticleInfo::~RefinementPackageParticleInfo( ) {
}

RefinementPackage::RefinementPackage( ) {
    asset_id          = -1;
    stack_filename    = "";
    name              = "";
    stack_box_size    = -1;
    output_pixel_size = -1;

    number_of_classes        = -1;
    number_of_run_refinments = -1;
    last_refinment_id        = -1;

    estimated_particle_size_in_angstroms                = 0.0;
    estimated_particle_weight_in_kda                    = 0.0;
    lowest_resolution_of_intial_parameter_generated_3ds = -1;

    stack_has_white_protein = false;
}

RefinementPackage::~RefinementPackage( ) {
}

long RefinementPackage::ReturnLastRefinementID( ) {
    return refinement_ids.Item(refinement_ids.GetCount( ) - 1);
}

RefinementPackageParticleInfo RefinementPackage::ReturnParticleInfoByPositionInStack(long wanted_position_in_stack) {
    for ( long counter = wanted_position_in_stack - 1; counter < contained_particles.GetCount( ); counter++ ) {
        if ( contained_particles.Item(counter).position_in_stack == wanted_position_in_stack )
            return contained_particles.Item(counter);
    }

    for ( long counter = 0; counter < wanted_position_in_stack; counter++ ) {
        if ( contained_particles.Item(counter).position_in_stack == wanted_position_in_stack )
            return contained_particles.Item(counter);
    }

    MyDebugPrintWithDetails("Shouldn't get here, means i didn't find the particle");
    DEBUG_ABORT;
}

bool RefinementPackage::ContainsMultiViewData( ) const {
    // Check if any particle has non-default multi-view values
    // Early return as soon as we find any non-default value

    // TODO: Migrate contained_particles from wxArray to std::vector<RefinementPackageParticleInfo>
    // This would allow us to:
    //   1. Use std::any_of with a lambda for more idiomatic C++:
    //      return std::any_of(contained_particles.begin(), contained_particles.end(),
    //                         [](const auto& p) { return p.particle_group != 1 ||
    //                                                    p.pre_exposure != 0.0f ||
    //                                                    p.total_exposure != 0.1f; });
    //   2. Consider making contained_particles private with getter/setter methods for better encapsulation
    //   3. Potentially use parallel algorithms (std::execution::par) for very large particle sets

    for ( long counter = 0; counter < contained_particles.GetCount( ); counter++ ) {
        const RefinementPackageParticleInfo& particle = contained_particles.Item(counter);

        // Check for any non-default values
        if ( particle.particle_group != 1 ||
             particle.pre_exposure != 0.0f ||
             particle.total_exposure != 0.1f ) {
            return true;
        }
    }

    return false;
}
