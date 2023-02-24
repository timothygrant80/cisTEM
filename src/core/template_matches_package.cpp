#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
//WX_DEFINE_OBJARRAY(ArrayOfRefinmentPackageParticleInfos);
WX_DEFINE_OBJARRAY(ArrayOfTemplateMatchesPackages);

//WX_DEFINE_OBJARRAY(ArrayofSingleRefinementResults);
//WX_DEFINE_OBJARRAY(ArrayofMultiClassRefinementResults);
//WX_DEFINE_OBJARRAY(ArrayofWholeRefinementResults);

/* RefinementPackageParticleInfo::RefinementPackageParticleInfo( ) {
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
}

RefinementPackageParticleInfo::~RefinementPackageParticleInfo( ) {
} */

TemplateMatchesPackage::TemplateMatchesPackage( ) {
    asset_id          = -1;
    starfile_filename = "";
    name              = "";
}

RefinementPackage::~RefinementPackage( ) {
}

long RefinementPackage::ReturnLastRefinementID( ) {
    return refinement_ids.Item(refinement_ids.GetCount( ) - 1);
}

/* RefinementPackageParticleInfo RefinementPackage::ReturnParticleInfoByPositionInStack(long wanted_position_in_stack) {
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
} */
