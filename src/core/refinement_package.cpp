#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfRefinmentPackageParticleInfos);
WX_DEFINE_OBJARRAY(ArrayOfRefinementPackages);
//WX_DEFINE_OBJARRAY(ArrayofSingleRefinementResults);
//WX_DEFINE_OBJARRAY(ArrayofMultiClassRefinementResults);
//WX_DEFINE_OBJARRAY(ArrayofWholeRefinementResults);

RefinementPackageParticleInfo::RefinementPackageParticleInfo()
{
	parent_image_id = -1;
	original_particle_position_asset_id = -1;
	position_in_stack = -1;
	x_pos = -1;
	y_pos = -1;
	pixel_size = -1;
	defocus_1 = 0;
	defocus_2 = 0;
	defocus_angle = 0;
	phase_shift = 0;
	spherical_aberration = 0;
	microscope_voltage = 0;
	amplitude_contrast = 0.07;
}

RefinementPackageParticleInfo::~RefinementPackageParticleInfo()
{


}


RefinementPackage::RefinementPackage()
{
	asset_id = -1;
	stack_filename = "";
	name = "";
	stack_box_size = -1;
	output_pixel_size = -1;

	number_of_classes = -1;
	number_of_run_refinments = -1;
	last_refinment_id = -1;

	estimated_particle_size_in_angstroms = 0.0;
	estimated_particle_weight_in_kda = 0.0;
	lowest_resolution_of_intial_parameter_generated_3ds = -1;

	stack_has_white_protein = false;
}

RefinementPackage::~RefinementPackage()
{


}


long RefinementPackage::ReturnLastRefinementID()
{
	return refinement_ids.Item(refinement_ids.GetCount() - 1);
}

RefinementPackageParticleInfo RefinementPackage::ReturnParticleInfoByPositionInStack(long wanted_position_in_stack)
{
	for (long counter = wanted_position_in_stack - 1; counter < contained_particles.GetCount(); counter++)
	{
		if (contained_particles.Item(counter).position_in_stack == wanted_position_in_stack) return contained_particles.Item(counter);
	}

	for (long counter = 0; counter < wanted_position_in_stack; counter++)
	{
		if (contained_particles.Item(counter).position_in_stack == wanted_position_in_stack) return contained_particles.Item(counter);
	}

	MyDebugPrintWithDetails("Shouldn't get here, means i didn't find the particle");
	DEBUG_ABORT;
}
