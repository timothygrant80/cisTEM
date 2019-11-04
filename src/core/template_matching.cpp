#include "core_headers.h"

#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfTemplateMatchFoundPeakInfos);
WX_DEFINE_OBJARRAY(ArrayOfTemplateMatchJobResults);

TemplateMatchJobResults::TemplateMatchJobResults()
{
	job_name = "";
	job_type = -1;
	input_job_id = -1;
	job_id = -1;
	datetime_of_run = 0;
	image_asset_id = -1;
	ref_volume_asset_id = -1;
	symmetry = "C1";
	pixel_size = 0.0f;
	voltage = 0.0f;
	spherical_aberration = 0.0f;
	amplitude_contrast = 0.0f;
	defocus1 = 0.0f;
	defocus2 = 0.0f;
	defocus_angle = 0.0f;
	phase_shift = 0.0f;
	low_res_limit = 0.0f;
	high_res_limit = 0.0f;
	out_of_plane_step = 0.0f;
	in_plane_step = 0.0f;
	defocus_search_range = 0.0f;
	defocus_step = 0.0f;
	pixel_size_search_range = 0.0f;
	pixel_size_step = 0.0f;
	mask_radius  = 0.0f;
	min_peak_radius = 0.0f;
	xy_change_threshold = 0.0f;
	exclude_above_xy_threshold = false;

	mip_filename = "";
	scaled_mip_filename = "";
	psi_filename = "";
	theta_filename = "";
	phi_filename = "";
	defocus_filename = "";
	pixel_size_filename = "";
	histogram_filename = "";
	projection_result_filename = "";
	sum_filename = "";
	variance_filename = '';

	refinement_threshold = 0.0f;
	used_threshold = 0.0f;
	reference_box_size_in_angstroms = 0;
}
