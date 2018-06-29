#include "../../core/core_headers.h"


class
MakeTemplateResult : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};

IMPLEMENT_APP(MakeTemplateResult)


// override the DoInteractiveUserInput

void MakeTemplateResult::DoInteractiveUserInput()
{

	wxString	input_reconstruction_filename;
	wxString    input_mip_filename;
	wxString    input_best_psi_filename;
	wxString    input_best_theta_filename;
	wxString    input_best_phi_filename;
	wxString    output_result_image_filename;

	float wanted_threshold;
	float min_peak_radius;

	UserInput *my_input = new UserInput("MakeTemplateResult", 1.00);

	input_reconstruction_filename = my_input->GetFilenameFromUser("Input template reconstruction", "The 3D reconstruction from which projections are calculated", "my_reconstruction.mrc", true);
	input_mip_filename = my_input->GetFilenameFromUser("Input MIP file", "The file for saving the maximum intensity projection image", "my_mip.mrc", false);
	input_best_psi_filename = my_input->GetFilenameFromUser("Input psi file", "The file containing the best psi image", "my_psi.mrc", false);
	input_best_theta_filename = my_input->GetFilenameFromUser("Input theta file", "The file containing the best psi image", "my_theta.mrc", false);
	input_best_phi_filename = my_input->GetFilenameFromUser("Input phi file", "The file containing the best psi image", "my_phi.mrc", false);
	output_result_image_filename = my_input->GetFilenameFromUser("Output found result file", "The file for saving the found result", "my_result.mrc", false);
	wanted_threshold = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
	min_peak_radius = my_input->GetFloatFromUser("Min Peak Radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 0.0);

	delete my_input;

	my_current_job.Reset(8);
	my_current_job.ManualSetArguments("ttttttff",	input_reconstruction_filename.ToUTF8().data(),
													input_mip_filename.ToUTF8().data(),
													input_best_psi_filename.ToUTF8().data(),
													input_best_theta_filename.ToUTF8().data(),
													input_best_phi_filename.ToUTF8().data(),
													output_result_image_filename.ToUTF8().data(),
													wanted_threshold,
													min_peak_radius);

}

// override the do calculation method which will be what is actually run..

bool MakeTemplateResult::DoCalculation()
{

	wxDateTime start_time = wxDateTime::Now();

	wxString	input_reconstruction_filename = my_current_job.arguments[0].ReturnStringArgument();
	wxString	input_mip_filename = my_current_job.arguments[1].ReturnStringArgument();
	wxString	input_best_psi_filename = my_current_job.arguments[2].ReturnStringArgument();
	wxString	input_best_theta_filename = my_current_job.arguments[3].ReturnStringArgument();
	wxString	input_best_phi_filename = my_current_job.arguments[4].ReturnStringArgument();
	wxString	output_result_image_filename = my_current_job.arguments[5].ReturnStringArgument();
	float		wanted_threshold = my_current_job.arguments[6].ReturnFloatArgument();
	float		min_peak_radius = my_current_job.arguments[7].ReturnFloatArgument();

	ImageFile input_reconstruction_file;

	input_reconstruction_file.OpenFile(input_reconstruction_filename.ToStdString(), false);

	Image output_image;
	Image mip_image;
	Image psi_image;
	Image theta_image;
	Image phi_image;
	Image input_reconstruction;
	Image current_projection;

	Peak current_peak;

	AnglesAndShifts angles;

	float current_phi;
	float current_theta;
	float current_psi;

	int number_of_peaks_found = 0;

	// pre square min peak radius

	min_peak_radius = powf(min_peak_radius, 2);

	mip_image.QuickAndDirtyReadSlice(input_mip_filename.ToStdString(), 1);
	psi_image.QuickAndDirtyReadSlice(input_best_psi_filename.ToStdString(), 1);
	theta_image.QuickAndDirtyReadSlice(input_best_theta_filename.ToStdString(), 1);
	phi_image.QuickAndDirtyReadSlice(input_best_phi_filename.ToStdString(), 1);

	output_image.Allocate(mip_image.logical_x_dimension, mip_image.logical_y_dimension, 1);
	output_image.SetToConstant(0.0f);

	input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices());
	input_reconstruction.ForwardFFT();
	input_reconstruction.MultiplyByConstant(sqrtf(input_reconstruction.logical_x_dimension * input_reconstruction.logical_y_dimension * sqrtf(input_reconstruction.logical_z_dimension)));
	//input_reconstruction.CosineMask(0.1, 0.01, true);
	//input_reconstruction.Whiten();
	//if (first_search_position == 0) input_reconstruction.QuickAndDirtyWriteSlices("/tmp/filter.mrc", 1, input_reconstruction.logical_z_dimension);
	input_reconstruction.ZeroCentralPixel();
	input_reconstruction.SwapRealSpaceQuadrants();

	// assume cube

	current_projection.Allocate(input_reconstruction.logical_x_dimension, input_reconstruction.logical_x_dimension, false);

	// loop until the found peak is below the threhold

	while (1==1)
	{

		// look for a peak..

		current_peak = mip_image.FindPeakWithIntegerCoordinates();
		if (current_peak.value < wanted_threshold) break;

		// ok we have peak..

		number_of_peaks_found++;

		// get angles and mask out the local area so it won't be picked again..

		float sq_dist_x, sq_dist_y;
		long address = 0;

		int i,j;

		current_peak.x = current_peak.x + mip_image.physical_address_of_box_center_x;
		current_peak.y = current_peak.y + mip_image.physical_address_of_box_center_y;

		wxPrintf("Peak = %f, %f, %f : %f\n", current_peak.x, current_peak.y, current_peak.value);

		for ( j = 0; j < mip_image.logical_y_dimension; j ++ )
		{
			sq_dist_y = float(pow(j-current_peak.y, 2));
			for ( i = 0; i < mip_image.logical_x_dimension; i ++ )
			{
				sq_dist_x = float(pow(i-current_peak.x,2));

				// The square centered at the pixel
				if ( sq_dist_x + sq_dist_y <= min_peak_radius )
				{
					mip_image.real_values[address] = -FLT_MAX;
				}

				if (sq_dist_x == 0 && sq_dist_y == 0)
				{
					current_phi = phi_image.real_values[address];
					current_theta = theta_image.real_values[address];
					current_psi = psi_image.real_values[address];
				}

				address++;
			}
			address += mip_image.padding_jump_value;
		}


			// ok get a projection


		angles.Init(current_phi, current_theta, current_psi, 0.0, 0.0);
		input_reconstruction.ExtractSlice(current_projection, angles, 1.0, false);
		current_projection.SwapRealSpaceQuadrants();
		current_projection.MultiplyByConstant(sqrtf(current_projection.logical_x_dimension * current_projection.logical_y_dimension));
		current_projection.BackwardFFT();
		current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges());

		// insert it into the output image

		output_image.InsertOtherImageAtSpecifiedPosition(&current_projection, current_peak.x - output_image.physical_address_of_box_center_x, current_peak.y - output_image.physical_address_of_box_center_y, 0, 0.0f);

	}

	// save the output image

	output_image.QuickAndDirtyWriteSlice(output_result_image_filename.ToStdString(), 1);

	if (is_running_locally == true)
	{
		wxPrintf("\nFound %i peaks.\n\n", number_of_peaks_found);
		wxPrintf("\nMatch Template: Normal termination\n");
		wxDateTime finish_time = wxDateTime::Now();
		wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
	}


	return true;
}
