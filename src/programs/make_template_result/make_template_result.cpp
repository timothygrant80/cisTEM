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
	wxString    input_best_defocus_filename;
	wxString    input_best_pixel_size_filename;
	wxString    output_result_image_filename;
	wxString    output_slab_filename;
	wxString    xyz_coords_filename;

	float wanted_threshold;
	float min_peak_radius;
	float slab_thickness;
	float pixel_size;
	float binning_factor;
	int	result_number;
	int mip_x_dimension = 0;
	int mip_y_dimension = 0;
	bool read_coordinates;

	UserInput *my_input = new UserInput("MakeTemplateResult", 1.00);

	read_coordinates = my_input->GetYesNoFromUser("Read coordinates from file?", "Should the target coordinates be read from a file instead of search results?", "No");
	if (! read_coordinates)
	{
		input_mip_filename = my_input->GetFilenameFromUser("Input MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", false);
		input_best_psi_filename = my_input->GetFilenameFromUser("Input psi file", "The file containing the best psi image", "psi.mrc", false);
		input_best_theta_filename = my_input->GetFilenameFromUser("Input theta file", "The file containing the best psi image", "theta.mrc", false);
		input_best_phi_filename = my_input->GetFilenameFromUser("Input phi file", "The file containing the best psi image", "phi.mrc", false);
		input_best_defocus_filename = my_input->GetFilenameFromUser("Input defocus file", "The file with the best defocus image", "defocus.mrc", true);
		input_best_pixel_size_filename = my_input->GetFilenameFromUser("Input pixel size file", "The file with the best pixel size image", "pixel_size.mrc", true);
		xyz_coords_filename = my_input->GetFilenameFromUser("Output x,y,z coordinate file", "The file for saving the x,y,z coordinates of the found targets", "coordinates.txt", false);
		wanted_threshold = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
		min_peak_radius = my_input->GetFloatFromUser("Min Peak Radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 1.0);
		result_number = my_input->GetIntFromUser("Result number to process", "If input files contain results from several searches, which one should be used?", "1", 1);
	}
	else
	{
		mip_x_dimension = my_input->GetIntFromUser("X-dimension of original MIP", "The x-dimension of the MIP that contained the peaks listed in the input coordinate file", "5760", 100);
		mip_y_dimension = my_input->GetIntFromUser("Y-dimension of original MIP", "The y-dimension of the MIP that contained the peaks listed in the input coordinate file", "4092", 100);
		xyz_coords_filename = my_input->GetFilenameFromUser("Input x,y,z coordinate file", "The file containing the x,y,z coordinates of the found targets", "coordinates.txt", false);
	}
	input_reconstruction_filename = my_input->GetFilenameFromUser("Input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
	output_result_image_filename = my_input->GetFilenameFromUser("Output 2D projection montage", "The file for saving the found result", "result.mrc", false);
	output_slab_filename = my_input->GetFilenameFromUser("Output slab volume montage", "The file for saving the slab with the found targets", "slab.mrc", false);
	slab_thickness = my_input->GetFloatFromUser("Sample thickness (A)", "The thickness of the sample that was searched", "2000.0", 100.0);
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	binning_factor = my_input->GetFloatFromUser("Binning factor for slab", "Factor to reduce size of output slab", "4.0", 0.0);

	delete my_input;

//	my_current_job.Reset(14);
	my_current_job.ManualSetArguments("ttttttttttfffffbiii",	input_reconstruction_filename.ToUTF8().data(),
													input_mip_filename.ToUTF8().data(),
													input_best_psi_filename.ToUTF8().data(),
													input_best_theta_filename.ToUTF8().data(),
													input_best_phi_filename.ToUTF8().data(),
													input_best_defocus_filename.ToUTF8().data(),
													input_best_pixel_size_filename.ToUTF8().data(),
													output_result_image_filename.ToUTF8().data(),
													output_slab_filename.ToUTF8().data(),
													xyz_coords_filename.ToUTF8().data(),
													wanted_threshold,
													min_peak_radius,
													slab_thickness,
													pixel_size, binning_factor,
													read_coordinates,
													mip_x_dimension, mip_y_dimension,
													result_number);
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
	wxString	input_best_defocus_filename = my_current_job.arguments[5].ReturnStringArgument();
	wxString	input_best_pixel_size_filename = my_current_job.arguments[6].ReturnStringArgument();
	wxString	output_result_image_filename = my_current_job.arguments[7].ReturnStringArgument();
	wxString	output_slab_filename = my_current_job.arguments[8].ReturnStringArgument();
	wxString	xyz_coords_filename = my_current_job.arguments[9].ReturnStringArgument();
	float		wanted_threshold = my_current_job.arguments[10].ReturnFloatArgument();
	float		min_peak_radius = my_current_job.arguments[11].ReturnFloatArgument();
	float		slab_thickness = my_current_job.arguments[12].ReturnFloatArgument();
	float		pixel_size = my_current_job.arguments[13].ReturnFloatArgument();
	float		binning_factor = my_current_job.arguments[14].ReturnFloatArgument();
	bool	 	read_coordinates = my_current_job.arguments[15].ReturnBoolArgument();
	int 		mip_x_dimension = my_current_job.arguments[16].ReturnIntegerArgument();
	int 		mip_y_dimension = my_current_job.arguments[17].ReturnIntegerArgument();
	int 		result_number = my_current_job.arguments[18].ReturnIntegerArgument();

	float padding = 2.0f;

	ImageFile input_reconstruction_file;

	input_reconstruction_file.OpenFile(input_reconstruction_filename.ToStdString(), false);

	Image output_image;
	Image mip_image;
	Image psi_image;
	Image theta_image;
	Image phi_image;
	Image defocus_image;
	Image pixel_size_image;
	Image input_reconstruction;
	Image binned_reconstruction;
	Image rotated_reconstruction;
	Image current_projection;
	Image padded_projection;
	Image slab;

	Peak current_peak;

	AnglesAndShifts angles;

	float current_phi;
	float current_theta;
	float current_psi;
	float current_defocus;
	float current_pixel_size;

	int number_of_peaks_found = 0;
	int slab_thickness_in_pixels;
	int binned_dimension_3d;
	float binned_pixel_size;
	float max_density;
	float sq_dist_x, sq_dist_y;
	long address;
	long text_file_access_type;
	int i,j;

	float coordinates[8];
	if (read_coordinates) text_file_access_type = OPEN_TO_READ;
	else text_file_access_type = OPEN_TO_WRITE;
	NumericTextFile coordinate_file(xyz_coords_filename, text_file_access_type, 8);
	if (! read_coordinates)
	{
		coordinate_file.WriteCommentLine("         Psi          Theta            Phi              X              Y              Z      PixelSize           Peak");

		mip_image.QuickAndDirtyReadSlice(input_mip_filename.ToStdString(), result_number);
		psi_image.QuickAndDirtyReadSlice(input_best_psi_filename.ToStdString(), result_number);
		theta_image.QuickAndDirtyReadSlice(input_best_theta_filename.ToStdString(), result_number);
		phi_image.QuickAndDirtyReadSlice(input_best_phi_filename.ToStdString(), result_number);
		defocus_image.QuickAndDirtyReadSlice(input_best_defocus_filename.ToStdString(), result_number);
		pixel_size_image.QuickAndDirtyReadSlice(input_best_pixel_size_filename.ToStdString(), result_number);
		mip_x_dimension = mip_image.logical_x_dimension;
		mip_y_dimension = mip_image.logical_y_dimension;

		min_peak_radius = powf(min_peak_radius, 2);
	}

	output_image.Allocate(mip_x_dimension, mip_y_dimension, 1);
	output_image.SetToConstant(0.0f);

	input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices());
	binned_reconstruction.CopyFrom(&input_reconstruction);
	binned_dimension_3d = myroundint(float(input_reconstruction.logical_x_dimension) / binning_factor);
	if (IsOdd(binned_dimension_3d)) binned_dimension_3d++;
	binning_factor = float(input_reconstruction.logical_x_dimension) / float(binned_dimension_3d);
	binned_pixel_size = pixel_size * binning_factor;
	slab_thickness_in_pixels = myroundint(slab_thickness / binned_pixel_size);
	wxPrintf("\nSlab dimensions = %i %i %i\n", myroundint(mip_x_dimension / binning_factor), myroundint(mip_y_dimension / binning_factor), slab_thickness_in_pixels);

	slab.Allocate(myroundint(mip_x_dimension / binning_factor), myroundint(mip_y_dimension / binning_factor), slab_thickness_in_pixels);
	slab.SetToConstant(0.0f);

	if (binned_dimension_3d != input_reconstruction.logical_x_dimension)
	{
		binned_reconstruction.ForwardFFT();
		binned_reconstruction.Resize(binned_dimension_3d, binned_dimension_3d, binned_dimension_3d);
		binned_reconstruction.BackwardFFT();
	}
	max_density = binned_reconstruction.ReturnAverageOfMaxN();
	binned_reconstruction.DivideByConstant(max_density);

	if (padding != 1.0f)
	{
		input_reconstruction.Resize(input_reconstruction.logical_x_dimension * padding, input_reconstruction.logical_y_dimension * padding, input_reconstruction.logical_z_dimension * padding, input_reconstruction.ReturnAverageOfRealValuesOnEdges());
	}
	input_reconstruction.ForwardFFT();
	input_reconstruction.MultiplyByConstant(sqrtf(input_reconstruction.logical_x_dimension * input_reconstruction.logical_y_dimension * sqrtf(input_reconstruction.logical_z_dimension)));
	//input_reconstruction.CosineMask(0.1, 0.01, true);
	//input_reconstruction.Whiten();
	//if (first_search_position == 0) input_reconstruction.QuickAndDirtyWriteSlices("/tmp/filter.mrc", 1, input_reconstruction.logical_z_dimension);
	input_reconstruction.ZeroCentralPixel();
	input_reconstruction.SwapRealSpaceQuadrants();

	// assume cube

	current_projection.Allocate(input_reconstruction.logical_x_dimension, input_reconstruction.logical_x_dimension, false);
	if (padding != 1.0f) padded_projection.Allocate(input_reconstruction_file.ReturnXSize() * padding, input_reconstruction_file.ReturnXSize() * padding, false);

	// loop until the found peak is below the threshold

	wxPrintf("\n");
	while (1 == 1)
	{
		if (! read_coordinates)
		{
			// look for a peak..

			current_peak = mip_image.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, input_reconstruction_file.ReturnXSize() / 2 + 1);
			if (current_peak.value < wanted_threshold) break;

			// ok we have peak..

			number_of_peaks_found++;

			// get angles and mask out the local area so it won't be picked again..

			address = 0;

			current_peak.x = current_peak.x + mip_image.physical_address_of_box_center_x;
			current_peak.y = current_peak.y + mip_image.physical_address_of_box_center_y;

//			wxPrintf("Peak = %f, %f, %f : %f\n", current_peak.x, current_peak.y, current_peak.value);

			for ( j = 0; j < mip_y_dimension; j ++ )
			{
				sq_dist_y = float(pow(j-current_peak.y, 2));
				for ( i = 0; i < mip_x_dimension; i ++ )
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
						current_defocus = defocus_image.real_values[address];
						current_pixel_size = pixel_size_image.real_values[address];
					}

					address++;
				}
				address += mip_image.padding_jump_value;
			}
			coordinates[0] = current_psi;
			coordinates[1] = current_theta;
			coordinates[2] = current_phi;
			coordinates[3] = current_peak.x * pixel_size;
			coordinates[4] = current_peak.y * pixel_size;
//			coordinates[5] = binned_pixel_size * (slab.physical_address_of_box_center_z - binned_reconstruction.physical_address_of_box_center_z) - current_defocus;
//			coordinates[5] = binned_pixel_size * slab.physical_address_of_box_center_z - current_defocus;
			coordinates[5] = - current_defocus;
			coordinates[6] = current_pixel_size;
			coordinates[7] = current_peak.value;
			coordinate_file.WriteLine(coordinates);
		}
		else
		{
			coordinate_file.ReadLine(coordinates);
			number_of_peaks_found++;
			current_psi = coordinates[0];
			current_theta = coordinates[1];
			current_phi = coordinates[2];
			current_peak.x = coordinates[3] / pixel_size;
			current_peak.y = coordinates[4] / pixel_size;
			current_defocus = - coordinates[5];
			current_pixel_size = coordinates[6];
			current_peak.value = coordinates[7];
		}

		wxPrintf("Peak %4i at x, y, psi, theta, phi, defocus, pixel size = %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f : %10.6f\n", number_of_peaks_found, current_peak.x, current_peak.y, current_psi, current_theta, current_phi, current_defocus, current_pixel_size, current_peak.value);

			// ok get a projection

		angles.Init(current_phi, current_theta, current_psi, 0.0, 0.0);

		if (padding != 1.0f)
		{
			input_reconstruction.ExtractSlice(padded_projection, angles, 1.0f, false);
			padded_projection.SwapRealSpaceQuadrants();
			padded_projection.BackwardFFT();
			padded_projection.ClipInto(&current_projection);
			current_projection.ForwardFFT();
		}
		else
		{
			input_reconstruction.ExtractSlice(current_projection, angles, 1.0f, false);
			current_projection.SwapRealSpaceQuadrants();
		}

		angles.Init(-current_psi, -current_theta, -current_phi, 0.0, 0.0);
		rotated_reconstruction.CopyFrom(&binned_reconstruction);
		rotated_reconstruction.Rotate3DByRotationMatrixAndOrApplySymmetry(angles.euler_matrix);

		current_projection.MultiplyByConstant(sqrtf(current_projection.logical_x_dimension * current_projection.logical_y_dimension));
		current_projection.BackwardFFT();
		current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges());

		// insert it into the output image

		output_image.InsertOtherImageAtSpecifiedPosition(&current_projection, current_peak.x - output_image.physical_address_of_box_center_x, current_peak.y - output_image.physical_address_of_box_center_y, 0, 0.0f);
		slab.InsertOtherImageAtSpecifiedPosition(&rotated_reconstruction, myroundint((current_peak.x - output_image.physical_address_of_box_center_x) / binning_factor), myroundint((current_peak.y - output_image.physical_address_of_box_center_y) / binning_factor), - myroundint(current_defocus / binned_pixel_size), 0.0f);

		if (read_coordinates && coordinate_file.number_of_lines == number_of_peaks_found) break;
	}

	// save the output image

	output_image.QuickAndDirtyWriteSlice(output_result_image_filename.ToStdString(), 1, true, pixel_size);
	slab.QuickAndDirtyWriteSlices(output_slab_filename.ToStdString(), 1, slab_thickness_in_pixels, true, binned_pixel_size);

	if (is_running_locally == true)
	{
		wxPrintf("\nFound %i peaks.\n\n", number_of_peaks_found);
		wxPrintf("\nMake Template Results: Normal termination\n");
		wxDateTime finish_time = wxDateTime::Now();
		wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
	}

	return true;
}
