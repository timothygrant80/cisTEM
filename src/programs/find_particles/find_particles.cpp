#include "../../core/core_headers.h"

class
FindParticlesApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();


	private:

};

IMPLEMENT_APP(FindParticlesApp)

void ComputeLocalMeanAndStandardDeviation(Image *micrograph, Image *mask_image, float mask_radius_in_pixels, long number_of_pixels_within_mask, Image *micrograph_local_mean, Image *micrograph_local_stdev);
void ComputeNormalizedCrossCorrelationFunction(Image *micrograph, Image *micrograph_local_stdev, Image *template_image, float mask_radius_in_pixels, long number_of_pixels_within_mask, Image *nccf);
void ComputeScheresPickingFunction(Image *micrograph, Image *micrograph_local_mean, Image *micrograph_local_stdev, Image *template_image, float mask_radius, long number_of_pixels_within_mask, Image *scoring_function);
void SetAreaToIgnore(Image *my_image, int central_pixel_address_x, int central_pixel_address_y, Image *box_image, float wanted_value);

// override the DoInteractiveUserInput

void FindParticlesApp::DoInteractiveUserInput()
{

	UserInput *my_input = new UserInput("FindParticles", 0.0);

	wxString	micrograph_filename		=	my_input->GetFilenameFromUser("Input micrograph filename","The input micrograph, in which we will look for particles","micrograph.mrc",true);
	bool		already_have_templates	=	my_input->GetYesNoFromUser("Do you already have templates available?","Say yes here if you already have a template or templates to use as references for picking, for example projections from an existing 3D reconstruction","no");
	wxString	templates_filename		=	"templates.mrc";
	if (already_have_templates)
	{
				templates_filename		=	my_input->GetFilenameFromUser("Input templates filename","Set of templates to use in the search","templates.mrc",true);
	}
	float		maximum_radius			=	my_input->GetFloatFromUser("Maximum radius of the particle (in pixels)","The maximum radius of the templates, in pixels","32.0",0.0);

	delete my_input;

	my_current_job.Reset(4);
	my_current_job.ManualSetArguments("tbtf",     			 micrograph_filename.ToStdString().c_str(),
															 already_have_templates,
															 templates_filename.ToStdString().c_str(),
															 maximum_radius
															 );


}

// override the do calculation method which will be what is actually run..

bool FindParticlesApp::DoCalculation()
{

	// Get the arguments for this job..
	wxString 	micrograph_filename 		= 	my_current_job.arguments[0].ReturnStringArgument();
	bool		already_have_templates		=	my_current_job.arguments[1].ReturnBoolArgument();
	wxString 	templates_filename			= 	my_current_job.arguments[2].ReturnStringArgument();
	float		mask_radius_in_pixels		=	my_current_job.arguments[3].ReturnFloatArgument();


	// Parameters which could be set by the user
	const int number_of_background_boxes_to_skip = 20;
	const int number_of_background_boxes = 40;

	// Let's decide on a box size for picking
	int box_size_for_picking = mask_radius_in_pixels * 2 + 12;


	// We will estimate the amplitude spectrum of the templates using curve objects
	// with spatial frequency (0.5 is Nyquist) on the X axis
	Curve template_power_spectrum;
	Curve background_power_spectrum;
	Curve current_power_spectrum;
	Curve current_number_of_fourier_elements;
	Curve temp_curve;
	template_power_spectrum.SetupXAxis(0.0,sqrtf(2.0)*0.5,box_size_for_picking);
	background_power_spectrum = template_power_spectrum;
	current_power_spectrum = template_power_spectrum;
	current_number_of_fourier_elements = template_power_spectrum;
	temp_curve = template_power_spectrum;


	// If the user is supplying templates, read them in. If not, generate a template
	if (already_have_templates)
	{
		MRCFile template_file(templates_filename.ToStdString(),false);
		Image template_image;
		for (int template_counter = 0; template_counter < template_file.ReturnNumberOfSlices(); template_counter ++)
		{
			template_image.ReadSlice(&template_file,template_counter+1);
			template_image.ForwardFFT(false);
			template_image.NormalizeFT();
			//template_image.complex_values[0] = (0.0,0.0);
			template_image.Compute1DPowerSpectrumCurve(&current_power_spectrum,&current_number_of_fourier_elements);
			template_power_spectrum.AddWith(&current_power_spectrum);
		}
		template_power_spectrum.MultiplyByConstant(1.0/float(template_file.ReturnNumberOfSlices()));
	}
	else // User did not supply a template, we will generate one
	{
		Image template_image;
		template_image.Allocate(box_size_for_picking,box_size_for_picking,1);
		template_image.SetToConstant(1.0);
		template_image.CosineMask(mask_radius_in_pixels,5.0,false,true,0.0);
		template_image.QuickAndDirtyWriteSlice("dbg_template.mrc",1);
		template_image.ForwardFFT(false);
		template_image.NormalizeFT();
		template_image.Compute1DPowerSpectrumCurve(&template_power_spectrum,&current_number_of_fourier_elements);
	}
	template_power_spectrum.WriteToFile("dbg_template_power.txt");

	// Normalize the curve to turn it into a band-pass filter
	template_power_spectrum.NormalizeMaximumValue();
	template_power_spectrum.SquareRoot();

	// Read in the micrograph
	MRCFile micrograph_file(micrograph_filename.ToStdString(),false);
	MyDebugAssertTrue(micrograph_file.ReturnNumberOfSlices() == 1,"Input micrograph file should only contain one image for now");
	Image micrograph;
	micrograph.ReadSlice(&micrograph_file,1);

	// Write the raw micrograph's spectrum to disk
	micrograph.ForwardFFT(false);
	micrograph.NormalizeFT();
	micrograph.Compute1DPowerSpectrumCurve(&current_power_spectrum, &current_number_of_fourier_elements);
	micrograph.BackwardFFT();
	micrograph.NormalizeFT();
	current_power_spectrum.WriteToFile("dbg_micrograph_power.txt");

	// Band-pass filter the micrograph to emphasize features similar to the templates
	Image micrograph_bp;
	micrograph_bp = micrograph;
	micrograph_bp.ForwardFFT(false);
	micrograph_bp.NormalizeFT();
	micrograph_bp.ApplyCurveFilter(&template_power_spectrum);
	micrograph_bp.BackwardFFT();
	micrograph_bp.NormalizeFT();
	micrograph_bp.QuickAndDirtyWriteSlice("dbg_micrograph_filtered.mrc",1);

	// We will need a few images with the same dimensions as the micrograph
	Image mask_image;
	Image local_mean;
	Image local_sigma;
	mask_image.Allocate(micrograph.logical_x_dimension, micrograph.logical_y_dimension,1);
	local_mean.Allocate(micrograph.logical_x_dimension, micrograph.logical_y_dimension,1);
	local_sigma.Allocate(micrograph.logical_x_dimension, micrograph.logical_y_dimension,1);


	// Prepare a mask
	mask_image.Allocate(micrograph.logical_x_dimension, micrograph.logical_y_dimension,1);
	mask_image.SetToConstant(1.0);
	mask_image.CircleMaskWithValue(mask_radius_in_pixels,0.0);
	long number_of_pixels_within_mask = mask_image.ReturnAverageOfRealValues() * mask_image.logical_x_dimension * mask_image.logical_y_dimension;

	// Compute local average and local sigma
	micrograph_bp.ComputeLocalMeanAndVarianceMaps(&local_mean,&local_sigma,&mask_image,number_of_pixels_within_mask);
	local_mean.QuickAndDirtyWriteSlice("dbg_local_average.mrc",1);
	local_sigma.QuickAndDirtyWriteSlice("dbg_local_variance.mrc",1);
	local_sigma.SetMinimumValue(0.0);
	local_sigma.SquareRootRealValues();
	//local_sigma.ForwardFFT();
	//local_sigma.ApplyCurveFilter(&template_amplitude_spectrum);
	//local_sigma.BackwardFFT();

	// Debug dumps
	local_mean.QuickAndDirtyWriteSlice("dbg_local_average.mrc",1);
	local_sigma.QuickAndDirtyWriteSlice("dbg_local_sigma.mrc",1);


	// Let's look for the areas of lowest variance, which we will assume are plain ice, so we can work out a whitening filter later on
	local_sigma.MultiplyByConstant(-1.0);
	background_power_spectrum.ZeroYData();
	Image box;
	box.Allocate(mask_radius_in_pixels * 2 + 2, mask_radius_in_pixels * 2 + 2, 1);
	wxPrintf("DBG: micrograph var, std = %f, %f\n",micrograph.ReturnVarianceOfRealValues(),sqrt(micrograph.ReturnVarianceOfRealValues()));

	for (int background_box_counter = 0; background_box_counter < number_of_background_boxes; background_box_counter ++ )
	{
		// Find the area of lowest variance
		local_sigma.QuickAndDirtyWriteSlice("dbg_latest_variance.mrc",background_box_counter+1);
		Peak my_peak = local_sigma.FindPeakWithIntegerCoordinates(0.0,FLT_MAX,box.physical_address_of_box_center_x+1);

		if (background_box_counter >= number_of_background_boxes_to_skip) {
			// Box out an image from the micrograph at that location
			micrograph.ClipInto(&box,0.0,false,1.0,int(my_peak.x),int(my_peak.y),0);
			wxPrintf("Boxed out background at position %i, %i = %i, %i; peak value = %f\n",int(my_peak.x),int(my_peak.y),int(my_peak.x)+local_sigma.physical_address_of_box_center_x,int(my_peak.y)+local_sigma.physical_address_of_box_center_y,my_peak.value);
			box.QuickAndDirtyWriteSlice("dbg_background_box.mrc",background_box_counter+1-number_of_background_boxes_to_skip);
			box.ForwardFFT(false);
			box.NormalizeFT();
			box.Compute1DPowerSpectrumCurve(&current_power_spectrum,&current_number_of_fourier_elements);
			background_power_spectrum.AddWith(&current_power_spectrum);
		}

		// Before we look for the next background box, we need to set the pixels we have already extracted from
		//  the variance map to a terrible value so they don't get picked again
		SetAreaToIgnore(&local_sigma,int(my_peak.x) + local_sigma.physical_address_of_box_center_x, int(my_peak.y) + local_sigma.physical_address_of_box_center_y,&box,-99999.99); // TODO: use a better value, such as the minimum value found in the image

	}
	background_power_spectrum.MultiplyByConstant(1.0/float(number_of_background_boxes - number_of_background_boxes_to_skip));
	background_power_spectrum.WriteToFile("dbg_background_spectrum.txt");


	// average_amplitude_spectrum should now contain a decent estimate of the the input micrograph's noise spectrum

	// Next, we need to whiten the noise in the micrograph and ensure that at each pixel
	// it has a variance of 1.0
	for (int counter = 0; counter < background_power_spectrum.number_of_points; counter ++ )
	{
		if (background_power_spectrum.data_y[counter] > 0.0)
		{
			background_power_spectrum.data_y[counter] = 1.0 / sqrtf(background_power_spectrum.data_y[counter]);
		}
	}
	background_power_spectrum.WriteToFile("dbg_background_whitening_filter.txt");
	micrograph.ForwardFFT(false);
	micrograph.NormalizeFT();
	micrograph.ApplyCurveFilter(&background_power_spectrum);
	micrograph.BackwardFFT();
	micrograph.NormalizeFT();
	micrograph.QuickAndDirtyWriteSlice("dbg_micrograph_whitened.mrc",1);
	wxPrintf("DBG: micrograph var, std = %f, %f\n",micrograph.ReturnVarianceOfRealValues(),sqrt(micrograph.ReturnVarianceOfRealValues()));

	// Check the micrograph amplitude spectrum
	micrograph.ForwardFFT(false);
	micrograph.NormalizeFT();
	micrograph.Compute1DPowerSpectrumCurve(&current_power_spectrum,&current_number_of_fourier_elements);
	current_power_spectrum.WriteToFile("dbg_micrograph_whitened_spectrum.txt");

	// Check the background boxes again, recompute their average amplitude spectrum
	temp_curve.ZeroYData();
	EmpiricalDistribution dist(false);
	for (int background_box_counter = 0; background_box_counter < number_of_background_boxes; background_box_counter ++ )
	{
		if (background_box_counter >= number_of_background_boxes_to_skip) {
			box.QuickAndDirtyReadSlice("dbg_background_box.mrc",background_box_counter+1-number_of_background_boxes_to_skip);
			dist = box.ReturnDistributionOfRealValues();
			wxPrintf("Background box %i of %i, mean = %f, std = %f\n",background_box_counter+1, number_of_background_boxes,dist.GetSampleMean(),sqrtf(dist.GetSampleVariance()));
			box.ForwardFFT(false);
			box.NormalizeFT();
			box.ApplyCurveFilter(&background_power_spectrum);
			box.Compute1DPowerSpectrumCurve(&current_power_spectrum,&current_number_of_fourier_elements);
			temp_curve.AddWith(&current_power_spectrum);
			box.BackwardFFT();
			box.NormalizeFT();
			dist = box.ReturnDistributionOfRealValues();
			box.QuickAndDirtyWriteSlice("dbg_background_box_whitened.mrc",background_box_counter+1-number_of_background_boxes_to_skip);
		}
	}
	temp_curve.MultiplyByConstant(1.0/float(number_of_background_boxes - number_of_background_boxes_to_skip));
	temp_curve.WriteToFile("dbg_whitened_background_spectrum.txt");



	return true;
}

// The box_image is just used to get dimensions and for addressing convenience
void SetAreaToIgnore(Image *my_image, int central_pixel_address_x, int central_pixel_address_y, Image *box_image, float wanted_value)
{

	const int box_lbound_x = central_pixel_address_x - box_image->physical_address_of_box_center_x;
	const int box_ubound_x = box_lbound_x + box_image->logical_x_dimension - 1;

	const int box_lbound_y = central_pixel_address_y - box_image->physical_address_of_box_center_y;
	const int box_ubound_y = box_lbound_y + box_image->logical_y_dimension - 1;

	long address = 0;
	for ( int j = 0; j < my_image->logical_y_dimension; j ++ )
	{
		for ( int i = 0; i < my_image->logical_x_dimension; i ++ )
		{
			// The square centered at the pixel
			if ( i >= box_lbound_x && i <= box_ubound_x && j >= box_lbound_y && j <= box_ubound_y )
			{
				my_image->real_values[address] = wanted_value;
			}
			address++;
		}
		address += my_image->padding_jump_value;
	}

}

// Implementation of Equation 8 of Scheres (JSB 2015)
void ComputeScheresPickingFunction(Image *micrograph, Image *micrograph_local_mean, Image *micrograph_local_stdev, Image *template_image, float mask_radius, long number_of_pixels_within_mask, Image *scoring_function)
{
	// We assume the template has been normalized such that its mean is 0.0 and its stdev 1.0 outside the mask
	// (I'm not sure that this is exactly what Sjors does, nor whether this is the correct thing to do)
#ifdef DEBUG
	EmpiricalDistribution template_values_outside_radius = template_image->ReturnDistributionOfRealValues(mask_radius,true);
	MyDebugAssertTrue(abs(template_values_outside_radius.GetSampleMean()) < 0.001,"Template should be normalized to have mean value of 0.0 outside radius");
	const float template_sum_outside_of_mask = template_values_outside_radius.GetSampleSum();
	const float template_sum_of_squares_outside_of_mask = template_values_outside_radius.GetSampleSumOfSquares();
#endif
	EmpiricalDistribution template_values_inside_radius  = template_image->ReturnDistributionOfRealValues(mask_radius,false);
	const float template_sum_inside_of_mask = template_values_inside_radius.GetSampleSum();
	const float template_sum_of_squares_inside_of_mask = template_values_inside_radius.GetSampleSumOfSquares();
	wxPrintf("Template sum of squares inside of mask = %e\n", template_sum_of_squares_inside_of_mask);

	Image template_image_large; // TODO: don't allocate this within this subroutine, pass the memory around

	// We need a version of the masked template padded to the same dimensions as the micrograph
	template_image_large.Allocate(micrograph->logical_x_dimension,micrograph->logical_y_dimension,1);
	template_image->ClipIntoLargerRealSpace2D(&template_image_large,template_image->ReturnAverageOfRealValuesOnEdges());

	//
	long number_of_pixels_in_template = template_image->logical_x_dimension * template_image->logical_y_dimension;

	// Cross-correlation
	if (micrograph->is_in_real_space) micrograph->ForwardFFT(false);
	//template_image_large.DivideByConstant(number_of_pixels_within_mask);
	template_image_large.ForwardFFT(false);
	scoring_function->CopyFrom(micrograph);
	scoring_function->ConjugateMultiplyPixelWise(template_image_large);
	scoring_function->SwapRealSpaceQuadrants();
	scoring_function->BackwardFFT();




	// Equation 6 & equation 8
	long pixel_counter=0;
	for (int counter_y=0; counter_y < micrograph->logical_y_dimension; counter_y++)
	{
		for (int counter_x=0; counter_x < micrograph->logical_x_dimension; counter_x++)
		{
			// Equation 6
			scoring_function->real_values[pixel_counter] =   exp( scoring_function->real_values[pixel_counter] / micrograph_local_stdev->real_values[pixel_counter]
															    - micrograph_local_mean->real_values[pixel_counter] * template_sum_inside_of_mask / micrograph_local_stdev->real_values[pixel_counter]
															    - 0.5 * template_sum_of_squares_inside_of_mask);

			// Equation 8
			//scoring_function->real_values[pixel_counter] = (exp(scoring_function->real_values[pixel_counter]) - 1.0) / (exp(template_sum_of_squares_inside_of_mask/(2.0*number_of_pixels_within_mask)) - 1.0);


			// Equation 8 (with an extra normalization by the number of pixels in the template - I'm not sure why) (this gives a function very very similar to the normalized CCF
			//scoring_function->real_values[pixel_counter] = (exp(scoring_function->real_values[pixel_counter]/number_of_pixels_in_template) - 1.0) / (exp(template_sum_of_squares_inside_of_mask/(2.0*number_of_pixels_within_mask)) - 1.0);


			pixel_counter++;
		}
		pixel_counter += micrograph->padding_jump_value;
	}

}

// It is assumed that the template image has been normalized and masked
void ComputeNormalizedCrossCorrelationFunction(Image *micrograph, Image *micrograph_local_stdev, Image *template_image, float mask_radius_in_pixels, long number_of_pixels_within_mask, Image *nccf)
{

	MyDebugAssertTrue(template_image->is_in_real_space, "Template image must be in real space");

	Image template_image_large; // TODO: don't allocate this within this subroutine, pass the memory around

	// We need a version of the masked template padded to the same dimensions as the micrograph
	template_image_large.Allocate(micrograph->logical_x_dimension,micrograph->logical_y_dimension,1);
	template_image->ClipIntoLargerRealSpace2D(&template_image_large,template_image->ReturnAverageOfRealValuesOnEdges());

	// Let's compute the local normalized correlation
	// First, convolve the masked tempate with the micrograph
	// Then divide the result by the local std dev times the number of pixels within mask
	if (micrograph->is_in_real_space) micrograph->ForwardFFT();
	nccf->CopyFrom(micrograph);
	if (template_image_large.is_in_real_space) template_image_large.ForwardFFT(false);
	MyDebugAssertFalse(micrograph->is_in_real_space,"Micrograph must be in Fourier space");
	MyDebugAssertFalse(template_image_large.is_in_real_space,"Template must be in Fourier space");
	nccf->ConjugateMultiplyPixelWise(template_image_large);
	nccf->SwapRealSpaceQuadrants();
	nccf->BackwardFFT();
	nccf->DivideByConstant(number_of_pixels_within_mask);
	nccf->DividePixelWise(*micrograph_local_stdev);

}


// Compute the local standard deviation in the image, a la Roseman (Ultramicroscopy, 2003), Eqn 6.
void ComputeLocalMeanAndStandardDeviation(Image *micrograph, Image *mask_image, float mask_radius_in_pixels, long number_of_pixels_within_mask, Image *micrograph_local_mean, Image *micrograph_local_stdev)
{

	micrograph_local_stdev->CopyFrom(micrograph);
	micrograph_local_mean->CopyFrom(micrograph);

	MyDebugAssertFalse(micrograph_local_mean->is_in_real_space,"Need to be in Fourier space (local average)");
	MyDebugAssertFalse(mask_image->is_in_real_space,"Need to be in Fourier space (mask image)");
	micrograph_local_mean->MultiplyPixelWise(*mask_image);
	micrograph_local_mean->SwapRealSpaceQuadrants();
	micrograph_local_mean->BackwardFFT();
	micrograph_local_mean->DivideByConstant(number_of_pixels_within_mask);


	// The square of the local average and the square of the micrograph are now needed in preparation
	// for computing the local variance of the micrograph
	MyDebugAssertFalse(micrograph_local_stdev->is_in_real_space,"Thought this was in F space");
	micrograph_local_stdev->CopyFrom(micrograph);
	micrograph_local_stdev->BackwardFFT();
	micrograph_local_stdev->SquareRealValues();


	// Convolute the squared micrograph with the mask image
	MyDebugAssertTrue(micrograph_local_stdev->is_in_real_space,"Thought this would be in R space");
	MyDebugAssertFalse(mask_image->is_in_real_space,"Thought mask was already in Fourier space");
	micrograph_local_stdev->ForwardFFT();
	micrograph_local_stdev->MultiplyPixelWise(*mask_image);
	micrograph_local_stdev->SwapRealSpaceQuadrants();
	micrograph_local_stdev->BackwardFFT();

	// Compute the local variance (Eqn 10 in Roseman 2003)
	micrograph_local_stdev->DivideByConstant(number_of_pixels_within_mask);
	micrograph_local_stdev->SubtractSquaredImage(micrograph_local_mean);

	// Square root to get local standard deviation
	micrograph_local_stdev->SetMinimumValue(0.0); // Otherwise, the image is not save for sqrt
	micrograph_local_stdev->SquareRootRealValues();
}



