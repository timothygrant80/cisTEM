#include "../../core/core_headers.h"

class
LocalResolutionFinalize : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};


IMPLEMENT_APP(LocalResolutionFinalize)

// override the DoInteractiveUserInput

void LocalResolutionFinalize::DoInteractiveUserInput()
{

	UserInput *my_input = new UserInput("LocalResolutionFinalize", 0.1);

	ImageFile input_image_file;

	wxString input_volume_fn	= my_input->GetFilenameFromUser("First input volume", "The first input 3D reconstruction used for FSC calculation. If more than one volumes are needed, make sure there is the suffix looks like _1.mrc", "my_reconstruction_1.mrc", true);
	input_image_file.OpenFile(input_volume_fn.ToStdString(), false, false);
	int	num_slices_per_volume	= my_input->GetIntFromUser("Number of local res slices per volume", "Number of slices of the local resolution map present in each volume", "1", 1, input_image_file.ReturnNumberOfSlices());
	int sampling_step			= my_input->GetIntFromUser("Sampling step","How frequently the local resolution was estimated","2",1,9999);
	wxString output_volume_fn 	= my_input->GetFilenameFromUser("Output volume","Local resolution map volume","local_resolution.mrc",false);

	input_image_file.CloseFile();

	delete my_input;

	my_current_job.Reset(4);
	my_current_job.ManualSetArguments("tiit", input_volume_fn.ToUTF8().data(), num_slices_per_volume, sampling_step, output_volume_fn.ToUTF8().data());
}


bool LocalResolutionFinalize::DoCalculation()
{
	wxString input_volume_fn				= my_current_job.arguments[0].ReturnStringArgument();
	int num_slices_to_read_from_each_volume	= my_current_job.arguments[1].ReturnIntegerArgument();
	int sampling_step						= my_current_job.arguments[2].ReturnIntegerArgument();
	wxString output_volume_fn				= my_current_job.arguments[3].ReturnStringArgument();

	// Local variables
	wxFileName input_volume_wxfn = wxFileName(input_volume_fn);
	wxFileName current_input_wxfn;
	wxRegEx reSuffix("_[[:digit:]]+$",wxRE_EXTENDED);
	Image combined_resolution_map;
	Image final_resolution_map;
	Image temp_image;
	ImageFile current_input_imagefile;
	wxString wx_str;
	int first_slice;
	int last_slice;
	int index_of_current_file;
	int num_regex_matches;

	// Open the first file
	current_input_imagefile.OpenFile(input_volume_fn.ToStdString(),false,false);
	int number_of_slices = current_input_imagefile.ReturnNumberOfSlices();

	// Set up the final map
	combined_resolution_map.Allocate(current_input_imagefile.ReturnXSize(), current_input_imagefile.ReturnYSize(), current_input_imagefile.ReturnNumberOfSlices(), true);

	// Loop over volumes
	first_slice = 1;
	index_of_current_file = 1;
	while (first_slice <= number_of_slices)
	{
		last_slice = std::min(first_slice + num_slices_to_read_from_each_volume - 1,number_of_slices);

		// Work out filename
		current_input_wxfn = input_volume_wxfn;
		current_input_wxfn.ClearExt();
		wx_str = current_input_wxfn.GetFullName();
		num_regex_matches = reSuffix.Replace(&wx_str,wxString::Format("_%i.mrc",index_of_current_file));

		// Open the file & read in the relevant sections
		current_input_imagefile.OpenFile(wx_str.ToStdString(), false);
		temp_image.ReadSlices(&current_input_imagefile, long(first_slice), long(last_slice));

		MyDebugAssertTrue(temp_image.logical_x_dimension == combined_resolution_map.logical_x_dimension && temp_image.logical_y_dimension == combined_resolution_map.logical_y_dimension,"Oops... dimension mismatch");

		// Copy the sections into the final volume
		long counter_in_destination = (first_slice-1) * (temp_image.logical_x_dimension+2) * (temp_image.logical_y_dimension);
		long counter_in_source = 0;

		for (int k = 0; k < temp_image.logical_z_dimension; k ++)
		{
			for (int j = 0; j < temp_image.logical_y_dimension; j ++)
			{
				for (int i = 0; i < temp_image.logical_x_dimension + 2; i ++)
				{
					combined_resolution_map.real_values[counter_in_destination] = temp_image.real_values[counter_in_source];
					counter_in_destination++;
					counter_in_source++;
				}
			}
		}

		// Increment for the next iteration
		first_slice += num_slices_to_read_from_each_volume;
		index_of_current_file++;
	}

	/*
	 * Now we have a full volume, but unless the local resolution was estimated at every
	 * single voxel, we have to interpolate between estimates
	 */
	final_resolution_map = combined_resolution_map;
	if (sampling_step != 1)
	{
		final_resolution_map.SetToConstant(0.0);
		/*
		 * Let's do interpolation. We loop over the combined volume, and anytime we have
		 * an estimate, we interpolate it around it (linearly)
		 */
		long counter_in = 0;
		long counter_out = 0;
		int k_out_min, k_out_max, k_out;
		int j_out_min, j_out_max, j_out;
		int i_out_min, i_out_max, i_out;
		float weight_k, weight_j, weight_i;
		float inverse_sampling_step = 1.0 / float(sampling_step);
		for (int k_in = 0; k_in < combined_resolution_map.logical_z_dimension; k_in ++)
		{
			k_out_min = std::max(0,k_in-sampling_step+1);
			k_out_max = std::min(combined_resolution_map.logical_z_dimension-1,k_in+sampling_step);
			for (int j_in = 0; j_in < combined_resolution_map.logical_y_dimension; j_in ++)
			{
				j_out_min = std::max(0,j_in-sampling_step+1);
				j_out_max = std::min(combined_resolution_map.logical_y_dimension-1,j_in+sampling_step);
				for (int i_in = 0; i_in < combined_resolution_map.logical_x_dimension; i_in ++)
				{
					if (combined_resolution_map.real_values[counter_in] > 0.0)
					{
						i_out_min = std::max(0,i_in-sampling_step+1);
						i_out_max = std::min(combined_resolution_map.logical_x_dimension-1,i_in+sampling_step);

						// Let's interpolate
						for (k_out = k_out_min; k_out <= k_out_max; k_out++)
						{
							weight_k = 1.0 - float(abs(k_in-k_out)) * inverse_sampling_step;
							for (j_out = j_out_min; j_out <= j_out_max; j_out++)
							{
								counter_out = final_resolution_map.ReturnReal1DAddressFromPhysicalCoord(i_out_min, j_out, k_out);
								weight_j = 1.0 - float(abs(j_in-j_out)) * inverse_sampling_step;
								for (i_out = i_out_min; i_out <= i_out_max; i_out++)
								{
									weight_i = 1.0 - float(abs(i_in-i_out)) * inverse_sampling_step;

									final_resolution_map.real_values[counter_out] += combined_resolution_map.real_values[counter_in] * weight_k * weight_j * weight_i;

									counter_out++;
								}
							}
						}

					}

					counter_in ++;
				}
				counter_in += combined_resolution_map.padding_jump_value;
			}
		}

	}

	/*
	 * High-pass filter the resolution map.
	 * This is to compensate for the fact that a local resolution estimate is computed from
	 * a local volume, which may be on the order of 20 Angstroms dimension and we want
	 * better locality than that for our estimate.
	 * Not working (yet?) - deconvolution is not easy... ;)
	 */
	/*
	final_resolution_map.ForwardFFT();
	final_resolution_map.ApplyBFactor(-100.0);
	final_resolution_map.CosineMask(0.25, 0.1);
	final_resolution_map.BackwardFFT();
	*/
	/*
	{
		const int box_size = 20;
		// Generate the real-space mask that was used
		Image box_mask(final_resolution_map);
		box_mask.SetToConstant(1.0);
		//box_mask.CosineMask(box_size / 4, box_size / 2, false, true, 0.0);
		box_mask.TriangleMask(float(box_size/2));
		//box_mask.MultiplyByConstant(box_mask.number_of_real_space_pixels);
		//box_mask.SetToConstant(0.0);
		//box_mask.real_values[box_mask.ReturnReal1DAddressFromPhysicalCoord(box_mask.physical_address_of_box_center_x, box_mask.physical_address_of_box_center_y, box_mask.physical_address_of_box_center_z)] = float(box_mask.number_of_real_space_pixels);
		//box_mask.real_values[box_mask.ReturnReal1DAddressFromPhysicalCoord(box_mask.physical_address_of_box_center_x, box_mask.physical_address_of_box_center_y, box_mask.physical_address_of_box_center_z)] = 1.0;
		box_mask.MultiplyByConstant(float(box_mask.number_of_real_space_pixels)/box_mask.ReturnSumOfRealValues());
		box_mask.QuickAndDirtyWriteSlices("dbg_mask.mrc", 1, box_mask.logical_z_dimension);
		//
		//Image box_mask_amplitudes;
		box_mask.ForwardFFT(false);
		box_mask.SwapRealSpaceQuadrants();
		box_mask.DivideByConstant(sqrt(float(box_mask.number_of_real_space_pixels))); // proper scaling to satisfy Parseval's theorem
		//box_mask.AddConstant(0.01);
		Image box_mask_amplitudes;
		box_mask.ComputeAmplitudeSpectrum(&box_mask_amplitudes);
		box_mask_amplitudes.QuickAndDirtyWriteSlices("dbg_mask_amp.mrc", 1, box_mask_amplitudes.logical_z_dimension);


		// Apply this as a filter
		final_resolution_map.QuickAndDirtyWriteSlices("dbg_res_beforefilter.mrc", 1, final_resolution_map.logical_z_dimension);
		final_resolution_map.ForwardFFT();
		final_resolution_map.SwapRealSpaceQuadrants();
		final_resolution_map.DivideByConstant(sqrt(float(final_resolution_map.number_of_real_space_pixels)));
		//final_resolution_map.DividePixelWise(box_mask);
		//final_resolution_map.MultiplyPixelWise(box_mask);

		std::complex<float> cvalue;
		for (long pixel_counter = 0; pixel_counter < box_mask.real_memory_allocated / 2; pixel_counter++)
		{
			cvalue = box_mask.complex_values[pixel_counter];
			if (abs(cvalue) < 0.001) cvalue = 0.001;
			final_resolution_map.complex_values[pixel_counter] /= cvalue;
		}
		//final_resolution_map.DivideByConstant(sqrt(float(final_resolution_map.number_of_real_space_pixels)));
		//final_resolution_map.MultiplyByConstant(float(final_resolution_map.number_of_real_space_pixels));
		final_resolution_map.SwapRealSpaceQuadrants();
		final_resolution_map.BackwardFFT();
	}
	*/

	// Write out the volume
	MRCFile output_file(output_volume_fn.ToStdString(),true);
	output_file.SetPixelSize(current_input_imagefile.ReturnPixelSize());
	// TODO: avoid calling 4 different metods to set the density statistics - don't we have a method to do this all in one go?
	output_file.SetDensityStatistics(final_resolution_map.ReturnMinimumValue(), final_resolution_map.ReturnMaximumValue(), final_resolution_map.ReturnAverageOfRealValues(), sqrt(final_resolution_map.ReturnVarianceOfRealValues()));
	final_resolution_map.WriteSlices(&output_file, 1, number_of_slices);


	return true;
}
