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
	 * single voxel, we have gaps between the estimates. Before we interpolate in
	 * between them, let's remove "hot spots": voxels where the local resolution estimate
	 * is much worse than the neighbors. This is usually a symptom of a numerical glitch
	 * in the very low-radius shells.
	 */
	// TODO: make this part faster by using address counters rather than the addressing methods
	const bool remove_hotspots = true;
	if (remove_hotspots)
	{
		int k_out_min, k_out_max, k_out;
		int j_out_min, j_out_max, j_out;
		int i_out_min, i_out_max, i_out;
		float weight_k, weight_j, weight_i;
		float inverse_sampling_step = 1.0 / float(sampling_step);
		long address_in;
		long address_neighbor;

		Image temp_combined_map;
		temp_combined_map = combined_resolution_map;
		bool is_an_outlier_hotspot;
		EmpiricalDistribution neighborhood;
		Curve neighborhood_histogram;
		float neighborhood_worst_res;
		float neighborhood_second_worst_res;
		for (int k_in = sampling_step; k_in < combined_resolution_map.logical_z_dimension-sampling_step; k_in ++)
		{
			k_out_min = k_in-sampling_step;
			k_out_max = k_in+sampling_step;
			for (int j_in = sampling_step; j_in < combined_resolution_map.logical_y_dimension-sampling_step; j_in ++)
			{
				j_out_min = j_in-sampling_step;
				j_out_max = j_in+sampling_step;
				for (int i_in = sampling_step; i_in < combined_resolution_map.logical_x_dimension-sampling_step; i_in ++)
				{
					address_in = combined_resolution_map.ReturnReal1DAddressFromPhysicalCoord(i_in, j_in, k_in);
					if (combined_resolution_map.real_values[address_in] > 0.0)
					{
						i_out_min = i_in-sampling_step;
						i_out_max = i_in+sampling_step;

						// Let's accumulate the local resolution values for all our neighbors
						neighborhood.Reset();
						neighborhood_worst_res = 0.0;
						neighborhood_second_worst_res = 0.0;
						MyDebugAssertTrue(k_out_min + 2*sampling_step == k_out_max,"Oops. bad k_out_min and k_out_max");
						for (k_out = k_out_min; k_out <= k_out_max; k_out += sampling_step)
						{
							for (j_out = j_out_min; j_out <= j_out_max; j_out += sampling_step)
							{
								for (i_out = i_out_min; i_out <= i_out_max; i_out += sampling_step)
								{
									if (i_out == i_in && j_out == j_in && k_out == k_in)
									{
										continue;
									}
									else
									{
										address_neighbor = combined_resolution_map.ReturnReal1DAddressFromPhysicalCoord(i_out, j_out, k_out);
										neighborhood.AddSampleValue(combined_resolution_map.real_values[address_neighbor]);
										if (combined_resolution_map.real_values[address_neighbor] > neighborhood_worst_res)
										{
											neighborhood_second_worst_res = neighborhood_worst_res;
											neighborhood_worst_res = combined_resolution_map.real_values[address_neighbor];
										}
										else if (combined_resolution_map.real_values[address_neighbor] > neighborhood_second_worst_res)
										{
											neighborhood_second_worst_res = combined_resolution_map.real_values[address_neighbor];
										}
									}
								}
							}
						}
						MyDebugAssertTrue(neighborhood.GetNumberOfSamples() == 26, "Unexpected number of neighbors: %i",int(neighborhood.GetNumberOfSamples()));

						/*
						 * If the current voxel is a clear outlier (its resolution is much worse
						 * than the neighborhood), we will replace it
						 */
						is_an_outlier_hotspot = temp_combined_map.real_values[address_in] > neighborhood_second_worst_res * 3.0;
						if (is_an_outlier_hotspot)
						{
							wxPrintf("Found an outlier hotspot: %i %i %i used to be %f, will be replaced by %f\n",i_in,j_in,k_in,temp_combined_map.real_values[address_in],neighborhood_second_worst_res);
							temp_combined_map.real_values[address_in] = neighborhood_second_worst_res;
						}

					}

				}
			}
		}
		combined_resolution_map = temp_combined_map;
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

	// Write out the volume
	final_resolution_map.WriteSlicesAndFillHeader(output_volume_fn.ToStdString(), current_input_imagefile.ReturnPixelSize());


	return true;
}
