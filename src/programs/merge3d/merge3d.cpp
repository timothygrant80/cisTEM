#include "../../core/core_headers.h"

class
    	Merge3DApp : public MyApp {
  public:
	bool DoCalculation( );
	void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(Merge3DApp)

// override the DoInteractiveUserInput

void Merge3DApp::DoInteractiveUserInput( ) {
	wxString output_reconstruction_1;
	wxString output_reconstruction_2;
	wxString output_reconstruction_filtered;
	wxString output_resolution_statistics;
	float	molecular_mass_kDa = 1000.0;
	float	inner_mask_radius  = 0.0;
	float	outer_mask_radius  = 100.0;
	wxString dump_file_seed_1;
	wxString dump_file_seed_2;
	int  	number_of_dump_files;

	UserInput* my_input = new UserInput("Merge3D", 1.01);

	output_reconstruction_1    	= my_input->GetFilenameFromUser("Output reconstruction 1", "The first output 3D reconstruction, calculated form half the data", "my_reconstruction_1.mrc", false);
	output_reconstruction_2    	= my_input->GetFilenameFromUser("Output reconstruction 2", "The second output 3D reconstruction, calculated form half the data", "my_reconstruction_2.mrc", false);
	output_reconstruction_filtered = my_input->GetFilenameFromUser("Output filtered reconstruction", "The final 3D reconstruction, containing from all data and optimally filtered", "my_filtered_reconstruction.mrc", false);
	output_resolution_statistics   = my_input->GetFilenameFromUser("Output resolution statistics", "The text file with the resolution statistics for the final reconstruction", "my_statistics.txt", false);
	molecular_mass_kDa         	= my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	inner_mask_radius          	= my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the final reconstruction in Angstroms", "0.0", 0.0);
	outer_mask_radius          	= my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", inner_mask_radius);
	dump_file_seed_1           	= my_input->GetFilenameFromUser("Seed for input dump filenames for odd particles", "The seed name of the first dump files with the intermediate reconstruction arrays", "dump_file_seed_1_.dat", false);
	dump_file_seed_2           	= my_input->GetFilenameFromUser("Seed for input dump filenames for even particles", "The seed name of the second dump files with the intermediate reconstruction arrays", "dump_file_seed_2_.dat", false);
	number_of_dump_files       	= my_input->GetIntFromUser("Number of dump files", "The number of dump files that should be read from disk and merged", "1", 1);

	delete my_input;

	int  	class_number_for_gui    	= 1;
	bool 	save_orthogonal_views_image = false;
	wxString orthogonal_views_filename   = "";
	float	weiner_nominator        	= 1.0f;
	float	alignment_res           	= 5.0f;
	//    my_current_job.Reset(14);
	my_current_job.ManualSetArguments("ttttfffttibtiff", output_reconstruction_1.ToUTF8( ).data( ),
                                  	output_reconstruction_2.ToUTF8( ).data( ),
                                  	output_reconstruction_filtered.ToUTF8( ).data( ),
                                  	output_resolution_statistics.ToUTF8( ).data( ),
                                  	molecular_mass_kDa,
                                  	inner_mask_radius,
                                  	outer_mask_radius,
                                  	dump_file_seed_1.ToUTF8( ).data( ),
                                  	dump_file_seed_2.ToUTF8( ).data( ),
                                  	class_number_for_gui,
                                  	save_orthogonal_views_image,
                                  	orthogonal_views_filename.ToUTF8( ).data( ),
                                  	number_of_dump_files,
                                  	weiner_nominator,
                                  	alignment_res);
}

// override the do calculation method which will be what is actually run..

bool Merge3DApp::DoCalculation( ) {
	wxString output_reconstruction_1    	= my_current_job.arguments[0].ReturnStringArgument( );
	wxString output_reconstruction_2    	= my_current_job.arguments[1].ReturnStringArgument( );
	wxString output_reconstruction_filtered = my_current_job.arguments[2].ReturnStringArgument( );
	wxString output_resolution_statistics   = my_current_job.arguments[3].ReturnStringArgument( );
	float	molecular_mass_kDa         	= my_current_job.arguments[4].ReturnFloatArgument( );
	float	inner_mask_radius          	= my_current_job.arguments[5].ReturnFloatArgument( );
	float	outer_mask_radius          	= my_current_job.arguments[6].ReturnFloatArgument( );
	wxString dump_file_seed_1           	= my_current_job.arguments[7].ReturnStringArgument( );
	wxString dump_file_seed_2           	= my_current_job.arguments[8].ReturnStringArgument( );
	int  	class_number_for_gui       	= my_current_job.arguments[9].ReturnIntegerArgument( );
	bool 	save_orthogonal_views_image	= my_current_job.arguments[10].ReturnBoolArgument( );
	wxString orthogonal_views_filename  	= my_current_job.arguments[11].ReturnStringArgument( );
	int  	number_of_dump_files       	= my_current_job.arguments[12].ReturnIntegerArgument( );
	float	weiner_nominator           	= my_current_job.arguments[13].ReturnFloatArgument( );
	// FOR LOCRES HACK..
	float alignment_res = my_current_job.arguments[14].ReturnFloatArgument( );

	ResolutionStatistics* resolution_statistics = NULL;
	resolution_statistics                   	= new ResolutionStatistics;

	ReconstructedVolume output_3d(molecular_mass_kDa);
	ReconstructedVolume output_3d1(molecular_mass_kDa);
	ReconstructedVolume output_3d2(molecular_mass_kDa);

	int    	i;
	int    	logical_x_dimension;
	int    	logical_y_dimension;
	int    	logical_z_dimension;
	int    	original_x_dimension;
	int    	original_y_dimension;
	int    	original_z_dimension;
	int    	count;
	int    	intermediate_box_size;
	int    	images_processed;
	float  	mask_volume_fraction;
	float  	mask_falloff = 10.0;
	float  	pixel_size;
	float  	original_pixel_size;
	float  	average_occupancy;
	float  	average_sigma;
	float  	sigma_bfactor_conversion;
	float  	particle_area_in_pixels;
	float  	scale;
	float  	binning_factor;
	wxString   my_symmetry;
	wxDateTime my_time_in;
	wxFileName dump_file_name = wxFileName::FileName(dump_file_seed_1);
	wxString   extension  	= dump_file_name.GetExt( );
	wxString   dump_file;
	bool   	insert_even;
	bool   	center_mass;
	bool   	crop_images;

	NumericTextFile output_statistics_file(output_resolution_statistics, OPEN_TO_WRITE, 7);

	my_time_in = wxDateTime::Now( );
	output_statistics_file.WriteCommentLine("C Merge3D run date and time:           	" + my_time_in.FormatISOCombined(' '));
	output_statistics_file.WriteCommentLine("C Output reconstruction 1:             	" + output_reconstruction_1);
	output_statistics_file.WriteCommentLine("C Output reconstruction 2:             	" + output_reconstruction_2);
	output_statistics_file.WriteCommentLine("C Output filtered reconstruction:      	" + output_reconstruction_filtered);
	output_statistics_file.WriteCommentLine("C Output resolution statistics:        	" + output_resolution_statistics);
	output_statistics_file.WriteCommentLine("C Molecular mass of particle (kDa):    	" + wxString::Format("%f", molecular_mass_kDa));
	output_statistics_file.WriteCommentLine("C Inner mask radius (A):               	" + wxString::Format("%f", inner_mask_radius));
	output_statistics_file.WriteCommentLine("C Outer mask radius (A):               	" + wxString::Format("%f", outer_mask_radius));
	output_statistics_file.WriteCommentLine("C Seed for dump files for odd particles:   " + dump_file_seed_1);
	output_statistics_file.WriteCommentLine("C Seed for dump files for even particles:  " + dump_file_seed_2);
	output_statistics_file.WriteCommentLine("C");

	dump_file = wxFileName::StripExtension(dump_file_seed_1) + wxString::Format("%i", 1) + "." + extension;

	if ( (is_running_locally && DoesFileExist(dump_file)) || (! is_running_locally && DoesFileExistWithWait(dump_file, 90)) ) // C++ standard says if LHS of OR is true, RHS never gets evaluated
	{
    	//
	}
	else {
    	SendError(wxString::Format("Error: Dump file %s not found\n", dump_file));
    	exit(-1);
	}

	Reconstruct3D temp_reconstruction;
	temp_reconstruction.ReadArrayHeader(dump_file, logical_x_dimension, logical_y_dimension, logical_z_dimension,
                                    	original_x_dimension, original_y_dimension, original_z_dimension, images_processed, pixel_size, original_pixel_size,
                                    	average_occupancy, average_sigma, sigma_bfactor_conversion, my_symmetry, insert_even, center_mass);
	wxPrintf("\nReconstruction dimensions = %i, %i, %i, pixel size = %f, symmetry = %s\n", logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, my_symmetry);
	temp_reconstruction.Init(logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_sigma, sigma_bfactor_conversion);
	Reconstruct3D my_reconstruction_1(logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_sigma, sigma_bfactor_conversion, my_symmetry);
	Reconstruct3D my_reconstruction_2(logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_sigma, sigma_bfactor_conversion, my_symmetry);

	wxPrintf("\nReading reconstruction arrays...\n\n");

	for ( count = 1; count <= number_of_dump_files; count++ ) {
    	dump_file = wxFileName::StripExtension(dump_file_seed_1) + wxString::Format("%i", count) + "." + extension;
    	wxPrintf("%s\n", dump_file);
    	if ( (is_running_locally && DoesFileExist(dump_file)) || (! is_running_locally && DoesFileExistWithWait(dump_file, 90)) ) // C++ standard says if LHS of OR is true, RHS never gets evaluated
    	{
        	temp_reconstruction.ReadArrays(dump_file);
        	my_reconstruction_1 += temp_reconstruction;
    	}
    	else {
        	SendError(wxString::Format("Error: Dump file %s not found\n", dump_file));
        	exit(-1);
    	}
	}

	for ( count = 1; count <= number_of_dump_files; count++ ) {
    	dump_file = wxFileName::StripExtension(dump_file_seed_2) + wxString::Format("%i", count) + "." + extension;
    	wxPrintf("%s\n", dump_file);
    	if ( (is_running_locally && DoesFileExist(dump_file)) || (! is_running_locally && DoesFileExistWithWait(dump_file, 90)) ) // C++ standard says if LHS of OR is true, RHS never gets evaluated
    	{
        	temp_reconstruction.ReadArrays(dump_file);
        	my_reconstruction_2 += temp_reconstruction;
    	}
    	else {
        	SendError(wxString::Format("Error: Dump file %s not found\n", dump_file));
        	exit(-1);
    	}
	}

	wxPrintf("\nFinished reading arrays\n");

	output_3d1.FinalizeSimple(my_reconstruction_1, original_x_dimension, original_pixel_size, pixel_size,
                          	inner_mask_radius, outer_mask_radius, mask_falloff, output_reconstruction_1);
	output_3d2.FinalizeSimple(my_reconstruction_2, original_x_dimension, original_pixel_size, pixel_size,
                          	inner_mask_radius, outer_mask_radius, mask_falloff, output_reconstruction_2);

	output_3d.mask_volume_in_voxels = output_3d1.mask_volume_in_voxels;
	my_reconstruction_1 += my_reconstruction_2;
	my_reconstruction_2.FreeMemory( );

	output_3d.FinalizeOptimal(my_reconstruction_1, output_3d1.density_map, output_3d2.density_map,
                          	original_pixel_size, pixel_size, inner_mask_radius, outer_mask_radius, mask_falloff,
                          	center_mass, output_reconstruction_filtered, output_statistics_file, resolution_statistics, weiner_nominator);


	//float orientation_distribution_efficiency = output_3d.ComputeOrientationDistributionEfficiency(my_reconstruction_1);
	//SendInfo(wxString::Format("Orientation distribution efficiency: %0.2f\n",orientation_distribution_efficiency));

	// LOCAL RESOLUTION HACK - REMOVE!!

	///////////// LOCAL RES HACK!! TODO REMOVE!

	// MASKING

	const bool test_locres_filtering = true;
	const int  number_of_threads 	= 64;
	if ( test_locres_filtering ) {
    	{
        	Image local_resolution_volume; // define local_resolution_volume image class
        	Image original_volume; // define original_volume image class
       	// const float krr = 5.0;
       	int box_size;
       	// box_size = int(krr * resolution_statistics->ReturnEstimatedResolution(true) / original_pixel_size);
       	box_size = 18.0f / original_pixel_size; // set box size using original pixel size
       	// if (box_size < 15) box_size = 15;

        	wxPrintf("Will estimate local resolution using a box size of %i\n", box_size);
        	// these values are used in the local_resolution_estimator
        	const float threshold_snr    	= 1;
        	const float threshold_confidence = 2.0;
        	float   	fixed_fsc_threshold  = .95;
        	const bool  use_fixed_threshold  = true; // Use fixed or unfixed threshold (Alexis code)

        	MyDebugPrint("About to estimate loc res, with %.2f cutoff\n", fixed_fsc_threshold); // Checking that the half maps are in real space, then do a backwards Fourier transform if they are in Fourier space?
        	if ( ! output_3d1.density_map->is_in_real_space )
            	output_3d1.density_map->BackwardFFT( );
        	if ( ! output_3d2.density_map->is_in_real_space )
            	output_3d2.density_map->BackwardFFT( );

        	original_volume.CopyFrom(output_3d.density_map); // Original volume is the same as the output from the last run or last program
        	wxFileName temp_filename;
        	temp_filename = output_reconstruction_filtered; // I think this is just setting the filename of the filtered reconstruction (volume_#_#.mrc)


        	output_3d.density_map->QuickAndDirtyWriteSlices(wxString::Format("/tmp/locres_original_%s", temp_filename.GetFullName()).ToStdString(), 1, output_3d.density_map->logical_z_dimension, true, original_pixel_size);
        	// Debug out put that needs be changed before this gets pushed to git

        	Image size_image;
        	size_image.CopyFrom(output_3d.density_map); // set size image to be the same as output map, which makes it the same as the original volume but we don't want to change the original volume


#ifdef DEBUG
        	size_image.QuickAndDirtyWriteSlices("/tmp/locres_filtered_input.mrc", 1, size_image.logical_z_dimension, true, original_pixel_size);
#endif

        	float original_average_value = size_image.ReturnAverageOfRealValues(outer_mask_radius / original_pixel_size, true); // Returns average of all values
        	size_image.ConvertToAutoMask(original_pixel_size, outer_mask_radius, original_pixel_size * 2.0f, 0.2f); // Unsure about this (check ConvertToAutoMask)

#ifdef DEBUG
        	size_image.QuickAndDirtyWriteSlices("/tmp/locres_mask.mrc", 1, size_image.logical_z_dimension, true, original_pixel_size);
#endif

        	local_resolution_volume.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);
        	local_resolution_volume.SetToConstant(0.0f); //Allocate the local_resolution_volume and set it to a constant

        	Image local_resolution_volume_all; //Define local_res_volume_all and allocate it to the correct dimensions
        	local_resolution_volume_all.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);
        	local_resolution_volume_all.SetToConstant(0.0f);

        	int first_slice_with_data;
        	int last_slice_with_data;

        	Image slice_image;

        	for ( int counter = 1; counter <= local_resolution_volume.logical_z_dimension; counter++ ) {
            	slice_image.AllocateAsPointingToSliceIn3D(&size_image, counter);
            	if ( slice_image.IsConstant( ) == false ) {
                	first_slice_with_data = counter; //Find the first slice with data by finding the slice where the values are not constant
                	break;
            	}
        	}

        	for ( int counter = local_resolution_volume.logical_z_dimension; counter >= 1; counter-- ) {
            	slice_image.AllocateAsPointingToSliceIn3D(&size_image, counter);
            	if ( slice_image.IsConstant( ) == false ) {
                	last_slice_with_data = counter; // start from the top this time and go down to find the last slice with data
                	break;
            	}
        	}

        	int   slices_with_data  = last_slice_with_data - first_slice_with_data; // define slices with data
        	float slices_per_thread = slices_with_data / float(number_of_threads); // split slices with different threads
        	int   number_averaged   = 0;


       		 for ( float current_res = 6.0f; current_res < 7.0f; current_res += 6.0f ) { //set current_res to 18, 24, 30, and 36
            	//    float current_res = 24;
        	box_size = current_res / original_pixel_size; //set box size based on current_res and pixel size
            	wxPrintf("box size is %i\n", box_size);
            	// TODO: think about whether alignment resolution is good to use here or if there is an alternative
            	if ( alignment_res > 12 ) // alignment res is obtained from the user then used to set the fixed_FSC_threshold
                	fixed_fsc_threshold = 0.95;
            	else if ( alignment_res > 10 )
                	fixed_fsc_threshold = 0.93;
            	else if ( alignment_res > 8 )
                	fixed_fsc_threshold = 0.90;
            	else if ( alignment_res > 6 )
            		fixed_fsc_threshold = 0.75;
            	else if ( alignment_res > 4 )
            	    fixed_fsc_threshold = 0.5;
            	else
                	fixed_fsc_threshold = 0.2f;

            	local_resolution_volume.SetToConstant(0.0f); // doing this again but within the for loop

#pragma omp parallel default(shared) num_threads(number_of_threads) //run in parallel with a certain number of threads
            	{ // for omp

                	int first_slice = (first_slice_with_data - 1) + myroundint(ReturnThreadNumberOfCurrentThread( ) * slices_per_thread) + 1; // set the first slice
                	int last_slice  = (first_slice_with_data - 1) + myroundint((ReturnThreadNumberOfCurrentThread( ) + 1) * slices_per_thread); // set the last slice

                	Image local_resolution_volume_local; //define new image classes
                	Image input_volume_one_local;
                	Image input_volume_two_local;
                	Image input_original_volume;

                	input_volume_one_local.CopyFrom(output_3d1.density_map); // copy half maps to local volumes
                	input_volume_two_local.CopyFrom(output_3d2.density_map);
                	input_original_volume.CopyFrom(output_3d.density_map); //Setting up a new image to pull the original volume into the estimator

                	local_resolution_volume_local.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);
                	local_resolution_volume_local.SetToConstant(0.0f); // allocate and set constant the local volume
                	LocalResolutionEstimator* estimator = new LocalResolutionEstimator( );
                	estimator->SetAllUserParameters(&input_volume_one_local, &input_volume_two_local, &input_original_volume, &size_image, first_slice, last_slice, 1, original_pixel_size, box_size, threshold_snr, threshold_confidence, use_fixed_threshold, fixed_fsc_threshold, my_reconstruction_1.symmetry_matrices.symmetry_symbol, false, 2);
                	//set all inputs for the local res estimator code
                	estimator->EstimateLocalResolution(&local_resolution_volume_local); // run estimate local resolution on the local volume
                	delete estimator;

#pragma omp critical
                	{
                    	for ( long pixel_counter = 0; pixel_counter < local_resolution_volume.number_of_real_space_pixels; pixel_counter++ ) {
                        	if ( local_resolution_volume_local.real_values[pixel_counter] != 0.0f )
                            	local_resolution_volume.real_values[pixel_counter] = local_resolution_volume_local.real_values[pixel_counter]; //if local2x volume pixel value is 0, set the local volume to 0 as well
                    	}
                	}
            	} // end omp

      	local_resolution_volume.QuickAndDirtyWriteSlices(wxString::Format("/tmp/local_res_%i", int(current_res)).ToStdString(), 1, local_resolution_volume.logical_z_dimension);

            	// fill in gaps..

            	float max_res = local_resolution_volume.ReturnMaximumValue( ); //find max value
            	float min_res = local_resolution_volume.ReturnMinimumValue( ); //find min value (this is currently not used but may be in the future so I'll keep it for now)

           		 local_resolution_volume_all.AddImage(&local_resolution_volume); // adds the current volume to the all volumes class
           		 float box_size_counter = 1;
           		 number_averaged+=box_size_counter;  //Takes number_averaged and adds the weight value to it
       		 }

        	// divide and copy
        	local_resolution_volume_all.QuickAndDirtyWriteSlices(wxString::Format("/tmp/locres_all").ToStdString(), 1, size_image.logical_z_dimension, true, original_pixel_size);
        	local_resolution_volume_all.DivideByConstant(number_averaged); // divides by the number averaged
        	local_resolution_volume.CopyFrom(&local_resolution_volume_all); // makes the local volume the combined volume
       		local_resolution_volume.QuickAndDirtyWriteSlices(wxString::Format("/tmp/locres_box_weighting_%s", temp_filename.GetFullName()).ToStdString(), 1, size_image.logical_z_dimension, true, original_pixel_size);

        	// get scaler for resolution

        	int number_of_top_pixels_to_use = local_resolution_volume.number_of_real_space_pixels * 0.00001;
        	if ( number_of_top_pixels_to_use < 50 )
            	number_of_top_pixels_to_use = 50; // find top pixels to use to find the highest resolution

        	float highest_resolution  = local_resolution_volume.ReturnAverageOfMinN(number_of_top_pixels_to_use); //get highest resolution
        	float measured_resolution = resolution_statistics->ReturnEstimatedResolution(true); //get the measured resolution using the cut off

        	float average_resolution = 0.0f;
        	long  voxels_in_the_mask = 0;

        	int  i, j, k;
        	long pixel_counter = 0;

        	for ( k = 0; k < local_resolution_volume.logical_z_dimension; k++ ) {
            	for ( j = 0; j < local_resolution_volume.logical_y_dimension; j++ ) {
                	for ( i = 0; i < local_resolution_volume.logical_x_dimension; i++ ) {
                    	if ( size_image.real_values[pixel_counter] == 1.0f ) {
                        	if (local_resolution_volume.real_values[pixel_counter] < highest_resolution) highest_resolution = local_resolution_volume.real_values[pixel_counter];
                        	average_resolution += local_resolution_volume.real_values[pixel_counter];
                        	voxels_in_the_mask++;
                    	}

                    	pixel_counter++;
                	}
                	pixel_counter += local_resolution_volume.padding_jump_value;
            	}
        	}

        	average_resolution /= voxels_in_the_mask;
        	wxPrintf("Local high / Measured Average / Local Average = %.2f / %.2f / %.2f\n", highest_resolution, measured_resolution, average_resolution);


#ifdef DEBUG
        	local_resolution_volume.QuickAndDirtyWriteSlices("/tmp/locres_scaled.mrc", 1, size_image.logical_z_dimension, true, pixel_size);
#endif


        	local_resolution_volume.QuickAndDirtyWriteSlices(wxString::Format("/tmp/locres_real_value_weighting").ToStdString(), 1, size_image.logical_z_dimension, true, pixel_size);

        	MyDebugPrint("About to apply locres filter\n");

        //	int number_of_levels = box_size;
        //	output_3d.density_map->ApplyLocalResolutionFilter(local_resolution_volume, original_pixel_size, number_of_levels); // apply the filter

        	// Need to filter the original map to 20 Angstroms

        	Image filtered_20_volume;
        	float cosine_falloff_width = 10.0 / float(logical_x_dimension); // 5 Fourier voxels
			filtered_20_volume.CopyFrom(output_3d.density_map);
			filtered_20_volume.ForwardFFT();
            filtered_20_volume.CosineMask(original_pixel_size / 20, cosine_falloff_width);
            filtered_20_volume.BackwardFFT();

        	filtered_20_volume.QuickAndDirtyWriteSlices(wxString::Format("/tmp/20_A_filtered_original").ToStdString(), 1, size_image.logical_z_dimension, true, pixel_size);

        	for ( long address = 0; address < output_3d.density_map->real_memory_allocated; address++ ) {
            	    output_3d.density_map->real_values[address] = local_resolution_volume.real_values[address];
            	    if (output_3d.density_map->real_values[address] == 0) {
            	    	output_3d.density_map->real_values[address] = filtered_20_volume.real_values[address];
            	    }

        	}
        	output_3d.density_map->QuickAndDirtyWriteSlices(wxString::Format("/tmp/locres_to_output").ToStdString(), 1, size_image.logical_z_dimension, true, pixel_size);


        //	output_3d.density_map->CosineMask(outer_mask_radius / original_pixel_size, 1.0, false, true, 0.0);
    	}

    	output_3d.density_map->WriteSlicesAndFillHeader(output_reconstruction_filtered.ToStdString( ), original_pixel_size);
	}
	/////////////////////// END HACK..

	if ( save_orthogonal_views_image == true ) {
    	Image orth_image;
    	orth_image.Allocate(output_3d.density_map->logical_x_dimension * 3, output_3d.density_map->logical_y_dimension * 2, 1, true);
    	output_3d.density_map->CreateOrthogonalProjectionsImage(&orth_image);
    	orth_image.QuickAndDirtyWriteSlice(orthogonal_views_filename.ToStdString( ), 1);
	}

	if ( is_running_locally == false ) {
    	int number_of_points = resolution_statistics->FSC.number_of_points;
    	int array_size   	= (number_of_points * 5) + 2;
    	wxPrintf("number of points = %i, class is %i, array size = %i\n", number_of_points, class_number_for_gui, array_size);

    	float* statistics = new float[array_size];
    	resolution_statistics->WriteStatisticsToFloatArray(statistics, class_number_for_gui);
    	my_result.SetResult(array_size, statistics);
    	delete[] statistics;
	}

	wxPrintf("\nMerge3D: Normal termination\n\n");

	delete resolution_statistics;
	return true;
}
