#include "../core/gui_core_headers.h"
#include <wx/gauge.h>

/* NOTE: this file contains large sections that are commented out; this is mostly logic that could/will 
be useful in the future, but isn't simple to fully implement now. These regions of code are being left 
in intentionally. */

extern MyRefinementPackageAssetPanel* refinement_package_asset_panel;
extern MyVolumeAssetPanel*            volume_asset_panel;

// On construction, we want to load in all of the classes, and default initialize the selected refinement to the first in the currently selected class (both will just be the first)
ResampleDialog::ResampleDialog(wxWindow* parent, bool is_volume_resample) : ResampleDialogParent(parent) {
    resample_pixel_size         = 0.0f;
    previously_entered_box_size = 32;
    resample_box_size           = 0;
    resampling_volume           = is_volume_resample;
    BoxSizeSpinCtrl->SetIncrement(2);
    if ( resampling_volume ) {
        ResampleInfoText->SetLabel("This action will create a new volume based on the selected volume, resampled to the chosen box size and pixel size displayed.");
    }
    else {
        ResampleInfoText->SetLabel("This action will create a new refinement package and particle stack based on the selected package, resampled with the chosen box size and the pixel size displayed.");
    }
    ResampleInfoText->Wrap(400);
    Layout( );
    Fit( );
    wxCommandEvent tmp_event;
    OnBoxSizeSpinCtrl(tmp_event);
}

void ResampleDialog::OnBoxSizeSpinCtrl(wxCommandEvent& event) {
    // This is a general check that is useful for both stacks and volumes
    resample_box_size = BoxSizeSpinCtrl->GetValue( );
    if ( (resample_box_size < 32 || resample_box_size > 1600) || resample_box_size % 2 != 0 )
        OKButton->Enable(false);
    else
        OKButton->Enable(true);

    if ( resample_box_size != previously_entered_box_size ) {
        previously_entered_box_size = resample_box_size;

        int   original_box_size;
        float original_pixel_size;
        if ( resampling_volume ) {
            original_pixel_size = volume_asset_panel->all_assets_list->ReturnVolumeAssetPointer(volume_asset_panel->all_groups_list->ReturnGroupMember(volume_asset_panel->selected_group, volume_asset_panel->selected_content))->pixel_size;
            original_box_size   = volume_asset_panel->all_assets_list->ReturnVolumeAssetPointer(volume_asset_panel->all_groups_list->ReturnGroupMember(volume_asset_panel->selected_group, volume_asset_panel->selected_content))->x_size;
        }
        else {
            original_box_size   = refinement_package_asset_panel->all_refinement_packages[refinement_package_asset_panel->selected_refinement_package].stack_box_size;
            original_pixel_size = refinement_package_asset_panel->all_refinement_packages[refinement_package_asset_panel->selected_refinement_package].output_pixel_size;
        }

        if ( original_box_size == resample_box_size ) {
            OKButton->Enable(false);
        }
        // Calculation is: new_pixel_size / old_pixel_size = old_box_size / resample_box_size
        float ratio         = float(original_box_size) / float(resample_box_size);
        resample_pixel_size = original_pixel_size * ratio;
        NewPixelSizeText->SetLabel(wxString::Format("New Pixel Size: %0.4f", resample_pixel_size));
        Layout( );
        Fit( );
    }
}

void ResampleDialog::OnBoxSizeTextEnter(wxCommandEvent& event) {
    OnBoxSizeSpinCtrl(event);
    event.Skip( );
}

// TODO: Load in the class and refinement selection panels, then fill the comboxes properly -- this would go in the constructor
// RefinementPackage current_package = refinement_package_asset_panel->all_refinement_packages[refinement_package_asset_panel->selected_refinement_package];

// Refinement selection
// refinement_selection_panel = new CombinedPackageRefinementSelectPanel(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
// refinement_selection_panel->RefinementText->SetLabel("Refinement from " + current_package.name + ": ");
// refinement_selection_panel->RefinementText->Wrap(300);
// refinement_selection_panel->RefinementComboBox->FillComboBox(refinement_package_asset_panel->selected_refinement_package, false);
// BinningRefinementSelectionSizer->Add(refinement_selection_panel, 1, wxALL | wxALIGN_CENTER_HORIZONTAL, 5);

// Class selection
// class_selection_panel = new CombinedPackageClassSelectionPanel(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
// class_selection_panel->ClassText->SetLabel("Class from " + current_package.name + ": ");
// class_selection_panel->ClassText->Wrap(300);
// class_selection_panel->FillComboBox(refinement_package_asset_panel->selected_refinement_package);
// BinningClassSelectionSizer->Add(class_selection_panel, 1, wxALL | wxALIGN_CENTER_HORIZONTAL, 5);

// // Initial reference selection (ClassVolumeSelectPanel)
// initial_reference_panel = new ClassVolumeSelectPanel(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
// initial_reference_panel->ClassText->SetLabel("Initial reference: ");
// initial_reference_panel->ClassText->Wrap(300);
// initial_reference_panel->VolumeComboBox->FillComboBox(true, true);
// InitialReferenceSelectionSizer->Add(initial_reference_panel, 1, wxALL | wxALIGN_CENTER_HORIZONTAL, 5);

ResampleDialog::~ResampleDialog( ) {
}

void ResampleDialog::OnOK(wxCommandEvent& event) {
    // long class_id               = class_selection_panel->ClassComboBox->GetSelection( ); // TEST: Does this get the ID itself or the index of the selection?
    // long selected_refinement_id = refinement_package_asset_panel->all_refinement_packages[refinement_package_asset_panel->selected_refinement_package].refinement_ids[refinement_selection_panel->RefinementComboBox->GetSelection( )];
    // FIXME: this method for pulling the reference index needs to be tested because of generate from params...
    // int      initial_reference_index = initial_reference_panel->VolumeComboBox->GetSelection( ) - 1;

    EndModal(0);
    RefinementPackage* binned_pkg;
    Refinement*        binned_refinement;
    VolumeAsset*       tmp_asset;
    if ( ! resampling_volume ) {
        RefinementPackage package_to_bin = refinement_package_asset_panel->all_refinement_packages.Item(refinement_package_asset_panel->selected_refinement_package);

        // Pixel size gets multipled by the binning factor to adjust size
        // float   new_pixel_size = package_to_bin.output_pixel_size * binning_val;
        MRCFile original_particle_stack(package_to_bin.stack_filename.ToStdString( ), false);

        long total_progress_increments = package_to_bin.contained_particles.GetCount( ) * 2;

        OneSecondProgressDialog* my_dialog;
        if ( ! resampling_volume )
            my_dialog = new OneSecondProgressDialog("Refinement Package", "Resampling original particle stack...", package_to_bin.contained_particles.GetCount( ) * 2, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE | wxPD_APP_MODAL);
        // int                      resample_box_size = myroundint(original_particle_stack.ReturnXSize( ) / binning_val);

        Image                 current_image;
        EmpiricalDistribution stack_distribution;

        std::string combined_stack_filename = wxString::Format(main_frame->current_project.particle_stack_directory.GetFullPath( ) + "/particle_stack_%li.mrc", refinement_package_asset_panel->current_asset_number).ToStdString( );
        MRCFile     binned_stack_file(combined_stack_filename);

        // Now handle the actual resample
        long overall_progress = 0;
        for ( long img_counter = 0; img_counter < original_particle_stack.ReturnNumberOfSlices( ); img_counter++ ) {
            current_image.ReadSlice(&original_particle_stack, img_counter + 1);

            current_image.ForwardFFT( );
            current_image.Resize(resample_box_size, resample_box_size, 1, 0.0f);
            current_image.BackwardFFT( );

            current_image.WriteSlice(&binned_stack_file, img_counter + 1);
            current_image.UpdateDistributionOfRealValues(&stack_distribution);
            overall_progress++;
            my_dialog->Update(overall_progress);
        }
        float std = stack_distribution.GetSampleVariance( );
        if ( std > 0.0 ) {
            std = sqrt(std);
        }

        // This should populate the new mrc file
        binned_stack_file.SetDensityStatistics(stack_distribution.GetMinimum( ), stack_distribution.GetMaximum( ), stack_distribution.GetSampleMean( ), std);
        binned_stack_file.SetPixelSize(resample_pixel_size);
        binned_stack_file.WriteHeader( );
        binned_stack_file.CloseFile( );
        original_particle_stack.CloseFile( );

        // Now, create a new RefinementPackage asset that will be added to the list of refinement packages
        // Also, update the refinement parameters where necessary
        binned_pkg        = new RefinementPackage( );
        binned_refinement = new Refinement;
        // Gets the first refinement containing the randomized angles
        Refinement* old_refinement = main_frame->current_project.database.GetRefinementByID(package_to_bin.refinement_ids[0]);
        // This value is the basis for all subsequent refinement_ids -- will have to use a loop to go over all refinements, and add counter + 1 to this to get it to work correctly
        binned_pkg->asset_id                             = refinement_package_asset_panel->current_asset_number + 1;
        binned_pkg->name                                 = wxString::Format("Refinement Package #%li", refinement_package_asset_panel->current_asset_number);
        binned_pkg->estimated_particle_size_in_angstroms = package_to_bin.estimated_particle_size_in_angstroms;
        binned_pkg->estimated_particle_weight_in_kda     = package_to_bin.estimated_particle_weight_in_kda;
        binned_pkg->output_pixel_size                    = resample_pixel_size;
        binned_pkg->stack_box_size                       = resample_box_size;
        binned_pkg->stack_filename                       = combined_stack_filename;
        binned_pkg->stack_has_white_protein              = package_to_bin.stack_has_white_protein;
        binned_pkg->symmetry                             = package_to_bin.symmetry;

        // This is for getting all the refinements and classes resampled as well
        binned_pkg->number_of_run_refinments = package_to_bin.number_of_run_refinments;
        binned_pkg->number_of_classes        = 1;
        binned_pkg->references_for_next_refinement.Add(-1);

        // Now the refinement
        binned_refinement->name                             = "Generate from params.";
        binned_refinement->refinement_id                    = main_frame->current_project.database.ReturnHighestRefinementID( ) + 1;
        binned_refinement->resolution_statistics_box_size   = resample_box_size;
        binned_refinement->resolution_statistics_pixel_size = resample_pixel_size;
        // See if this needs to be specified or if it just defaults
        // binned_refinement->starting_refinement_id           = binned_refinement->refinement_id;
        binned_refinement->number_of_classes           = 1;
        binned_refinement->percent_used                = 100.0f;
        binned_refinement->datetime_of_run             = wxDateTime::Now( );
        binned_refinement->refinement_package_asset_id = binned_pkg->asset_id;
        binned_refinement->number_of_particles         = old_refinement->number_of_particles;

        binned_pkg->refinement_ids.Add(binned_refinement->refinement_id);

        // Fill in particle info
        RefinementPackageParticleInfo current_particle_info;
        // Only using a single refinement, so only one class is being included; and we need to fill up the class_refinement_results with blank particles that will be filled during the loop
        binned_refinement->SizeAndFillWithEmpty(old_refinement->number_of_particles, 1);
        for ( int particle_counter = 0; particle_counter < package_to_bin.contained_particles.GetCount( ); particle_counter++ ) {
            current_particle_info                   = package_to_bin.contained_particles[particle_counter];
            current_particle_info.position_in_stack = particle_counter + 1;
            current_particle_info.pixel_size        = resample_pixel_size;
            binned_pkg->contained_particles.Add(current_particle_info);

            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].position_in_stack                  = particle_counter + 1;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].defocus1                           = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].defocus1;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].defocus2                           = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].defocus2;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].defocus_angle                      = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].defocus_angle;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].phase_shift                        = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].phase_shift;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].logp                               = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].logp;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].pixel_size                         = resample_pixel_size;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].microscope_voltage_kv              = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].microscope_voltage_kv;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].microscope_spherical_aberration_mm = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].microscope_spherical_aberration_mm;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].amplitude_contrast                 = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].amplitude_contrast;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].beam_tilt_x                        = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].beam_tilt_x;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].beam_tilt_y                        = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].beam_tilt_y;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].image_shift_x                      = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].image_shift_x;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].image_shift_y                      = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].image_shift_y;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].occupancy                          = 100.0f;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].phi                                = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].phi;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].theta                              = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].theta;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].psi                                = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].psi;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].score                              = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].score;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].image_is_active                    = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].image_is_active;
            binned_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].sigma                              = old_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].sigma;
            overall_progress++;
            my_dialog->Update(overall_progress, "Filling Refinement Package with particles...");
        }
    }

    else {
        tmp_asset                      = new VolumeAsset( );
        wxString resampled_3d_filename = main_frame->current_project.volume_asset_directory.GetFullPath( ) + wxString::Format("/resampled_%i_%s", resample_box_size, volume_asset_panel->all_assets_list->ReturnVolumeAssetPointer(volume_asset_panel->all_groups_list->ReturnGroupMember(volume_asset_panel->selected_group, volume_asset_panel->selected_content))->ReturnShortNameString( ));

        VolumeAsset* original_volume_asset = volume_asset_panel->all_assets_list->ReturnVolumeAssetPointer(volume_asset_panel->all_groups_list->ReturnGroupMember(volume_asset_panel->current_group_number, volume_asset_panel->selected_content));
        tmp_asset->CopyFrom(original_volume_asset);

        // Now that we have the particular asset, we'll change the parameters that need changing
        tmp_asset->asset_id   = volume_asset_panel->current_asset_number;
        tmp_asset->pixel_size = resample_pixel_size;
        tmp_asset->x_size     = resample_box_size;
        tmp_asset->y_size     = resample_box_size;
        tmp_asset->z_size     = resample_box_size;

        // for ( int i = 0; i < volume_asset_panel->all_assets_list->ReturnNumberOfAssets( ); i++ ) {
        //     if ( volume_asset_panel->all_assets_list->ReturnVolumeAssetPointer(i)->asset_name == new_asset_name ) {
        //         new_asset_name = original_volume_asset->asset_name;
        //         break;
        //     }
        // }
        // tmp_asset->asset_name          = new_asset_name;
        tmp_asset->half_map_1_filename = "";
        tmp_asset->half_map_2_filename = "";
        tmp_asset->Update(resampled_3d_filename);
        tmp_asset->asset_name            = tmp_asset->filename.GetName( );
        tmp_asset->reconstruction_job_id = -1;
        // binned_pkg->references_for_next_refinement.Add(tmp_asset->asset_id);

        MRCFile               original_volume_file(original_volume_asset->filename.GetFullPath( ).ToStdString( ));
        MRCFile               resampled_volume_file(resampled_3d_filename.ToStdString( ));
        Image                 my_volume;
        EmpiricalDistribution volume_distribution;
        const long            number_of_slices = original_volume_file.ReturnNumberOfSlices( );
        my_volume.ReadSlices(&original_volume_file, 1, number_of_slices);
        my_volume.ForwardFFT( );
        my_volume.Resize(resample_box_size, resample_box_size, resample_box_size, 0.f);
        my_volume.BackwardFFT( );
        my_volume.WriteSlices(&resampled_volume_file, 1, resample_box_size);
        // current_slice.UpdateDistributionOfRealValues(&volume_distribution);

        float std = volume_distribution.GetSampleVariance( );
        if ( std > 0.0 ) {
            std = sqrt(std);
        }

        // This should populate the new mrc file
        resampled_volume_file.SetDensityStatistics(volume_distribution.GetMinimum( ), volume_distribution.GetMaximum( ), volume_distribution.GetSampleMean( ), std);
        resampled_volume_file.SetPixelSize(resample_pixel_size);
        resampled_volume_file.WriteHeader( );
        resampled_volume_file.CloseFile( );
        original_volume_file.CloseFile( );
    }

    // Resample op complete; now register in database
    main_frame->current_project.database.Begin( );
    if ( ! resampling_volume ) {
        refinement_package_asset_panel->AddAsset(binned_pkg);
        main_frame->current_project.database.AddRefinementPackageAsset(binned_pkg);
        binned_refinement->class_refinement_results[0].class_resolution_statistics.Init(binned_pkg->output_pixel_size, binned_refinement->resolution_statistics_box_size);
        binned_refinement->class_refinement_results[0].class_resolution_statistics.GenerateDefaultStatistics(binned_pkg->estimated_particle_weight_in_kda);
        main_frame->current_project.database.AddRefinement(binned_refinement);

        ArrayofAngularDistributionHistograms all_histograms = binned_refinement->ReturnAngularDistributions(binned_pkg->symmetry);
        main_frame->current_project.database.AddRefinementAngularDistribution(all_histograms[0], binned_refinement->refinement_id, 1);

        ShortRefinementInfo binned_info;
        binned_info = binned_refinement;
        refinement_package_asset_panel->all_refinement_short_infos.Add(binned_info);
    }
    else {
        volume_asset_panel->AddAsset(tmp_asset);
        main_frame->current_project.database.BeginVolumeAssetInsert( );
        main_frame->current_project.database.AddNextVolumeAsset(tmp_asset->asset_id, tmp_asset->asset_name, tmp_asset->filename.GetFullPath( ), tmp_asset->reconstruction_job_id, tmp_asset->pixel_size, tmp_asset->x_size, tmp_asset->y_size, tmp_asset->z_size, tmp_asset->half_map_1_filename.GetFullPath( ), tmp_asset->half_map_2_filename.GetFullPath( ));
        main_frame->current_project.database.EndVolumeAssetInsert( );
        delete tmp_asset;
    }

    main_frame->current_project.database.Commit( );
    Destroy( );
}

void ResampleDialog::OnCancel(wxCommandEvent& event) {
    EndModal(0);
    Destroy( );
}
