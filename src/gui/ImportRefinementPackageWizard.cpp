//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel* refinement_package_asset_panel;

ImportRefinementPackageWizard::ImportRefinementPackageWizard(wxWindow* parent)
    : ImportRefinementPackageWizardParent(parent) {
    SetPageSize(wxSize(600, 400));
    SymmetryComboBox->Clear( );
    SymmetryComboBox->Append("C1");
    SymmetryComboBox->Append("C2");
    SymmetryComboBox->Append("C3");
    SymmetryComboBox->Append("C4");
    SymmetryComboBox->Append("D2");
    SymmetryComboBox->Append("D3");
    SymmetryComboBox->Append("D4");
    SymmetryComboBox->Append("I");
    SymmetryComboBox->Append("I2");
    SymmetryComboBox->Append("O");
    SymmetryComboBox->Append("T");
    SymmetryComboBox->Append("T2");
    SymmetryComboBox->SetSelection(0);
    PixelSizeTextCtrl->SetPrecision(4);

    if ( cisTEMRadioButton->GetValue( ) == true ) {
        MicroscopeVoltageTextCtrl->Show(false);
        MicroscopeVoltageTextCtrlLabel->Show(false);
        PixelSizeTextCtrl->Show(false);
        PixelSizeTextCtrlLabel->Show(false);
        AmplitudeContrastTextCtrl->Show(false);
        AmplitudeContrastTextCtrlLabel->Show(false);
    }
}

void ImportRefinementPackageWizard::CheckPaths( ) {
    if ( GetCurrentPage( ) == m_pages.Item(1) ) {
        Freeze( );

        EnableNextButton( );

        if ( DoesFileExist(ParticleStackFileTextCtrl->GetLineText(0)) == false )
            DisableNextButton( );
        if ( DoesFileExist(MetaDataFileTextCtrl->GetLineText(0)) == false )
            DisableNextButton( );

        Thaw( );
    }
}

void ImportRefinementPackageWizard::OnStackBrowseButtonClick(wxCommandEvent& event) {
    wxFileDialog openFileDialog(this, _("Select Stack File"), "", "", "MRC files (*.mrc)|*.mrc;*.mrcs", wxFD_OPEN | wxFD_FILE_MUST_EXIST);

    if ( openFileDialog.ShowModal( ) == wxID_OK ) {
        ParticleStackFileTextCtrl->SetValue(openFileDialog.GetPath( ));
    }
}

void ImportRefinementPackageWizard::OnMetaBrowseButtonClick(wxCommandEvent& event) {
    wxFileDialog* openFileDialog;

    if ( FrealignRadioButton->GetValue( ) == true ) {
        openFileDialog = new wxFileDialog(this, _("Select PAR File"), "", "", "PAR files (*.par)|*.par;", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    }
    else if ( RelionRadioButton->GetValue( ) == true ) {
        openFileDialog = new wxFileDialog(this, _("Select STAR File"), "", "", "STAR files (*.star)|*.star;", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    }
    else if ( cisTEMRadioButton->GetValue( ) == true ) {
        openFileDialog = new wxFileDialog(this, _("Select STAR File"), "", "", "STAR files (*.star)|*.star;", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    }

    if ( openFileDialog->ShowModal( ) == wxID_OK ) {
        MetaDataFileTextCtrl->SetValue(openFileDialog->GetPath( ));
    }

    openFileDialog->Destroy( );
}

void ImportRefinementPackageWizard::OnPageChanged(wxWizardEvent& event) {
    if ( event.GetPage( ) == m_pages.Item(0) ) {
        EnableNextButton( );
    }
    else if ( event.GetPage( ) == m_pages.Item(1) ) {
        if ( FrealignRadioButton->GetValue( ) == true ) {
            MetaFilenameStaticText->SetLabel("PAR Filename :-    ");
        }
        else if ( RelionRadioButton->GetValue( ) == true ) {
            MetaFilenameStaticText->SetLabel("STAR Filename :-   ");
        }
        else if ( cisTEMRadioButton->GetValue( ) == true ) {
            MetaFilenameStaticText->SetLabel("STAR Filename :-   ");
        }
        CheckPaths( );
    }
    else if ( event.GetPage( ) == m_pages.Item(2) ) {
        if ( FrealignRadioButton->GetValue( ) == true ) {
            BlackProteinRadioButton->SetValue(true);
        }
        else if ( RelionRadioButton->GetValue( ) == true ) {
            WhiteProteinRadioButton->SetValue(true);
        }
        if ( cisTEMRadioButton->GetValue( ) == true ) {
            BlackProteinRadioButton->SetValue(true);
        }
        else
            CheckPaths( );
        EnableNextButton( );
    }

    if ( cisTEMRadioButton->GetValue( ) == true ) {
        MicroscopeVoltageTextCtrl->Show(false);
        MicroscopeVoltageTextCtrlLabel->Show(false);
        PixelSizeTextCtrl->Show(false);
        PixelSizeTextCtrlLabel->Show(false);
        AmplitudeContrastTextCtrl->Show(false);
        AmplitudeContrastTextCtrlLabel->Show(false);
    }
    else {
        MicroscopeVoltageTextCtrl->Show(true);
        MicroscopeVoltageTextCtrlLabel->Show(true);
        PixelSizeTextCtrl->Show(true);
        PixelSizeTextCtrlLabel->Show(true);
        AmplitudeContrastTextCtrl->Show(true);
        AmplitudeContrastTextCtrlLabel->Show(true);
    }
}

void ImportRefinementPackageWizard::OnPathChange(wxCommandEvent& event) {
    CheckPaths( );
}

void ImportRefinementPackageWizard::OnUpdateUI(wxUpdateUIEvent& event) {
    if ( GetCurrentPage( ) == m_pages.Item(2) ) {
        wxString symmetry = SymmetryComboBox->GetValue( );
        if ( IsAValidSymmetry(&symmetry) == true )
            EnableNextButton( );
        else
            DisableNextButton( );
    }
}

void ImportRefinementPackageWizard::OnFinished(wxWizardEvent& event) {
    // get the stack details..

    int stack_x_size;
    int stack_y_size;
    int stack_number_of_images;

    RefinementPackage*            temp_refinement_package;
    RefinementPackageParticleInfo temp_particle_info;
    Refinement                    temp_refinement;

    bool stack_is_ok = GetMRCDetails(ParticleStackFileTextCtrl->GetLineText(0).ToUTF8( ).data( ), stack_x_size, stack_y_size, stack_number_of_images);

    if ( stack_is_ok == false ) {
        wxMessageBox(wxT("Error: Cannot read the stack file - aborting."), wxT("Error Reading particle stack"), wxICON_ERROR);
        return;
    }

    if ( stack_x_size != stack_y_size ) {
        wxMessageBox(wxT("Error: Only square images are currently supported - aborting."), wxT("Error images are not square"), wxICON_ERROR);
        return;
    }

    // hmm, so now I guess we have to actually do the import..
    if ( cisTEMRadioButton->GetValue( ) == true ) // cisTEM
    {

        cisTEMParameters input_star_file;
        // Method creates a cisTEMStarFileReader object and reads in
        input_star_file.ReadFromcisTEMStarFile(MetaDataFileTextCtrl->GetLineText(0), true);

        if ( stack_number_of_images != input_star_file.ReturnNumberofLines( ) ) {
            wxMessageBox(wxT("Error: Number of images in stack is different from\nthe number of lines in the star file - aborting."), wxT("Error: Number Mismatch"), wxICON_ERROR);
            return;
        }

        OneSecondProgressDialog* my_dialog = new OneSecondProgressDialog("Refinement Package", "Creating Refinement Package...", stack_number_of_images, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE | wxPD_APP_MODAL);

        // create the refinement package and intial refinement..

        temp_refinement_package = new RefinementPackage;
        temp_refinement.SizeAndFillWithEmpty(stack_number_of_images, 1);

        temp_refinement_package->name                     = wxString::Format("Refinement Package #%li (cisTEM Import)", refinement_package_asset_panel->current_asset_number);
        temp_refinement_package->number_of_classes        = 1;
        temp_refinement_package->number_of_run_refinments = 0;
        temp_refinement_package->stack_has_white_protein  = WhiteProteinRadioButton->GetValue( );
        temp_refinement_package->output_pixel_size        = input_star_file.ReturnPixelSize(0); // FIXME should be for all lines.
        //		temp_refinement_package->output_pixel_size = PixelSizeTextCtrl->ReturnValue();

        temp_refinement.number_of_classes                = temp_refinement_package->number_of_classes;
        temp_refinement.number_of_particles              = stack_number_of_images;
        temp_refinement.name                             = "Imported Parameters";
        temp_refinement.resolution_statistics_box_size   = stack_x_size;
        temp_refinement.resolution_statistics_pixel_size = input_star_file.ReturnPixelSize(0); // FIXME should be for all lines.
        //		temp_refinement.resolution_statistics_pixel_size = PixelSizeTextCtrl->ReturnValue();
        temp_refinement.refinement_package_asset_id = refinement_package_asset_panel->current_asset_number + 1;

        temp_refinement_package->stack_box_size                       = stack_x_size;
        temp_refinement_package->stack_filename                       = ParticleStackFileTextCtrl->GetLineText(0);
        temp_refinement_package->symmetry                             = SymmetryComboBox->GetValue( ).Upper( );
        temp_refinement_package->estimated_particle_weight_in_kda     = MolecularWeightTextCtrl->ReturnValue( );
        temp_refinement_package->estimated_particle_size_in_angstroms = LargestDimensionTextCtrl->ReturnValue( );

        long refinement_id = main_frame->current_project.database.ReturnHighestRefinementID( ) + 1;
        temp_refinement_package->refinement_ids.Add(refinement_id);
        temp_refinement_package->references_for_next_refinement.Add(-1);

        temp_refinement.refinement_id                       = refinement_id;
        temp_refinement.resolution_statistics_are_generated = true;

        temp_particle_info.spherical_aberration = SphericalAberrationTextCtrl->ReturnValue( );
        temp_particle_info.microscope_voltage   = input_star_file.ReturnMicroscopekV(0); // FIXME should be for all lines.
        //		temp_particle_info.microscope_voltage = MicroscopeVoltageTextCtrl->ReturnValue();
        temp_particle_info.parent_image_id    = -1;
        temp_particle_info.amplitude_contrast = input_star_file.ReturnAmplitudeContrast(0); // FIXME should be for all lines.
        //		temp_particle_info.pixel_size = PixelSizeTextCtrl->ReturnValue();
        //		temp_particle_info.amplitude_contrast = AmplitudeContrastTextCtrl->ReturnValue();
        temp_particle_info.x_pos = 0;
        temp_particle_info.y_pos = 0;

        temp_refinement.class_refinement_results[0].class_resolution_statistics.Init(temp_particle_info.pixel_size, temp_refinement.resolution_statistics_box_size);
        temp_refinement.class_refinement_results[0].class_resolution_statistics.GenerateDefaultStatistics(temp_refinement_package->estimated_particle_weight_in_kda);

        // loop over all particles

        for ( int particle_counter = 0; particle_counter < stack_number_of_images; particle_counter++ ) {

            temp_particle_info.original_particle_position_asset_id = (int)input_star_file.ReturnPositionInStack(particle_counter);
            temp_particle_info.position_in_stack                   = (int)input_star_file.ReturnPositionInStack(particle_counter);
            temp_particle_info.defocus_1                           = input_star_file.ReturnDefocus1(particle_counter);
            temp_particle_info.defocus_2                           = input_star_file.ReturnDefocus2(particle_counter);
            temp_particle_info.defocus_angle                       = input_star_file.ReturnDefocusAngle(particle_counter);
            temp_particle_info.phase_shift                         = input_star_file.ReturnPhaseShift(particle_counter);
            temp_particle_info.pixel_size                          = input_star_file.ReturnPixelSize(particle_counter);
            temp_particle_info.amplitude_contrast                  = input_star_file.ReturnAmplitudeContrast(particle_counter);
            temp_particle_info.assigned_subset                     = input_star_file.ReturnAssignedSubset(particle_counter);

            temp_refinement_package->contained_particles.Add(temp_particle_info);

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].position_in_stack = (int)input_star_file.ReturnPositionInStack(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus1          = input_star_file.ReturnDefocus1(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus2          = input_star_file.ReturnDefocus2(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus_angle     = input_star_file.ReturnDefocusAngle(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].phase_shift       = input_star_file.ReturnPhaseShift(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].logp              = input_star_file.ReturnLogP(particle_counter);

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].occupancy       = input_star_file.ReturnOccupancy(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].phi             = input_star_file.ReturnPhi(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].theta           = input_star_file.ReturnTheta(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].psi             = input_star_file.ReturnPsi(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].xshift          = input_star_file.ReturnXShift(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].yshift          = input_star_file.ReturnYShift(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].score           = input_star_file.ReturnScore(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].image_is_active = (int)input_star_file.ReturnImageIsActive(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].sigma           = input_star_file.ReturnSigma(particle_counter);

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].pixel_size                         = input_star_file.ReturnPixelSize(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].microscope_voltage_kv              = input_star_file.ReturnMicroscopekV(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].microscope_spherical_aberration_mm = input_star_file.ReturnMicroscopeCs(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].amplitude_contrast                 = input_star_file.ReturnAmplitudeContrast(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].beam_tilt_x                        = input_star_file.ReturnBeamTiltX(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].beam_tilt_y                        = input_star_file.ReturnBeamTiltY(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].image_shift_x                      = input_star_file.ReturnImageShiftX(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].image_shift_y                      = input_star_file.ReturnImageShiftY(particle_counter);

            //
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].beam_tilt_group = input_star_file.ReturnBeamTiltGroup(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].particle_group  = input_star_file.ReturnParticleGroup(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].assigned_subset = input_star_file.ReturnAssignedSubset(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].pre_exposure    = input_star_file.ReturnPreExposure(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].total_exposure  = input_star_file.ReturnTotalExposure(particle_counter);

            my_dialog->Update(particle_counter + 1);
        }

        // add to the database and panel..

        main_frame->current_project.database.Begin( );
        wxPrintf("\t\t\n\nAdding the temp refinement from the cisTEM import dialog\n\n");
        refinement_package_asset_panel->AddAsset(temp_refinement_package);
        main_frame->current_project.database.AddRefinement(&temp_refinement);

        ArrayofAngularDistributionHistograms all_histograms = temp_refinement.ReturnAngularDistributions(temp_refinement_package->symmetry);
        for ( int class_counter = 0; class_counter < temp_refinement.number_of_classes; class_counter++ ) {
            main_frame->current_project.database.AddRefinementAngularDistribution(all_histograms[class_counter], temp_refinement.refinement_id, class_counter + 1);
        }

        ShortRefinementInfo temp_info;
        temp_info = temp_refinement;

        refinement_package_asset_panel->all_refinement_short_infos.Add(temp_info);
        main_frame->current_project.database.Commit( );
        my_dialog->Destroy( );
    }
    else if ( FrealignRadioButton->GetValue( ) == true ) // FREALIGN
    {
        FrealignParameterFile input_par_file(MetaDataFileTextCtrl->GetLineText(0), OPEN_TO_READ);
        input_par_file.ReadFile(false, stack_number_of_images);

        if ( stack_number_of_images != input_par_file.number_of_lines ) {
            wxMessageBox(wxT("Error: Number of images in stack is different from\nthe number of lines in the par file - aborting."), wxT("Error: Number Mismatch"), wxICON_ERROR);
            return;
        }

        OneSecondProgressDialog* my_dialog = new OneSecondProgressDialog("Refinement Package", "Creating Refinement Package...", stack_number_of_images, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE | wxPD_APP_MODAL);

        float input_parameters[17];

        // create the refinement package and intial refinement..

        temp_refinement_package = new RefinementPackage;
        temp_refinement.SizeAndFillWithEmpty(stack_number_of_images, 1);

        temp_refinement_package->name                     = wxString::Format("Refinement Package #%li (Frealign Import)", refinement_package_asset_panel->current_asset_number);
        temp_refinement_package->number_of_classes        = 1;
        temp_refinement_package->number_of_run_refinments = 0;
        temp_refinement_package->stack_has_white_protein  = WhiteProteinRadioButton->GetValue( );
        temp_refinement_package->output_pixel_size        = PixelSizeTextCtrl->ReturnValue( );

        temp_refinement.number_of_classes                = temp_refinement_package->number_of_classes;
        temp_refinement.number_of_particles              = stack_number_of_images;
        temp_refinement.name                             = "Imported Parameters";
        temp_refinement.resolution_statistics_box_size   = stack_x_size;
        temp_refinement.resolution_statistics_pixel_size = PixelSizeTextCtrl->ReturnValue( );
        temp_refinement.refinement_package_asset_id      = refinement_package_asset_panel->current_asset_number + 1;

        temp_refinement_package->stack_box_size                       = stack_x_size;
        temp_refinement_package->stack_filename                       = ParticleStackFileTextCtrl->GetLineText(0);
        temp_refinement_package->symmetry                             = SymmetryComboBox->GetValue( ).Upper( );
        temp_refinement_package->estimated_particle_weight_in_kda     = MolecularWeightTextCtrl->ReturnValue( );
        temp_refinement_package->estimated_particle_size_in_angstroms = LargestDimensionTextCtrl->ReturnValue( );

        long refinement_id = main_frame->current_project.database.ReturnHighestRefinementID( ) + 1;
        temp_refinement_package->refinement_ids.Add(refinement_id);
        temp_refinement_package->references_for_next_refinement.Add(-1);

        temp_refinement.refinement_id                       = refinement_id;
        temp_refinement.resolution_statistics_are_generated = true;

        temp_particle_info.spherical_aberration = SphericalAberrationTextCtrl->ReturnValue( );
        temp_particle_info.microscope_voltage   = MicroscopeVoltageTextCtrl->ReturnValue( );
        temp_particle_info.parent_image_id      = -1;
        temp_particle_info.pixel_size           = PixelSizeTextCtrl->ReturnValue( );
        temp_particle_info.amplitude_contrast   = AmplitudeContrastTextCtrl->ReturnValue( );
        temp_particle_info.x_pos                = 0;
        temp_particle_info.y_pos                = 0;

        temp_refinement.class_refinement_results[0].class_resolution_statistics.Init(temp_particle_info.pixel_size, temp_refinement.resolution_statistics_box_size);
        temp_refinement.class_refinement_results[0].class_resolution_statistics.GenerateDefaultStatistics(temp_refinement_package->estimated_particle_weight_in_kda);

        // loop over all particles
        int current_subset;
        for ( int particle_counter = 0; particle_counter < stack_number_of_images; particle_counter++ ) {
            input_par_file.ReadLine(input_parameters);

            temp_particle_info.original_particle_position_asset_id = int(input_parameters[0]);
            temp_particle_info.position_in_stack                   = int(input_parameters[0]);
            temp_particle_info.defocus_1                           = input_parameters[8];
            temp_particle_info.defocus_2                           = input_parameters[9];
            temp_particle_info.defocus_angle                       = input_parameters[10];
            temp_particle_info.phase_shift                         = input_parameters[11];

            // we split the stack in 10 and assign half-dataset membership accordingly
            if ( particle_counter / (stack_number_of_images / 10) % 2 ) {
                current_subset = 1;
            }
            else {
                current_subset = 2;
            }
            temp_particle_info.assigned_subset = current_subset;

            temp_refinement_package->contained_particles.Add(temp_particle_info);

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].position_in_stack = int(input_parameters[0]);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus1          = input_parameters[8];
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus2          = input_parameters[9];
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus_angle     = input_parameters[10];
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].phase_shift       = input_parameters[11];
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].logp              = input_parameters[13];

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].occupancy       = input_parameters[12];
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].phi             = input_parameters[3];
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].theta           = input_parameters[2];
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].psi             = input_parameters[1];
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].xshift          = input_parameters[4];
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].yshift          = input_parameters[5];
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].score           = input_parameters[15];
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].image_is_active = int(input_parameters[7]);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].sigma           = input_parameters[14];

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].pixel_size                         = PixelSizeTextCtrl->ReturnValue( );
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].microscope_voltage_kv              = MicroscopeVoltageTextCtrl->ReturnValue( );
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].microscope_spherical_aberration_mm = SphericalAberrationTextCtrl->ReturnValue( );
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].amplitude_contrast                 = AmplitudeContrastTextCtrl->ReturnValue( );

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].assigned_subset = current_subset;

            my_dialog->Update(particle_counter + 1);
        }

        // add to the database and panel..

        main_frame->current_project.database.Begin( );

        refinement_package_asset_panel->AddAsset(temp_refinement_package);
        main_frame->current_project.database.AddRefinement(&temp_refinement);

        ArrayofAngularDistributionHistograms all_histograms = temp_refinement.ReturnAngularDistributions(temp_refinement_package->symmetry);
        for ( int class_counter = 0; class_counter < temp_refinement.number_of_classes; class_counter++ ) {
            main_frame->current_project.database.AddRefinementAngularDistribution(all_histograms[class_counter], temp_refinement.refinement_id, class_counter + 1);
        }

        ShortRefinementInfo temp_info;
        temp_info = temp_refinement;

        refinement_package_asset_panel->all_refinement_short_infos.Add(temp_info);
        main_frame->current_project.database.Commit( );
        my_dialog->Destroy( );
    }
    else if ( RelionRadioButton->GetValue( ) == true ) // RELION
    {

        BasicStarFileReader input_star_file;
        wxString            star_error_text;

        if ( input_star_file.ReadFile(MetaDataFileTextCtrl->GetLineText(0), &star_error_text) == false ) {
            wxMessageBox(wxString::Format("Error: Encountered the following error - aborting :-\n%s", star_error_text), wxT("Error: Cannot read star file"), wxICON_ERROR);
            return;
        }

        if ( stack_number_of_images != input_star_file.cached_parameters.GetCount( ) ) {
            wxMessageBox(wxString::Format("Error: Number of images(%i) in stack is different from\nthe number of parameters read from the star file(%i) - aborting.", stack_number_of_images, input_star_file.cached_parameters.GetCount( )), wxT("Error: Number Mismatch"), wxICON_ERROR);
            return;
        }

        OneSecondProgressDialog* my_dialog = new OneSecondProgressDialog("Refinement Package", "Creating Refinement Package...", stack_number_of_images, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE | wxPD_APP_MODAL);

        // create the refinement package and intial refinement..

        temp_refinement_package = new RefinementPackage;
        temp_refinement.SizeAndFillWithEmpty(stack_number_of_images, 1);

        temp_refinement_package->name                     = wxString::Format("Refinement Package #%li (Relion Import)", refinement_package_asset_panel->current_asset_number);
        temp_refinement_package->number_of_classes        = 1;
        temp_refinement_package->number_of_run_refinments = 0;
        temp_refinement_package->stack_has_white_protein  = WhiteProteinRadioButton->GetValue( );
        temp_refinement_package->output_pixel_size        = PixelSizeTextCtrl->ReturnValue( );

        temp_refinement.number_of_classes                = temp_refinement_package->number_of_classes;
        temp_refinement.number_of_particles              = stack_number_of_images;
        temp_refinement.name                             = "Imported Parameters";
        temp_refinement.resolution_statistics_box_size   = stack_x_size;
        temp_refinement.resolution_statistics_pixel_size = PixelSizeTextCtrl->ReturnValue( );
        temp_refinement.refinement_package_asset_id      = refinement_package_asset_panel->current_asset_number + 1;

        temp_refinement_package->stack_box_size                       = stack_x_size;
        temp_refinement_package->stack_filename                       = ParticleStackFileTextCtrl->GetLineText(0);
        temp_refinement_package->symmetry                             = SymmetryComboBox->GetValue( ).Upper( );
        temp_refinement_package->estimated_particle_weight_in_kda     = MolecularWeightTextCtrl->ReturnValue( );
        temp_refinement_package->estimated_particle_size_in_angstroms = LargestDimensionTextCtrl->ReturnValue( );

        long refinement_id = main_frame->current_project.database.ReturnHighestRefinementID( ) + 1;
        temp_refinement_package->refinement_ids.Add(refinement_id);
        temp_refinement_package->references_for_next_refinement.Add(-1);

        temp_refinement.refinement_id                       = refinement_id;
        temp_refinement.resolution_statistics_are_generated = true;

        temp_particle_info.spherical_aberration = SphericalAberrationTextCtrl->ReturnValue( );
        temp_particle_info.microscope_voltage   = MicroscopeVoltageTextCtrl->ReturnValue( );
        temp_particle_info.parent_image_id      = -1;
        temp_particle_info.pixel_size           = PixelSizeTextCtrl->ReturnValue( );
        temp_particle_info.amplitude_contrast   = AmplitudeContrastTextCtrl->ReturnValue( );
        temp_particle_info.x_pos                = 0;
        temp_particle_info.y_pos                = 0;

        temp_refinement.class_refinement_results[0].class_resolution_statistics.Init(temp_particle_info.pixel_size, temp_refinement.resolution_statistics_box_size);
        temp_refinement.class_refinement_results[0].class_resolution_statistics.GenerateDefaultStatistics(temp_refinement_package->estimated_particle_weight_in_kda);

        // loop over all particles

        for ( int particle_counter = 0; particle_counter < stack_number_of_images; particle_counter++ ) {
            temp_particle_info.original_particle_position_asset_id = input_star_file.ReturnPositionInStack(particle_counter);
            temp_particle_info.position_in_stack                   = input_star_file.ReturnPositionInStack(particle_counter);
            temp_particle_info.defocus_1                           = input_star_file.ReturnDefocus1(particle_counter);
            temp_particle_info.defocus_2                           = input_star_file.ReturnDefocus2(particle_counter);
            temp_particle_info.defocus_angle                       = input_star_file.ReturnDefocusAngle(particle_counter);
            temp_particle_info.phase_shift                         = input_star_file.ReturnPhaseShift(particle_counter);

            temp_particle_info.x_pos = input_star_file.ReturnXCoordinate(particle_counter) * PixelSizeTextCtrl->ReturnValue( );
            temp_particle_info.y_pos = input_star_file.ReturnYCoordinate(particle_counter) * PixelSizeTextCtrl->ReturnValue( );

            temp_particle_info.assigned_subset = input_star_file.ReturnAssignedSubset(particle_counter);

            temp_refinement_package->contained_particles.Add(temp_particle_info);

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].position_in_stack = input_star_file.ReturnPositionInStack(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus1          = input_star_file.ReturnDefocus1(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus2          = input_star_file.ReturnDefocus2(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus_angle     = input_star_file.ReturnDefocusAngle(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].phase_shift       = input_star_file.ReturnPhaseShift(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].logp              = 0;

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].occupancy = 100.0;
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].phi       = input_star_file.ReturnPhi(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].theta     = input_star_file.ReturnTheta(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].psi       = input_star_file.ReturnPsi(particle_counter);

            if ( input_star_file.x_shifts_are_in_angst ) {
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].xshift = -input_star_file.ReturnXShift(particle_counter);
            }
            else {
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].xshift = -input_star_file.ReturnXShift(particle_counter) * PixelSizeTextCtrl->ReturnValue( );
            }

            if ( input_star_file.y_shifts_are_in_angst ) {
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].yshift = -input_star_file.ReturnYShift(particle_counter);
            }
            else {
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].yshift = -input_star_file.ReturnYShift(particle_counter) * PixelSizeTextCtrl->ReturnValue( );
            }

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].score           = 0.0;
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].image_is_active = 1;
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].sigma           = 10.0;

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].pixel_size                         = PixelSizeTextCtrl->ReturnValue( );
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].microscope_voltage_kv              = MicroscopeVoltageTextCtrl->ReturnValue( );
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].microscope_spherical_aberration_mm = SphericalAberrationTextCtrl->ReturnValue( );
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].amplitude_contrast                 = AmplitudeContrastTextCtrl->ReturnValue( );

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].assigned_subset = input_star_file.ReturnAssignedSubset(particle_counter);

            my_dialog->Update(particle_counter + 1);
        }

        // add to the database and panel..

        main_frame->current_project.database.Begin( );
        refinement_package_asset_panel->AddAsset(temp_refinement_package);
        main_frame->current_project.database.AddRefinement(&temp_refinement);

        ArrayofAngularDistributionHistograms all_histograms;
        all_histograms = temp_refinement.ReturnAngularDistributions(temp_refinement_package->symmetry);
        for ( int class_counter = 0; class_counter < temp_refinement.number_of_classes; class_counter++ ) {
            main_frame->current_project.database.AddRefinementAngularDistribution(all_histograms[class_counter], temp_refinement.refinement_id, class_counter + 1);
        }

        ShortRefinementInfo temp_info;
        temp_info = temp_refinement;

        refinement_package_asset_panel->all_refinement_short_infos.Add(temp_info);
        main_frame->current_project.database.Commit( );
        my_dialog->Destroy( );
    }

    /*
	// create the directory if it doesn't exist (which it shouldn't)

	wxFileName current_dirname = wxFileName::DirName(ProjectPathTextCtrl->GetValue());

	if (current_dirname.Exists())
	{
		MyDebugPrintWithDetails("Directory should not already exist, and does!\n");
	}
	else current_dirname.Mkdir();

	wxString wanted_database_file = ProjectPathTextCtrl->GetValue();
	if (wanted_database_file.EndsWith("/") == false) wanted_database_file += "/";
	wanted_database_file += ProjectNameTextCtrl->GetValue();
	wanted_database_file += ".db";

	main_frame->current_project.CreateNewProject(wanted_database_file, ProjectPathTextCtrl->GetValue(), ProjectNameTextCtrl->GetValue());
	*/
}
