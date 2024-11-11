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

template <typename StarFileSource_t>
void ImportRefinementPackageWizard::ImportRefinementPackage(StarFileSource_t& input_params_file, const int stack_x_size, const int stack_num_images) {
    float    pixel_size;
    float    spherical_aberration_nm;
    float    microscope_voltage_kv;
    float    amplitude_contrast;
    float    molecular_weight_kda;
    float    largest_dimension;
    wxString refinement_package_name;
    wxString symmetry;

    RefinementPackage*            temp_refinement_package = new RefinementPackage( );
    RefinementPackageParticleInfo temp_particle_info;
    Refinement                    temp_refinement;

    constexpr bool is_cistem_import   = std::is_same_v<StarFileSource_t, cisTEMParameters>;
    constexpr bool is_frealign_import = std::is_same_v<StarFileSource_t, FrealignParameterFile>;
    constexpr bool is_relion_import   = std::is_same_v<StarFileSource_t, BasicStarFileReader>;

    if constexpr ( is_cistem_import ) {
        refinement_package_name = wxString::Format("Refinement Package #%li (cisTEM Import)",
                                                   refinement_package_asset_panel->current_asset_number);

        pixel_size            = input_params_file.ReturnPixelSize(0);
        microscope_voltage_kv = input_params_file.ReturnMicroscopekV(0);
        amplitude_contrast    = input_params_file.ReturnAmplitudeContrast(0);
    }
    else if constexpr ( is_frealign_import ) {
        refinement_package_name = wxString::Format("Refinement Package #%li (Frealign Import)",
                                                   refinement_package_asset_panel->current_asset_number);

        pixel_size            = PixelSizeTextCtrl->ReturnValue( );
        microscope_voltage_kv = MicroscopeVoltageTextCtrl->ReturnValue( );
        amplitude_contrast    = AmplitudeContrastTextCtrl->ReturnValue( );
    }
    else if constexpr ( is_relion_import ) {
        refinement_package_name = wxString::Format("Refinement Package #%li (Relion Import)",
                                                   refinement_package_asset_panel->current_asset_number);

        pixel_size            = PixelSizeTextCtrl->ReturnValue( );
        microscope_voltage_kv = MicroscopeVoltageTextCtrl->ReturnValue( );
        amplitude_contrast    = AmplitudeContrastTextCtrl->ReturnValue( );
    }

    spherical_aberration_nm = SphericalAberrationTextCtrl->ReturnValue( );
    molecular_weight_kda    = MolecularWeightTextCtrl->ReturnValue( );
    largest_dimension       = LargestDimensionTextCtrl->ReturnValue( );
    symmetry                = SymmetryComboBox->GetValue( ).Upper( );

    OneSecondProgressDialog* my_dialog = new OneSecondProgressDialog("Refinement Package", "Creating Refinement Package...", stack_num_images, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE | wxPD_APP_MODAL);

    // create the refinement package and intial refinement..
    temp_refinement_package->name                     = refinement_package_name;
    temp_refinement_package->number_of_classes        = 1;
    temp_refinement_package->number_of_run_refinments = 0;
    temp_refinement_package->stack_has_white_protein  = WhiteProteinRadioButton->GetValue( );
    temp_refinement_package->output_pixel_size        = pixel_size;

    temp_refinement.number_of_classes                = temp_refinement_package->number_of_classes;
    temp_refinement.number_of_particles              = stack_num_images;
    temp_refinement.name                             = "Imported Parameters";
    temp_refinement.resolution_statistics_box_size   = stack_x_size;
    temp_refinement.resolution_statistics_pixel_size = pixel_size;
    temp_refinement.refinement_package_asset_id      = refinement_package_asset_panel->current_asset_number + 1;

    temp_refinement_package->stack_box_size                       = stack_x_size;
    temp_refinement_package->stack_filename                       = ParticleStackFileTextCtrl->GetLineText(0);
    temp_refinement_package->symmetry                             = symmetry;
    temp_refinement_package->estimated_particle_weight_in_kda     = molecular_weight_kda;
    temp_refinement_package->estimated_particle_size_in_angstroms = largest_dimension;

    long refinement_id = main_frame->current_project.database.ReturnHighestRefinementID( ) + 1;
    temp_refinement_package->refinement_ids.Add(refinement_id);
    temp_refinement_package->references_for_next_refinement.Add(-1);

    temp_refinement.refinement_id                       = refinement_id;
    temp_refinement.resolution_statistics_are_generated = true;

    // These will all be the same and don't need to be changed during each iteration
    temp_particle_info.spherical_aberration = spherical_aberration_nm;
    temp_particle_info.microscope_voltage   = microscope_voltage_kv;
    temp_particle_info.parent_image_id      = -1;
    temp_particle_info.amplitude_contrast   = amplitude_contrast;

    temp_particle_info.pixel_size = pixel_size;
    temp_particle_info.x_pos      = 0;
    temp_particle_info.y_pos      = 0;

    temp_refinement.SizeAndFillWithEmpty(stack_num_images, 1);
    temp_refinement.class_refinement_results[0].class_resolution_statistics.Init(pixel_size, temp_refinement.resolution_statistics_box_size);
    temp_refinement.class_refinement_results[0].class_resolution_statistics.GenerateDefaultStatistics(temp_refinement_package->estimated_particle_weight_in_kda);

    for ( int particle_counter = 0; particle_counter < stack_num_images; particle_counter++ ) {
        // cisTEMParameters and BasicStarFileReader (for reading Relion format) have several identical function names that can be lumped together; so, to save code,
        // they are included in this if block, with further specific checks for divergence in parameter loading.
        if constexpr ( is_cistem_import || is_relion_import ) {

            // Start with particles' info, then perform checks to see which format we're using to finish filling the info
            temp_particle_info.original_particle_position_asset_id = input_params_file.ReturnPositionInStack(particle_counter);
            temp_particle_info.position_in_stack                   = input_params_file.ReturnPositionInStack(particle_counter);
            temp_particle_info.defocus_1                           = input_params_file.ReturnDefocus1(particle_counter);
            temp_particle_info.defocus_2                           = input_params_file.ReturnDefocus2(particle_counter);
            temp_particle_info.defocus_angle                       = input_params_file.ReturnDefocusAngle(particle_counter);
            temp_particle_info.phase_shift                         = input_params_file.ReturnPhaseShift(particle_counter);
            temp_particle_info.assigned_subset                     = input_params_file.ReturnAssignedSubset(particle_counter);

            // Finish the particles' info with specific checks, and while doing so, fill in specific refinement result parameters as well
            if constexpr ( is_relion_import ) {
                temp_particle_info.amplitude_contrast = amplitude_contrast;
                temp_particle_info.pixel_size         = pixel_size;
                temp_refinement_package->contained_particles.Add(temp_particle_info);

                if ( input_params_file.x_shifts_are_in_angst )
                    temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].xshift = -input_params_file.ReturnXShift(particle_counter);
                else
                    temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].xshift = -input_params_file.ReturnXShift(particle_counter) * pixel_size;

                if ( input_params_file.y_shifts_are_in_angst )
                    temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].yshift = -input_params_file.ReturnYShift(particle_counter);
                else
                    temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].yshift = -input_params_file.ReturnYShift(particle_counter) * pixel_size;

                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].score           = 0.0;
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].image_is_active = 1;
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].sigma           = 10.0;
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].logp            = 0.0;
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].occupancy       = 100.0;
            }
            else if constexpr ( is_cistem_import ) {
                temp_particle_info.pixel_size         = input_params_file.ReturnPixelSize(particle_counter);
                temp_particle_info.amplitude_contrast = input_params_file.ReturnAmplitudeContrast(particle_counter);
                temp_refinement_package->contained_particles.Add(temp_particle_info);

                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].xshift          = input_params_file.ReturnXShift(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].yshift          = input_params_file.ReturnYShift(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].score           = input_params_file.ReturnScore(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].image_is_active = (int)input_params_file.ReturnImageIsActive(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].sigma           = input_params_file.ReturnSigma(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].beam_tilt_x     = input_params_file.ReturnBeamTiltX(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].beam_tilt_y     = input_params_file.ReturnBeamTiltY(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].image_shift_x   = input_params_file.ReturnImageShiftX(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].image_shift_y   = input_params_file.ReturnImageShiftY(particle_counter);

                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].beam_tilt_group = input_params_file.ReturnBeamTiltGroup(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].particle_group  = input_params_file.ReturnParticleGroup(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].pre_exposure    = input_params_file.ReturnPreExposure(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].total_exposure  = input_params_file.ReturnTotalExposure(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].logp            = input_params_file.ReturnLogP(particle_counter);
                temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].occupancy       = input_params_file.ReturnOccupancy(particle_counter);
            }

            // Finish filling in the parameters that have the same function names and can be retrieved the same way for cisTEM/Relion imports
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].position_in_stack = (int)input_params_file.ReturnPositionInStack(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus1          = input_params_file.ReturnDefocus1(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus2          = input_params_file.ReturnDefocus2(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus_angle     = input_params_file.ReturnDefocusAngle(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].phase_shift       = input_params_file.ReturnPhaseShift(particle_counter);

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].phi             = input_params_file.ReturnPhi(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].theta           = input_params_file.ReturnTheta(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].psi             = input_params_file.ReturnPsi(particle_counter);
            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].assigned_subset = input_params_file.ReturnAssignedSubset(particle_counter);
        }
        else if constexpr ( is_frealign_import ) {
            float input_parameters[17];
            int   current_subset;
            input_params_file.ReadLine(input_parameters);

            temp_particle_info.original_particle_position_asset_id = int(input_parameters[0]);
            temp_particle_info.position_in_stack                   = int(input_parameters[0]);
            temp_particle_info.defocus_1                           = input_parameters[8];
            temp_particle_info.defocus_2                           = input_parameters[9];
            temp_particle_info.defocus_angle                       = input_parameters[10];
            temp_particle_info.phase_shift                         = input_parameters[11];

            // we split the stack in 10 and assign half-dataset membership accordingly
            if ( particle_counter / (stack_num_images / 10) % 2 ) {
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

            temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].assigned_subset = current_subset;
        }
        // These are general to ALL types of parameters files? Worth checking with Tim about
        temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].pixel_size                         = pixel_size;
        temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].microscope_voltage_kv              = microscope_voltage_kv;
        temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].microscope_spherical_aberration_mm = spherical_aberration_nm;
        temp_refinement.class_refinement_results[0].particle_refinement_results[particle_counter].amplitude_contrast                 = amplitude_contrast;

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

template void ImportRefinementPackageWizard::ImportRefinementPackage<cisTEMParameters>(cisTEMParameters& input_params_file, const int stack_x_size, const int stack_num_images);
template void ImportRefinementPackageWizard::ImportRefinementPackage<FrealignParameterFile>(FrealignParameterFile& input_params_file, const int stack_x_size, const int stack_num_images);
template void ImportRefinementPackageWizard::ImportRefinementPackage<BasicStarFileReader>(BasicStarFileReader& input_params_file, const int stack_x_size, const int stack_num_images);

void ImportRefinementPackageWizard::OnFinished(wxWizardEvent& event) {
    int stack_x_size;
    int stack_y_size;
    int stack_number_of_images;

    // Verify stack integrity, check which type of import we're doing
    bool stack_is_ok = GetMRCDetails(ParticleStackFileTextCtrl->GetLineText(0).ToUTF8( ).data( ), stack_x_size, stack_y_size, stack_number_of_images);

    if ( ! stack_is_ok ) {
        wxMessageBox(wxT("Error: Cannot read the stack file - aborting."), wxT("Error Reading particle stack"), wxICON_ERROR);
        return;
    }

    if ( stack_x_size != stack_y_size ) {
        wxMessageBox(wxT("Error: Only square images are currently supported - aborting."), wxT("Error images are not square"), wxICON_ERROR);
        return;
    }

    if ( cisTEMRadioButton->GetValue( ) == true ) {

        cisTEMParameters input_star_file;
        input_star_file.ReadFromcisTEMStarFile(MetaDataFileTextCtrl->GetLineText(0), true);

        if ( stack_number_of_images != input_star_file.ReturnNumberofLines( ) ) {
            wxMessageBox(wxT("Error: Number of images in stack is different from\nthe number of lines in the star file - aborting."), wxT("Error: Number Mismatch"), wxICON_ERROR);
            return;
        }
        ImportRefinementPackage(input_star_file, stack_x_size, stack_number_of_images);
    }
    else if ( FrealignRadioButton->GetValue( ) == true ) {
        FrealignParameterFile input_par_file(MetaDataFileTextCtrl->GetLineText(0), OPEN_TO_READ);
        input_par_file.ReadFile(false, stack_number_of_images);

        if ( stack_number_of_images != input_par_file.number_of_lines ) {
            wxMessageBox(wxT("Error: Number of images in stack is different from\nthe number of lines in the par file - aborting."), wxT("Error: Number Mismatch"), wxICON_ERROR);
            return;
        }
        ImportRefinementPackage(input_par_file, stack_x_size, stack_number_of_images);
    }
    else if ( RelionRadioButton->GetValue( ) == true ) {
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
        ImportRefinementPackage(input_star_file, stack_x_size, stack_number_of_images);
    }
}
