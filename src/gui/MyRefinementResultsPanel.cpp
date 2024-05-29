//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel* refinement_package_asset_panel;
extern MyRefinementResultsPanel*      refinement_results_panel;
extern MyVolumeAssetPanel*            volume_asset_panel;

MyRefinementResultsPanel::MyRefinementResultsPanel(wxWindow* parent)
    : RefinementResultsPanel(parent) {
    refinement_package_is_dirty = false;
    input_params_are_dirty      = false;
    current_class               = 0;
    FSCPlotPanel->Clear( );

    currently_displayed_refinement = NULL;
    buffered_full_refinement       = NULL;

    OrthPanel->EnableStartWithFourierScaling( );
    OrthPanel->EnableDoNotShowStatusBar( );
    OrthPanel->Initialise( );
    OrthPanel->my_notebook->Bind(wxEVT_AUINOTEBOOK_PAGE_CHANGED, &MyRefinementResultsPanel::OnDisplayTabChange, this);

    RefinementPackageComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MyRefinementResultsPanel::OnRefinementPackageComboBox, this);
    InputParametersComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MyRefinementResultsPanel::OnInputParametersComboBox, this);

#include "icons/show_angles.cpp"
#include "icons/show_text.cpp"

    wxBitmap angles_popup_bmp     = wxBITMAP_PNG_FROM_DATA(show_angles);
    wxBitmap parameters_popup_bmp = wxBITMAP_PNG_FROM_DATA(show_text);
    AngularPlotDetailsButton->SetBitmap(angles_popup_bmp);
    ParametersDetailButton->SetBitmap(parameters_popup_bmp);

    //	FSCPlotPanel->ClassComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( MyRefinementResultsPanel::OnClassComboBoxChange ), NULL, this );
}

void MyRefinementResultsPanel::OnClassComboBoxChange(wxCommandEvent& event) {
    //wxPrintf("Changed\n");
    //	current_class = FSCPlotPanel->ClassComboBox->GetSelection();
    //FillAngles();
    //	ParameterListCtrl->RefreshItems(0, ParameterListCtrl->GetItemCount() - 1);
    //	FSCPlotPanel->PlotCurrentClass();
    event.Skip( );
}

void MyRefinementResultsPanel::OnDisplayTabChange(wxAuiNotebookEvent& event) {
    wxPrintf("Changed to tab %i\n", OrthPanel->my_notebook->GetSelection( ));
    wxPrintf("highlighting %i\n", OrthPanel->my_notebook->GetSelection( ));
    if ( OrthPanel->my_notebook->GetSelection( ) >= 0 ) {
        FSCPlotPanel->HighlightClass(OrthPanel->my_notebook->GetSelection( ));
        FillAngles(OrthPanel->my_notebook->GetSelection( ));
        WriteJobInfo(OrthPanel->my_notebook->GetSelection( ));
    }

    event.Skip( );
}

void MyRefinementResultsPanel::FillRefinementPackageComboBox(void) {

    RefinementPackageComboBox->FillComboBox( );
    FillInputParametersComboBox( );
}

void MyRefinementResultsPanel::FillInputParametersComboBox(void) {
    if ( RefinementPackageComboBox->GetSelection( ) >= 0 ) {
        InputParametersComboBox->FillComboBox(RefinementPackageComboBox->GetSelection( ));

        UpdateCachedRefinement( );
        FSCPlotPanel->AddRefinement(currently_displayed_refinement);
        DrawOrthViews( ); // I hope this will cause  redraw of the angles also..
        if ( OrthPanel->my_notebook->GetSelection( ) >= 0 )
            FillAngles(OrthPanel->my_notebook->GetSelection( ));
        if ( OrthPanel->my_notebook->GetSelection( ) >= 0 )
            WriteJobInfo(OrthPanel->my_notebook->GetSelection( ));
    }
}

void MyRefinementResultsPanel::UpdateCachedRefinement( ) {
    //	wxPrintf("refinement package selection = %i, parameter selcetion = %i\n", refinement_results_panel->RefinementPackageComboBox->GetSelection(), refinement_results_panel->InputParametersComboBox->GetSelection());
    if ( currently_displayed_refinement == NULL || refinement_package_asset_panel->all_refinement_packages[refinement_results_panel->RefinementPackageComboBox->GetSelection( )].refinement_ids[refinement_results_panel->InputParametersComboBox->GetSelection( )] != currently_displayed_refinement->refinement_id ) {
        //wxProgressDialog progress_dialog("Please wait", "Retrieving refinement result from database...", 0, this);

        if ( currently_displayed_refinement != NULL )
            delete currently_displayed_refinement;
        currently_displayed_refinement = main_frame->current_project.database.GetRefinementByID(refinement_package_asset_panel->all_refinement_packages[refinement_results_panel->RefinementPackageComboBox->GetSelection( )].refinement_ids[refinement_results_panel->InputParametersComboBox->GetSelection( )], false);
    }
}

void MyRefinementResultsPanel::UpdateBufferedFullRefinement( ) {
    if ( refinement_id_of_buffered_refinement != currently_displayed_refinement->refinement_id && currently_displayed_refinement != NULL ) {
        if ( buffered_full_refinement != NULL )
            delete buffered_full_refinement;

        wxProgressDialog progress_dialog("Please wait", "Retrieving refinement result from database...", 0, this);
        buffered_full_refinement             = main_frame->current_project.database.GetRefinementByID(currently_displayed_refinement->refinement_id);
        refinement_id_of_buffered_refinement = currently_displayed_refinement->refinement_id;
    }
}

void MyRefinementResultsPanel::OnJobDetailsToggle(wxCommandEvent& event) {
    Freeze( );

    if ( JobDetailsToggleButton->GetValue( ) == true ) {
        JobDetailsPanel->Show(true);
    }
    else {
        JobDetailsPanel->Show(false);
    }

    RightPanel->Layout( );
    Thaw( );
}

void MyRefinementResultsPanel::Clear( ) {
    JobDetailsToggleButton->SetValue(false);
    JobDetailsPanel->Show(false);
    RefinementPackageComboBox->Clear( );
    InputParametersComboBox->Clear( );
    AngularPlotPanel->Clear( );
    AngularPlotPanel->Refresh( );
    OrthPanel->Clear( );
    Layout( );
}

void MyRefinementResultsPanel::OnUpdateUI(wxUpdateUIEvent& event) {
    if ( main_frame->current_project.is_open == false ) {
        Enable(false);
        RefinementPackageComboBox->Clear( );
        RefinementPackageComboBox->ChangeValue("");
        InputParametersComboBox->Clear( );
        InputParametersComboBox->ChangeValue("");
        //		ParameterListCtrl->ClearAll();
        //		ParameterListCtrl->SetItemCount(0);
        FSCPlotPanel->Clear( );
    }
    else {
        Enable(true);

        if ( refinement_package_is_dirty == true ) {
            refinement_package_is_dirty = false;
            FillRefinementPackageComboBox( );
            //AngularPlotPanel->Clear();
            //	FillAngles();
        }

        if ( input_params_are_dirty == true ) {
            input_params_are_dirty = false;
            FillInputParametersComboBox( );
        }
    }
}

void MyRefinementResultsPanel::OnRefinementPackageComboBox(wxCommandEvent& event) {
    OrthPanel->my_notebook->Unbind(wxEVT_AUINOTEBOOK_PAGE_CHANGED, &MyRefinementResultsPanel::OnDisplayTabChange, this);
    OrthPanel->Clear( );
    OrthPanel->my_notebook->Bind(wxEVT_AUINOTEBOOK_PAGE_CHANGED, &MyRefinementResultsPanel::OnDisplayTabChange, this);

    if ( RefinementPackageComboBox->GetSelection( ) >= 0 ) {
        FillInputParametersComboBox( );
    }
}

void MyRefinementResultsPanel::OnInputParametersComboBox(wxCommandEvent& event) {
    if ( RefinementPackageComboBox->GetSelection( ) >= 0 ) {
        UpdateCachedRefinement( );
        FSCPlotPanel->AddRefinement(currently_displayed_refinement);
        DrawOrthViews( ); // Should redraw angles also
        if ( OrthPanel->my_notebook->GetSelection( ) >= 0 )
            FillAngles(OrthPanel->my_notebook->GetSelection( ));
        if ( OrthPanel->my_notebook->GetSelection( ) >= 0 )
            WriteJobInfo(OrthPanel->my_notebook->GetSelection( ));

        //AngularPlotPanel->Clear();
    }
}

void MyRefinementResultsPanel::FillAngles(int wanted_class) {
    if ( RefinementPackageComboBox->GetSelection( ) >= 0 && OrthPanel->my_notebook->GetSelection( ) >= 0 ) {
        AngularPlotPanel->Freeze( );
        //AngularPlotPanel->Clear();
        AngularPlotPanel->SetSymmetryAndNumber(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection( )].symmetry, refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection( )].contained_particles.GetCount( ));
        AngularPlotPanel->distribution_histogram.Init(18, 72); // hard coded size

        //  get the histogram from the database..

        main_frame->current_project.database.GetRefinementAngularDistributionHistogramData(currently_displayed_refinement->refinement_id, wanted_class + 1, AngularPlotPanel->distribution_histogram);
        AngularPlotPanel->Thaw( );
        AngularPlotPanel->SetupBitmap( );
        AngularPlotPanel->Refresh( );
    }
}

void MyRefinementResultsPanel::DrawOrthViews( ) {

    if ( RefinementPackageComboBox->GetSelection( ) >= 0 && currently_displayed_refinement != NULL ) {
        OrthPanel->Freeze( );
        wxString current_orth_filename;
        long     array_position;
        OrthPanel->my_notebook->Unbind(wxEVT_AUINOTEBOOK_PAGE_CHANGED, &MyRefinementResultsPanel::OnDisplayTabChange, this);

        if ( OrthPanel->my_notebook->GetPageCount( ) != currently_displayed_refinement->number_of_classes ) {
            OrthPanel->Clear( );

            for ( int class_counter = 0; class_counter < currently_displayed_refinement->number_of_classes; class_counter++ ) {
                // open the relevant orth view (if available)
                array_position = volume_asset_panel->ReturnArrayPositionFromAssetID(currently_displayed_refinement->class_refinement_results[class_counter].reconstructed_volume_asset_id);

                if ( array_position != -1 ) {
                    current_orth_filename = main_frame->current_project.volume_asset_directory.GetFullPath( ) + "/OrthViews/" + volume_asset_panel->ReturnAssetShortFilename(array_position);

                    if ( DoesFileExist(current_orth_filename) == true ) {
                        if ( currently_displayed_refinement->number_of_classes == 1 )
                            OrthPanel->OpenFile(current_orth_filename, wxString::Format(wxT("#%i (%.2fÅ)"), class_counter + 1, currently_displayed_refinement->class_refinement_results[0].estimated_resolution));
                        else {
                            OrthPanel->OpenFile(current_orth_filename, wxString::Format(wxT("#%i (%.2f%%, %.2fÅ)"), class_counter + 1, currently_displayed_refinement->class_refinement_results[0].average_occupancy, currently_displayed_refinement->class_refinement_results[0].estimated_resolution));
                        }
                        //OrthPanel->my_notebook->DoSizing();
                    }
                    else {
                        Image* dummy = new Image;
                        dummy->Allocate(300, 200, 1);
                        dummy->SetToConstant(0.0f);

                        if ( currently_displayed_refinement->number_of_classes == 1 )
                            OrthPanel->ChangeImage(dummy, wxString::Format(wxT("#%i (%.2fÅ)"), class_counter + 1, currently_displayed_refinement->class_refinement_results[0].estimated_resolution));
                        else {
                            OrthPanel->ChangeImage(dummy, wxString::Format(wxT("#%i (%.2f%%, %.2fÅ)"), class_counter + 1, currently_displayed_refinement->class_refinement_results[class_counter].average_occupancy, currently_displayed_refinement->class_refinement_results[class_counter].estimated_resolution));
                        }
                    }
                }
                else {
                    Image* dummy = new Image;
                    dummy->Allocate(300, 200, 1);
                    dummy->SetToConstant(0.0f);

                    if ( currently_displayed_refinement->number_of_classes == 1 )
                        OrthPanel->ChangeImage(dummy, wxString::Format(wxT("#%i (%.2fÅ)"), class_counter + 1, currently_displayed_refinement->class_refinement_results[0].estimated_resolution));
                    else {
                        OrthPanel->ChangeImage(dummy, wxString::Format(wxT("#%i (%.2f%%, %.2fÅ)"), class_counter + 1, currently_displayed_refinement->class_refinement_results[class_counter].average_occupancy, currently_displayed_refinement->class_refinement_results[class_counter].estimated_resolution));
                    }
                }
            }

            OrthPanel->my_notebook->Bind(wxEVT_AUINOTEBOOK_PAGE_CHANGED, &MyRefinementResultsPanel::OnDisplayTabChange, this);
            OrthPanel->my_notebook->SetSelection(0);
        }
        else {
            int current_page = OrthPanel->my_notebook->GetSelection( );

            for ( int class_counter = 0; class_counter < currently_displayed_refinement->number_of_classes; class_counter++ ) {
                // open the relevant orth view (if available)
                OrthPanel->my_notebook->ChangeSelection(class_counter);
                array_position = volume_asset_panel->ReturnArrayPositionFromAssetID(currently_displayed_refinement->class_refinement_results[class_counter].reconstructed_volume_asset_id);
                if ( array_position != -1 ) {
                    current_orth_filename = main_frame->current_project.volume_asset_directory.GetFullPath( ) + "/OrthViews/" + volume_asset_panel->ReturnAssetShortFilename(array_position);

                    if ( DoesFileExist(current_orth_filename) == true ) {
                        wxPrintf("current res = %.2f\n", currently_displayed_refinement->class_refinement_results[0].estimated_resolution);
                        if ( currently_displayed_refinement->number_of_classes == 1 )
                            OrthPanel->ChangeFile(current_orth_filename, wxString::Format(wxT("#%i (%.2fÅ)"), class_counter + 1, currently_displayed_refinement->class_refinement_results[0].estimated_resolution));
                        else {
                            OrthPanel->ChangeFile(current_orth_filename, wxString::Format(wxT("#%i (%.2f%%, %.2fÅ)"), class_counter + 1, currently_displayed_refinement->class_refinement_results[class_counter].average_occupancy, currently_displayed_refinement->class_refinement_results[class_counter].estimated_resolution));
                        }
                        //OrthPanel->my_notebook->DoSizing();
                    }
                    else {
                        Image* dummy = new Image;
                        dummy->Allocate(300, 200, 1);
                        dummy->SetToConstant(0.0f);

                        if ( currently_displayed_refinement->number_of_classes == 1 )
                            OrthPanel->ChangeImage(dummy, wxString::Format(wxT("#%i (%.2fÅ)"), class_counter + 1, currently_displayed_refinement->class_refinement_results[0].estimated_resolution));
                        else {
                            OrthPanel->ChangeImage(dummy, wxString::Format(wxT("#%i (%.2f%%, %.2fÅ)"), class_counter + 1, currently_displayed_refinement->class_refinement_results[class_counter].average_occupancy, currently_displayed_refinement->class_refinement_results[class_counter].estimated_resolution));
                        }
                    }
                }
                else {
                    Image* dummy = new Image;
                    dummy->Allocate(300, 200, 1);
                    dummy->SetToConstant(0.0f);

                    if ( currently_displayed_refinement->number_of_classes == 1 )
                        OrthPanel->ChangeImage(dummy, wxString::Format(wxT("#%i (%.2fÅ)"), class_counter + 1, currently_displayed_refinement->class_refinement_results[0].estimated_resolution));
                    else {
                        OrthPanel->ChangeImage(dummy, wxString::Format(wxT("#%i (%.2f%%, %.2fÅ)"), class_counter + 1, currently_displayed_refinement->class_refinement_results[class_counter].average_occupancy, currently_displayed_refinement->class_refinement_results[class_counter].estimated_resolution));
                    }
                }
            }

            //	wxPrintf("Setting selection %i\n", current_page);
            OrthPanel->my_notebook->Bind(wxEVT_AUINOTEBOOK_PAGE_CHANGED, &MyRefinementResultsPanel::OnDisplayTabChange, this);
            OrthPanel->my_notebook->SetSelection(current_page);
        }

        OrthPanel->my_notebook->Refresh( );
        OrthPanel->my_notebook->Update( );
        OrthPanel->Thaw( );
    }
}

void MyRefinementResultsPanel::AngularPlotPopupClick(wxCommandEvent& event) {
    if ( RefinementPackageComboBox->GetSelection( ) >= 0 && OrthPanel->my_notebook->GetPageCount( ) > 0 ) {
        long        particle_counter;
        int         class_counter;
        int         wanted_class = OrthPanel->my_notebook->GetSelection( );
        wxArrayLong images_to_add;

        UpdateBufferedFullRefinement( );
        LargeAngularPlotDialog* plot_dialog = new LargeAngularPlotDialog(this, wxID_ANY, "Angular Plot");

        for ( particle_counter = 0; particle_counter < buffered_full_refinement->number_of_particles; particle_counter++ ) {
            if ( buffered_full_refinement->ReturnClassWithHighestOccupanyForGivenParticle(particle_counter) == wanted_class ) {
                if ( buffered_full_refinement->class_refinement_results[wanted_class].particle_refinement_results[particle_counter].image_is_active >= 0 )
                    images_to_add.Add(particle_counter);
            }
        }

        plot_dialog->AngularPlotPanel->draw_axis_overlay_instead_of_underlay = true;
        plot_dialog->AngularPlotPanel->SetSymmetryAndNumber(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection( )].symmetry, images_to_add.GetCount( ));

        for ( particle_counter = 0; particle_counter < images_to_add.GetCount( ); particle_counter++ ) {
            plot_dialog->AngularPlotPanel->AddRefinementResult(&buffered_full_refinement->class_refinement_results[wanted_class].particle_refinement_results[images_to_add[particle_counter]]);
        }

        //buffered_full_refinement->FillAngularDistributionHistogram(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].symmetry, OrthPanel->my_notebook->GetSelection(), 72, 288, plot_dialog->AngularPlotPanel->distribution_histogram);
        plot_dialog->ShowModal( );
        plot_dialog->Destroy( );
    }
}

void MyRefinementResultsPanel::PopupParametersClick(wxCommandEvent& event) {
    if ( RefinementPackageComboBox->GetSelection( ) >= 0 && OrthPanel->my_notebook->GetPageCount( ) > 0 ) {
        UpdateBufferedFullRefinement( );
        RefinementParametersDialog* parameters_dialog = new RefinementParametersDialog(this, wxID_ANY, "Parameters");
        parameters_dialog->ShowModal( );
        parameters_dialog->Destroy( );
    }
}

void MyRefinementResultsPanel::WriteJobInfo(int wanted_class) {
    RefinementIDStaticText->SetLabel(wxString::Format("%li", currently_displayed_refinement->refinement_id));
    DateOfRunStaticText->SetLabel(currently_displayed_refinement->datetime_of_run.FormatISODate( ));
    TimeOfRunStaticText->SetLabel(currently_displayed_refinement->datetime_of_run.FormatISOTime( ));
    PercentUsedStaticText->SetLabel(wxString::Format("%.2f %%", currently_displayed_refinement->percent_used));
    ReferenceVolumeIDStaticText->SetLabel(wxString::Format("%li", currently_displayed_refinement->reference_volume_ids[wanted_class]));
    ReferenceRefinementIDStaticText->SetLabel(wxString::Format("%li", currently_displayed_refinement->starting_refinement_id));
    LowResLimitStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].low_resolution_limit));
    HighResLimitStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].high_resolution_limit));
    MaskRadiusStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].mask_radius));
    SignedCCResLimitStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].signed_cc_resolution_limit));
    GlobalResLimitStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].global_resolution_limit));
    GlobalMaskRadiusStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].global_mask_radius));
    NumberResultsRefinedStaticText->SetLabel(wxString::Format(wxT("%i"), currently_displayed_refinement->class_refinement_results[wanted_class].number_results_to_refine));
    AngularSearchStepStaticText->SetLabel(wxString::Format(wxT("%.2f °"), currently_displayed_refinement->class_refinement_results[wanted_class].angular_search_step));
    SearchRangeXStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].search_range_x));
    SearchRangeYStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].search_range_y));
    ClassificationResLimitStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].classification_resolution_limit));

    if ( currently_displayed_refinement->class_refinement_results[wanted_class].should_focus_classify == true )
        ShouldFocusClassifyStaticText->SetLabel("Yes");
    else
        ShouldFocusClassifyStaticText->SetLabel("No");

    SphereXCoordStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].sphere_x_coord));
    SphereYCoordStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].sphere_y_coord));
    SphereZCoordStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].sphere_z_coord));
    SphereRadiusStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].sphere_radius));

    if ( currently_displayed_refinement->class_refinement_results[wanted_class].should_refine_ctf == true )
        ShouldRefineCTFStaticText->SetLabel("Yes");
    else
        ShouldRefineCTFStaticText->SetLabel("No");

    DefocusSearchRangeStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].defocus_search_range));
    DefocusSearchStepStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].defocus_search_step));

    if ( currently_displayed_refinement->class_refinement_results[wanted_class].should_auto_mask == true )
        ShouldAutoMaskStaticText->SetLabel("Yes");
    else
        ShouldAutoMaskStaticText->SetLabel("No");

    if ( currently_displayed_refinement->class_refinement_results[wanted_class].should_refine_input_params == true )
        RefineInputParamsStaticText->SetLabel("Yes");
    else
        RefineInputParamsStaticText->SetLabel("No");

    if ( currently_displayed_refinement->class_refinement_results[wanted_class].should_use_supplied_mask == true )
        UseSuppliedMaskStaticText->SetLabel("Yes");
    else
        UseSuppliedMaskStaticText->SetLabel("No");

    MaskAssetIDStaticText->SetLabel(wxString::Format("%li", currently_displayed_refinement->class_refinement_results[wanted_class].mask_asset_id));
    MaskEdgeWidthStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), currently_displayed_refinement->class_refinement_results[wanted_class].mask_edge_width));
    MaskOutsideWeightStaticText->SetLabel(wxString::Format(wxT("%.2f"), currently_displayed_refinement->class_refinement_results[wanted_class].outside_mask_weight));

    if ( currently_displayed_refinement->class_refinement_results[wanted_class].should_low_pass_filter_mask == true )
        ShouldFilterOutsideMaskStaticText->SetLabel("Yes");
    else
        ShouldFilterOutsideMaskStaticText->SetLabel("No");

    MaskFilterResolutionStaticText->SetLabel(wxString::Format(wxT("%.2f"), currently_displayed_refinement->class_refinement_results[wanted_class].filter_resolution));
    ReconstructionIDStaticText->SetLabel(wxString::Format(wxT("%li"), currently_displayed_refinement->class_refinement_results[wanted_class].reconstruction_id));

    // get the reconstruction job details..

    long     refinement_package_asset_id;
    long     refinement_id;
    wxString name;
    float    inner_mask_radius;
    float    outer_mask_radius;
    float    resolution_limit;
    float    score_weight_conversion;
    bool     should_adjust_score;
    bool     should_crop_images;
    bool     should_save_half_maps;
    bool     should_likelihood_blur;
    float    smoothing_factor;
    int      class_number;
    long     volume_asset_id;

    main_frame->current_project.database.GetReconstructionJob(currently_displayed_refinement->class_refinement_results[wanted_class].reconstruction_id, refinement_package_asset_id, refinement_id, name, inner_mask_radius, outer_mask_radius, resolution_limit, score_weight_conversion, should_adjust_score, should_crop_images, should_save_half_maps, should_likelihood_blur, smoothing_factor, class_number, volume_asset_id);

    InnerMaskRadiusStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), inner_mask_radius));
    OuterMaskRadiusStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), outer_mask_radius));
    ResolutionCutOffStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), resolution_limit));
    ScoreWeightConstantStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), score_weight_conversion));

    if ( should_adjust_score == true )
        AdjustScoresStaticText->SetLabel("Yes");
    else
        AdjustScoresStaticText->SetLabel("No");

    if ( should_crop_images == true )
        ShouldCropImagesStaticText->SetLabel("Yes");
    else
        ShouldCropImagesStaticText->SetLabel("No");

    if ( should_likelihood_blur == true )
        ShouldLikelihoodBlurStaticText->SetLabel("Yes");
    else
        ShouldLikelihoodBlurStaticText->SetLabel("No");

    SmoothingFactorStaticText->SetLabel(wxString::Format(wxT("%.2f"), smoothing_factor));

    JobDetailsPanel->Layout( );
}

void MyRefinementResultsPanel::ClearJobInfo( ) {
    RefinementIDStaticText->SetLabel("");
    DateOfRunStaticText->SetLabel("");
    TimeOfRunStaticText->SetLabel("");
    PercentUsedStaticText->SetLabel("");
    ReferenceVolumeIDStaticText->SetLabel("");
    ReferenceRefinementIDStaticText->SetLabel("");
    LowResLimitStaticText->SetLabel("");
    HighResLimitStaticText->SetLabel("");
    MaskRadiusStaticText->SetLabel("");
    SignedCCResLimitStaticText->SetLabel("");
    GlobalResLimitStaticText->SetLabel("");
    GlobalMaskRadiusStaticText->SetLabel("");
    NumberResultsRefinedStaticText->SetLabel("");
    AngularSearchStepStaticText->SetLabel("");
    SearchRangeXStaticText->SetLabel("");
    SearchRangeYStaticText->SetLabel("");
    ClassificationResLimitStaticText->SetLabel("");
    ShouldFocusClassifyStaticText->SetLabel("");
    SphereXCoordStaticText->SetLabel("");
    SphereYCoordStaticText->SetLabel("");
    SphereZCoordStaticText->SetLabel("");
    SphereRadiusStaticText->SetLabel("");
    ShouldRefineCTFStaticText->SetLabel("");
    DefocusSearchRangeStaticText->SetLabel("");
    DefocusSearchStepStaticText->SetLabel("");
    ShouldAutoMaskStaticText->SetLabel("");
    RefineInputParamsStaticText->SetLabel("");
    UseSuppliedMaskStaticText->SetLabel("");
    MaskAssetIDStaticText->SetLabel("");
    MaskEdgeWidthStaticText->SetLabel("");
    MaskOutsideWeightStaticText->SetLabel("");
    ShouldFilterOutsideMaskStaticText->SetLabel("");
    MaskFilterResolutionStaticText->SetLabel("");
    ReconstructionIDStaticText->SetLabel("");
    InnerMaskRadiusStaticText->SetLabel("");
    OuterMaskRadiusStaticText->SetLabel("");
    ResolutionCutOffStaticText->SetLabel("");
    ScoreWeightConstantStaticText->SetLabel("");
    AdjustScoresStaticText->SetLabel("");
    ShouldCropImagesStaticText->SetLabel("");
    ShouldLikelihoodBlurStaticText->SetLabel("");
    SmoothingFactorStaticText->SetLabel("");

    JobDetailsPanel->Layout( );
}
