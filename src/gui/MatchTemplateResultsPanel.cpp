//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMainFrame*       main_frame;
extern MyImageAssetPanel* image_asset_panel;
extern MyMovieAssetPanel* movie_asset_panel;

MatchTemplateResultsPanel::MatchTemplateResultsPanel(wxWindow* parent)
    : MatchTemplateResultsPanelParent(parent) {

    Bind(wxEVT_DATAVIEW_ITEM_VALUE_CHANGED, wxDataViewEventHandler(MatchTemplateResultsPanel::OnValueChanged), this);

    per_row_asset_id       = NULL;
    per_row_array_position = NULL;
    number_of_assets       = 0;

    selected_row     = -1;
    selected_column  = -1;
    doing_panel_fill = false;

    current_fill_command = "SELECT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST";
    is_dirty             = false;
    group_combo_is_dirty = false;

    FillGroupComboBox( );

    Bind(wxEVT_CHAR_HOOK, &MatchTemplateResultsPanel::OnCharHook, this);
}

void MatchTemplateResultsPanel::OnCharHook(wxKeyEvent& event) {
    if ( event.GetUnicodeKey( ) == 'N' ) {
        ResultDataView->NextEye( );
    }
    else if ( event.GetUnicodeKey( ) == 'P' ) {
        ResultDataView->PreviousEye( );
    }
    else
        event.Skip( );
}

void MatchTemplateResultsPanel::FillGroupComboBox( ) {
    GroupComboBox->FillWithImageGroups(false);
}

void MatchTemplateResultsPanel::OnUpdateUI(wxUpdateUIEvent& event) {
    if ( main_frame->current_project.is_open == false ) {
        Enable(false);
    }
    else {
        Enable(true);

        if ( ByFilterButton->GetValue( ) == true ) {
            FilterButton->Enable(true);
        }
        else {
            FilterButton->Enable(false);
        }

        if ( GroupComboBox->GetCount( ) > 0 && ResultDataView->GetItemCount( ) > 0 ) {
            AddToGroupButton->Enable(true);
            DeleteFromGroupButton->Enable(true);
            AddAllToGroupButton->Enable(true);
        }
        else {
            AddToGroupButton->Enable(false);
            DeleteFromGroupButton->Enable(false);
            AddAllToGroupButton->Enable(false);
        }

        if ( is_dirty == true ) {
            is_dirty = false;
            FillBasedOnSelectCommand(current_fill_command);
        }

        if ( group_combo_is_dirty == true ) {
            FillGroupComboBox( );
            group_combo_is_dirty = false;
        }
    }
}

void MatchTemplateResultsPanel::OnAllImagesSelect(wxCommandEvent& event) {
    FillBasedOnSelectCommand("SELECT DISTINCT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST");
}

void MatchTemplateResultsPanel::OnByFilterSelect(wxCommandEvent& event) {
    if ( GetFilter( ) == wxID_CANCEL ) {
        AllImagesButton->SetValue(true);
    }
}

int MatchTemplateResultsPanel::GetFilter( ) {
    MyCTFFilterDialog* filter_dialog = new MyCTFFilterDialog(this);

    // set initial settings..

    // show modal

    if ( filter_dialog->ShowModal( ) == wxID_OK ) {
        //wxPrintf("Command = %s\n", filter_dialog->search_command);
        FillBasedOnSelectCommand(filter_dialog->search_command);

        filter_dialog->Destroy( );
        return wxID_OK;
    }
    else
        return wxID_CANCEL;
}

void MatchTemplateResultsPanel::FillBasedOnSelectCommand(wxString wanted_command) {
    wxVector<wxVariant> data;
    wxVariant           temp_variant;
    long                asset_counter;
    long                job_counter;
    bool                should_continue;
    int                 selected_job_id;
    int                 current_asset;
    int                 array_position;
    int                 current_row;
    int                 start_from_row;

    // append columns..

    doing_panel_fill     = true;
    current_fill_command = wanted_command;

    Freeze( );
    Clear( );

    ResultDataView->AppendTextColumn("ID"); //, wxDATAVIEW_CELL_INERT,1, wxALIGN_LEFT, 0);
    ResultDataView->AppendTextColumn("File"); //, wxDATAVIEW_CELL_INERT,1, wxALIGN_LEFT,wxDATAVIEW_COL_RESIZABLE);

    //
    // find out how many alignment jobs there are :-

    number_of_template_match_ids = main_frame->current_project.database.ReturnNumberOfTemplateMatchingJobs( );
    if ( number_of_template_match_ids == 0 ) {
        Thaw( );
        return;
    }
    // cache the various  alignment_job_ids

    if ( template_match_job_ids != NULL )
        delete[] template_match_job_ids;
    template_match_job_ids = new long[number_of_template_match_ids];

    main_frame->current_project.database.GetUniqueTemplateMatchIDs(template_match_job_ids, number_of_template_match_ids);

    // retrieve their ids

    for ( job_counter = 0; job_counter < number_of_template_match_ids; job_counter++ ) {
        ResultDataView->AppendCheckColumn(wxString::Format("#%li", template_match_job_ids[job_counter]));
    }

    // assign memory to the maximum..

    if ( per_row_asset_id != NULL )
        delete[] per_row_asset_id;
    if ( per_row_array_position != NULL )
        delete[] per_row_array_position;

    per_row_asset_id       = new int[image_asset_panel->ReturnNumberOfAssets( )];
    per_row_array_position = new int[image_asset_panel->ReturnNumberOfAssets( )];

    // execute the select command, to retrieve all the ids..

    number_of_assets = 0;
    should_continue  = main_frame->current_project.database.BeginBatchSelect(wanted_command);

    if ( should_continue == true ) {
        while ( should_continue == true ) {
            should_continue = main_frame->current_project.database.GetFromBatchSelect("i", &current_asset);
            array_position  = image_asset_panel->ReturnArrayPositionFromAssetID(current_asset);

            if ( array_position < 0 || current_asset < 0 ) {
                MyPrintWithDetails("Error: Something wrong finding asset %i, skipping - have you deleted an image?", current_asset);
            }
            else {
                per_row_asset_id[number_of_assets]       = current_asset;
                per_row_array_position[number_of_assets] = array_position;
                number_of_assets++;
            }
        }

        main_frame->current_project.database.EndBatchSelect( );

        // now we know which movies are included, and their order.. draw the dataviewlistctrl

        for ( asset_counter = 0; asset_counter < number_of_assets; asset_counter++ ) {
            data.clear( );
            data.push_back(wxVariant(wxString::Format("%i", per_row_asset_id[asset_counter])));
            data.push_back(wxVariant(image_asset_panel->ReturnAssetShortFilename(per_row_array_position[asset_counter])));

            for ( job_counter = 0; job_counter < number_of_template_match_ids; job_counter++ ) {
                data.push_back(wxVariant(long(-1)));
            }

            ResultDataView->AppendItem(data);
        }

        // all assets should be added.. now go job by job and fill the appropriate columns..

        for ( job_counter = 0; job_counter < number_of_template_match_ids; job_counter++ ) {
            should_continue = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST WHERE TEMPLATE_MATCH_JOB_ID=%li", template_match_job_ids[job_counter]));

            if ( should_continue == false ) {
                MyPrintWithDetails("Error getting template match jobs..");
                DEBUG_ABORT;
            }

            start_from_row = 0;

            while ( 1 == 1 ) {
                should_continue = main_frame->current_project.database.GetFromBatchSelect("i", &current_asset);
                current_row     = ReturnRowFromAssetID(current_asset, start_from_row);

                if ( current_row != -1 ) {
                    ResultDataView->SetValue(wxVariant(UNCHECKED), current_row, 2 + job_counter);
                    start_from_row = current_row;
                }

                if ( should_continue == false )
                    break;
            }

            main_frame->current_project.database.EndBatchSelect( );
        }

        // set the checked ones..

        should_continue = main_frame->current_project.database.BeginBatchSelect("SELECT IMAGE_ASSET_ID, TEMPLATE_MATCH_JOB_ID FROM TEMPLATE_MATCH_LIST WHERE IS_ACTIVE=1;");

        if ( should_continue == false ) {
            MyPrintWithDetails("Error getting selected template matches..");
            DEBUG_ABORT;
        }

        start_from_row = 0;

        while ( 1 == 1 ) {
            should_continue = main_frame->current_project.database.GetFromBatchSelect("ii", &current_asset, &selected_job_id);
            current_row     = ReturnRowFromAssetID(current_asset, start_from_row);

            if ( current_row != -1 ) {
                start_from_row = current_row;

                for ( job_counter = 0; job_counter < number_of_template_match_ids; job_counter++ ) {
                    if ( template_match_job_ids[job_counter] == selected_job_id ) {
                        ResultDataView->SetValue(wxVariant(CHECKED), current_row, 2 + job_counter);
                        break;
                    }
                }
            }

            if ( should_continue == false )
                break;
        }

        main_frame->current_project.database.EndBatchSelect( );

        // select the first row..
        doing_panel_fill = false;

        selected_column = -1;
        selected_row    = -1;

        if ( number_of_assets > 0 ) {
            ResultDataView->ChangeDisplayTo(0, ResultDataView->ReturnCheckedColumn(0));
        }
        ResultDataView->SizeColumns( );
    }
    else {
        main_frame->current_project.database.EndBatchSelect( );
    }

    Thaw( );
}

int MatchTemplateResultsPanel::ReturnRowFromAssetID(int asset_id, int start_location) {
    int counter;

    for ( counter = start_location; counter < number_of_assets; counter++ ) {
        if ( per_row_asset_id[counter] == asset_id )
            return counter;
    }

    // if we got here, we should do the begining..

    for ( counter = 0; counter < start_location; counter++ ) {
        if ( per_row_asset_id[counter] == asset_id )
            return counter;
    }

    return -1;
}

void MatchTemplateResultsPanel::FillResultsPanelAndDetails(int row, int column) {
    bool should_continue;

    int  current_image_id                         = per_row_asset_id[row];
    long current_template_match_estimation_job_id = template_match_job_ids[column - 2];

    long template_match_id = main_frame->current_project.database.ReturnSingleLongFromSelectCommand(wxString::Format("SELECT TEMPLATE_MATCH_ID FROM TEMPLATE_MATCH_LIST WHERE IMAGE_ASSET_ID=%i AND TEMPLATE_MATCH_JOB_ID=%li", current_image_id, current_template_match_estimation_job_id));

    TemplateMatchJobResults current_result;
    current_result = main_frame->current_project.database.GetTemplateMatchingResultByID(template_match_id);

    ResultPanel->SetActiveResult(current_result);

    if ( current_result.job_type == TEMPLATE_MATCH_FULL_SEARCH )
        JobTitleStaticText->SetLabel(current_result.job_name);
    else
        JobTitleStaticText->SetLabel(wxString::Format("%s, Refinement of job #%li", current_result.job_name, current_result.input_job_id));

    JobIDStaticText->SetLabel(wxString::Format("%li", template_match_id));
    wxDateTime wxdatetime_of_run;
    wxdatetime_of_run.SetFromDOS((unsigned long)current_result.datetime_of_run);
    DateOfRunStaticText->SetLabel(wxdatetime_of_run.FormatISODate( ));
    TimeOfRunStaticText->SetLabel(wxdatetime_of_run.FormatISOTime( ));
    RefVolumeIDStaticText->SetLabel(wxString::Format("%li", current_result.ref_volume_asset_id));
    SymmetryStaticText->SetLabel(current_result.symmetry);
    PixelSizeStaticText->SetLabel(wxString::Format(wxT("%.4f Å"), current_result.pixel_size));
    VoltageStaticText->SetLabel(wxString::Format(wxT("%.2f kV"), current_result.voltage));
    CsStaticText->SetLabel(wxString::Format(wxT("%.2f mm"), current_result.spherical_aberration));
    AmplitudeContrastStaticText->SetLabel(wxString::Format(wxT("%.2f"), current_result.amplitude_contrast));
    Defocus1StaticText->SetLabel(wxString::Format(wxT("%.2f Å"), current_result.defocus1));
    Defocus2StaticText->SetLabel(wxString::Format(wxT("%.2f Å"), current_result.defocus2));
    DefocusAngleStaticText->SetLabel(wxString::Format(wxT("%.2f °"), current_result.defocus_angle));
    PhaseShiftStaticText->SetLabel(wxString::Format(wxT("%.2f °"), current_result.phase_shift));
    LowResLimitStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), current_result.low_res_limit));
    HighResLimitStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), current_result.high_res_limit));
    OOPAngluarStepStaticText->SetLabel(wxString::Format(wxT("%.2f °"), current_result.out_of_plane_step));
    IPAngluarStepStaticText->SetLabel(wxString::Format(wxT("%.2f °"), current_result.in_plane_step));
    DefocusRangeStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), current_result.defocus_search_range));
    DefocusStepStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), current_result.defocus_step));
    PixelSizeRangeStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), current_result.pixel_size_search_range));
    PixelSizeStepStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), current_result.pixel_size_step));
    MinPeakRadiusStaticText->SetLabel(wxString::Format(wxT("%.2f px"), current_result.min_peak_radius));
    ShiftThresholdStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), current_result.xy_change_threshold));

    if ( current_result.exclude_above_xy_threshold == true )
        IgnoreShiftedPeaksStaticText->SetLabel("Yes");
    else
        IgnoreShiftedPeaksStaticText->SetLabel("No");

    RightPanel->Layout( );
}

void MatchTemplateResultsPanel::OnValueChanged(wxDataViewEvent& event) {

    if ( doing_panel_fill == false ) {
        wxDataViewItem current_item = event.GetItem( );
        int            row          = ResultDataView->ItemToRow(current_item);
        int            column       = event.GetColumn( );
        long           value;

        int old_selected_row    = -1;
        int old_selected_column = -1;

        wxVariant temp_variant;
        ResultDataView->GetValue(temp_variant, row, column);
        value = temp_variant.GetLong( );

        if ( (value == CHECKED_WITH_EYE || value == UNCHECKED_WITH_EYE) && (selected_row != row || selected_column != column) ) {
            old_selected_row    = selected_row;
            old_selected_column = selected_column;

            selected_row    = row;
            selected_column = column;

            FillResultsPanelAndDetails(row, column);

            //wxPrintf("drawing curve\n");
        }
        else // This is dodgy, and relies on the fact that a box will be deselected, before a new box is selected...
        {
            if ( (value == CHECKED && (selected_row != row || selected_column != column)) || (value == CHECKED_WITH_EYE) ) {
                // we need to update the database for the resulting image asset

                int image_asset_position = per_row_array_position[row];
                int image_asset_id       = image_asset_panel->ReturnAssetID(image_asset_position);
                int estimation_job_id    = template_match_job_ids[column - 2];

                MyDebugAssertTrue(image_asset_id >= 0, "Something went wrong finding an image asset");
                main_frame->current_project.database.SetActiveTemplateMatchJobForGivenImageAssetID(image_asset_id, estimation_job_id);

                // we need to get the details of the selected movie alignment, and update the image asset.

                //int estimation_id = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(wxString::Format("SELECT TEMPLATE_MATCH_ID FROM TEMPLATE_MATCH_LIST WHERE IMAGE_ASSET_ID=%i AND TEMPLATE_MATCH_JOB_ID=%i",image_asset_id, estimation_job_id));
                //bool should_continue;

                //should_continue = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT CTF_ESTIMATION_ID FROM ESTIMATED_CTF_PARAMETERS WHERE IMAGE_ASSET_ID=%i AND CTF_ESTIMATION_JOB_ID=%i",image_asset_id, estimation_job_id));

                //if (should_continue == false)
                //{
                //	MyPrintWithDetails("Error getting information about alignment!")
                //	DEBUG_ABORT;
                //}

                //main_frame->current_project.database.GetFromBatchSelect("it", &alignment_id, &output_file);
                //main_frame->current_project.database.EndBatchSelect();

                //main_frame->current_project.database.BeginImageAssetInsert();
                //main_frame->current_project.database.AddNextImageAsset(image_asset_id,  image_asset_panel->ReturnAssetPointer(image_asset_position)->asset_name, image_asset_panel->ReturnAssetPointer(image_asset_position)->filename.GetFullPath(),  image_asset_panel->ReturnAssetPointer(image_asset_position)->position_in_stack, image_asset_panel->ReturnAssetPointer(image_asset_position)->parent_id,  image_asset_panel->ReturnAssetPointer(image_asset_position)->alignment_id, estimation_id, image_asset_panel->ReturnAssetPointer(image_asset_position)->x_size, image_asset_panel->ReturnAssetPointer(image_asset_position)->y_size, image_asset_panel->ReturnAssetPointer(image_asset_position)->microscope_voltage, image_asset_panel->ReturnAssetPointer(image_asset_position)->pixel_size, image_asset_panel->ReturnAssetPointer(image_asset_position)->spherical_aberration, image_asset_panel->ReturnAssetPointer(image_asset_position)->protein_is_white);
                //main_frame->current_project.database.EndImageAssetInsert();
                //image_asset_panel->ReturnAssetPointer(image_asset_position)->ctf_estimation_id = estimation_id;
                //image_asset_panel->is_dirty = true;
            }
        }
    }
}

void MatchTemplateResultsPanel::OnNextButtonClick(wxCommandEvent& event) {
    ResultDataView->NextEye( );
}

void MatchTemplateResultsPanel::OnPreviousButtonClick(wxCommandEvent& event) {
    ResultDataView->PreviousEye( );
}

void MatchTemplateResultsPanel::Clear( ) {
    selected_row    = -1;
    selected_column = -1;

    ResultDataView->Clear( );
    ResultPanel->Clear( );
    ResultPanel->ImageDisplayPanel->Clear( );

    JobDetailsToggleButton->SetValue(false);
    JobDetailsPanel->Show(false);

    JobIDStaticText->SetLabel("");
    RefVolumeIDStaticText->SetLabel("");
    DateOfRunStaticText->SetLabel("");
    TimeOfRunStaticText->SetLabel("");
    SymmetryStaticText->SetLabel("");
    PixelSizeStaticText->SetLabel("");
    VoltageStaticText->SetLabel("");
    CsStaticText->SetLabel("");
    AmplitudeContrastStaticText->SetLabel("");
    Defocus1StaticText->SetLabel("");
    Defocus2StaticText->SetLabel("");
    DefocusAngleStaticText->SetLabel("");
    PhaseShiftStaticText->SetLabel("");
    LowResLimitStaticText->SetLabel("");
    HighResLimitStaticText->SetLabel("");
    OOPAngluarStepStaticText->SetLabel("");
    IPAngluarStepStaticText->SetLabel("");
    DefocusRangeStaticText->SetLabel("");
    DefocusStepStaticText->SetLabel("");
    PixelSizeRangeStaticText->SetLabel("");
    PixelSizeStepStaticText->SetLabel("");
    Layout( );
    RightPanel->Layout( );
}

void MatchTemplateResultsPanel::OnJobDetailsToggle(wxCommandEvent& event) {
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

void MatchTemplateResultsPanel::OnAddToGroupClick(wxCommandEvent& event) {
    image_asset_panel->AddArrayItemToGroup(GroupComboBox->GetSelection( ) + 1, per_row_array_position[selected_row]);
}

void MatchTemplateResultsPanel::OnRemoveFromGroupClick(wxCommandEvent& event) {
    image_asset_panel->DeleteArrayItemFromGroup(GroupComboBox->GetSelection( ) + 1, per_row_array_position[selected_row]);
}

void MatchTemplateResultsPanel::OnAddAllToGroupClick(wxCommandEvent& event) {
    wxArrayLong items_to_add;

    for ( long counter = 0; counter < ResultDataView->GetItemCount( ); counter++ ) {
        items_to_add.Add(per_row_array_position[counter]);
    }
    OneSecondProgressDialog* progress_bar = new OneSecondProgressDialog("Add all to group", "Adding all to group", ResultDataView->GetItemCount( ), this, wxPD_APP_MODAL);
    image_asset_panel->AddArrayofArrayItemsToGroup(GroupComboBox->GetSelection( ) + 1, &items_to_add, progress_bar);
    progress_bar->Destroy( );
}

void MatchTemplateResultsPanel::OnDefineFilterClick(wxCommandEvent& event) {
    GetFilter( );
}
