#include "../core/gui_core_headers.h"

extern MyVolumeAssetPanel* volume_asset_panel;
extern MyRefine2DPanel*    classification_panel;
extern AbInitio3DPanel*    ab_initio_3d_panel;
extern AutoRefine3DPanel*  auto_refine_3d_panel;
extern MyRefine3DPanel*    refine_3d_panel;

MyRefinementPackageAssetPanel::MyRefinementPackageAssetPanel(wxWindow* parent)
    : RefinementPackageAssetPanel(parent) {
    current_asset_number        = 0;
    selected_refinement_package = -1;
    is_dirty                    = false;

    RefinementPackageListCtrl->InsertColumn(0, "Packages", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);
}

void MyRefinementPackageAssetPanel::OnCreateClick(wxCommandEvent& event) {
    MyNewRefinementPackageWizard* my_wizard = new MyNewRefinementPackageWizard(this);
    my_wizard->RunWizard(my_wizard->template_page);
    my_wizard->Destroy( );
}

void MyRefinementPackageAssetPanel::OnDisplayStackButton(wxCommandEvent& event) {
    if ( selected_refinement_package >= 0 ) {
        wxString execution_command = wxStandardPaths::Get( ).GetExecutablePath( );
        execution_command          = execution_command.BeforeLast('/');
        execution_command += "/cisTEM_display ";

        execution_command += all_refinement_packages[selected_refinement_package].stack_filename;
        execution_command += "&";
        //wxPrintf("Launching %s\n", execution_command);
        system(execution_command.ToUTF8( ).data( ));
    }
}

void MyRefinementPackageAssetPanel::OnRenameClick(wxCommandEvent& event) {
    if ( selected_refinement_package >= 0 ) {
        RefinementPackageListCtrl->EditLabel(selected_refinement_package);
    }
}

void MyRefinementPackageAssetPanel::OnCombineClick(wxCommandEvent& event) {
    CombineRefinementPackagesWizard* my_wizard = new CombineRefinementPackagesWizard(this);
    my_wizard->RunWizard(my_wizard->package_selection_page);
    my_wizard->Destroy( );
}

void MyRefinementPackageAssetPanel::OnImportClick(wxCommandEvent& event) {
    ImportRefinementPackageWizard* my_wizard = new ImportRefinementPackageWizard(this);
    my_wizard->GetPageAreaSizer( )->Add(my_wizard->m_pages.Item(0));
    my_wizard->RunWizard(my_wizard->m_pages.Item(0));
    my_wizard->Destroy( );
}

void MyRefinementPackageAssetPanel::OnExportClick(wxCommandEvent& event) {
    ExportRefinementPackageWizard* my_wizard = new ExportRefinementPackageWizard(this);
    my_wizard->GetPageAreaSizer( )->Add(my_wizard->m_pages.Item(0));
    my_wizard->RunWizard(my_wizard->m_pages.Item(0));
    my_wizard->Destroy( );
}

void MyRefinementPackageAssetPanel::RemoveVolumeFromAllRefinementPackages(long wanted_volume_asset_id) {
    int refinement_package_counter;
    int reference_id_counter;

    for ( refinement_package_counter = 0; refinement_package_counter < all_refinement_packages.GetCount( ); refinement_package_counter++ ) {
        for ( reference_id_counter = all_refinement_packages[refinement_package_counter].references_for_next_refinement.GetCount( ) - 1; reference_id_counter >= 0; reference_id_counter-- ) {
            if ( all_refinement_packages[refinement_package_counter].references_for_next_refinement[reference_id_counter] == wanted_volume_asset_id )
                all_refinement_packages[refinement_package_counter].references_for_next_refinement[reference_id_counter] = -1;
        }
    }
}

void MyRefinementPackageAssetPanel::RemoveImageFromAllRefinementPackages(long wanted_image_asset_id) {
    int refinement_package_counter;
    int contained_particle_counter;

    for ( refinement_package_counter = 0; refinement_package_counter < all_refinement_packages.GetCount( ); refinement_package_counter++ ) {
        for ( contained_particle_counter = all_refinement_packages[refinement_package_counter].contained_particles.GetCount( ) - 1; contained_particle_counter >= 0; contained_particle_counter-- ) {
            if ( all_refinement_packages[refinement_package_counter].contained_particles[contained_particle_counter].parent_image_id == wanted_image_asset_id )
                all_refinement_packages[refinement_package_counter].contained_particles[contained_particle_counter].parent_image_id = -1;
        }
    }
}

void MyRefinementPackageAssetPanel::OnDeleteClick(wxCommandEvent& event) {
    if ( selected_refinement_package >= 0 ) {
        // check if there is a running job which uses refinement packages, if there is refuse to delete the refinement package.

        if ( classification_panel->running_job == true || ab_initio_3d_panel->running_job == true || auto_refine_3d_panel->running_job == true || refine_3d_panel->running_job == true ) {

            wxMessageDialog error_dialog(this, "Sorry! - You cannot delete refinement packages when any of the following are running (jobs are not marked finished until you press the finish button) :-\n\n2D Classification\nAb-Inito 3D\nAuto Refine\nManual Refine\n\nThis is to stop the database getting messed up. Sorry again!", "There are jobs running", wxICON_ERROR);
            error_dialog.ShowModal( );
            return;
        }

        wxMessageDialog* check_dialog = new wxMessageDialog(this, "This will remove the refinement package from your entire project, including all classifications / refinements that have been run on it.\n\nAlso note that this is just a database deletion, the particle stacks are not deleted.\n\nAre you sure you want to continue?", "Are you sure?", wxYES_NO);

        if ( check_dialog->ShowModal( ) == wxID_YES ) {
            long counter;
            int  class_counter;

            main_frame->current_project.database.Begin( );

            main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=%li", all_refinement_packages.Item(selected_refinement_package).asset_id));
            main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%li", all_refinement_packages.Item(selected_refinement_package).asset_id));
            main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li", all_refinement_packages.Item(selected_refinement_package).asset_id));
            main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li", all_refinement_packages.Item(selected_refinement_package).asset_id));
            main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_%li", all_refinement_packages.Item(selected_refinement_package).asset_id));

            // Delete Refinements

            for ( counter = all_refinement_short_infos.GetCount( ) - 1; counter >= 0; counter-- ) {
                if ( all_refinement_short_infos.Item(counter).refinement_package_asset_id == all_refinement_packages.Item(selected_refinement_package).asset_id ) {
                    main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM REFINEMENT_LIST WHERE REFINEMENT_ID=%li", all_refinement_short_infos.Item(counter).refinement_id));

                    for ( class_counter = 1; class_counter <= all_refinement_short_infos.Item(counter).number_of_classes; class_counter++ ) {
                        main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_DETAILS_%li", all_refinement_short_infos.Item(counter).refinement_id));
                        main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_RESULT_%li_%i", all_refinement_short_infos.Item(counter).refinement_id, class_counter));
                        main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_RESOLUTION_STATISTICS_%li_%i", all_refinement_short_infos.Item(counter).refinement_id, class_counter));
                    }
                    all_refinement_short_infos.RemoveAt(counter);
                }
            }

            for ( counter = all_classification_short_infos.GetCount( ) - 1; counter >= 0; counter-- ) {
                if ( all_classification_short_infos.Item(counter).refinement_package_asset_id == all_refinement_packages.Item(selected_refinement_package).asset_id ) {
                    main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=%li", all_classification_short_infos.Item(counter).classification_id));
                    main_frame->current_project.database.DeleteTable(wxString::Format("CLASSIFICATION_RESULT_%li", all_classification_short_infos.Item(counter).classification_id));

                    all_classification_short_infos.RemoveAt(counter);
                }
            }

            for ( counter = all_classification_selections.GetCount( ) - 1; counter >= 0; counter-- ) {
                if ( all_classification_selections.Item(counter).refinement_package_asset_id == all_refinement_packages.Item(selected_refinement_package).asset_id ) {
                    main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM CLASSIFICATION_SELECTION_LIST WHERE SELECTION_ID=%li", all_classification_selections.Item(counter).selection_id));
                    main_frame->current_project.database.DeleteTable(wxString::Format("CLASSIFICATION_SELECTION_%li", all_classification_selections.Item(counter).selection_id));

                    all_classification_selections.RemoveAt(counter);
                }
            }

            all_refinement_packages.RemoveAt(selected_refinement_package);

            main_frame->current_project.database.Commit( );

            if ( all_refinement_packages.GetCount( ) > 0 )
                selected_refinement_package = 0;
            else
                selected_refinement_package = -1;
            main_frame->DirtyRefinementPackages( );
        }
    }
}

void MyRefinementPackageAssetPanel::AddAsset(RefinementPackage* refinement_package) {
    // add into memory..

    current_asset_number++;
    refinement_package->asset_id = current_asset_number;

    all_refinement_packages.Add(refinement_package);

    // now add it to the database..

    main_frame->current_project.database.AddRefinementPackageAsset(refinement_package);
    main_frame->DirtyRefinementPackages( );
}

void MyRefinementPackageAssetPanel::FillRefinementPackages( ) {
    Freeze( );

    if ( all_refinement_packages.GetCount( ) > 0 ) {

        RefinementPackageListCtrl->SetItemCount(all_refinement_packages.GetCount( ));
        RefinementPackageListCtrl->SetColumnWidth(0, RefinementPackageListCtrl->ReturnGuessAtColumnTextWidth( ));

        if ( selected_refinement_package >= 0 && selected_refinement_package < all_refinement_packages.GetCount( ) ) {
            RefinementPackageListCtrl->SetItemState(selected_refinement_package, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
        }
        else {
            selected_refinement_package = 0;
            RefinementPackageListCtrl->SetItemState(selected_refinement_package, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
        }

        if ( all_refinement_packages.GetCount( ) > 0 )
            RefinementPackageListCtrl->RefreshItems(0, all_refinement_packages.GetCount( ) - 1);

        StackFileNameText->SetLabel(all_refinement_packages.Item(selected_refinement_package).stack_filename);
        StackBoxSizeText->SetLabel(wxString::Format("%i px", all_refinement_packages.Item(selected_refinement_package).stack_box_size));
        NumberofClassesText->SetLabel(wxString::Format("%i", all_refinement_packages.Item(selected_refinement_package).number_of_classes));
        NumberofRefinementsText->SetLabel(wxString::Format("%i", all_refinement_packages.Item(selected_refinement_package).number_of_run_refinments));
        LastRefinementIDText->SetLabel(wxString::Format("%li", all_refinement_packages.Item(selected_refinement_package).last_refinment_id));
        SymmetryText->SetLabel(all_refinement_packages.Item(selected_refinement_package).symmetry);
        MolecularWeightText->SetLabel(wxString::Format(wxT("%.0f kDa"), all_refinement_packages.Item(selected_refinement_package).estimated_particle_weight_in_kda));
        LargestDimensionText->SetLabel(wxString::Format(wxT("%.0f Å"), all_refinement_packages.Item(selected_refinement_package).estimated_particle_size_in_angstroms));

        // setup the contents panel..

        ContainedParticlesListCtrl->ClearAll( );
        ContainedParticlesListCtrl->InsertColumn(0, "Orig. Pos. ID", wxLIST_FORMAT_LEFT);
        ContainedParticlesListCtrl->InsertColumn(1, "Image ID", wxLIST_FORMAT_LEFT);
        ContainedParticlesListCtrl->InsertColumn(2, "X Pos.", wxLIST_FORMAT_LEFT);
        ContainedParticlesListCtrl->InsertColumn(3, "Y Pos.", wxLIST_FORMAT_LEFT);
        ContainedParticlesListCtrl->InsertColumn(4, "Pixel Size", wxLIST_FORMAT_LEFT);
        ContainedParticlesListCtrl->InsertColumn(5, "Cs", wxLIST_FORMAT_LEFT);
        ContainedParticlesListCtrl->InsertColumn(6, "Voltage", wxLIST_FORMAT_LEFT);
        ContainedParticlesListCtrl->InsertColumn(7, "Amp. Contrast", wxLIST_FORMAT_LEFT);
        ContainedParticlesListCtrl->InsertColumn(8, "Init. Defocus 1", wxLIST_FORMAT_LEFT);
        ContainedParticlesListCtrl->InsertColumn(9, "Init. Defocus 2", wxLIST_FORMAT_LEFT);
        ContainedParticlesListCtrl->InsertColumn(10, "Init. Defocus Angle", wxLIST_FORMAT_LEFT);
        ContainedParticlesListCtrl->InsertColumn(11, "Init. Phase Shift", wxLIST_FORMAT_LEFT);
        //	ContainedParticlesListCtrl->InsertColumn(10, wxT("Init. Psi"), wxLIST_FORMAT_LEFT);
        //	ContainedParticlesListCtrl->InsertColumn(11, wxT("Init. Theta"), wxLIST_FORMAT_LEFT);
        //	ContainedParticlesListCtrl->InsertColumn(12, wxT("Init. Phi"), wxLIST_FORMAT_LEFT);
        //	ContainedParticlesListCtrl->InsertColumn(13, "Init. X-shift", wxLIST_FORMAT_LEFT);
        //	ContainedParticlesListCtrl->InsertColumn(14, "Init. Y-Shift", wxLIST_FORMAT_LEFT);

        ContainedParticlesListCtrl->SetItemCount(all_refinement_packages.Item(selected_refinement_package).contained_particles.GetCount( ));

        if ( all_refinement_packages.Item(selected_refinement_package).contained_particles.GetCount( ) > 0 ) {
            ContainedParticlesListCtrl->RefreshItems(0, all_refinement_packages.Item(selected_refinement_package).contained_particles.GetCount( ) - 1);

            ContainedParticlesListCtrl->SetColumnWidth(0, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(0));
            ContainedParticlesListCtrl->SetColumnWidth(1, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(1));
            ContainedParticlesListCtrl->SetColumnWidth(2, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(2));
            ContainedParticlesListCtrl->SetColumnWidth(3, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(3));
            ContainedParticlesListCtrl->SetColumnWidth(4, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(4));
            ContainedParticlesListCtrl->SetColumnWidth(5, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(5));
            ContainedParticlesListCtrl->SetColumnWidth(6, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(6));
            ContainedParticlesListCtrl->SetColumnWidth(7, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(7));
            ContainedParticlesListCtrl->SetColumnWidth(8, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(8));
            ContainedParticlesListCtrl->SetColumnWidth(9, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(9));
            ContainedParticlesListCtrl->SetColumnWidth(10, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(10));
            ContainedParticlesListCtrl->SetColumnWidth(11, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(11));
        }

        ContainedParticlesStaticText->SetLabel(wxString::Format("Contained Particles (%li) : ", all_refinement_packages.Item(selected_refinement_package).contained_particles.GetCount( )));

        // 3D references..

        ReDrawActiveReferences( );
    }
    else {
        RefinementPackageListCtrl->SetItemCount(0);
        ContainedParticlesListCtrl->SetItemCount(0);
        Active3DReferencesListCtrl->SetItemCount(0);

        StackFileNameText->SetLabel("");
        StackBoxSizeText->SetLabel("");
        NumberofClassesText->SetLabel("");
        NumberofRefinementsText->SetLabel("");
        LastRefinementIDText->SetLabel("");

        ContainedParticlesStaticText->SetLabel("Contained Particles : ");
    }

    //	RefinementPackageListCtrl->Thaw();
    Thaw( );
}

void MyRefinementPackageAssetPanel::ReDrawActiveReferences( ) {

    Active3DReferencesListCtrl->ClearAll( );
    Active3DReferencesListCtrl->InsertColumn(0, "Class No.", wxLIST_FORMAT_LEFT);
    Active3DReferencesListCtrl->InsertColumn(1, "Reference Volume", wxLIST_FORMAT_LEFT);

    Active3DReferencesListCtrl->SetItemCount(all_refinement_packages.Item(selected_refinement_package).references_for_next_refinement.GetCount( ));

    if ( all_refinement_packages.Item(selected_refinement_package).references_for_next_refinement.GetCount( ) > 0 ) {
        Active3DReferencesListCtrl->RefreshItems(0, all_refinement_packages.Item(selected_refinement_package).references_for_next_refinement.GetCount( ) - 1);

        Active3DReferencesListCtrl->SetColumnWidth(0, Active3DReferencesListCtrl->ReturnGuessAtColumnTextWidth(0));
        Active3DReferencesListCtrl->SetColumnWidth(1, Active3DReferencesListCtrl->ReturnGuessAtColumnTextWidth(1));
    }
}

void MyRefinementPackageAssetPanel::OnUpdateUI(wxUpdateUIEvent& event) {
    if ( main_frame->current_project.is_open == true ) {
        Enable(true);

        if ( is_dirty == true ) {
            FillRefinementPackages( );
            is_dirty = false;
        }

        if ( selected_refinement_package >= 0 ) {
            RenameButton->Enable(true);
            DeleteButton->Enable(true);
            ExportButton->Enable(true);
            DisplayStackButton->Enable(true);
        }
        else {
            RenameButton->Enable(false);
            DeleteButton->Enable(false);
            ExportButton->Enable(false);
            DisplayStackButton->Enable(false);
        }
    }
    else
        Enable(false);
}

void MyRefinementPackageAssetPanel::ImportAllFromDatabase( ) {
    int                counter;
    RefinementPackage* temp_package;

    all_refinement_packages.Clear( );

    // Now the groups..

    main_frame->current_project.database.BeginAllRefinementPackagesSelect( );

    while ( main_frame->current_project.database.last_return_code == SQLITE_ROW ) {
        temp_package = main_frame->current_project.database.GetNextRefinementPackage( );
        if ( temp_package->asset_id > current_asset_number )
            current_asset_number = temp_package->asset_id;

        all_refinement_packages.Add(temp_package);
    }

    main_frame->current_project.database.EndAllRefinementPackagesSelect( );

    ImportAllRefinementInfosFromDatabase( );
    ImportAllClassificationInfosFromDatabase( );
    ImportAllClassificationSelectionsFromDatabase( );

    main_frame->DirtyRefinementPackages( );
}

void MyRefinementPackageAssetPanel::MouseVeto(wxMouseEvent& event) {
    //Do nothing
}

void MyRefinementPackageAssetPanel::MouseCheckPackagesVeto(wxMouseEvent& event) {
    VetoInvalidMouse(RefinementPackageListCtrl, event);
}

void MyRefinementPackageAssetPanel::MouseCheckParticlesVeto(wxMouseEvent& event) {
    VetoInvalidMouse(ContainedParticlesListCtrl, event);
}

void MyRefinementPackageAssetPanel::VetoInvalidMouse(wxListCtrl* wanted_list, wxMouseEvent& event) {
    // Don't allow clicking on anything other than item, to stop the selection bar changing

    int flags;

    if ( wanted_list->HitTest(event.GetPosition( ), flags) != wxNOT_FOUND ) {
        should_veto_motion = false;
        event.Skip( );
    }
    else
        should_veto_motion = true;
}

void MyRefinementPackageAssetPanel::OnMotion(wxMouseEvent& event) {
    if ( should_veto_motion == false )
        event.Skip( );
}

void MyRefinementPackageAssetPanel::OnPackageFocusChange(wxListEvent& event) {
    if ( event.GetIndex( ) != selected_refinement_package ) {
        selected_refinement_package = event.GetIndex( );
        is_dirty                    = true;

        if ( selected_refinement_package >= 0 ) {
            ContainedParticlesStaticText->SetLabel(wxString::Format("Contained Particles (%li) : ", all_refinement_packages.Item(selected_refinement_package).contained_particles.GetCount( )));
        }
        else
            ContainedParticlesStaticText->SetLabel("Contained Particles : ");
    }

    //wxPrintf("Selected refinement package = %li\n", selected_refinement_package);

    event.Skip( );
}

void MyRefinementPackageAssetPanel::OnPackageActivated(wxListEvent& event) {
    RefinementPackageListCtrl->EditLabel(event.GetIndex( ));
}

void MyRefinementPackageAssetPanel::OnVolumeListItemActivated(wxListEvent& event) {
    MyVolumeChooserDialog* dialog = new MyVolumeChooserDialog(this);

    dialog->ComboBox->SetSelection(volume_asset_panel->ReturnArrayPositionFromAssetID(all_refinement_packages.Item(selected_refinement_package).references_for_next_refinement.Item(event.GetIndex( ))) + 1);
    dialog->Fit( );
    if ( dialog->ShowModal( ) == wxID_OK ) {
        if ( dialog->selected_volume_id != all_refinement_packages.Item(selected_refinement_package).references_for_next_refinement.Item(event.GetIndex( )) ) {
            all_refinement_packages.Item(selected_refinement_package).references_for_next_refinement.Item(event.GetIndex( )) = dialog->selected_volume_id;
            // Change in database..
            main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li SET VOLUME_ASSET_ID=%li WHERE CLASS_NUMBER=%li;", all_refinement_packages.Item(selected_refinement_package).asset_id, dialog->selected_volume_id, event.GetIndex( ) + 1));

            ReDrawActiveReferences( );
        }
    }
    dialog->Destroy( );
}

void MyRefinementPackageAssetPanel::OnBeginEdit(wxListEvent& event) {
    event.Skip( );
}

void MyRefinementPackageAssetPanel::OnEndEdit(wxListEvent& event) {

    if ( event.GetLabel( ) == wxEmptyString ) {
        RefinementPackageListCtrl->SetItemText(event.GetIndex( ), all_refinement_packages.Item(event.GetIndex( )).name);
        event.Veto( );
    }
    else {
        if ( all_refinement_packages.Item(event.GetIndex( )).name != event.GetLabel( ) ) {
            all_refinement_packages.Item(event.GetIndex( )).name = event.GetLabel( );

            // do in database also..

            wxString safe_name = all_refinement_packages.Item(event.GetIndex( )).name;

            //escape apostrophes
            safe_name.Replace("'", "''");

            wxString sql_command = wxString::Format("UPDATE REFINEMENT_PACKAGE_ASSETS SET NAME='%s' WHERE REFINEMENT_PACKAGE_ASSET_ID=%li", safe_name, all_refinement_packages.Item(event.GetIndex( )).asset_id);
            main_frame->current_project.database.ExecuteSQL(sql_command.ToUTF8( ).data( ));

            main_frame->DirtyRefinementPackages( );
            event.Skip( );
        }
        else
            event.Veto( );
    }
}

void MyRefinementPackageAssetPanel::Reset( ) {
    all_refinement_packages.Clear( );
    main_frame->DirtyRefinementPackages( );
    ContainedParticlesStaticText->SetLabel("Contained Particles : ");
}

long MyRefinementPackageAssetPanel::ReturnArrayPositionFromAssetID(long wanted_asset_id) {
    for ( long counter = 0; counter < all_refinement_packages.GetCount( ); counter++ ) {
        if ( all_refinement_packages[counter].asset_id == wanted_asset_id )
            return counter;
    }

    return -1;
}

void MyRefinementPackageAssetPanel::ImportAllRefinementInfosFromDatabase( ) {

    bool                more_data;
    long                current_id;
    int                 counter = 0;
    ShortRefinementInfo temp_info;
    all_refinement_short_infos.Clear( );
    double        average_occupancy;
    double        estimated_resolution;
    long          reconstructed_volume_asset_id;
    int           class_counter;
    sqlite3_stmt* list_statement = NULL;

    more_data = main_frame->current_project.database.BeginBatchSelect("SELECT REFINEMENT_ID, REFINEMENT_PACKAGE_ASSET_ID, NAME, NUMBER_OF_PARTICLES, NUMBER_OF_CLASSES FROM REFINEMENT_LIST");

    while ( more_data == true ) {
        more_data = main_frame->current_project.database.GetFromBatchSelect("lltli", &temp_info.refinement_id, &temp_info.refinement_package_asset_id, &temp_info.name, &temp_info.number_of_particles, &temp_info.number_of_classes);
        main_frame->current_project.database.Prepare(wxString::Format("SELECT AVERAGE_OCCUPANCY, ESTIMATED_RESOLUTION, RECONSTRUCTED_VOLUME_ASSET_ID FROM REFINEMENT_DETAILS_%li", temp_info.refinement_id), &list_statement);

        temp_info.average_occupancy.Clear( );
        temp_info.estimated_resolution.Clear( );
        temp_info.reconstructed_volume_asset_ids.Clear( );

        for ( class_counter = 0; class_counter < temp_info.number_of_classes; class_counter++ ) {
            main_frame->current_project.database.Step(list_statement);
            average_occupancy             = sqlite3_column_double(list_statement, 0);
            estimated_resolution          = sqlite3_column_double(list_statement, 1);
            reconstructed_volume_asset_id = sqlite3_column_int64(list_statement, 2);

            //wxPrintf("Average Occ. = %f, Res. = %f\n", average_occupancy, estimated_resolution);

            temp_info.average_occupancy.Add(average_occupancy);
            temp_info.estimated_resolution.Add(estimated_resolution);
            temp_info.reconstructed_volume_asset_ids.Add(reconstructed_volume_asset_id);
        }

        main_frame->current_project.database.Finalize(list_statement);
        all_refinement_short_infos.Add(temp_info);
    }

    main_frame->current_project.database.EndBatchSelect( );
}

void MyRefinementPackageAssetPanel::ImportAllClassificationInfosFromDatabase( ) {

    bool more_data;

    ShortClassificationInfo temp_info;
    all_classification_short_infos.Clear( );

    more_data = main_frame->current_project.database.BeginBatchSelect("SELECT CLASSIFICATION_ID, REFINEMENT_PACKAGE_ASSET_ID, NAME, CLASS_AVERAGE_FILE, NUMBER_OF_PARTICLES, NUMBER_OF_CLASSES, HIGH_RESOLUTION_LIMIT FROM CLASSIFICATION_LIST");

    while ( more_data == true ) {
        more_data = main_frame->current_project.database.GetFromBatchSelect("llttlis", &temp_info.classification_id, &temp_info.refinement_package_asset_id, &temp_info.name, &temp_info.class_average_file, &temp_info.number_of_particles, &temp_info.number_of_classes, &temp_info.high_resolution_limit);
        all_classification_short_infos.Add(temp_info);
    }

    main_frame->current_project.database.EndBatchSelect( );
}

void MyRefinementPackageAssetPanel::ImportAllClassificationSelectionsFromDatabase( ) {

    bool more_data;
    long temp_long;

    ClassificationSelection temp_selection;
    all_classification_selections.Clear( );

    more_data = main_frame->current_project.database.BeginBatchSelect("SELECT * FROM CLASSIFICATION_SELECTION_LIST");

    while ( more_data == true ) {
        more_data = main_frame->current_project.database.GetFromBatchSelect("ltlllii", &temp_selection.selection_id, &temp_selection.name, &temp_long, &temp_selection.refinement_package_asset_id, &temp_selection.classification_id, &temp_selection.number_of_classes, &temp_selection.number_of_selections);
        temp_selection.creation_date.SetFromDOS((unsigned long)temp_long);
        all_classification_selections.Add(temp_selection);
    }

    main_frame->current_project.database.EndBatchSelect( );

    for ( int counter = 0; counter < all_classification_selections.GetCount( ); counter++ ) {
        more_data = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT CLASS_AVERAGE_NUMBER FROM CLASSIFICATION_SELECTION_%li", all_classification_selections.Item(counter).selection_id));

        while ( more_data == true ) {
            more_data = main_frame->current_project.database.GetFromBatchSelect("l", &temp_long);
            all_classification_selections.Item(counter).selections.Add(temp_long);
        }

        main_frame->current_project.database.EndBatchSelect( );
    }
}

ShortRefinementInfo* MyRefinementPackageAssetPanel::ReturnPointerToShortRefinementInfoByRefinementID(long wanted_id) {
    long counter;

    //wxPrintf("looking for %li\n", wanted_id);
    for ( counter = 0; counter < all_refinement_short_infos.GetCount( ); counter++ ) {
        if ( all_refinement_short_infos[counter].refinement_id == wanted_id ) {
            //wxPrintf("found refinement\n");
            return &all_refinement_short_infos.Item(counter);
        }
    }

    //wxPrintf("returning NULL\n");
    MyDebugAssertFalse(1 == 1, "Returning NULL here, wanted_id is %li", wanted_id);
    return NULL;
}

ShortClassificationInfo* MyRefinementPackageAssetPanel::ReturnPointerToShortClassificationInfoByClassificationID(long wanted_id) {
    long counter;

    //wxPrintf("looking for %li\n", wanted_id);
    for ( counter = 0; counter < all_classification_short_infos.GetCount( ); counter++ ) {
        if ( all_classification_short_infos[counter].classification_id == wanted_id ) {
            //wxPrintf("found refinement\n");
            return &all_classification_short_infos.Item(counter);
        }
    }

    //wxPrintf("returning NULL\n");
    MyDebugAssertFalse(1 == 1, "Returning NULL here, wanted_id is %li", wanted_id);
    return NULL;
}

/*
Refinement* MyRefinementPackageAssetPanel::ReturnPointerToRefinementByRefinementID(long wanted_id)
{
	long counter;

	//wxPrintf("looking for %li\n", wanted_id);
	for (counter = 0; counter < all_refinements.GetCount(); counter++)
	{
		if (all_refinements[counter].refinement_id == wanted_id)
		{
		//wxPrintf("found refinement\n");
			return &all_refinements.Item(counter);
		}
	}

	wxPrintf("returning NULL\n");
	return NULL;

}*/
