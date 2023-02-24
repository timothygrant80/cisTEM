#include "../core/gui_core_headers.h"

//extern MyVolumeAssetPanel* volume_asset_panel;
//extern MyRefine2DPanel*    classification_panel;
//extern AbInitio3DPanel*    ab_initio_3d_panel;
//extern AutoRefine3DPanel*  auto_refine_3d_panel;
//extern MyRefine3DPanel*    refine_3d_panel;

TemplateMatchesPackageAssetPanel::TemplateMatchesPackageAssetPanel(wxWindow* parent)
    : TemplateMatchesPackageAssetPanelParent(parent) {
    current_asset_number        = 0;
    selected_refinement_package = -1;
    is_dirty                    = false;

    RefinementPackageListCtrl->InsertColumn(0, "Packages", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);
}

void TemplateMatchesPackageAssetPanel::OnCreateClick(wxCommandEvent& event) {
    NewTemplateMatchesPackageWizard* my_wizard = new NewTemplateMatchesPackageWizard(this);
    my_wizard->RunWizard(my_wizard->page);
    my_wizard->Destroy( );
}

void TemplateMatchesPackageAssetPanel::OnDisplayStackButton(wxCommandEvent& event) {
    if ( selected_refinement_package >= 0 ) {
        wxString execution_command = wxStandardPaths::Get( ).GetExecutablePath( );
    }
}

void TemplateMatchesPackageAssetPanel::OnRenameClick(wxCommandEvent& event) {
    if ( selected_refinement_package >= 0 ) {
        RefinementPackageListCtrl->EditLabel(selected_refinement_package);
    }
}

void TemplateMatchesPackageAssetPanel::OnCombineClick(wxCommandEvent& event) {
    CombineRefinementPackagesWizard* my_wizard = new CombineRefinementPackagesWizard(this);
    my_wizard->RunWizard(my_wizard->package_selection_page);
    my_wizard->Destroy( );
}

void TemplateMatchesPackageAssetPanel::OnImportClick(wxCommandEvent& event) {
    ImportRefinementPackageWizard* my_wizard = new ImportRefinementPackageWizard(this);
    my_wizard->GetPageAreaSizer( )->Add(my_wizard->m_pages.Item(0));
    my_wizard->RunWizard(my_wizard->m_pages.Item(0));
    my_wizard->Destroy( );
}

void TemplateMatchesPackageAssetPanel::OnExportClick(wxCommandEvent& event) {
    ExportRefinementPackageWizard* my_wizard = new ExportRefinementPackageWizard(this);
    my_wizard->GetPageAreaSizer( )->Add(my_wizard->m_pages.Item(0));
    my_wizard->RunWizard(my_wizard->m_pages.Item(0));
    my_wizard->Destroy( );
}

void TemplateMatchesPackageAssetPanel::OnDeleteClick(wxCommandEvent& event) {
    if ( selected_refinement_package >= 0 ) {
        // check if there is a running job which uses refinement packages, if there is refuse to delete the refinement package.

        wxMessageDialog* check_dialog = new wxMessageDialog(this, "This will remove the refinement package from your entire project, including all classifications / refinements that have been run on it.\n\nAlso note that this is just a database deletion, the particle stacks are not deleted.\n\nAre you sure you want to continue?", "Are you sure?", wxYES_NO);

        if ( check_dialog->ShowModal( ) == wxID_YES ) {
            long counter;
            int  class_counter;

            main_frame->current_project.database.Begin( );

            main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=%li", all_template_matches_packages.Item(selected_refinement_package).asset_id));
            //main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%li", all_template_matches_packages.Item(selected_refinement_package).asset_id));
            //main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li", all_template_matches_packages.Item(selected_refinement_package).asset_id));
            //main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li", all_template_matches_packages.Item(selected_refinement_package).asset_id));
            //main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_%li", all_template_matches_packages.Item(selected_refinement_package).asset_id));

            all_template_matches_packages.RemoveAt(selected_refinement_package);

            main_frame->current_project.database.Commit( );

            if ( all_template_matches_packages.GetCount( ) > 0 )
                selected_refinement_package = 0;
            else
                selected_refinement_package = -1;
            main_frame->DirtyTemplateMatchesPackages( );
        }
    }
}

void TemplateMatchesPackageAssetPanel::AddAsset(TemplateMatchesPackage* refinement_package) {
    // add into memory..

    current_asset_number++;
    refinement_package->asset_id = current_asset_number;

    all_template_matches_packages.Add(refinement_package);

    // now add it to the database..

    main_frame->current_project.database.AddTemplateMatchesPackageAsset(refinement_package);
    main_frame->DirtyTemplateMatchesPackages( );
}

void TemplateMatchesPackageAssetPanel::FillRefinementPackages( ) {
    Freeze( );

    if ( all_template_matches_packages.GetCount( ) > 0 ) {

        RefinementPackageListCtrl->SetItemCount(all_template_matches_packages.GetCount( ));
        RefinementPackageListCtrl->SetColumnWidth(0, RefinementPackageListCtrl->ReturnGuessAtColumnTextWidth( ));

        if ( selected_refinement_package >= 0 && selected_refinement_package < all_template_matches_packages.GetCount( ) ) {
            RefinementPackageListCtrl->SetItemState(selected_refinement_package, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
        }
        else {
            selected_refinement_package = 0;
            RefinementPackageListCtrl->SetItemState(selected_refinement_package, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
        }

        if ( all_template_matches_packages.GetCount( ) > 0 )
            RefinementPackageListCtrl->RefreshItems(0, all_template_matches_packages.GetCount( ) - 1);

        //StackFileNameText->SetLabel(all_template_matches_packages.Item(selected_refinement_package).stack_filename);
        //StackBoxSizeText->SetLabel(wxString::Format("%i px", all_template_matches_packages.Item(selected_refinement_package).stack_box_size));
        //NumberofClassesText->SetLabel(wxString::Format("%i", all_template_matches_packages.Item(selected_refinement_package).number_of_classes));
        //NumberofRefinementsText->SetLabel(wxString::Format("%i", all_template_matches_packages.Item(selected_refinement_package).number_of_run_refinments));
        //LastRefinementIDText->SetLabel(wxString::Format("%li", all_template_matches_packages.Item(selected_refinement_package).last_refinment_id));
        //SymmetryText->SetLabel(all_template_matches_packages.Item(selected_refinement_package).symmetry);
        //MolecularWeightText->SetLabel(wxString::Format(wxT("%.0f kDa"), all_template_matches_packages.Item(selected_refinement_package).estimated_particle_weight_in_kda));
        //LargestDimensionText->SetLabel(wxString::Format(wxT("%.0f Å"), all_template_matches_packages.Item(selected_refinement_package).estimated_particle_size_in_angstroms));

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

        /* ContainedParticlesListCtrl->SetItemCount(all_template_matches_packages.Item(selected_refinement_package).contained_particles.GetCount( ));

        if ( all_template_matches_packages.Item(selected_refinement_package).contained_particles.GetCount( ) > 0 ) {
            ContainedParticlesListCtrl->RefreshItems(0, all_template_matches_packages.Item(selected_refinement_package).contained_particles.GetCount( ) - 1);

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

        ContainedParticlesStaticText->SetLabel(wxString::Format("Contained Particles (%li) : ", all_template_matches_packages.Item(selected_refinement_package).contained_particles.GetCount( )));

        // 3D references..

        ReDrawActiveReferences( ); */
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

/*void TemplateMatchesPackageAssetPanel::ReDrawActiveReferences( ) {

    Active3DReferencesListCtrl->ClearAll( );
    Active3DReferencesListCtrl->InsertColumn(0, "Class No.", wxLIST_FORMAT_LEFT);
    Active3DReferencesListCtrl->InsertColumn(1, "Reference Volume", wxLIST_FORMAT_LEFT);

    Active3DReferencesListCtrl->SetItemCount(all_template_matches_packages.Item(selected_refinement_package).references_for_next_refinement.GetCount( ));

    if ( all_template_matches_packages.Item(selected_refinement_package).references_for_next_refinement.GetCount( ) > 0 ) {
        Active3DReferencesListCtrl->RefreshItems(0, all_template_matches_packages.Item(selected_refinement_package).references_for_next_refinement.GetCount( ) - 1);

        Active3DReferencesListCtrl->SetColumnWidth(0, Active3DReferencesListCtrl->ReturnGuessAtColumnTextWidth(0));
        Active3DReferencesListCtrl->SetColumnWidth(1, Active3DReferencesListCtrl->ReturnGuessAtColumnTextWidth(1));
    }
} */

void TemplateMatchesPackageAssetPanel::OnUpdateUI(wxUpdateUIEvent& event) {
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

void TemplateMatchesPackageAssetPanel::ImportAllFromDatabase( ) {
    int                     counter;
    TemplateMatchesPackage* temp_package;

    all_template_matches_packages.Clear( );

    // Now the groups..

    main_frame->current_project.database.BeginAllTemplateMatchesPackagesSelect( );

    while ( main_frame->current_project.database.last_return_code == SQLITE_ROW ) {
        temp_package = main_frame->current_project.database.GetNextTemplateMatchesPackage( );
        if ( temp_package->asset_id > current_asset_number )
            current_asset_number = temp_package->asset_id;

        all_template_matches_packages.Add(temp_package);
    }

    main_frame->current_project.database.EndAllTemplateMatchesPackagesSelect( );

    main_frame->DirtyTemplateMatchesPackages( );
}

void TemplateMatchesPackageAssetPanel::MouseVeto(wxMouseEvent& event) {
    //Do nothing
}

void TemplateMatchesPackageAssetPanel::MouseCheckPackagesVeto(wxMouseEvent& event) {
    VetoInvalidMouse(RefinementPackageListCtrl, event);
}

void TemplateMatchesPackageAssetPanel::MouseCheckParticlesVeto(wxMouseEvent& event) {
    VetoInvalidMouse(ContainedParticlesListCtrl, event);
}

void TemplateMatchesPackageAssetPanel::VetoInvalidMouse(wxListCtrl* wanted_list, wxMouseEvent& event) {
    // Don't allow clicking on anything other than item, to stop the selection bar changing

    int flags;

    if ( wanted_list->HitTest(event.GetPosition( ), flags) != wxNOT_FOUND ) {
        should_veto_motion = false;
        event.Skip( );
    }
    else
        should_veto_motion = true;
}

void TemplateMatchesPackageAssetPanel::OnMotion(wxMouseEvent& event) {
    if ( should_veto_motion == false )
        event.Skip( );
}

void TemplateMatchesPackageAssetPanel::OnPackageFocusChange(wxListEvent& event) {
    if ( event.GetIndex( ) != selected_refinement_package ) {
        selected_refinement_package = event.GetIndex( );
        is_dirty                    = true;

        if ( selected_refinement_package >= 0 ) {
            ContainedParticlesStaticText->SetLabel(wxString::Format("Contained Matches (%li) : ", all_template_matches_packages.Item(selected_refinement_package).contained_match_count));
        }
        else
            ContainedParticlesStaticText->SetLabel("Contained Matches : ");
    }

    //wxPrintf("Selected refinement package = %li\n", selected_refinement_package);

    event.Skip( );
}

void TemplateMatchesPackageAssetPanel::OnPackageActivated(wxListEvent& event) {
    RefinementPackageListCtrl->EditLabel(event.GetIndex( ));
}

/*void TemplateMatchesPackageAssetPanel::OnVolumeListItemActivated(wxListEvent& event) {
    MyVolumeChooserDialog* dialog = new MyVolumeChooserDialog(this);

    dialog->ComboBox->SetSelection(volume_asset_panel->ReturnArrayPositionFromAssetID(all_template_matches_packages.Item(selected_refinement_package).references_for_next_refinement.Item(event.GetIndex( ))) + 1);
    dialog->Fit( );
    if ( dialog->ShowModal( ) == wxID_OK ) {
        if ( dialog->selected_volume_id != all_template_matches_packages.Item(selected_refinement_package).references_for_next_refinement.Item(event.GetIndex( )) ) {
            all_template_matches_packages.Item(selected_refinement_package).references_for_next_refinement.Item(event.GetIndex( )) = dialog->selected_volume_id;
            // Change in database..
            main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li SET VOLUME_ASSET_ID=%li WHERE CLASS_NUMBER=%li;", all_template_matches_packages.Item(selected_refinement_package).asset_id, dialog->selected_volume_id, event.GetIndex( ) + 1));

            ReDrawActiveReferences( );
        }
    }
    dialog->Destroy( );
}*/

void TemplateMatchesPackageAssetPanel::OnBeginEdit(wxListEvent& event) {
    event.Skip( );
}

void TemplateMatchesPackageAssetPanel::OnEndEdit(wxListEvent& event) {

    if ( event.GetLabel( ) == wxEmptyString ) {
        RefinementPackageListCtrl->SetItemText(event.GetIndex( ), all_template_matches_packages.Item(event.GetIndex( )).name);
        event.Veto( );
    }
    else {
        if ( all_template_matches_packages.Item(event.GetIndex( )).name != event.GetLabel( ) ) {
            all_template_matches_packages.Item(event.GetIndex( )).name = event.GetLabel( );

            // do in database also..

            wxString safe_name = all_template_matches_packages.Item(event.GetIndex( )).name;

            //escape apostrophes
            safe_name.Replace("'", "''");

            wxString sql_command = wxString::Format("UPDATE TEMPLATE_MATCHES_PACKAGE_ASSETS SET NAME='%s' WHERE TEMPLATE_MATCHES_PACKAGE_ASSET_ID=%li", safe_name, all_template_matches_packages.Item(event.GetIndex( )).asset_id);
            main_frame->current_project.database.ExecuteSQL(sql_command.ToUTF8( ).data( ));

            main_frame->DirtyTemplateMatchesPackages( );
            event.Skip( );
        }
        else
            event.Veto( );
    }
}

void TemplateMatchesPackageAssetPanel::Reset( ) {
    all_template_matches_packages.Clear( );
    main_frame->DirtyTemplateMatchesPackages( );
    ContainedParticlesStaticText->SetLabel("Contained Particles : ");
}

long TemplateMatchesPackageAssetPanel::ReturnArrayPositionFromAssetID(long wanted_asset_id) {
    for ( long counter = 0; counter < all_template_matches_packages.GetCount( ); counter++ ) {
        if ( all_template_matches_packages[counter].asset_id == wanted_asset_id )
            return counter;
    }

    return -1;
}
