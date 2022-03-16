//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern AtomicCoordinatesAssetPanel* atomic_coordinates_asset_panel;
extern MyMainFrame*                 main_frame;

AtomicCoordinatesImportDialog::AtomicCoordinatesImportDialog(wxWindow* parent)
    : AtomicCoordinatesImportDialogParent(parent) {
    int list_height;
    int list_width;

    PathListCtrl->GetClientSize(&list_width, &list_height);
    PathListCtrl->InsertColumn(0, "Files", wxLIST_FORMAT_LEFT, list_width);
}

void AtomicCoordinatesImportDialog::AddFilesClick(wxCommandEvent& event) {
    wxFileDialog openFileDialog(this, _("Select PDBx/mmCIF files - basic wildcards are allowed"), "", "", "Atomic Coord files (*.pdb*;*.cif*)|*.*cif*;*.CIF;*.pdb*;*.PDB*", wxFD_OPEN | wxFD_MULTIPLE);

    if ( openFileDialog.ShowModal( ) == wxID_OK ) {
        wxArrayString selected_paths;
        openFileDialog.GetPaths(selected_paths);

        PathListCtrl->Freeze( );

        for ( unsigned long counter = 0; counter < selected_paths.GetCount( ); counter++ ) {
            // is this an actual filename, that exists - in which case add it.

            if ( DoesFileExist(selected_paths.Item(counter)) == true )
                PathListCtrl->InsertItem(PathListCtrl->GetItemCount( ), selected_paths.Item(counter), PathListCtrl->GetItemCount( ));
            else {
                // perhaps it is a wildcard..
                int           wildcard_counter;
                wxArrayString wildcard_files;
                wxString      directory_string;
                wxString      file_string;
                wxString      current_extension;

                SplitFileIntoDirectoryAndFile(selected_paths.Item(counter), directory_string, file_string);
                wxDir::GetAllFiles(directory_string, &wildcard_files, file_string, wxDIR_FILES);

                for ( int wildcard_counter = 0; wildcard_counter < wildcard_files.GetCount( ); wildcard_counter++ ) {
                    current_extension = wxFileName(wildcard_files.Item(wildcard_counter)).GetExt( );
                    current_extension = current_extension.MakeLower( );

                    if ( current_extension == "pdb" || current_extension == "pdbx" || current_extension == "cif" || current_extension == "mmcif" )
                        PathListCtrl->InsertItem(PathListCtrl->GetItemCount( ), wildcard_files.Item(wildcard_counter), PathListCtrl->GetItemCount( ));
                }
            }
        }

        PathListCtrl->SetColumnWidth(0, wxLIST_AUTOSIZE);
        PathListCtrl->Thaw( );

        CheckImportButtonStatus( );
    }
}

void AtomicCoordinatesImportDialog::ClearClick(wxCommandEvent& event) {
    PathListCtrl->DeleteAllItems( );
    CheckImportButtonStatus( );
}

void AtomicCoordinatesImportDialog::CancelClick(wxCommandEvent& event) {
    Destroy( );
}

void AtomicCoordinatesImportDialog::AddDirectoryClick(wxCommandEvent& event) {
    wxDirDialog dlg(NULL, "Choose import directory", "", wxDD_DEFAULT_STYLE | wxDD_DIR_MUST_EXIST);

    wxArrayString all_files;

    if ( dlg.ShowModal( ) == wxID_OK ) {
        wxDir::GetAllFiles(dlg.GetPath( ), &all_files, "*.pdb*", wxDIR_FILES);
        wxDir::GetAllFiles(dlg.GetPath( ), &all_files, "*.*cif*", wxDIR_FILES);

        all_files.Sort( );

        PathListCtrl->Freeze( );

        for ( unsigned long counter = 0; counter < all_files.GetCount( ); counter++ ) {
            PathListCtrl->InsertItem(PathListCtrl->GetItemCount( ), all_files.Item(counter), PathListCtrl->GetItemCount( ));
        }

        PathListCtrl->SetColumnWidth(0, wxLIST_AUTOSIZE);
        PathListCtrl->Thaw( );

        CheckImportButtonStatus( );
    }
}

void AtomicCoordinatesImportDialog::OnTextKeyPress(wxKeyEvent& event) {

    int  keycode      = event.GetKeyCode( );
    bool is_valid_key = false;

    if ( keycode > 31 && keycode < 127 ) {
        if ( keycode > 47 && keycode < 58 )
            is_valid_key = true;
        else if ( keycode > 44 && keycode < 47 )
            is_valid_key = true;
    }
    else
        is_valid_key = true;

    if ( is_valid_key == true )
        event.Skip( );
}

void AtomicCoordinatesImportDialog::TextChanged(wxCommandEvent& event) {
    CheckImportButtonStatus( );
}

void AtomicCoordinatesImportDialog::CheckImportButtonStatus( ) {
    bool   enable_import_box = true;
    double temp_double;

    if ( PathListCtrl->GetItemCount( ) < 1 )
        enable_import_box = false;

    // if (PixelSizeText->GetLineLength(0) == 0 ) enable_import_box = false;

    if ( enable_import_box == true )
        ImportButton->Enable(true);
    else
        ImportButton->Enable(false);

    Update( );
    Refresh( );
}

void AtomicCoordinatesImportDialog::ImportClick(wxCommandEvent& event) {
    //

    bool have_errors = false;
    // PixelSizeText->GetLineText(0).ToDouble(&pixel_size);

    //  In case we need it make an error dialog..

    MyErrorDialog* my_error = new MyErrorDialog(this);

    if ( PathListCtrl->GetItemCount( ) > 0 ) {
        AtomicCoordinatesAsset temp_asset;

        // ProgressBar..

        OneSecondProgressDialog* my_progress_dialog = new OneSecondProgressDialog("Import AtomicCoordinates", "Importing AtomicCoordinates...", PathListCtrl->GetItemCount( ), this, wxPD_AUTO_HIDE | wxPD_APP_MODAL | wxPD_ELAPSED_TIME);

        // loop through all the files and add them as assets..

        // for adding to the database..
        main_frame->current_project.database.BeginAtomicCoordinatesAssetInsert( );

        for ( long counter = 0; counter < PathListCtrl->GetItemCount( ); counter++ ) {
            temp_asset.Update(PathListCtrl->GetItemText(counter));

            // Check this movie is not already an asset..

            if ( atomic_coordinates_asset_panel->IsFileAnAsset(temp_asset.filename) == false ) {
                if ( temp_asset.is_valid == true ) {
                    temp_asset.asset_id   = atomic_coordinates_asset_panel->current_asset_number;
                    temp_asset.asset_name = temp_asset.filename.GetName( );
                    atomic_coordinates_asset_panel->AddAsset(&temp_asset);

                    main_frame->current_project.database.AddNextAtomicCoordinatesAsset(&temp_asset);
                }
                else {
                    my_error->ErrorText->AppendText(wxString::Format(wxT("%s is not a valid Atomic Coordinate file, skipping\n"), temp_asset.ReturnFullPathString( )));
                    have_errors = true;
                }
            }
            else {
                my_error->ErrorText->AppendText(wxString::Format(wxT("%s is already an asset, skipping\n"), temp_asset.ReturnFullPathString( )));
                have_errors = true;
            }

            my_progress_dialog->Update(counter);
        }

        // do database write..

        main_frame->current_project.database.EndAtomicCoordinatesAssetInsert( );

        my_progress_dialog->Destroy( );

        atomic_coordinates_asset_panel->SetSelectedGroup(0);
        atomic_coordinates_asset_panel->FillGroupList( );
        atomic_coordinates_asset_panel->FillContentsList( );
        //main_frame->RecalculateAssetBrowser();
        //main_frame->Dirtyatomic_coordinatesGroups();
        // main_frame->DirtyAtomicCoordinates(); FIXME
    }

    if ( have_errors == true ) {
        Hide( );
        my_error->ShowModal( );
    }

    my_error->Destroy( );
    Destroy( );
}
