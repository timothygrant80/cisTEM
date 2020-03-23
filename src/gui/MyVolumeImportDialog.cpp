//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyVolumeAssetPanel *volume_asset_panel;
extern MyMainFrame *main_frame;

MyVolumeImportDialog::MyVolumeImportDialog( wxWindow* parent )
:
VolumeImportDialog( parent )
{
	int list_height;
	int list_width;

	PathListCtrl->GetClientSize(&list_width, &list_height);
	PathListCtrl->InsertColumn(0, "Files", wxLIST_FORMAT_LEFT, list_width);
}

void MyVolumeImportDialog::AddFilesClick( wxCommandEvent& event )
{
	  wxFileDialog openFileDialog(this, _("Select MRC files - basic wildcards are allowed"), "", "", "MRC files (*.mrc)|*.mrc;*.mrcs", wxFD_OPEN | wxFD_MULTIPLE);



	    if (openFileDialog.ShowModal() == wxID_OK)
	    {
	    	wxArrayString selected_paths;
	    	openFileDialog.GetPaths(selected_paths);

	      	PathListCtrl->Freeze();

	    	for (unsigned long counter = 0; counter < selected_paths.GetCount(); counter++)
	    	{
	    		// is this an actual filename, that exists - in which case add it.

	    		if (DoesFileExist(selected_paths.Item(counter)) == true) PathListCtrl->InsertItem(PathListCtrl->GetItemCount(), selected_paths.Item(counter), PathListCtrl->GetItemCount());
	    		else
	    		{
	    			// perhaps it is a wildcard..
	    			int wildcard_counter;
	    			wxArrayString wildcard_files;
	    			wxString directory_string;
	    			wxString file_string;
	    			wxString current_extension;

	    			SplitFileIntoDirectoryAndFile(selected_paths.Item(counter), directory_string, file_string);
	    			wxDir::GetAllFiles 	( directory_string, &wildcard_files, file_string, wxDIR_FILES);

	    			for (int wildcard_counter = 0; wildcard_counter < wildcard_files.GetCount(); wildcard_counter++)
	    			{
	    				current_extension = wxFileName(wildcard_files.Item(wildcard_counter)).GetExt();
	    				current_extension = current_extension.MakeLower();

	    				if ( current_extension == "mrc" || current_extension == "mrcs") PathListCtrl->InsertItem(PathListCtrl->GetItemCount(), wildcard_files.Item(wildcard_counter), PathListCtrl->GetItemCount());
	    			}

	    		}
	    	}

	    	PathListCtrl->SetColumnWidth(0, wxLIST_AUTOSIZE);
	    	PathListCtrl->Thaw();

	    	CheckImportButtonStatus();
	    }
}

void MyVolumeImportDialog::ClearClick( wxCommandEvent& event )
{
	PathListCtrl->DeleteAllItems();
	CheckImportButtonStatus();

}

void MyVolumeImportDialog::CancelClick( wxCommandEvent& event )
{
	Destroy();

}

void MyVolumeImportDialog::AddDirectoryClick( wxCommandEvent& event )
{
	wxDirDialog dlg(NULL, "Choose import directory", "", wxDD_DEFAULT_STYLE | wxDD_DIR_MUST_EXIST);

	wxArrayString all_files;

    if (dlg.ShowModal() == wxID_OK)
    {
    	wxDir::GetAllFiles 	( dlg.GetPath(), &all_files, "*.mrc", wxDIR_FILES);
    	wxDir::GetAllFiles 	( dlg.GetPath(), &all_files, "*.mrcs", wxDIR_FILES);

    	all_files.Sort();

    	PathListCtrl->Freeze();

    	for (unsigned long counter = 0; counter < all_files.GetCount(); counter++)
    	{
    		PathListCtrl->InsertItem(PathListCtrl->GetItemCount(), all_files.Item(counter), PathListCtrl->GetItemCount());
    	}

    	PathListCtrl->SetColumnWidth(0, wxLIST_AUTOSIZE);
    	PathListCtrl->Thaw();

    	CheckImportButtonStatus();

    }

}

void MyVolumeImportDialog::OnTextKeyPress( wxKeyEvent& event )
{

			int keycode = event.GetKeyCode();
			bool is_valid_key = false;


			if (keycode > 31 && keycode < 127)
			{
				if (keycode > 47 && keycode < 58) is_valid_key = true;
				else if (keycode > 44 && keycode < 47) is_valid_key = true;
			}
			else is_valid_key = true;

			if (is_valid_key == true) event.Skip();


}

void MyVolumeImportDialog::TextChanged( wxCommandEvent& event)
{
	CheckImportButtonStatus();
}

void MyVolumeImportDialog::CheckImportButtonStatus()
{
	bool enable_import_box = true;
	double temp_double;

	if (PathListCtrl->GetItemCount() < 1) enable_import_box = false;

	if (PixelSizeText->GetLineLength(0) == 0 ) enable_import_box = false;

	if (enable_import_box == true) ImportButton->Enable(true);
	else  ImportButton->Enable(false);

	Update();
	Refresh();
}



void MyVolumeImportDialog::ImportClick( wxCommandEvent& event )
{
	// Get the microscope values


	double microscope_voltage;
	double pixel_size;
	double spherical_aberration;

	bool have_errors = false;
	PixelSizeText->GetLineText(0).ToDouble(&pixel_size);

	//  In case we need it make an error dialog..

	MyErrorDialog *my_error = new MyErrorDialog(this);

	if (PathListCtrl->GetItemCount() > 0)
	{
		VolumeAsset temp_asset;

		temp_asset.pixel_size = pixel_size;
		temp_asset.half_map_1_filename = "";
		temp_asset.half_map_2_filename = "";


		// ProgressBar..

		OneSecondProgressDialog *my_progress_dialog = new OneSecondProgressDialog("Import Volumes",	"Importing Volumes...", PathListCtrl->GetItemCount(), this,  wxPD_AUTO_HIDE|wxPD_APP_MODAL|wxPD_ELAPSED_TIME);

		// loop through all the files and add them as assets..

		// for adding to the database..
		main_frame->current_project.database.BeginVolumeAssetInsert();

		for (long counter = 0; counter < PathListCtrl->GetItemCount(); counter++)
		{
			temp_asset.Update(PathListCtrl->GetItemText(counter));

			// Check this movie is not already an asset..

			if (volume_asset_panel->IsFileAnAsset(temp_asset.filename) == false)
			{
				if (temp_asset.is_valid == true)
					{
						temp_asset.asset_id = volume_asset_panel->current_asset_number;
						temp_asset.asset_name = temp_asset.filename.GetName();
						volume_asset_panel->AddAsset(&temp_asset);

					//	void AddNextVolumeAsset(int image_asset_id,  wxString name, wxString filename, int reconstruction_job_id, double pixel_size, int x_size, int y_size, int z_size);
						main_frame->current_project.database.AddNextVolumeAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath(), -1, temp_asset.pixel_size, temp_asset.x_size, temp_asset.y_size, temp_asset.z_size, temp_asset.half_map_1_filename.GetFullPath(), temp_asset.half_map_2_filename.GetFullPath());

					}
					else
					{
						my_error->ErrorText->AppendText(wxString::Format(wxT("%s is not a valid MRC file, skipping\n"), temp_asset.ReturnFullPathString()));
						have_errors = true;
					}
			}
			else
			{
				my_error->ErrorText->AppendText(wxString::Format(wxT("%s is already an asset, skipping\n"), temp_asset.ReturnFullPathString()));
				have_errors = true;
			}

			my_progress_dialog->Update(counter);
		}

		// do database write..

		main_frame->current_project.database.EndVolumeAssetInsert();

		my_progress_dialog->Destroy();

		volume_asset_panel->SetSelectedGroup(0);
		volume_asset_panel->FillGroupList();
		volume_asset_panel->FillContentsList();
		//main_frame->RecalculateAssetBrowser();
		//main_frame->DirtyVolumeGroups();
		main_frame->DirtyVolumes();
	}



	if (have_errors == true)
	{
		Hide();
		my_error->ShowModal();
	}

	my_error->Destroy();
	Destroy();
}
