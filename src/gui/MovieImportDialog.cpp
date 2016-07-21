//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMovieAssetPanel *movie_asset_panel;
extern MyMainFrame *main_frame;

MyMovieImportDialog::MyMovieImportDialog( wxWindow* parent )
:
MovieImportDialog( parent )
{
	int list_height;
	int list_width;

	PathListCtrl->GetClientSize(&list_width, &list_height);
	PathListCtrl->InsertColumn(0, "Files", wxLIST_FORMAT_LEFT, list_width);
}

void MyMovieImportDialog::AddFilesClick( wxCommandEvent& event )
{
    wxFileDialog openFileDialog(this, _("Select MRC files"), "", "", "MRC files (*.mrc)|*.mrc;*.mrcs", wxFD_OPEN |wxFD_FILE_MUST_EXIST | wxFD_MULTIPLE);



    if (openFileDialog.ShowModal() == wxID_OK)
    {
    	wxArrayString selected_paths;
    	openFileDialog.GetPaths(selected_paths);

      	PathListCtrl->Freeze();

    	for (unsigned long counter = 0; counter < selected_paths.GetCount(); counter++)
    	{
    		PathListCtrl->InsertItem(PathListCtrl->GetItemCount(), selected_paths.Item(counter), PathListCtrl->GetItemCount());
    	}

    	PathListCtrl->SetColumnWidth(0, wxLIST_AUTOSIZE);
    	PathListCtrl->Thaw();

    	CheckImportButtonStatus();
    }

}

void MyMovieImportDialog::ClearClick( wxCommandEvent& event )
{
	PathListCtrl->DeleteAllItems();
	CheckImportButtonStatus();

}

void MyMovieImportDialog::CancelClick( wxCommandEvent& event )
{
	Destroy();

}

void MyMovieImportDialog::AddDirectoryClick( wxCommandEvent& event )
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

void MyMovieImportDialog::OnTextKeyPress( wxKeyEvent& event )
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

void MyMovieImportDialog::TextChanged( wxCommandEvent& event)
{
	CheckImportButtonStatus();
}

void MyMovieImportDialog::CheckImportButtonStatus()
{
	bool enable_import_box = true;
	double temp_double;

	if (PathListCtrl->GetItemCount() < 1) enable_import_box = false;

	if (VoltageCombo->IsTextEmpty() == true || PixelSizeText->GetLineLength(0) == 0 || CsText->GetLineLength (0) == 0 || DoseText->GetLineLength(0) == 0) enable_import_box = false;

	if (enable_import_box == true) ImportButton->Enable(true);
	else  ImportButton->Enable(false);

	Update();
	Refresh();
}



void MyMovieImportDialog::ImportClick( wxCommandEvent& event )
{
	// Get the microscope values


	double microscope_voltage;
	double pixel_size;
	double spherical_aberration;
	double dose_per_frame;

	bool have_errors = false;

	VoltageCombo->GetValue().ToDouble(&microscope_voltage);
	//VoltageCombo->GetStringSelection().ToDouble(&microscope_voltage);
	PixelSizeText->GetLineText(0).ToDouble(&pixel_size);
	CsText->GetLineText(0).ToDouble(&spherical_aberration);
	DoseText->GetLineText(0).ToDouble(&dose_per_frame);

	//  In case we need it make an error dialog..

	MyErrorDialog *my_error = new MyErrorDialog(this);

	if (PathListCtrl->GetItemCount() > 0)
	{
		MovieAsset temp_asset;



		temp_asset.microscope_voltage = microscope_voltage;
		temp_asset.pixel_size = pixel_size;
		temp_asset.dose_per_frame = dose_per_frame;
		temp_asset.spherical_aberration = spherical_aberration;

		// ProgressBar..

		wxGenericProgressDialog *my_progress_dialog = new wxGenericProgressDialog("Import Movie",	"Importing Movies...", PathListCtrl->GetItemCount(), this,  wxPD_AUTO_HIDE|wxPD_APP_MODAL|wxPD_ELAPSED_TIME);

		// loop through all the files and add them as assets..

		// for adding to the database..
		main_frame->current_project.database.BeginMovieAssetInsert();

		for (long counter = 0; counter < PathListCtrl->GetItemCount(); counter++)
		{
			temp_asset.Update(PathListCtrl->GetItemText(counter));

			// Check this movie is not already an asset..

			if (movie_asset_panel->IsFileAnAsset(temp_asset.filename) == false)
			{
				if (temp_asset.is_valid == true)
					{
						temp_asset.asset_id = movie_asset_panel->current_asset_number;
						temp_asset.total_dose = double(temp_asset.number_of_frames) * dose_per_frame;
						movie_asset_panel->AddAsset(&temp_asset);

						main_frame->current_project.database.AddNextMovieAsset(temp_asset.asset_id, temp_asset.filename.GetFullPath(), 1, temp_asset.x_size, temp_asset.y_size, temp_asset.number_of_frames, temp_asset.microscope_voltage, temp_asset.pixel_size, temp_asset.dose_per_frame, temp_asset.spherical_aberration);

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

		main_frame->current_project.database.EndMovieAssetInsert();

		my_progress_dialog->Destroy();

		movie_asset_panel->SetSelectedGroup(0);
		movie_asset_panel->FillGroupList();
		movie_asset_panel->FillContentsList();
		main_frame->RecalculateAssetBrowser();
	}



	if (have_errors == true)
	{
		Hide();
		my_error->ShowModal();
	}

	my_error->Destroy();
	Destroy();
}
