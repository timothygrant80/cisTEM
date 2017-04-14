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

	DesiredPixelSizeTextCtrl->SetMinMaxValue(0, 100);
	MajorScaleTextCtrl->SetMinMaxValue(0, FLT_MAX);
	MinorScaleTextCtrl->SetMinMaxValue(0, FLT_MAX);
	MajorScaleTextCtrl->SetPrecision(3);
	MinorScaleTextCtrl->SetPrecision(3);
}

void MyMovieImportDialog::AddFilesClick( wxCommandEvent& event )
{
    wxFileDialog openFileDialog(this, _("Select movie files"), "", "", "MRC or TIFF files (*.mrc;*.mrcs;*.tif)|*.mrc;*.mrcs;*.tif", wxFD_OPEN |wxFD_FILE_MUST_EXIST | wxFD_MULTIPLE);



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
    	wxDir::GetAllFiles 	( dlg.GetPath(), &all_files, "*.tif", wxDIR_FILES);

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
	double current_pixel_size;

	if (PathListCtrl->GetItemCount() < 1) enable_import_box = false;

	if (VoltageCombo->IsTextEmpty() == true || PixelSizeText->GetLineLength(0) == 0 || CsText->GetLineLength (0) == 0 || DoseText->GetLineLength(0) == 0) enable_import_box = false;

	if ((! MoviesAreGainCorrectedCheckBox->IsChecked()) && (! GainFilePicker->GetFileName().Exists()) ) enable_import_box = false;

	//if ((! MoviesAreGainCorrectedCheckBox->IsChecked())) wxPrintf("Movies are not gain corrected\n");
	//if (GainFilePicker->GetFileName().Exists()) wxPrintf("Gain file exists\n");

	if (ResampleMoviesCheckBox->IsChecked() == true)
	{
		if (PixelSizeText->GetLineText(0).ToDouble(&current_pixel_size) == false) enable_import_box = false;
		else
		if (current_pixel_size >= DesiredPixelSizeTextCtrl->ReturnValue()) enable_import_box = false;
	}

	if (CorrectMagDistortionCheckBox->IsChecked() == true)
	{
		if (MajorScaleTextCtrl->ReturnValue() < MinorScaleTextCtrl->ReturnValue()) enable_import_box = false;
	}

	ImportButton->Enable(enable_import_box);

	Update();
	Refresh();
}

void MyMovieImportDialog::OnGainFilePickerChanged(wxFileDirPickerEvent & event )
{
	CheckImportButtonStatus();
}


void MyMovieImportDialog::OnMoviesAreGainCorrectedCheckBox( wxCommandEvent & event )
{
	GainFilePicker->Enable(! MoviesAreGainCorrectedCheckBox->IsChecked());
	GainFileStaticText->Enable(! MoviesAreGainCorrectedCheckBox->IsChecked());
	CheckImportButtonStatus();
}

void MyMovieImportDialog::OnResampleMoviesCheckBox( wxCommandEvent & event )
{
	DesiredPixelSizeStaticText->Enable(ResampleMoviesCheckBox->IsChecked());
	DesiredPixelSizeTextCtrl->Enable(ResampleMoviesCheckBox->IsChecked());
	//DesiredPixelSizeTextCtrl->ChangeValue("1.00");
	CheckImportButtonStatus();
}

void MyMovieImportDialog::OnCorrectMagDistortionCheckBox( wxCommandEvent & event )
{
	DistortionAngleTextCtrl->Enable(CorrectMagDistortionCheckBox->IsChecked());
	MajorScaleTextCtrl->Enable(CorrectMagDistortionCheckBox->IsChecked());
	MinorScaleTextCtrl->Enable(CorrectMagDistortionCheckBox->IsChecked());
	DistortionAngleStaticText->Enable(CorrectMagDistortionCheckBox->IsChecked());
	MajorScaleStaticText->Enable(CorrectMagDistortionCheckBox->IsChecked());
	MinorScaleStaticText->Enable(CorrectMagDistortionCheckBox->IsChecked());

	CheckImportButtonStatus();
}

void MyMovieImportDialog::ImportClick( wxCommandEvent& event )
{
	// Get the microscope values


	double microscope_voltage;
	double pixel_size;
	double spherical_aberration;
	double dose_per_frame;

	bool have_errors = false;

	int gain_ref_x_size = -1;
	int gain_ref_y_size = -1;

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

		if (MoviesAreGainCorrectedCheckBox->IsChecked())
		{
			temp_asset.gain_filename = "";
		}
		else
		{
			temp_asset.gain_filename = GainFilePicker->GetFileName().GetFullPath();

			ImageFile gain_ref;
			if (gain_ref.OpenFile(temp_asset.gain_filename.ToStdString(), false) == false)
			{
				my_error->ErrorText->AppendText(wxString::Format(wxT("cannot open gain reference file (%s)\n"), temp_asset.gain_filename));
				have_errors = true;
			}
			else
			{
				gain_ref_x_size = gain_ref.ReturnXSize();
				gain_ref_y_size = gain_ref.ReturnYSize();
			}

		}

		// get the size of the gain ref..



		if (ResampleMoviesCheckBox->IsChecked() == false)
		{
			temp_asset.output_binning_factor = 1;
		}
		else
		{
			float desired_pixel_size = DesiredPixelSizeTextCtrl->ReturnValue();

			// if we are correcting for a mag distortion - calculate the binning factor off the corrected pixel size

			if (CorrectMagDistortionCheckBox->IsChecked() == false)
			{
				temp_asset.output_binning_factor = desired_pixel_size / temp_asset.pixel_size;
			}
			else
			{
				temp_asset.output_binning_factor = desired_pixel_size / ReturnMagDistortionCorrectedPixelSize(temp_asset.pixel_size, MajorScaleTextCtrl->ReturnValue(), MinorScaleTextCtrl->ReturnValue());
			}

		}

		if (CorrectMagDistortionCheckBox->IsChecked() == false)
		{
			temp_asset.correct_mag_distortion = false;
			temp_asset.mag_distortion_angle = 0.0;
			temp_asset.mag_distortion_major_scale = 1.0;
			temp_asset.mag_distortion_minor_scale = 1.0;
		}
		else
		{
			temp_asset.correct_mag_distortion = true;
			temp_asset.mag_distortion_angle = DistortionAngleTextCtrl->ReturnValue();
			temp_asset.mag_distortion_major_scale = MajorScaleTextCtrl->ReturnValue();
			temp_asset.mag_distortion_minor_scale = MinorScaleTextCtrl->ReturnValue();

					}

		// ProgressBar..

		OneSecondProgressDialog *my_progress_dialog = new OneSecondProgressDialog("Import Movie",	"Importing Movies...", PathListCtrl->GetItemCount(), this,  wxPD_AUTO_HIDE|wxPD_APP_MODAL|wxPD_REMAINING_TIME);

		// loop through all the files and add them as assets..

		// for adding to the database..
		main_frame->current_project.database.BeginMovieAssetInsert();

		for (long counter = 0; counter < PathListCtrl->GetItemCount(); counter++)
		{
			wxDateTime start = wxDateTime::UNow();
			temp_asset.Update(PathListCtrl->GetItemText(counter));
			wxDateTime end = wxDateTime::UNow();
			wxPrintf("File parsing (disk access) took %li milliseconds\n", (end-start).GetMilliseconds());
			start = wxDateTime::UNow();

			// check everything ok with gain_ref

			if (CorrectMagDistortionCheckBox->IsChecked() == true && (temp_asset.x_size != gain_ref_x_size || temp_asset.y_size != gain_ref_y_size))
			{
				my_error->ErrorText->AppendText(wxString::Format(wxT("%s does not have the same size as the provided gain reference, skipping\n"), temp_asset.ReturnFullPathString()));
				have_errors = true;
			}
			else
			if (movie_asset_panel->IsFileAnAsset(temp_asset.filename) == false) // Check this movie is not already an asset..
			{
				end = wxDateTime::UNow();

				wxPrintf("File searching took %li milliseconds\n", (end-start).GetMilliseconds());

				if (temp_asset.is_valid == true)
				{
					if (temp_asset.number_of_frames < 3 )
					{
						my_error->ErrorText->AppendText(wxString::Format(wxT("%s contains less than 3 frames, skipping\n"), temp_asset.ReturnFullPathString()));
						have_errors = true;
					}
					else
					{

						temp_asset.asset_id = movie_asset_panel->current_asset_number;
						temp_asset.asset_name = temp_asset.filename.GetName();
						temp_asset.total_dose = double(temp_asset.number_of_frames) * dose_per_frame;
						movie_asset_panel->AddAsset(&temp_asset);

						main_frame->current_project.database.AddNextMovieAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath(), 1, temp_asset.x_size, temp_asset.y_size, temp_asset.number_of_frames, temp_asset.microscope_voltage, temp_asset.pixel_size, temp_asset.dose_per_frame, temp_asset.spherical_aberration,temp_asset.gain_filename,temp_asset.output_binning_factor, temp_asset.correct_mag_distortion, temp_asset.mag_distortion_angle, temp_asset.mag_distortion_major_scale, temp_asset.mag_distortion_minor_scale);
					}
				}
				else
				{
					my_error->ErrorText->AppendText(wxString::Format(wxT("%s is not a valid image file, skipping\n"), temp_asset.ReturnFullPathString()));
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
		main_frame->DirtyMovieGroups();
//		movie_asset_panel->SetSelectedGroup(0);
	//	movie_asset_panel->FillGroupList();
		//movie_asset_panel->FillContentsList();
		//main_frame->RecalculateAssetBrowser();
	}



	if (have_errors == true)
	{
		Hide();
		my_error->ShowModal();
	}

	my_error->Destroy();
	Destroy();
}
