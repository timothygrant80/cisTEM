//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMovieAssetPanel *movie_asset_panel;
extern MyImageAssetPanel *image_asset_panel;

MyMovieAlignResultsPanel::MyMovieAlignResultsPanel( wxWindow* parent )
:
MovieAlignResultsPanel( parent )
{

	Bind(wxEVT_DATAVIEW_ITEM_VALUE_CHANGED, wxDataViewEventHandler( MyMovieAlignResultsPanel::OnValueChanged), this);

	alignment_job_ids = NULL;
	number_of_alignmnet_jobs = 0;
	per_row_asset_id = NULL;
	per_row_array_position = NULL;
	number_of_assets = 0;

	selected_row = -1;
	selected_column = -1;
	doing_panel_fill = false;

	current_fill_command = "SELECT MOVIE_ASSET_ID FROM MOVIE_ALIGNMENT_LIST";
	is_dirty=false;
	group_combo_is_dirty=false;

	FillGroupComboBox();

	Bind(wxEVT_CHAR_HOOK, &MyMovieAlignResultsPanel::OnCharHook, this);

	ResultPanel->SpectraPanel->use_auto_contrast = false;

}

void MyMovieAlignResultsPanel::OnCharHook( wxKeyEvent& event )
{
	if (event.GetUnicodeKey() == 'N')
	{
		ResultDataView->NextEye();
	}
	else
	if (event.GetUnicodeKey() == 'P')
	{
		ResultDataView->PreviousEye();
	}
	else
	event.Skip();
}

void MyMovieAlignResultsPanel::OnValueChanged(wxDataViewEvent &event)
{
	if (doing_panel_fill == false)
	{
		wxDataViewItem current_item = event.GetItem();
		int row =  ResultDataView->ItemToRow(current_item);
		int column = event.GetColumn();
		long value;

		int old_selected_row = -1;
		int old_selected_column = -1;

		wxVariant temp_variant;
		ResultDataView->GetValue(temp_variant, row, column);
		value = temp_variant.GetLong();

		if ((value == CHECKED_WITH_EYE || value == UNCHECKED_WITH_EYE) && (selected_row != row || selected_column != column))
		{
			old_selected_row = selected_row;
			old_selected_column = selected_column;

			selected_row = row;
			selected_column = column;

			DrawCurveAndFillDetails(row, column);
			//wxPrintf("drawing curve\n");

		}
		else // This is dodgy, and relies on the fact that a box will be deselected, before a new box is selected...
		{
			if ((value == CHECKED  && (selected_row != row || selected_column != column)) || (value == CHECKED_WITH_EYE))
			{
				// we need to update the database for the resulting image asset

				int movie_asset_id = movie_asset_panel->ReturnAssetID(per_row_array_position[row]);
				int image_asset = image_asset_panel->ReturnArrayPositionFromParentID(movie_asset_id);
				int image_asset_id = image_asset_panel->ReturnAssetID(image_asset);
				int alignment_job_id = alignment_job_ids[column - 2];

				MyDebugAssertTrue(image_asset >= 0, "Something went wrong finding an image asset");

				// we need to get the details of the selected movie alignment, and update the image asset.

				int alignment_id;
				wxString output_file;
				bool should_continue;

				should_continue = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT ALIGNMENT_ID, OUTPUT_FILE FROM MOVIE_ALIGNMENT_LIST WHERE MOVIE_ASSET_ID=%i AND ALIGNMENT_JOB_ID=%i", movie_asset_id, alignment_job_id));

				if (should_continue == false)
				{
					MyPrintWithDetails("Error getting information about alignment!")
					abort();
				}

				main_frame->current_project.database.GetFromBatchSelect("it", &alignment_id, &output_file);
				main_frame->current_project.database.EndBatchSelect();
//				main_frame->current_project.database.InsertOrReplace("IMAGE_ASSETS", "itiiiiirrr", "IMAGE_ASSET_ID", "FILENAME", "POSITION_IN_STACK", "PARENT_MOVIE_ID", "ALIGNMENT_ID", "X_SIZE", "Y_SIZE", "PIXEL_SIZE", "VOLTAGE", "SPHERICAL_ABERRATION", image_asset_id, output_file.ToUTF8().data(), 1, image_asset_panel->ReturnAssetPointer(image_asset)->parent_id,  alignment_id, image_asset_panel->ReturnAssetPointer(image_asset)->x_size, image_asset_panel->ReturnAssetPointer(image_asset)->y_size, image_asset_panel->ReturnAssetPointer(image_asset)->pixel_size, image_asset_panel->ReturnAssetPointer(image_asset)->microscope_voltage, image_asset_panel->ReturnAssetPointer(image_asset)->spherical_aberration);


				main_frame->current_project.database.BeginImageAssetInsert();
				main_frame->current_project.database.AddNextImageAsset(image_asset_id, image_asset_panel->ReturnAssetPointer(image_asset)->asset_name, output_file.ToUTF8().data(), image_asset_panel->ReturnAssetPointer(image_asset)->position_in_stack, image_asset_panel->ReturnAssetPointer(image_asset)->parent_id,  alignment_id, image_asset_panel->ReturnAssetPointer(image_asset)->ctf_estimation_id, image_asset_panel->ReturnAssetPointer(image_asset)->x_size, image_asset_panel->ReturnAssetPointer(image_asset)->y_size, image_asset_panel->ReturnAssetPointer(image_asset)->microscope_voltage, image_asset_panel->ReturnAssetPointer(image_asset)->pixel_size, image_asset_panel->ReturnAssetPointer(image_asset)->spherical_aberration);
				main_frame->current_project.database.EndImageAssetInsert();

				image_asset_panel->ReturnAssetPointer(image_asset)->filename = output_file;
				image_asset_panel->ReturnAssetPointer(image_asset)->alignment_id = alignment_id;
				image_asset_panel->is_dirty = true;


			}
		}
	}
}

int MyMovieAlignResultsPanel::GetFilter()
{
	MyMovieFilterDialog *filter_dialog = new MyMovieFilterDialog(this);

	// set initial settings..


	// show modal

	if (filter_dialog->ShowModal() == wxID_OK)
	{
		//wxPrintf("Command = %s\n", filter_dialog->search_command);
		FillBasedOnSelectCommand( filter_dialog->search_command);

		filter_dialog->Destroy();
		return wxID_OK;
	}
	else return wxID_CANCEL;

}

void MyMovieAlignResultsPanel::FillGroupComboBox()
{
	GroupComboBox->FillWithMovieGroups(false);
}



void MyMovieAlignResultsPanel::OnAllMoviesSelect( wxCommandEvent& event )
{
	FillBasedOnSelectCommand("SELECT DISTINCT MOVIE_ASSET_ID FROM MOVIE_ALIGNMENT_LIST");
}

void MyMovieAlignResultsPanel::OnByFilterSelect( wxCommandEvent& event )
{
	if (GetFilter() == wxID_CANCEL)
	{
		AllMoviesButton->SetValue(true);
	}
}

void MyMovieAlignResultsPanel::OnDefineFilterClick( wxCommandEvent& event )
{
	GetFilter();
}

void MyMovieAlignResultsPanel::OnNextButtonClick( wxCommandEvent& event )
{
	ResultDataView->NextEye();
}

void MyMovieAlignResultsPanel::OnPreviousButtonClick( wxCommandEvent& event )
{
	ResultDataView->PreviousEye();
}

void MyMovieAlignResultsPanel::OnAddToGroupClick( wxCommandEvent& event )
{
	movie_asset_panel->AddArrayItemToGroup(GroupComboBox->GetSelection() + 1, per_row_array_position[selected_row]);
}

void MyMovieAlignResultsPanel::OnAddAllToGroupClick( wxCommandEvent& event )
{
	wxArrayLong items_to_add;

	for (long counter = 0; counter < ResultDataView->GetItemCount(); counter++)
	{
		items_to_add.Add(per_row_array_position[counter]);

	}
	OneSecondProgressDialog *progress_bar = new OneSecondProgressDialog("Add all to group", "Adding all to group", ResultDataView->GetItemCount(), this, wxPD_APP_MODAL);
	movie_asset_panel->AddArrayofArrayItemsToGroup(GroupComboBox->GetSelection() + 1, &items_to_add, progress_bar);
	progress_bar->Destroy();


}

void MyMovieAlignResultsPanel::DrawCurveAndFillDetails(int row, int column)
{
	bool should_continue;

	// get the correct result from the database..

	int current_movie_id = per_row_asset_id[row];
	int current_alignment_job_id = alignment_job_ids[column - 2];
	MovieAsset *current_asset = movie_asset_panel->ReturnAssetPointer(movie_asset_panel->ReturnArrayPositionFromAssetID(current_movie_id));

	float current_dose_per_frame = current_asset->dose_per_frame;

	double current_x_shift;
	double current_y_shift;

	int frame_counter;

	int alignment_id;
	long datetime_of_run;
	wxString output_file;
	double voltage;
	double pixel_size;
	double exposure_per_frame;
	double pre_exposure_amount;
	double min_shift;
	double max_shift;
	int should_dose_filter;
	int should_restore_power;
	double termination_threshold;
	int max_iterations;
	int bfactor;
	int should_mask_central_cross;
	int horizontal_mask;
	int vertical_mask;

	// get the alignment_id and all the other details..;

	should_continue = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT * FROM MOVIE_ALIGNMENT_LIST WHERE MOVIE_ASSET_ID=%i AND ALIGNMENT_JOB_ID=%i", current_movie_id, current_alignment_job_id));

	if (should_continue == false)
	{
		MyPrintWithDetails("Error getting information about alignment!")
		abort();
	}

	main_frame->current_project.database.GetFromBatchSelect("iliitrrrrrriiriiiii", &alignment_id, &datetime_of_run, &current_alignment_job_id, &current_movie_id, &output_file, &voltage, &pixel_size, &exposure_per_frame, &pre_exposure_amount, &min_shift, &max_shift, &should_dose_filter, &should_restore_power, &termination_threshold, &max_iterations, &bfactor, &should_mask_central_cross, &horizontal_mask, &vertical_mask);
	main_frame->current_project.database.EndBatchSelect();

	RightPanel->Freeze();

	// Set the appropriate text..

	AlignmentIDStaticText->SetLabel(wxString::Format("%i", alignment_id));
	wxDateTime wxdatetime_of_run;
	wxdatetime_of_run.SetFromDOS((unsigned long) datetime_of_run);
	DateOfRunStaticText->SetLabel(wxdatetime_of_run.FormatISODate());
	TimeOfRunStaticText->SetLabel(wxdatetime_of_run.FormatISOTime());
	VoltageStaticText->SetLabel(wxString::Format(wxT("%.2f kV"), voltage));
	PixelSizeStaticText->SetLabel(wxString::Format(wxT("%.4f Å"), pixel_size));
	BfactorStaticText->SetLabel(wxString::Format(wxT("%i Å²"), bfactor));
	ExposureStaticText->SetLabel(wxString::Format(wxT("%.2f e¯/Å²"), exposure_per_frame));
	PreExposureStaticText->SetLabel(wxString::Format(wxT("%.2f e¯/Å²"), pre_exposure_amount));
	MinShiftStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), min_shift));
	MaxShiftStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), max_shift));
	TerminationThresholdStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), termination_threshold));
	MaxIterationsStaticText->SetLabel(wxString::Format(wxT("%i"), max_iterations));

	if (should_dose_filter == 1) ExposureFilterStaticText->SetLabel("Yes");
	else ExposureFilterStaticText->SetLabel("No");

	if (should_restore_power == 1) RestorePowerStaticText->SetLabel("Yes");
	else RestorePowerStaticText->SetLabel("No");

	if (should_mask_central_cross == 1) MaskCrossStaticText->SetLabel("Yes");
	else MaskCrossStaticText->SetLabel("No");

	HorizontalMaskStaticText->SetLabel(wxString::Format("%i px", horizontal_mask));
	VerticalMaskStaticText->SetLabel(wxString::Format("%i px", vertical_mask));

	// now get the result, and draw it as we go..


	ResultPanel->ClearGraph();

	should_continue = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT * FROM MOVIE_ALIGNMENT_PARAMETERS_%i", alignment_id));

	if (should_continue == false)
	{
		MyPrintWithDetails("Error getting alignment result!")
		abort();
	}

	while (should_continue == true)
	{
		should_continue = main_frame->current_project.database.GetFromBatchSelect("irr", &frame_counter, &current_x_shift, &current_y_shift);
		ResultPanel->AddPoint(frame_counter * current_dose_per_frame, current_x_shift, current_y_shift);
	}

	main_frame->current_project.database.EndBatchSelect();

	float current_corrected_pixel_size = current_asset->pixel_size / current_asset->output_binning_factor;
	// if we corrected a mag distortion, we have to adjust the pixel size appropriately.

	if (current_asset->correct_mag_distortion == true) // correct mag distortion
	{
		current_corrected_pixel_size = ReturnMagDistortionCorrectedPixelSize(current_corrected_pixel_size, current_asset->mag_distortion_major_scale, current_asset->mag_distortion_minor_scale);
	}

	float current_nyquist;

	if (current_corrected_pixel_size < 1.4) current_nyquist = 2.8;
	else current_nyquist = current_corrected_pixel_size * 2.0;

	ResultPanel->SpectraNyquistStaticText->SetLabel(wxString::Format(wxT("%.2f Å)   "), current_nyquist));

	wxString amplitude_spectrum_filename = main_frame->current_project.image_asset_directory.GetFullPath();;
	amplitude_spectrum_filename += wxString::Format("/Spectra/%s", wxFileName(output_file).GetFullName());

	ResultPanel->FilenameStaticText->SetLabel(wxString::Format("(%s)"        , wxFileName(output_file).GetFullName()));

	// draw..

	if (DoesFileExist(amplitude_spectrum_filename) == true)
	{
		ResultPanel->SpectraPanel->PanelImage.QuickAndDirtyReadSlice(amplitude_spectrum_filename.ToStdString(), 1);
		ResultPanel->SpectraPanel->should_show = true;
		ResultPanel->SpectraPanel->Refresh();
	}
	else
	{
		ResultPanel->SpectraPanel->should_show = false;
		ResultPanel->SpectraPanel->Refresh();
	}



	ResultPanel->Draw();

	wxString small_image_filename = main_frame->current_project.image_asset_directory.GetFullPath();;
	small_image_filename += wxString::Format("/Scaled/%s", wxFileName(output_file).GetFullName());

	if (DoesFileExist(small_image_filename) == true)
	{
		ResultPanel->ImageDisplayPanel->ChangeFile(small_image_filename, "");
	}
	else
	if (DoesFileExist(output_file) == true)
	{
		ResultPanel->ImageDisplayPanel->ChangeFile(output_file, "");
	}



	RightPanel->Layout();
	RightPanel->Thaw();

}

void MyMovieAlignResultsPanel::Clear()
{
	selected_row = -1;
	selected_column = -1;

	ResultDataView->Clear();
	ResultPanel->Clear();
}

/*
void MyMovieAlignResultsPanel::OnShowTypeRadioBoxChange(wxCommandEvent& event)
{
	wxPrintf("Changed\n");

}*/

int MyMovieAlignResultsPanel::ReturnRowFromAssetID(int asset_id, int start_location)
{
	int counter;

	for (counter = start_location; counter < number_of_assets; counter++)
	{
		if (per_row_asset_id[counter] == asset_id) return counter;
	}

	// if we got here, we should do the begining..

	for (counter = 0; counter < start_location; counter++)
	{
		if (per_row_asset_id[counter] == asset_id) return counter;
	}

	return -1;
}

void MyMovieAlignResultsPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	if (main_frame->current_project.is_open == false)
	{
		Enable(false);
	}
	else
	{
		Enable(true);

		if (ByFilterButton->GetValue() == true)
		{
			FilterButton->Enable(true);
		}
		else
		{
			FilterButton->Enable(false);
		}

		if (GroupComboBox->GetCount() > 0 && ResultDataView->GetItemCount() > 0)
		{
			AddToGroupButton->Enable(true);
			AddAllToGroupButton->Enable(true);
		}
		else
		{
			AddToGroupButton->Enable(false);
			AddAllToGroupButton->Enable(false);
		}

		if (is_dirty == true)
		{
			is_dirty = false;
			FillBasedOnSelectCommand(current_fill_command);
		}

		if (group_combo_is_dirty == true)
		{
			FillGroupComboBox();
			group_combo_is_dirty = false;
		}



	}
}

void MyMovieAlignResultsPanel::FillBasedOnSelectCommand(wxString wanted_command)
{
	wxVector<wxVariant> data;
	wxVariant temp_variant;
	long asset_counter;
	long job_counter;
	bool should_continue;
	int selected_job_id;
	int current_asset;
	int array_position;
	int current_row;
	int start_from_row;


	// append columns..

	doing_panel_fill = true;
	current_fill_command = wanted_command;

	ResultDataView->Freeze();
	ResultDataView->Clear();

	ResultDataView->AppendTextColumn("ID");//, wxDATAVIEW_CELL_INERT,1, wxALIGN_LEFT, 0);
	ResultDataView->AppendTextColumn("File");//, wxDATAVIEW_CELL_INERT,1, wxALIGN_LEFT,wxDATAVIEW_COL_RESIZABLE);

	//
	// find out how many alignment jobs there are :-

	number_of_alignmnet_jobs = main_frame->current_project.database.ReturnNumberOfAlignmentJobs();

	// cache the various  alignment_job_ids

	if (alignment_job_ids != NULL) delete [] alignment_job_ids;
	alignment_job_ids = new int[number_of_alignmnet_jobs];

	main_frame->current_project.database.GetUniqueAlignmentIDs(alignment_job_ids, number_of_alignmnet_jobs);

	// retrieve their ids

	for (job_counter = 0; job_counter < number_of_alignmnet_jobs; job_counter++)
	{
		ResultDataView->AppendCheckColumn(wxString::Format("#%i", alignment_job_ids[job_counter]));
	}

	// assign memory to the maximum..

	if (per_row_asset_id != NULL) delete [] per_row_asset_id;
	if (per_row_array_position != NULL) delete [] per_row_array_position;

	per_row_asset_id = new int[movie_asset_panel->ReturnNumberOfAssets()];
	per_row_array_position = new int[movie_asset_panel->ReturnNumberOfAssets()];

	// execute the select command, to retrieve all the ids..

	number_of_assets = 0;
	should_continue = main_frame->current_project.database.BeginBatchSelect(wanted_command);

	if (should_continue == true)
	{
		while(should_continue == true)
		{
			should_continue = main_frame->current_project.database.GetFromBatchSelect("i", &current_asset);
			array_position = movie_asset_panel->ReturnArrayPositionFromAssetID(current_asset);

			if (array_position < 0 || current_asset < 0)
			{
				MyPrintWithDetails("Error: Something wrong finding asset %i, skipping", current_asset);
			}
			else
			{
				per_row_asset_id[number_of_assets] = current_asset;
				per_row_array_position[number_of_assets] = array_position;
				number_of_assets++;

			}


		}

		main_frame->current_project.database.EndBatchSelect();

		// now we know which movies are included, and their order.. draw the dataviewlistctrl

		for (asset_counter = 0; asset_counter < number_of_assets; asset_counter++)
		{
			data.clear();
			data.push_back(wxVariant(wxString::Format("%i", per_row_asset_id[asset_counter])));
			data.push_back(wxVariant(movie_asset_panel->ReturnAssetShortFilename(per_row_array_position[asset_counter])));

			for (job_counter = 0; job_counter < number_of_alignmnet_jobs; job_counter++)
			{
				data.push_back(wxVariant(long(-1)));
			}

			ResultDataView->AppendItem( data );
		}

		// all assets should be added.. now go job by job and fill the appropriate columns..


		for (job_counter = 0; job_counter < number_of_alignmnet_jobs; job_counter++)
		{
			should_continue = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT MOVIE_ASSET_ID FROM MOVIE_ALIGNMENT_LIST WHERE ALIGNMENT_JOB_ID=%i", alignment_job_ids[job_counter]));

			if (should_continue == false)
			{
				MyPrintWithDetails("Error getting alignment jobs..");
				abort();
			}

			start_from_row = 0;

			while(1==1)
			{
				should_continue = main_frame->current_project.database.GetFromBatchSelect("i", &current_asset);
				current_row = ReturnRowFromAssetID(current_asset, start_from_row);

				if (current_row != -1)
				{
					ResultDataView->SetValue(wxVariant(UNCHECKED), current_row, 2 + job_counter);
					start_from_row = current_row;
				}

				if (should_continue == false) break;
			}

			main_frame->current_project.database.EndBatchSelect();

		}

		//SELECT MOVIE_ASSET_ID, ALIGNMENT_JOB_ID FROM MOVIE_ALIGNMENT_LIST, IMAGE_ASSETS WHERE MOVIE_ALIGNMENT_LIST.ALIGNMENT_ID=IMAGE_ASSETS.ALIGNMENT_ID AND IMAGE_ASSETS.PARENT_MOVIE_ID=MOVIE_ALIGNMENT_LIST.MOVIE_ASSET_ID;

		// set the checked ones..

		should_continue = main_frame->current_project.database.BeginBatchSelect("SELECT MOVIE_ASSET_ID, ALIGNMENT_JOB_ID FROM MOVIE_ALIGNMENT_LIST, IMAGE_ASSETS WHERE MOVIE_ALIGNMENT_LIST.ALIGNMENT_ID=IMAGE_ASSETS.ALIGNMENT_ID AND IMAGE_ASSETS.PARENT_MOVIE_ID=MOVIE_ALIGNMENT_LIST.MOVIE_ASSET_ID;");

		if (should_continue == false)
		{
			MyPrintWithDetails("Error getting selected alignments..");
			abort();
		}

		start_from_row = 0;

		while(1==1)
		{
			should_continue = main_frame->current_project.database.GetFromBatchSelect("ii", &current_asset, &selected_job_id);
			current_row = ReturnRowFromAssetID(current_asset, start_from_row);

			if (current_row != -1)
			{
				start_from_row = current_row;

				for (job_counter = 0; job_counter < number_of_alignmnet_jobs; job_counter++)
				{
					if (alignment_job_ids[job_counter] == selected_job_id)
					{
						ResultDataView->SetValue(wxVariant(CHECKED), current_row, 2 + job_counter);
						break;
					}
				}
			}

			if (should_continue == false) break;
		}

		main_frame->current_project.database.EndBatchSelect();

		// select the first row..
		doing_panel_fill = false;
		if (number_of_assets > 0)
		{
			ResultDataView->ChangeDisplayTo(0, ResultDataView->ReturnCheckedColumn(0));
			ResultDataView->EnsureVisible(ResultDataView->RowToItem(-1), ResultDataView->GetColumn(ResultDataView->ReturnCheckedColumn(0)));
		}
		ResultDataView->SizeColumns();
		ResultDataView->Thaw();
	}
	else
	{
		main_frame->current_project.database.EndBatchSelect();
	}



}

void MyMovieAlignResultsPanel::OnJobDetailsToggle( wxCommandEvent& event )
{
	Freeze();

	if (JobDetailsToggleButton->GetValue() == true)
	{
		JobDetailsPanel->Show(true);
	}
	else
	{
		JobDetailsPanel->Show(false);
	}


	RightPanel->Layout();
	Thaw();


}

