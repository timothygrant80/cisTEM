//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMainFrame *main_frame;
extern MyImageAssetPanel *image_asset_panel;
extern MyMovieAssetPanel *movie_asset_panel;

MyFindCTFResultsPanel::MyFindCTFResultsPanel( wxWindow* parent )
:
FindCTFResultsPanel( parent )
{


	Bind(wxEVT_DATAVIEW_ITEM_VALUE_CHANGED, wxDataViewEventHandler( MyFindCTFResultsPanel::OnValueChanged), this);

	ctf_estimation_job_ids = NULL;
	number_of_ctf_estimation_jobs = 0;
	per_row_asset_id = NULL;
	per_row_array_position = NULL;
	number_of_assets = 0;

	selected_row = -1;
	selected_column = -1;
	doing_panel_fill = false;

	current_fill_command = "SELECT IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS";
	is_dirty=false;
	group_combo_is_dirty=false;

	FillGroupComboBox();

	Bind(wxEVT_CHAR_HOOK, &MyFindCTFResultsPanel::OnCharHook, this);
}

void MyFindCTFResultsPanel::OnCharHook( wxKeyEvent& event )
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

void MyFindCTFResultsPanel::FillGroupComboBox()
{
	GroupComboBox->FillWithImageGroups(false);
}

void MyFindCTFResultsPanel::OnUpdateUI( wxUpdateUIEvent& event )
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
			DeleteFromGroupButton->Enable(true);
			AddAllToGroupButton->Enable(true);
		}
		else
		{
			AddToGroupButton->Enable(false);
			DeleteFromGroupButton->Enable(false);
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

void MyFindCTFResultsPanel::OnAllMoviesSelect( wxCommandEvent& event )
{
	FillBasedOnSelectCommand("SELECT DISTINCT IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS");
}

void MyFindCTFResultsPanel::OnByFilterSelect( wxCommandEvent& event )
{
	if (GetFilter() == wxID_CANCEL)
	{
		AllImagesButton->SetValue(true);
	}
}

int MyFindCTFResultsPanel::GetFilter()
{
	MyCTFFilterDialog *filter_dialog = new MyCTFFilterDialog(this);

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


void MyFindCTFResultsPanel::FillBasedOnSelectCommand(wxString wanted_command)
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

	Freeze();
	Clear();

	ResultDataView->AppendTextColumn("ID");//, wxDATAVIEW_CELL_INERT,1, wxALIGN_LEFT, 0);
	ResultDataView->AppendTextColumn("File");//, wxDATAVIEW_CELL_INERT,1, wxALIGN_LEFT,wxDATAVIEW_COL_RESIZABLE);

	//
	// find out how many alignment jobs there are :-

	number_of_ctf_estimation_jobs = main_frame->current_project.database.ReturnNumberOfCTFEstimationJobs();

	// cache the various  alignment_job_ids

	if (ctf_estimation_job_ids != NULL) delete [] ctf_estimation_job_ids;
	ctf_estimation_job_ids = new int[number_of_ctf_estimation_jobs];

	main_frame->current_project.database.GetUniqueCTFEstimationIDs(ctf_estimation_job_ids, number_of_ctf_estimation_jobs);

	// retrieve their ids

	for (job_counter = 0; job_counter < number_of_ctf_estimation_jobs; job_counter++)
	{
		ResultDataView->AppendCheckColumn(wxString::Format("#%i", ctf_estimation_job_ids[job_counter]));
	}

	// assign memory to the maximum..

	if (per_row_asset_id != NULL) delete [] per_row_asset_id;
	if (per_row_array_position != NULL) delete [] per_row_array_position;

	per_row_asset_id = new int[image_asset_panel->ReturnNumberOfAssets()];
	per_row_array_position = new int[image_asset_panel->ReturnNumberOfAssets()];

	// execute the select command, to retrieve all the ids..

	number_of_assets = 0;
	should_continue = main_frame->current_project.database.BeginBatchSelect(wanted_command);

	if (should_continue == true)
	{
		while(should_continue == true)
		{
			should_continue = main_frame->current_project.database.GetFromBatchSelect("i", &current_asset);
			array_position = image_asset_panel->ReturnArrayPositionFromAssetID(current_asset);

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
			data.push_back(wxVariant(image_asset_panel->ReturnAssetShortFilename(per_row_array_position[asset_counter])));

			for (job_counter = 0; job_counter < number_of_ctf_estimation_jobs; job_counter++)
			{
				data.push_back(wxVariant(long(-1)));
			}

			ResultDataView->AppendItem( data );
		}

		// all assets should be added.. now go job by job and fill the appropriate columns..


		for (job_counter = 0; job_counter < number_of_ctf_estimation_jobs; job_counter++)
		{
			should_continue = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS WHERE CTF_ESTIMATION_JOB_ID=%i", ctf_estimation_job_ids[job_counter]));

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

		// set the checked ones..

		should_continue = main_frame->current_project.database.BeginBatchSelect("SELECT IMAGE_ASSETS.IMAGE_ASSET_ID, CTF_ESTIMATION_JOB_ID FROM ESTIMATED_CTF_PARAMETERS, IMAGE_ASSETS WHERE ESTIMATED_CTF_PARAMETERS.CTF_ESTIMATION_ID=IMAGE_ASSETS.CTF_ESTIMATION_ID AND IMAGE_ASSETS.IMAGE_ASSET_ID=ESTIMATED_CTF_PARAMETERS.IMAGE_ASSET_ID;");

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

				for (job_counter = 0; job_counter < number_of_ctf_estimation_jobs; job_counter++)
				{
					if (ctf_estimation_job_ids[job_counter] == selected_job_id)
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

		selected_column = -1;
		selected_row = -1;

		if (number_of_assets > 0)
		{
			ResultDataView->ChangeDisplayTo(0, ResultDataView->ReturnCheckedColumn(0));

		}
		ResultDataView->SizeColumns();
	}
	else
	{
		main_frame->current_project.database.EndBatchSelect();
	}

	Thaw();



}

int MyFindCTFResultsPanel::ReturnRowFromAssetID(int asset_id, int start_location)
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

void MyFindCTFResultsPanel::FillResultsPanelAndDetails(int row, int column)
{
	bool should_continue;

	// get the correct result from the database..

	int current_image_id = per_row_asset_id[row];
	int current_ctf_estimation_job_id = ctf_estimation_job_ids[column - 2];

	int ctf_estimation_id;
	int ctf_estimation_job_id;
	long datetime_of_run;
	int image_asset_id;
	int estimated_on_movie_frames;
	double voltage;
	double spherical_aberration;
	double pixel_size;
	double amplitude_contrast;
	int box_size;
	double min_resolution;
	double max_resolution;
	double min_defocus;
	double max_defocus;
	double defocus_step;
	int large_astigmatism_expected;
	int restrain_astigmatism;
	double tolerated_astigmatism;
	int find_additional_phase_shift;
	double min_phase_shift;
	double max_phase_shift;
	double phase_shift_step;
	double defocus1;
	double defocus2;
	double defocus_angle;
	double additional_phase_shift;
	double score;
	double detected_ring_resolution;
	double detected_alias_resolution;
	wxString output_diagnostic_file;
	int number_of_frames_averaged;
	//wxString input_filename;

	// get the alignment_id and all the other details..;

	should_continue = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT * FROM ESTIMATED_CTF_PARAMETERS WHERE IMAGE_ASSET_ID=%i AND CTF_ESTIMATION_JOB_ID=%i", current_image_id, current_ctf_estimation_job_id));

	//wxPrintf("SELECT * FROM ESTIMATED_CTF_PARAMETERS WHERE IMAGE_ASSET_ID=%i AND CTF_ESTIMATION_JOB_ID=%i\n", current_image_id, current_ctf_estimation_job_id);

	if (should_continue == false)
	{
		MyPrintWithDetails("Error getting information about alignment!")
		abort();
	}

	main_frame->current_project.database.GetFromBatchSelect("iiliirrrrirrrrririrrrrrrrrrrtii", &ctf_estimation_id, &ctf_estimation_job_id,&datetime_of_run, &image_asset_id, &estimated_on_movie_frames, &voltage, &spherical_aberration, &pixel_size, &amplitude_contrast, &box_size, &min_resolution, &max_resolution, &min_defocus, &max_defocus, &defocus_step, &restrain_astigmatism, &tolerated_astigmatism, &find_additional_phase_shift, &min_phase_shift, &max_phase_shift, &phase_shift_step, &defocus1, &defocus2, &defocus_angle, &additional_phase_shift, &score, &detected_ring_resolution, &detected_alias_resolution, &output_diagnostic_file, &number_of_frames_averaged, &large_astigmatism_expected);
	//wxPrintf("%i,%i,%li,%i,%i,%f,%f,%f,%f,%i,%f,%f,%f,%f,%f,%i,%f,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s\n", ctf_estimation_id, ctf_estimation_job_id,datetime_of_run, image_asset_id, estimated_on_movie_frames, voltage, spherical_aberration, pixel_size, amplitude_contrast, box_size, min_resolution, max_resolution, min_defocus, max_defocus, defocus_step, restrain_astigmatism, tolerated_astigmatism, find_additional_phase_shift, min_phase_shift, max_phase_shift, phase_shift_step, defocus1, defocus2, defocus_angle, additional_phase_shift, score, detected_ring_resolution, detected_alias_resolution, output_diagnostic_file);
	main_frame->current_project.database.EndBatchSelect();

	// Set the appropriate text..

	EstimationIDStaticText->SetLabel(wxString::Format("%i", ctf_estimation_id));
	wxDateTime wxdatetime_of_run;
	wxdatetime_of_run.SetFromDOS((unsigned long) datetime_of_run);
	DateOfRunStaticText->SetLabel(wxdatetime_of_run.FormatISODate());
	TimeOfRunStaticText->SetLabel(wxdatetime_of_run.FormatISOTime());
	VoltageStaticText->SetLabel(wxString::Format(wxT("%.2f kV"), voltage));
	CsStaticText->SetLabel(wxString::Format(wxT("%.2f mm"), spherical_aberration));
	PixelSizeStaticText->SetLabel(wxString::Format(wxT("%.4f Å"), pixel_size));
	AmplitudeContrastStaticText->SetLabel(wxString::Format(wxT("%.2f"), amplitude_contrast));
	BoxSizeStaticText->SetLabel(wxString::Format(wxT("%i"), box_size));
	MinResStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), min_resolution));
	MaxResStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), max_resolution));
	MinDefocusStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), min_defocus));
	MaxDefocusStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), max_defocus));
	DefocusStepStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), defocus_step));
	if (estimated_on_movie_frames == 1)
	{
		NumberOfAveragedFramesStaticText->SetLabel(wxString::Format(wxT("%i"), number_of_frames_averaged));
	}
	else
	{
		NumberOfAveragedFramesStaticText->SetLabel("n/a");
	}


	if (large_astigmatism_expected == 1)
	{
		LargeAstigExpectedStaticText->SetLabel("Yes");
	}
	else
	{
		LargeAstigExpectedStaticText->SetLabel("No");
	}

	if (restrain_astigmatism == 1)
	{
		RestrainAstigStaticText->SetLabel("Yes");
		ToleratedAstigLabel->Show(true);
		ToleratedAstigStaticText->Show(true);
		ToleratedAstigStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), tolerated_astigmatism));

	}
	else
	{
		RestrainAstigStaticText->SetLabel("No");
		RestrainAstigStaticText->SetLabel("Yes");
		ToleratedAstigLabel->Show(false);
		ToleratedAstigStaticText->Show(false);
	}

	if (find_additional_phase_shift == 1)
	{
		AddtionalPhaseShiftStaticText->SetLabel("Yes");
		MinPhaseShiftStaticText->Show(true);
		MinPhaseShiftLabel->Show(true);
		MaxPhaseshiftStaticText->Show(true);
		MaxPhaseShiftLabel->Show(true);
		PhaseShiftStepStaticText->Show(true);
		PhaseShiftStepLabel->Show(true);

		MinPhaseShiftStaticText->SetLabel(wxString::Format(wxT("%.2f °"), rad_2_deg(min_phase_shift)));
		MaxPhaseshiftStaticText->SetLabel(wxString::Format(wxT("%.2f °"), rad_2_deg(max_phase_shift)));
		PhaseShiftStepStaticText->SetLabel(wxString::Format(wxT("%.2f °"), rad_2_deg(phase_shift_step)));

	}
	else
	{
		AddtionalPhaseShiftStaticText->SetLabel("No");
		MinPhaseShiftStaticText->Show(false);
		MinPhaseShiftLabel->Show(false);
		MaxPhaseshiftStaticText->Show(false);
		MaxPhaseShiftLabel->Show(false);
		PhaseShiftStepStaticText->Show(false);
		PhaseShiftStepLabel->Show(false);
	}


	// now get the result, and draw it as we go..

	wxString ImageFile = image_asset_panel->ReturnAssetLongFilename(image_asset_panel->ReturnArrayPositionFromAssetID(current_image_id));
	ResultPanel->Draw(output_diagnostic_file, find_additional_phase_shift, defocus1, defocus2, defocus_angle, additional_phase_shift, score, detected_ring_resolution, detected_alias_resolution, ImageFile);
	RightPanel->Layout();

}

void MyFindCTFResultsPanel::OnValueChanged(wxDataViewEvent &event)
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

			FillResultsPanelAndDetails(row, column);
			//wxPrintf("drawing curve\n");

		}
		else // This is dodgy, and relies on the fact that a box will be deselected, before a new box is selected...
		{
			if ((value == CHECKED  && (selected_row != row || selected_column != column)) || (value == CHECKED_WITH_EYE))
			{
				// we need to update the database for the resulting image asset

				int image_asset_position = per_row_array_position[row];
				int image_asset_id = image_asset_panel->ReturnAssetID(image_asset_position);
				int estimation_job_id = ctf_estimation_job_ids[column - 2];

				MyDebugAssertTrue(image_asset_id >= 0, "Something went wrong finding an image asset");

				// we need to get the details of the selected movie alignment, and update the image asset.

				int estimation_id = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(wxString::Format("SELECT CTF_ESTIMATION_ID FROM ESTIMATED_CTF_PARAMETERS WHERE IMAGE_ASSET_ID=%i AND CTF_ESTIMATION_JOB_ID=%i",image_asset_id, estimation_job_id));
				bool should_continue;

				//should_continue = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT CTF_ESTIMATION_ID FROM ESTIMATED_CTF_PARAMETERS WHERE IMAGE_ASSET_ID=%i AND CTF_ESTIMATION_JOB_ID=%i",image_asset_id, estimation_job_id));

				//if (should_continue == false)
				//{
				//	MyPrintWithDetails("Error getting information about alignment!")
				//	abort();
				//}

				//main_frame->current_project.database.GetFromBatchSelect("it", &alignment_id, &output_file);
				//main_frame->current_project.database.EndBatchSelect();

				main_frame->current_project.database.BeginImageAssetInsert();
				main_frame->current_project.database.AddNextImageAsset(image_asset_id,  image_asset_panel->ReturnAssetPointer(image_asset_position)->asset_name, image_asset_panel->ReturnAssetPointer(image_asset_position)->filename.GetFullPath(),  image_asset_panel->ReturnAssetPointer(image_asset_position)->position_in_stack, image_asset_panel->ReturnAssetPointer(image_asset_position)->parent_id,  image_asset_panel->ReturnAssetPointer(image_asset_position)->alignment_id, estimation_id, image_asset_panel->ReturnAssetPointer(image_asset_position)->x_size, image_asset_panel->ReturnAssetPointer(image_asset_position)->y_size, image_asset_panel->ReturnAssetPointer(image_asset_position)->microscope_voltage, image_asset_panel->ReturnAssetPointer(image_asset_position)->pixel_size, image_asset_panel->ReturnAssetPointer(image_asset_position)->spherical_aberration, image_asset_panel->ReturnAssetPointer(image_asset_position)->protein_is_white);
				main_frame->current_project.database.EndImageAssetInsert();
				image_asset_panel->ReturnAssetPointer(image_asset_position)->ctf_estimation_id = estimation_id;
				image_asset_panel->is_dirty = true;


			}
		}
	}
}

void MyFindCTFResultsPanel::OnNextButtonClick( wxCommandEvent& event )
{
	ResultDataView->NextEye();
}

void MyFindCTFResultsPanel::OnPreviousButtonClick( wxCommandEvent& event )
{
	ResultDataView->PreviousEye();
}

void MyFindCTFResultsPanel::Clear()
{
	selected_row = -1;
	selected_column = -1;

	ResultDataView->Clear();
	ResultPanel->Clear();
	ResultPanel->CTF2DResultsPanel->should_show = false;
	ResultPanel->CTF2DResultsPanel->Clear();
	ResultPanel->CTFPlotPanel->Clear();
	ResultPanel->ImageDisplayPanel->Clear();
}

void MyFindCTFResultsPanel::OnJobDetailsToggle( wxCommandEvent& event )
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

void MyFindCTFResultsPanel::OnAddToGroupClick( wxCommandEvent& event )
{
	image_asset_panel->AddArrayItemToGroup(GroupComboBox->GetSelection() + 1, per_row_array_position[selected_row]);
}

void MyFindCTFResultsPanel::OnRemoveFromGroupClick( wxCommandEvent& event )
{
	image_asset_panel->DeleteArrayItemFromGroup(GroupComboBox->GetSelection() + 1, per_row_array_position[selected_row]);
}

void MyFindCTFResultsPanel::OnAddAllToGroupClick( wxCommandEvent& event )
{
	wxArrayLong items_to_add;

	for (long counter = 0; counter < ResultDataView->GetItemCount(); counter++)
	{
		items_to_add.Add(per_row_array_position[counter]);

	}
	OneSecondProgressDialog *progress_bar = new OneSecondProgressDialog("Add all to group", "Adding all to group", ResultDataView->GetItemCount(), this, wxPD_APP_MODAL);
	image_asset_panel->AddArrayofArrayItemsToGroup(GroupComboBox->GetSelection() + 1, &items_to_add, progress_bar);
	progress_bar->Destroy();
}


void MyFindCTFResultsPanel::OnDefineFilterClick( wxCommandEvent& event )
{
	GetFilter();
}


