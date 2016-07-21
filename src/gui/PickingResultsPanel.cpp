//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMainFrame *main_frame;
extern MyImageAssetPanel *image_asset_panel;
extern MyParticlePositionAssetPanel *particle_position_asset_panel;

MyPickingResultsPanel::MyPickingResultsPanel( wxWindow* parent )
:
PickingResultsPanel( parent )
{


	Bind(wxEVT_DATAVIEW_ITEM_VALUE_CHANGED, wxDataViewEventHandler( MyPickingResultsPanel::OnValueChanged), this);

	picking_job_ids = NULL;
	number_of_picking_jobs = 0;
	per_row_asset_id = NULL;
	per_row_array_position = NULL;
	number_of_assets = 0;

	selected_row = -1;
	selected_column = -1;
	doing_panel_fill = false;

	current_fill_command = "SELECT PARENT_IMAGE_ASSET_ID FROM PARTICLE_PICKING_LIST";
	is_dirty=false;
	group_combo_is_dirty=false;

	FillGroupComboBox();

}

void MyPickingResultsPanel::FillGroupComboBox()
{
	GroupComboBox->Freeze();
	GroupComboBox->ChangeValue("");
	GroupComboBox->Clear();

	for (long counter = 1; counter < image_asset_panel->ReturnNumberOfGroups(); counter++)
	{
		GroupComboBox->Append(image_asset_panel->ReturnGroupName(counter) +  " (" + wxString::Format(wxT("%li"), image_asset_panel->ReturnGroupSize(counter)) + ")");

	}

	if (GroupComboBox->GetCount() > 0) GroupComboBox->SetSelection(0);

	GroupComboBox->Thaw();

}

void MyPickingResultsPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	if ( ! main_frame->current_project.is_open)
	{
		Enable(false);
	}
	else
	{
		Enable(true);

		FilterButton->Enable(ByFilterButton->GetValue());

		if (GroupComboBox->GetCount() > 0)
		{
			AddToGroupButton->Enable(true);
		}
		else AddToGroupButton->Enable(false);

		if (is_dirty)
		{
			is_dirty = false;
			FillBasedOnSelectCommand(current_fill_command);
		}

		if (group_combo_is_dirty)
		{
			FillGroupComboBox();
			group_combo_is_dirty = false;
		}



	}
}

void MyPickingResultsPanel::OnAllMoviesSelect( wxCommandEvent& event )
{
	MyDebugAssertTrue(false,"to be written");
	FillBasedOnSelectCommand("SELECT DISTINCT IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS");
}

void MyPickingResultsPanel::OnByFilterSelect( wxCommandEvent& event )
{
	if (GetFilter() == wxID_CANCEL)
	{
		AllImagesButton->SetValue(true);
	}
}

int MyPickingResultsPanel::GetFilter()
{
	MyDebugAssertTrue(false,"to be written");
	/*
	MyPickingFilterDialog *filter_dialog = new MyPickingFilterDialog(this);

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

	*/
}


void MyPickingResultsPanel::FillBasedOnSelectCommand(wxString wanted_command)
{
	wxVector<wxVariant> data;
	wxVariant temp_variant;
	long asset_counter;
	long job_counter;
	bool should_continue;
	int selected_job_id;
	int current_image_asset_id;
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
	// find out how many picking jobs there are :-

	number_of_picking_jobs = main_frame->current_project.database.ReturnNumberOfPickingJobs();

	// cache the various  alignment_job_ids

	if (picking_job_ids != NULL) delete [] picking_job_ids;
	picking_job_ids = new int[number_of_picking_jobs];

	main_frame->current_project.database.GetUniquePickingJobIDs(picking_job_ids, number_of_picking_jobs);

	// retrieve their ids

	for (job_counter = 0; job_counter < number_of_picking_jobs; job_counter++)
	{
		ResultDataView->AppendCheckColumn(wxString::Format("#%i", picking_job_ids[job_counter]));
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
			should_continue = main_frame->current_project.database.GetFromBatchSelect("i", &current_image_asset_id);
			array_position = image_asset_panel->ReturnArrayPositionFromAssetID(current_image_asset_id);

			if (array_position < 0 || current_image_asset_id < 0)
			{
				MyPrintWithDetails("Error: Something wrong finding image asset %i, skipping", current_image_asset_id);
			}
			else
			{
				per_row_asset_id[number_of_assets] = current_image_asset_id;
				per_row_array_position[number_of_assets] = array_position;
				number_of_assets++;

			}


		}

		main_frame->current_project.database.EndBatchSelect();

		// now we know which images are included, and their order.. draw the dataviewlistctrl

		for (asset_counter = 0; asset_counter < number_of_assets; asset_counter++)
		{
			data.clear();
			data.push_back(wxVariant(wxString::Format("%i", per_row_asset_id[asset_counter])));
			data.push_back(wxVariant(image_asset_panel->ReturnAssetShortFilename(per_row_array_position[asset_counter])));

			for (job_counter = 0; job_counter < number_of_picking_jobs; job_counter++)
			{
				data.push_back(wxVariant(long(-1)));
			}

			ResultDataView->AppendItem( data );
		}

		// all assets should be added.. now go job by job and fill the appropriate columns..


		for (job_counter = 0; job_counter < number_of_picking_jobs; job_counter++)
		{
			should_continue = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT PARENT_IMAGE_ASSET_ID FROM PARTICLE_PICKING_LIST WHERE PICKING_JOB_ID=%i", picking_job_ids[job_counter]));

			if (!should_continue)
			{
				MyPrintWithDetails("Error getting alignment jobs..");
				abort();
			}

			start_from_row = 0;

			while(true)
			{
				should_continue = main_frame->current_project.database.GetFromBatchSelect("i", &current_image_asset_id);
				current_row = ReturnRowFromAssetID(current_image_asset_id, start_from_row);

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

		should_continue = main_frame->current_project.database.BeginBatchSelect("SELECT PARENT_IMAGE_ASSET_ID, PICK_JOB_ID FROM PARTICLE_POSITION_ASSETS;");

		if (!should_continue)
		{
			MyPrintWithDetails("Error getting selected alignments..");
			abort();
		}

		start_from_row = 0;

		while(true)
		{
			should_continue = main_frame->current_project.database.GetFromBatchSelect("ii", &current_image_asset_id, &selected_job_id);
			current_row = ReturnRowFromAssetID(current_image_asset_id, start_from_row);

			if (current_row != -1)
			{
				start_from_row = current_row;

				for (job_counter = 0; job_counter < number_of_picking_jobs; job_counter++)
				{
					if (picking_job_ids[job_counter] == selected_job_id)
					{
						ResultDataView->SetValue(wxVariant(CHECKED), current_row, 2 + job_counter);
						break;
					}
				}
			}

			if (!should_continue) break;
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

		ResultDataView->Thaw();
	}
	else
	{
		main_frame->current_project.database.EndBatchSelect();
	}



}

int MyPickingResultsPanel::ReturnRowFromAssetID(int asset_id, int start_location)
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

void MyPickingResultsPanel::FillResultsPanelAndDetails(int row, int column)
{
	bool should_continue;

	// get the correct result from the database..

	int current_image_id = per_row_asset_id[row];
	int current_picking_job_id = picking_job_ids[column - 2];
	wxString parent_image_filename;

	bool keep_going = main_frame->current_project.database.BeginBatchSelect(wxString::Format("select filename from image_assets where image_asset_id = %i",current_image_id));
	if (!keep_going)
	{
		MyPrintWithDetails("Error dealing with assets table");
		abort();
	}
	main_frame->current_project.database.GetFromBatchSelect("t",&parent_image_filename);
	main_frame->current_project.database.EndBatchSelect();

	int number_of_particles = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(wxString::Format("select count(*) from particle_picking_results_%i where parent_image_asset_id = %i",current_picking_job_id,current_image_id));
	double *x_coordinates;
	double *y_coordinates;
	x_coordinates = new double[number_of_particles];
	y_coordinates = new double[number_of_particles];

	float maximum_radius_of_particle = main_frame->current_project.database.ReturnSingleDoubleFromSelectCommand(wxString::Format("select maximum_radius from particle_picking_list where picking_job_id = %i",current_picking_job_id));
	float pixel_size = main_frame->current_project.database.ReturnSingleDoubleFromSelectCommand(wxString::Format("select pixel_size from image_assets where image_asset_id = %i",current_image_id));

	if (number_of_particles > 0)
	{
		keep_going = main_frame->current_project.database.BeginBatchSelect(wxString::Format("select x_position, y_position from particle_picking_results_%i where parent_image_asset_id = %i",current_picking_job_id,current_image_id));
		if (!keep_going)
		{
			MyPrintWithDetails("Error dealing with results table");
			abort();
		}
		for (int counter = 0; counter < number_of_particles; counter ++ )
		{
			main_frame->current_project.database.GetFromBatchSelect("rr",&x_coordinates[counter],&y_coordinates[counter]);
		}
		main_frame->current_project.database.EndBatchSelect();
	}

	ResultDisplayPanel->Draw(parent_image_filename,number_of_particles,x_coordinates,y_coordinates,maximum_radius_of_particle, pixel_size);
	RightPanel->Layout();

}

void MyPickingResultsPanel::OnValueChanged(wxDataViewEvent &event)
{

	if (!doing_panel_fill)
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

				// First remove from particle_position assets any assets with parent_image_asset_id corresponding to the image of the current row
				for (int group_counter = 1; group_counter < particle_position_asset_panel->all_groups_list->number_of_groups; group_counter++)
				{
					main_frame->current_project.database.RemoveParticlePositionsWithGivenParentImageIDFromGroup(group_counter,per_row_asset_id[row]);
				}
				main_frame->current_project.database.RemoveParticlePositionAssetsPickedFromImageWithGivenID(per_row_asset_id[row]);

				// Now, add particle position assets from the relevant results table which have the correct parent_image_asset_id
				main_frame->current_project.database.CopyParticleAssetsFromResultsTable(picking_job_ids[column - 2],per_row_asset_id[row]);


				particle_position_asset_panel->ImportAllFromDatabase();
				particle_position_asset_panel->is_dirty = true;

			}
		}
	}
}

void MyPickingResultsPanel::OnNextButtonClick( wxCommandEvent& event )
{
	ResultDataView->NextEye();
}

void MyPickingResultsPanel::OnPreviousButtonClick( wxCommandEvent& event )
{
	ResultDataView->PreviousEye();
}

void MyPickingResultsPanel::Clear()
{
	selected_row = -1;
	selected_column = -1;

	ResultDataView->Clear();
}

void MyPickingResultsPanel::OnJobDetailsToggle( wxCommandEvent& event )
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

void MyPickingResultsPanel::OnAddToGroupClick( wxCommandEvent& event )
{
	image_asset_panel->AddArrayItemToGroup(GroupComboBox->GetCurrentSelection() + 1, per_row_array_position[selected_row]);

}

void MyPickingResultsPanel::OnDefineFilterClick( wxCommandEvent& event )
{
	GetFilter();
}


