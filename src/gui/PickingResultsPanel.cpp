//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMainFrame *main_frame;
extern MyImageAssetPanel *image_asset_panel;
extern MyParticlePositionAssetPanel *particle_position_asset_panel;
extern MyFindParticlesPanel *findparticles_panel;

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

MyPickingResultsPanel::~MyPickingResultsPanel()
{
	// The destrictor is called when the application is closed, so we need to make sure we've saved any manual edits to the database
	UpdateResultsFromBitmapPanel();
}

void MyPickingResultsPanel::OnProjectOpen()
{
	FillBasedOnSelectCommand("SELECT DISTINCT PARENT_IMAGE_ASSET_ID FROM PARTICLE_PICKING_LIST");
}

void MyPickingResultsPanel::OnProjectClose()
{
	UpdateResultsFromBitmapPanel();
	Clear();
}

void MyPickingResultsPanel::FillGroupComboBox()
{
	FillGroupComboBoxSlave(GroupComboBox,false);
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

		if (should_continue)
		{

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

		}
		else
		{
			MyDebugPrint("No particle position assets\n");
		}

		main_frame->current_project.database.EndBatchSelect();

		// It could be that some assets still don't have any checked boxes, for example
		// because the latest pick job found 0 particles, so that the asset table has no assets for
		// this image.
		// In this case, we want to select the job with the largest pick_job_id
		/*
		wxVariant temp_variant;
				ResultDataView->GetValue(temp_variant, row, column);
				value = temp_variant.GetLong();

				if ((value == CHECKED_WITH_EYE || value == UNCHECKED_WITH_EYE)
				*/
		wxVariant temp_variant;
		bool current_row_has_something_checked;
		int last_unchecked_column = -1;
		long value;
		for (int row = 0; row < number_of_assets; row ++ )
		{
			current_row_has_something_checked = false;
			for (int col = 2; col < number_of_picking_jobs + 2; col ++)
			{
				ResultDataView->GetValue(temp_variant,row,col);
				value = temp_variant.GetLong();
				if (value == CHECKED_WITH_EYE || value == CHECKED) current_row_has_something_checked = true;
				if (value == UNCHECKED) last_unchecked_column = col;
			}
			if (! current_row_has_something_checked) ResultDataView->CheckItem(row,last_unchecked_column);
		}


		// select the first row..
		doing_panel_fill = false;

		selected_column = -1;
		selected_row = -1;

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


void MyPickingResultsPanel::UpdateResultsFromBitmapPanel()
{
	UpdateResultsFromBitmapPanel(ResultDataView->ReturnEyeRow(),ResultDataView->ReturnEyeColumn());
}

// Grab the set of coordinates from the bitmap panel and send them to the database (if the user changed them with the bitmap panel)
void MyPickingResultsPanel::UpdateResultsFromBitmapPanel(const int row, const int column)
{


	// Work out whether the user changed anything in the Bitmap panel since we last saved
	if (ResultDisplayPanel->PickingResultsImagePanel->UserHasEditedParticleCoordinates())
	{
		//wxPrintf("User has edited particle coordinates, let's update the database\n");

		int current_image_id = per_row_asset_id[row];
		int current_picking_job_id = picking_job_ids[column - 2];
		int picking_id = main_frame->current_project.database.ReturnPickingIDGivenPickingJobIDAndParentImageID(current_picking_job_id, current_image_id);


		// Get a pointer to the array of particle position assets
		ArrayOfParticlePositionAssets * assets_in_bitmap_panel = &ResultDisplayPanel->PickingResultsImagePanel->particle_coordinates_in_angstroms;
		//wxPrintf("Number of assets in bitmap panel = %li\n",assets_in_bitmap_panel->GetCount());
		ResultDisplayPanel->PickingResultsImagePanel->ResetHistory();

		// Before we can add new particle coordinates to the database, we need to assign IDs to positions that were added
		// by the user and therefore don't have an ID yet
		int highest_position_id_so_far = main_frame->current_project.database.ReturnHighestParticlePositionID();
		ParticlePositionAsset * current_asset;
		for (size_t counter = 0; counter < assets_in_bitmap_panel->GetCount(); counter ++ )
		{
			current_asset = & assets_in_bitmap_panel->Item(counter);
			if (current_asset->asset_id == -1)
			{
				highest_position_id_so_far ++;
				current_asset->asset_id = highest_position_id_so_far;
				current_asset->parent_id = current_image_id;
				current_asset->picking_id = picking_id;
				current_asset->pick_job_id = current_picking_job_id;
			}
		}

		// Remove results from database and the asset panel
		main_frame->current_project.database.RemoveParticlePositionsFromResultsList(current_picking_job_id,current_image_id);
		if (CheckBoxIsChecked(row, column))
		{
			particle_position_asset_panel->RemoveParticlePositionAssetsWithGivenParentImageID(current_image_id);
			particle_position_asset_panel->is_dirty = true;
		}


		// Add results to picking_results_*** table
		main_frame->current_project.database.AddArrayOfParticlePositionAssetsToResultsTable(current_picking_job_id,assets_in_bitmap_panel);

		// If the current results are the ones used as assets, update the assets table also
		if (CheckBoxIsChecked(row,column))
		{
			main_frame->current_project.database.CopyParticleAssetsFromResultsTable(current_picking_job_id, current_image_id);
			for (size_t counter = 0; counter < assets_in_bitmap_panel->GetCount(); counter ++ )
			{
				current_asset = & assets_in_bitmap_panel->Item(counter);
				current_asset->pick_job_id = current_picking_job_id;
				current_asset->parent_id = current_image_id;
				current_asset->picking_id = picking_id;
				particle_position_asset_panel->AddAsset(current_asset);
				//wxPrintf("Adding asset to particel position asset panel list\n");
			}
			particle_position_asset_panel->is_dirty = true;
		}

		// Mark the picking results as manually edited
		main_frame->current_project.database.SetManualEditForPickingID(picking_id, true);

	}

}

void MyPickingResultsPanel::FillResultsPanelAndDetails(int row, int column)
{
	bool should_continue;

	// get the correct result from the database..

	int current_image_id = per_row_asset_id[row];
	int current_picking_job_id = picking_job_ids[column - 2];
	wxString parent_image_filename;
	bool keep_going;

	// Variables for job details
	int picking_id;
	long datetime_of_run;
	int picking_job_id_check;
	int parent_image_id_check;
	int picking_algorithm;
	double characteristic_radius;
	double maximum_radius;
	double threshold_peak_height;
	double highest_resolution_used;
	int minimum_distance_from_edges;
	int avoid_high_variance;
	int avoid_high_low_mean;
	int number_of_background_boxes;
	int manual_edit;

	// Get job details
	keep_going = main_frame->current_project.database.BeginBatchSelect(wxString::Format("select * from particle_picking_list where picking_job_id = %i",current_picking_job_id));
	if (!keep_going)
	{
		MyPrintWithDetails("Error dealing with picking_list table");
		abort();
	}
	main_frame->current_project.database.GetFromBatchSelect("iliiirrrriiiii",&picking_id, &datetime_of_run, &picking_job_id_check, &parent_image_id_check, &picking_algorithm, &characteristic_radius, &maximum_radius, &threshold_peak_height, &highest_resolution_used, &minimum_distance_from_edges, &avoid_high_variance, &avoid_high_low_mean, &number_of_background_boxes, &manual_edit);

	main_frame->current_project.database.EndBatchSelect();


	// Set text in the details panel
	PickIDStaticText->SetLabel(wxString::Format("%i", picking_id));
	wxDateTime wxdatetime_of_run;
	wxdatetime_of_run.SetFromDOS((unsigned long) datetime_of_run);
	DateOfRunStaticText->SetLabel(wxdatetime_of_run.FormatISODate());
	TimeOfRunStaticText->SetLabel(wxdatetime_of_run.FormatISOTime());
	AlgorithmStaticText->SetLabel(findparticles_panel->ReturnNameOfPickingAlgorithm(picking_algorithm));
	if (manual_edit == 1)
	{
		ManualEditStaticText->SetLabel("yes");
	}
	else
	{
		ManualEditStaticText->SetLabel("no");
	}
	ThresholdStaticText->SetLabel(wxString::Format("%0.1f",threshold_peak_height));
	MaximumRadiusStaticText->SetLabel(wxString::Format("%0.1f A",maximum_radius));
	CharacteristicRadiusStaticText->SetLabel(wxString::Format("%0.1f A",characteristic_radius));
	HighestResStaticText->SetLabel(wxString::Format("%0.1f A",highest_resolution_used));
	MinEdgeDistStaticText->SetLabel(wxString::Format("%i px",minimum_distance_from_edges));
	if (avoid_high_variance == 1)
	{
		AvoidHighVarStaticText->SetLabel("yes");
	}
	else
	{
		AvoidHighVarStaticText->SetLabel("no");
	}
	if (avoid_high_low_mean == 1)
	{
		AvoidHighLowMeanStaticText->SetLabel("yes");
	}
	else
	{
		AvoidHighLowMeanStaticText->SetLabel("no");
	}
	NumBackgroundBoxesStaticText->SetLabel(wxString::Format("%i",number_of_background_boxes));




	// Get the filename of the image we will need to display

	keep_going = main_frame->current_project.database.BeginBatchSelect(wxString::Format("select filename from image_assets where image_asset_id = %i",current_image_id));
	if (!keep_going)
	{
		MyPrintWithDetails("Error dealing with assets table");
		abort();
	}
	main_frame->current_project.database.GetFromBatchSelect("t",&parent_image_filename);
	main_frame->current_project.database.EndBatchSelect();


	// Get the coordinates of picked particles
	ArrayOfParticlePositionAssets array_of_particle_positions = main_frame->current_project.database.ReturnArrayOfParticlePositionAssetsFromResultsTable(current_picking_job_id,current_image_id);

	float maximum_radius_of_particle = main_frame->current_project.database.ReturnSingleDoubleFromSelectCommand(wxString::Format("select maximum_radius from particle_picking_list where picking_job_id = %i",current_picking_job_id));
	float pixel_size = main_frame->current_project.database.ReturnSingleDoubleFromSelectCommand(wxString::Format("select pixel_size from image_assets where image_asset_id = %i",current_image_id));

	ResultDisplayPanel->Draw(parent_image_filename,array_of_particle_positions, maximum_radius_of_particle, pixel_size);
	RightPanel->Layout();

}

bool MyPickingResultsPanel::CheckBoxIsChecked(const int row, const int column)
{
	wxVariant temp_variant;
	ResultDataView->GetValue(temp_variant, row, column);
	long value = temp_variant.GetLong();
	return value == CHECKED_WITH_EYE || value == CHECKED;
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

			UpdateResultsFromBitmapPanel(old_selected_row,old_selected_column);

			selected_row = row;
			selected_column = column;

			FillResultsPanelAndDetails(row, column);

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


