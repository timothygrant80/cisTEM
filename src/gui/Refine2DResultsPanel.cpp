#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;

Refine2DResultsPanel::Refine2DResultsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
: Refine2DResultsPanelParent(parent, id, pos, size, style)
{
	#include "icons/add_file_icon.cpp"
	#include "icons/delete_file_icon.cpp"

	ClassumDisplayPanel->Initialise(CAN_FFT | NO_NOTEBOOK | CAN_SELECT_IMAGES | NO_POPUP | START_WITH_FOURIER_SCALING | SKIP_LEFTCLICK_TO_PARENT | DO_NOT_SHOW_STATUS_BAR);
	ParticleDisplayPanel->Initialise(CAN_FFT | START_WITH_INVERTED_CONTRAST | START_WITH_AUTO_CONTRAST | NO_NOTEBOOK | START_WITH_NO_LABEL | START_WITH_FOURIER_SCALING | DO_NOT_SHOW_STATUS_BAR);
	selected_class = 1;
	refinement_package_combo_is_dirty = false;
	input_params_combo_is_dirty = false;
	classification_selections_are_dirty = false;

	ClassumDisplayPanel->Bind(wxEVT_RIGHT_DOWN, &Refine2DResultsPanel::OnClassumRightClick, this);
	ClassumDisplayPanel->Bind(wxEVT_LEFT_DOWN, &Refine2DResultsPanel::OnClassumLeftClick, this);
	Layout();
}

Refine2DResultsPanel::~Refine2DResultsPanel()
{
	ClassumDisplayPanel->Unbind(wxEVT_RIGHT_DOWN, &Refine2DResultsPanel::OnClassumRightClick, this);

}


void Refine2DResultsPanel::FillRefinementPackageComboBox(void)
{
	RefinementPackageComboBox->FillWithRefinementPackages();
	FillInputParametersComboBox();
}

void Refine2DResultsPanel::OnClassumRightClick( wxMouseEvent& event )
{
	long current_class = refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].classification_ids[InputParametersComboBox->GetSelection()];
	ShortClassificationInfo *current_classification = refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].classification_ids[InputParametersComboBox->GetSelection()]);

	// work out which has been clicked on..
	if (selected_class != event.GetId())
	{
		selected_class = event.GetId();
		if (selected_class < 1 || selected_class > current_classification->number_of_classes) selected_class = 1;

		wxArrayLong wanted_images = main_frame->current_project.database.Return2DClassMembers(current_class, selected_class);
		ParticleDisplayPanel->ChangeFile(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].stack_filename, "", &wanted_images);
		ClassumDisplayPanel->SetSelectionSquareLocation(selected_class);
		ClassNumberStaticText->SetLabel(wxString::Format("Class Members - Class #%i", selected_class));
	}


}


void Refine2DResultsPanel::OnClassumLeftClick( wxMouseEvent& event )
{
	// work out which has been clicked on..

	if (SelectionManagerListCtrl->ReturnCurrentSelection() != -1 && event.GetId() != -1)
	{
		// ok so we have a selection.. which one is it..

		ClassificationSelection *current_selection = &refinement_package_asset_panel->all_classification_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelectionOriginalArrayPosition());

		// ok toggle the effective selection..

		if (ClassumDisplayPanel->ReturnCurrentPanel()->image_is_selected[event.GetId()] == false)
		{
			// need to add it..
			current_selection->selections.Add(event.GetId());
			current_selection->number_of_selections++;

			// and from listctrl..

			SelectionManagerListCtrl->all_valid_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelection()).selections.Add(event.GetId());
			SelectionManagerListCtrl->all_valid_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelection()).number_of_selections++;

			// now the database..

			main_frame->current_project.database.InsertOrReplace(wxString::Format("CLASSIFICATION_SELECTION_%li", current_selection->selection_id), "i", "CLASS_AVERAGE_NUMBER", event.GetId());
			main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE CLASSIFICATION_SELECTION_LIST SET NUMBER_OF_SELECTIONS=%i WHERE SELECTION_ID=%li", current_selection->number_of_selections, current_selection->selection_id));

		}
		else
		{
			// need to add it..
			current_selection->selections.Remove(event.GetId());
			current_selection->number_of_selections--;

			// and from listctrl..

			SelectionManagerListCtrl->all_valid_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelection()).selections.Remove(event.GetId());
			SelectionManagerListCtrl->all_valid_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelection()).number_of_selections--;

			// now the databse..

			main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM CLASSIFICATION_SELECTION_%li WHERE CLASS_AVERAGE_NUMBER=%i", current_selection->selection_id, event.GetId()));
			main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE CLASSIFICATION_SELECTION_LIST SET NUMBER_OF_SELECTIONS=%i WHERE SELECTION_ID=%li", current_selection->number_of_selections, current_selection->selection_id));


		}

		ClassumDisplayPanel->ToggleImageSelected(event.GetId());
		SelectionManagerListCtrl->RefreshItem(SelectionManagerListCtrl->ReturnCurrentSelection());

	}




}

void Refine2DResultsPanel::FillInputParametersComboBox(void)
{
	if (RefinementPackageComboBox->GetSelection() >= 0)
	{

		if (InputParametersComboBox->FillWithClassifications(RefinementPackageComboBox->GetSelection(), false) == false && InputParametersComboBox->GetSelection() >= 0)
		{
			long current_class = refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].classification_ids[InputParametersComboBox->GetSelection()];
			ShortClassificationInfo *current_classification = refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].classification_ids[InputParametersComboBox->GetSelection()]);

			if (selected_class < 1 || selected_class > current_classification->number_of_classes) selected_class = 1;

			wxArrayLong wanted_images = main_frame->current_project.database.Return2DClassMembers(current_class, selected_class);

			ClassumDisplayPanel->ChangeFile(current_classification->class_average_file, "");
			ParticleDisplayPanel->ChangeFile(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].stack_filename, "", &wanted_images);
			ClassumDisplayPanel->SetSelectionSquareLocation(selected_class);
			ClassNumberStaticText->SetLabel(wxString::Format("Class Members - Class #%i", selected_class));

			WriteJobInfo(current_classification->classification_id);
			FillSelectionManagerListCtrl();

		}
		else
		if (InputParametersComboBox->GetSelection() == -1)
		{
			Clear();
		}

	}
}

void Refine2DResultsPanel::OnJobDetailsToggle( wxCommandEvent& event )
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


	LeftPanel->Layout();
	Thaw();
}

void Refine2DResultsPanel::Clear()
{
	ClassumDisplayPanel->Clear();
	ParticleDisplayPanel->Clear();
	ClassNumberStaticText->SetLabel("Class Members");
	ClearJobInfo();
	SelectionManagerListCtrl->ClearAll();
	Layout();

}

void Refine2DResultsPanel::ClearJobInfo()
{
	ClassificationIDStaticText->SetLabel("-");
	DateOfRunStaticText->SetLabel("-");
	TimeOfRunStaticText->SetLabel("-");
	RefinementPackageIDStaticText->SetLabel("-");
	StartClassificationIDStaticText->SetLabel("-");
	NumberClassesStaticText->SetLabel("-");
	NumberParticlesStaticText->SetLabel("-");
	LowResLimitStaticText->SetLabel("-");
	HighResLimitStaticText->SetLabel("-");
	MaskRadiusStaticText->SetLabel("-");
	AngularSearchStepStaticText->SetLabel("-");
	SearchRangeXStaticText->SetLabel("-");
	SearchRangeYStaticText->SetLabel("-");
	SmoothingFactorStaticText->SetLabel("-");
	ExcludeBlankEdgesStaticText->SetLabel("-");
	AutoPercentUsedStaticText->SetLabel("-");
	PercentUsedStaticText->SetLabel("-");

	LeftPanel->Layout();
}

void Refine2DResultsPanel::WriteJobInfo(long wanted_classification_id)
{
	wxString sql_select_command;
	int return_code;
	sqlite3_stmt *list_statement = NULL;
	Classification *temp_classification = new Classification;
	bool more_data;
	long records_retrieved = 0;

	// general data

	sql_select_command = wxString::Format("SELECT * FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=%li", wanted_classification_id);
	return_code = sqlite3_prepare_v2(main_frame->current_project.database.sqlite_database, sql_select_command.ToUTF8().data(), sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\nSQL Command : %s\n", return_code , sql_select_command);
	return_code = sqlite3_step(list_statement);

	temp_classification->classification_id = sqlite3_column_int64(list_statement, 0);
	temp_classification->refinement_package_asset_id = sqlite3_column_int64(list_statement, 1);
	temp_classification->name = sqlite3_column_text(list_statement, 2);
	temp_classification->class_average_file = sqlite3_column_text(list_statement, 3);
	temp_classification->classification_was_imported_or_generated = sqlite3_column_int(list_statement, 4);
	temp_classification->datetime_of_run.SetFromDOS((unsigned long) sqlite3_column_int64(list_statement, 5));
	temp_classification->starting_classification_id = sqlite3_column_int64(list_statement, 6);
	temp_classification->number_of_particles = sqlite3_column_int64(list_statement, 7);
	temp_classification->number_of_classes = sqlite3_column_int(list_statement, 8);
	temp_classification->low_resolution_limit = sqlite3_column_double(list_statement, 9);
	temp_classification->high_resolution_limit = sqlite3_column_double(list_statement, 10);
	temp_classification->mask_radius = sqlite3_column_double(list_statement, 11);
	temp_classification->angular_search_step = sqlite3_column_double(list_statement, 12);
	temp_classification->search_range_x = sqlite3_column_double(list_statement, 13);
	temp_classification->search_range_y = sqlite3_column_double(list_statement, 14);
	temp_classification->smoothing_factor = sqlite3_column_double(list_statement, 15);
	temp_classification->exclude_blank_edges = sqlite3_column_int(list_statement, 16);
	temp_classification->auto_percent_used = sqlite3_column_int(list_statement, 17);
	temp_classification->percent_used = sqlite3_column_double(list_statement, 18);

	sqlite3_finalize(list_statement);


	ClassificationIDStaticText->SetLabel(wxString::Format("%li", temp_classification->classification_id));
	DateOfRunStaticText->SetLabel(temp_classification->datetime_of_run.FormatISODate());
	TimeOfRunStaticText->SetLabel(temp_classification->datetime_of_run.FormatISOTime());
	RefinementPackageIDStaticText->SetLabel(wxString::Format("%li", temp_classification->refinement_package_asset_id));
	StartClassificationIDStaticText->SetLabel(wxString::Format("%li", temp_classification->starting_classification_id));
	NumberClassesStaticText->SetLabel(wxString::Format("%i", temp_classification->number_of_classes));
	NumberParticlesStaticText->SetLabel(wxString::Format("%li", temp_classification->number_of_particles));
	LowResLimitStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), temp_classification->low_resolution_limit));
	HighResLimitStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), temp_classification->high_resolution_limit));
	MaskRadiusStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), temp_classification->mask_radius));
	AngularSearchStepStaticText->SetLabel(wxString::Format(wxT("%.2f °"), temp_classification->angular_search_step));
	SearchRangeXStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), temp_classification->search_range_x));
	SearchRangeYStaticText->SetLabel(wxString::Format(wxT("%.2f Å"), temp_classification->search_range_y));
	SmoothingFactorStaticText->SetLabel(wxString::Format(wxT("%.2f"), temp_classification->smoothing_factor));

	if (temp_classification->exclude_blank_edges == true) ExcludeBlankEdgesStaticText->SetLabel("Yes");
	else ExcludeBlankEdgesStaticText->SetLabel("No");

	if (temp_classification->auto_percent_used == true) AutoPercentUsedStaticText->SetLabel("Yes");
	else AutoPercentUsedStaticText->SetLabel("No");

	PercentUsedStaticText->SetLabel(wxString::Format(wxT("%.2f %%"), temp_classification->percent_used));

	LeftPanel->Layout();
	delete temp_classification;
}


void Refine2DResultsPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	if (main_frame->current_project.is_open == false)
	{
		Enable(false);
		RefinementPackageComboBox->Clear();
		RefinementPackageComboBox->ChangeValue("");
		InputParametersComboBox->Clear();
		InputParametersComboBox->ChangeValue("");
//		ClassumDisplayPanel->Clear();
//  	ParticleDisplayPanel->Clear();
	}
	else
	{
		Enable(true);

		if (refinement_package_combo_is_dirty == true)
		{
			refinement_package_combo_is_dirty = false;
			FillRefinementPackageComboBox();
		}

		if (input_params_combo_is_dirty == true)
		{
			input_params_combo_is_dirty = false;
			FillInputParametersComboBox();
		}

		if (classification_selections_are_dirty == true)
		{
			FillSelectionManagerListCtrl();
			classification_selections_are_dirty = false;
		}

		if (SelectionManagerListCtrl->ReturnCurrentSelection() == -1)
		{
			DeleteButton->Enable(false);
			RenameButton->Enable(false);
			ClearButton->Enable(false);
			InvertButton->Enable(false);
		}
		else
		{
			DeleteButton->Enable(true);
			RenameButton->Enable(true);
			ClearButton->Enable(true);
			InvertButton->Enable(true);

		}

		if (InputParametersComboBox->GetSelection() >= 0) AddButton->Enable(true);
		else AddButton->Enable(false);

		if (refinement_package_asset_panel->all_classification_selections.GetCount() - SelectionManagerListCtrl->all_valid_selections.GetCount() > 0 && InputParametersComboBox->GetSelection() >= 0) CopyOtherButton->Enable(true);
		else CopyOtherButton->Enable(false);
	}
}
void Refine2DResultsPanel::OnRefinementPackageComboBox( wxCommandEvent& event )
{
	if (RefinementPackageComboBox->GetSelection() >= 0)
	{
		FillInputParametersComboBox();
	}
}

void Refine2DResultsPanel::OnInputParametersComboBox( wxCommandEvent& event )
{
	if (RefinementPackageComboBox->GetSelection() >= 0)
	{
		long current_class = refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].classification_ids[InputParametersComboBox->GetSelection()];
		wxArrayLong wanted_images = main_frame->current_project.database.Return2DClassMembers(current_class, selected_class);

		ShortClassificationInfo *current_classification = refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].classification_ids[InputParametersComboBox->GetSelection()]);

		ClassumDisplayPanel->OpenFile(current_classification->class_average_file, "");
		ParticleDisplayPanel->OpenFile(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].stack_filename, "", &wanted_images);
		ClassumDisplayPanel->SetSelectionSquareLocation(selected_class);
		WriteJobInfo(current_classification->classification_id);
		FillSelectionManagerListCtrl();
	}
}

void Refine2DResultsPanel::FillSelectionManagerListCtrl(bool select_latest)
{
	SelectionManagerListCtrl->ClearAll();

	if (RefinementPackageComboBox->GetSelection() >= 0 && InputParametersComboBox->GetSelection() >= 0)
	{
		long current_class = refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].classification_ids[InputParametersComboBox->GetSelection()];
		ShortClassificationInfo *current_classification = refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].classification_ids[InputParametersComboBox->GetSelection()]);

		SelectionManagerListCtrl->Fill(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].asset_id, current_classification->classification_id, select_latest);
	}

}

void Refine2DResultsPanel::OnDeselected( wxListEvent& event )
{
	SelectionManagerListCtrl->current_selection = -1;
	SelectionManagerListCtrl->current_selection_id = -10;
	ClassumDisplayPanel->ClearSelection(true);
}

void Refine2DResultsPanel::OnSelected( wxListEvent& event )
{
	SelectionManagerListCtrl->current_selection = event.GetIndex();
	SelectionManagerListCtrl->current_selection_id = SelectionManagerListCtrl->all_valid_selections.Item(event.GetIndex()).selection_id;

	ClassificationSelection *current_selection = &refinement_package_asset_panel->all_classification_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelectionOriginalArrayPosition());

	for (int counter = 0; counter < current_selection->selections.GetCount(); counter++)
	{
		ClassumDisplayPanel->SetImageSelected(current_selection->selections.Item(counter));
	}
}

void Refine2DResultsPanel::OnActivated( wxListEvent& event )
{
	SelectionManagerListCtrl->EditLabel(event.GetIndex());
}

void Refine2DResultsPanel::OnEndLabelEdit( wxListEvent& event )
{
	if (event.GetLabel() == wxEmptyString)
	{
		event.Veto();
	}
	else
	{
		ClassificationSelection *current_selection = &refinement_package_asset_panel->all_classification_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelectionOriginalArrayPosition());
		current_selection->name = event.GetLabel();
		SelectionManagerListCtrl->all_valid_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelection()).name = event.GetLabel();
		main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE CLASSIFICATION_SELECTION_LIST SET SELECTION_NAME=\"%s\" WHERE SELECTION_ID=%li", current_selection->name, current_selection->selection_id));
		SelectionManagerListCtrl->RefreshItem(SelectionManagerListCtrl->ReturnCurrentSelection());

	}

}

void Refine2DResultsPanel::OnAddButtonClick(wxCommandEvent& event )
{
	// work out current refinement package asset id, and number of classes.

	if (RefinementPackageComboBox->GetSelection() >= 0 && InputParametersComboBox->GetSelection() >= 0)
	{
		ClassificationSelection new_selection;

		ShortClassificationInfo *current_classification = refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].classification_ids[InputParametersComboBox->GetSelection()]);

		new_selection.refinement_package_asset_id = refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].asset_id;
		new_selection.classification_id =  current_classification->classification_id;
		new_selection.number_of_classes = current_classification->number_of_classes;
		new_selection.name = "New Selection";
		new_selection.number_of_selections = 0;
		new_selection.selection_id = main_frame->current_project.database.ReturnHighestClassumSelectionID() + 1;

		refinement_package_asset_panel->all_classification_selections.Add(new_selection);

		// add to the database..

		main_frame->current_project.database.AddClassificationSelection(&new_selection);
		ClassumDisplayPanel->ClearSelection(true);
		FillSelectionManagerListCtrl(true);

	}
}

void Refine2DResultsPanel::OnClearButtonClick(wxCommandEvent& event )
{
	// work out current refinement package asset id, and number of classes.

	if (SelectionManagerListCtrl->ReturnCurrentSelectionOriginalArrayPosition() != -1)
	{
		int current_array_position = SelectionManagerListCtrl->ReturnCurrentSelectionOriginalArrayPosition();
		ClassificationSelection *current_selection = &refinement_package_asset_panel->all_classification_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelectionOriginalArrayPosition());

		refinement_package_asset_panel->all_classification_selections.Item(current_array_position).selections.Clear();
		refinement_package_asset_panel->all_classification_selections.Item(current_array_position).number_of_selections = 0;

		SelectionManagerListCtrl->all_valid_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelection()).selections.Clear();
		SelectionManagerListCtrl->all_valid_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelection()).number_of_selections = 0;

		main_frame->current_project.database.ExecuteSQL(wxString::Format("BEGIN"));
		main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE CLASSIFICATION_SELECTION_LIST SET NUMBER_OF_SELECTIONS=0 WHERE SELECTION_ID=%li", current_selection->selection_id));
		main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE CLASSIFICATION_SELECTION_%li", current_selection->selection_id));
		main_frame->current_project.database.CreateClassificationSelectionTable(current_selection->selection_id);
		main_frame->current_project.database.ExecuteSQL(wxString::Format("COMMIT"));

		ClassumDisplayPanel->ClearSelection(true);
		SelectionManagerListCtrl->RefreshItem(SelectionManagerListCtrl->ReturnCurrentSelection());

	}
}

void Refine2DResultsPanel::OnInvertButtonClick(wxCommandEvent& event )
{
	// work out current refinement package asset id, and number of classes.

	if (SelectionManagerListCtrl->ReturnCurrentSelectionOriginalArrayPosition() != -1)
	{
		// we are going to use the display array for the inversion - this may be dodgy, but maybe ok!

		int current_array_position = SelectionManagerListCtrl->ReturnCurrentSelectionOriginalArrayPosition();
		ClassificationSelection *current_selection = &refinement_package_asset_panel->all_classification_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelectionOriginalArrayPosition());

		refinement_package_asset_panel->all_classification_selections.Item(current_array_position).selections.Clear();
		refinement_package_asset_panel->all_classification_selections.Item(current_array_position).number_of_selections = 0;

		SelectionManagerListCtrl->all_valid_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelection()).selections.Clear();
		SelectionManagerListCtrl->all_valid_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelection()).number_of_selections = 0;

		main_frame->current_project.database.ExecuteSQL(wxString::Format("BEGIN"));
		main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE CLASSIFICATION_SELECTION_%li", current_selection->selection_id));
		main_frame->current_project.database.CreateClassificationSelectionTable(current_selection->selection_id);

		for (int counter = 0; counter < current_selection->number_of_classes; counter++)
		{
			if (ClassumDisplayPanel->IsImageSelected(counter + 1) == false)
			{
				refinement_package_asset_panel->all_classification_selections.Item(current_array_position).selections.Add(counter + 1);
				refinement_package_asset_panel->all_classification_selections.Item(current_array_position).number_of_selections++;

				SelectionManagerListCtrl->all_valid_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelection()).selections.Add(counter + 1);
				SelectionManagerListCtrl->all_valid_selections.Item(SelectionManagerListCtrl->ReturnCurrentSelection()).number_of_selections++;

				main_frame->current_project.database.InsertOrReplace(wxString::Format("CLASSIFICATION_SELECTION_%li", current_selection->selection_id), "i", "CLASS_AVERAGE_NUMBER", counter + 1);
			}

			ClassumDisplayPanel->ToggleImageSelected(counter + 1, false);
		}

		main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE CLASSIFICATION_SELECTION_LIST SET NUMBER_OF_SELECTIONS=%i WHERE SELECTION_ID=%li", current_selection->number_of_selections, current_selection->selection_id));
		main_frame->current_project.database.ExecuteSQL(wxString::Format("COMMIT"));

		SelectionManagerListCtrl->RefreshItem(SelectionManagerListCtrl->ReturnCurrentSelection());
		ClassumDisplayPanel->RefreshCurrentPanel();
	}
}



void Refine2DResultsPanel::OnDeleteButtonClick(wxCommandEvent& event )
{
	if (SelectionManagerListCtrl->ReturnCurrentSelectionOriginalArrayPosition() != -1)
	{
		int current_array_position = SelectionManagerListCtrl->ReturnCurrentSelectionOriginalArrayPosition();
		ClassificationSelection *current_selection = &refinement_package_asset_panel->all_classification_selections.Item(current_array_position);

		main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM CLASSIFICATION_SELECTION_LIST WHERE SELECTION_ID=%li", current_selection->selection_id));
		main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE CLASSIFICATION_SELECTION_%li", current_selection->selection_id));
		SelectionManagerListCtrl->current_selection = -1;
		refinement_package_asset_panel->all_classification_selections.RemoveAt(current_array_position);
		ClassumDisplayPanel->ClearSelection(true);
		FillSelectionManagerListCtrl(false);

	}

}

void Refine2DResultsPanel::OnCopyOtherButtonClick(wxCommandEvent& event )
{
	ClassumSelectionCopyFromDialog *select_dialog = new ClassumSelectionCopyFromDialog(this);

	ShortClassificationInfo *current_classification = refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].classification_ids[InputParametersComboBox->GetSelection()]);
	select_dialog->FillWithSelections(current_classification->number_of_classes);
	if (select_dialog->ShowModal() == wxID_OK)
	{
		// add a new one copied from this one but updated with classification id / refinement package id..

		int current_array_position = select_dialog->ReturnSelectedPosition();

		ClassificationSelection new_selection = refinement_package_asset_panel->all_classification_selections.Item(current_array_position);
		new_selection.name = "Copy of " + refinement_package_asset_panel->all_classification_selections.Item(current_array_position).name;
		new_selection.selection_id = main_frame->current_project.database.ReturnHighestClassumSelectionID() + 1;
		new_selection.refinement_package_asset_id = refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].asset_id;
		new_selection.classification_id =  current_classification->classification_id;

		refinement_package_asset_panel->all_classification_selections.Add(new_selection);

		// add to the database..

		main_frame->current_project.database.AddClassificationSelection(&new_selection);
		ClassumDisplayPanel->ClearSelection(true);
		FillSelectionManagerListCtrl(true);

	}

	select_dialog->Destroy();

}

void Refine2DResultsPanel::OnRenameButtonClick(wxCommandEvent& event )
{
	if (SelectionManagerListCtrl->ReturnCurrentSelection() != -1)
	{

		SelectionManagerListCtrl->EditLabel(SelectionManagerListCtrl->ReturnCurrentSelection());
	}

}

