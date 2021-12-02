#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyVolumeAssetPanel *volume_asset_panel;
extern MyRefinementResultsPanel *refinement_results_panel;

RefineCTFPanel::RefineCTFPanel( wxWindow* parent )
:
RefineCTFPanelParent( parent )
{
	buffered_results = NULL;

	// Fill combo box..

	//FillGroupComboBox();

	my_job_id = -1;
	running_job = false;

//	group_combo_is_dirty = false;
//	run_profiles_are_dirty = false;

	SetInfo();
//	FillGroupComboBox()t
//	FillRunProfileComboBox();

	wxSize input_size = InputSizer->GetMinSize();
	input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
	input_size.y = -1;
	ExpertPanel->SetMinSize(input_size);
	ExpertPanel->SetSize(input_size);


	// set values //

	/*
	AmplitudeContrastNumericCtrl->SetMinMaxValue(0.0f, 1.0f);
	MinResNumericCtrl->SetMinMaxValue(0.0f, 50.0f);
	MaxResNumericCtrl->SetMinMaxValue(0.0f, 50.0f);
	DefocusStepNumericCtrl->SetMinMaxValue(1.0f, FLT_MAX);
	ToleratedAstigmatismNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
	MinPhaseShiftNumericCtrl->SetMinMaxValue(-3.15, 3.15);
	MaxPhaseShiftNumericCtrl->SetMinMaxValue(-3.15, 3.15);
	PhaseShiftStepNumericCtrl->SetMinMaxValue(0.001, 3.15);

	result_bitmap.Create(1,1, 24);
	time_of_last_result_update = time(NULL);*/

	refinement_package_combo_is_dirty = false;
	run_profiles_are_dirty = false;
	input_params_combo_is_dirty = false;
	selected_refinement_package = -1;

	RefinementPackageComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &RefineCTFPanel::OnRefinementPackageComboBox, this);
	Bind(RETURN_PROCESSED_IMAGE_EVT, &RefineCTFPanel::OnOrthThreadComplete, this);
	Bind(wxEVT_MULTIPLY3DMASKTHREAD_COMPLETED, &RefineCTFPanel::OnMaskerThreadComplete, this);
	Bind(wxEVT_AUTOMASKERTHREAD_COMPLETED, &RefineCTFPanel::OnMaskerThreadComplete, this);

	my_refinement_manager.SetParent(this);

	FillRefinementPackagesComboBox();

	long time_of_last_result_update;

	active_orth_thread_id = -1;
	active_mask_thread_id = -1;
	next_thread_id = 1;

	ShowRefinementResultsPanel->DefocusHistorgramPlotPanel->Initialise(wxT("Defocus Change (Å)"), "Number of Images", false, false, 20, 50, 60, 20, true, false, true);
	ShowRefinementResultsPanel->DefocusHistorgramPlotPanel->SetXAxisMinStep(100);

	Active3DReferencesListCtrl->refinement_package_picker_to_use = RefinementPackageComboBox;

}

void RefineCTFPanel::Reset()
{
	ProgressBar->SetValue(0);
	TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
	FinishButton->Show(false);

	InputParamsPanel->Show(true);
	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	output_textctrl->Clear();
	ShowRefinementResultsPanel->Show(false);
	ShowRefinementResultsPanel->Clear();

	UseMaskCheckBox->SetValue(false);
	HighResolutionLimitTextCtrl->ChangeValueFloat(30.0f);

	//CTFResultsPanel->Show(false);
	//graph_is_hidden = true;
	InfoPanel->Show(true);

	RefinementPackageComboBox->Clear();
	InputParametersComboBox->Clear();
	RefinementRunProfileComboBox->Clear();
	ReconstructionRunProfileComboBox->Clear();

	UseMaskCheckBox->SetValue(false);
	ExpertToggleButton->SetValue(false);
	ExpertPanel->Show(false);

	if (running_job == true)
	{
		main_frame->job_controller.KillJob(my_job_id);

		active_mask_thread_id = -1;
		active_orth_thread_id = -1;
		running_job = false;
	}

	if (my_refinement_manager.output_refinement != NULL)
	{
		delete my_refinement_manager.output_refinement;
		my_refinement_manager.output_refinement = NULL;
	}

	SetDefaults();
	global_delete_refinectf_scratch();
	Layout();
}

void RefineCTFPanel::SetInfo()
{

	wxLogNull *suppress_png_warnings = new wxLogNull;
//	#include "icons/niko_picture1.cpp"
//	wxBitmap niko_picture1_bmp = wxBITMAP_PNG_FROM_DATA(niko_picture1);

	InfoText->GetCaret()->Hide();

	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("CTF Refinement"));
	InfoText->EndFontSize();
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("Makes stuff better..."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

/*	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(niko_picture1_bmp);
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();*/

}

void RefineCTFPanel::OnInfoURL(wxTextUrlEvent& event)
{
	 const wxMouseEvent& ev = event.GetMouseEvent();

	 // filter out mouse moves, too many of them
	 if ( ev.Moving() ) return;

	 long start = event.GetURLStart();

	 wxTextAttr my_style;

	 InfoText->GetStyle(start, my_style);

	 // Launch the URL

	 wxLaunchDefaultBrowser(my_style.GetURL());
}


void RefineCTFPanel::ResetAllDefaultsClick( wxCommandEvent& event )
{
	// TODO : should probably check that the user hasn't changed the defaults yet in the future
	SetDefaults();
}

void RefineCTFPanel::SetDefaults()
{
	if (RefinementPackageComboBox->GetCount() > 0)
	{
		float calculated_high_resolution_cutoff;
		float local_mask_radius;
		float global_mask_radius;
		float global_angular_step;
		float search_range;

		ExpertPanel->Freeze();

	// calculate high resolution limit..

		long current_input_refinement_id = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).refinement_ids[InputParametersComboBox->GetSelection()];
		calculated_high_resolution_cutoff = 3.5;

		for (int class_counter = 0; class_counter < refinement_package_asset_panel->ReturnPointerToShortRefinementInfoByRefinementID(current_input_refinement_id)->number_of_classes; class_counter++)
		{
		//if (refinement_package_asset_panel->ReturnPointerToRefinementByRefinementID(current_input_refinement_id)->class_refinement_results[class_counter].class_resolution_statistics.Return0p5Resolution() > calculated_high_resolution_cutoff) calculated_high_resolution_cutoff = refinement_package_asset_panel->ReturnPointerToRefinementByRefinementID(current_input_refinement_id)->class_refinement_results[class_counter].class_resolution_statistics.Return0p8Resolution();
		}

		local_mask_radius = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.65;
		global_mask_radius = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.8;

		global_angular_step = CalculateAngularStep(calculated_high_resolution_cutoff, local_mask_radius);

		search_range = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.15;
		// Set the values..

		float low_res_limit = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 1.5;
		if (low_res_limit > 300.00) low_res_limit = 300.00;

		auto_mask_value = true;
		LowResolutionLimitTextCtrl->SetValue(wxString::Format("%.2f", low_res_limit));
		HighResolutionLimitTextCtrl->SetValue(wxString::Format("%.2f", calculated_high_resolution_cutoff));
		MaskRadiusTextCtrl->SetValue(wxString::Format("%.2f", local_mask_radius));
		SignedCCResolutionTextCtrl->SetValue("0.00");

		DefocusSearchRangeTextCtrl->SetValue("500.00");
		DefocusSearchStepTextCtrl->SetValue("20.00");

		InnerMaskRadiusTextCtrl->SetValue("0.00");
		ScoreToWeightConstantTextCtrl->SetValue("2.00");

		AdjustScoreForDefocusYesRadio->SetValue(true);
		AdjustScoreForDefocusNoRadio->SetValue(false);
		ReconstructionScoreThreshold->SetValue("0.00");
		ReconstructionResolutionLimitTextCtrl->SetValue("0.00");
		AutoCropYesRadioButton->SetValue(false);
		AutoCropNoRadioButton->SetValue(true);

		ApplyBlurringNoRadioButton->SetValue(true);
		ApplyBlurringYesRadioButton->SetValue(false);
		SmoothingFactorTextCtrl->SetValue("1.00");

		MaskEdgeTextCtrl->ChangeValueFloat(10.00);
		MaskWeightTextCtrl->ChangeValueFloat(0.00);
		LowPassMaskYesRadio->SetValue(false);
		LowPassMaskNoRadio->SetValue(true);
		MaskFilterResolutionText->ChangeValueFloat(20.00);

		ExpertPanel->Thaw();
	}

}

void RefineCTFPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	// are there enough members in the selected group.
	if (main_frame->current_project.is_open == false)
	{
		RefinementPackageComboBox->Enable(false);
		InputParametersComboBox->Enable(false);
		RefinementRunProfileComboBox->Enable(false);
		ReconstructionRunProfileComboBox->Enable(false);
		ExpertToggleButton->Enable(false);
		StartRefinementButton->Enable(false);
		UseMaskCheckBox->Enable(false);
		MaskSelectPanel->Enable(false);
		HighResolutionLimitTextCtrl->Enable(false);
		RefineCTFCheckBox->Enable(false);
		RefineBeamTiltCheckBox->Enable(false);
		HighResolutionLimitTextCtrl->Enable(false); // Why is this duplicated
		HiResLimitStaticText->Enable(false);


		if (ExpertPanel->IsShown() == true)
		{
			ExpertToggleButton->SetValue(false);
			ExpertPanel->Show(false);
			Layout();

		}

		if (RefinementPackageComboBox->GetCount() > 0)
		{
			RefinementPackageComboBox->Clear();
			RefinementPackageComboBox->ChangeValue("");

		}

		if (InputParametersComboBox->GetCount() > 0)
		{
			InputParametersComboBox->Clear();
			InputParametersComboBox->ChangeValue("");
		}

		if (ReconstructionRunProfileComboBox->GetCount() > 0)
		{
			ReconstructionRunProfileComboBox->Clear();
			ReconstructionRunProfileComboBox->ChangeValue("");
		}

		if (RefinementRunProfileComboBox->GetCount() > 0)
		{
			RefinementRunProfileComboBox->Clear();
			RefinementRunProfileComboBox->ChangeValue("");
		}

		if (PleaseCreateRefinementPackageText->IsShown())
		{
			PleaseCreateRefinementPackageText->Show(false);
			Layout();
		}

	}
	else
	{
		RefinementRunProfileComboBox->Enable(true);
		ReconstructionRunProfileComboBox->Enable(true);

		if (running_job == false)
		{
			UseMaskCheckBox->Enable(true);

			RefineCTFCheckBox->Enable(true);
			RefineBeamTiltCheckBox->Enable(true);

			if (RefineCTFCheckBox->IsChecked() == true)
			{
				HighResolutionLimitTextCtrl->Enable(true);
				HiResLimitStaticText->Enable(true);
			}
			else
			{
				HighResolutionLimitTextCtrl->Enable(false);
				HiResLimitStaticText->Enable(false);

			}

			if (RefinementPackageComboBox->GetCount() > 0)
			{

				RefinementPackageComboBox->Enable(true);
				InputParametersComboBox->Enable(true);

				if (UseMaskCheckBox->GetValue() == true)
				{
					MaskSelectPanel->Enable(true);
				}
				else
				{
					MaskSelectPanel->Enable(false);
					if (MaskSelectPanel->GetCount() > 0)
					{
						MaskSelectPanel->Clear();
						MaskSelectPanel->AssetComboBox->ChangeValue("");
					}
				}

				if (PleaseCreateRefinementPackageText->IsShown())
				{
					PleaseCreateRefinementPackageText->Show(false);
					Layout();
				}

			}
			else
			{
				UseMaskCheckBox->Enable(false);
				MaskSelectPanel->Enable(false);
				MaskSelectPanel->AssetComboBox->ChangeValue("");
				RefinementPackageComboBox->ChangeValue("");
				RefinementPackageComboBox->Enable(false);
				InputParametersComboBox->ChangeValue("");
				InputParametersComboBox->Enable(false);

				if (PleaseCreateRefinementPackageText->IsShown() == false)
				{
					PleaseCreateRefinementPackageText->Show(true);
					Layout();
				}
			}

			if (ExpertToggleButton->GetValue() == true)
			{
					if (ApplyBlurringYesRadioButton->GetValue() == true)
				{
					SmoothingFactorTextCtrl->Enable(true);
					SmoothingFactorStaticText->Enable(true);
				}
				else
				{
					SmoothingFactorTextCtrl->Enable(false);
					SmoothingFactorStaticText->Enable(false);
				}

				if (UseMaskCheckBox->GetValue() == false)
				{
					MaskEdgeStaticText->Enable(false);
					MaskEdgeTextCtrl->Enable(false);
					MaskWeightStaticText->Enable(false);
					MaskWeightTextCtrl->Enable(false);
					LowPassYesNoStaticText->Enable(false);
					LowPassMaskYesRadio->Enable(false);
					LowPassMaskNoRadio->Enable(false);
					FilterResolutionStaticText->Enable(false);
					MaskFilterResolutionText->Enable(false);

					AutoMaskStaticText->Enable(true);
					AutoMaskYesRadioButton->Enable(true);
					AutoMaskNoRadioButton->Enable(true);

					if (AutoMaskYesRadioButton->GetValue() != auto_mask_value)
					{
						if (auto_mask_value == true) AutoMaskYesRadioButton->SetValue(true);
						else AutoMaskNoRadioButton->SetValue(true);
					}
				}
				else
				{
					AutoMaskStaticText->Enable(false);
					AutoMaskYesRadioButton->Enable(false);
					AutoMaskNoRadioButton->Enable(false);

					if (AutoMaskYesRadioButton->GetValue() != false)
					{
						AutoMaskNoRadioButton->SetValue(true);
					}

					MaskEdgeStaticText->Enable(true);
					MaskEdgeTextCtrl->Enable(true);
					MaskWeightStaticText->Enable(true);
					MaskWeightTextCtrl->Enable(true);
					LowPassYesNoStaticText->Enable(true);
					LowPassMaskYesRadio->Enable(true);
					LowPassMaskNoRadio->Enable(true);

					if (LowPassMaskYesRadio->GetValue() == true)
					{
						FilterResolutionStaticText->Enable(true);
						MaskFilterResolutionText->Enable(true);
					}
					else
					{
						FilterResolutionStaticText->Enable(false);
						MaskFilterResolutionText->Enable(false);
					}




				}



			}

			bool estimation_button_status = false;

			if (RefinementPackageComboBox->GetCount() > 0 && ReconstructionRunProfileComboBox->GetCount() > 0)
			{
				if (run_profiles_panel->run_profile_manager.ReturnTotalJobs(RefinementRunProfileComboBox->GetSelection()) > 0 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(ReconstructionRunProfileComboBox->GetSelection()) > 0)
				{
					if (RefinementPackageComboBox->GetSelection() != wxNOT_FOUND && InputParametersComboBox->GetSelection() != wxNOT_FOUND)
					{
						if (UseMaskCheckBox->GetValue() == false || MaskSelectPanel->AssetComboBox->GetSelection() != wxNOT_FOUND)
						estimation_button_status = true;
					}

				}
			}

			StartRefinementButton->Enable(estimation_button_status);

			if (refinement_package_combo_is_dirty == true)
			{
				FillRefinementPackagesComboBox();
				refinement_package_combo_is_dirty = false;
			}

			if (run_profiles_are_dirty == true)
			{
				FillRunProfileComboBoxes();
				run_profiles_are_dirty = false;
			}

			if (input_params_combo_is_dirty == true)
			{
				FillInputParamsComboBox();
				input_params_combo_is_dirty = false;
			}

			if (volumes_are_dirty == true)
			{
				MaskSelectPanel->FillComboBox();
				volumes_are_dirty = false;
			}
		}
		else
		{
			HiResLimitStaticText->Enable(false);
			RefinementPackageComboBox->Enable(false);
			InputParametersComboBox->Enable(false);
			HighResolutionLimitTextCtrl->Enable(false);
			UseMaskCheckBox->Enable(false);
			MaskSelectPanel->Enable(false);

		}
	}

}

void RefineCTFPanel::OnAutoMaskButton( wxCommandEvent& event )
{
	auto_mask_value = AutoMaskYesRadioButton->GetValue();
}

void RefineCTFPanel::OnUseMaskCheckBox( wxCommandEvent& event )
{
	if (UseMaskCheckBox->GetValue() == true)
	{
		auto_mask_value = AutoMaskYesRadioButton->GetValue();
		MaskSelectPanel->FillComboBox();
	}
	else
	{
		if (auto_mask_value == true) AutoMaskYesRadioButton->SetValue(true);
		else AutoMaskNoRadioButton->SetValue(true);
	}
}

void RefineCTFPanel::OnExpertOptionsToggle( wxCommandEvent& event )
{

	if (ExpertToggleButton->GetValue() == true)
	{
		ExpertPanel->Show(true);
		Layout();
	}
	else
	{
		ExpertPanel->Show(false);
		Layout();
	}
}

void RefineCTFPanel::ReDrawActiveReferences()
{
	Active3DReferencesListCtrl->ClearAll();

	if (RefinementPackageComboBox->GetSelection() >= 0)
	{
		Active3DReferencesListCtrl->InsertColumn(0, "Class No.", wxLIST_FORMAT_LEFT);
		Active3DReferencesListCtrl->InsertColumn(1, "Active Reference Volume", wxLIST_FORMAT_LEFT);

		Active3DReferencesListCtrl->SetItemCount(refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).references_for_next_refinement.GetCount());

		if (refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).references_for_next_refinement.GetCount() > 0)
		{
			Active3DReferencesListCtrl->RefreshItems(0, refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).references_for_next_refinement.GetCount() -1);

			Active3DReferencesListCtrl->SetColumnWidth(0, Active3DReferencesListCtrl->ReturnGuessAtColumnTextWidth(0));
			Active3DReferencesListCtrl->SetColumnWidth(1, Active3DReferencesListCtrl->ReturnGuessAtColumnTextWidth(1));
		}
	}
}

void RefineCTFPanel::FillRefinementPackagesComboBox()
{
	if (RefinementPackageComboBox->FillComboBox() == false) NewRefinementPackageSelected();
}

void RefineCTFPanel::FillInputParamsComboBox()
{
	if (RefinementPackageComboBox->GetCount() > 0 ) InputParametersComboBox->FillComboBox(RefinementPackageComboBox->GetSelection(), true);
}

void RefineCTFPanel::NewRefinementPackageSelected()
{
	selected_refinement_package = RefinementPackageComboBox->GetSelection();
	FillInputParamsComboBox();
	SetDefaults();
	ReDrawActiveReferences();
	//wxPrintf("New Refinement Package Selection\n");

}

void RefineCTFPanel::OnRefinementPackageComboBox( wxCommandEvent& event )
{

	NewRefinementPackageSelected();

}

void RefineCTFPanel::OnInputParametersComboBox( wxCommandEvent& event )
{
	//SetDefaults();
}

void RefineCTFPanel::TerminateButtonClick( wxCommandEvent& event )
{
	main_frame->job_controller.KillJob(my_job_id);

	active_mask_thread_id = -1;
	active_orth_thread_id = -1;

	WriteBlueText("Terminated Job");
	TimeRemainingText->SetLabel("Time Remaining : Terminated");
	CancelAlignmentButton->Show(false);
	FinishButton->Show(true);
	ProgressPanel->Layout();
/*
	if (buffered_results != NULL)
	{
		delete [] buffered_results;
		buffered_results = NULL;
	}*/
}

void RefineCTFPanel::OnVolumeListItemActivated( wxListEvent& event )
{
	MyVolumeChooserDialog *dialog = new MyVolumeChooserDialog(this);

	dialog->ComboBox->SetSelection(volume_asset_panel->ReturnArrayPositionFromAssetID(refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).references_for_next_refinement.Item(event.GetIndex())) + 1);
	dialog->Fit();
	if (dialog->ShowModal() == wxID_OK)
	{
		if (dialog->selected_volume_id != refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).references_for_next_refinement.Item(event.GetIndex()))
		{
			refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).references_for_next_refinement.Item(event.GetIndex()) = dialog->selected_volume_id;
					// Change in database..
			main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li SET VOLUME_ASSET_ID=%li WHERE CLASS_NUMBER=%li;", refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).asset_id, dialog->selected_volume_id, event.GetIndex() + 1));

			ReDrawActiveReferences();
		}
	}
	dialog->Destroy();
}

void RefineCTFPanel::FinishButtonClick( wxCommandEvent& event )
{
	ProgressBar->SetValue(0);
	TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
	FinishButton->Show(false);

	InputParamsPanel->Show(true);
	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	output_textctrl->Clear();
	ShowRefinementResultsPanel->Show(false);
	ShowRefinementResultsPanel->Clear();
	//CTFResultsPanel->Show(false);
	//graph_is_hidden = true;
	InfoPanel->Show(true);

	if (my_refinement_manager.output_refinement != NULL)
	{
		delete my_refinement_manager.output_refinement;
		my_refinement_manager.output_refinement = NULL;
	}

	if (ExpertToggleButton->GetValue() == true) ExpertPanel->Show(true);
	else ExpertPanel->Show(false);
	running_job = false;
	Layout();

	//CTFResultsPanel->CTF2DResultsPanel->should_show = false;
	//CTFResultsPanel->CTF2DResultsPanel->Refresh();

}




void RefineCTFPanel::StartRefinementClick( wxCommandEvent& event )
{
	stopwatch.Start();
	my_refinement_manager.BeginRefinementCycle();
}

void RefineCTFPanel::WriteInfoText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void RefineCTFPanel::WriteBlueText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLUE));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void RefineCTFPanel::WriteErrorText(wxString text_to_write)
{
	 output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
	 output_textctrl->AppendText(text_to_write);

	 if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}


void RefineCTFPanel::FillRunProfileComboBoxes()
{
	ReconstructionRunProfileComboBox->FillWithRunProfiles();
	RefinementRunProfileComboBox->FillWithRunProfiles();
}
void RefineCTFPanel::OnSocketJobResultMsg(JobResult &received_result)
{
	my_refinement_manager.ProcessJobResult(&received_result);


}

void RefineCTFPanel::OnSocketJobResultQueueMsg(ArrayofJobResults &received_queue)
{
	if (my_refinement_manager.running_job_type == ESTIMATE_BEAMTILT)
	{
		int results_in_job = received_queue.GetCount();
		long current_time = time(NULL);

		if (my_refinement_manager.number_of_received_results == 0)
		{
			my_refinement_manager.current_job_starttime = current_time;
			my_refinement_manager.time_of_last_update = 0;
			my_refinement_manager.number_of_received_results += results_in_job;
		}
		else
		{
			my_refinement_manager.number_of_received_results += results_in_job;

			if (current_time != my_refinement_manager.time_of_last_update)
			{
				int current_percentage = float(my_refinement_manager.number_of_received_results) / float(my_refinement_manager.number_of_expected_results) * 100.0;
				my_refinement_manager.time_of_last_update = current_time;
				if (current_percentage > 100) current_percentage = 100;
				ProgressBar->SetValue(current_percentage);

				long job_time = current_time - my_refinement_manager.current_job_starttime;
				float seconds_per_job = float(job_time) / float(my_refinement_manager.number_of_received_results - 1);
				long seconds_remaining = float(my_refinement_manager.number_of_expected_results - my_refinement_manager.number_of_received_results) * seconds_per_job;

				wxTimeSpan time_remaining = wxTimeSpan(0,0,seconds_remaining);
				TimeRemainingText->SetLabel(time_remaining.Format("Time Remaining : %Hh:%Mm:%Ss"));
			}
		}
	}
	else
	{
		for (int counter = 0; counter < received_queue.GetCount(); counter++)
		{
			my_refinement_manager.ProcessJobResult(&received_queue.Item(counter));
		}
	}
}

void RefineCTFPanel::SetNumberConnectedText(wxString wanted_text)
{
	NumberConnectedText->SetLabel(wanted_text);
}

void RefineCTFPanel::SetTimeRemainingText(wxString wanted_text)
{
	TimeRemainingText->SetLabel(wanted_text);
}

void RefineCTFPanel::OnSocketAllJobsFinished()
{
	my_refinement_manager.ProcessAllJobsFinished();
}

CTFRefinementManager::CTFRefinementManager()
{
	input_refinement = NULL;
	output_refinement = NULL;

}

void CTFRefinementManager::SetParent(RefineCTFPanel *wanted_parent)
{
	my_parent = wanted_parent;
}

void CTFRefinementManager::BeginRefinementCycle()
{
	start_with_reconstruction = false;

	active_low_resolution_limit = my_parent->LowResolutionLimitTextCtrl->ReturnValue();
	active_high_resolution_limit = my_parent->HighResolutionLimitTextCtrl->ReturnValue();
	active_mask_radius = my_parent->MaskRadiusTextCtrl->ReturnValue();
	active_signed_cc_limit = 0.0f;
	active_defocus_search_range = my_parent->DefocusSearchRangeTextCtrl->ReturnValue();
	active_defocus_search_step = my_parent->DefocusSearchStepTextCtrl->ReturnValue();
	active_inner_mask_radius = my_parent->InnerMaskRadiusTextCtrl->ReturnValue();
	active_resolution_limit_rec = my_parent->ReconstructionResolutionLimitTextCtrl->ReturnValue();
	active_score_weight_conversion	= my_parent->ScoreToWeightConstantTextCtrl->ReturnValue();
	active_score_threshold	= my_parent->ReconstructionScoreThreshold->ReturnValue();
	active_adjust_scores = my_parent->AdjustScoreForDefocusYesRadio->GetValue();
	active_crop_images	= my_parent->AutoCropYesRadioButton->GetValue();
	active_should_apply_blurring = my_parent->ApplyBlurringYesRadioButton->GetValue();
	active_smoothing_factor = my_parent->SmoothingFactorTextCtrl->ReturnValue();
	active_should_mask = my_parent->UseMaskCheckBox->GetValue();
	active_should_auto_mask = my_parent->AutoMaskYesRadioButton->GetValue();

	active_ctf_refinement = my_parent->RefineCTFCheckBox->IsChecked();
	active_beamtilt_refinement = my_parent->RefineBeamTiltCheckBox->IsChecked();

	if (my_parent->MaskSelectPanel->ReturnSelection() >= 0) active_mask_asset_id = volume_asset_panel->ReturnAssetID(my_parent->MaskSelectPanel->ReturnSelection());
	else active_mask_asset_id = -1;
	if (my_parent->MaskSelectPanel->ReturnSelection() >= 0)	active_mask_filename = volume_asset_panel->ReturnAssetLongFilename(my_parent->MaskSelectPanel->ReturnSelection());
	else active_mask_filename = "";

	active_should_low_pass_filter_mask = my_parent->LowPassMaskYesRadio->GetValue();
	active_mask_filter_resolution = my_parent->MaskFilterResolutionText->ReturnValue();
	active_mask_edge = my_parent->MaskEdgeTextCtrl->ReturnValue();
	active_mask_weight = my_parent->MaskWeightTextCtrl->ReturnValue();

	active_refinement_run_profile = run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()];
	active_reconstruction_run_profile = run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()];

	active_refinement_package = &refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection());

	current_refinement_package_asset_id = active_refinement_package->asset_id;
	current_input_refinement_id = active_refinement_package->refinement_ids[my_parent->InputParametersComboBox->GetSelection()];

	int class_counter;
	int number_of_classes = active_refinement_package->number_of_classes;

	wxString blank_string = "";
	current_reference_filenames.Clear();
	current_reference_filenames.Add(blank_string, number_of_classes);

	current_reference_asset_ids.Clear();
	current_reference_asset_ids.Add(-1, number_of_classes);

	// check scratch directory.
	global_delete_refinectf_scratch();

	// get the data..

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		if (active_refinement_package->references_for_next_refinement[class_counter] == -1) start_with_reconstruction = true;
	}

	my_parent->Freeze();

	my_parent->InputParamsPanel->Show(false);
	my_parent->StartPanel->Show(false);
	my_parent->ProgressPanel->Show(true);
	my_parent->ExpertPanel->Show(false);
	my_parent->InfoPanel->Show(false);
	my_parent->OutputTextPanel->Show(true);

	if (active_ctf_refinement == true)
	{
		my_parent->ShowRefinementResultsPanel->Clear();
		if (my_parent->ShowRefinementResultsPanel->LeftRightSplitter->IsSplit() == true) my_parent->ShowRefinementResultsPanel->LeftRightSplitter->Unsplit();
		if (my_parent->ShowRefinementResultsPanel->TopBottomSplitter->IsSplit() == true) my_parent->ShowRefinementResultsPanel->TopBottomSplitter->Unsplit();
		my_parent->ShowRefinementResultsPanel->Show(true);
	}

	my_parent->Layout();
	my_parent->Thaw();

	if (start_with_reconstruction == true)
	{
		input_refinement = main_frame->current_project.database.GetRefinementByID(current_input_refinement_id);
		output_refinement = input_refinement;
		current_output_refinement_id = input_refinement->refinement_id;

		// after this job, the resolution statistics will be real, so update..

		output_refinement->resolution_statistics_are_generated = false;

		SetupReconstructionJob();
		RunReconstructionJob();
	}
	else
	{
		input_refinement = main_frame->current_project.database.GetRefinementByID(current_input_refinement_id);
		output_refinement = new Refinement;

		*output_refinement = *input_refinement;

		// we need to set the currently selected reference filenames..

		for (class_counter = 0; class_counter < number_of_classes; class_counter++)
		{
			if (volume_asset_panel->ReturnAssetPointer(volume_asset_panel->ReturnArrayPositionFromAssetID(active_refinement_package->references_for_next_refinement[class_counter]))->x_size != active_refinement_package->stack_box_size ||
				volume_asset_panel->ReturnAssetPointer(volume_asset_panel->ReturnArrayPositionFromAssetID(active_refinement_package->references_for_next_refinement[class_counter]))->y_size != active_refinement_package->stack_box_size ||
				volume_asset_panel->ReturnAssetPointer(volume_asset_panel->ReturnArrayPositionFromAssetID(active_refinement_package->references_for_next_refinement[class_counter]))->z_size != active_refinement_package->stack_box_size ||
				fabsf(float(volume_asset_panel->ReturnAssetPointer(volume_asset_panel->ReturnArrayPositionFromAssetID(active_refinement_package->references_for_next_refinement[class_counter]))->pixel_size) - input_refinement->resolution_statistics_pixel_size) > 0.01f)
			{
				my_parent->WriteErrorText("Error: Reference volume has different dimensions / pixel size from the input stack.  This will currently not work.");
			}

			current_reference_filenames.Item(class_counter) = volume_asset_panel->ReturnAssetLongFilename(volume_asset_panel->ReturnArrayPositionFromAssetID(active_refinement_package->references_for_next_refinement[class_counter]));
			current_reference_asset_ids.Item(class_counter) = volume_asset_panel->ReturnAssetID(volume_asset_panel->ReturnArrayPositionFromAssetID(active_refinement_package->references_for_next_refinement[class_counter]));
		}

		if (my_parent->UseMaskCheckBox->GetValue() == true || my_parent->AutoMaskYesRadioButton->GetValue() == true)
		{
			DoMasking();
		}
		else
		{
			SetupRefinementJob();
			RunRefinementJob();
		}
	}
}


void CTFRefinementManager::RunRefinementJob()
{
	running_job_type = REFINEMENT;

	number_of_received_results = 0;
	number_of_expected_results = input_refinement->number_of_particles;

	// setup curve..

	defocus_change_histogram.ClearData();
	//defocus_change_histogram.SetupXAxis(-active_defocus_search_range, active_defocus_search_range, int(active_defocus_search_range / active_defocus_search_step));

	for (int defocus_i = -myround(float(active_defocus_search_range)/float(active_defocus_search_step)); defocus_i <= myround(float(active_defocus_search_range)/float(active_defocus_search_step)); defocus_i++)
	{
		defocus_change_histogram.AddPoint(defocus_i * active_defocus_search_step, 0.0f);
	}

	time_of_last_histogram_update = 0;

	my_parent->ShowRefinementResultsPanel->DefocusHistorgramPlotPanel->Clear();
	my_parent->ShowRefinementResultsPanel->DefocusHistorgramPlotPanel->AddCurve(defocus_change_histogram, *wxBLUE);
	my_parent->ShowRefinementResultsPanel->DefocusHistorgramPlotPanel->Draw(-active_defocus_search_range, active_defocus_search_range, 0, 1);



	output_refinement->SizeAndFillWithEmpty(input_refinement->number_of_particles, input_refinement->number_of_classes);
	//wxPrintf("Output refinement has %li particles and %i classes\n", output_refinement->number_of_particles, input_refinement->number_of_classes);
	current_output_refinement_id = main_frame->current_project.database.ReturnHighestRefinementID() + 1;

	output_refinement->refinement_id = current_output_refinement_id;
	output_refinement->refinement_package_asset_id = current_refinement_package_asset_id;

	if (my_parent->RefineCTFCheckBox->IsChecked() == true && my_parent->RefineBeamTiltCheckBox->IsChecked() == true)
	{
		output_refinement->name = wxString::Format("Defocus & Beam Tilt Refinement #%li", current_output_refinement_id);
	}
	else if (my_parent->RefineCTFCheckBox->IsChecked() == true)
	{
		output_refinement->name = wxString::Format("Defocus Refinement #%li", current_output_refinement_id);
	}
	else output_refinement->name = wxString::Format("Beam Tilt Refinement #%li", current_output_refinement_id);

	output_refinement->resolution_statistics_are_generated = false;
	output_refinement->datetime_of_run = wxDateTime::Now();
	output_refinement->starting_refinement_id = current_input_refinement_id;

	for (int class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		output_refinement->class_refinement_results[class_counter].low_resolution_limit = active_low_resolution_limit;

		if (active_ctf_refinement == true) output_refinement->class_refinement_results[class_counter].high_resolution_limit = active_high_resolution_limit;
		else output_refinement->class_refinement_results[class_counter].high_resolution_limit = input_refinement->class_refinement_results[class_counter].high_resolution_limit;

		output_refinement->class_refinement_results[class_counter].mask_radius = active_mask_radius;
		output_refinement->class_refinement_results[class_counter].signed_cc_resolution_limit = active_signed_cc_limit;
		output_refinement->class_refinement_results[class_counter].global_resolution_limit = active_high_resolution_limit;
		output_refinement->class_refinement_results[class_counter].global_mask_radius = 0.0f;
		output_refinement->class_refinement_results[class_counter].number_results_to_refine = 0;
		output_refinement->class_refinement_results[class_counter].angular_search_step = 0.0f;
		output_refinement->class_refinement_results[class_counter].search_range_x = 0.0f;
		output_refinement->class_refinement_results[class_counter].search_range_y = 0.0f;
		output_refinement->class_refinement_results[class_counter].classification_resolution_limit = 0.0f;
		output_refinement->class_refinement_results[class_counter].should_focus_classify = false;
		output_refinement->class_refinement_results[class_counter].sphere_x_coord = 0.0f;
		output_refinement->class_refinement_results[class_counter].sphere_y_coord = 0.0f;
		output_refinement->class_refinement_results[class_counter].sphere_z_coord = 0.0f;
		output_refinement->class_refinement_results[class_counter].sphere_radius = active_sphere_radius;
		output_refinement->class_refinement_results[class_counter].should_refine_ctf = active_ctf_refinement;
		output_refinement->class_refinement_results[class_counter].defocus_search_range = active_defocus_search_range;
		output_refinement->class_refinement_results[class_counter].defocus_search_step = active_defocus_search_step;

		output_refinement->class_refinement_results[class_counter].should_auto_mask = active_should_auto_mask;
		output_refinement->class_refinement_results[class_counter].should_refine_input_params = false;
		output_refinement->class_refinement_results[class_counter].should_use_supplied_mask = active_should_mask;
		output_refinement->class_refinement_results[class_counter].mask_asset_id = active_mask_asset_id;
		output_refinement->class_refinement_results[class_counter].mask_edge_width = active_mask_edge;
		output_refinement->class_refinement_results[class_counter].outside_mask_weight = active_mask_weight;
		output_refinement->class_refinement_results[class_counter].should_low_pass_filter_mask = active_should_low_pass_filter_mask;
		output_refinement->class_refinement_results[class_counter].filter_resolution = active_mask_filter_resolution;
	}

	output_refinement->percent_used = 100.0f;

	output_refinement->resolution_statistics_box_size = input_refinement->resolution_statistics_box_size;
	output_refinement->resolution_statistics_pixel_size = input_refinement->resolution_statistics_pixel_size;

	// launch a controller

	current_job_starttime = time(NULL);
	time_of_last_update = current_job_starttime;
	my_parent->ShowRefinementResultsPanel->DefocusHistorgramPlotPanel->Clear();

	if (active_ctf_refinement == true && active_beamtilt_refinement == true)
	{
		my_parent->WriteBlueText("Refining defocus and calculating phase difference image...\n");
	}
	else
	if (active_ctf_refinement == true)
	{
		my_parent->WriteBlueText("Refining defocus...\n");
	}
	else
	{
		my_parent->WriteBlueText("Calculating phase difference image...\n");
	}

	current_job_id = main_frame->job_controller.AddJob(my_parent, active_refinement_run_profile.manager_command, active_refinement_run_profile.gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		my_parent->SetNumberConnectedTextToZeroAndStartTracking();
	}




	my_parent->ProgressBar->Pulse();
}

void CTFRefinementManager::RunBeamTiltEstimationJob()
{
	running_job_type = ESTIMATE_BEAMTILT;
	number_of_received_results = 0;
	best_score = FLT_MAX;

	int number_of_refinement_processes = active_refinement_run_profile.ReturnTotalJobs();
	int number_of_refinement_jobs = number_of_refinement_processes;

	const int total_number_of_positions = 290880; // hard coded based on FindBeamTilt, not very nice.
	number_of_expected_results = total_number_of_positions;
	int number_of_positions_per_thread = int(ceilf(total_number_of_positions / number_of_refinement_jobs));


	my_parent->current_job_package.Reset(active_refinement_run_profile, "estimate_beamtilt", number_of_refinement_jobs);

	for (int counter = 0; counter < number_of_refinement_jobs; counter++)
	{
		wxString input_phase_difference_image		= main_frame->ReturnRefineCTFScratchDirectory() + "phase_output.mrc";//"/tmp/phase_diff.mrc";
		float 	 pixel_size							= active_refinement_package->output_pixel_size;
		float 	 voltage_kV							= active_refinement_package->contained_particles[0].microscope_voltage; // assume all the same
		float 	 spherical_aberration_mm			= active_refinement_package->contained_particles[0].spherical_aberration; // assume all the same
		int 	 first_position_to_search			= counter*number_of_positions_per_thread;
		int 	 last_position_to_search 			= (counter + 1) * number_of_positions_per_thread - 1;

		if (first_position_to_search < 0) first_position_to_search = 0;
		if (last_position_to_search > total_number_of_positions - 1) last_position_to_search = total_number_of_positions - 1;


		my_parent->current_job_package.AddJob("tfffii", 	input_phase_difference_image.ToUTF8().data(),
															pixel_size,
															voltage_kV,
															spherical_aberration_mm,
															first_position_to_search,
															last_position_to_search);
	}

	my_parent->WriteBlueText("Estimating beam tilt...");

	// ok launch it..

	current_job_id = main_frame->job_controller.AddJob(my_parent, active_refinement_run_profile.manager_command, active_refinement_run_profile.gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		my_parent->SetNumberConnectedTextToZeroAndStartTracking();
	}

	my_parent->ProgressBar->Pulse();
}

void CTFRefinementManager::SetupMerge3dJob()
{

	int number_of_reconstruction_jobs = active_reconstruction_run_profile.ReturnTotalJobs();

	int class_counter;

	my_parent->current_job_package.Reset(active_reconstruction_run_profile, "merge3d", active_refinement_package->number_of_classes);

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		wxString output_reconstruction_1			= "/dev/null";
		wxString output_reconstruction_2			= "/dev/null";
		wxString output_reconstruction_filtered		= main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/volume_%li_%i.mrc", output_refinement->refinement_id, class_counter + 1);

		current_reference_filenames.Item(class_counter) = output_reconstruction_filtered;

		wxString output_resolution_statistics		= "/dev/null";
		float 	 molecular_mass_kDa					= active_refinement_package->estimated_particle_weight_in_kda;
		float    inner_mask_radius					= active_inner_mask_radius;
		float    outer_mask_radius					= active_mask_radius;
		wxString dump_file_seed_1 					= main_frame->ReturnRefineCTFScratchDirectory() + wxString::Format("dump_file_%li_%i_odd_.dump", current_output_refinement_id, class_counter);
		wxString dump_file_seed_2 					= main_frame->ReturnRefineCTFScratchDirectory() + wxString::Format("dump_file_%li_%i_even_.dump", current_output_refinement_id, class_counter);

		bool save_orthogonal_views_image = true;
		wxString orthogonal_views_filename = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/OrthViews/volume_%li_%i.mrc", output_refinement->refinement_id, class_counter + 1);
		float weiner_nominator = 1.0f;
	  float alignment_res = 5.0f;

		my_parent->current_job_package.AddJob("ttttfffttibtiff",	output_reconstruction_1.ToUTF8().data(),
                                                              output_reconstruction_2.ToUTF8().data(),
                                                              output_reconstruction_filtered.ToUTF8().data(),
                                                              output_resolution_statistics.ToUTF8().data(),
                                                              molecular_mass_kDa, 
                                                              inner_mask_radius, 
                                                              outer_mask_radius,
                                                              dump_file_seed_1.ToUTF8().data(),
                                                              dump_file_seed_2.ToUTF8().data(),
                                                              class_counter + 1,
                                                              save_orthogonal_views_image,
                                                              orthogonal_views_filename.ToUTF8().data(),
                                                              number_of_reconstruction_jobs, 
                                                              weiner_nominator,
                                                              alignment_res);
	}
}



void CTFRefinementManager::RunMerge3dJob()
{
	running_job_type = MERGE;

	// start job..

	if (output_refinement->number_of_classes > 1) my_parent->WriteBlueText("Merging and Filtering Reconstructions...");
	else
	my_parent->WriteBlueText("Merging and Filtering Reconstruction...");

	current_job_id = main_frame->job_controller.AddJob(my_parent, active_reconstruction_run_profile.manager_command, active_reconstruction_run_profile.gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		my_parent->StartPanel->Show(false);
		my_parent->ProgressPanel->Show(true);

		my_parent->ExpertPanel->Show(false);
		my_parent->InfoPanel->Show(false);
		my_parent->OutputTextPanel->Show(true);
			//	CTFResultsPanel->Show(true);

		my_parent->ExpertToggleButton->Enable(false);
		my_parent->RefinementPackageComboBox->Enable(false);
		my_parent->InputParametersComboBox->Enable(false);

		my_parent->SetNumberConnectedTextToZeroAndStartTracking();

		}

		my_parent->ProgressBar->Pulse();
}


void CTFRefinementManager::SetupReconstructionJob()
{
	wxArrayString written_parameter_files;
	written_parameter_files = output_refinement->WritecisTEMStarFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/beam_tilt_output_par", 1.0f, 0.0f, true);

	int class_counter;
	long counter;
	int job_counter;
	long number_of_reconstruction_jobs;
	long number_of_reconstruction_processes;
	float current_particle_counter;

	long number_of_particles;
	float particles_per_job;

	number_of_reconstruction_processes = active_reconstruction_run_profile.ReturnTotalJobs();
	number_of_reconstruction_jobs = number_of_reconstruction_processes;

	number_of_particles = active_refinement_package->contained_particles.GetCount();

	if (number_of_particles - number_of_reconstruction_jobs < number_of_reconstruction_jobs) particles_per_job = 1;
	else particles_per_job = float(number_of_particles - number_of_reconstruction_jobs) / float(number_of_reconstruction_jobs);

	my_parent->current_job_package.Reset(active_reconstruction_run_profile, "reconstruct3d", number_of_reconstruction_jobs * active_refinement_package->number_of_classes);

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		current_particle_counter = 1.0;

		for (job_counter = 0; job_counter < number_of_reconstruction_jobs; job_counter++)
		{
			wxString input_particle_stack 		= active_refinement_package->stack_filename;
			wxString input_parameter_file 		= written_parameter_files[class_counter];
			wxString output_reconstruction_1    		= "/dev/null";
			wxString output_reconstruction_2			= "/dev/null";
			wxString output_reconstruction_filtered		= "/dev/null";
			wxString output_resolution_statistics		= "/dev/null";
			wxString my_symmetry						= active_refinement_package->symmetry;

			long	 first_particle						= myroundint(current_particle_counter);

			current_particle_counter += particles_per_job;
			if (current_particle_counter > number_of_particles  || job_counter == number_of_reconstruction_jobs - 1) current_particle_counter = number_of_particles;

			long	 last_particle						= myroundint(current_particle_counter);
			current_particle_counter+=1.0;

			//float 	 pixel_size							= active_refinement_package->contained_particles[0].pixel_size;
			//float    voltage_kV							= active_refinement_package->contained_particles[0].microscope_voltage;
			//float 	 spherical_aberration_mm			= active_refinement_package->contained_particles[0].spherical_aberration;
			//float    amplitude_contrast					= active_refinement_package->contained_particles[0].amplitude_contrast;
			float 	 molecular_mass_kDa					= active_refinement_package->estimated_particle_weight_in_kda;
			float    inner_mask_radius					= active_inner_mask_radius;
			float    outer_mask_radius					= active_mask_radius;
			float    resolution_limit_rec				= active_resolution_limit_rec;
			float    score_weight_conversion			= active_score_weight_conversion;
			float    score_threshold					= active_score_threshold;
			bool	 adjust_scores						= active_adjust_scores;
			bool	 invert_contrast					= active_refinement_package->stack_has_white_protein;
			bool	 crop_images						= active_crop_images;
			bool	 dump_arrays						= true;
			wxString dump_file_1 						= main_frame->ReturnRefineCTFScratchDirectory() + wxString::Format("dump_file_%li_%i_odd_%i.dump", current_output_refinement_id, class_counter, job_counter +1);
			wxString dump_file_2 						= main_frame->ReturnRefineCTFScratchDirectory() + wxString::Format("dump_file_%li_%i_even_%i.dump", current_output_refinement_id, class_counter, job_counter + 1);

			wxString input_reconstruction;
			bool	 use_input_reconstruction;
			float pixel_size_of_reference = active_refinement_package->output_pixel_size;

			if (active_should_apply_blurring == true)
			{
				// do we have a reference..

				if (active_refinement_package->references_for_next_refinement[class_counter] == -1)
				{
					input_reconstruction			= "/dev/null";
					use_input_reconstruction		= false;
				}
				else
				{
					input_reconstruction = current_reference_filenames.Item(class_counter);//volume_asset_panel->ReturnAssetLongFilename(volume_asset_panel->ReturnArrayPositionFromAssetID(refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).references_for_next_refinement[class_counter]));
					use_input_reconstruction = true;
				}


			}
			else
			{
				input_reconstruction			= "/dev/null";
				use_input_reconstruction		= false;
			}

			float    resolution_limit_ref               = active_high_resolution_limit;
			float	 smoothing_factor					= active_smoothing_factor;
			float    padding							= 1.0f;
			bool	 normalize_particles				= true;
			bool	 exclude_blank_edges				= false;
			bool	 split_even_odd						= false;
			bool     centre_mass                        = false;
			int		 max_threads						= 1;



			bool threshold_input_3d = true;
			int correct_ewald_sphere = 0;

			my_parent->current_job_package.AddJob("ttttttttiiffffffffffbbbbbbbbbbttii",
																		input_particle_stack.ToUTF8().data(),
																		input_parameter_file.ToUTF8().data(),
																		input_reconstruction.ToUTF8().data(),
																		output_reconstruction_1.ToUTF8().data(),
																		output_reconstruction_2.ToUTF8().data(),
																		output_reconstruction_filtered.ToUTF8().data(),
																		output_resolution_statistics.ToUTF8().data(),
																		my_symmetry.ToUTF8().data(),
																		first_particle,
																		last_particle,
																		pixel_size_of_reference,
																		molecular_mass_kDa,
																		inner_mask_radius,
																		outer_mask_radius,
																		resolution_limit_rec,
																		resolution_limit_ref,
																		score_weight_conversion,
																		score_threshold,
																		smoothing_factor,
																		padding,
																		normalize_particles,
																		adjust_scores,
																		invert_contrast,
																		exclude_blank_edges,
																		crop_images,
																		split_even_odd,
																		centre_mass,
																		use_input_reconstruction,
																		threshold_input_3d,
																		dump_arrays,
																		dump_file_1.ToUTF8().data(),
																		dump_file_2.ToUTF8().data(),
																		correct_ewald_sphere,
																		max_threads);




		}
	}
}


// for now we take the paramter

void CTFRefinementManager::RunReconstructionJob()
{
	running_job_type = RECONSTRUCTION;
	number_of_received_results = 0;
	number_of_expected_results = output_refinement->ReturnNumberOfActiveParticlesInFirstClass() * output_refinement->number_of_classes;

	if (output_refinement->number_of_classes > 1) my_parent->WriteBlueText("Calculating Reconstructions...");
	else my_parent->WriteBlueText("Calculating Reconstruction...");

	current_job_id = main_frame->job_controller.AddJob(my_parent, active_reconstruction_run_profile.manager_command, active_reconstruction_run_profile.gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		my_parent->SetNumberConnectedTextToZeroAndStartTracking();
	}
	my_parent->ProgressBar->Pulse();
}

void CTFRefinementManager::SetupRefinementJob()
{

	int class_counter;
	long counter;
	long number_of_refinement_jobs;
	int number_of_refinement_processes;
	float current_particle_counter;

	long number_of_particles;
	float particles_per_job;

	int image_number_for_gui = 0;;
	int number_of_jobs_per_image_in_gui;

	int best_class;
	float highest_occupancy;


	//input_refinement->WritecisTEMStarFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/input_par");
	input_refinement->WriteResolutionStatistics(main_frame->current_project.parameter_file_directory.GetFullPath() + "/input_stats");

	// make a merged star file for all the classes, so that each image is refined against it's best class..

	cisTEMParameters output_parameters;
	output_parameters.PreallocateMemoryAndBlank(input_refinement->number_of_particles);
	output_parameters.parameters_to_write.SetActiveParameters(POSITION_IN_STACK | IMAGE_IS_ACTIVE | PSI | THETA | PHI | X_SHIFT | Y_SHIFT | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | PHASE_SHIFT | OCCUPANCY | LOGP | SIGMA | SCORE | PIXEL_SIZE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y | REFERENCE_3D_FILENAME);


	for (counter = 0; counter < input_refinement->number_of_particles; counter++)
	{
		highest_occupancy = -FLT_MAX;
		for (class_counter = 0; class_counter < input_refinement->number_of_classes; class_counter++)
		{
			if (input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].occupancy > highest_occupancy)
			{
				best_class = class_counter;
				highest_occupancy = input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].occupancy;
			}
		}

		output_parameters.all_parameters[counter].position_in_stack = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].position_in_stack;
		output_parameters.all_parameters[counter].image_is_active = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].image_is_active;
		output_parameters.all_parameters[counter].psi = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].psi;
		output_parameters.all_parameters[counter].theta = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].theta;
		output_parameters.all_parameters[counter].phi = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].phi;
		output_parameters.all_parameters[counter].x_shift = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].xshift;
		output_parameters.all_parameters[counter].y_shift = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].yshift;
		output_parameters.all_parameters[counter].defocus_1 = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].defocus1;
		output_parameters.all_parameters[counter].defocus_2 = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].defocus2;
		output_parameters.all_parameters[counter].defocus_angle = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].defocus_angle;
		output_parameters.all_parameters[counter].phase_shift = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].phase_shift;
		output_parameters.all_parameters[counter].occupancy = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].occupancy;
		output_parameters.all_parameters[counter].logp = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].logp;
		output_parameters.all_parameters[counter].score = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].score;
		output_parameters.all_parameters[counter].pixel_size = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].pixel_size;
		output_parameters.all_parameters[counter].microscope_voltage_kv = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].microscope_voltage_kv;
		output_parameters.all_parameters[counter].microscope_spherical_aberration_mm = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].microscope_spherical_aberration_mm;
		output_parameters.all_parameters[counter].amplitude_contrast = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].amplitude_contrast;
		output_parameters.all_parameters[counter].beam_tilt_x = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].beam_tilt_x;
		output_parameters.all_parameters[counter].beam_tilt_y = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].beam_tilt_y;
		output_parameters.all_parameters[counter].image_shift_x = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].image_shift_x;
		output_parameters.all_parameters[counter].image_shift_y = input_refinement->class_refinement_results[best_class].particle_refinement_results[counter].image_shift_y;
		output_parameters.all_parameters[counter].reference_3d_filename = current_reference_filenames.Item(best_class);
	}

	wxString output_star_filename = wxString::Format("%s/refine_ctf_input_star_%li.cistem", main_frame->current_project.parameter_file_directory.GetFullPath(), input_refinement->refinement_id);
	output_parameters.WriteTocisTEMBinaryFile(output_star_filename);

	//	wxPrintf("Input refinement has %li particles\n", input_refinement->number_of_particles);

	// for now, number of jobs is number of processes -1 (master)..

	number_of_refinement_processes = active_refinement_run_profile.ReturnTotalJobs();
	number_of_refinement_jobs = number_of_refinement_processes;

	number_of_particles = active_refinement_package->contained_particles.GetCount();
	if (number_of_particles - number_of_refinement_jobs < number_of_refinement_jobs) particles_per_job = 1;
	else particles_per_job = float(number_of_particles - number_of_refinement_jobs) / float(number_of_refinement_jobs);



	my_parent->current_job_package.Reset(active_refinement_run_profile, "refine_ctf", number_of_refinement_jobs * active_refinement_package->number_of_classes);

	current_particle_counter = 1;

	for (counter = 0; counter < number_of_refinement_jobs; counter++)
	{
		wxString input_particle_images					= active_refinement_package->stack_filename;
		wxString input_reconstruction					= current_reference_filenames.Item(0);
		wxString input_reconstruction_statistics 		= main_frame->current_project.parameter_file_directory.GetFullPath() + wxString::Format("/input_stats_%li_1.txt", current_input_refinement_id, class_counter + 1);
		bool	 use_statistics							= true;

		wxString ouput_shift_filename					= "/dev/null";
		wxString ouput_star_filename					= "/dev/null";
		wxString ouput_phase_difference_image			= main_frame->ReturnRefineCTFScratchDirectory() + "phase_output.mrc";//"/tmp/phase_diff.mrc";
		wxString ouput_beamtilt_image					= "/dev/null";
		wxString ouput_difference_image					= "/dev/null";

		long	 first_particle							= myroundint(current_particle_counter);

		current_particle_counter += particles_per_job;
		if (current_particle_counter > number_of_particles  || counter == number_of_refinement_jobs) current_particle_counter = number_of_particles;

		long	 last_particle							= myroundint(current_particle_counter);
		current_particle_counter++;


#ifdef DEBUG
			wxString output_parameter_file = wxString::Format("/tmp/output_par_%li_%li_%i.star", first_particle, last_particle, class_counter);
#else
			wxString output_parameter_file = "/dev/null";
#endif

		// for now we take the paramters of the first image!!!!

		float 	 output_pixel_size						= active_refinement_package->output_pixel_size;
		float	 molecular_mass_kDa						= active_refinement_package->estimated_particle_weight_in_kda;
		float    outer_mask_radius						= active_mask_radius;
		float    inner_mask_radius                      = active_inner_mask_radius;
		float    low_resolution_limit					= active_low_resolution_limit;
		float    high_resolution_limit					= active_high_resolution_limit;
		//float	 signed_CC_limit						= active_signed_cc_limit;

		float	 defocus_search_range					= active_defocus_search_range;
		float	 defocus_step							= active_defocus_search_step;
		float	 padding								= 1.0;

		bool ctf_refinement 							= active_ctf_refinement;
		bool beamtilt_refinement 						= active_beamtilt_refinement;

		bool invert_contrast							= active_refinement_package->stack_has_white_protein;

		bool normalize_particles = true;
		bool exclude_blank_edges = false;
		bool normalize_input_3d;

		if (active_should_apply_blurring == true) normalize_input_3d = false;
		else normalize_input_3d = true;

		bool threshold_input_3d = true;
		int max_threads = 1;

		image_number_for_gui = counter;
		number_of_jobs_per_image_in_gui = number_of_refinement_jobs;

		my_parent->current_job_package.AddJob("ttttbtttttiifffffffffbbbbbbbiii", 	input_particle_images.ToUTF8().data(),
																					output_star_filename.ToUTF8().data(),
																					input_reconstruction.ToUTF8().data(),
																					input_reconstruction_statistics.ToUTF8().data(),
																					use_statistics,
																					ouput_star_filename.ToUTF8().data(),
																					ouput_shift_filename.ToUTF8().data(),
																					ouput_phase_difference_image.ToUTF8().data(),
																					ouput_beamtilt_image.ToUTF8().data(),
																					ouput_difference_image.ToUTF8().data(),
																					first_particle,
																					last_particle,
																					output_pixel_size,
																					molecular_mass_kDa,
																					inner_mask_radius,
																					outer_mask_radius,
																					low_resolution_limit,
																					high_resolution_limit,
																					defocus_search_range,
																					defocus_step,
																					padding,
																					ctf_refinement,
																					beamtilt_refinement,
																					normalize_particles,
																					invert_contrast,
																					exclude_blank_edges,
																					normalize_input_3d,
																					threshold_input_3d,
																					image_number_for_gui,
																					number_of_jobs_per_image_in_gui,
																					max_threads);
	}
}

void CTFRefinementManager::ProcessJobResult(JobResult *result_to_process)
{
	if (running_job_type == REFINEMENT)
	{

		long current_particle = long(result_to_process->result_data[0] + 0.5) - 1;

		MyDebugAssertTrue(current_particle != -1, "Current Particle (%li) == -1", current_particle);

	//	wxPrintf("Received a refinement result for class #%i, particle %li\n", current_class + 1, current_particle + 1);
		//wxPrintf("output refinement has %i classes and %li particles\n", output_refinement->number_of_classes, output_refinement->number_of_particles);

		for (int class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
		{

			output_refinement->class_refinement_results[class_counter].particle_refinement_results[current_particle].defocus1 = result_to_process->result_data[1];
			output_refinement->class_refinement_results[class_counter].particle_refinement_results[current_particle].defocus2 = result_to_process->result_data[2];

			float defocus_change = output_refinement->class_refinement_results[class_counter].particle_refinement_results[current_particle].defocus1 - input_refinement->class_refinement_results[class_counter].particle_refinement_results[current_particle].defocus1;
			defocus_change_histogram.AddValueAtXUsingNearestNeighborInterpolation(defocus_change, 1);

			if (output_refinement->number_of_classes == 1)
			{
				output_refinement->class_refinement_results[class_counter].particle_refinement_results[current_particle].logp = result_to_process->result_data[3];
				output_refinement->class_refinement_results[class_counter].particle_refinement_results[current_particle].score = result_to_process->result_data[4];
			}
		}

		number_of_received_results++;
		//wxPrintf("received result!\n");
		long current_time = time(NULL);

		if (number_of_received_results == 1)
		{
			current_job_starttime = current_time;
			time_of_last_update = 0;
			my_parent->Layout();
		}
		else
		if (current_time != time_of_last_update)
		{
			int current_percentage = float(number_of_received_results) / float(number_of_expected_results) * 100.0;
			time_of_last_update = current_time;
			if (current_percentage > 100) current_percentage = 100;
			my_parent->ProgressBar->SetValue(current_percentage);
			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_results - 1);
			long seconds_remaining = float(number_of_expected_results - number_of_received_results) * seconds_per_job;

			wxTimeSpan time_remaining = wxTimeSpan(0,0,seconds_remaining);
			my_parent->TimeRemainingText->SetLabel(time_remaining.Format("Time Remaining : %Hh:%Mm:%Ss"));		
		}

		if (current_time - time_of_last_histogram_update > 5)
		{
			time_of_last_histogram_update = current_time;

			my_parent->ShowRefinementResultsPanel->DefocusHistorgramPlotPanel->Clear();
			my_parent->ShowRefinementResultsPanel->DefocusHistorgramPlotPanel->AddCurve(defocus_change_histogram, *wxBLUE);
			my_parent->ShowRefinementResultsPanel->DefocusHistorgramPlotPanel->Draw(-active_defocus_search_range, active_defocus_search_range, 0, defocus_change_histogram.ReturnMaximumValue());

			wxPrintf("\n\n\n\n");
		}

	}
	else
	if (running_job_type == ESTIMATE_BEAMTILT)
	{
		if (result_to_process->result_data[0] < best_score)
		{
			best_score = result_to_process->result_data[0];
			best_beamtilt_x = result_to_process->result_data[1];
			best_beamtilt_y = result_to_process->result_data[2];
			best_particle_shift_x = result_to_process->result_data[3];
			best_particle_shift_y = result_to_process->result_data[4];
		}

//		wxPrintf("Got BeamTilt Result : %f, %f, %f, %f = %f\n", result_to_process->result_data[1] * 1000.0f, result_to_process->result_data[2] * 1000.0f, result_to_process->result_data[3], result_to_process->result_data[4], result_to_process->result_data[0]);
	}
	if (running_job_type == RECONSTRUCTION)
	{
		//wxPrintf("Got reconstruction job \n");
		number_of_received_results++;
	//	wxPrintf("Received a reconstruction intermmediate result\n");

		long current_time = time(NULL);

		if (number_of_received_results == 1)
		{
			time_of_last_update = 0;
			current_job_starttime = current_time;
		}
		else
		if (current_time - time_of_last_update >= 1)
		{
			time_of_last_update = current_time;
			int current_percentage = float(number_of_received_results) / float(number_of_expected_results) * 100.0;
			if (current_percentage > 100) current_percentage = 100;
			my_parent->ProgressBar->SetValue(current_percentage);
			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_results - 1);
			long seconds_remaining = float(number_of_expected_results - number_of_received_results) * seconds_per_job;

			wxTimeSpan time_remaining = wxTimeSpan(0,0,seconds_remaining);
			my_parent->TimeRemainingText->SetLabel(time_remaining.Format("Time Remaining : %Hh:%Mm:%Ss"));		
		}
	}
	else
	if (running_job_type == MERGE)
	{
	//	wxPrintf("received merge result!\n");

		// add to the correct resolution statistics..

		int number_of_points = result_to_process->result_data[0];
		int class_number = int(result_to_process->result_data[1] + 0.5);
		int array_position = 2;
		float current_resolution;
		float fsc;
		float part_fsc;
		float part_ssnr;
		float rec_ssnr;

	//	wxPrintf("class_number = %i\n", class_number);
		// add the points..

		output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.Init(output_refinement->resolution_statistics_pixel_size, output_refinement->resolution_statistics_box_size);

		output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.FSC.ClearData();
		output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_FSC.ClearData();
		output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_SSNR.ClearData();
		output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.rec_SSNR.ClearData();


		for (int counter = 0; counter < number_of_points; counter++)
		{
			current_resolution = result_to_process->result_data[array_position];
			array_position++;
			fsc = result_to_process->result_data[array_position];
			array_position++;
			part_fsc = result_to_process->result_data[array_position];
			array_position++;
			part_ssnr = result_to_process->result_data[array_position];
			array_position++;
			rec_ssnr = result_to_process->result_data[array_position];
			array_position++;


			output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.FSC.AddPoint(current_resolution, fsc);
			output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_FSC.AddPoint(current_resolution, part_fsc);
			output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_SSNR.AddPoint(current_resolution, part_ssnr);
			output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.rec_SSNR.AddPoint(current_resolution, rec_ssnr);

		}

	}
}



void CTFRefinementManager::ProcessAllJobsFinished()
{

	// Update the GUI with project timings
	extern MyOverviewPanel *overview_panel;
	overview_panel->SetProjectInfo();


	if (running_job_type == REFINEMENT)
	{
		main_frame->job_controller.KillJob(my_parent->my_job_id);

		if (active_beamtilt_refinement == true)
		{
			RunBeamTiltEstimationJob();
		}
		else
		{
			SetupReconstructionJob();
			RunReconstructionJob();
		}


	}
	else
	if (running_job_type == ESTIMATE_BEAMTILT)
	{
		main_frame->job_controller.KillJob(my_parent->my_job_id);

		// work out significance..
		CTF input_ctf;
		input_ctf.Init(active_refinement_package->contained_particles[0].microscope_voltage, active_refinement_package->contained_particles[0].spherical_aberration, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, active_refinement_package->output_pixel_size, 0.0f);

		wxString phase_difference_image_filename = main_frame->ReturnRefineCTFScratchDirectory() + "phase_output.mrc";

		Image phase_difference_image;
		Image phase_difference_spectrum;
		Image beam_tilt_image;

//		int number_of_previous_searches = main_frame->current_project.database.ReturnNumberOfPreviousBeamtiltEstimationsFromList();

		std::string model_applied = "CTF";
		wxString phase_difference_output_file = main_frame->current_project.phase_difference_asset_directory.GetFullPath();
		phase_difference_output_file += wxString::Format("/%s_phase_difference_%s_%i.mrc","temp",model_applied,(int)current_input_refinement_id);
//		phase_difference_output_file = wxString::Format("/%s_mip_%i_%i.mrc", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);

		wxString beam_tilt_output_file = main_frame->current_project.phase_difference_asset_directory.GetFullPath();
		beam_tilt_output_file += wxString::Format("/%s_beam_tilt_%s_%i.mrc","temp",model_applied,(int)current_input_refinement_id);
//		phase_difference_output_file = wxString::Format("/%s_mip_%i_%i.mrc", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);

		phase_difference_image.QuickAndDirtyReadSlice(phase_difference_image_filename.ToStdString(), 1);

		phase_difference_image.ForwardFFT();
		phase_difference_spectrum.Allocate(phase_difference_image.logical_x_dimension, phase_difference_image.logical_y_dimension, false);
		beam_tilt_image.Allocate(phase_difference_image.logical_x_dimension, phase_difference_image.logical_y_dimension, false);
		phase_difference_image.ComputeAmplitudeSpectrumFull2D(&phase_difference_spectrum, true, 1.0f);
		phase_difference_spectrum.CosineRingMask(5.0f, phase_difference_spectrum.ReturnLargestLogicalDimension(), 2.0f);

		input_ctf.SetBeamTilt(best_beamtilt_x, best_beamtilt_y, best_particle_shift_x / active_refinement_package->output_pixel_size, best_particle_shift_y / active_refinement_package->output_pixel_size);
		phase_difference_image.CalculateBeamTiltImage(input_ctf); // reuse
		phase_difference_image.ComputeAmplitudeSpectrumFull2D(&beam_tilt_image, true, 1.0);
		phase_difference_spectrum.QuickAndDirtyWriteSlice(phase_difference_output_file.ToStdString(), 1);
		beam_tilt_image.QuickAndDirtyWriteSlice(beam_tilt_output_file.ToStdString(), 1);
//		phase_difference_spectrum.QuickAndDirtyWriteSlice("/tmp/phase_diff.mrc", 1);
//		beam_tilt_image.QuickAndDirtyWriteSlice("/tmp/beam_tilt.mrc", 1);

		float significance = phase_difference_spectrum.ReturnBeamTiltSignificanceScore(beam_tilt_image);
//		phase_difference_spectrum.MultiplyByConstant(float(phase_difference_spectrum.logical_x_dimension));
//		phase_difference_spectrum.Binarise(0.00002f);
//		beam_tilt_image.Binarise(0.0f);
//
//		float mask_radius_local = sqrtf((phase_difference_spectrum.ReturnAverageOfRealValues() * phase_difference_spectrum.logical_x_dimension * phase_difference_spectrum.logical_y_dimension) / PI);
//
//		phase_difference_spectrum.SubtractImage(&beam_tilt_image);
//		float binarized_score = phase_difference_spectrum.ReturnSumOfSquares(mask_radius_local);
//
//		float significance = 0.5f * PI * powf((0.5f - binarized_score) * mask_radius_local, 2);

		wxPrintf("%.2f, %.2f, %.2f %.2f\n", best_beamtilt_x * 1000.0f, best_beamtilt_y * 1000.0f, best_particle_shift_x, best_particle_shift_y);

		wxPrintf("Significance = %.2f\n", significance);

		if (significance <= MINIMUM_BEAM_TILT_SIGNIFICANCE_SCORE)
		{
			best_beamtilt_x = 0.0f;
			best_beamtilt_y = 0.0f;
			best_particle_shift_x = 0.0f;
			best_particle_shift_y = 0.0f;
		}

		for (int class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
		{
			for (long particle_counter = 0; particle_counter < output_refinement->number_of_particles; particle_counter++)
			{
				output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].beam_tilt_x = best_beamtilt_x * 1000.0f;
				output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].beam_tilt_y = best_beamtilt_y * 1000.0f;
				output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_shift_x = best_particle_shift_x;
				output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_shift_y = best_particle_shift_y;
			}
		}

		SetupReconstructionJob();
		RunReconstructionJob();

	}
	else
	if (running_job_type == RECONSTRUCTION)
	{
		main_frame->job_controller.KillJob(my_parent->my_job_id);
		//wxPrintf("Reconstruction has finished\n");
		SetupMerge3dJob();
		RunMerge3dJob();
	}
	else
	if (running_job_type == MERGE)
	{
		long current_reconstruction_id;

		OrthDrawerThread *result_thread;
		my_parent->active_orth_thread_id = my_parent->next_thread_id;
		my_parent->next_thread_id++;

		result_thread = new OrthDrawerThread(my_parent, current_reference_filenames, wxString::Format("Reconstruction"), 1.0f, active_mask_radius / input_refinement->resolution_statistics_pixel_size, my_parent->active_orth_thread_id);

		if ( result_thread->Run() != wxTHREAD_NO_ERROR )
		{
			my_parent->WriteErrorText("Error: Cannot start result creation thread, results not displayed");
			delete result_thread;
		}

		int class_counter;

		main_frame->job_controller.KillJob(my_parent->my_job_id);

		VolumeAsset temp_asset;

		temp_asset.pixel_size = output_refinement->resolution_statistics_pixel_size;
		temp_asset.x_size = output_refinement->resolution_statistics_box_size;
		temp_asset.y_size = output_refinement->resolution_statistics_box_size;
		temp_asset.z_size = output_refinement->resolution_statistics_box_size;

		// add the volumes etc to the database..
		main_frame->current_project.database.Begin();
		output_refinement->reference_volume_ids.Clear();
		active_refinement_package->references_for_next_refinement.Clear();

		main_frame->current_project.database.BeginVolumeAssetInsert();

		my_parent->WriteInfoText("");

		for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
		{
			temp_asset.asset_id = volume_asset_panel->current_asset_number;

			output_refinement->reference_volume_ids.Add(current_reference_asset_ids[class_counter]);
			current_reference_asset_ids[class_counter] = temp_asset.asset_id;

			temp_asset.asset_name = wxString::Format("CTF Refine #%li - Class #%i", current_output_refinement_id, class_counter + 1);
			
			current_reconstruction_id = main_frame->current_project.database.ReturnHighestReconstructionID() + 1;
			temp_asset.reconstruction_job_id = current_reconstruction_id;
			
			// set the output volume
			output_refinement->class_refinement_results[class_counter].reconstructed_volume_asset_id = temp_asset.asset_id;
			output_refinement->class_refinement_results[class_counter].reconstruction_id = current_reconstruction_id;

			// add the reconstruction job


			main_frame->current_project.database.AddReconstructionJob(current_reconstruction_id, active_refinement_package->asset_id, output_refinement->refinement_id, "", active_inner_mask_radius, active_mask_radius, active_resolution_limit_rec, active_score_weight_conversion, active_adjust_scores, active_crop_images, false, active_should_apply_blurring, active_smoothing_factor, class_counter + 1, long(temp_asset.asset_id));


			temp_asset.filename = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/volume_%li_%i.mrc", output_refinement->refinement_id, class_counter + 1);
			output_refinement->reference_volume_ids.Add(temp_asset.asset_id);

			active_refinement_package->references_for_next_refinement.Add(temp_asset.asset_id);
			main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li SET VOLUME_ASSET_ID=%i WHERE CLASS_NUMBER=%i", current_refinement_package_asset_id, temp_asset.asset_id, class_counter + 1 ));


			volume_asset_panel->AddAsset(&temp_asset);
			main_frame->current_project.database.AddNextVolumeAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath(), temp_asset.reconstruction_job_id, temp_asset.pixel_size, temp_asset.x_size, temp_asset.y_size, temp_asset.z_size, temp_asset.half_map_1_filename.GetFullPath(), temp_asset.half_map_2_filename.GetFullPath());
		}

		main_frame->current_project.database.EndVolumeAssetInsert();

		wxArrayFloat average_occupancies = output_refinement->UpdatePSSNR();

		my_parent->WriteInfoText("");

		if (output_refinement->number_of_classes > 1)
		{
			for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
			{
				my_parent->WriteInfoText(wxString::Format(wxT("Est. Res. Class %2i = %2.2f Å (%2.2f %%)"), class_counter + 1, output_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnEstimatedResolution(), average_occupancies[class_counter]));
			}
		}
		else
		{
			my_parent->WriteInfoText(wxString::Format(wxT("Est. Res. = %2.2f Å"), output_refinement->class_refinement_results[0].class_resolution_statistics.ReturnEstimatedResolution()));
		}

		my_parent->WriteInfoText("");

		for (class_counter = 1; class_counter <= output_refinement->number_of_classes; class_counter++)
		{
			main_frame->current_project.database.CopyRefinementAngularDistributions(input_refinement->refinement_id, output_refinement->refinement_id, class_counter);
		}

		main_frame->current_project.database.AddRefinement(output_refinement);
		ShortRefinementInfo temp_info;
		temp_info = output_refinement;
		refinement_package_asset_panel->all_refinement_short_infos.Add(temp_info);

		// add this refinment to the refinement package..

		active_refinement_package->last_refinment_id = output_refinement->refinement_id;
		active_refinement_package->refinement_ids.Add(output_refinement->refinement_id);

		main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_ASSETS SET LAST_REFINEMENT_ID=%li WHERE REFINEMENT_PACKAGE_ASSET_ID=%li", output_refinement->refinement_id, current_refinement_package_asset_id));
		main_frame->current_project.database.ExecuteSQL(wxString::Format("INSERT INTO REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li (REFINEMENT_NUMBER, REFINEMENT_ID) VALUES (%li, %li);", current_refinement_package_asset_id, main_frame->current_project.database.ReturnSingleLongFromSelectCommand(wxString::Format("SELECT MAX(REFINEMENT_NUMBER) FROM REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li", current_refinement_package_asset_id)) + 1,  output_refinement->refinement_id));

		main_frame->DirtyVolumes();
		main_frame->DirtyRefinements();

		my_parent->ShowRefinementResultsPanel->FSCResultsPanel->AddRefinement(output_refinement);


		my_parent->ShowRefinementResultsPanel->Show(true);

		if (my_parent->ShowRefinementResultsPanel->TopBottomSplitter->IsSplit() == false)
		{
			my_parent->ShowRefinementResultsPanel->TopBottomSplitter->SplitHorizontally(my_parent->ShowRefinementResultsPanel->TopPanel, my_parent->ShowRefinementResultsPanel->BottomPanel);
			my_parent->ShowRefinementResultsPanel->FSCResultsPanel->Show(true);
		}

		main_frame->current_project.database.Commit();
		global_delete_refinectf_scratch();

		delete input_refinement;
		input_refinement = NULL;
//			delete output_refinement;
		my_parent->WriteBlueText("Refinement finished!");
		my_parent->CancelAlignmentButton->Show(false);
		my_parent->FinishButton->Show(true);
		my_parent->TimeRemainingText->SetLabel(wxString::Format("All Done! (%s)", wxTimeSpan::Milliseconds(my_parent->stopwatch.Time()).Format(wxT("%Hh:%Mm:%Ss"))));
		my_parent->ProgressBar->SetValue(100);

		my_parent->Layout();

		//wxPrintf("Calling cycle refinement\n");
		main_frame->DirtyVolumes();
		main_frame->DirtyRefinements();
	}

}

void CTFRefinementManager::DoMasking()
{
	MyDebugAssertTrue(active_should_mask == true || active_should_auto_mask == true, "DoMasking called, when masking not selected!");

	wxArrayString masked_filenames;
	wxString current_masked_filename;
	wxString filename_of_mask = active_mask_filename;

	for (int class_counter = 0; class_counter < current_reference_filenames.GetCount(); class_counter++)
	{
		current_masked_filename = main_frame->ReturnRefineCTFScratchDirectory();
		current_masked_filename += wxFileName(current_reference_filenames.Item(class_counter)).GetName();
		current_masked_filename += "_masked.mrc";

		masked_filenames.Add(current_masked_filename);
	}

	if (active_should_mask == true) // user selected masking
	{

		my_parent->WriteInfoText("Masking reference reconstruction with selected mask");

		float wanted_cosine_edge_width = active_mask_edge;
		float wanted_weight_outside_mask = active_mask_weight;

		float wanted_low_pass_filter_radius;

		if (active_should_low_pass_filter_mask == true)
		{
			wanted_low_pass_filter_radius = active_mask_filter_resolution;
		}
		else
		{
			wanted_low_pass_filter_radius = 0.0;
		}

		my_parent->active_mask_thread_id = my_parent->next_thread_id;
		my_parent->next_thread_id++;

		Multiply3DMaskerThread *mask_thread = new Multiply3DMaskerThread(my_parent, current_reference_filenames, masked_filenames, filename_of_mask, wanted_cosine_edge_width, wanted_weight_outside_mask, wanted_low_pass_filter_radius, input_refinement->resolution_statistics_pixel_size, my_parent->active_mask_thread_id);

		if ( mask_thread->Run() != wxTHREAD_NO_ERROR )
		{
			my_parent->WriteErrorText("Error: Cannot start masking thread, masking will not be performed");
			delete mask_thread;
		}
		else
		{
			current_reference_filenames = masked_filenames;
			return;
		}
	}
	else
	{

		my_parent->WriteInfoText("Automasking reference reconstruction");

		my_parent->active_mask_thread_id = my_parent->next_thread_id;
		my_parent->next_thread_id++;

		AutoMaskerThread *mask_thread = new AutoMaskerThread(my_parent, current_reference_filenames, masked_filenames, input_refinement->resolution_statistics_pixel_size, active_refinement_package->estimated_particle_size_in_angstroms * 0.75, my_parent->active_mask_thread_id);

		if ( mask_thread->Run() != wxTHREAD_NO_ERROR )
		{
			my_parent->WriteErrorText("Error: Cannot start masking thread, masking will not be performed");
			delete mask_thread;
		}
		else
		{
			current_reference_filenames = masked_filenames;
			return;
		}
	}



}

void RefineCTFPanel::OnMaskerThreadComplete(wxThreadEvent& my_event)
{
	if (my_event.GetInt() == active_mask_thread_id)	my_refinement_manager.OnMaskerThreadComplete();
}


void RefineCTFPanel::OnOrthThreadComplete(ReturnProcessedImageEvent& my_event)
{

	Image *new_image = my_event.GetImage();

	if (my_event.GetInt() == active_orth_thread_id)
	{
		if (new_image != NULL)
		{
			ShowRefinementResultsPanel->ShowOrthDisplayPanel->OpenImage(new_image, my_event.GetString(), true);

			if (ShowRefinementResultsPanel->LeftRightSplitter->IsSplit() == false)
			{
				ShowRefinementResultsPanel->LeftRightSplitter->SplitVertically(ShowRefinementResultsPanel->LeftPanel, ShowRefinementResultsPanel->RightPanel, 600);
				Layout();
			}
		}
	}
	else
	{
		delete new_image;
	}

}


void CTFRefinementManager::OnMaskerThreadComplete()
{
	//my_parent->WriteInfoText("Masking Finished");
	SetupRefinementJob();
	RunRefinementJob();
}

