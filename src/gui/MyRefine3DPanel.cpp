#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyVolumeAssetPanel *volume_asset_panel;

MyRefine3DPanel::MyRefine3DPanel( wxWindow* parent )
:
Refine3DPanel( parent )
{
	buffered_results = NULL;

	// Fill combo box..

	//FillGroupComboBox();

	my_job_id = -1;
	running_job = false;

//	group_combo_is_dirty = false;
//	run_profiles_are_dirty = false;

	SetInfo();
//	FillGroupComboBox();
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

	my_refinement_manager.SetParent(this);

	FillRefinementPackagesComboBox();



}

void MyRefine3DPanel::SetInfo()
{
	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("Refinement"));
	InfoText->EndFontSize();
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("Refinement is a good idea. Niko will write more about it!"));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

}

void MyRefine3DPanel::ResetAllDefaultsClick( wxCommandEvent& event )
{
	// TODO : should probably check that the user hasn't changed the defaults yet in the future
	SetDefaults();
}

void MyRefine3DPanel::SetDefaults()
{
	float calculated_high_resolution_cutoff;
	float local_mask_radius;
	float global_mask_radius;
	float global_angular_step;
	float search_range;

	ExpertPanel->Freeze();

	// calculate high resolution limit..

	if (InputParametersComboBox->GetSelection() == 0)
	{
		if (refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).lowest_resolution_of_intial_parameter_generated_3ds > 0)
		{
			calculated_high_resolution_cutoff = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).lowest_resolution_of_intial_parameter_generated_3ds;
		}
		else calculated_high_resolution_cutoff = 30.00;
	}
	else
	{
		// calculate it based on refinements..
	}

	local_mask_radius = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.6;
	global_mask_radius = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms;

	global_angular_step = CalculateAngularStep(calculated_high_resolution_cutoff, local_mask_radius);

	search_range = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.5;
	// Set the values..

	RefinePhiCheckBox->SetValue(true);
	RefineThetaCheckBox->SetValue(true);
	RefinePsiCheckBox->SetValue(true);
	RefineXShiftCheckBox->SetValue(true);
	RefineYShiftCheckBox->SetValue(true);

	LowResolutionLimitTextCtrl->SetValue("300.00");
	HighResolutionLimitTextCtrl->SetValue(wxString::Format("%.2f", calculated_high_resolution_cutoff));
	MaskRadiusTextCtrl->SetValue(wxString::Format("%.2f", local_mask_radius));
	SignedCCResolutionTextCtrl->SetValue("0.00");

	GlobalResolutionLimitTextCtrl->SetValue(wxString::Format("%.2f", calculated_high_resolution_cutoff));
	GlobalMaskRadiusTextCtrl->SetValue(wxString::Format("%.2f", global_mask_radius));
	NumberToRefineSpinCtrl->SetValue(20);
	AngularStepTextCtrl->SetValue(wxString::Format("%.2f", global_angular_step));
	SearchRangeXTextCtrl->SetValue(wxString::Format("%.2f", search_range));
	SearchRangeYTextCtrl->SetValue(wxString::Format("%.2f", search_range));

	ClassificationHighResLimitTextCtrl->SetValue(wxString::Format("%.2f", calculated_high_resolution_cutoff));
	SphereClassificatonYesRadio->SetValue(false);
	SphereClassificatonNoRadio->SetValue(true);

	SphereXTextCtrl->SetValue("0.00");
	SphereYTextCtrl->SetValue("0.00");
	SphereZTextCtrl->SetValue("0.00");
	SphereRadiusTextCtrl->SetValue("0.00");

	RefineCTFYesRadio->SetValue(false);
	RefineCTFNoRadio->SetValue(true);
	DefocusSearchRangeTextCtrl->SetValue("500.00");
	DefocusSearchStepTextCtrl->SetValue("50.00");

	ReconstructionInnerRadiusTextCtrl->SetValue("0.00");
	ReconstructionOuterRadiusTextCtrl->SetValue(wxString::Format("%.2f", local_mask_radius));
	ScoreToBFactorConstantTextCtrl->SetValue("2.00");

	AdjustScoreForDefocusYesRadio->SetValue(true);
	AdjustScoreForDefocusNoRadio->SetValue(false);
	ReconstructioScoreThreshold->SetValue("0.00");
	ReconstructionResolutionLimitTextCtrl->SetValue("0.00");
	AutoCropYesRadioButton->SetValue(false);
	AutoCropNoRadioButton->SetValue(true);

	ExpertPanel->Thaw();

}


void MyRefine3DPanel::OnUpdateUI( wxUpdateUIEvent& event )
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
		LocalRefinementRadio->Enable(false);
		GlobalRefinementRadio->Enable(false);
		NumberRoundsSpinCtrl->Enable(false);

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

	}
	else
	{
		if (running_job == false)
		{
			RefinementRunProfileComboBox->Enable(true);
			ReconstructionRunProfileComboBox->Enable(true);

			ExpertToggleButton->Enable(true);

			if (RefinementPackageComboBox->GetCount() > 0)
			{
				RefinementPackageComboBox->Enable(true);
				InputParametersComboBox->Enable(true);

			}
			else
			{
				RefinementPackageComboBox->ChangeValue("");
				RefinementPackageComboBox->Enable(false);
				InputParametersComboBox->ChangeValue("");
				InputParametersComboBox->Enable(false);
			}

			LocalRefinementRadio->Enable(true);
			GlobalRefinementRadio->Enable(true);
			NumberRoundsSpinCtrl->Enable(true);

			if (ExpertToggleButton->GetValue() == true)
			{
				if (GlobalRefinementRadio->GetValue() == false)
				{
					GlobalResolutionLimitStaticText->Enable(false);
					GlobalResolutionLimitTextCtrl->Enable(false);
					GlobalMaskRadiusStaticText->Enable(false);
					GlobalMaskRadiusTextCtrl->Enable(false);
					NumberToRefineSpinCtrl->Enable(false);
					NumberToRefineStaticText->Enable(false);
					AngularStepTextCtrl->Enable(false);
					AngularStepStaticText->Enable(false);
					SearchRangeXTextCtrl->Enable(false);
					SearchRangeXStaticText->Enable(false);
					SearchRangeYTextCtrl->Enable(false);
					SearchRangeYStaticText->Enable(false);
				}
				else
				{
					GlobalResolutionLimitStaticText->Enable(true);
					GlobalResolutionLimitTextCtrl->Enable(true);
					GlobalMaskRadiusStaticText->Enable(true);
					GlobalMaskRadiusTextCtrl->Enable(true);
					NumberToRefineSpinCtrl->Enable(true);
					NumberToRefineStaticText->Enable(true);
					AngularStepTextCtrl->Enable(true);
					AngularStepStaticText->Enable(true);
					SearchRangeXTextCtrl->Enable(true);
					SearchRangeXStaticText->Enable(true);
					SearchRangeYTextCtrl->Enable(true);
					SearchRangeYStaticText->Enable(true);

				}

				if (SphereClassificatonYesRadio->GetValue() == false)
				{
					SphereXTextCtrl->Enable(false);
					SphereXStaticText->Enable(false);
					SphereYTextCtrl->Enable(false);
					SphereYStaticText->Enable(false);
					SphereZTextCtrl->Enable(false);
					SphereZStaticText->Enable(false);
					SphereRadiusTextCtrl->Enable(false);
					SphereRadiusStaticText->Enable(false);

				}
				else
				{
					SphereXTextCtrl->Enable(true);
					SphereXStaticText->Enable(true);
					SphereYTextCtrl->Enable(true);
					SphereYStaticText->Enable(true);
					SphereZTextCtrl->Enable(true);
					SphereZStaticText->Enable(true);
					SphereRadiusTextCtrl->Enable(true);
					SphereRadiusStaticText->Enable(true);
				}

				if (RefineCTFYesRadio->GetValue() == false)
				{
					DefocusSearchRangeStaticText->Enable(false);
					DefocusSearchRangeTextCtrl->Enable(false);
					DefocusSearchStepTextCtrl->Enable(false);
					DefocusSearchStepStaticText->Enable(false);
				}
				else
				{
					DefocusSearchRangeStaticText->Enable(true);
					DefocusSearchRangeTextCtrl->Enable(true);
					DefocusSearchStepTextCtrl->Enable(true);
					DefocusSearchStepStaticText->Enable(true);
				}


			}

			bool estimation_button_status = false;

			if (RefinementPackageComboBox->GetCount() > 0 && ReconstructionRunProfileComboBox->GetCount() > 0)
			{
				if (run_profiles_panel->run_profile_manager.ReturnTotalJobs(RefinementRunProfileComboBox->GetSelection()) > 1 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(ReconstructionRunProfileComboBox->GetSelection()) > 1)
				{
					if (RefinementPackageComboBox->GetSelection() != wxNOT_FOUND && InputParametersComboBox->GetSelection() != wxNOT_FOUND)
					{
						estimation_button_status = true;
					}

				}
			}

			StartRefinementButton->Enable(estimation_button_status);
		}
		else
		{
			//	ExpertToggleButton->Enable(false);
			//	GroupComboBox->Enable(false);
			//	RunProfileComboBox->Enable(false);
			//  StartAlignmentButton->SetLabel("Stop Job");
			//  StartAlignmentButton->Enable(true);
		}

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

	}

}

void MyRefine3DPanel::OnExpertOptionsToggle( wxCommandEvent& event )
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

void MyRefine3DPanel::FillRefinementPackagesComboBox()
{
	RefinementPackageComboBox->Freeze();
	RefinementPackageComboBox->Clear();
	RefinementPackageComboBox->ChangeValue("");

	if (refinement_package_asset_panel->all_refinement_packages.GetCount() > 0)
	{
		long currently_selected_asset_id = -1;
		bool selection_has_changed = true;

		if (selected_refinement_package >= 0 && selected_refinement_package <= RefinementPackageComboBox->GetCount() && RefinementPackageComboBox->GetCount() > 0)
		{
			currently_selected_asset_id = refinement_package_asset_panel->all_refinement_packages.Item(selected_refinement_package).asset_id;
		}

		AppendRefinementPackagesToComboBox(RefinementPackageComboBox);

		if (currently_selected_asset_id == -1)
		{
			selected_refinement_package = 0;
			RefinementPackageComboBox->SetSelection(0);
		}
		else
		{
			selected_refinement_package = 0;
			RefinementPackageComboBox->SetSelection(0);

			for (long counter = 0; counter < RefinementPackageComboBox->GetCount(); counter++)
			{
				if (refinement_package_asset_panel->all_refinement_packages.Item(counter).asset_id == currently_selected_asset_id)
				{
					selected_refinement_package = counter;
					RefinementPackageComboBox->SetSelection(counter);
					selection_has_changed = false;
				}
			}
		}

		if (selection_has_changed == true)
		{
			NewRefinementPackageSelected();
		}
	}

	RefinementPackageComboBox->Thaw();
}

void MyRefine3DPanel::FillInputParamsComboBox()
{
	InputParametersComboBox->Freeze();
	InputParametersComboBox->Clear();
	InputParametersComboBox->ChangeValue("");
	InputParametersComboBox->Append("Initial Parameters");
	InputParametersComboBox->SetSelection(0);
}

void MyRefine3DPanel::NewRefinementPackageSelected()
{
	selected_refinement_package = RefinementPackageComboBox->GetSelection();
	FillInputParamsComboBox();
	SetDefaults();
	//wxPrintf("New Refinement Package Selection\n");

}

void MyRefine3DPanel::OnRefinementPackageComboBox( wxCommandEvent& event )
{

	NewRefinementPackageSelected();

}


void MyRefine3DPanel::OnInfoURL( wxTextUrlEvent& event )
{

}

void MyRefine3DPanel::TerminateButtonClick( wxCommandEvent& event )
{
	main_frame->job_controller.KillJob(my_job_id);

	WriteInfoText("Terminated Job");
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

void MyRefine3DPanel::FinishButtonClick( wxCommandEvent& event )
{
	ProgressBar->SetValue(0);
	TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
	FinishButton->Show(false);

	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	output_textctrl->Clear();
	//CTFResultsPanel->Show(false);
	//graph_is_hidden = true;
	InfoPanel->Show(true);

	if (ExpertToggleButton->GetValue() == true) ExpertPanel->Show(true);
	else ExpertPanel->Show(false);
	running_job = false;
	Layout();

	//CTFResultsPanel->CTF2DResultsPanel->should_show = false;
	//CTFResultsPanel->CTF2DResultsPanel->Refresh();

}




void MyRefine3DPanel::StartRefinementClick( wxCommandEvent& event )
{
	my_refinement_manager.BeginRefinementCycle();
}

void MyRefine3DPanel::WriteInfoText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void MyRefine3DPanel::WriteErrorText(wxString text_to_write)
{
	 output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
	 output_textctrl->AppendText(text_to_write);

	 if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}


void MyRefine3DPanel::FillRunProfileComboBoxes()
{
	int old_reconstruction_selection = 0;
	int old_refinement_selection = 0;

	// get the current selection..

	if (ReconstructionRunProfileComboBox->GetCount() > 0) old_reconstruction_selection = ReconstructionRunProfileComboBox->GetSelection();
	if (RefinementRunProfileComboBox->GetCount() > 0) old_refinement_selection = RefinementRunProfileComboBox->GetSelection();

	// refresh..

	ReconstructionRunProfileComboBox->Freeze();
	RefinementRunProfileComboBox->Freeze();

	ReconstructionRunProfileComboBox->Clear();
	RefinementRunProfileComboBox->Clear();

	for (long counter = 0; counter < run_profiles_panel->run_profile_manager.number_of_run_profiles; counter++)
	{
		ReconstructionRunProfileComboBox->Append(run_profiles_panel->run_profile_manager.ReturnProfileName(counter) + wxString::Format(" (%li)", run_profiles_panel->run_profile_manager.ReturnTotalJobs(counter)));
		RefinementRunProfileComboBox->Append(run_profiles_panel->run_profile_manager.ReturnProfileName(counter) + wxString::Format(" (%li)", run_profiles_panel->run_profile_manager.ReturnTotalJobs(counter)));
	}

	if (ReconstructionRunProfileComboBox->GetCount() > 0)
	{
		if (ReconstructionRunProfileComboBox->GetCount() >= old_reconstruction_selection) ReconstructionRunProfileComboBox->SetSelection(old_reconstruction_selection);
		else ReconstructionRunProfileComboBox->SetSelection(0);
	}

	if (RefinementRunProfileComboBox->GetCount() > 0)
	{
		if (RefinementRunProfileComboBox->GetCount() >= old_refinement_selection) RefinementRunProfileComboBox->SetSelection(old_refinement_selection);
		else RefinementRunProfileComboBox->SetSelection(0);
	}

	ReconstructionRunProfileComboBox->Thaw();
	RefinementRunProfileComboBox->Thaw();


}

void MyRefine3DPanel::OnJobSocketEvent(wxSocketEvent& event)
{
      SETUP_SOCKET_CODES

	  wxString s = _("OnSocketEvent: ");
	  wxSocketBase *sock = event.GetSocket();

	  MyDebugAssertTrue(sock == main_frame->job_controller.job_list[my_job_id].socket, "Socket event from Non conduit socket??");

	  // First, print a message
	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_INPUT : s.Append(_("wxSOCKET_INPUT\n")); break;
	    case wxSOCKET_LOST  : s.Append(_("wxSOCKET_LOST\n")); break;
	    default             : s.Append(_("Unexpected event !\n")); break;
	  }

	  //m_text->AppendText(s);

	  //MyDebugPrint(s);

	  // Now we process the event
	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_INPUT:
	    {
	      // We disable input events, so that the test doesn't trigger
	      // wxSocketEvent again.
	      sock->SetNotify(wxSOCKET_LOST_FLAG);
	      sock->ReadMsg(&socket_input_buffer, SOCKET_CODE_SIZE);

	      if (memcmp(socket_input_buffer, socket_send_job_details, SOCKET_CODE_SIZE) == 0) // identification
	      {
	    	  // send the job details..

	    	  //wxPrintf("Sending Job Details...\n");
	    	  my_job_package.SendJobPackage(sock);

	      }
	      else
	      if (memcmp(socket_input_buffer, socket_i_have_an_error, SOCKET_CODE_SIZE) == 0) // identification
	      {

	    	  wxString error_message;
   			  error_message = ReceivewxStringFromSocket(sock);

   			  WriteErrorText(error_message);
    	  }
	      else
	      if (memcmp(socket_input_buffer, socket_i_have_info, SOCKET_CODE_SIZE) == 0) // identification
	      {

	    	  wxString info_message;
   			  info_message = ReceivewxStringFromSocket(sock);

   			  WriteInfoText(info_message);
    	  }
	      else
	      if (memcmp(socket_input_buffer, socket_job_finished, SOCKET_CODE_SIZE) == 0) // identification
	 	  {
	 		 // which job is finished?

	 		 int finished_job;
	 		 sock->ReadMsg(&finished_job, 4);
	 		 my_job_tracker.MarkJobFinished();

//	 		 if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();
	 		 //WriteInfoText(wxString::Format("Job %i has finished!", finished_job));
	 	  }
	      else
	      if (memcmp(socket_input_buffer, socket_job_result, SOCKET_CODE_SIZE) == 0) // identification
	 	  {
	    	  JobResult temp_result;
	    	  temp_result.ReceiveFromSocket(sock);

	    	  // send the result to the

	    	  my_refinement_manager.ProcessJobResult(&temp_result);

	 	  }
	      else
		  if (memcmp(socket_input_buffer, socket_number_of_connections, SOCKET_CODE_SIZE) == 0) // identification
		  {
			  // how many connections are there?

			  int number_of_connections;
              sock->ReadMsg(&number_of_connections, 4);

              my_job_tracker.AddConnection();

    //          if (graph_is_hidden == true) ProgressBar->Pulse();

              //WriteInfoText(wxString::Format("There are now %i connections\n", number_of_connections));

              // send the info to the gui

              int total_processes = my_job_package.my_profile.ReturnTotalJobs();

		 	  if (number_of_connections == total_processes) WriteInfoText(wxString::Format("All %i processes are connected.", number_of_connections));

			  if (length_of_process_number == 6) NumberConnectedText->SetLabel(wxString::Format("%6i / %6i processes connected.", number_of_connections, total_processes));
			  else
			  if (length_of_process_number == 5) NumberConnectedText->SetLabel(wxString::Format("%5i / %5i processes connected.", number_of_connections, total_processes));
		      else
			  if (length_of_process_number == 4) NumberConnectedText->SetLabel(wxString::Format("%4i / %4i processes connected.", number_of_connections, total_processes));
			  else
			  if (length_of_process_number == 3) NumberConnectedText->SetLabel(wxString::Format("%3i / %3i processes connected.", number_of_connections, total_processes));
			  else
			  if (length_of_process_number == 2) NumberConnectedText->SetLabel(wxString::Format("%2i / %2i processes connected.", number_of_connections, total_processes));
			  else
		      NumberConnectedText->SetLabel(wxString::Format("%1i / %1i processes connected.", number_of_connections, total_processes));
		  }
	      else
		  if (memcmp(socket_input_buffer, socket_all_jobs_finished, SOCKET_CODE_SIZE) == 0) // identification
		  {
			  my_refinement_manager.ProcessAllJobsFinished();
		  }

	      // Enable input events again.

	      sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

	      break;
	    }


	    case wxSOCKET_LOST:
	    {

	    	//MyDebugPrint("Socket Disconnected!!\n");
	        sock->Destroy();
	        break;
	    }
	    default: ;
	  }

}


void RefinementManager::SetParent(MyRefine3DPanel *wanted_parent)
{
	my_parent = wanted_parent;
}

void RefinementManager::BeginRefinementCycle()
{
	number_of_rounds_run = 0;
	number_of_rounds_to_run = my_parent->NumberRoundsSpinCtrl->GetValue();

	current_refinement_package_asset_id = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).asset_id;
	current_input_refinement_id = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).refinement_ids[my_parent->InputParametersComboBox->GetSelection()];

	// get the data..

	input_refinement = main_frame->current_project.database.GetRefinementByID(current_input_refinement_id);
	output_refinement = new Refinement;

	if (start_with_reconstruction == true)
	{
		SetupReconstructionJob();
		RunReconstructionJob();
	}
	else
	{
		SetupRefinementJob();
		RunRefinementJob();
	}
}


void RefinementManager::RunRefinementJob()
{
	running_job_type = REFINEMENT;

	output_refinement->SizeAndFillWithEmpty(input_refinement->number_of_particles, input_refinement->number_of_classes);
	wxPrintf("Output refinement has %li particles\n", output_refinement->number_of_particles);
	current_output_refinement_id = main_frame->current_project.database.ReturnHighestRefinementID() + 1;

	output_refinement->refinement_id = current_output_refinement_id;
	output_refinement->refinement_package_asset_id = current_refinement_package_asset_id;

	if (my_parent->GlobalRefinementRadio->GetValue() == true)
	{
		output_refinement->name = wxString::Format("Global Search #%li", current_output_refinement_id);
	}
	else output_refinement->name = wxString::Format("Local Refinement #%li", current_output_refinement_id);

	output_refinement->refinement_was_imported_or_generated = false;
	output_refinement->datetime_of_run = wxDateTime::Now();
	output_refinement->starting_refinement_id = current_input_refinement_id;

	output_refinement->low_resolution_limit = my_parent->LowResolutionLimitTextCtrl->ReturnValue();
	output_refinement->high_resolution_limit = my_parent->HighResolutionLimitTextCtrl->ReturnValue();
	output_refinement->mask_radius = my_parent->MaskRadiusTextCtrl->ReturnValue();
	output_refinement->signed_cc_resolution_limit = my_parent->SignedCCResolutionTextCtrl->ReturnValue();
	output_refinement->global_resolution_limit = my_parent->GlobalResolutionLimitTextCtrl->ReturnValue();
	output_refinement->global_mask_radius = my_parent->GlobalMaskRadiusTextCtrl->ReturnValue();
	output_refinement->number_results_to_refine = my_parent->NumberToRefineSpinCtrl->GetValue();
	output_refinement->angular_search_step = my_parent->AngularStepTextCtrl->ReturnValue();
	output_refinement->search_range_x = my_parent->SearchRangeXTextCtrl->ReturnValue();
	output_refinement->search_range_x = my_parent->SearchRangeXTextCtrl->ReturnValue();
	output_refinement->classification_resolution_limit = my_parent->ClassificationHighResLimitTextCtrl->ReturnValue();
	output_refinement->should_focus_classify = my_parent->SphereClassificatonYesRadio->GetValue();
	output_refinement->sphere_x_coord = my_parent->SphereXTextCtrl->ReturnValue();
	output_refinement->sphere_y_coord = my_parent->SphereYTextCtrl->ReturnValue();
	output_refinement->sphere_z_coord = my_parent->SphereZTextCtrl->ReturnValue();
	output_refinement->should_refine_ctf = my_parent->RefineCTFYesRadio->GetValue();
	output_refinement->defocus_search_range = my_parent->DefocusSearchRangeTextCtrl->ReturnValue();
	output_refinement->defocus_search_step = my_parent->DefocusSearchStepTextCtrl->ReturnValue();

	output_refinement->resolution_statistics_box_size = input_refinement->resolution_statistics_box_size;
	output_refinement->resolution_statistics_pixel_size = input_refinement->resolution_statistics_pixel_size;

	// launch a controller

	my_parent->WriteInfoText(wxString::Format("Running refinement round %2i of %2i\n", number_of_rounds_run + 1, number_of_rounds_to_run));
	current_job_id = main_frame->job_controller.AddJob(my_parent, run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes =  my_parent->my_job_package.my_profile.ReturnTotalJobs();

		if (number_of_refinement_processes >= 100000) my_parent->length_of_process_number = 6;
		else
		if (number_of_refinement_processes >= 10000) my_parent->length_of_process_number = 5;
		else
		if (number_of_refinement_processes >= 1000) my_parent->length_of_process_number = 4;
		else
		if (number_of_refinement_processes >= 100) my_parent->length_of_process_number = 3;
		else
		if (number_of_refinement_processes >= 10) my_parent->length_of_process_number = 2;
		else
		my_parent->length_of_process_number = 1;

		if (my_parent->length_of_process_number == 6) my_parent->NumberConnectedText->SetLabel(wxString::Format("%6i / %6li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 5) my_parent->NumberConnectedText->SetLabel(wxString::Format("%5i / %5li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 4) my_parent->NumberConnectedText->SetLabel(wxString::Format("%4i / %4li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 3) my_parent->NumberConnectedText->SetLabel(wxString::Format("%3i / %3li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 2) my_parent->NumberConnectedText->SetLabel(wxString::Format("%2i / %2li processes connected.", 0, number_of_refinement_processes));
		else

		my_parent->NumberConnectedText->SetLabel(wxString::Format("%i / %li processes connected.", 0, number_of_refinement_processes));

		my_parent->StartPanel->Show(false);
		my_parent->ProgressPanel->Show(true);

		my_parent->ExpertPanel->Show(false);
		my_parent->InfoPanel->Show(false);
		my_parent->OutputTextPanel->Show(true);
		//	CTFResultsPanel->Show(true);

		my_parent->ExpertToggleButton->Enable(false);
		my_parent->RefinementPackageComboBox->Enable(false);
		my_parent->InputParametersComboBox->Enable(false);

		my_parent->Layout();
		my_parent->running_job = true;
		my_parent->my_job_tracker.StartTracking(my_parent->my_job_package.number_of_jobs);

	}



	my_parent->ProgressBar->Pulse();
}

void RefinementManager::SetupMerge3dJob()
{
	int class_counter;

	my_parent->my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()], "merge3d", refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes);

	for (class_counter = 0; class_counter < refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes; class_counter++)
	{
		wxString output_reconstruction_1			= "/dev/null";
		wxString output_reconstruction_2			= "/dev/null";
		wxString output_reconstruction_filtered		= main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/volume_%li_%i.mrc", output_refinement->refinement_id, class_counter + 1);
		wxString output_resolution_statistics		= "/tmp/stats.txt";///dev/null";
		float 	 molecular_mass_kDa					= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_weight_in_kda;
		float    inner_mask_radius					= my_parent->ReconstructionInnerRadiusTextCtrl->ReturnValue();
		float    outer_mask_radius					= my_parent->ReconstructionOuterRadiusTextCtrl->ReturnValue();
		wxString dump_file_seed_1 					= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/dump_file_%li_%i_odd_.dump", current_output_refinement_id, class_counter);
		wxString dump_file_seed_2 					= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/dump_file_%li_%i_even_.dump", current_output_refinement_id, class_counter);

		my_parent->my_job_package.AddJob("ttttffftti",	output_reconstruction_1.ToUTF8().data(),
														output_reconstruction_2.ToUTF8().data(),
														output_reconstruction_filtered.ToUTF8().data(),
														output_resolution_statistics.ToUTF8().data(),
														molecular_mass_kDa, inner_mask_radius, outer_mask_radius,
														dump_file_seed_1.ToUTF8().data(),
														dump_file_seed_2.ToUTF8().data(),
														class_counter + 1);
	}
}

void RefinementManager::RunMerge3dJob()
{
	running_job_type = MERGE;


	current_job_id = main_frame->job_controller.AddJob(my_parent, run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()].gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes =  my_parent->my_job_package.my_profile.ReturnTotalJobs();

		if (number_of_refinement_processes >= 100000) my_parent->length_of_process_number = 6;
		else
		if (number_of_refinement_processes >= 10000) my_parent->length_of_process_number = 5;
		else
		if (number_of_refinement_processes >= 1000) my_parent->length_of_process_number = 4;
		else
		if (number_of_refinement_processes >= 100) my_parent->length_of_process_number = 3;
		else
		if (number_of_refinement_processes >= 10) my_parent->length_of_process_number = 2;
		else
		my_parent->length_of_process_number = 1;

		if (my_parent->length_of_process_number == 6) my_parent->NumberConnectedText->SetLabel(wxString::Format("%6i / %6li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 5) my_parent->NumberConnectedText->SetLabel(wxString::Format("%5i / %5li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 4) my_parent->NumberConnectedText->SetLabel(wxString::Format("%4i / %4li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 3) my_parent->NumberConnectedText->SetLabel(wxString::Format("%3i / %3li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 2) my_parent->NumberConnectedText->SetLabel(wxString::Format("%2i / %2li processes connected.", 0, number_of_refinement_processes));
		else

		my_parent->NumberConnectedText->SetLabel(wxString::Format("%i / %li processes connected.", 0, number_of_refinement_processes));

		my_parent->StartPanel->Show(false);
		my_parent->ProgressPanel->Show(true);

		my_parent->ExpertPanel->Show(false);
		my_parent->InfoPanel->Show(false);
		my_parent->OutputTextPanel->Show(true);
			//	CTFResultsPanel->Show(true);

		my_parent->ExpertToggleButton->Enable(false);
		my_parent->RefinementPackageComboBox->Enable(false);
		my_parent->InputParametersComboBox->Enable(false);

		my_parent->Layout();
		my_parent->running_job = true;
		my_parent->my_job_tracker.StartTracking(my_parent->my_job_package.number_of_jobs);

		}

		my_parent->ProgressBar->Pulse();
}

void RefinementManager::SetupReconstructionJob()
{
	wxArrayString written_parameter_files = output_refinement->WriteFrealignParameterFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/output_par");

	int class_counter;
	long counter;
	int job_counter;
	long number_of_reconstruction_jobs;
	long number_of_reconstruction_processes;
	long current_particle_counter;

	long number_of_particles;
	long particles_per_job;

	// for now, number of jobs is number of processes -1 (master)..

	number_of_reconstruction_processes = run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()].ReturnTotalJobs();
	number_of_reconstruction_jobs = number_of_reconstruction_processes - 1;

	number_of_particles = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles.GetCount();
	particles_per_job = ceil(number_of_particles / number_of_reconstruction_jobs);

	my_parent->my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()], "reconstruct3d", number_of_reconstruction_jobs * refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes);

	for (class_counter = 0; class_counter < refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes; class_counter++)
	{
		for (job_counter = 0; job_counter < number_of_reconstruction_jobs; job_counter++)
		{

			current_particle_counter = 1;

			wxString input_particle_stack 		= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).stack_filename;
			wxString input_parameter_file 		= written_parameter_files[class_counter];
			wxString output_reconstruction_1    = "/dev/null";
			wxString output_reconstruction_2			= "/dev/null";
			wxString output_reconstruction_filtered		= "/dev/null";
			wxString output_resolution_statistics		= "/dev/null";
			wxString my_symmetry						= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).symmetry;

			long	 first_particle						= current_particle_counter;

			current_particle_counter += particles_per_job;
			if (current_particle_counter > number_of_particles) current_particle_counter = number_of_particles;

			long	 last_particle						= current_particle_counter;
			current_particle_counter++;

			float 	 pixel_size							= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].pixel_size;
			float    voltage_kV							= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].microscope_voltage;
			float 	 spherical_aberration_mm			= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].spherical_aberration;
			float    amplitude_contrast					= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].amplitude_contrast;
			float 	 molecular_mass_kDa					= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_weight_in_kda;
			float    inner_mask_radius					= my_parent->ReconstructionInnerRadiusTextCtrl->ReturnValue();
			float    outer_mask_radius					= my_parent->ReconstructionOuterRadiusTextCtrl->ReturnValue();
			float    resolution_limit					= my_parent->ReconstructionResolutionLimitTextCtrl->ReturnValue();
			float    score_bfactor_conversion			= my_parent->ScoreToBFactorConstantTextCtrl->ReturnValue();
			float    score_threshold					= my_parent->ReconstructioScoreThreshold->ReturnValue();
			bool	 normalize_particles				= false;
			bool	 adjust_scores						= my_parent->AdjustScoreForDefocusYesRadio->GetValue();
			bool	 invert_contrast					= false;
			bool	 crop_images						= my_parent->AutoCropYesRadioButton->GetValue();
			bool	 dump_arrays						= true;
			wxString dump_file_1 						= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/dump_file_%li_%i_odd_%i.dump", current_output_refinement_id, class_counter, job_counter +1);
			wxString dump_file_2 						= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/dump_file_%li_%i_even_%i.dump", current_output_refinement_id, class_counter, job_counter + 1);

			my_parent->my_job_package.AddJob("tttttttiiffffffffffbbbbbtt",
																		input_particle_stack.ToUTF8().data(),
																		input_parameter_file.ToUTF8().data(),
																		output_reconstruction_1.ToUTF8().data(),
																		output_reconstruction_2.ToUTF8().data(),
																		output_reconstruction_filtered.ToUTF8().data(),
																		output_resolution_statistics.ToUTF8().data(),
																		my_symmetry.ToUTF8().data(),
																		first_particle, last_particle,
																		pixel_size, voltage_kV, spherical_aberration_mm, amplitude_contrast,
																		molecular_mass_kDa, inner_mask_radius, outer_mask_radius,
																		resolution_limit, score_bfactor_conversion, score_threshold,
																		normalize_particles, adjust_scores,
																		invert_contrast, crop_images, dump_arrays,
																		dump_file_1.ToUTF8().data(),
																		dump_file_2.ToUTF8().data());
		}
	}
}

// for now we take the paramter

void RefinementManager::RunReconstructionJob()
{
	running_job_type = RECONSTRUCTION;

	// in the future store the reconstruction parameters..


	// launch a controller

	current_job_id = main_frame->job_controller.AddJob(my_parent, run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()].gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes =  my_parent->my_job_package.my_profile.ReturnTotalJobs();

		if (number_of_refinement_processes >= 100000) my_parent->length_of_process_number = 6;
		else
		if (number_of_refinement_processes >= 10000) my_parent->length_of_process_number = 5;
		else
		if (number_of_refinement_processes >= 1000) my_parent->length_of_process_number = 4;
		else
		if (number_of_refinement_processes >= 100) my_parent->length_of_process_number = 3;
		else
		if (number_of_refinement_processes >= 10) my_parent->length_of_process_number = 2;
		else
		my_parent->length_of_process_number = 1;

		if (my_parent->length_of_process_number == 6) my_parent->NumberConnectedText->SetLabel(wxString::Format("%6i / %6li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 5) my_parent->NumberConnectedText->SetLabel(wxString::Format("%5i / %5li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 4) my_parent->NumberConnectedText->SetLabel(wxString::Format("%4i / %4li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 3) my_parent->NumberConnectedText->SetLabel(wxString::Format("%3i / %3li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 2) my_parent->NumberConnectedText->SetLabel(wxString::Format("%2i / %2li processes connected.", 0, number_of_refinement_processes));
		else

		my_parent->NumberConnectedText->SetLabel(wxString::Format("%i / %li processes connected.", 0, number_of_refinement_processes));

		my_parent->StartPanel->Show(false);
		my_parent->ProgressPanel->Show(true);

		my_parent->ExpertPanel->Show(false);
		my_parent->InfoPanel->Show(false);
		my_parent->OutputTextPanel->Show(true);
			//	CTFResultsPanel->Show(true);

		my_parent->ExpertToggleButton->Enable(false);
		my_parent->RefinementPackageComboBox->Enable(false);
		my_parent->InputParametersComboBox->Enable(false);

		my_parent->Layout();
		my_parent->running_job = true;
		my_parent->my_job_tracker.StartTracking(my_parent->my_job_package.number_of_jobs);

		}

		my_parent->ProgressBar->Pulse();
}

void RefinementManager::SetupRefinementJob()
{
	int class_counter;
	long counter;
	long number_of_refinement_jobs;
	int number_of_refinement_processes;
	long current_particle_counter;

	long number_of_particles;
	long particles_per_job;

	// get the last refinement for the currently selected refinement package..

	input_refinement->WriteFrealignParameterFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/input_par");
	input_refinement->WriteResolutionStatistics(main_frame->current_project.parameter_file_directory.GetFullPath() + "/input_stats");

//	wxPrintf("Input refinement has %li particles\n", input_refinement->number_of_particles);

	// for now, number of jobs is number of processes -1 (master)..

	number_of_refinement_processes = run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].ReturnTotalJobs();
	number_of_refinement_jobs = number_of_refinement_processes - 1;

	number_of_particles = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles.GetCount();
	particles_per_job = ceil(number_of_particles / number_of_refinement_jobs);

	my_parent->my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()], "refine3d", number_of_refinement_jobs * refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes);

	for (class_counter = 0; class_counter < refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes; class_counter++)
	{
		current_particle_counter = 1;

		for (counter = 0; counter < number_of_refinement_jobs; counter++)
		{

			wxString input_particle_images					= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).stack_filename;
			wxString input_parameter_file 					= main_frame->current_project.parameter_file_directory.GetFullPath() + wxString::Format("/input_par_%li_%i.par", current_input_refinement_id, class_counter + 1);
			wxString input_reconstruction					= volume_asset_panel->ReturnAssetLongFilename(volume_asset_panel->ReturnArrayPositionFromAssetID(refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).references_for_next_refinement[class_counter]));
			wxString input_reconstruction_statistics 		= main_frame->current_project.parameter_file_directory.GetFullPath() + wxString::Format("/input_stats_%li_%i.txt", current_input_refinement_id, class_counter + 1);
			bool	 use_statistics							= true;

			wxString ouput_matching_projections		 		= "";
			wxString output_parameter_file					= "/dev/null";
			wxString ouput_shift_file						= "/dev/null";
			wxString my_symmetry							= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).symmetry;
			long	 first_particle							= current_particle_counter;

			current_particle_counter += particles_per_job;
			if (current_particle_counter > number_of_particles) current_particle_counter = number_of_particles;

			long	 last_particle							= current_particle_counter;
			current_particle_counter++;

			// for now we take the paramters of the first image!!!!

			float 	 pixel_size								= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].pixel_size;
			float    voltage_kV								= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].microscope_voltage;
			float 	 spherical_aberration_mm				= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].spherical_aberration;
			float    amplitude_contrast						= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].amplitude_contrast;
			float	 molecular_mass_kDa						= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_weight_in_kda;
			float    mask_radius							= my_parent->MaskRadiusTextCtrl->ReturnValue();
			float    low_resolution_limit					= my_parent->LowResolutionLimitTextCtrl->ReturnValue();
			float    high_resolution_limit					= my_parent->HighResolutionLimitTextCtrl->ReturnValue();
			float	 signed_CC_limit						= my_parent->SignedCCResolutionTextCtrl->ReturnValue();
			float	 classification_resolution_limit		= my_parent->ClassificationHighResLimitTextCtrl->ReturnValue();
			float    mask_radius_search						= my_parent->GlobalMaskRadiusTextCtrl->ReturnValue();
			float	 high_resolution_limit_search			= my_parent->GlobalResolutionLimitTextCtrl->ReturnValue();
			float	 angular_step							= my_parent->AngularStepTextCtrl->ReturnValue();
			int		 best_parameters_to_keep				= my_parent->NumberToRefineSpinCtrl->GetValue();
			float	 max_search_x							= my_parent->SearchRangeXTextCtrl->ReturnValue();
			float	 max_search_y							= my_parent->SearchRangeYTextCtrl->ReturnValue();
			float    mask_center_2d_x						= my_parent->SphereXTextCtrl->ReturnValue();
			float 	 mask_center_2d_y						= my_parent->SphereYTextCtrl->ReturnValue();
			float    mask_center_2d_z						= my_parent->SphereZTextCtrl->ReturnValue();
			float    mask_radius_2d							= my_parent->SphereRadiusTextCtrl->ReturnValue();

			float	 defocus_search_range					= my_parent->DefocusSearchRangeTextCtrl->ReturnValue();
			float	 defocus_step							= my_parent->DefocusSearchStepTextCtrl->ReturnValue();
			float	 padding								= 1.0;

			bool global_search;
			bool local_refinement;

			if (my_parent->GlobalRefinementRadio->GetValue() == true)
			{
				global_search = true;
				local_refinement = false;
			}
			else
			{
				global_search = false;
				local_refinement = true;
			}


			bool refine_psi 								= my_parent->RefinePsiCheckBox->GetValue();
			bool refine_theta								= my_parent->RefineThetaCheckBox->GetValue();
			bool refine_phi									= my_parent->RefinePhiCheckBox->GetValue();
			bool refine_x_shift								= my_parent->RefineXShiftCheckBox->GetValue();
			bool refine_y_shift								= my_parent->RefineYShiftCheckBox->GetValue();
			bool calculate_matching_projections				= false;
			bool apply_2d_masking							= my_parent->SphereClassificatonYesRadio->GetValue();
			bool ctf_refinement								= my_parent->RefineCTFYesRadio->GetValue();
			bool invert_contrast							= false;


			my_parent->my_job_package.AddJob("ttttbttttiifffffffffffffifffffffffbbbbbbbbbbbi",
																input_particle_images.ToUTF8().data(), 				// 0
																input_parameter_file.ToUTF8().data(), 				// 1
																input_reconstruction.ToUTF8().data(), 				// 2
																input_reconstruction_statistics.ToUTF8().data(),	// 3
																use_statistics, 							// 4
																ouput_matching_projections.ToUTF8().data(),			// 5
																output_parameter_file.ToUTF8().data(),				// 6
																ouput_shift_file.ToUTF8().data(),					// 7
																my_symmetry.ToUTF8().data(),						// 8
																first_particle, 							// 9
																last_particle,								// 10
																pixel_size, 								// 11
																voltage_kV,									// 12
																spherical_aberration_mm,					// 13
																amplitude_contrast,							// 14
																molecular_mass_kDa,							// 15
																mask_radius,								// 16
																low_resolution_limit,						// 17
																high_resolution_limit,						// 18
																signed_CC_limit,							// 19
																classification_resolution_limit,			// 20
																mask_radius_search,							// 21
																high_resolution_limit_search,				// 22
																angular_step,								// 23
																best_parameters_to_keep,					// 24
																max_search_x,								// 25
																max_search_y,								// 26
																mask_center_2d_x,							// 27
																mask_center_2d_y,							// 28
																mask_center_2d_z,							// 29
																mask_radius_2d,								// 30
																defocus_search_range,						// 31
																defocus_step,								// 32
																padding,									// 33
																global_search,								// 34
																local_refinement,							// 35
																refine_psi,									// 36
																refine_theta,								// 37
																refine_phi, 								// 38
																refine_x_shift,								// 39
																refine_y_shift,								// 40
																calculate_matching_projections,				// 41
																apply_2d_masking,							// 42
																ctf_refinement,								// 43
																invert_contrast,
																class_counter);								// 44


		}

	}
}

void RefinementManager::ProcessJobResult(JobResult *result_to_process)
{
	if (running_job_type == REFINEMENT)
	{
		int current_class = int(result_to_process->result_data[0] + 0.5);
		long current_particle = long(result_to_process->result_data[1] + 0.5) - 1;

		//wxPrintf("Received a refinement result for class #%i, particle %li\n", current_class + 1, current_particle + 1);
		//wxPrintf("output refinement has %i classes and %li particles\n", output_refinement->number_of_classes, output_refinement->number_of_particles);


		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].position_in_stack = long(result_to_process->result_data[1] + 0.5);
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].psi = result_to_process->result_data[2];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].theta = result_to_process->result_data[3];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].phi = result_to_process->result_data[4];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].xshift = result_to_process->result_data[5];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].yshift = result_to_process->result_data[6];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].defocus1 = result_to_process->result_data[9];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].defocus2 = result_to_process->result_data[10];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].defocus_angle = result_to_process->result_data[11];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].occupancy = result_to_process->result_data[12];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].logp = result_to_process->result_data[13];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].sigma = result_to_process->result_data[14];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].score = result_to_process->result_data[15];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].score_change = result_to_process->result_data[16];

	}
	else
	if (running_job_type == MERGE)
	{
		wxPrintf("received merge result!\n");

		// add to the correct resolution statistics..

		int number_of_points = result_to_process->result_data[0];
		int class_number = int(result_to_process->result_data[1] + 0.5);
		int array_position = 2;
		float current_resolution;
		float fsc;
		float part_fsc;
		float part_ssnr;
		float rec_ssnr;

		//wxPrintf("class_number = %i\n", class_number);
		// add the points..

		output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.Init(output_refinement->resolution_statistics_pixel_size, output_refinement->resolution_statistics_box_size);

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

void RefinementManager::ProcessAllJobsFinished()
{

	if (running_job_type == REFINEMENT)
	{
		wxPrintf("Refinement has finished\n");
		main_frame->job_controller.KillJob(my_parent->my_job_id);
		SetupReconstructionJob();
		RunReconstructionJob();

	}
	else
	if (running_job_type == RECONSTRUCTION)
	{
		main_frame->job_controller.KillJob(my_parent->my_job_id);
		wxPrintf("Reconstruction has finished\n");
		SetupMerge3dJob();
		RunMerge3dJob();
	}
	else
	if (running_job_type == MERGE)
	{
		main_frame->job_controller.KillJob(my_parent->my_job_id);

		VolumeAsset temp_asset;

		temp_asset.reconstruction_job_id = current_output_refinement_id;
		temp_asset.pixel_size = output_refinement->resolution_statistics_pixel_size;
		temp_asset.x_size = output_refinement->resolution_statistics_box_size;
		temp_asset.y_size = output_refinement->resolution_statistics_box_size;
		temp_asset.z_size = output_refinement->resolution_statistics_box_size;

		// add the volumes to the database..

		output_refinement->reference_volume_ids.Clear();
		refinement_package_asset_panel->all_refinement_packages[my_parent->RefinementPackageComboBox->GetSelection()].references_for_next_refinement.Clear();
		main_frame->current_project.database.BeginVolumeAssetInsert();


		for (int class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
		{
			temp_asset.asset_id = volume_asset_panel->current_asset_number;
			temp_asset.asset_name = wxString::Format("Volume #%li - Class #%i", current_output_refinement_id, class_counter + 1);
			temp_asset.filename = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/volume_%li_%i.mrc", output_refinement->refinement_id, class_counter + 1);
			output_refinement->reference_volume_ids.Add(temp_asset.asset_id);
			refinement_package_asset_panel->all_refinement_packages[my_parent->RefinementPackageComboBox->GetSelection()].references_for_next_refinement.Add(temp_asset.asset_id);
			volume_asset_panel->AddAsset(&temp_asset);
			main_frame->current_project.database.AddNextVolumeAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath(), temp_asset.reconstruction_job_id, temp_asset.pixel_size, temp_asset.x_size, temp_asset.y_size, temp_asset.z_size);
		}
		main_frame->current_project.database.EndVolumeAssetInsert();

		// write the refinement to the database
		main_frame->current_project.database.AddRefinement(output_refinement);
		volume_asset_panel->is_dirty = true;
		refinement_package_asset_panel->is_dirty = true;

		// add this refinment to the refinement package..

		refinement_package_asset_panel->all_refinement_packages[my_parent->RefinementPackageComboBox->GetSelection()].last_refinment_id = output_refinement->refinement_id;

		CycleRefinement();
	}
}

void RefinementManager::CycleRefinement()
{
	number_of_rounds_run++;

	if (number_of_rounds_run < number_of_rounds_to_run)
	{
		current_input_refinement_id = output_refinement->refinement_id;
		delete input_refinement;
		input_refinement = output_refinement;
		output_refinement = new Refinement;

		SetupRefinementJob();
		RunRefinementJob();
	}
	else
	{
		delete input_refinement;
		delete output_refinement;
		my_parent->WriteInfoText("All refinement cycles are finished!");
	}
}

