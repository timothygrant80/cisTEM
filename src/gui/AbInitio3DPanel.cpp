#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyVolumeAssetPanel *volume_asset_panel;

wxDEFINE_EVENT(wxEVT_COMMAND_MASKERTHREAD_COMPLETED, wxThreadEvent);

AbInitio3DPanel::AbInitio3DPanel( wxWindow* parent )
:
AbInitio3DPanelParent( parent )
{

	my_job_id = -1;
	running_job = false;

	SetInfo();

	ShowOrthDisplayPanel->Initialise(START_WITH_FOURIER_SCALING | DO_NOT_SHOW_STATUS_BAR);

	wxSize input_size = InputSizer->GetMinSize();
	input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
	input_size.y = -1;
	ExpertPanel->SetMinSize(input_size);
	ExpertPanel->SetSize(input_size);

	refinement_package_combo_is_dirty = false;
	run_profiles_are_dirty = false;
	selected_refinement_package = -1;

	my_abinitio_manager.SetParent(this);
	RefinementPackageComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &AbInitio3DPanel::OnRefinementPackageComboBox, this);
	Bind(wxEVT_COMMAND_MASKERTHREAD_COMPLETED, &AbInitio3DPanel::OnMaskerThreadComplete, this);
	Bind(MY_ORTH_DRAW_EVENT, &AbInitio3DPanel::OnOrthThreadComplete, this);
	FillRefinementPackagesComboBox();

	// limits

	InitialResolutionLimitTextCtrl->SetMinMaxValue(0, 300);
	FinalResolutionLimitTextCtrl->SetMinMaxValue(0, 300);
	StartPercentUsedTextCtrl->SetMinMaxValue(0.001, 100);
	EndPercentUsedTextCtrl->SetMinMaxValue(0.001, 100);
	SmoothingFactorTextCtrl->SetMinMaxValue(0, 1);
}

void AbInitio3DPanel::SetInfo()
{
	InfoText->GetCaret()->Hide();

	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("Ab-Initio 3D Reconstruction"));
	InfoText->EndFontSize();
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("When a prior 3D reconstruction of a molecule or complex is available that has a closely related structure, it is usually fastest and safest to use it to initialize 3D refinement and reconstruction of a new dataset. However, in many cases such a structure is not available, or an independently determined structure is desired. Ab-initio 3D reconstruction offers a way to start refinement and reconstruction without any prior structural information."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->WriteText(wxT("It is advisable to precede the ab-initio step with a 2D classification step to remove junk particles and select a high-quality subset of the data. A refinement package has to be created (in Assets) that will be used with the ab-initio procedure, either from selected 2D classes or using picked particle position. The idea of the ab-initio algorithm is to iteratively improve a 3D reconstruction, starting with a reconstruction calculated from random Euler angles, by aligning a small percentage of the data against the current reconstruction and increasing the refinement resolution and percentage at each iteration (Grigorieff, 2016). This procedure can also be carried out using multiple references that must be specified when creating the refinement package. The procedure stops after a user-specified number of refinement cycles and restarts if more than one starts are specified."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->WriteText(wxT("The progress of the ab-initio reconstruction is displayed as a plot of the average sigma value that measures the average apparent noise-to-signal ratio in the data. The sigma value should decrease as the reconstruction gets closer to the true structure. The current reconstruction is also displayed as three orthogonal central slices and three orthogonal projections."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->WriteText(wxT("If the ab-initio procedure fails on a symmetrical particle, users should repeat it using C1 (no symmetry). This can be specified by creating a new refinement package that is based on the previous refinement package, and changing the symmetry to C1. If a particle is close to spherical, such as apoferritin, it may be necessary to change the initial and final resolution limits from 40 Å and 9 Å (default) to higher resolution, e.g. 15 Å and 6 Å (see Expert Options). Finally, it is worth repeating the procedure a few times if a good reconstruction is not obtained in the first trial."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("Program Options"));
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Input Refinement Package : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The name of the refinement package previously set up in the Assets panel (providing details of particle locations, box size and imaging parameters)."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Number of Starts : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The number of times the ab-initio reconstruction is restarted, using the result from the previous run in each restart."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("No. of Cycles per Start : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The number of refinement cycles to run for each start. The percentage of particles and the refinement resolution limit will be adjusted automatically from cycle to cycle using initial and final values specified under Expert Options."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("Expert Options"));
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Initial Resolution Limit (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The starting resolution limit used to align particles against the current 3D reconstruction. In most cases, this should specify a relatively low resolution to make sure the reconstructions generated in the initial refinement cycles do not develop spurious high-resolution features."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Final Resolution Limit (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The resolution limit used in the final refinement cycle. In most cases, this should specify a resolution at which expected secondary structure becomes apparent, i.e. around 9 Å."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Use Auto-Masking? "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Should the 3D reconstructions be masked? Masking is important to suppress weak density features that usually appear in the early stages of ab-initio reconstruction, thus preventing them to get amplified during the iterative refinement. Masking should only be disabled if it appears to interfere with the reconstruction process."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Auto Percent used? "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Should the percentage of particles used in each refinement cycle be set automatically? If reconstructions appear very noisy or reconstructions settle into a wrong structure that does not change anymore during iterations, disable this option and specify initial and final percentages manually. To reduce noise, increase the percentage; to make reconstructions more variable, decrease the percentage. By default, the initial percentage is set to include an equivalent of 2500 asymmetric units and the final percentage corresponds to 10,000 asymmetric units used."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Initial % Used / Final % Used : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("User-specified percentages of particles used when Auto Percent Used is disabled."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Apply Likelihood Blurring? "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Should the reconstructions be blurred by inserting each particle image at multiple orientations, weighted by a likelihood function? Enable this option if the ab-initio procedure appears to suffer from over-fitting and the appearance of spurious high-resolution features."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Smoothing Factor : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("A factor that reduces the range of likelihoods used for blurring. A smaller number leads to more blurring. The user should try values between 0.1 and 1."));
	InfoText->Newline();
	InfoText->Newline();


	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("References"));
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Grigorieff, N.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2016. Frealign: An exploratory tool for single-particle cryo-EM. Methods Enzymol. 579, 191-226. "));
	InfoText->BeginURL("http://dx.doi.org/10.1016/bs.mie.2016.04.013");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);

	InfoText->WriteText(wxT("doi:10.1016/bs.mie.2016.04.013"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->EndAlignment();
	InfoText->Newline();
	InfoText->Newline();
}

void AbInitio3DPanel::OnInfoURL( wxTextUrlEvent& event )
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


void AbInitio3DPanel::WriteInfoText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void AbInitio3DPanel::WriteBlueText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLUE));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void AbInitio3DPanel::WriteWarningText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(wxColor(255,165,0)));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}


void AbInitio3DPanel::WriteErrorText(wxString text_to_write)
{
	 output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
	 output_textctrl->AppendText(text_to_write);

	 if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}


void AbInitio3DPanel::FillRefinementPackagesComboBox()
{
	if (RefinementPackageComboBox->FillComboBox() == false) NewRefinementPackageSelected();
}

void AbInitio3DPanel::FillRunProfileComboBoxes()
{
	ReconstructionRunProfileComboBox->FillWithRunProfiles();
	RefinementRunProfileComboBox->FillWithRunProfiles();
}

void AbInitio3DPanel::NewRefinementPackageSelected()
{
	selected_refinement_package = RefinementPackageComboBox->GetSelection();
	SetDefaults();
	//wxPrintf("New Refinement Package Selection\n");

}

void AbInitio3DPanel::AbInitio3DPanel::SetDefaults()
{
	if (RefinementPackageComboBox->GetCount() > 0)
	{
		wxString current_symmetry_string = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).symmetry;
		wxChar   symmetry_type;

		current_symmetry_string = current_symmetry_string.Trim();
		current_symmetry_string = current_symmetry_string.Trim(false);

		MyDebugAssertTrue(current_symmetry_string.Length() > 0, "symmetry string is blank");
		symmetry_type = current_symmetry_string.Capitalize()[0];

		ExpertPanel->Freeze();

		if (symmetry_type == 'O' || symmetry_type == 'I')
		{
			NumberStartsSpinCtrl->SetValue(1);
			AutoMaskNoRadio->SetValue(true);
		}
		else
		{
			NumberStartsSpinCtrl->SetValue(2);
			AutoMaskYesRadio->SetValue(true);
		}

//		NumberStartsSpinCtrl->SetValue(2);
		NumberRoundsSpinCtrl->SetValue(40);

		InitialResolutionLimitTextCtrl->ChangeValueFloat(40);
		FinalResolutionLimitTextCtrl->ChangeValueFloat(9);
		AutoPercentUsedYesRadio->SetValue(true);

		StartPercentUsedTextCtrl->ChangeValueFloat(10);
		EndPercentUsedTextCtrl->ChangeValueFloat(10);

		ApplyBlurringNoRadioButton->SetValue(true);
		SmoothingFactorTextCtrl->ChangeValueFloat(1.00);

		ExpertPanel->Thaw();
	}
}

void AbInitio3DPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	if (main_frame->current_project.is_open == false)
	{
		RefinementPackageComboBox->Enable(false);
		RefinementRunProfileComboBox->Enable(false);
		ReconstructionRunProfileComboBox->Enable(false);
		ExpertToggleButton->Enable(false);
		StartRefinementButton->Enable(false);
		NumberStartsSpinCtrl->Enable(false);
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
		if (running_job == false)
		{
			RefinementRunProfileComboBox->Enable(true);
			ReconstructionRunProfileComboBox->Enable(true);
			ExpertToggleButton->Enable(true);

			if (RefinementPackageComboBox->GetCount() > 0)
			{
				RefinementPackageComboBox->Enable(true);

				if (PleaseCreateRefinementPackageText->IsShown())
				{
					PleaseCreateRefinementPackageText->Show(false);
					Layout();
				}

			}
			else
			{
				RefinementPackageComboBox->ChangeValue("");
				RefinementPackageComboBox->Enable(false);

				if (PleaseCreateRefinementPackageText->IsShown() == false)
				{
					PleaseCreateRefinementPackageText->Show(true);
					Layout();
				}
			}

			NumberRoundsSpinCtrl->Enable(true);
			NumberStartsSpinCtrl->Enable(true);

			if (ExpertToggleButton->GetValue() == true)
			{
				if (AutoPercentUsedYesRadio->GetValue() == true)
				{
					InitialPercentUsedStaticText->Enable(false);
					StartPercentUsedTextCtrl->Enable(false);
					FinalPercentUsedStaticText->Enable(false);
					EndPercentUsedTextCtrl->Enable(false);
				}
				else
				{
					InitialPercentUsedStaticText->Enable(true);
					StartPercentUsedTextCtrl->Enable(true);
					FinalPercentUsedStaticText->Enable(true);
					EndPercentUsedTextCtrl->Enable(true);
				}

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

			}



			bool estimation_button_status = false;

			if (RefinementPackageComboBox->GetCount() > 0 && ReconstructionRunProfileComboBox->GetCount() > 0)
			{
				if (run_profiles_panel->run_profile_manager.ReturnTotalJobs(RefinementRunProfileComboBox->GetSelection()) > 1 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(ReconstructionRunProfileComboBox->GetSelection()) > 1)
				{
					if (RefinementPackageComboBox->GetSelection() != wxNOT_FOUND)
					{
						estimation_button_status = true;
					}

				}
			}

			StartRefinementButton->Enable(estimation_button_status);
		}
		else
		{
			RefinementPackageComboBox->Enable(false);
			NumberStartsSpinCtrl->Enable(false);
			NumberRoundsSpinCtrl->Enable(false);
			ExpertToggleButton->Enable(false);

			if (ExpertToggleButton->GetValue() == true) ExpertToggleButton->SetValue(false);

			if (my_abinitio_manager.number_of_rounds_run > 0 || my_abinitio_manager.number_of_starts_run > 0)
			{
				TakeCurrentResultButton->Enable(true);
			}
			else TakeCurrentResultButton->Enable(false);

			if (my_abinitio_manager.number_of_starts_run > 0)
			{
				TakeLastStartResultButton->Enable(true);
			}
			else TakeLastStartResultButton->Enable(false);

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
	}


}

void AbInitio3DPanel::OnExpertOptionsToggle( wxCommandEvent& event )
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

void AbInitio3DPanel::TerminateButtonClick( wxCommandEvent& event )
{
	main_frame->job_controller.KillJob(my_job_id);
	Freeze();
	WriteBlueText("Terminated Job");
	TimeRemainingText->SetLabel("Time Remaining : Terminated");
	CancelAlignmentButton->Show(false);
	CurrentLineOne->Show(false);
	CurrentLineTwo->Show(false);
	TakeCurrentResultButton->Show(false);
	TakeLastStartResultButton->Show(false);
	FinishButton->Show(true);
	ProgressPanel->Layout();
	Thaw();
}

void AbInitio3DPanel::FinishButtonClick( wxCommandEvent& event )
{
	ProgressBar->SetValue(0);
	TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
	CurrentLineOne->Show(true);
	CurrentLineTwo->Show(true);
	TakeCurrentResultButton->Show(true);
	TakeLastStartResultButton->Show(true);
	FinishButton->Show(false);

	InputParamsPanel->Show(true);

	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	output_textctrl->Clear();
	PlotPanel->Show(false);
	PlotPanel->Clear();
	OrthResultsPanel->Show(false);
	ShowOrthDisplayPanel->Clear();
	//FSCResultsPanel->Show(false);
	//AngularPlotPanel->Show(false);
	//CTFResultsPanel->Show(false);
	//graph_is_hidden = true;
	InfoPanel->Show(true);

	if (ExpertToggleButton->GetValue() == true) ExpertPanel->Show(true);
	else ExpertPanel->Show(false);
	running_job = false;
	Layout();


	global_delete_startup_scratch();


}

void AbInitio3DPanel::StartRefinementClick( wxCommandEvent& event )
{
	my_abinitio_manager.BeginRefinementCycle();
}

void AbInitio3DPanel::ResetAllDefaultsClick( wxCommandEvent& event )
{
	SetDefaults();
}

void AbInitio3DPanel::OnJobSocketEvent(wxSocketEvent& event)
{
    SETUP_SOCKET_CODES

	 wxString s = _("OnSocketEvent: ");
	 wxSocketBase *sock = event.GetSocket();
	 sock->SetFlags(wxSOCKET_BLOCK | wxSOCKET_WAITALL);

	//  MyDebugAssertTrue(sock == main_frame->job_controller.job_list[my_job_id].socket, "Socket event from Non conduit socket??");

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
	     ReadFromSocket(sock, &socket_input_buffer, SOCKET_CODE_SIZE);


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
	 		 ReadFromSocket(sock, &finished_job, 4);

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

	    	  my_abinitio_manager.ProcessJobResult(&temp_result);
	    	  wxPrintf("Warning: Received socket_job_result - should this happen?");

	 	  }
	      else
	      if (memcmp(socket_input_buffer, socket_job_result_queue, SOCKET_CODE_SIZE) == 0) // identification
	 	  {
	    	  ArrayofJobResults temp_queue;
	    	  ReceiveResultQueueFromSocket(sock, temp_queue);

	    	  for (int counter = 0; counter < temp_queue.GetCount(); counter++)
	    	  {
	    		  my_abinitio_manager.ProcessJobResult(&temp_queue.Item(counter));
	    	  }
	 	  }
	      else
		  if (memcmp(socket_input_buffer, socket_number_of_connections, SOCKET_CODE_SIZE) == 0) // identification
		  {
			  // how many connections are there?

			  int number_of_connections;
			  ReadFromSocket(sock, &number_of_connections, 4);

             my_job_tracker.AddConnection();

   //          if (graph_is_hidden == true) ProgressBar->Pulse();

             //WriteInfoText(wxString::Format("There are now %i connections\n", number_of_connections));

             // send the info to the gui

             int total_processes = my_job_package.my_profile.ReturnTotalJobs();
             if (my_job_package.number_of_jobs + 1 < my_job_package.my_profile.ReturnTotalJobs()) total_processes = my_job_package.number_of_jobs + 1;
             else total_processes =  my_job_package.my_profile.ReturnTotalJobs();


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
			  my_abinitio_manager.ProcessAllJobsFinished();
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

void AbInitio3DPanel::OnRefinementPackageComboBox( wxCommandEvent& event )
{
	NewRefinementPackageSelected();
}


void AbInitio3DPanel::TakeLastStartClicked( wxCommandEvent& event )
{
	main_frame->job_controller.KillJob(my_job_id);
	Freeze();
	WriteBlueText("Terminating job, and importing the result at the end of the previous start.");
	TimeRemainingText->SetLabel("Time Remaining : Stopped");
	CancelAlignmentButton->Show(false);
	CurrentLineOne->Show(false);
	CurrentLineTwo->Show(false);
	TakeCurrentResultButton->Show(false);
	TakeLastStartResultButton->Show(false);
	FinishButton->Show(true);
	ProgressPanel->Layout();
	Thaw();

	TakeLastStart();

}

void AbInitio3DPanel::TakeCurrentClicked( wxCommandEvent& event )
{
	main_frame->job_controller.KillJob(my_job_id);
	Freeze();
	WriteBlueText("Terminating job, and importing the current result.");
	TimeRemainingText->SetLabel("Time Remaining : Stopped");
	CancelAlignmentButton->Show(false);
	CurrentLineOne->Show(false);
	CurrentLineTwo->Show(false);
	TakeCurrentResultButton->Show(false);
	TakeLastStartResultButton->Show(false);
	FinishButton->Show(true);
	ProgressPanel->Layout();
	Thaw();

	TakeCurrent();
}


void AbInitio3DPanel::TakeCurrent()
{
	// import the current 3ds into the database.

	wxString current_output_filename;
	wxString input_file;
	wxArrayString all_output_filenames;

	bool have_error = false;
	long current_startup_id = main_frame->current_project.database.ReturnHighestStartupID() + 1;

	// copy the files..

	for (int class_counter = 0; class_counter < my_abinitio_manager.input_refinement->number_of_classes; class_counter++)
	{

		current_output_filename = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/startup_volume_%li_%i.mrc", current_startup_id, class_counter + 1);
		all_output_filenames.Add(current_output_filename);

		// what is the current round number

		int current_round_number = (my_abinitio_manager.number_of_rounds_to_run * my_abinitio_manager.number_of_starts_run) + my_abinitio_manager.number_of_rounds_run - 1;
		input_file = main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup3d_%i_%i.mrc", current_round_number, class_counter);

		if (wxCopyFile(input_file, current_output_filename) == false)
		{
			have_error = true;
			WriteErrorText(wxString::Format("Cannot Copy File (%s to %s) - Startup volumes were not imported", input_file, current_output_filename));
		}
	}

	if (have_error == false) // now add to database
	{
		VolumeAsset temp_asset;
		wxArrayLong volume_asset_ids;

		main_frame->current_project.database.Begin();
		main_frame->current_project.database.BeginVolumeAssetInsert();

		for (int class_counter = 0; class_counter < my_abinitio_manager.current_reference_filenames.GetCount(); class_counter++)
		{

			temp_asset.reconstruction_job_id = current_startup_id;
			temp_asset.pixel_size = my_abinitio_manager.input_refinement->resolution_statistics_pixel_size;
			temp_asset.x_size = my_abinitio_manager.input_refinement->resolution_statistics_box_size;
			temp_asset.y_size = my_abinitio_manager.input_refinement->resolution_statistics_box_size;
			temp_asset.z_size = my_abinitio_manager.input_refinement->resolution_statistics_box_size;

			temp_asset.asset_id = volume_asset_panel->current_asset_number;
			volume_asset_ids.Add(temp_asset.asset_id);
			temp_asset.asset_name = wxString::Format("Volume From Startup #%li - Class #%i", current_startup_id, class_counter + 1);
			temp_asset.filename = all_output_filenames[class_counter];

			volume_asset_panel->AddAsset(&temp_asset);
			main_frame->current_project.database.AddNextVolumeAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath(), temp_asset.reconstruction_job_id, temp_asset.pixel_size, temp_asset.x_size, temp_asset.y_size, temp_asset.z_size);
		}

		main_frame->current_project.database.EndVolumeAssetInsert();

		// now add the details of the startup job..

		main_frame->current_project.database.AddStartupJob(current_startup_id, my_abinitio_manager.input_refinement->refinement_package_asset_id, wxString::Format("Refinement #%li", current_startup_id), my_abinitio_manager.number_of_starts_run, my_abinitio_manager.number_of_rounds_to_run, InitialResolutionLimitTextCtrl->ReturnValue(), FinalResolutionLimitTextCtrl->ReturnValue(), AutoMaskYesRadio->GetValue(), AutoPercentUsedYesRadio->GetValue(), my_abinitio_manager.start_percent_used, my_abinitio_manager.end_percent_used, my_abinitio_manager.input_refinement->resolution_statistics_box_size * 0.45 * my_abinitio_manager.input_refinement->resolution_statistics_pixel_size,  ApplyBlurringYesRadioButton->GetValue(), SmoothingFactorTextCtrl->ReturnValue(), volume_asset_ids);
		main_frame->current_project.database.Commit();
	}

	main_frame->DirtyVolumes();



}

void AbInitio3DPanel::TakeLastStart()
{
	// import the current 3ds into the database.

	wxString current_output_filename;
	wxString input_file;
	wxArrayString all_output_filenames;

	bool have_error = false;
	long current_startup_id = main_frame->current_project.database.ReturnHighestStartupID() + 1;

	// copy the files..

	for (int class_counter = 0; class_counter < my_abinitio_manager.input_refinement->number_of_classes; class_counter++)
	{

		current_output_filename = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/startup_volume_%li_%i.mrc", current_startup_id, class_counter + 1);
		all_output_filenames.Add(current_output_filename);

		// what is the round number of the end of the last start..

		int last_start_round_number = (my_abinitio_manager.number_of_rounds_to_run * my_abinitio_manager.number_of_starts_run) - 1;
		input_file = main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup3d_%i_%i.mrc", last_start_round_number, class_counter);

		if (wxCopyFile(input_file, current_output_filename) == false)
		{
			have_error = true;
			WriteErrorText(wxString::Format("Cannot Copy File (%s to %s) - Startup volumes were not imported", input_file, current_output_filename));
		}
	}

	if (have_error == false) // now add to database
	{
		VolumeAsset temp_asset;
		wxArrayLong volume_asset_ids;

		main_frame->current_project.database.Begin();
		main_frame->current_project.database.BeginVolumeAssetInsert();

		for (int class_counter = 0; class_counter < my_abinitio_manager.current_reference_filenames.GetCount(); class_counter++)
		{

			temp_asset.reconstruction_job_id = current_startup_id;
			temp_asset.pixel_size = my_abinitio_manager.input_refinement->resolution_statistics_pixel_size;
			temp_asset.x_size = my_abinitio_manager.input_refinement->resolution_statistics_box_size;
			temp_asset.y_size = my_abinitio_manager.input_refinement->resolution_statistics_box_size;
			temp_asset.z_size = my_abinitio_manager.input_refinement->resolution_statistics_box_size;

			temp_asset.asset_id = volume_asset_panel->current_asset_number;
			volume_asset_ids.Add(temp_asset.asset_id);
			temp_asset.asset_name = wxString::Format("Volume From Startup #%li - Class #%i", current_startup_id, class_counter + 1);
			temp_asset.filename = all_output_filenames[class_counter];

			volume_asset_panel->AddAsset(&temp_asset);
			main_frame->current_project.database.AddNextVolumeAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath(), temp_asset.reconstruction_job_id, temp_asset.pixel_size, temp_asset.x_size, temp_asset.y_size, temp_asset.z_size);
		}

		main_frame->current_project.database.EndVolumeAssetInsert();

		// now add the details of the startup job..

		main_frame->current_project.database.AddStartupJob(current_startup_id, my_abinitio_manager.input_refinement->refinement_package_asset_id, wxString::Format("Refinement #%li", current_startup_id), my_abinitio_manager.number_of_starts_run, my_abinitio_manager.number_of_rounds_to_run, InitialResolutionLimitTextCtrl->ReturnValue(), FinalResolutionLimitTextCtrl->ReturnValue(), AutoMaskYesRadio->GetValue(), AutoPercentUsedYesRadio->GetValue(), my_abinitio_manager.start_percent_used, my_abinitio_manager.end_percent_used, my_abinitio_manager.input_refinement->resolution_statistics_box_size * 0.45 * my_abinitio_manager.input_refinement->resolution_statistics_pixel_size,  ApplyBlurringYesRadioButton->GetValue(), SmoothingFactorTextCtrl->ReturnValue(), volume_asset_ids);
		main_frame->current_project.database.Commit();
	}

	main_frame->DirtyVolumes();


}

AbInitioManager::AbInitioManager()
{

	number_of_starts_to_run = 2;
	number_of_starts_run = 0;


}

void AbInitioManager::SetParent(AbInitio3DPanel *wanted_parent)
{
	my_parent = wanted_parent;
}

void AbInitioManager::BeginRefinementCycle()
{
	long counter;
	int class_counter;

	start_with_reconstruction = true;

	number_of_starts_run = 0;
	number_of_rounds_run = 0;

	number_of_starts_to_run = my_parent->NumberStartsSpinCtrl->GetValue();
	number_of_rounds_to_run = my_parent->NumberRoundsSpinCtrl->GetValue();

	my_parent->PlotPanel->Clear();
	my_parent->PlotPanel->my_notebook->SetSelection(0);


	current_refinement_package_asset_id = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).asset_id;

	// this should be the random start..
	current_input_refinement_id = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).refinement_ids[0];
	// create a refinement with random angles etc..

	input_refinement = main_frame->current_project.database.GetRefinementByID(current_input_refinement_id);
	input_refinement->refinement_id = 0;
	output_refinement = input_refinement;
	current_output_refinement_id = input_refinement->refinement_id;

	// work out the percent used

	long number_of_particles = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles.GetCount();
	int number_of_classes = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes;

	// re-randomise the input parameters, and set the default resolution statistics..

	for (class_counter = 0; class_counter < number_of_classes; class_counter++)
	{
		for ( counter = 0; counter < number_of_particles; counter++)
		{
			if (number_of_classes == 1) input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = 100.0;
			else input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = 100.00 / input_refinement->number_of_classes;

			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].phi = global_random_number_generator.GetUniformRandom() * 180.0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].theta = global_random_number_generator.GetUniformRandom() * 180.0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].psi = global_random_number_generator.GetUniformRandom() * 180.0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].xshift = 0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].yshift = 0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].score = 0.0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].image_is_active = 1;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].sigma = 1.0;
		}

		input_refinement->class_refinement_results[class_counter].class_resolution_statistics.GenerateDefaultStatistics(refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_weight_in_kda);
	}

	// need to take into account symmetry

	wxString current_symmetry_string = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).symmetry;
	wxChar   symmetry_type;
	long     symmetry_number;

	current_symmetry_string = current_symmetry_string.Trim();
	current_symmetry_string = current_symmetry_string.Trim(false);

	MyDebugAssertTrue(current_symmetry_string.Length() > 0, "symmetry string is blank");
	symmetry_type = current_symmetry_string.Capitalize()[0];

	if (current_symmetry_string.Length() == 1)
	{
		symmetry_number = 0;
	}
	else
	{
		if (! current_symmetry_string.Mid(1).ToLong(&symmetry_number))
		{
			MyPrintWithDetails("Error: Invalid n after symmetry symbol: %s\n", current_symmetry_string.Mid(1));
			abort();
		}
	}

	if (my_parent->AutoPercentUsedYesRadio->GetValue() == true)
	{
		int symmetry_number = ReturnNumberofAsymmetricUnits(refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).symmetry);

		long number_of_asym_units = number_of_particles * symmetry_number;

		long wanted_start_number_of_asym_units = 2500 * number_of_classes;
		long wanted_end_number_of_asym_units = 10000 * number_of_classes;

		// what percentage is this.

		start_percent_used = (float(wanted_start_number_of_asym_units) / float(number_of_asym_units)) * 100.0;
		end_percent_used = (float(wanted_end_number_of_asym_units) / float(number_of_asym_units)) * 100.0;

		if (start_percent_used > 100.0) start_percent_used = 100.0;
		if (end_percent_used > 100.0) end_percent_used = 100.0;

		if (end_percent_used > 25)
		{
			if (number_of_classes > 1) my_parent->WriteWarningText(wxString::Format("Warning : Using max %.2f %% of the images per round, this is quite high, you may wish to increase the number of particles or reduce the number of classes", end_percent_used));
			else my_parent->WriteWarningText(wxString::Format("Warning : Using max %.2f %% of the images per round, this is quite high, you may wish to increase the number of particles", end_percent_used));
		}
	}
	else
	{
		start_percent_used = my_parent->StartPercentUsedTextCtrl->ReturnValue();
		end_percent_used = my_parent->EndPercentUsedTextCtrl->ReturnValue();
	}

	current_percent_used = start_percent_used;

/*
	startup_percent_used = (float(wanted_number_of_asym_units) / float(number_of_asym_units)) * 100.0;
	wxPrintf("percent used = %.2f\n", startup_percent_used);
	if (startup_percent_used > 100) startup_percent_used = 100;

	if (startup_percent_used > 20 && startup_percent_used < 30) my_parent->WriteWarningText(wxString::Format("Warning : Using %.2f %% of the images per round, this is quite high, you may wish to increase the number of particles or reduce the number of classes", startup_percent_used));
	else
	if (startup_percent_used > 30) my_parent->WriteWarningText(wxString::Format("Warning : Using %.2f %% of the images per round, this is very high, you may wish to increase the number of particles or reduce the number of classes", startup_percent_used));
*/
	current_high_res_limit = my_parent->InitialResolutionLimitTextCtrl->ReturnValue();
	next_high_res_limit = current_high_res_limit;



	wxString blank_string = "";
	current_reference_filenames.Clear();
	current_reference_filenames.Add(blank_string, number_of_classes);

	// empty scratch
	if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath() + "/Startup/") == true) wxFileName::Rmdir(main_frame->current_project.scratch_directory.GetFullPath() + "/Startup", wxPATH_RMDIR_RECURSIVE);
	if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath() + "/Startup/") == false) wxFileName::Mkdir(main_frame->current_project.scratch_directory.GetFullPath() + "/Startup");

	my_parent->InputParamsPanel->Show(false);
	my_parent->StartPanel->Show(false);
	my_parent->ProgressPanel->Show(true);

	my_parent->ExpertPanel->Show(false);
	my_parent->InfoPanel->Show(false);
	my_parent->OutputTextPanel->Show(true);

	my_parent->ExpertToggleButton->Enable(false);


	SetupReconstructionJob();
	RunReconstructionJob();
}

void AbInitioManager::UpdatePlotPanel()
{
	long particle_counter;
	int class_counter;
	double number_active = 0;
	double average_likelihood = 0;
	double average_sigma = 0;

	for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
	{
		for (particle_counter = 0; particle_counter < output_refinement->number_of_particles; particle_counter++)
		{
			if (output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_is_active >= 0)
			{
				number_active += output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy * 0.01;
				average_likelihood += output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp * output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy * 0.01;
				average_sigma += output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].sigma * output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy * 0.01;
			}
		}
	}


	average_likelihood /= number_active;
	average_sigma /= number_active;
	//wxPrintf("\nLogp = %f, sigma = %f, number_active = %li\n\n", average_likelihood, average_sigma, number_active);

	my_parent->PlotPanel->AddPoints((number_of_rounds_to_run * number_of_starts_run) + number_of_rounds_run, average_sigma);
	my_parent->PlotPanel->Draw();

	if (number_of_rounds_run == 1)
	{
		my_parent->PlotPanel->Show(true);
		my_parent->Layout();
	}
}


void AbInitioManager::CycleRefinement()
{
	if (start_with_reconstruction == true)
	{
		output_refinement = new Refinement;
		output_refinement->refinement_id = 0;
		output_refinement->number_of_classes = input_refinement->number_of_classes;
		start_with_reconstruction = false;

		if (my_parent->AutoMaskYesRadio->GetValue() == true)
		{
			DoMasking();
		}
		else
		{
			SetupRefinementJob();
			RunRefinementJob();
		}
	}
	else
	{
		number_of_rounds_run++;

		if (number_of_rounds_run < number_of_rounds_to_run)
		{
			UpdatePlotPanel();

			delete input_refinement;
			input_refinement = output_refinement;
			output_refinement = new Refinement;
			output_refinement->refinement_id = (number_of_rounds_to_run * number_of_starts_run) + number_of_rounds_run;
			output_refinement->number_of_classes = input_refinement->number_of_classes;

			// we need to update the resolution..

			float start_res = my_parent->InitialResolutionLimitTextCtrl->ReturnValue();
			float end_res = my_parent->FinalResolutionLimitTextCtrl->ReturnValue();

			//current_high_res_limit = start_res + (end_res - start_res) * sqrtf(float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
			//next_high_res_limit = start_res + (end_res - start_res) * sqrtf(float(number_of_rounds_run + 1) / float(number_of_rounds_to_run - 1));

			current_high_res_limit = start_res + (end_res - start_res) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
			next_high_res_limit = start_res + (end_res - start_res) * (float(number_of_rounds_run + 1) / float(number_of_rounds_to_run - 1));


			if (next_high_res_limit < 0.0) next_high_res_limit = 0.0;

			//current_percent_used = start_percent_used + (end_percent_used - start_percent_used) * sqrtf(float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
			current_percent_used = start_percent_used + (end_percent_used - start_percent_used) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));


			if (my_parent->AutoMaskYesRadio->GetValue() == true)
			{
				DoMasking();
			}
			else
			{
				SetupRefinementJob();
				RunRefinementJob();
			}
		}
		else
		{
			number_of_starts_run++;

			if (my_parent->AutoPercentUsedYesRadio->GetValue() == true && number_of_starts_run == 1)
			{
				start_percent_used *=10;
				if (start_percent_used > end_percent_used) start_percent_used = end_percent_used;
			}

			if (number_of_starts_run < number_of_starts_to_run)
			{
				number_of_rounds_run = 0;
				UpdatePlotPanel();

				delete input_refinement;
				input_refinement = output_refinement;
				output_refinement = new Refinement;
				output_refinement->refinement_id = (number_of_rounds_to_run * number_of_starts_run) + number_of_rounds_run;
				output_refinement->number_of_classes = input_refinement->number_of_classes;

				float start_res = my_parent->InitialResolutionLimitTextCtrl->ReturnValue();
				float end_res = my_parent->FinalResolutionLimitTextCtrl->ReturnValue();

				//current_high_res_limit = start_res + (end_res - start_res) * sqrtf(float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
				//next_high_res_limit = start_res + (end_res - start_res) * sqrtf(float(number_of_rounds_run + 1) / float(number_of_rounds_to_run - 1));

				current_high_res_limit = start_res + (end_res - start_res) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
				next_high_res_limit = start_res + (end_res - start_res) * (float(number_of_rounds_run + 1) / float(number_of_rounds_to_run - 1));

				if (next_high_res_limit < 0.0) next_high_res_limit = 0.0;

				//current_percent_used = start_percent_used + (end_percent_used - start_percent_used) * sqrtf(float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
				current_percent_used = start_percent_used + (end_percent_used - start_percent_used) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));

				if (my_parent->AutoMaskYesRadio->GetValue() == true)
				{
					DoMasking();
				}
				else
				{
					SetupRefinementJob();
					RunRefinementJob();
				}


			}
			else
			{
				UpdatePlotPanel();
				number_of_starts_run--;
				my_parent->TakeCurrent();

				delete input_refinement;
				delete output_refinement;

				// write to the database..


				my_parent->WriteBlueText("All refinement cycles are finished!");
				my_parent->CancelAlignmentButton->Show(false);
				my_parent->CurrentLineOne->Show(false);
				my_parent->CurrentLineTwo->Show(false);
				my_parent->TakeCurrentResultButton->Show(false);
				my_parent->TakeLastStartResultButton->Show(false);
				my_parent->FinishButton->Show(true);
				my_parent->TimeRemainingText->SetLabel("Time Remaining : Finished!");
				my_parent->ProgressBar->SetValue(100);
				my_parent->ProgressPanel->Layout();

							// empty scratch
				//			if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath() + "/Startup/") == true) wxFileName::Rmdir(main_frame->current_project.scratch_directory.GetFullPath() + "/Startup", wxPATH_RMDIR_RECURSIVE);
					//		if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath() + "/Startup/") == false) wxFileName::Mkdir(main_frame->current_project.scratch_directory.GetFullPath() + "/Startup");


			}

		}
	}
}

void AbInitioManager::SetupReconstructionJob()
{
	wxArrayString written_parameter_files;

	if (start_with_reconstruction == true) written_parameter_files = output_refinement->WriteFrealignParameterFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/output_par", current_percent_used / 100.0, 10.0);
	else
	{
		//if (current_high_res_limit > 20) 	written_parameter_files = output_refinement->WriteFrealignParameterFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/output_par", 1.0, 1.0);
		//else 	written_parameter_files = output_refinement->WriteFrealignParameterFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/output_par", 1.0);
		written_parameter_files = output_refinement->WriteFrealignParameterFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/output_par", 1.0, 1.0);
	}


	int class_counter;
	long counter;
	int job_counter;
	long number_of_reconstruction_jobs;
	long number_of_reconstruction_processes;
	float current_particle_counter;

	long number_of_particles;
	float particles_per_job;

	// for now, number of jobs is number of processes -1 (master)..

	number_of_reconstruction_processes = run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()].ReturnTotalJobs();
	number_of_reconstruction_jobs = number_of_reconstruction_processes - 1;

	number_of_particles = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles.GetCount();

	if (number_of_particles - number_of_reconstruction_jobs < number_of_reconstruction_jobs) particles_per_job = 1;
	particles_per_job = float(number_of_particles - number_of_reconstruction_jobs) / float(number_of_reconstruction_jobs);

	my_parent->my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()], "reconstruct3d", number_of_reconstruction_jobs * refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes);

	for (class_counter = 0; class_counter < refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes; class_counter++)
	{
		current_particle_counter = 1.0;

		for (job_counter = 0; job_counter < number_of_reconstruction_jobs; job_counter++)
		{
			wxString input_particle_stack 		= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).stack_filename;
			wxString input_parameter_file 		= written_parameter_files[class_counter];
			wxString output_reconstruction_1    = "/dev/null";
			wxString output_reconstruction_2			= "/dev/null";
			wxString output_reconstruction_filtered		= "/dev/null";
			wxString output_resolution_statistics		= "/dev/null";
			wxString my_symmetry						= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).symmetry;

			long	 first_particle						= myroundint(current_particle_counter);

			current_particle_counter += particles_per_job;
			if (current_particle_counter > number_of_particles) current_particle_counter = number_of_particles;

			long	 last_particle						= myroundint(current_particle_counter);
			current_particle_counter+=1.0;

			float 	 pixel_size							= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].pixel_size;
			float    voltage_kV							= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].microscope_voltage;
			float 	 spherical_aberration_mm			= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].spherical_aberration;
			float    amplitude_contrast					= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].amplitude_contrast;
			float 	 molecular_mass_kDa					= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_weight_in_kda;
			float    inner_mask_radius					= 0;//my_parent->ReconstructionInnerRadiusTextCtrl->ReturnValue();
			float    outer_mask_radius					= input_refinement->resolution_statistics_box_size * 0.45 * input_refinement->resolution_statistics_pixel_size;

			float    resolution_limit_rec;

			resolution_limit_rec					= next_high_res_limit;//current_high_res_limit;//current_my_parent->ReconstructionResolutionLimitTextCtrl->ReturnValue();
			//if (number_of_rounds_run == number_of_rounds_to_run - 1) resolution_limit_rec = 0.0;
			//else resolution_limit_rec					= next_high_res_limit;//current_high_res_limit;//current_my_parent->ReconstructionResolutionLimitTextCtrl->ReturnValue();

			float    score_weight_conversion			= 0;//my_parent->ScoreToWeightConstantTextCtrl->ReturnValue();
			float    score_threshold					= 0;//my_parent->ReconstructioScoreThreshold->ReturnValue();
			bool	 adjust_scores						= true;//my_parent->AdjustScoreForDefocusYesRadio->GetValue();
			bool	 invert_contrast					= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).stack_has_white_protein;
			bool	 crop_images						= false;//my_parent->AutoCropYesRadioButton->GetValue();
			bool	 dump_arrays						= true;
			wxString dump_file_1 						= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup_dump_file_%i_odd_%i.dump", class_counter, job_counter +1);
			wxString dump_file_2 						= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup_dump_file_%i_even_%i.dump", class_counter, job_counter + 1);

			wxString input_reconstruction;
			bool	 use_input_reconstruction;


			if (my_parent->ApplyBlurringYesRadioButton->GetValue() == true)
			{
				// do we have a reference..

				if (number_of_rounds_run < 1)
				{
					input_reconstruction			= "/dev/null";
					use_input_reconstruction		= false;
				}
				else
				{
					input_reconstruction = current_reference_filenames.Item(class_counter);
					use_input_reconstruction = true;
				}


			}
			else
			{
				input_reconstruction			= "/dev/null";
				use_input_reconstruction		= false;
			}

			float    resolution_limit_ref               = current_high_res_limit;//my_parent->HighResolutionLimitTextCtrl->ReturnValue();
			float	 smoothing_factor					= my_parent->SmoothingFactorTextCtrl->ReturnValue();
			float    padding							= 1.0f;
			bool	 normalize_particles				= true;
			bool	 exclude_blank_edges				= false;
			bool	 split_even_odd						= false;
			bool     centre_mass                        = true;

			bool threshold_input_3d = false;

			my_parent->my_job_package.AddJob("ttttttttiifffffffffffffbbbbbbbbbbtt",
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
																		pixel_size,
																		voltage_kV,
																		spherical_aberration_mm,
																		amplitude_contrast,
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
																		dump_file_2.ToUTF8().data());
		}

	}
}


// for now we take the paramter

void AbInitioManager::RunReconstructionJob()
{
	running_job_type = RECONSTRUCTION;
	number_of_received_particle_results = 0;

	// in the future store the reconstruction parameters..

	// empty scratch directory..

//	if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath()) == true) wxFileName::Rmdir(main_frame->current_project.scratch_directory.GetFullPath(), wxPATH_RMDIR_RECURSIVE);
//	if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath()) == false) wxFileName::Mkdir(main_frame->current_project.scratch_directory.GetFullPath());

	// launch a controller

	if (start_with_reconstruction == true)
	{
		if (output_refinement->number_of_classes > 1) my_parent->WriteBlueText("Calculating Initial Reconstructions...");
		else my_parent->WriteBlueText("Calculating Initial Reconstruction...");

	}
	else
	{
		if (output_refinement->number_of_classes > 1) my_parent->WriteBlueText("Calculating Reconstructions...");
		else my_parent->WriteBlueText("Calculating Reconstruction...");

	}

	current_job_id = main_frame->job_controller.AddJob(my_parent, run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()].gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes;
	    if (my_parent->my_job_package.number_of_jobs + 1 < my_parent->my_job_package.my_profile.ReturnTotalJobs()) number_of_refinement_processes = my_parent->my_job_package.number_of_jobs + 1;
	    else number_of_refinement_processes =  my_parent->my_job_package.my_profile.ReturnTotalJobs();

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

		my_parent->NumberConnectedText->SetLabel(wxString::Format("%i / %li processes connected.", 0, number_of_refinement_processes));

		my_parent->TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
		my_parent->Layout();
		my_parent->running_job = true;
		my_parent->my_job_tracker.StartTracking(my_parent->my_job_package.number_of_jobs);

	}
		my_parent->ProgressBar->Pulse();
}

void AbInitioManager::SetupMerge3dJob()
{
	int class_counter;

	my_parent->my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()], "merge3d", refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes);

	for (class_counter = 0; class_counter < refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes; class_counter++)
	{
		wxString output_reconstruction_1			= "/dev/null";
		wxString output_reconstruction_2			= "/dev/null";

		wxString output_reconstruction_filtered;
		int output_number = (number_of_rounds_to_run * number_of_starts_run) + number_of_rounds_run;
		if (start_with_reconstruction == true) output_reconstruction_filtered = main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup3d_initial_%i_%i.mrc", output_number, class_counter);
		else output_reconstruction_filtered		= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup3d_%i_%i.mrc", output_number, class_counter);

		current_reference_filenames.Item(class_counter) = output_reconstruction_filtered;

		wxString output_resolution_statistics		= "/dev/null";
		float 	 molecular_mass_kDa					= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_weight_in_kda;
		float    inner_mask_radius					= 0;
		float    outer_mask_radius					= input_refinement->resolution_statistics_box_size * 0.45 * input_refinement->resolution_statistics_pixel_size;
		wxString dump_file_seed_1 					= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup_dump_file_%i_odd_.dump",  class_counter);
		wxString dump_file_seed_2 					= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup_dump_file_%i_even_.dump", class_counter);

		bool save_orthogonal_views_image = false;
		wxString orthogonal_views_filename = "";

		my_parent->my_job_package.AddJob("ttttfffttibt",	output_reconstruction_1.ToUTF8().data(),
														output_reconstruction_2.ToUTF8().data(),
														output_reconstruction_filtered.ToUTF8().data(),
														output_resolution_statistics.ToUTF8().data(),
														molecular_mass_kDa, inner_mask_radius, outer_mask_radius,
														dump_file_seed_1.ToUTF8().data(),
														dump_file_seed_2.ToUTF8().data(),
														class_counter + 1,
														save_orthogonal_views_image,
														orthogonal_views_filename.ToUTF8().data());
	}
}



void AbInitioManager::RunMerge3dJob()
{
	running_job_type = MERGE;

	// start job..

	if (output_refinement->number_of_classes > 1) my_parent->WriteBlueText("Merging and Filtering Reconstructions...");
	else
	my_parent->WriteBlueText("Merging and Filtering Reconstruction...");

	current_job_id = main_frame->job_controller.AddJob(my_parent, run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()].gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes;
	    if (my_parent->my_job_package.number_of_jobs + 1 < my_parent->my_job_package.my_profile.ReturnTotalJobs()) number_of_refinement_processes = my_parent->my_job_package.number_of_jobs + 1;
	    else number_of_refinement_processes =  my_parent->my_job_package.my_profile.ReturnTotalJobs();

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

		my_parent->TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
		my_parent->Layout();
		my_parent->running_job = true;
		my_parent->my_job_tracker.StartTracking(my_parent->my_job_package.number_of_jobs);

		}

		my_parent->ProgressBar->Pulse();
}

void AbInitioManager::SetupRefinementJob()
{
	int class_counter;
	long counter;
	long number_of_refinement_jobs;
	int number_of_refinement_processes;
	float current_particle_counter;

	long number_of_particles;
	float particles_per_job;

	// get the last refinement for the currently selected refinement package..


	wxArrayString written_parameter_files;
	wxArrayString written_res_files;

	// re-randomise the input parameters so that old results become meaningless..

	for (class_counter = 0; class_counter < input_refinement->number_of_classes; class_counter++)
	{
		for ( counter = 0; counter < number_of_particles; counter++)
		{
			if (input_refinement->number_of_classes == 1) input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = 100.0;
			else input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = 100.00 / input_refinement->number_of_classes;

			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].phi = global_random_number_generator.GetUniformRandom() * 180.0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].theta = global_random_number_generator.GetUniformRandom() * 180.0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].psi = global_random_number_generator.GetUniformRandom() * 180.0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].xshift = 0.0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].yshift = 0.0;

			//input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].image_is_active = 1;
			//input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].sigma = 1.0;
		}

		input_refinement->class_refinement_results[class_counter].class_resolution_statistics.GenerateDefaultStatistics(refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_weight_in_kda);
	}


	//wxPrintf("refinement_id = %li\n", input_refinement->refinement_id);
	written_parameter_files = input_refinement->WriteFrealignParameterFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/startup_input_par");
	written_res_files = input_refinement->WriteResolutionStatistics(main_frame->current_project.parameter_file_directory.GetFullPath() + "/startup_input_stats");

//	wxPrintf("Input refinement has %li particles\n", input_refinement->number_of_particles);

	// for now, number of jobs is number of processes -1 (master)..

	number_of_refinement_processes = run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].ReturnTotalJobs();
	number_of_refinement_jobs = number_of_refinement_processes - 1;

	number_of_particles = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles.GetCount();
	if (number_of_particles - number_of_refinement_jobs < number_of_refinement_jobs) particles_per_job = 1;
	else particles_per_job = float(number_of_particles - number_of_refinement_jobs) / float(number_of_refinement_jobs);

	my_parent->my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()], "refine3d", number_of_refinement_jobs * refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes);

	for (class_counter = 0; class_counter < refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes; class_counter++)
	{
		current_particle_counter = 1;

		for (counter = 0; counter < number_of_refinement_jobs; counter++)
		{

			wxString input_particle_images					= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).stack_filename;
			wxString input_parameter_file 					= written_parameter_files.Item(class_counter);
			wxString input_reconstruction					= current_reference_filenames.Item(class_counter);
			wxString input_reconstruction_statistics 		= written_res_files.Item(class_counter);
			bool	 use_statistics							= true;

			wxString ouput_matching_projections		 		= "";
			//wxString output_parameter_file					= "/tmp/output_par.par";
			//wxString ouput_shift_file						= "/tmp/output_shift.shft";
			wxString ouput_shift_file						= "/dev/null";

			wxString my_symmetry							= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).symmetry;
			long	 first_particle							= myroundint(current_particle_counter);

			current_particle_counter += particles_per_job;
			if (current_particle_counter > number_of_particles) current_particle_counter = number_of_particles;

			long	 last_particle							= myroundint(current_particle_counter);
			current_particle_counter++;

			float	 percent_used							= current_percent_used / 100.0;

#ifdef DEBUG
			wxString output_parameter_file = wxString::Format("/tmp/output_par_%li_%li.par", first_particle, last_particle);
#else
			wxString output_parameter_file = "/dev/null";
#endif

			// for now we take the paramters of the first image!!!!

			float 	 pixel_size								= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].pixel_size;
			float    voltage_kV								= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].microscope_voltage;
			float 	 spherical_aberration_mm				= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].spherical_aberration;
			float    amplitude_contrast						= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].amplitude_contrast;
			float	 molecular_mass_kDa						= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_weight_in_kda;

//			float    mask_radius							= input_refinement->resolution_statistics_box_size * 0.45 * input_refinement->resolution_statistics_pixel_size;//my_parent->MaskRadiusTextCtrl->ReturnValue();
			float    mask_radius							= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.6;
			float    inner_mask_radius						= 0;

			float low_resolution_limit = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms;
			//if (low_resolution_limit > 100.00) low_resolution_limit = 100.00;

			float    high_resolution_limit					= current_high_res_limit;
			float	 signed_CC_limit						= 0.0;
			float	 classification_resolution_limit		= current_high_res_limit;
//			float    mask_radius_search						= input_refinement->resolution_statistics_box_size * 0.45 * input_refinement->resolution_statistics_pixel_size;
			float    mask_radius_search						= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.6;
			float	 high_resolution_limit_search			= current_high_res_limit;
			float	 angular_step							= CalculateAngularStep(current_high_res_limit, refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.5);

		//	if (angular_step < 30.0) angular_step = 30.0;

			int		 best_parameters_to_keep				= 20;
			float	 max_search_x							= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.5;
			float	 max_search_y							= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.5;
			float    mask_center_2d_x						= 0.0;
			float 	 mask_center_2d_y						= 0.0;
			float    mask_center_2d_z						= 0.0;
			float    mask_radius_2d							= 0.0;

			float	 defocus_search_range					= 0.0;
			float	 defocus_step							= 0.0;
			float	 padding								= 1.0;

			bool global_search = true;
			bool local_refinement = false;
			bool local_global_refine = false;

			bool refine_psi 								= true;
			bool refine_theta								= true;
			bool refine_phi									= true;
			bool refine_x_shift								= true;
			bool refine_y_shift								= true;
			bool calculate_matching_projections				= false;
			bool apply_2d_masking							= false;
			bool ctf_refinement								= false;
			bool invert_contrast							= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).stack_has_white_protein;

			bool normalize_particles = true;
			bool exclude_blank_edges = false;
			bool normalize_input_3d;

			if (my_parent->ApplyBlurringYesRadioButton->GetValue() == true) normalize_input_3d = false;
			else normalize_input_3d = true;

			bool threshold_input_3d = false;
			bool ignore_input_parameters = false;
			my_parent->my_job_package.AddJob("ttttbttttiifffffffffffffffifffffffffbbbbbbbbbbbbbbbbib",
																											input_particle_images.ToUTF8().data(),
																											input_parameter_file.ToUTF8().data(),
																											input_reconstruction.ToUTF8().data(),
																											input_reconstruction_statistics.ToUTF8().data(),
																											use_statistics,
																											ouput_matching_projections.ToUTF8().data(),
																											output_parameter_file.ToUTF8().data(),
																											ouput_shift_file.ToUTF8().data(),
																											my_symmetry.ToUTF8().data(),
																											first_particle,
																											last_particle,
																											percent_used,
																											pixel_size,
																											voltage_kV,
																											spherical_aberration_mm,
																											amplitude_contrast,
																											molecular_mass_kDa,
																											inner_mask_radius,
																											mask_radius,
																											low_resolution_limit,
																											high_resolution_limit,
																											signed_CC_limit,
																											classification_resolution_limit,
																											mask_radius_search,
																											high_resolution_limit_search,
																											angular_step,
																											best_parameters_to_keep,
																											max_search_x,
																											max_search_y,
																											mask_center_2d_x,
																											mask_center_2d_y,
																											mask_center_2d_z,
																											mask_radius_2d,
																											defocus_search_range,
																											defocus_step,
																											padding,
																											global_search,
																											local_refinement,
																											refine_psi,
																											refine_theta,
																											refine_phi,
																											refine_x_shift,
																											refine_y_shift,
																											calculate_matching_projections,
																											apply_2d_masking,
																											ctf_refinement,
																											normalize_particles,
																											invert_contrast,
																											exclude_blank_edges,
																											normalize_input_3d,
																											threshold_input_3d,
																											local_global_refine,
																											class_counter,
																											ignore_input_parameters);
		}

	}
}

void AbInitioManager::RunRefinementJob()
{
	running_job_type = REFINEMENT;
	number_of_received_particle_results = 0;

	output_refinement->SizeAndFillWithEmpty(input_refinement->number_of_particles, input_refinement->number_of_classes);
	output_refinement->refinement_package_asset_id = current_refinement_package_asset_id;

	output_refinement->resolution_statistics_are_generated = true;
	output_refinement->datetime_of_run = wxDateTime::Now();

/*	output_refinement->low_resolution_limit = my_parent->LowResolutionLimitTextCtrl->ReturnValue();
	output_refinement->high_resolution_limit = my_parent->HighResolutionLimitTextCtrl->ReturnValue();
	output_refinement->mask_radius = my_parent->MaskRadiusTextCtrl->ReturnValue();
	output_refinement->signed_cc_resolution_limit = my_parent->SignedCCResolutionTextCtrl->ReturnValue();
	output_refinement->global_resolution_limit = my_parent->HighResolutionLimitTextCtrl->ReturnValue();
	output_refinement->global_mask_radius = my_parent->GlobalMaskRadiusTextCtrl->ReturnValue();
	output_refinement->number_results_to_refine = my_parent->NumberToRefineSpinCtrl->GetValue();
	output_refinement->angular_search_step = my_parent->AngularStepTextCtrl->ReturnValue();
	output_refinement->search_range_x = my_parent->SearchRangeXTextCtrl->ReturnValue();
	output_refinement->search_range_y = my_parent->SearchRangeYTextCtrl->ReturnValue();
	output_refinement->classification_resolution_limit = my_parent->ClassificationHighResLimitTextCtrl->ReturnValue();
	output_refinement->should_focus_classify = my_parent->SphereClassificatonYesRadio->GetValue();
	output_refinement->sphere_x_coord = my_parent->SphereXTextCtrl->ReturnValue();
	output_refinement->sphere_y_coord = my_parent->SphereYTextCtrl->ReturnValue();
	output_refinement->sphere_z_coord = my_parent->SphereZTextCtrl->ReturnValue();
	output_refinement->should_refine_ctf = my_parent->RefineCTFYesRadio->GetValue();
	output_refinement->defocus_search_range = my_parent->DefocusSearchRangeTextCtrl->ReturnValue();
	output_refinement->defocus_search_step = my_parent->DefocusSearchStepTextCtrl->ReturnValue();

	output_refinement->percent_used = my_parent->PercentUsedTextCtrl->ReturnValue();
*/
	output_refinement->resolution_statistics_box_size = input_refinement->resolution_statistics_box_size;
	output_refinement->resolution_statistics_pixel_size = input_refinement->resolution_statistics_pixel_size;

	// launch a controller

	current_job_starttime = time(NULL);
	time_of_last_update = current_job_starttime;
	//my_parent->AngularPlotPanel->Clear();

	my_parent->WriteBlueText(wxString::Format(wxT("Running refinement round %2i of %2i (%.2f  Å / %.2f %%) - Start %2i of %2i \n"), number_of_rounds_run + 1, number_of_rounds_to_run, current_high_res_limit, current_percent_used, number_of_starts_run + 1, number_of_starts_to_run));
	current_job_id = main_frame->job_controller.AddJob(my_parent, run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes;
	    if (my_parent->my_job_package.number_of_jobs + 1 < my_parent->my_job_package.my_profile.ReturnTotalJobs()) number_of_refinement_processes = my_parent->my_job_package.number_of_jobs + 1;
	    else number_of_refinement_processes =  my_parent->my_job_package.my_profile.ReturnTotalJobs();

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
		my_parent->TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
		my_parent->Layout();
		my_parent->running_job = true;
		my_parent->my_job_tracker.StartTracking(my_parent->my_job_package.number_of_jobs);

	}




	my_parent->ProgressBar->Pulse();
}



void AbInitioManager::ProcessJobResult(JobResult *result_to_process)
{
	if (running_job_type == REFINEMENT)
	{

		int current_class = int(result_to_process->result_data[0] + 0.5);
		long current_particle = long(result_to_process->result_data[1] + 0.5) - 1;

		MyDebugAssertTrue(current_particle != -1 && current_class != -1, "Current Particle (%li) or Current Class(%i) = -1!", current_particle, current_class);

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
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].phase_shift = result_to_process->result_data[12];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].occupancy = result_to_process->result_data[13];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].logp = result_to_process->result_data[14];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].sigma = result_to_process->result_data[15];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].score = result_to_process->result_data[16];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].image_is_active = int(result_to_process->result_data[8]);

		number_of_received_particle_results++;
		//wxPrintf("received result!\n");
		long current_time = time(NULL);

		if (number_of_received_particle_results == 1)
		{
			current_job_starttime = current_time;
			time_of_last_update = 0;
//			my_parent->AngularPlotPanel->SetSymmetryAndNumber(refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).symmetry,output_refinement->number_of_particles);
	//		my_parent->AngularPlotPanel->Show(true);
		//	my_parent->FSCResultsPanel->Show(false);
			my_parent->Layout();
		}
		else
		if (current_time != time_of_last_update)
		{
			int current_percentage = float(number_of_received_particle_results) / float(output_refinement->number_of_particles * output_refinement->number_of_classes) * 100.0;
			time_of_last_update = current_time;
			if (current_percentage > 100) current_percentage = 100;
			my_parent->ProgressBar->SetValue(current_percentage);
			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float((input_refinement->number_of_particles * output_refinement->number_of_classes) - number_of_received_particle_results) * seconds_per_job;

			TimeRemaining time_remaining;

			if (seconds_remaining > 3600) time_remaining.hours = seconds_remaining / 3600;
			else time_remaining.hours = 0;

			if (seconds_remaining > 60) time_remaining.minutes = (seconds_remaining / 60) - (time_remaining.hours * 60);
			else time_remaining.minutes = 0;

			time_remaining.seconds = seconds_remaining - ((time_remaining.hours * 60 + time_remaining.minutes) * 60);
			my_parent->TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_remaining.hours, time_remaining.minutes, time_remaining.seconds));
		}


	        // Add this result to the list of results to be plotted onto the angular plot
		/*
		if (current_class == 0)
		{
		//		my_parent->AngularPlotPanel->AddRefinementResult( &output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle]);
		         // Plot this new result onto the angular plot immediately if it's one of the first few results to come in. Otherwise, only plot at regular intervals.

		        if(my_parent->AngularPlotPanel->refinement_results_to_plot.Count() * my_parent->AngularPlotPanel->symmetry_matrices.number_of_matrices < 1500 || current_time - my_parent->time_of_last_result_update > 0)
		        {

		            my_parent->AngularPlotPanel->Refresh();
		            my_parent->time_of_last_result_update = current_time;
		        }

			}
		}*/
	}
	else
	if (running_job_type == RECONSTRUCTION)
	{
		number_of_received_particle_results++;

		long current_time = time(NULL);

		if (number_of_received_particle_results == 1)
		{
			time_of_last_update = 0;
			current_job_starttime = current_time;
		}
		else
		if (current_time - time_of_last_update >= 1)
		{
			time_of_last_update = current_time;
			int current_percentage = float(number_of_received_particle_results) / float(output_refinement->number_of_particles * output_refinement->number_of_classes) * 100.0;
			if (current_percentage > 100) current_percentage = 100;
			my_parent->ProgressBar->SetValue(current_percentage);
			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float((input_refinement->number_of_particles * input_refinement->number_of_classes) - number_of_received_particle_results) * seconds_per_job;

			TimeRemaining time_remaining;
			if (seconds_remaining > 3600) time_remaining.hours = seconds_remaining / 3600;
			else time_remaining.hours = 0;

			if (seconds_remaining > 60) time_remaining.minutes = (seconds_remaining / 60) - (time_remaining.hours * 60);
			else time_remaining.minutes = 0;

			time_remaining.seconds = seconds_remaining - ((time_remaining.hours * 60 + time_remaining.minutes) * 60);
			my_parent->TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_remaining.hours, time_remaining.minutes, time_remaining.seconds));
		}
	}
	else
	if (running_job_type == MERGE)
	{
		// add to the correct resolution statistics..

		int number_of_points = result_to_process->result_data[0];
		int class_number = int(result_to_process->result_data[1] + 0.5);
		int array_position = 2;
		float current_resolution;
		float fsc;
		float part_fsc;
		float part_ssnr;
		float rec_ssnr;

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


void AbInitioManager::ProcessAllJobsFinished()
{
	if (running_job_type == REFINEMENT)
	{

		main_frame->job_controller.KillJob(my_parent->my_job_id);

		// calculate occupancies..
		output_refinement->UpdateOccupancies();

		SetupReconstructionJob();
		RunReconstructionJob();

	}
	else
	if (running_job_type == RECONSTRUCTION)
	{
		main_frame->job_controller.KillJob(my_parent->my_job_id);
		SetupMerge3dJob();
		RunMerge3dJob();
	}
	else
	if (running_job_type == MERGE)
	{
		int class_counter;
		main_frame->job_controller.KillJob(my_parent->my_job_id);

		// prepare the orth projections..

		OrthDrawerThread *result_thread;

		if (start_with_reconstruction == true)	result_thread = new OrthDrawerThread(my_parent, current_reference_filenames, "Random Angles");
		else
		result_thread = new OrthDrawerThread(my_parent, current_reference_filenames, wxString::Format("Iter. #%i,%i", number_of_starts_run, number_of_rounds_run));

		if ( result_thread->Run() != wxTHREAD_NO_ERROR )
		{
			my_parent->WriteErrorText("Error: Cannot start result creation thread, results not displayed");
			delete result_thread;
		}

		wxArrayFloat average_occupancies = output_refinement->UpdatePSSNR();

		if (output_refinement->number_of_classes > 1)
		{
			my_parent->WriteInfoText("");

			for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
			{
				my_parent->WriteInfoText(wxString::Format(wxT("   Occupancy for Class %2i = %2.2f %%"), class_counter + 1, average_occupancies[class_counter]));
			}

			my_parent->WriteInfoText("");

		}

		CycleRefinement();
	}
}

void AbInitio3DPanel::OnOrthThreadComplete(MyOrthDrawEvent& my_event)
{
	// in theory the long data should contain a pointer to wxPanel that we are going to add to the notebook..

	Image *new_image = my_event.GetImage();

	if (new_image != NULL)
	{
		ShowOrthDisplayPanel->OpenImage(new_image, my_event.GetString(), true);

		if (OrthResultsPanel->IsShown() == false)
		{
			OrthResultsPanel->Show(true);
			Layout();
		}
	}
}


void AbInitio3DPanel::OnMaskerThreadComplete(wxThreadEvent& my_event)
{
	my_abinitio_manager.OnMaskerThreadComplete();
}


void AbInitioManager::OnMaskerThreadComplete()
{
	//my_parent->WriteInfoText("Masking Finished");
	SetupRefinementJob();
	RunRefinementJob();
}

wxThread::ExitCode MaskerThread::Entry()
{
	//  Read in the files, threshold them write them out again...

	Image input_image;
	Image mask_image;
	ImageFile input_file;
	MRCFile output_file;

	float average_value;
	float average_of_other_values;
	float average_value_on_edge;
	int i,j,k;
	long address;
	long number_of_pixels;

	for (int class_counter = 0; class_counter < input_files.GetCount(); class_counter++)
	{
		input_file.OpenFile(input_files.Item(class_counter).ToStdString(), false);
		input_image.ReadSlices(&input_file, 1, input_file.ReturnNumberOfSlices());
		input_file.CloseFile();
/*
		average_value = input_image.ReturnAverageOfRealValues();
		average_value_on_edge = input_image.ReturnAverageOfRealValuesOnEdges();
		address = 0;
		number_of_pixels = 0;
		average_of_other_values = 0.0f;

		for (k = 0; k < input_image.logical_z_dimension; k++)
		{
			for (j = 0; j < input_image.logical_y_dimension; j++)
			{
				for (i = 0; i < input_image.logical_x_dimension; i++)
				{
					if (input_image.real_values[address] < average_value)
					{
						average_of_other_values += input_image.real_values[address];
						number_of_pixels++;
					}

					address++;
				}

				address += input_image.padding_jump_value;

			}
		}

		if (number_of_pixels > 0)
		{
			average_of_other_values /= float(number_of_pixels);
		}

		address = 0;

		for (k = 0; k < input_image.logical_z_dimension; k++)
		{
			for (j = 0; j < input_image.logical_y_dimension; j++)
			{
				for (i = 0; i < input_image.logical_x_dimension; i++)
				{
					if (input_image.real_values[address] < average_value)
					{
						input_image.real_values[address] = average_of_other_values;
					}

					address++;
				}

				address += input_image.padding_jump_value;
			}
		}

		input_image.CosineMask(mask_radius / pixel_size, 1.0, false, true, average_value_on_edge);

*/

		input_image.SetMinimumValue(input_image.ReturnAverageOfRealValues());
		input_image.CosineMask(mask_radius / pixel_size, 1.0, false, true, 0.0);


		//input_image.CosineMask(mask_radius / pixel_size, 1.0);
		//input_image.AddConstant(-input_image.ReturnAverageOfRealValuesOnEdges());
		//input_image.SetMinimumValue(0);




/*
		mask_image.CopyFrom(&input_image);
		mask_image.Binarise(input_image.ReturnAverageOfRealValues());
		mask_image.CosineMask(mask_radius / pixel_size, 1.0f, false, true, 0.0f);

		input_image.ApplyMask(mask_image, 5.0f  / pixel_size, 1.0f, pixel_size / 50.0f, pixel_size / 5.0f);
*/

		output_file.OpenFile(output_files.Item(class_counter).ToStdString(), true);
		input_image.WriteSlices(&output_file, 1, input_image.logical_z_dimension);
		output_file.CloseFile();
	}


	// send finished event..

	wxThreadEvent *my_thread_event = new wxThreadEvent(wxEVT_COMMAND_MASKERTHREAD_COMPLETED);
	wxQueueEvent(main_thread_pointer, my_thread_event);

	return (wxThread::ExitCode)0;     // success
}


void AbInitioManager::DoMasking()
{
	// right now do nothing. Take out event if changing back to thread.
//	MyDebugAssertTrue(my_parent->AutoMaskYesRadio->GetValue() == true, "DoMasking called, when masking not ticked!");
//	wxThreadEvent *my_thread_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_COMPLETED);
//	wxQueueEvent(my_parent, my_thread_event);

	wxArrayString masked_filenames;
	wxFileName current_ref_filename;
	wxString current_masked_filename;

	for (int class_counter = 0; class_counter < current_reference_filenames.GetCount(); class_counter++)
	{
		current_ref_filename = current_reference_filenames.Item(class_counter);
		current_ref_filename.ClearExt();
		current_masked_filename = current_ref_filename.GetFullPath();
		current_masked_filename += "_masked.mrc";

		masked_filenames.Add(current_masked_filename);
	}

	MaskerThread *mask_thread = new MaskerThread(my_parent, current_reference_filenames, masked_filenames, input_refinement->resolution_statistics_pixel_size, 35.0f, input_refinement->resolution_statistics_box_size * 0.45 * input_refinement->resolution_statistics_pixel_size);

	if ( mask_thread->Run() != wxTHREAD_NO_ERROR )
	{
		my_parent->WriteErrorText("Error: Cannot start masking thread, masking will not be performed");
		delete mask_thread;
	}
	else
	{
		current_reference_filenames = masked_filenames;
		return; // just return, we will startup again whent he mask thread finishes.
	}
}


