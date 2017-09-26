#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyVolumeAssetPanel *volume_asset_panel;
extern MyRefinementResultsPanel *refinement_results_panel;

Generate3DPanel::Generate3DPanel( wxWindow* parent )
:
Generate3DPanelParent( parent )
{
//	buffered_results = NULL;
	my_job_id = -1;
	running_job = false;

	SetInfo();

	wxSize input_size = InputSizer->GetMinSize();
	input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
	input_size.y = -1;
	ExpertPanel->SetMinSize(input_size);
	ExpertPanel->SetSize(input_size);

	refinement_package_combo_is_dirty = false;
	run_profiles_are_dirty = false;
	input_params_combo_is_dirty = false;
	selected_refinement_package = -1;

	RefinementPackageComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &Generate3DPanel::OnRefinementPackageComboBox, this);
	Bind(RETURN_PROCESSED_IMAGE_EVT, &Generate3DPanel::OnOrthThreadComplete, this);

	if (ShowRefinementResultsPanel->TopBottomSplitter->IsSplit() == true) ShowRefinementResultsPanel->TopBottomSplitter->Unsplit(ShowRefinementResultsPanel->TopPanel);

	ShowRefinementResultsPanel->AngularPlotText->Show(false);
	ShowRefinementResultsPanel->AngularPlotLine->Show(false);

	ShowRefinementResultsPanel->AngularPlotPanel->Show(false);
	ShowRefinementResultsPanel->FSCResultsPanel->Show(true);

	FillRefinementPackagesComboBox();
}

void Generate3DPanel::SetInfo()
{

	wxLogNull *suppress_png_warnings = new wxLogNull;
//	#include "icons/niko_picture1.cpp"
//	wxBitmap niko_picture1_bmp = wxBITMAP_PNG_FROM_DATA(niko_picture1);

	//#include "icons/niko_picture2.cpp"
	//wxBitmap niko_picture2_bmp = wxBITMAP_PNG_FROM_DATA(niko_picture2);
	delete suppress_png_warnings;

	InfoText->GetCaret()->Hide();

	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("Generate 3D"));
	InfoText->EndFontSize();
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("This panel is used to (re-)generate 3D's from a given set of input parameters without running any refinement.  It could be used to generate a 3D from imported parameters for example. It can also be used to try different reconstruction parameters to see how they influence the final map.  Finally, it can also be used to save the reconstruction half maps."));
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
	InfoText->WriteText(wxT("Input Parameters : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The source of the starting parameters for this reconstruction run."));
	InfoText->Newline();

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
	InfoText->WriteText(wxT("Inner/Outer Mask Radius (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Radii describing a spherical mask with an inner and outer radius that will be applied to the final reconstruction and to the half reconstructions to calculate Fourier Shell Correlation curve. The inner radius is normally set to 0.0 but can assume non-zero values to remove density inside a particle if it represents largely disordered features, such as the genomic RNA or DNA of a virus."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Score to B-factor Constant (Å2) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The particles inserted into a reconstruction will be weighted according to their scores. The weighting function is akin to a B-factor, attenuating high-resolution signal of particles with lower scores more strongly than of particles with higher scores. The B-factor applied to each particle prior to insertion into the reconstruction is calculated as B = (score – average score) * constant * 0.25. Users are encouraged to calculate reconstructions with different values to find a value that produces the highest resolution. Values between 0 and 10 are reasonable (0 will disable weighting)."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Adjust Score for Defocus? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Scores sometimes depend on the amount of image defocus. A larger defocus amplifies low-resolution features in the image and this may lead to higher particle scores compared to particles from an image with a small defocus. Adjusting the scores for this difference makes sure that particles with smaller defocus are not systematically downweighted by the above B-factor weighting."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Score Threshold : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Particles with a score lower than the threshold will be excluded from the reconstruction. This provides a way to exclude particles that may score low because of misalignment or damage."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Resolution Limit (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The reconstruction calculation can be accelerated by limiting its resolution. It is important to make sure that the resolution limit entered here is higher than the resolution used for refinement in the following cycle."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Autocrop Images? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The reconstruction calculation can also be accelerated by cropping the boxes containing the particles. Cropping will slightly reduce the overall quality of the reconstruction due to increased aliasing effects and should not be used when finalizing refinement. However, during refinement, cropping can greatly increase the speed of reconstruction without noticeable impact on the refinement results."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Also Save Half-Maps? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If yes, the reconstruction half maps will also be saved as '_map1' and '_map2' in the volume assets folder."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Overwrite Statistics? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If yes, the resolution statistics (essentially the FSC) for the input refinement will be overwritten."));
	InfoText->Newline();

}

void Generate3DPanel::OnInfoURL(wxTextUrlEvent& event)
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


void Generate3DPanel::ResetAllDefaultsClick( wxCommandEvent& event )
{
	// TODO : should probably check that the user hasn't changed the defaults yet in the future
	SetDefaults();
}

void Generate3DPanel::SetDefaults()
{

	if (RefinementPackageComboBox->GetCount() > 0)
	{
		ExpertPanel->Freeze();

		float local_mask_radius = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.6;
		MaskRadiusTextCtrl->SetValue(wxString::Format("%.2f", local_mask_radius));
		InnerMaskRadiusTextCtrl->SetValue("0.00");

		ScoreToWeightConstantTextCtrl->SetValue("2.00");

		AdjustScoreForDefocusYesRadio->SetValue(true);
		AdjustScoreForDefocusNoRadio->SetValue(false);
		ReconstructionScoreThreshold->SetValue("0.00");
		ReconstructionResolutionLimitTextCtrl->SetValue("0.00");
		AutoCropYesRadioButton->SetValue(false);
		AutoCropNoRadioButton->SetValue(true);
		SaveHalfMapsNoButton->SetValue(true);
		OverwriteStatisticsYesButton->SetValue(true);

		ExpertPanel->Thaw();
	}
}

void Generate3DPanel::OnUpdateUI( wxUpdateUIEvent& event )
{

	// are there enough members in the selected group.
	if (main_frame->current_project.is_open == false)
	{
		RefinementPackageComboBox->Enable(false);
		InputParametersComboBox->Enable(false);
		ReconstructionRunProfileComboBox->Enable(false);
		ExpertToggleButton->Enable(false);

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

		if (PleaseCreateRefinementPackageText->IsShown())
		{
			PleaseCreateRefinementPackageText->Show(false);
			Layout();
		}
	}
	else
	{
		ReconstructionRunProfileComboBox->Enable(true);

		if (running_job == false)
		{
			ExpertToggleButton->Enable(true);

			if (RefinementPackageComboBox->GetCount() > 0)
			{
				RefinementPackageComboBox->Enable(true);
				InputParametersComboBox->Enable(true);

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
				InputParametersComboBox->ChangeValue("");
				InputParametersComboBox->Enable(false);

				if (PleaseCreateRefinementPackageText->IsShown() == false)
				{
					PleaseCreateRefinementPackageText->Show(true);
					Layout();
				}
			}

			bool estimation_button_status = false;

			if (RefinementPackageComboBox->GetCount() > 0 && ReconstructionRunProfileComboBox->GetCount() > 0)
			{
				if (run_profiles_panel->run_profile_manager.ReturnTotalJobs(ReconstructionRunProfileComboBox->GetSelection()) > 1)
				{
					if (RefinementPackageComboBox->GetSelection() != wxNOT_FOUND && InputParametersComboBox->GetSelection() != wxNOT_FOUND)
					{
						estimation_button_status = true;
					}

				}
			}

			StartReconstructionButton->Enable(estimation_button_status);

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
		else
		{
			RefinementPackageComboBox->Enable(false);
			InputParametersComboBox->Enable(false);
			ExpertToggleButton->Enable(false);

		}
	}
}


void Generate3DPanel::OnExpertOptionsToggle( wxCommandEvent& event )
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


void Generate3DPanel::FillRefinementPackagesComboBox()
{
	if (RefinementPackageComboBox->FillComboBox() == false) NewRefinementPackageSelected();
}

void Generate3DPanel::FillInputParamsComboBox()
{
	if (RefinementPackageComboBox->GetCount() > 0 ) InputParametersComboBox->FillComboBox(RefinementPackageComboBox->GetSelection(), true);
}

void Generate3DPanel::NewRefinementPackageSelected()
{
	selected_refinement_package = RefinementPackageComboBox->GetSelection();
	FillInputParamsComboBox();
	SetDefaults();
}

void Generate3DPanel::OnRefinementPackageComboBox( wxCommandEvent& event )
{

	NewRefinementPackageSelected();

}

void Generate3DPanel::OnInputParametersComboBox( wxCommandEvent& event )
{
	//SetDefaults();
}

void Generate3DPanel::TerminateButtonClick( wxCommandEvent& event )
{
	main_frame->job_controller.KillJob(my_job_id);

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


void Generate3DPanel::FinishButtonClick( wxCommandEvent& event )
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

//	if (my_refinement_manager.output_refinement != NULL) delete my_refinement_manager.output_refinement;

	if (ExpertToggleButton->GetValue() == true) ExpertPanel->Show(true);
	else ExpertPanel->Show(false);
	running_job = false;
	delete input_refinement;
	Layout();

	//CTFResultsPanel->CTF2DResultsPanel->should_show = false;
	//CTFResultsPanel->CTF2DResultsPanel->Refresh();

}




void Generate3DPanel::StartReconstructionClick( wxCommandEvent& event )
{
	active_mask_radius = MaskRadiusTextCtrl->ReturnValue();
	active_inner_mask_radius = InnerMaskRadiusTextCtrl->ReturnValue();
	active_resolution_limit_rec = ReconstructionResolutionLimitTextCtrl->ReturnValue();
	active_score_weight_conversion	= ScoreToWeightConstantTextCtrl->ReturnValue();
	active_score_threshold	= ReconstructionScoreThreshold->ReturnValue();
	active_adjust_scores = AdjustScoreForDefocusYesRadio->GetValue();
	active_crop_images	= AutoCropYesRadioButton->GetValue();
	active_save_half_maps = SaveHalfMapsYesButton->GetValue();
	active_update_statistics = OverwriteStatisticsYesButton->GetValue();

	active_refinement_package = &refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection());
	active_reconstruction_run_profile = run_profiles_panel->run_profile_manager.run_profiles[ReconstructionRunProfileComboBox->GetSelection()];
	long current_input_refinement_id = active_refinement_package->refinement_ids[InputParametersComboBox->GetSelection()];


	// check scratch directory.
	global_delete_refine3d_scratch();

	input_refinement = main_frame->current_project.database.GetRefinementByID(current_input_refinement_id);

	Freeze();
	StartPanel->Show(false);
	ProgressPanel->Show(true);
	ExpertPanel->Show(false);
	InfoPanel->Show(false);
	OutputTextPanel->Show(true);
	ShowRefinementResultsPanel->Clear();

	//if (ShowRefinementResultsPanel->LeftRightSplitter->IsSplit() == true) ShowRefinementResultsPanel->LeftRightSplitter->Unsplit();

	Layout();
	Thaw();

	SetupReconstructionJob();
	RunReconstructionJob();
}

void Generate3DPanel::WriteInfoText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void Generate3DPanel::WriteBlueText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLUE));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void Generate3DPanel::WriteErrorText(wxString text_to_write)
{
	 output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
	 output_textctrl->AppendText(text_to_write);

	 if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}


void Generate3DPanel::FillRunProfileComboBoxes()
{
	ReconstructionRunProfileComboBox->FillWithRunProfiles();
}

void Generate3DPanel::OnJobSocketEvent(wxSocketEvent& event)
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

			ProcessJobResult(&temp_result);
			wxPrintf("Warning: Received socket_job_result - should this happen?");

		}
		else
		if (memcmp(socket_input_buffer, socket_job_result_queue, SOCKET_CODE_SIZE) == 0) // identification
		{
			ArrayofJobResults temp_queue;
			ReceiveResultQueueFromSocket(sock, temp_queue);

			for (int counter = 0; counter < temp_queue.GetCount(); counter++)
			{
				ProcessJobResult(&temp_queue.Item(counter));
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
			// As soon as it sends us the message that all jobs are finished, the controller should also
			// send timing info - we need to remember this
			long timing_from_controller;
			ReadFromSocket(sock, &timing_from_controller, sizeof(long));
			main_frame->current_project.total_cpu_hours += timing_from_controller / 3600000.0;
			main_frame->current_project.total_jobs_run += my_job_tracker.total_number_of_jobs;

			// Other stuff to do once all jobs finished
			ProcessAllJobsFinished();
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

void Generate3DPanel::SetupReconstructionJob()
{
	wxArrayString written_parameter_files;
	written_parameter_files = input_refinement->WriteFrealignParameterFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/generate3d_par");

	int class_counter;
	long counter;
	int job_counter;
	long number_of_reconstruction_jobs;
	long number_of_reconstruction_processes;
	float current_particle_counter;

	long number_of_particles;
	float particles_per_job;

	// for now, number of jobs is number of processes -1 (master)..

	number_of_reconstruction_processes = active_reconstruction_run_profile.ReturnTotalJobs();
	number_of_reconstruction_jobs = number_of_reconstruction_processes - 1;

	number_of_particles = active_refinement_package->contained_particles.GetCount();

	if (number_of_particles - number_of_reconstruction_jobs < number_of_reconstruction_jobs) particles_per_job = 1;
	particles_per_job = float(number_of_particles - number_of_reconstruction_jobs) / float(number_of_reconstruction_jobs);

	my_job_package.Reset(active_reconstruction_run_profile, "reconstruct3d", number_of_reconstruction_jobs * active_refinement_package->number_of_classes);

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		current_particle_counter = 1.0;

		for (job_counter = 0; job_counter < number_of_reconstruction_jobs; job_counter++)
		{
			wxString input_particle_stack 		= active_refinement_package->stack_filename;
			wxString input_parameter_file 		= written_parameter_files[class_counter];
			wxString output_reconstruction_1    = "/dev/null";
			wxString output_reconstruction_2			= "/dev/null";
			wxString output_reconstruction_filtered		= "/dev/null";
			wxString output_resolution_statistics		= "/dev/null";
			wxString my_symmetry						= active_refinement_package->symmetry;

			long	 first_particle						= myroundint(current_particle_counter);

			current_particle_counter += particles_per_job;
			if (current_particle_counter > number_of_particles) current_particle_counter = number_of_particles;

			long	 last_particle						= myroundint(current_particle_counter);
			current_particle_counter+=1.0;

			float 	 pixel_size							= active_refinement_package->contained_particles[0].pixel_size;
			float    voltage_kV							= active_refinement_package->contained_particles[0].microscope_voltage;
			float 	 spherical_aberration_mm			= active_refinement_package->contained_particles[0].spherical_aberration;
			float    amplitude_contrast					= active_refinement_package->contained_particles[0].amplitude_contrast;
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
			wxString dump_file_1 						= main_frame->ReturnGenerate3DScratchDirectory() + wxString::Format("dump_file_%li_%i_odd_%i.dump", input_refinement->refinement_id, class_counter, job_counter +1);
			wxString dump_file_2 						= main_frame->ReturnGenerate3DScratchDirectory() + wxString::Format("dump_file_%li_%i_even_%i.dump", input_refinement->refinement_id, class_counter, job_counter + 1);

			wxString input_reconstruction = "";
			bool	 use_input_reconstruction = false;

/*
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
*/
			float    resolution_limit_ref               = 0.0;
			float	 smoothing_factor					= 1.0f;;
			float    padding							= 1.0f;
			bool	 normalize_particles				= true;
			bool	 exclude_blank_edges				= false;
			bool	 split_even_odd						= false;
			bool     centre_mass                        = false;

			bool threshold_input_3d = true;

			my_job_package.AddJob("ttttttttiifffffffffffffbbbbbbbbbbtt",
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

void Generate3DPanel::RunReconstructionJob()
{
	running_job_type = RECONSTRUCTION;
	number_of_received_particle_results = 0;

	// in the future store the reconstruction parameters..

	// empty scratch directory..

//	if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath() + "/Refine3D/") == true) wxFileName::Rmdir(main_frame->current_project.scratch_directory.GetFullPath() + "/Refine3D/", wxPATH_RMDIR_RECURSIVE);
//	if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath() + "/Refine3D/") == false) wxFileName::Mkdir(main_frame->current_project.scratch_directory.GetFullPath() + "/Refine3D/");

	// launch a controller

	if (input_refinement->number_of_classes > 1) WriteBlueText("Calculating Reconstructions...");
	else WriteBlueText("Calculating Reconstruction...");

	current_job_id = main_frame->job_controller.AddJob(this, active_reconstruction_run_profile.manager_command, active_reconstruction_run_profile.gui_address);
	my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes;
	    if (my_job_package.number_of_jobs + 1 < my_job_package.my_profile.ReturnTotalJobs()) number_of_refinement_processes = my_job_package.number_of_jobs + 1;
	    else number_of_refinement_processes =  my_job_package.my_profile.ReturnTotalJobs();

		if (number_of_refinement_processes >= 100000) length_of_process_number = 6;
		else
		if (number_of_refinement_processes >= 10000) length_of_process_number = 5;
		else
		if (number_of_refinement_processes >= 1000) length_of_process_number = 4;
		else
		if (number_of_refinement_processes >= 100) length_of_process_number = 3;
		else
		if (number_of_refinement_processes >= 10) length_of_process_number = 2;
		else
		length_of_process_number = 1;

		if (length_of_process_number == 6) NumberConnectedText->SetLabel(wxString::Format("%6i / %6li processes connected.", 0, number_of_refinement_processes));
		else
		if (length_of_process_number == 5) NumberConnectedText->SetLabel(wxString::Format("%5i / %5li processes connected.", 0, number_of_refinement_processes));
		else
		if (length_of_process_number == 4) NumberConnectedText->SetLabel(wxString::Format("%4i / %4li processes connected.", 0, number_of_refinement_processes));
		else
		if (length_of_process_number == 3) NumberConnectedText->SetLabel(wxString::Format("%3i / %3li processes connected.", 0, number_of_refinement_processes));
		else
		if (length_of_process_number == 2) NumberConnectedText->SetLabel(wxString::Format("%2i / %2li processes connected.", 0, number_of_refinement_processes));

		NumberConnectedText->SetLabel(wxString::Format("%i / %li processes connected.", 0, number_of_refinement_processes));
		TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
		Layout();
		running_job = true;
		my_job_tracker.StartTracking(my_job_package.number_of_jobs);

	}
	ProgressBar->Pulse();
}


void Generate3DPanel::SetupMerge3dJob()
{
	int class_counter;

	long number_of_3d_jobs = main_frame->current_project.database.ReturnSingleLongFromSelectCommand("select count(*) from refinement_list;");

	my_job_package.Reset(active_reconstruction_run_profile, "merge3d", active_refinement_package->number_of_classes);
	output_filenames.Clear();

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{

		wxString output_reconstruction_1;
		wxString output_reconstruction_2;

		if (active_save_half_maps == true)
		{
			output_reconstruction_1 = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/generate3d_volume_%li_%li_%i_map1.mrc", number_of_3d_jobs, input_refinement->refinement_id, class_counter + 1);
			output_reconstruction_2 = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/generate3d_volume_%li_%li_%i_map2.mrc", number_of_3d_jobs, input_refinement->refinement_id, class_counter + 1);

		}
		else
		{
			output_reconstruction_1 = "/dev/null";
			output_reconstruction_2 = "/dev/null";
		}
		wxString output_reconstruction_filtered		= main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/generate3d_volume_%li_%li_%i.mrc", number_of_3d_jobs, input_refinement->refinement_id, class_counter + 1);
		output_filenames.Add(output_reconstruction_filtered);

		wxString output_resolution_statistics		= "/dev/null";
		float 	 molecular_mass_kDa					= active_refinement_package->estimated_particle_weight_in_kda;
		float    inner_mask_radius					= active_inner_mask_radius;
		float    outer_mask_radius					= active_mask_radius;
		wxString dump_file_seed_1 					= main_frame->ReturnGenerate3DScratchDirectory() + wxString::Format("dump_file_%li_%i_odd_.dump", input_refinement->refinement_id, class_counter);
		wxString dump_file_seed_2 					= main_frame->ReturnGenerate3DScratchDirectory() + wxString::Format("dump_file_%li_%i_even_.dump", input_refinement->refinement_id, class_counter);

		bool save_orthogonal_views_image = true;
		wxString orthogonal_views_filename = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/OrthViews/generate3d_volume_%li_%li_%i.mrc", number_of_3d_jobs, input_refinement->refinement_id, class_counter + 1);

		my_job_package.AddJob("ttttfffttibt",	output_reconstruction_1.ToUTF8().data(),
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



void Generate3DPanel::RunMerge3dJob()
{
	running_job_type = MERGE;

	// start job..

	if (input_refinement->number_of_classes > 1) WriteBlueText("Merging and Filtering Reconstructions...");
	else
	WriteBlueText("Merging and Filtering Reconstruction...");

	current_job_id = main_frame->job_controller.AddJob(this, active_reconstruction_run_profile.manager_command, active_reconstruction_run_profile.gui_address);
	my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes;
	    if (my_job_package.number_of_jobs + 1 < my_job_package.my_profile.ReturnTotalJobs()) number_of_refinement_processes = my_job_package.number_of_jobs + 1;
	    else number_of_refinement_processes =  my_job_package.my_profile.ReturnTotalJobs();

		if (number_of_refinement_processes >= 100000) length_of_process_number = 6;
		else
		if (number_of_refinement_processes >= 10000) length_of_process_number = 5;
		else
		if (number_of_refinement_processes >= 1000) length_of_process_number = 4;
		else
		if (number_of_refinement_processes >= 100) length_of_process_number = 3;
		else
		if (number_of_refinement_processes >= 10) length_of_process_number = 2;
		else
		length_of_process_number = 1;

		if (length_of_process_number == 6) NumberConnectedText->SetLabel(wxString::Format("%6i / %6li processes connected.", 0, number_of_refinement_processes));
		else
		if (length_of_process_number == 5) NumberConnectedText->SetLabel(wxString::Format("%5i / %5li processes connected.", 0, number_of_refinement_processes));
		else
		if (length_of_process_number == 4) NumberConnectedText->SetLabel(wxString::Format("%4i / %4li processes connected.", 0, number_of_refinement_processes));
		else
		if (length_of_process_number == 3) NumberConnectedText->SetLabel(wxString::Format("%3i / %3li processes connected.", 0, number_of_refinement_processes));
		else
		if (length_of_process_number == 2) NumberConnectedText->SetLabel(wxString::Format("%2i / %2li processes connected.", 0, number_of_refinement_processes));
		else

		NumberConnectedText->SetLabel(wxString::Format("%i / %li processes connected.", 0, number_of_refinement_processes));

		StartPanel->Show(false);
		ProgressPanel->Show(true);

		ExpertPanel->Show(false);
		InfoPanel->Show(false);
		OutputTextPanel->Show(true);
			//	CTFResultsPanel->Show(true);

		ExpertToggleButton->Enable(false);
		RefinementPackageComboBox->Enable(false);
		InputParametersComboBox->Enable(false);

		TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
		Layout();
		running_job = true;
		my_job_tracker.StartTracking(my_job_package.number_of_jobs);

		}

		ProgressBar->Pulse();
}


void Generate3DPanel::ProcessJobResult(JobResult *result_to_process)
{

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
			int current_percentage = float(number_of_received_particle_results) / float(input_refinement->number_of_particles * input_refinement->number_of_classes) * 100.0;
			if (current_percentage > 100) current_percentage = 100;
			ProgressBar->SetValue(current_percentage);
			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float((input_refinement->number_of_particles * input_refinement->number_of_classes) - number_of_received_particle_results) * seconds_per_job;

			TimeRemaining time_remaining;
			if (seconds_remaining > 3600) time_remaining.hours = seconds_remaining / 3600;
			else time_remaining.hours = 0;

			if (seconds_remaining > 60) time_remaining.minutes = (seconds_remaining / 60) - (time_remaining.hours * 60);
			else time_remaining.minutes = 0;

			time_remaining.seconds = seconds_remaining - ((time_remaining.hours * 60 + time_remaining.minutes) * 60);
			TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_remaining.hours, time_remaining.minutes, time_remaining.seconds));
		}


	}
	else
	if (running_job_type == MERGE)
	{
		int number_of_points = result_to_process->result_data[0];
		int class_number = int(result_to_process->result_data[1] + 0.5);
		int array_position = 2;
		float current_resolution;
		float fsc;
		float part_fsc;
		float part_ssnr;
		float rec_ssnr;

		input_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.Init(input_refinement->resolution_statistics_pixel_size, input_refinement->resolution_statistics_box_size);

		input_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.FSC.ClearData();
		input_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_FSC.ClearData();
		input_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_SSNR.ClearData();
		input_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.rec_SSNR.ClearData();


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

			input_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.FSC.AddPoint(current_resolution, fsc);
			input_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_FSC.AddPoint(current_resolution, part_fsc);
			input_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_SSNR.AddPoint(current_resolution, part_ssnr);
			input_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.rec_SSNR.AddPoint(current_resolution, rec_ssnr);
		}
	}
}



void Generate3DPanel::ProcessAllJobsFinished()
{

	// Update the GUI with project timings
	extern MyOverviewPanel *overview_panel;
	overview_panel->SetProjectInfo();


	if (running_job_type == RECONSTRUCTION)
	{
		main_frame->job_controller.KillJob(my_job_id);
		SetupMerge3dJob();
		RunMerge3dJob();
	}
	else
	if (running_job_type == MERGE)
	{
		long current_reconstruction_id;

		OrthDrawerThread *result_thread;
		ShortRefinementInfo *pointer_to_refinement_info = refinement_package_asset_panel->ReturnPointerToShortRefinementInfoByRefinementID(input_refinement->refinement_id);

		if (input_refinement->number_of_classes > 1) result_thread = new OrthDrawerThread(this, output_filenames, "Output Reconstructions");
		else result_thread = new OrthDrawerThread(this, output_filenames, "Output Reconstruction");

		if ( result_thread->Run() != wxTHREAD_NO_ERROR )
		{
			WriteErrorText("Error: Cannot start result creation thread, results not displayed");
			delete result_thread;
		}

		int class_counter;
		main_frame->job_controller.KillJob(my_job_id);

		VolumeAsset temp_asset;


		temp_asset.pixel_size = input_refinement->resolution_statistics_pixel_size;
		temp_asset.x_size = input_refinement->resolution_statistics_box_size;
		temp_asset.y_size = input_refinement->resolution_statistics_box_size;
		temp_asset.z_size = input_refinement->resolution_statistics_box_size;

		// add the volumes etc to the database..
		main_frame->current_project.database.Begin();
		main_frame->current_project.database.BeginVolumeAssetInsert();

		WriteInfoText("");

		for (class_counter = 0; class_counter < input_refinement->number_of_classes; class_counter++)
		{
			current_reconstruction_id = main_frame->current_project.database.ReturnHighestReconstructionID() + 1;
			temp_asset.reconstruction_job_id = current_reconstruction_id;
			temp_asset.asset_id = volume_asset_panel->current_asset_number;

			if (active_update_statistics == true)
			{
				main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_DETAILS_%li SET RECONSTRUCTED_VOLUME_ASSET_ID=%li, RECONSTRUCTION_ID=%li WHERE CLASS_NUMBER=%i;", input_refinement->refinement_id, long(temp_asset.asset_id), current_reconstruction_id, class_counter + 1));
				pointer_to_refinement_info->reconstructed_volume_asset_ids[class_counter] = temp_asset.asset_id;
			}

			temp_asset.asset_name = wxString::Format("Generated from #%li - Class #%i", input_refinement->refinement_id, class_counter + 1);
			temp_asset.filename = output_filenames[class_counter];
			volume_asset_panel->AddAsset(&temp_asset);
			main_frame->current_project.database.AddNextVolumeAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath(), temp_asset.reconstruction_job_id, temp_asset.pixel_size, temp_asset.x_size, temp_asset.y_size, temp_asset.z_size);

			// add the reconstruction job..

			main_frame->current_project.database.AddReconstructionJob(current_reconstruction_id, active_refinement_package->asset_id, input_refinement->refinement_id, "", active_inner_mask_radius, active_mask_radius, active_resolution_limit_rec, active_score_weight_conversion, active_adjust_scores, active_crop_images, active_save_half_maps, false, 1.0, class_counter + 1, long(temp_asset.asset_id));

		}

		main_frame->current_project.database.EndVolumeAssetInsert();

		wxArrayFloat average_occupancies = input_refinement->UpdatePSSNR();
		WriteInfoText("");

		if (input_refinement->number_of_classes > 1)
		{
			for (class_counter = 0; class_counter < input_refinement->number_of_classes; class_counter++)
			{
				WriteInfoText(wxString::Format(wxT("Est. Res. Class %2i = %2.2f Å (%2.2f %%)"), class_counter + 1, input_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnEstimatedResolution(), average_occupancies[class_counter]));
			}
		}
		else
		{
			WriteInfoText(wxString::Format(wxT("Est. Res. = %2.2f Å"), input_refinement->class_refinement_results[0].class_resolution_statistics.ReturnEstimatedResolution()));
		}

		WriteInfoText("");


		if (active_update_statistics == true)
		{
			main_frame->current_project.database.UpdateRefinementResolutionStatistics(input_refinement);
			refinement_results_panel->is_dirty = true;
		}

		long point_counter;

		ShowRefinementResultsPanel->FSCResultsPanel->AddRefinement(input_refinement);

		/*if (ShowRefinementResultsPanel->TopBottomSplitter->IsSplit() == false)
		{
			ShowRefinementResultsPanel->TopBottomSplitter->SplitHorizontally(ShowRefinementResultsPanel->TopPanel, ShowRefinementResultsPanel->BottomPanel);
			ShowRefinementResultsPanel->FSCResultsPanel->Show(true);
		}*/

		main_frame->current_project.database.Commit();
		main_frame->DirtyVolumes();
		global_delete_generate3d_scratch();

		WriteBlueText("Reconstruction is finished!");
		CancelAlignmentButton->Show(false);
		FinishButton->Show(true);
		TimeRemainingText->SetLabel("Time Remaining : Finished!");
		ProgressBar->SetValue(100);
		ShowRefinementResultsPanel->Show(true);
		Layout();
	}

}

void Generate3DPanel::OnOrthThreadComplete(ReturnProcessedImageEvent& my_event)
{

	Image *new_image = my_event.GetImage();

	if (new_image != NULL)
	{
		ShowRefinementResultsPanel->ShowOrthDisplayPanel->OpenImage(new_image, my_event.GetString(), true);

	/*	if (ShowRefinementResultsPanel->LeftRightSplitter->IsSplit() == false)
		{
			ShowRefinementResultsPanel->LeftRightSplitter->SplitVertically(ShowRefinementResultsPanel->LeftPanel, ShowRefinementResultsPanel->RightPanel, 600);
			Layout();
		}
		*/
	}
}
