#include "../core/gui_core_headers.h"
wxDEFINE_EVENT(wxEVT_RESAMPLE_VOLUME_EVENT, ReturnProcessedImageEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_IMPOSESYMMETRY_DONE, wxThreadEvent);

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyVolumeAssetPanel *volume_asset_panel;



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
	Bind(wxEVT_AUTOMASKERTHREAD_COMPLETED, &AbInitio3DPanel::OnMaskerThreadComplete, this);
	Bind(RETURN_PROCESSED_IMAGE_EVT, &AbInitio3DPanel::OnOrthThreadComplete, this);
	Bind(wxEVT_RESAMPLE_VOLUME_EVENT, &AbInitio3DPanel::OnVolumeResampled, this);
	Bind(wxEVT_COMMAND_IMPOSESYMMETRY_DONE, &AbInitio3DPanel::OnImposeSymmetryThreadComplete, this);
	FillRefinementPackagesComboBox();

	// limits

	InitialResolutionLimitTextCtrl->SetMinMaxValue(0, 300);
	FinalResolutionLimitTextCtrl->SetMinMaxValue(0, 300);
	StartPercentUsedTextCtrl->SetMinMaxValue(0.001, 100);
	EndPercentUsedTextCtrl->SetMinMaxValue(0.001, 100);
	SmoothingFactorTextCtrl->SetMinMaxValue(0, 1);

	active_orth_thread_id = -1;
	active_mask_thread_id = -1;
	active_sym_thread_id = -1;
	next_thread_id = 1;
}

void AbInitio3DPanel::SetInfo()
{
	wxLogNull *suppress_png_warnings = new wxLogNull;

	#include "icons/VSV_startup_800.cpp"
	wxBitmap vsv_startup_bmp = wxBITMAP_PNG_FROM_DATA(VSV_startup_800);
	delete suppress_png_warnings;

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
	InfoText->WriteText(wxT("When a prior 3D reconstruction of a molecule or complex is available that has a closely related structure, it is usually fastest and safest to use it to initialize 3D refinement and reconstruction of a new dataset. However, in many cases such a structure is not available, or an independently determined structure is desired. Ab-initio 3D reconstruction offers a way to start refinement and reconstruction without any prior structural information. The following figure shows an ab-initio 3D reconstruction of VSV polymerase (Liang et al. 2015):"));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(vsv_startup_bmp);
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

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
		ExpertPanel->Freeze();

		float 	 molecular_mass_kDa = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_weight_in_kda;
		wxString current_symmetry_string = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).symmetry;
		wxChar   symmetry_type;

		current_symmetry_string = current_symmetry_string.Trim();
		current_symmetry_string = current_symmetry_string.Trim(false);

		MyDebugAssertTrue(current_symmetry_string.Length() > 0, "symmetry string is blank");
		symmetry_type = current_symmetry_string.Capitalize()[0];

/*		if (symmetry_type == 'O' || symmetry_type == 'I')
		{
			NumberStartsSpinCtrl->SetValue(1);
			AutoMaskNoRadio->SetValue(true);
		}
		else
		{
			NumberStartsSpinCtrl->SetValue(2);
			AutoMaskYesRadio->SetValue(true);
		}
*/

		SearchRangeXTextCtrl->ChangeValueFloat(refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.15f);
		SearchRangeYTextCtrl->ChangeValueFloat(refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.15f);

		float    mask_radius = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.75;

		GlobalMaskRadiusTextCtrl->ChangeValueFloat(mask_radius);
		InnerMaskRadiusTextCtrl->ChangeValueFloat(0.0f);

		AutoMaskYesRadio->SetValue(true);
		NumberStartsSpinCtrl->SetValue(2);
		NumberRoundsSpinCtrl->SetValue(40);

		if (symmetry_type == 'O' || symmetry_type == 'I' || molecular_mass_kDa <= 200.0f)
		{
			InitialResolutionLimitTextCtrl->ChangeValueFloat(20);
		}
		else
		{
			InitialResolutionLimitTextCtrl->ChangeValueFloat(20);
		}
		FinalResolutionLimitTextCtrl->ChangeValueFloat(8);
		AutoPercentUsedYesRadio->SetValue(true);

		StartPercentUsedTextCtrl->ChangeValueFloat(10);
		EndPercentUsedTextCtrl->ChangeValueFloat(10);

		ApplyBlurringNoRadioButton->SetValue(true);
		SmoothingFactorTextCtrl->ChangeValueFloat(1.00);

		AlwaysApplySymmetryNoButton->SetValue(true);

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

	active_mask_thread_id = -1;
	active_sym_thread_id = -1;
	active_orth_thread_id = -1;

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

		MyDebugAssertTrue(sock == main_frame->job_controller.job_list[my_job_id].socket, "Socket event from Non conduit socket??");

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
			if (my_abinitio_manager.running_job_type == ALIGN_SYMMETRY)
			{
					// is this better than all the current_best?

				int current_class = temp_result.result_data[7] + 0.5;

				//wxPrintf("got final result %f, %f, %f - %f, %f, %f = %f\n", temp_result.result_data[0], temp_result.result_data[1], temp_result.result_data[2], temp_result.result_data[3], temp_result.result_data[4], temp_result.result_data[5], temp_result.result_data[6]);
				if (temp_result.result_data[6] > my_abinitio_manager.align_sym_best_correlations[current_class])
				{
					my_abinitio_manager.align_sym_best_correlations[current_class] = temp_result.result_data[6];
					my_abinitio_manager.align_sym_best_x_rots[current_class] = temp_result.result_data[0];
					my_abinitio_manager.align_sym_best_y_rots[current_class] = temp_result.result_data[1];
					my_abinitio_manager.align_sym_best_z_rots[current_class] = temp_result.result_data[2];
					my_abinitio_manager.align_sym_best_x_shifts[current_class] = temp_result.result_data[3];
					my_abinitio_manager.align_sym_best_y_shifts[current_class] = temp_result.result_data[4];
					my_abinitio_manager.align_sym_best_z_shifts[current_class] = temp_result.result_data[5];
				}
			}
			else
			{
				my_abinitio_manager.ProcessJobResult(&temp_result);
			}
			//wxPrintf("Warning: Received socket_job_result - should this happen?");

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
			// As soon as it sends us the message that all jobs are finished, the controller should also
			// send timing info - we need to remember this
			long timing_from_controller;
			ReadFromSocket(sock, &timing_from_controller, sizeof(long));
			main_frame->current_project.total_cpu_hours += timing_from_controller / 3600000.0;
			main_frame->current_project.total_jobs_run += my_job_tracker.total_number_of_jobs;

			// Update project statistics in the database
			main_frame->current_project.WriteProjectStatisticsToDatabase();

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

	active_mask_thread_id = -1;
	active_sym_thread_id = -1;
	active_orth_thread_id = -1;

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

	active_mask_thread_id = -1;
	active_sym_thread_id = -1;
	active_orth_thread_id = -1;

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
	wxString input_file;
	number_of_resampled_volumes_recieved = 0;

	current_startup_id = main_frame->current_project.database.ReturnHighestStartupID() + 1;

	for (int class_counter = 0; class_counter < my_abinitio_manager.input_refinement->number_of_classes; class_counter++)
	{
		int current_round_number = (my_abinitio_manager.number_of_rounds_to_run * my_abinitio_manager.number_of_starts_run) + my_abinitio_manager.number_of_rounds_run - 1;
		input_file = main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup3d_%i_%i.mrc", current_round_number, class_counter);
		ResampleVolumeThread *resample_thread = new ResampleVolumeThread(this, input_file, my_abinitio_manager.active_refinement_package->stack_box_size, my_abinitio_manager.active_refinement_package->contained_particles[0].pixel_size, class_counter + 1);

		if ( resample_thread->Run() != wxTHREAD_NO_ERROR )
		{
			WriteErrorText("Error: Cannot start resample thread, results not saved");
			delete resample_thread;
		}

	}
}

void AbInitio3DPanel::TakeLastStart()
{
	wxString input_file;
	number_of_resampled_volumes_recieved = 0;
		current_startup_id = main_frame->current_project.database.ReturnHighestStartupID() + 1;

	for (int class_counter = 0; class_counter < my_abinitio_manager.input_refinement->number_of_classes; class_counter++)
	{
		// what is the round number of the end of the last start..

		int last_start_round_number = (my_abinitio_manager.number_of_rounds_to_run * my_abinitio_manager.number_of_starts_run) - 1;
		input_file = main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup3d_%i_%i.mrc", last_start_round_number, class_counter);

		ResampleVolumeThread *resample_thread = new ResampleVolumeThread(this, input_file, my_abinitio_manager.active_refinement_package->stack_box_size, my_abinitio_manager.active_refinement_package->contained_particles[0].pixel_size, class_counter + 1);

		if ( resample_thread->Run() != wxTHREAD_NO_ERROR )
		{
			WriteErrorText("Error: Cannot start resample thread, results not saved");
			delete resample_thread;
		}
	}
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

	// set the active refinement package..

	active_refinement_package = &refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection());
	current_refinement_package_asset_id = active_refinement_package->asset_id;

	// this should be the random start..
	current_input_refinement_id = active_refinement_package->refinement_ids[0];
	// create a refinement with random angles etc..

	input_refinement = main_frame->current_project.database.GetRefinementByID(current_input_refinement_id);
	input_refinement->refinement_id = 0;
	output_refinement = input_refinement;
	current_output_refinement_id = input_refinement->refinement_id;


	number_of_starts_to_run = my_parent->NumberStartsSpinCtrl->GetValue();
	number_of_rounds_to_run = my_parent->NumberRoundsSpinCtrl->GetValue();

	active_start_res = my_parent->InitialResolutionLimitTextCtrl->ReturnValue();
	active_end_res = my_parent->FinalResolutionLimitTextCtrl->ReturnValue();

	active_start_percent_used = my_parent->StartPercentUsedTextCtrl->ReturnValue();
	active_end_percent_used = my_parent->EndPercentUsedTextCtrl->ReturnValue();

	active_inner_mask_radius = my_parent->InnerMaskRadiusTextCtrl->ReturnValue();
	active_always_apply_symmetry = my_parent->AlwaysApplySymmetryYesButton->GetValue();

	my_parent->PlotPanel->Clear();
	my_parent->PlotPanel->my_notebook->SetSelection(0);

	active_should_automask = my_parent->AutoMaskYesRadio->GetValue();
	active_global_mask_radius = my_parent->GlobalMaskRadiusTextCtrl->ReturnValue();
	active_global_mask_radius = std::min(active_global_mask_radius, input_refinement->resolution_statistics_box_size * 0.45f * input_refinement->resolution_statistics_pixel_size);

	active_should_apply_blurring = my_parent->ApplyBlurringYesRadioButton->GetValue();
	active_smoothing_factor = my_parent->SmoothingFactorTextCtrl->ReturnValue();

	active_search_range_x = my_parent->SearchRangeXTextCtrl->ReturnValue();
	active_search_range_y = my_parent->SearchRangeYTextCtrl->ReturnValue();

	active_refinement_run_profile = run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()];
	active_reconstruction_run_profile = run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()];

	active_auto_set_percent_used = my_parent->AutoPercentUsedYesRadio->GetValue();

	if (active_always_apply_symmetry == true && active_refinement_package->symmetry != "C1") apply_symmetry = true;
	else apply_symmetry = false;

	// work out the percent used

	long number_of_particles = active_refinement_package->contained_particles.GetCount();
	int number_of_classes = active_refinement_package->number_of_classes;

	// re-randomise the input parameters, and set the default resolution statistics..

	for (class_counter = 0; class_counter < number_of_classes; class_counter++)
	{
		for ( counter = 0; counter < number_of_particles; counter++)
		{
			if (number_of_classes == 1) input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = 100.0;
			else input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = 100.00 / input_refinement->number_of_classes;

			/* for a scheme that does not put more views at the top - use :-
			*/
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].phi = global_random_number_generator.GetUniformRandom() * 180.0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].theta = rad_2_deg(acosf(2.0f * fabsf(global_random_number_generator.GetUniformRandom()) - 1.0f));
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].psi = global_random_number_generator.GetUniformRandom() * 180.0;


			//input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].phi = global_random_number_generator.GetUniformRandom() * 180.0;
			//input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].theta = global_random_number_generator.GetUniformRandom() * 180.0;
			//input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].psi = global_random_number_generator.GetUniformRandom() * 180.0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].xshift = global_random_number_generator.GetUniformRandom() * 25.0f;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].yshift = global_random_number_generator.GetUniformRandom() * 25.0f;;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].score = 0.0;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].image_is_active = 1;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].sigma = 1.0;
		}

		input_refinement->class_refinement_results[class_counter].class_resolution_statistics.GenerateDefaultStatistics(active_refinement_package->estimated_particle_weight_in_kda);
	}

	// need to take into account symmetry

	wxString current_symmetry_string = active_refinement_package->symmetry;
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

	if (active_auto_set_percent_used == true)
	{
		int symmetry_number = ReturnNumberofAsymmetricUnits(active_refinement_package->symmetry);

		long number_of_asym_units = number_of_particles;

		long wanted_start_number_of_asym_units = 2500 * number_of_classes;
		long wanted_end_number_of_asym_units = 10000 * number_of_classes;

		// what percentage is this.

		start_percent_used = (float(wanted_start_number_of_asym_units) / float(number_of_asym_units)) * 100.0;
		end_percent_used = (float(wanted_end_number_of_asym_units) / float(number_of_asym_units)) * 100.0;

		symmetry_start_percent_used = (float(wanted_start_number_of_asym_units) / float(number_of_asym_units  * symmetry_number)) * 100.0;
		symmetry_end_percent_used = (float(wanted_end_number_of_asym_units) / float(number_of_asym_units  * symmetry_number)) * 100.0;

		if (start_percent_used > 100.0) start_percent_used = 100.0;
		if (end_percent_used > 100.0) end_percent_used = 100.0;

		if (symmetry_start_percent_used > 100.0f) symmetry_start_percent_used = 100.0f;
		if (symmetry_end_percent_used > 100.0) symmetry_end_percent_used = 100.0;

		//	if (end_percent_used > 25)
		//	{
		//		if (number_of_classes > 1) my_parent->WriteWarningText(wxString::Format("Warning : Using max %.2f %% of the images per round, this is quite high, you may wish to increase the number of particles or reduce the number of classes", end_percent_used));
		//		else my_parent->WriteWarningText(wxString::Format("Warning : Using max %.2f %% of the images per round, this is quite high, you may wish to increase the number of particles", end_percent_used));
		//	}
	}
	else
	{
		start_percent_used = active_start_percent_used;
		end_percent_used = active_end_percent_used;

		symmetry_start_percent_used = active_start_percent_used;
		symmetry_end_percent_used = active_start_percent_used;
	}

	if (apply_symmetry == false) current_percent_used = start_percent_used;
	else current_percent_used = symmetry_start_percent_used;


/*
	startup_percent_used = (float(wanted_number_of_asym_units) / float(number_of_asym_units)) * 100.0;
	wxPrintf("percent used = %.2f\n", startup_percent_used);
	if (startup_percent_used > 100) startup_percent_used = 100;

	if (startup_percent_used > 20 && startup_percent_used < 30) my_parent->WriteWarningText(wxString::Format("Warning : Using %.2f %% of the images per round, this is quite high, you may wish to increase the number of particles or reduce the number of classes", startup_percent_used));
	else
	if (startup_percent_used > 30) my_parent->WriteWarningText(wxString::Format("Warning : Using %.2f %% of the images per round, this is very high, you may wish to increase the number of particles or reduce the number of classes", startup_percent_used));
*/
	current_high_res_limit = active_start_res;
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

	// are we pre-preparing the stack?

	if (active_end_res > active_refinement_package->contained_particles[0].pixel_size * 3.0f )
	{
		SetupPrepareStackJob();
		RunPrepareStackJob();
	}
	else // just start the reconstruction job
	{
		stack_has_been_precomputed = false;
		active_pixel_size = active_refinement_package->contained_particles[0].pixel_size;
		active_stack_filename = active_refinement_package->stack_filename;
		stack_bin_factor = 1.0f;

		SetupReconstructionJob();
		RunReconstructionJob();
	}
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

		if (active_should_automask == true)
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

			//current_high_res_limit = start_res + (end_res - start_res) * sqrtf(float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
			//next_high_res_limit = start_res + (end_res - start_res) * sqrtf(float(number_of_rounds_run + 1) / float(number_of_rounds_to_run - 1));

//			current_high_res_limit = active_start_res + (active_end_res - active_start_res) * powf((float(number_of_rounds_run) / float(number_of_rounds_to_run - 1)), 1.0);
//			next_high_res_limit = active_start_res + (active_end_res - active_start_res) * powf((float(number_of_rounds_run + 1) / float(number_of_rounds_to_run - 1)), 1.0);
			current_high_res_limit = active_start_res + (active_end_res - active_start_res) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
			if (number_of_rounds_run % myroundint(number_of_rounds_to_run / 10.0f) <= 1 && number_of_rounds_run <= number_of_rounds_to_run * 0.65f) current_high_res_limit = active_start_res;
//			if (number_of_rounds_run % myroundint(number_of_rounds_to_run / 10.0f) <= 1 && number_of_rounds_run <= number_of_rounds_to_run * 0.8f)
//				current_high_res_limit = active_start_res + (active_end_res - active_start_res) * (0.3 * float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
			next_high_res_limit = active_start_res + (active_end_res - active_start_res) * (float(number_of_rounds_run + 1) / float(number_of_rounds_to_run - 1));


			if (next_high_res_limit < 0.0) next_high_res_limit = 0.0;

			//current_percent_used = start_percent_used + (end_percent_used - start_percent_used) * sqrtf(float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
		//	current_percent_used = start_percent_used + (end_percent_used - start_percent_used) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
			if (apply_symmetry == true)	current_percent_used = symmetry_start_percent_used + (symmetry_end_percent_used - symmetry_start_percent_used) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
			else current_percent_used = start_percent_used + (end_percent_used - start_percent_used) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));

			if (number_of_rounds_run == myroundint(float(number_of_rounds_to_run) * 0.75f) && active_refinement_package->symmetry != "C1" && active_always_apply_symmetry == false && number_of_starts_run == 0) // we need to align to the symmetry
			{
				SetupAlignSymmetryJob();
				RunAlignSymmetryJob();
			}
			else
			{
				if (active_should_automask == true)
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
		else
		{
			number_of_starts_run++;

			if (number_of_starts_run < number_of_starts_to_run)
			{
				number_of_rounds_run = 0;
				UpdatePlotPanel();

				delete input_refinement;
				input_refinement = output_refinement;
				output_refinement = new Refinement;
				output_refinement->refinement_id = (number_of_rounds_to_run * number_of_starts_run) + number_of_rounds_run;
				output_refinement->number_of_classes = input_refinement->number_of_classes;

				current_high_res_limit = active_start_res + (active_end_res - active_start_res) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
				next_high_res_limit = active_start_res + (active_end_res - active_start_res) * (float(number_of_rounds_run + 1) / float(number_of_rounds_to_run - 1));

				if (next_high_res_limit < 0.0) next_high_res_limit = 0.0;

				//current_percent_used = start_percent_used + (end_percent_used - start_percent_used) * sqrtf(float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
				if (apply_symmetry == true)	current_percent_used = symmetry_start_percent_used + (symmetry_end_percent_used - symmetry_start_percent_used) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
				else current_percent_used = start_percent_used + (end_percent_used - start_percent_used) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
				//current_percent_used = start_percent_used + (end_percent_used - start_percent_used) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));

				if (active_should_automask == true)
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

				main_frame->job_controller.KillJob(my_parent->my_job_id);
				my_parent->Freeze();
				my_parent->WriteBlueText("All refinement cycles are finished!");
				my_parent->TimeRemainingText->SetLabel("Time Remaining : Finished!");
				my_parent->CancelAlignmentButton->Show(false);
				my_parent->CurrentLineOne->Show(false);
				my_parent->CurrentLineTwo->Show(false);
				my_parent->TakeCurrentResultButton->Show(false);
				my_parent->TakeLastStartResultButton->Show(false);
				my_parent->FinishButton->Show(true);
				my_parent->ProgressPanel->Layout();
				my_parent->Thaw();

				my_parent->TakeCurrent();
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

	number_of_reconstruction_processes = active_reconstruction_run_profile.ReturnTotalJobs();
	number_of_reconstruction_jobs = number_of_reconstruction_processes - 1;

	number_of_particles = active_refinement_package->contained_particles.GetCount();

	if (number_of_particles - number_of_reconstruction_jobs < number_of_reconstruction_jobs) particles_per_job = 1;
	else particles_per_job = float(number_of_particles - number_of_reconstruction_jobs) / float(number_of_reconstruction_jobs);

	my_parent->my_job_package.Reset(active_reconstruction_run_profile, "reconstruct3d", number_of_reconstruction_jobs * active_refinement_package->number_of_classes);

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		current_particle_counter = 1.0;

		for (job_counter = 0; job_counter < number_of_reconstruction_jobs; job_counter++)
		{
			wxString input_particle_stack 		= active_stack_filename;//active_refinement_package->stack_filename;
			wxString input_parameter_file 		= written_parameter_files[class_counter];
			wxString output_reconstruction_1    = "/dev/null";
			wxString output_reconstruction_2			= "/dev/null";
			wxString output_reconstruction_filtered		= "/dev/null";
			wxString output_resolution_statistics		= "/dev/null";
			wxString my_symmetry;

			if (apply_symmetry == true)	my_symmetry = active_refinement_package->symmetry;
			else my_symmetry = "C1";

			long	 first_particle						= myroundint(current_particle_counter);

			current_particle_counter += particles_per_job;
			if (current_particle_counter > number_of_particles  || counter == number_of_reconstruction_jobs - 1) current_particle_counter = number_of_particles;

			long	 last_particle						= myroundint(current_particle_counter);
			current_particle_counter+=1.0;

			float 	 pixel_size							= active_pixel_size;//active_refinement_package->contained_particles[0].pixel_size;
			float    voltage_kV							= active_refinement_package->contained_particles[0].microscope_voltage;
			float 	 spherical_aberration_mm			= active_refinement_package->contained_particles[0].spherical_aberration;
			float    amplitude_contrast					= active_refinement_package->contained_particles[0].amplitude_contrast;
			float 	 molecular_mass_kDa					= active_refinement_package->estimated_particle_weight_in_kda;
			float    inner_mask_radius					= active_inner_mask_radius;
			float    outer_mask_radius					= active_global_mask_radius;


			float    resolution_limit_rec;

			resolution_limit_rec					= next_high_res_limit;

			float    score_weight_conversion			= 0;

			float    score_threshold;
			float percent_multiplier;

			percent_multiplier = 2.5 + (3.5 - 2.5) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));

			if (current_percent_used * percent_multiplier < 100.0f) score_threshold = 0.333f; // we are refining 3 times more then current_percent_used, we want to use current percent used so it is always 1/3.
			else score_threshold = current_percent_used / 100.0; 	// now 3 times current_percent_used is more than 100%, we therefire refined them all, and so just take current_percent used

			// overwrites above
			//score_threshold =0.0f;

			bool	 adjust_scores						= true;
			bool	 invert_contrast					= active_refinement_package->stack_has_white_protein;
			bool	 crop_images						= false;
			bool	 dump_arrays						= true;
			wxString dump_file_1 						= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup_dump_file_%i_odd_%i.dump", class_counter, job_counter +1);
			wxString dump_file_2 						= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup_dump_file_%i_even_%i.dump", class_counter, job_counter + 1);

			wxString input_reconstruction;
			bool	 use_input_reconstruction;


			if (active_should_apply_blurring == true)
			{
				// do we have a reference..

				if (start_with_reconstruction == true)
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

			float    resolution_limit_ref               = current_high_res_limit;
			float	 smoothing_factor					= active_smoothing_factor;
			float    padding							= 1.0f;
			bool	 normalize_particles;

			if (stack_has_been_precomputed == true) normalize_particles = false;
			else normalize_particles = true;

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

	current_job_id = main_frame->job_controller.AddJob(my_parent, active_reconstruction_run_profile.manager_command, active_reconstruction_run_profile.gui_address);
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

	int number_of_reconstruction_jobs = active_reconstruction_run_profile.ReturnTotalJobs() - 1;

	int class_counter;

	my_parent->my_job_package.Reset(active_reconstruction_run_profile, "merge3d", active_refinement_package->number_of_classes);

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		wxString output_reconstruction_1			= "/dev/null";
		wxString output_reconstruction_2			= "/dev/null";

		wxString output_reconstruction_filtered;
		int output_number = (number_of_rounds_to_run * number_of_starts_run) + number_of_rounds_run;
		if (start_with_reconstruction == true) output_reconstruction_filtered = main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup3d_initial_%i_%i.mrc", output_number, class_counter);
		else output_reconstruction_filtered		= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup3d_%i_%i.mrc", output_number, class_counter);

		current_reference_filenames.Item(class_counter) = output_reconstruction_filtered;

		wxString output_resolution_statistics		= "/dev/null";
		float 	 molecular_mass_kDa					= active_refinement_package->estimated_particle_weight_in_kda;
		float    inner_mask_radius					= active_inner_mask_radius;
		float    outer_mask_radius					= active_global_mask_radius;
		outer_mask_radius = std::min(outer_mask_radius, input_refinement->resolution_statistics_box_size * 0.45f * input_refinement->resolution_statistics_pixel_size);
		wxString dump_file_seed_1 					= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup_dump_file_%i_odd_.dump",  class_counter);
		wxString dump_file_seed_2 					= main_frame->current_project.scratch_directory.GetFullPath() + wxString::Format("/Startup/startup_dump_file_%i_even_.dump", class_counter);

		bool save_orthogonal_views_image = false;
		wxString orthogonal_views_filename = "";
		float weiner_nominator;

		/*if (start_with_reconstruction == true) weiner_nominator = 100.0f;
		else
		if (number_of_rounds_run == 0 && number_of_starts_run == 0) weiner_nominator = 50.0f;
		else iner_nominator = 1.0f;

		*/
		if (number_of_rounds_run == 0 && number_of_starts_run == 0) weiner_nominator = 200.0f;
		else
		if (number_of_starts_run == 0)
		{
			weiner_nominator = 100 + (1 - 100) * (float(number_of_rounds_run) / float(number_of_rounds_to_run / 2));
			if (weiner_nominator < 1.0f) weiner_nominator = 1.0f;
		}
		else weiner_nominator = 1.0f;

		//my_parent->WriteInfoText(wxString::Format("weiner nominator = %f", weiner_nominator));

		my_parent->my_job_package.AddJob("ttttfffttibtif",	output_reconstruction_1.ToUTF8().data(),
														output_reconstruction_2.ToUTF8().data(),
														output_reconstruction_filtered.ToUTF8().data(),
														output_resolution_statistics.ToUTF8().data(),
														molecular_mass_kDa, inner_mask_radius, outer_mask_radius,
														dump_file_seed_1.ToUTF8().data(),
														dump_file_seed_2.ToUTF8().data(),
														class_counter + 1,
														save_orthogonal_views_image,
														orthogonal_views_filename.ToUTF8().data(),
														number_of_reconstruction_jobs,
														weiner_nominator);
	}
}



void AbInitioManager::RunMerge3dJob()
{
	running_job_type = MERGE;

	// start job..

	if (output_refinement->number_of_classes > 1) my_parent->WriteBlueText("Merging and Filtering Reconstructions...");
	else
	my_parent->WriteBlueText("Merging and Filtering Reconstruction...");

	current_job_id = main_frame->job_controller.AddJob(my_parent,active_reconstruction_run_profile.manager_command, active_reconstruction_run_profile.gui_address);
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
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].xshift = 0.0f;
			input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].yshift = 0.0f;

			//input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].image_is_active = 1;
			//input_refinement->class_refinement_results[class_counter].particle_refinement_results[counter].sigma = 1.0;
		}

		if (number_of_rounds_run < 3) input_refinement->class_refinement_results[class_counter].class_resolution_statistics.GenerateDefaultStatistics(active_refinement_package->estimated_particle_weight_in_kda);
	}


	//wxPrintf("refinement_id = %li\n", input_refinement->refinement_id);
	written_parameter_files = input_refinement->WriteFrealignParameterFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/startup_input_par");
	written_res_files = input_refinement->WriteResolutionStatistics(main_frame->current_project.parameter_file_directory.GetFullPath() + "/startup_input_stats");

//	wxPrintf("Input refinement has %li particles\n", input_refinement->number_of_particles);

	// for now, number of jobs is number of processes -1 (master)..

	number_of_refinement_processes = active_refinement_run_profile.ReturnTotalJobs();
	number_of_refinement_jobs = number_of_refinement_processes - 1;

	number_of_particles = active_refinement_package->contained_particles.GetCount();
	if (number_of_particles - number_of_refinement_jobs < number_of_refinement_jobs) particles_per_job = 1;
	else particles_per_job = float(number_of_particles - number_of_refinement_jobs) / float(number_of_refinement_jobs);

	my_parent->my_job_package.Reset(active_refinement_run_profile, "refine3d", number_of_refinement_jobs * active_refinement_package->number_of_classes);

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		current_particle_counter = 1;

		for (counter = 0; counter < number_of_refinement_jobs; counter++)
		{

			wxString input_particle_images					= active_stack_filename;//active_refinement_package->stack_filename;
			wxString input_parameter_file 					= written_parameter_files.Item(class_counter);
			wxString input_reconstruction					= current_reference_filenames.Item(class_counter);
			wxString input_reconstruction_statistics 		= written_res_files.Item(class_counter);
			bool	 use_statistics							= true;

			wxString ouput_matching_projections		 		= "";
			//wxString output_parameter_file					= "/tmp/output_par.par";
			//wxString ouput_shift_file						= "/tmp/output_shift.shft";
			wxString ouput_shift_file						= "/dev/null";

			wxString my_symmetry;
			if (apply_symmetry == true) my_symmetry = active_refinement_package->symmetry;
			else my_symmetry = "C1";

			long	 first_particle							= myroundint(current_particle_counter);

			current_particle_counter += particles_per_job;
			if (current_particle_counter > number_of_particles  || counter == number_of_refinement_jobs - 1) current_particle_counter = number_of_particles;

			long	 last_particle							= myroundint(current_particle_counter);
			current_particle_counter++;



			float percent_used;
			float percent_multiplier;

			percent_multiplier = 2.5 + (3.5 - 2.5) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));
			percent_used = current_percent_used * percent_multiplier;
			if (percent_used > 100.0f) percent_used = 100.0f;
			percent_used *= 0.01f;

			// overides above
			//percent_used							= current_percent_used / 100.0;

#ifdef DEBUG
			wxString output_parameter_file = wxString::Format("/tmp/output_par_%li_%li.par", first_particle, last_particle);
#else
			wxString output_parameter_file = "/dev/null";
#endif

			// for now we take the paramters of the first image!!!!

			float 	 pixel_size								= active_pixel_size;//active_refinement_package->contained_particles[0].pixel_size;
			float    voltage_kV								= active_refinement_package->contained_particles[0].microscope_voltage;
			float 	 spherical_aberration_mm				= active_refinement_package->contained_particles[0].spherical_aberration;
			float    amplitude_contrast						= active_refinement_package->contained_particles[0].amplitude_contrast;
			float	 molecular_mass_kDa						= active_refinement_package->estimated_particle_weight_in_kda;

			float    mask_radius							= active_global_mask_radius;

			float    inner_mask_radius						= active_inner_mask_radius;

			float low_resolution_limit = active_refinement_package->estimated_particle_size_in_angstroms;
			//if (low_resolution_limit > 100.00) low_resolution_limit = 100.00;

			float    high_resolution_limit					= current_high_res_limit;
			float	 signed_CC_limit						= 0.0;
			float	 classification_resolution_limit		= 8.0;
//			float    mask_radius_search						= input_refinement->resolution_statistics_box_size * 0.45 * input_refinement->resolution_statistics_pixel_size;
			float    mask_radius_search						= mask_radius;
			float	 high_resolution_limit_search			= current_high_res_limit;
			float	 angular_step							= CalculateAngularStep(current_high_res_limit, 75.0f);
			//my_parent->WriteInfoText(wxString::Format("angular sampling = %f degrees", angular_step));

		//	if (angular_step < 30.0) angular_step = 30.0;

			int		 best_parameters_to_keep;

			//if (number_of_rounds_run < 5) best_parameters_to_keep = 20;
			//else best_parameters_to_keep = -10000;
			best_parameters_to_keep = -10000;

			float	 max_search_x							= active_search_range_x;
			float	 max_search_y							= active_search_range_y;
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

		/*	if (number_of_rounds_run == 0 && number_of_starts_run == 0)
			{
				refine_x_shift = false;
				refine_y_shift = false;
			}*/

			bool calculate_matching_projections				= false;
			bool apply_2d_masking							= false;
			bool ctf_refinement								= false;
			bool invert_contrast							= active_refinement_package->stack_has_white_protein;

			bool normalize_particles;

			if (stack_has_been_precomputed == true) normalize_particles = false;
			else normalize_particles = true;

			bool exclude_blank_edges = false;
			bool normalize_input_3d;

			if (active_should_apply_blurring == true) normalize_input_3d = false;
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

	output_refinement->resolution_statistics_box_size = input_refinement->resolution_statistics_box_size;
	output_refinement->resolution_statistics_pixel_size = input_refinement->resolution_statistics_pixel_size;

	// launch a controller

	current_job_starttime = time(NULL);
	time_of_last_update = current_job_starttime;

	my_parent->WriteBlueText(wxString::Format(wxT("Running refinement round %2i of %2i (%.2f  Å / %.2f %%) - Start %2i of %2i \n"), number_of_rounds_run + 1, number_of_rounds_to_run, current_high_res_limit, current_percent_used, number_of_starts_run + 1, number_of_starts_to_run));
	current_job_id = main_frame->job_controller.AddJob(my_parent, active_refinement_run_profile.manager_command, active_refinement_run_profile.gui_address);
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



void AbInitioManager::SetupPrepareStackJob()
{
	int class_counter;
	float particles_per_job;
	int number_of_refinement_processes = active_refinement_run_profile.ReturnTotalJobs();
	int	number_of_refinement_jobs = number_of_refinement_processes - 1;

	int	number_of_particles = active_refinement_package->contained_particles.GetCount();
//	if (number_of_particles < number_of_refinement_jobs) particles_per_job = 1.0f;
//	else particles_per_job = float(number_of_particles) / float(number_of_refinement_jobs);
	if (number_of_particles - number_of_refinement_jobs < number_of_refinement_jobs) particles_per_job = 1;
	else particles_per_job = float(number_of_particles - number_of_refinement_jobs) / float(number_of_refinement_jobs);

	my_parent->my_job_package.Reset(active_refinement_run_profile, "prepare_stack", number_of_refinement_jobs);

	float binning_factor = (active_end_res / 2.0f) / active_refinement_package->contained_particles[0].pixel_size;
	float current_particle_counter = 1.0f;

	for (int counter = 0; counter < number_of_refinement_jobs; counter++)
	{

		wxString input_particle_images 				= active_refinement_package->stack_filename;
		wxString output_particle_images 			= main_frame->ReturnStartupScratchDirectory() + "/temp_stack.mrc";
		float pixel_size				          	= active_refinement_package->contained_particles[0].pixel_size;;
		float  mask_radius                          = active_global_mask_radius;
		bool resample_box							= true;
		int	 wanted_output_box_size 				= ReturnClosestFactorizedUpper(ReturnSafeBinnedBoxSize(active_refinement_package->stack_box_size, binning_factor), 3, true);
		bool process_a_subset						= true;
		int	 first_particle							= myroundint(current_particle_counter);

		current_particle_counter += particles_per_job;
		if (current_particle_counter > number_of_particles  || counter == number_of_refinement_jobs - 1) current_particle_counter = number_of_particles;

		int	 last_particle							= myroundint(current_particle_counter);
		current_particle_counter++;

		wxPrintf("1st = %i, last = %i\n", first_particle, last_particle);

		my_parent->my_job_package.AddJob("ttffbibii",	input_particle_images.ToUTF8().data(),
													output_particle_images.ToUTF8().data(),
													pixel_size,
													mask_radius,
													resample_box,
													wanted_output_box_size,
													process_a_subset,
													first_particle,
													last_particle);

	}

}

void AbInitioManager::RunPrepareStackJob()
{
	running_job_type = PREPARE_STACK;
	number_of_received_particle_results = 0;

	// launch a controller

	current_job_starttime = time(NULL);
	time_of_last_update = current_job_starttime;

	my_parent->WriteBlueText("Preparing Input Stack...");
	current_job_id = main_frame->job_controller.AddJob(my_parent, active_refinement_run_profile.manager_command, active_refinement_run_profile.gui_address);
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

void AbInitioManager::SetupAlignSymmetryJob()
{

	int class_counter;
	float start_angle = -90.0f;
	float end_angle = 90.f;
	float wanted_intial_angular_step = 4.0f;

	float current_start_angle;
	float angle_step;

	int number_of_refinement_processes= active_refinement_run_profile.ReturnTotalJobs() ;
	int	number_of_refinement_jobs_per_class = (number_of_refinement_processes - 1) / active_refinement_package->number_of_classes;

	if (number_of_refinement_jobs_per_class < 1)
	{
		number_of_refinement_jobs_per_class = 1;
		angle_step = end_angle - start_angle;

	}
	else
	{
		angle_step = (end_angle - start_angle) / float(number_of_refinement_jobs_per_class - 1);
	}

	// i seem to be calculating it kind of wrong above, directy count it below (this is nasty, and shouldn't be necessary, but is quick)

	number_of_refinement_jobs_per_class = 0;
	for (current_start_angle = start_angle; current_start_angle <= end_angle; current_start_angle += angle_step )
	{
		number_of_refinement_jobs_per_class++;
	}

	number_of_expected_results = 0;

	align_sym_best_correlations.Clear();
	align_sym_best_x_rots.Clear();
	align_sym_best_y_rots.Clear();
	align_sym_best_z_rots.Clear();
	align_sym_best_x_shifts.Clear();
	align_sym_best_y_shifts.Clear();
	align_sym_best_z_shifts.Clear();

	//wxPrintf("reset job package with %i\n", number_of_refinement_jobs_per_class * active_refinement_package->number_of_classes);

	my_parent->my_job_package.Reset(active_refinement_run_profile, "align_symmetry", number_of_refinement_jobs_per_class * active_refinement_package->number_of_classes);

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		align_sym_best_correlations.Add(-FLT_MAX);
		align_sym_best_x_rots.Add(0.0f);
		align_sym_best_y_rots.Add(0.0f);
		align_sym_best_z_rots.Add(0.0f);
		align_sym_best_x_shifts.Add(0.0f);
		align_sym_best_y_shifts.Add(0.0f);
		align_sym_best_z_shifts.Add(0.0f);

		for (current_start_angle = start_angle; current_start_angle <= end_angle; current_start_angle += angle_step )
		{
			wxString input_volume_file = current_reference_filenames.Item(class_counter);
			wxString wanted_symmetry = active_refinement_package->symmetry;
			wxString output_volume_file_no_sym = "";
			wxString output_volume_file_with_sym = "";
			float start_angle_for_search = current_start_angle;
			float end_angle_for_search = current_start_angle + angle_step;
			float initial_angular_step = wanted_intial_angular_step;

			my_parent->my_job_package.AddJob("ttttfffi",	input_volume_file.ToUTF8().data(),
															wanted_symmetry.ToUTF8().data(),
															output_volume_file_no_sym.ToUTF8().data(),
															output_volume_file_with_sym.ToUTF8().data(),
															start_angle_for_search,
															end_angle_for_search,
															initial_angular_step,
															class_counter);

			// this is nasty, just copy and paste from align symmetry to count how many results will be expected

			float current_angular_step = initial_angular_step * 5.0f;
			float best_x_this_round = 0.0f;
			float best_y_this_round = 0.0f;
			float best_z_this_round = 0.0f;

			float low_search_limit_x = start_angle_for_search;
			float high_search_limit_x = end_angle_for_search;

			float low_search_limit_y = -90.0f;
			float high_search_limit_y = 90.0f;

			float low_search_limit_z = -90.0f;
			float high_search_limit_z = 90.0f;

			float current_z_angle;
			float current_y_angle;
			float current_x_angle;

			while (current_angular_step > 0.1f)
			{
				current_angular_step /= 5.0f;
				if (current_angular_step < 0.1f) current_angular_step = 0.1f;

				for (current_z_angle = low_search_limit_z; current_z_angle <= high_search_limit_z; current_z_angle += current_angular_step)
				{
					for(current_y_angle = low_search_limit_y; current_y_angle <= high_search_limit_y; current_y_angle += current_angular_step)
					{
						for (current_x_angle = low_search_limit_x; current_x_angle <= high_search_limit_x; current_x_angle += current_angular_step)
						{
							number_of_expected_results++;
						}
					}
				}

				low_search_limit_x = best_x_this_round - current_angular_step;
				high_search_limit_x = best_x_this_round + current_angular_step;

				low_search_limit_y = best_y_this_round - current_angular_step;
				high_search_limit_y = best_y_this_round + current_angular_step;

				low_search_limit_z = best_z_this_round - current_angular_step;
				high_search_limit_z = best_z_this_round + current_angular_step;
			}
		}
	}
}

void AbInitioManager::RunAlignSymmetryJob()
{
	running_job_type = ALIGN_SYMMETRY;
	number_of_received_particle_results = 0;

	// launch a controller

	current_job_starttime = time(NULL);
	time_of_last_update = current_job_starttime;

	if (active_refinement_package->number_of_classes > 1) my_parent->WriteBlueText("Aligning 3D's to symmetry axes...");
	else my_parent->WriteBlueText("Aligning 3D to symmetry axes...");

	current_job_id = main_frame->job_controller.AddJob(my_parent, active_refinement_run_profile.manager_command, active_refinement_run_profile.gui_address);
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
//			my_parent->AngularPlotPanel->SetSymmetryAndNumber(active_refinement_package->symmetry,output_refinement->number_of_particles);
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
	else
	if (running_job_type == PREPARE_STACK)
	{
		//wxPrintf("got result\n");
		number_of_received_particle_results++;
		long current_time = time(NULL);

		if (number_of_received_particle_results == 1)
		{
			current_job_starttime = current_time;
			time_of_last_update = 0;
		}
		else
		if (current_time != time_of_last_update)
		{
			int current_percentage = float(number_of_received_particle_results) / float(output_refinement->number_of_particles) * 100.0;
			time_of_last_update = current_time;
			if (current_percentage > 100) current_percentage = 100;
			my_parent->ProgressBar->SetValue(current_percentage);
			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float((input_refinement->number_of_particles) - number_of_received_particle_results) * seconds_per_job;

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
	if (running_job_type == ALIGN_SYMMETRY)
	{
		number_of_received_particle_results++;
		long current_time = time(NULL);

		if (number_of_received_particle_results == 1)
		{
			current_job_starttime = current_time;
			time_of_last_update = 0;
		}
		else
		if (current_time != time_of_last_update)
		{
			int current_percentage = float(number_of_received_particle_results) / float(number_of_expected_results) * 100.0;
			time_of_last_update = current_time;
			if (current_percentage > 100) current_percentage = 100;
			my_parent->ProgressBar->SetValue(current_percentage);
			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float((number_of_expected_results) - number_of_received_particle_results) * seconds_per_job;

			TimeRemaining time_remaining;

			if (seconds_remaining > 3600) time_remaining.hours = seconds_remaining / 3600;
			else time_remaining.hours = 0;

			if (seconds_remaining > 60) time_remaining.minutes = (seconds_remaining / 60) - (time_remaining.hours * 60);
			else time_remaining.minutes = 0;

			time_remaining.seconds = seconds_remaining - ((time_remaining.hours * 60 + time_remaining.minutes) * 60);
			my_parent->TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_remaining.hours, time_remaining.minutes, time_remaining.seconds));
		}
	}
}


void AbInitioManager::ProcessAllJobsFinished()
{

	// Update the GUI with project timings
	extern MyOverviewPanel *overview_panel;
	overview_panel->SetProjectInfo();

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
		my_parent->active_orth_thread_id = my_parent->next_thread_id;
		my_parent->next_thread_id++;

		if (start_with_reconstruction == true)	result_thread = new OrthDrawerThread(my_parent, current_reference_filenames, "Random Start", stack_bin_factor, active_global_mask_radius / active_pixel_size, my_parent->active_orth_thread_id);
		else
		result_thread = new OrthDrawerThread(my_parent, current_reference_filenames, wxString::Format("Iter. #%i,%i", number_of_starts_run, number_of_rounds_run), stack_bin_factor, active_global_mask_radius / active_pixel_size, my_parent->active_orth_thread_id);

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
	else
	if (running_job_type == PREPARE_STACK)
	{
		// get the box size of the stack so we can set the pixel size etc..

		/*active_pixel_size = event.GetPayload<float>();
		preparing_stack = false;*/
		wxString new_stack_file = main_frame->ReturnStartupScratchDirectory() + "/temp_stack.mrc";
		MRCFile check_file(new_stack_file.ToStdString(), false, true);

		stack_bin_factor = float(active_refinement_package->stack_box_size) / float(check_file.ReturnXSize());
		active_pixel_size = active_refinement_package->contained_particles[0].pixel_size * stack_bin_factor;
		active_stack_filename = new_stack_file;
		stack_has_been_precomputed = true;

		input_refinement->resolution_statistics_pixel_size = active_pixel_size;
		input_refinement->resolution_statistics_box_size = check_file.ReturnXSize();

		SetupReconstructionJob();
		RunReconstructionJob();
	}
	else
	if (running_job_type == ALIGN_SYMMETRY)
	{
		// we have the alignment, we now need to apply it - launch a seperate thread.

		wxArrayString symmetry_filenames;
		wxFileName current_ref_filename;
		wxString current_symmetry_filename;

		for (int class_counter = 0; class_counter < current_reference_filenames.GetCount(); class_counter++)
		{
			current_ref_filename = current_reference_filenames.Item(class_counter);
			current_ref_filename.ClearExt();
			current_symmetry_filename = current_ref_filename.GetFullPath();
			current_symmetry_filename += "_sym.mrc";

			symmetry_filenames.Add(current_symmetry_filename);
		}

		//my_parent->TimeRemainingText->SetLabel("Time Remaining : 000h:00m:01s");
		my_parent->ProgressBar->SetValue(100);

		my_parent->active_sym_thread_id = my_parent->next_thread_id;
		my_parent->next_thread_id++;

		ImposeAlignmentAndSymmetryThread *symmetry_thread = new ImposeAlignmentAndSymmetryThread(my_parent, current_reference_filenames, symmetry_filenames, align_sym_best_x_rots, align_sym_best_y_rots, align_sym_best_z_rots, align_sym_best_x_shifts, align_sym_best_y_shifts, align_sym_best_z_shifts, active_refinement_package->symmetry, my_parent->active_sym_thread_id);

		if ( symmetry_thread->Run() != wxTHREAD_NO_ERROR )
		{
			my_parent->WriteErrorText("Error: Cannot start symmetry imposition thread, things are about to go wrong!");
			delete symmetry_thread;
		}
		else
		{
			current_reference_filenames = symmetry_filenames;
			// turn on symmetry
			apply_symmetry = true;
			// recalculate percent used.
			current_percent_used = symmetry_start_percent_used + (symmetry_end_percent_used - symmetry_start_percent_used) * (float(number_of_rounds_run) / float(number_of_rounds_to_run - 1));

			return; // just return, we will startup again when the thread finishes.
		}
	}
}

void AbInitio3DPanel::OnOrthThreadComplete(ReturnProcessedImageEvent& my_event)
{
	// in theory the long data should contain a pointer to wxPanel that we are going to add to the notebook..

	Image *new_image = my_event.GetImage();

	if (my_event.GetInt() == active_orth_thread_id)
	{
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
	else
	{
		wxPrintf("No %i, not %i\n", my_event.GetInt(), active_orth_thread_id);
		delete new_image;
	}
}


void AbInitio3DPanel::OnMaskerThreadComplete(wxThreadEvent& my_event)
{
	if (my_event.GetInt() == active_mask_thread_id) my_abinitio_manager.OnMaskerThreadComplete();
}


void AbInitioManager::OnMaskerThreadComplete()
{
	//my_parent->WriteInfoText("Masking Finished");
	SetupRefinementJob();
	RunRefinementJob();
}

void AbInitioManager::DoMasking()
{
	// right now do nothing. Take out event if changing back to thread.
//	MyDebugAssertTrue(my_parent->AutoMaskYesRadio->GetValue() == true, "DoMasking called, when masking not ticked!");
//	wxThreadEvent *my_thread_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_COMPLETED);
//	wxQueueEvent(my_parent, my_thread_event);

	//my_parent->WriteInfoText("Doing Masking...");

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

	my_parent->active_mask_thread_id = my_parent->next_thread_id;
	my_parent->next_thread_id++;

	AutoMaskerThread *mask_thread = new AutoMaskerThread(my_parent, current_reference_filenames, masked_filenames, input_refinement->resolution_statistics_pixel_size, active_global_mask_radius, my_parent->active_mask_thread_id );

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

void AbInitio3DPanel::OnImposeSymmetryThreadComplete(wxThreadEvent& event)
{
	if (event.GetInt() == active_sym_thread_id)
	{
		if (my_abinitio_manager.active_should_automask == true)
		{
			my_abinitio_manager.DoMasking();
		}
		else
		{
			my_abinitio_manager.SetupRefinementJob();
			my_abinitio_manager.RunRefinementJob();
		}
	}

}

void AbInitio3DPanel::OnVolumeResampled(ReturnProcessedImageEvent& my_event)
{
	// in theory the long data should contain a pointer to wxPanel that we are going to add to the notebook..

	Image *new_image = my_event.GetImage();
	MRCFile output_file;
	number_of_resampled_volumes_recieved++;
	wxString current_output_filename;

	if (new_image != NULL)
	{
		current_output_filename = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/startup_volume_%li_%i.mrc", current_startup_id, my_event.GetInt());
		output_file.OpenFile(current_output_filename.ToStdString(), true);

		new_image->QuickAndDirtyWriteSlices(current_output_filename.ToStdString(), 1, new_image->logical_z_dimension);
		new_image->WriteSlices(&output_file,1,new_image->logical_z_dimension);
		output_file.SetPixelSize(my_abinitio_manager.active_refinement_package->contained_particles[0].pixel_size);

		EmpiricalDistribution density_distribution;
		new_image->UpdateDistributionOfRealValues(&density_distribution);
		output_file.SetDensityStatistics(density_distribution.GetMinimum(), density_distribution.GetMaximum(), density_distribution.GetSampleMean(), sqrtf(density_distribution.GetSampleVariance()));
		output_file.CloseFile();
	}

	if (number_of_resampled_volumes_recieved == my_abinitio_manager.input_refinement->number_of_classes)
	{
		VolumeAsset temp_asset;
		main_frame->current_project.database.Begin();
		main_frame->current_project.database.BeginVolumeAssetInsert();

		wxArrayLong volume_asset_ids;

		for (int class_counter = 0; class_counter < my_abinitio_manager.input_refinement->number_of_classes; class_counter++)
		{
			temp_asset.reconstruction_job_id = -1;
			temp_asset.pixel_size = my_abinitio_manager.active_refinement_package->contained_particles[0].pixel_size;
			temp_asset.x_size = new_image->logical_x_dimension;
			temp_asset.y_size = new_image->logical_y_dimension;
			temp_asset.z_size = new_image->logical_z_dimension;
			temp_asset.asset_id = volume_asset_panel->current_asset_number;
			volume_asset_ids.Add(temp_asset.asset_id);
			temp_asset.asset_name = wxString::Format("Volume From Startup #%li - Class #%i", current_startup_id, my_event.GetInt());

			current_output_filename = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/startup_volume_%li_%i.mrc", current_startup_id, class_counter + 1);
			temp_asset.filename = current_output_filename;
			volume_asset_panel->AddAsset(&temp_asset);
			main_frame->current_project.database.AddNextVolumeAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath(), temp_asset.reconstruction_job_id, temp_asset.pixel_size, temp_asset.x_size, temp_asset.y_size, temp_asset.z_size);
		}

		main_frame->current_project.database.EndVolumeAssetInsert();

		// now add the details of the startup job..

		main_frame->current_project.database.AddStartupJob(current_startup_id, my_abinitio_manager.input_refinement->refinement_package_asset_id, wxString::Format("Refinement #%li", current_startup_id), my_abinitio_manager.number_of_starts_run, my_abinitio_manager.number_of_rounds_to_run, InitialResolutionLimitTextCtrl->ReturnValue(), FinalResolutionLimitTextCtrl->ReturnValue(), AutoMaskYesRadio->GetValue(), AutoPercentUsedYesRadio->GetValue(), my_abinitio_manager.start_percent_used, my_abinitio_manager.end_percent_used, my_abinitio_manager.active_global_mask_radius,  ApplyBlurringYesRadioButton->GetValue(), SmoothingFactorTextCtrl->ReturnValue(), volume_asset_ids);
		main_frame->current_project.database.Commit();

		FinishButton->Enable(true);
		main_frame->DirtyVolumes();
	}

	if (new_image != NULL) delete new_image;

}

wxThread::ExitCode ResampleVolumeThread::Entry()
{
	ImageFile input_file(input_volume.ToStdString());
	Image *resampled_volume = new Image;

	ReturnProcessedImageEvent *finished_event = new ReturnProcessedImageEvent(wxEVT_RESAMPLE_VOLUME_EVENT); // for sending back the panel

	resampled_volume->ReadSlices(&input_file, 1, input_file.ReturnNumberOfSlices());
	resampled_volume->ForwardFFT();
	resampled_volume->Resize(box_size, box_size, box_size);
	resampled_volume->BackwardFFT();

	finished_event->SetImage(resampled_volume);
	finished_event->SetInt(class_number);

	wxQueueEvent(parent_window, finished_event);
}

wxThread::ExitCode ImposeAlignmentAndSymmetryThread::Entry()
{
	Image input_volume;

	ImageFile input_file;
	MRCFile output_file;

	RotationMatrix current_matrix;
	RotationMatrix inverse_matrix;

	for (int class_counter = 0; class_counter < input_volumes.GetCount(); class_counter++)
	{
		input_file.OpenFile(input_volumes.Item(class_counter).ToStdString(), false);
		input_volume.ReadSlices(&input_file, 1, input_file.ReturnNumberOfSlices());
		input_file.CloseFile();

		current_matrix.SetToRotation(x_rots[class_counter], y_rots[class_counter], z_rots[class_counter]);
		inverse_matrix = current_matrix.ReturnTransposed();

		//wxPrintf("Imposing %f, %f, %f - %f, %f, %f\n", x_rots[class_counter], y_rots[class_counter], z_rots[class_counter], x_shifts[class_counter], y_shifts[class_counter], z_shifts[class_counter]);
		input_volume.Rotate3DThenShiftThenApplySymmetry(inverse_matrix, x_shifts[class_counter], y_shifts[class_counter], z_shifts[class_counter], input_volume.logical_x_dimension / 2.0f, symmetry);

		output_file.OpenFile(output_volumes.Item(class_counter).ToStdString(), true);
		input_volume.WriteSlices(&output_file, 1, input_volume.logical_z_dimension);
		output_file.CloseFile();
	}

	// send event to say we are done..

	wxThreadEvent *my_thread_event = new wxThreadEvent(wxEVT_COMMAND_IMPOSESYMMETRY_DONE);
	my_thread_event->SetInt(thread_id);
	wxQueueEvent(parent_window, my_thread_event);
}
