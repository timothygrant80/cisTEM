#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMovieAssetPanel *movie_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyMainFrame *main_frame;

MyAlignMoviesPanel::MyAlignMoviesPanel( wxWindow* parent )
:
AlignMoviesPanel( parent )
{

		// Create a mpFXYVector layer for the plot

		mpFXYVector* vectorLayer = new mpFXYVector((""));

	//	accumulated_dose_data.push_back(0);
//		accumulated_dose_data.push_back(5);
//		accumulated_dose_data.push_back(10);
//		accumulated_dose_data.push_back(15);
		//accumulated_dose_data.push_back(20);

		//average_movement_data.push_back(10);
		//average_movement_data.push_back(6);
		//average_movement_data.push_back(4);
		//average_movement_data.push_back(2);
		//average_movement_data.push_back(1);

		vectorLayer->SetData(accumulated_dose_data, average_movement_data);
		vectorLayer->SetContinuity(true);
		wxPen vectorpen(*wxBLUE, 2, wxSOLID);
		vectorLayer->SetPen(vectorpen);
		vectorLayer->SetDrawOutsideMargins(false);


		wxFont graphFont(11, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);

		plot_window = new mpWindow( GraphPanel, -1, wxPoint(0,0), wxSize(100, 100), wxSUNKEN_BORDER );

		mpScaleX* xaxis = new mpScaleX(wxT("Accumulated Exposure  (e¯/Å²)"), mpALIGN_BOTTOM, true, mpX_NORMAL);
	    mpScaleY* yaxis = new mpScaleY(wxT("Average Movement (Å)"), mpALIGN_LEFT, true);

	    xaxis->SetFont(graphFont);
	    yaxis->SetFont(graphFont);
	    xaxis->SetDrawOutsideMargins(false);
	    yaxis->SetDrawOutsideMargins(false);

	    plot_window->SetMargins(30, 30, 60, 100);

	    plot_window->AddLayer(xaxis);
	    plot_window->AddLayer(yaxis);
		plot_window->AddLayer(vectorLayer);
		//plot_window->AddLayer( nfo = new mpInfoCoords(wxRect(500,20,10,10), wxTRANSPARENT_BRUSH));

	    GraphSizer->Add(plot_window, 1, wxEXPAND );

	    plot_window->EnableDoubleBuffer(true);
//   	    plot_window->SetMPScrollbars(false);
   	    plot_window->EnableMousePanZoom(false);
	    plot_window->Fit();

	    // Set variables

	    show_expert_options = false;

	    // Fill combo box..

	    FillGroupComboBox();

	    my_job_id = -1;
	    running_job = false;

	    SetInfo();

}

void MyAlignMoviesPanel::OnInfoURL(wxTextUrlEvent& event)
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

void MyAlignMoviesPanel::SetInfo()
{
	#include "icons/dlp_alignment.cpp"
	wxBitmap alignment_bmp = wxBITMAP_PNG_FROM_DATA(dlp_alignment);

	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("Movie Alignment"));
	InfoText->EndFontSize();
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("Physical drift and beam induced motion (Brilot et al., 2012; Campbell et al., 2012; Li et al., 2013; Scheres, 2014) of the specimen leads to a degradation of information within images, and will ultimately limit the resolution of any reconstruction.  Aligning a movie prior to calculating the sum will prevent a large amount of this degradation and lead to better data.  This panel therefore attempts to align movies based on the Unblur algorithm described in (Grant and Grigorieff, 2015)."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(alignment_bmp);
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("Additionally an exposure weighted sum can be calculated which attempts to maximize the signal-to-noise ratio in the final sums by taking into account the radiation damage the sample has suffered as the movie progresses.  This exposure weighting is described in (Grant and Grigorieff, 2015)."));
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
	InfoText->WriteText(wxT("Input Group : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The group of movie assets that will be aligned.  For each movie within the group, the output will be an image representing the sum of the aligned movie.  This movie will be automatically added to the image assets list."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Run Profile : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The selected run profile will be used to run the job.  The run profile describes how the job should be run (e.g. how many processors should be used, and on which different computers).  Run profiles are set in the Run Profile panel, located under settings."));
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

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Minimum Shift : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("This is the minimum shift that can be applied during the initial refinement stage.  Its purpose is to prevent images aligning to detector artifacts that may be reinforced in the initial sum which is used as the first reference. It is applied only during the first alignment round, and is ignored after that. "));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Maximum Shift : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("This is the maximum shift that can be applied in any single alignment round. Its purpose is to avoid alignment to spurious noise peaks by not considering unreasonably large shifts.  This limit is applied during every alignment round, but only for that round, such that it can be exceeded over a number of successive rounds."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Exposure Filter Sums? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If selected the resulting aligned movie sums will be calculated using the exposure filter as described in Grant and Grigorieff (2015)."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Restore Power? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If selected, and the exposure filter is used to calculate the sum then the sum will be high pass filtered to restore the noise power.  This is essentially the denominator of Eq. 9 in Grant and Grigorieff (2015)."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Termination Threshold : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The frames will be iteratively aligned until either the maximum number of iterations is reached, or if after an alignment round every frame was shifted by less than this threshold."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Max Iterations : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The maximum number of iterations that can be run for the movie alignment. If reached, the alignment will stop and the current best values will be taken."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("B-Factor : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("This B-Factor is applied to the reference sum prior to alignment.  It is intended to low-pass filter the images in order to prevent alignment to spurious noise peaks and detector artifacts."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Mask Central Cross : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If selected, the Fourier transform of the reference will be masked by a cross centred on the origin of the transform. This is intended to reduce the influence of detector artifacts which often have considerable power along the central cross."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Horizontal Mask : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The width of the horizontal line in the central cross mask. It is only used if Mask Central Cross is selected."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Vertical Mask : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The width of the vertical line in the central cross mask. It is only used if Mask Central Cross is selected."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();
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
	InfoText->WriteText(wxT("Brilot, A.F., Chen, J.Z., Cheng, A., Pan, J., Harrison, S.C., Potter, C.S., Carragher, B., Henderson, R., Grigorieff, N.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2012. Beam-induced motion of vitrified specimen on holey carbon film. J. Struct. Biol. 177, 630–637. "));
	InfoText->BeginURL("http://dx.doi.org/10.1016/j.jsb.2012.02.003");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.1016/j.jsb.2012.02.003"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Campbell, M.G., Cheng, A., Brilot, A.F., Moeller, A., Lyumkis, D., Veesler, D., Pan, J., Harrison, S.C., Potter, C.S., Carragher, B., Grigorieff, N.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2012. Movies of ice-embedded particles enhance resolution in electron cryo-microscopy. Structure 20, 1823–8. "));
	InfoText->BeginURL("http://dx.doi.org/10.1016/j.str.2012.08.026");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.1016/j.str.2012.08.026"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Grant, T., Grigorieff, N.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2015. Measuring the optimal exposure for single particle cryo-EM using a 2.6 Å reconstruction of rotavirus VP6. Elife 4, e06980. "));
	InfoText->BeginURL("http://dx.doi.org/10.7554/eLife.06980");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.7554/eLife.06980"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Li, X., Mooney, P., Zheng, S., Booth, C.R., Braunfeld, M.B., Gubbens, S., Agard, D.A., Cheng, Y.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2013. Electron counting and beam-induced motion correction enable near-atomic-resolution single-particle cryo-EM. Nat. Methods 10, 584–90. "));
	InfoText->BeginURL("http://dx.doi.org/10.1038/nmeth.2472");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.1038/nmeth.2472"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Scheres, S.H.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" Beam-induced motion correction for sub-megadalton cryo-EM particles. Elife 3, e03665. "));
	InfoText->BeginURL("http://dx.doi.org/10.7554/eLife.03665");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.7554/eLife.03665"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();

	InfoText->EndSuppressUndo();


}

void MyAlignMoviesPanel::OnExpertOptionsToggle( wxCommandEvent& event )
{
	if (show_expert_options == true)
	{
		show_expert_options = false;
		ExpertPanel->Show(false);
		Layout();
	}
	else
	{
		show_expert_options = true;
		ExpertPanel->Show(true);
		Layout();
	}
}

void MyAlignMoviesPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	// are there enough members in the selected group.
	if (main_frame->current_project.is_open == false)
	{
		RunProfileComboBox->Enable(false);
		GroupComboBox->Enable(false);
		ExpertToggleButton->Enable(false);
		StartAlignmentButton->Enable(false);
	}
	else
	{
		//Enable(true);


		if (running_job == false)
		{
			RunProfileComboBox->Enable(true);
			GroupComboBox->Enable(true);
			ExpertToggleButton->Enable(true);

			if (RunProfileComboBox->GetCount() > 0)
			{
				if (movie_asset_panel->ReturnGroupSize(GroupComboBox->GetCurrentSelection()) > 0 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(RunProfileComboBox->GetSelection()) > 1)
				{
					StartAlignmentButton->Enable(true);
				}
				else StartAlignmentButton->Enable(false);
			}
			else
			{
				StartAlignmentButton->Enable(false);
			}
		}
		else
		{
			ExpertToggleButton->Enable(false);
			GroupComboBox->Enable(false);
			RunProfileComboBox->Enable(false);
			//StartAlignmentButton->SetLabel("Stop Job");
			//StartAlignmentButton->Enable(true);
		}
	}




}

void MyAlignMoviesPanel::FillGroupComboBox()
{

	GroupComboBox->Freeze();
	GroupComboBox->Clear();

	for (long counter = 0; counter < movie_asset_panel->ReturnNumberOfGroups(); counter++)
	{
		GroupComboBox->Append(movie_asset_panel->ReturnGroupName(counter) +  " (" + wxString::Format(wxT("%li"), movie_asset_panel->ReturnGroupSize(counter)) + ")");

	}

	GroupComboBox->SetSelection(0);

	GroupComboBox->Thaw();
}

void MyAlignMoviesPanel::Refresh()
{
	FillGroupComboBox();
	FillRunProfileComboBox();
}

void MyAlignMoviesPanel::FillRunProfileComboBox()
{
	RunProfileComboBox->Freeze();
	RunProfileComboBox->Clear();

	for (long counter = 0; counter < run_profiles_panel->run_profile_manager.number_of_run_profiles; counter++)
	{
		RunProfileComboBox->Append(run_profiles_panel->run_profile_manager.ReturnProfileName(counter) + wxString::Format(" (%li)", run_profiles_panel->run_profile_manager.ReturnTotalJobs(counter)));
	}

	if (RunProfileComboBox->GetCount() > 0) RunProfileComboBox->SetSelection(0);
	RunProfileComboBox->Thaw();

}

void MyAlignMoviesPanel::StartAlignmentClick( wxCommandEvent& event )
{
	// Package the job details..

	long counter;
	long number_of_jobs = movie_asset_panel->ReturnGroupSize(GroupComboBox->GetCurrentSelection()); // how many movies in the selected group..
	long number_of_processes = 2; // NEED TO DO THIS PROPERLY LATER

	bool ok_number_conversion;

	double minimum_shift;
	double maximum_shift;

	bool should_dose_filter;
	bool should_restore_power;

	double termination_threshold;
	int max_iterations;
	int bfactor;

	bool should_mask_central_cross;
	int horizontal_mask;
	int vertical_mask;

	float current_pixel_size;

	std::string current_filename;
	wxString output_filename;

	// read the options form the gui..

	// min_shift
	ok_number_conversion = minimum_shift_text->GetLineText(0).ToDouble(&minimum_shift);
	MyDebugAssertTrue(ok_number_conversion == true, "Couldn't convert minimum shift text to number!");

	// max_shift
	ok_number_conversion = maximum_shift_text->GetLineText(0).ToDouble(&maximum_shift);
	MyDebugAssertTrue(ok_number_conversion == true, "Couldn't convert maximum shift text to number!");

	// dose filter
	should_dose_filter = dose_filter_checkbox->IsChecked();

	// restore power
	should_restore_power = restore_power_checkbox->IsChecked();

	// termination threshold
	ok_number_conversion = termination_threshold_text->GetLineText(0).ToDouble(&termination_threshold);
	MyDebugAssertTrue(ok_number_conversion == true, "Couldn't convert termination threshold text to number!");

	// max iterations

	max_iterations = max_iterations_spinctrl->GetValue();

	// b-factor

	bfactor = bfactor_spinctrl->GetValue();

	// mask central cross

	should_mask_central_cross = mask_central_cross_checkbox->IsChecked();

	// horizontal mask

	horizontal_mask = horizontal_mask_spinctrl->GetValue();

	// vertical mask

	vertical_mask = vertical_mask_spinctrl->GetValue();


	my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()], "unblur", number_of_jobs);

	for (counter = 0; counter < number_of_jobs; counter++)
	{
		// job is :-
		//
		// Input Filename (string)
		// Output Filename (string)
		// Minimum shift in angstroms (float)
		// Maximum Shift in angstroms (float)
		// Dose filter Sums? (bool)
		// Restore power? (bool)
		// Termination threshold in angstroms (float)
		// Max iterations (int)
		// B-Factor in angstroms (float)
		// Mask central cross (bool)
		// Horizontal mask size in pixels (int)
		// Vertical mask size in pixels (int)

		// OUTPUT FILENAME, MUST BE SET PROPERLY

		output_filename = movie_asset_panel->ReturnAssetLongFilename(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter));
		output_filename.Replace(".mrc", "_ali.mrc", false);

		current_filename = movie_asset_panel->ReturnAssetLongFilename(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter)).ToStdString();
		current_pixel_size = movie_asset_panel->ReturnAssetPixelSize(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter));
		my_job_package.AddJob("ssfffbbfifbii", current_filename.c_str(), output_filename.ToUTF8().data(), current_pixel_size, float(minimum_shift), float(maximum_shift), should_dose_filter, should_restore_power, float(termination_threshold), max_iterations, float(bfactor), should_mask_central_cross, horizontal_mask, vertical_mask);
	}

	// launch a controller

	my_job_id = main_frame->job_controller.AddJob(this, run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()].manager_command);

	if (my_job_id != -1)
	{

		StartPanel->Show(false);
		ProgressPanel->Show(true);


		ExpertPanel->Show(false);
		InfoPanel->Show(false);
		OutputTextPanel->Show(true);
		GraphPanel->Show(true);

		ExpertToggleButton->Enable(false);
		GroupComboBox->Enable(false);
		Layout();

		running_job = true;
		my_job_tracker.StartTracking(my_job_package.number_of_jobs);

	}
	ProgressBar->Pulse();

}

void MyAlignMoviesPanel::FinishButtonClick( wxCommandEvent& event )
{
	ProgressBar->SetValue(0);
	TimeRemainingText->SetLabel("Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
	FinishButton->Show(false);

	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	output_textctrl->Clear();
	GraphPanel->Show(false);
	InfoPanel->Show(true);

	if (show_expert_options == true) ExpertPanel->Show(true);
	else ExpertPanel->Show(false);
	running_job = false;
	Layout();



}

void MyAlignMoviesPanel::TerminateButtonClick( wxCommandEvent& event )
{

}

void MyAlignMoviesPanel::WriteInfoText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void MyAlignMoviesPanel::WriteErrorText(wxString text_to_write)
{
	 output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
	 output_textctrl->AppendText(text_to_write);

	 if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}


void MyAlignMoviesPanel::OnJobSocketEvent(wxSocketEvent& event)
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

	    	  MyDebugPrint("Sending Job Details...");
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
	      if (memcmp(socket_input_buffer, socket_job_finished, SOCKET_CODE_SIZE) == 0) // identification
	 	  {
	 		 // which job is finished?

	 		 int finished_job;
	 		 sock->ReadMsg(&finished_job, 4);
	 		 my_job_tracker.MarkJobFinished();

	 		 if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();
	 		 //WriteInfoText(wxString::Format("Job %i has finished!", finished_job));
	 	  }
	      else
		  if (memcmp(socket_input_buffer, socket_number_of_connections, SOCKET_CODE_SIZE) == 0) // identification
		  {
			  // how many connections are there?

			  int number_of_connections;
              sock->ReadMsg(&number_of_connections, 4);

              my_job_tracker.AddConnection();

              ProgressBar->Pulse();

              // send the info to the gui

		 	  if (number_of_connections == my_job_package.my_profile.ReturnTotalJobs()) WriteInfoText(wxString::Format("All %i jobs are running and connected.\n\n", number_of_connections));
		  }
	      else
		  if (memcmp(socket_input_buffer, socket_all_jobs_finished, SOCKET_CODE_SIZE) == 0) // identification
		  {
			  WriteInfoText("All Jobs have finished.");
			  ProgressBar->SetValue(100);
			  TimeRemainingText->SetLabel("Remaining : All Done!");
			  CancelAlignmentButton->Show(false);
			  FinishButton->Show(true);
			  ProgressPanel->Layout();
			  //running_job = false;

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


void MyAlignMoviesPanel::UpdateProgressBar()
{
	TimeRemaining time_left = my_job_tracker.ReturnRemainingTime();
	ProgressBar->SetValue(my_job_tracker.ReturnPercentCompleted());

	TimeRemainingText->SetLabel(wxString::Format("Remaining : %ih:%im:%is", time_left.hours, time_left.minutes, time_left.seconds));
}
