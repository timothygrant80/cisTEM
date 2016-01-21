#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMovieAssetPanel *movie_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyMainFrame *main_frame;

MyAlignMoviesPanel::MyAlignMoviesPanel( wxWindow* parent )
:
AlignMoviesPanel( parent )
{

		buffered_results = NULL;

		// Create a mpFXYVector layer for the plot

		current_x_shift_vector_layer = new mpFXYVector((""));
		current_y_shift_vector_layer = new mpFXYVector((""));
		//average_shift_vector_layer = new mpFXYVector((""));


		/*current_accumulated_dose_data.push_back(0);
		current_accumulated_dose_data.push_back(5);
		current_accumulated_dose_data.push_back(10);
		current_accumulated_dose_data.push_back(15);
		current_accumulated_dose_data.push_back(20);

		current_movement_data.push_back(10);
		current_movement_data.push_back(6);
		current_movement_data.push_back(4);
		current_movement_data.push_back(2);
		current_movement_data.push_back(1);*/

		wxPen vectorpen(*wxBLUE, 2, wxSOLID);
		wxPen redvectorpen(*wxRED, 2, wxSOLID);

		//current_x_shift_vector_layer->SetData(current_accumulated_dose_data, current_movement_data);
		current_x_shift_vector_layer->SetContinuity(true);
		current_x_shift_vector_layer->SetPen(vectorpen);
		current_x_shift_vector_layer->SetDrawOutsideMargins(false);

		current_y_shift_vector_layer->SetContinuity(true);
		current_y_shift_vector_layer->SetPen(redvectorpen);
		current_y_shift_vector_layer->SetDrawOutsideMargins(false);

		//average_shift_vector_layer->SetData(current_accumulated_dose_data, current_movement_data);
		//average_shift_vector_layer->SetContinuity(true);
		//average_shift_vector_layer->SetPen(vectorpen);
		//average_shift_vector_layer->SetDrawOutsideMargins(false);

		wxFont graphFont(11, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);

		current_plot_window = new mpWindow( GraphPanel, -1, wxPoint(0,0), wxSize(100, 100), wxSUNKEN_BORDER );
		//average_plot_window = new mpWindow( GraphPanel, -1, wxPoint(0,0), wxSize(100, 100), wxSUNKEN_BORDER );

		//mpScaleX* average_xaxis = new mpScaleX(wxT("Accum. Exposure  (e¯/Å²)"), mpALIGN_BOTTOM, true, mpX_NORMAL);
	    //mpScaleY* average_yaxis = new mpScaleY(wxT("Average Movement (Å)"), mpALIGN_LEFT, true);

	    mpScaleX* current_xaxis = new mpScaleX(wxT("Accumulated Exposure  (e¯/Å²)"), mpALIGN_BOTTOM, true, mpX_NORMAL);
	    mpScaleY* current_yaxis = new mpScaleY(wxT("Shifts (Å)"), mpALIGN_LEFT, true);

	   // legend = new mpInfoLegend(wxRect(400,30,100,50));

	   // average_xaxis->SetFont(graphFont);
	    //average_yaxis->SetFont(graphFont);
	    //average_xaxis->SetDrawOutsideMargins(false);
	    //average_yaxis->SetDrawOutsideMargins(false);

	    current_xaxis->SetFont(graphFont);
	    current_yaxis->SetFont(graphFont);
	    current_xaxis->SetDrawOutsideMargins(false);
	    current_yaxis->SetDrawOutsideMargins(false);

	    current_plot_window->SetMargins(30, 30, 60, 80);
	    //average_plot_window->SetMargins(20, 40, 50, 50);

	    current_plot_window->AddLayer(current_xaxis);
	    current_plot_window->AddLayer(current_yaxis);
		current_plot_window->AddLayer(current_x_shift_vector_layer);
		current_plot_window->AddLayer(current_y_shift_vector_layer);
		//current_plot_window->AddLayer(legend);
		//plot_window->AddLayer( nfo = new mpInfoCoords(wxRect(500,20,10,10), wxTRANSPARENT_BRUSH));

	    //average_plot_window->AddLayer(average_xaxis);
	    //average_plot_window->AddLayer(average_yaxis);
		//average_plot_window->AddLayer(average_shift_vector_layer);

	    GraphSizer->Add(current_plot_window, 1, wxEXPAND );
	  //  GraphSizer->Add(average_plot_window, 1, wxEXPAND );

	    current_plot_window->EnableDoubleBuffer(true);
//   	    plot_window->SetMPScrollbars(false);
   	    current_plot_window->EnableMousePanZoom(false);
	    current_plot_window->Fit();

	    // Set variables

	    show_expert_options = false;

	    // Fill combo box..

	    FillGroupComboBox();

	    my_job_id = -1;
	    running_job = false;

	    graph_is_hidden = true;

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
	int old_selection = 0;

	// get the current selection..

	if (RunProfileComboBox->GetCount() > 0) old_selection = RunProfileComboBox->GetSelection();

	// refresh..

	RunProfileComboBox->Freeze();
	RunProfileComboBox->Clear();

	for (long counter = 0; counter < run_profiles_panel->run_profile_manager.number_of_run_profiles; counter++)
	{
		RunProfileComboBox->Append(run_profiles_panel->run_profile_manager.ReturnProfileName(counter) + wxString::Format(" (%li)", run_profiles_panel->run_profile_manager.ReturnTotalJobs(counter)));
	}

	if (RunProfileComboBox->GetCount() > 0)
	{
		if (RunProfileComboBox->GetCount() >= old_selection) RunProfileComboBox->SetSelection(old_selection);
		else RunProfileComboBox->SetSelection(0);

	}
	RunProfileComboBox->Thaw();

}

void MyAlignMoviesPanel::StartAlignmentClick( wxCommandEvent& event )
{

	MyDebugAssertTrue(buffered_results == NULL, "Error: buffered results not null")

	// Package the job details..

	long counter;
	long number_of_jobs = movie_asset_panel->ReturnGroupSize(GroupComboBox->GetCurrentSelection()); // how many movies in the selected group..

	bool ok_number_conversion;

	double minimum_shift;
	double maximum_shift;

	bool should_dose_filter;
	bool should_restore_power;

	double termination_threshold;
	int max_iterations;
	int bfactor;
	int number_of_processes;

	bool should_mask_central_cross;
	int horizontal_mask;
	int vertical_mask;

	float current_pixel_size;

	float current_acceleration_voltage;
	float current_dose_per_frame;
	float current_pre_exposure;

	time_of_last_graph_update = 0;

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

	// allocate space for the buffered results..

	buffered_results = new JobResult[number_of_jobs];

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
		current_acceleration_voltage = movie_asset_panel->ReturnAssetAccelerationVoltage(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter));
		current_dose_per_frame = movie_asset_panel->ReturnAssetDosePerFrame(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter));
		current_pre_exposure = movie_asset_panel->ReturnAssetPreExposureAmount(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter));


		my_job_package.AddJob("ssfffbbfifbiifff",	current_filename.c_str(),
													output_filename.ToUTF8().data(),
													current_pixel_size,
													float(minimum_shift),
													float(maximum_shift),
													should_dose_filter,
													should_restore_power,
													float(termination_threshold),
													max_iterations,
													float(bfactor),
													should_mask_central_cross,
													horizontal_mask,
													vertical_mask,
													current_acceleration_voltage,
													current_dose_per_frame,
													current_pre_exposure);
	}

	// launch a controller

	my_job_id = main_frame->job_controller.AddJob(this, run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()].gui_address);

	if (my_job_id != -1)
	{
		number_of_processes =  my_job_package.my_profile.ReturnTotalJobs();

		if (number_of_processes >= 100000) length_of_process_number = 6;
		else
		if (number_of_processes >= 10000) length_of_process_number = 5;
		else
		if (number_of_processes >= 1000) length_of_process_number = 4;
		else
		if (number_of_processes >= 100) length_of_process_number = 3;
		else
		if (number_of_processes >= 10) length_of_process_number = 2;
		else
		length_of_process_number = 1;

		if (length_of_process_number == 6) NumberConnectedText->SetLabel(wxString::Format("%6i / %6i processes connected.", 0, number_of_processes));
		else
		if (length_of_process_number == 5) NumberConnectedText->SetLabel(wxString::Format("%5i / %5i processes connected.", 0, number_of_processes));
		else
		if (length_of_process_number == 4) NumberConnectedText->SetLabel(wxString::Format("%4i / %4i processes connected.", 0, number_of_processes));
		else
		if (length_of_process_number == 3) NumberConnectedText->SetLabel(wxString::Format("%3i / %3i processes connected.", 0, number_of_processes));
		else
		if (length_of_process_number == 2) NumberConnectedText->SetLabel(wxString::Format("%2i / %2i processes connected.", 0, number_of_processes));
		else
		NumberConnectedText->SetLabel(wxString::Format("%1i / %1i processes connected.", 0, number_of_processes));

		StartPanel->Show(false);
		ProgressPanel->Show(true);


		ExpertPanel->Show(false);
		InfoPanel->Show(false);
		OutputTextPanel->Show(true);
		//GraphPanel->Show(true);

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
	TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
	FinishButton->Show(false);

	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	output_textctrl->Clear();
	GraphPanel->Show(false);
	graph_is_hidden = true;
	InfoPanel->Show(true);

	if (show_expert_options == true) ExpertPanel->Show(true);
	else ExpertPanel->Show(false);
	running_job = false;
	Layout();



}

void MyAlignMoviesPanel::TerminateButtonClick( wxCommandEvent& event )
{
	// kill the job, this will kill the socket to terminate downstream processes
	// - this will have to be improved when clever network failure is incorporated


	main_frame->job_controller.KillJob(my_job_id);

	WriteInfoText("Terminated Job");
	TimeRemainingText->SetLabel("Time Remaining : Terminated");
	CancelAlignmentButton->Show(false);
	FinishButton->Show(true);
	ProgressPanel->Layout();

	if (buffered_results != NULL)
	{
		delete [] buffered_results;
		buffered_results = NULL;
	}
			  //running_job = false;


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

	 		 if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();
	 		 //WriteInfoText(wxString::Format("Job %i has finished!", finished_job));
	 	  }
	      if (memcmp(socket_input_buffer, socket_job_result, SOCKET_CODE_SIZE) == 0) // identification
	 	  {
	    	  JobResult temp_result;
	    	  temp_result.ReceiveFromSocket(sock);

			  if (temp_result.result_size > 0)
			  {
				 ProcessResult(&temp_result);
			  }
	 	  }
	      else
		  if (memcmp(socket_input_buffer, socket_number_of_connections, SOCKET_CODE_SIZE) == 0) // identification
		  {
			  // how many connections are there?

			  int number_of_connections;
              sock->ReadMsg(&number_of_connections, 4);

              my_job_tracker.AddConnection();

              if (graph_is_hidden == true) ProgressBar->Pulse();

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
			/*  WriteInfoText("All Jobs have finished.");
			  ProgressBar->SetValue(100);
			  TimeRemainingText->SetLabel("Time Remaining : All Done!");
			  CancelAlignmentButton->Show(false);
			  FinishButton->Show(true);
			  ProgressPanel->Layout();
			  //running_job = false;*/

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

void  MyAlignMoviesPanel::ProcessResult(JobResult *result_to_process) // this will have to be overidden in the parent clas when i make it.
{
	int number_of_frames;
	int frame_counter;

	long current_time = time(NULL);


	// mark the job as finished, update progress bar


	float exposure_per_frame = my_job_package.jobs[result_to_process->job_number].arguments[14].ReturnFloatArgument();

	// ok the result should be x-shifts, followed by y-shifts..
	//WriteInfoText(wxString::Format("Job #%i finished.", job_number));

	number_of_frames = result_to_process->result_size / 2;


	if (current_time - time_of_last_graph_update > 1)
	{
		current_x_shift_vector_layer->Clear();
		current_accumulated_dose_data.clear();
		current_x_movement_data.clear();
		current_y_movement_data.clear();

		for (frame_counter = 0; frame_counter < number_of_frames; frame_counter++)
		{
			current_accumulated_dose_data.push_back(double(exposure_per_frame * frame_counter));
			current_x_movement_data.push_back(double(result_to_process->result_data[frame_counter]));
			current_y_movement_data.push_back(double(result_to_process->result_data[frame_counter + number_of_frames]));

			//WriteInfoText(wxString::Format("Frame %i = %f, %f\n", frame_counter, result[frame_counter], result[frame_counter + number_of_frames]));
		}

		current_x_shift_vector_layer->SetData(current_accumulated_dose_data, current_x_movement_data);
		current_y_shift_vector_layer->SetData(current_accumulated_dose_data, current_y_movement_data);
		current_plot_window->Fit();
		current_plot_window->UpdateAll();

		if (graph_is_hidden == true)
		{
			GraphPanel->Show(true);
			Layout();
			graph_is_hidden = false;
		}

		time_of_last_graph_update = current_time;


	}

	my_job_tracker.MarkJobFinished();
	if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();

	// store the results..

	buffered_results[result_to_process->job_number] = result_to_process;

	if (my_job_tracker.total_number_of_finished_jobs == my_job_tracker.total_number_of_jobs)
	{
		// job has really finished, so we can write to the database...

		WriteResultToDataBase();

		if (buffered_results != NULL)
		{
			delete [] buffered_results;
			buffered_results = NULL;
		}

		WriteInfoText("All Jobs have finished.");
		ProgressBar->SetValue(100);
		TimeRemainingText->SetLabel("Time Remaining : All Done!");
		CancelAlignmentButton->Show(false);
		FinishButton->Show(true);
		ProgressPanel->Layout();
	}


}

void MyAlignMoviesPanel::WriteResultToDataBase()
{

	long counter;
	int frame_counter;
	wxString current_table_name;

	// find the current highest alignment number in the database, then increment by one

	int starting_alignment_id = main_frame->current_project.database.ReturnHighestAlignmentID();
	int alignment_id = starting_alignment_id + 1;
	int alignment_job_id =  main_frame->current_project.database.ReturnHighestAlignmentJobID() + 1;

	// loop over all the jobs, and add them..

	main_frame->current_project.database.BeginBatchInsert("MOVIE_ALIGNMENT_LIST", 18, "ALIGNMENT_ID", "DATETIME_OF_RUN", "ALIGNMENT_JOB_ID", "MOVIE_ASSET_ID", "VOLTAGE", "PIXEL_SIZE", "EXPOSURE_PER_FRAME", "PRE_EXPOSURE_AMOUNT", "MIN_SHIFT", "MAX_SHIFT", "SHOULD_DOSE_FILTER", "SHOULD_RESTORE_POWER", "TERMINATION_THRESHOLD", "MAX_ITERATIONS", "BFACTOR", "SHOULD_MASK_CENTRAL_CROSS", "HORIZONTAL_MASK", "VERTICAL_MASK" );

	for (counter = 0; counter < my_job_tracker.total_number_of_jobs; counter++)
	{
		main_frame->current_project.database.AddToBatchInsert("iiiirrrrrriiriiiii", alignment_id,
				                                                                    (unsigned long int) time(NULL),
																					alignment_job_id,
																					movie_asset_panel->ReturnAssetID(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter)),
																					my_job_package.jobs[counter].arguments[13].ReturnFloatArgument(), // voltage
																					my_job_package.jobs[counter].arguments[2].ReturnFloatArgument(), // pixel size
																					my_job_package.jobs[counter].arguments[14].ReturnFloatArgument(), // exposure per frame
																					my_job_package.jobs[counter].arguments[15].ReturnFloatArgument(), // current_pre_exposure
																					my_job_package.jobs[counter].arguments[3].ReturnFloatArgument(), // min shift
																					my_job_package.jobs[counter].arguments[4].ReturnFloatArgument(), // max shift
																					my_job_package.jobs[counter].arguments[5].ReturnBoolArgument(), // should dose filter
																					my_job_package.jobs[counter].arguments[6].ReturnBoolArgument(), // should restore power
																					my_job_package.jobs[counter].arguments[7].ReturnFloatArgument(), // termination threshold
																					my_job_package.jobs[counter].arguments[8].ReturnIntegerArgument(), // max_iterations
																					my_job_package.jobs[counter].arguments[9].ReturnFloatArgument(), // bfactor
																					my_job_package.jobs[counter].arguments[10].ReturnBoolArgument(), // should mask central cross
																					my_job_package.jobs[counter].arguments[11].ReturnIntegerArgument(), // horizonatal mask
																					my_job_package.jobs[counter].arguments[12].ReturnIntegerArgument() // vertical mask
																					);

		alignment_id++;


	}

	main_frame->current_project.database.EndBatchInsert();

	// now need to add the results of the job..

	alignment_id = starting_alignment_id + 1;

	for (counter = 0; counter < my_job_tracker.total_number_of_jobs; counter++)
	{
		current_table_name = wxString::Format("MOVIE_ALIGNMENT_PARAMETERS_%i", alignment_id);
		main_frame->current_project.database.CreateTable(current_table_name, "prr", "FRAME_NUMBER", "X_SHIFT", "Y_SHIFT");
		main_frame->current_project.database.BeginBatchInsert(current_table_name, 3, "FRAME_NUMBER", "X_SHIFT", "Y_SHIFT");

		for (frame_counter = 0; frame_counter < buffered_results[counter].result_size / 2; frame_counter++)
		{
			main_frame->current_project.database.AddToBatchInsert("irr", frame_counter + 1, buffered_results[counter].result_data[frame_counter], buffered_results[counter].result_data[frame_counter +  buffered_results[counter].result_size / 2]);
		}

		main_frame->current_project.database.EndBatchInsert();
		alignment_id++;

	}


}


void MyAlignMoviesPanel::UpdateProgressBar()
{
	TimeRemaining time_left = my_job_tracker.ReturnRemainingTime();
	ProgressBar->SetValue(my_job_tracker.ReturnPercentCompleted());

	TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_left.hours, time_left.minutes, time_left.seconds));
}


