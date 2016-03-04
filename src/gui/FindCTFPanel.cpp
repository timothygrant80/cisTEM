#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMovieAssetPanel *movie_asset_panel;
extern MyImageAssetPanel *image_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyMainFrame *main_frame;

MyFindCTFPanel::MyFindCTFPanel( wxWindow* parent )
:
FindCTFPanel( parent )
{
	// Set variables

	buffered_results = NULL;

	// Fill combo box..

	//FillGroupComboBox();

	my_job_id = -1;
	running_job = false;

	group_combo_is_dirty = false;
	run_profiles_are_dirty = false;

	SetInfo();
	FillGroupComboBox();
	FillRunProfileComboBox();

	wxSize input_size = InputSizer->GetMinSize();
	input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
	input_size.y = -1;
	ExpertPanel->SetMinSize(input_size);
	ExpertPanel->SetSize(input_size);

	AmplitudeContrastNumericCtrl->SetMinMaxValue(0.0f, 1.0f);
	MinResNumericCtrl->SetMinMaxValue(0.0f, 50.0f);
	MaxResNumericCtrl->SetMinMaxValue(0.0f, 50.0f);
	DefocusStepNumericCtrl->SetMinMaxValue(1.0f, FLT_MAX);
	ToleratedAstigmatismNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
	MinPhaseShiftNumericCtrl->SetMinMaxValue(-3.15, 3.15);
	MaxPhaseShiftNumericCtrl->SetMinMaxValue(-3.15, 3.15);
	PhaseShiftStepNumericCtrl->SetMinMaxValue(0.001, 3.15);

}

void MyFindCTFPanel::OnInfoURL(wxTextUrlEvent& event)
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


void MyFindCTFPanel::SetInfo()
{
	#include "icons/ctffind_definitions.cpp"
	#include "icons/ctffind_diagnostic_image.cpp"
	#include "icons/ctffind_example_1dfit.cpp"

	wxBitmap definitions_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_definitions);
	wxBitmap diagnostic_image_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_diagnostic_image);
	wxBitmap example_1dfit_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_example_1dfit);

	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("CTF Estimation"));
	InfoText->EndFontSize();
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("The contrast transfer function (CTF) of the microscope affects the relative signal-to-noise ratio (SNR) of Fourier components of each micrograph. Those Fourier components where the CTF is near 0.0 have very low SNR compared to others. It is therefore essential to obtain accurate estimates of the CTF for each micrograph so that  data from multiple micrographs may be combined in an optimal manner during later processing.\n\nIn this panel, you can use CTFfind (Mindell & Grigorieff, 2003; Rohou & Grigorieff, 2015) to estimate CTF parameter values for each micrograph. The main parameter to be determined for each micrograph is the objective lens defocus (in Angstroms). Because in general lenses are astigmatic, one actually needs to determine two defocus values (describing defocus along the lens' major and minor axes) and the angle of astigmatism."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(definitions_bmp);
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("To estimate the values of these three defocus parameters for a micrograph, CTFfind computes a filtered version of the amplitude spectrum of the micrograph and then fits a model of the CTF (Equation X of Rohou & Grigorieff) to this filtered amplitude spectrum. It then returns the values of the defocus parameters which maximize the quality of the fit, as well as an image of the filtered amplitude spectrum, with the CTF model overlayed onto the lower-left quadrant. Dashed lines are also overlayed onto Fourier components where the CTF is 0. "));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(diagnostic_image_bmp);
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT(" Another diagnostic output is a 1D plot of the experimental amplitude spectrum (green), the CTF fit (orange) and the quality of fit (blue). More details on how these plots are computed is given below."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(example_1dfit_bmp);
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
	InfoText->WriteText(wxT("The group of image assets to estimate the CTF for."));
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
	InfoText->WriteText(wxT("Box size : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The dimensions of the amplitude spectrum CTFfind will compute. Smaller box sizes make the fitting process significantly faster, but sometimes at the expense of fitting accuracy. If you see warnings regarding CTF aliasing, consider increasing this parameter."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Minimum resolution for fitting  : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The CTF model will not be fit to regions of the amplitude spectrum corresponding to this resolution or lower."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Maximum resolution for fitting : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The CTF model will not be fit to regions of the amplitude spectrum corresponding to this resolution or higher."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Lowest defocus to search over  : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Positive for underfocus. The Lower bound of initial defocus search."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Highest defocus to search over : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Positive for underfocus. Upper bound of initial defocus search."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Defocus search step : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Step size for the defocus search."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Expected (tolerated) astigmatism : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Astigmatism values much larger than this will be penalised. Set to negative to remove this restraint. In cases where the amplitude spectrum is very noisy, such a restraint can help achieve more accurate results."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Relative amplitude contrast : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The fraction (between 0.0 and 1.0) of total image contrast attributed to amplitude contrast (as opposed to phase contrast), arising for example from electron scattered outside the objective aperture, or those removed by energy filtering."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Find additional phase shift  : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Specifies the input micrograph was recorded using a phase plate with variable phase shift, which you want to find"));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Lower bound of initial phase shift search : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If finding an additional phase shift, this value sets the lower bound for the search."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Upper bound of initial phase shift search : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If finding an additional phase shift, this value sets the upper bound for the search."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Step size of initial phase shift search  : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If finding an additional phase shift, this value sets the step size for the search."));
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
	InfoText->WriteText(wxT("Rohou A., Grigorieff N.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2015. CTFFIND4: Fast and accurate defocus estimation from electron micrographs. J. Struct. Biol. 192, 216–221. "));
	InfoText->BeginURL("http://dx.doi.org/10.1016/j.jsb.2015.08.008");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.1016/j.jsb.2015.08.008"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->EndSuppressUndo();


}

void MyFindCTFPanel::FillGroupComboBox()
{

	GroupComboBox->Freeze();
	GroupComboBox->Clear();

	for (long counter = 0; counter < image_asset_panel->ReturnNumberOfGroups(); counter++)
	{
		GroupComboBox->Append(image_asset_panel->ReturnGroupName(counter) +  " (" + wxString::Format(wxT("%li"), image_asset_panel->ReturnGroupSize(counter)) + ")");

	}

	GroupComboBox->SetSelection(0);

	GroupComboBox->Thaw();
}

void MyFindCTFPanel::FillRunProfileComboBox()
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

void MyFindCTFPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	// are there enough members in the selected group.
	if (main_frame->current_project.is_open == false)
	{
		RunProfileComboBox->Enable(false);
		GroupComboBox->Enable(false);
		ExpertToggleButton->Enable(false);
		StartEstimationButton->Enable(false);
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
			if (image_asset_panel->ReturnGroupSize(GroupComboBox->GetCurrentSelection()) > 0 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(RunProfileComboBox->GetSelection()) > 1)
			{
				StartEstimationButton->Enable(true);
			}
			else StartEstimationButton->Enable(false);
		}
		else
		{
			StartEstimationButton->Enable(false);
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

		if (group_combo_is_dirty == true)
		{
			FillGroupComboBox();
			group_combo_is_dirty = false;
		}

		if (run_profiles_are_dirty == true)
		{
			FillRunProfileComboBox();
			run_profiles_are_dirty = false;
		}
	}




}

void MyFindCTFPanel::OnMovieRadioButton(wxCommandEvent& event )
{
	NoFramesToAverageSpinCtrl->Enable(true);
	NoMovieFramesStaticText->Enable(true);
}

void MyFindCTFPanel::OnImageRadioButton(wxCommandEvent& event )
{
	NoFramesToAverageSpinCtrl->Enable(false);
	NoMovieFramesStaticText->Enable(false);
}

void MyFindCTFPanel::OnFindAdditionalPhaseCheckBox(wxCommandEvent& event )
{
	if (AdditionalPhaseShiftCheckBox->IsChecked() == true)
	{
		Freeze();
		MaxPhaseShiftNumericCtrl->Enable(true);
		MinPhaseShiftNumericCtrl->Enable(true);
		PhaseShiftStepNumericCtrl->Enable(true);
		MaxPhaseShiftStaticText->Enable(true);
		MinPhaseShiftStaticText->Enable(true);
		PhaseShiftStepStaticText->Enable(true);
		Thaw();
	}
	else
	{
		Freeze();
		MaxPhaseShiftNumericCtrl->Enable(false);
		MinPhaseShiftNumericCtrl->Enable(false);
		PhaseShiftStepNumericCtrl->Enable(false);
		MaxPhaseShiftStaticText->Enable(false);
		MinPhaseShiftStaticText->Enable(false);
		PhaseShiftStepStaticText->Enable(false);
		Thaw();

	}
}

void MyFindCTFPanel::OnRestrainAstigmatismCheckBox(wxCommandEvent& event )
{
	if (RestrainAstigmatismCheckBox->IsChecked() == true)
	{
		Freeze();
		ToleratedAstigmatismNumericCtrl->Enable(true);
		ToleratedAstigmatismStaticText->Enable(true);
		Thaw();

	}
	else
	{
		Freeze();
		ToleratedAstigmatismNumericCtrl->Enable(false);
		ToleratedAstigmatismStaticText->Enable(false);
		Thaw();

	}

}

void MyFindCTFPanel::OnExpertOptionsToggle(wxCommandEvent& event )
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


void MyFindCTFPanel::StartEstimationClick( wxCommandEvent& event )
{

	wxPrintf("clicked\n");
	MyDebugAssertTrue(buffered_results == NULL, "Error: buffered results not null")


	// Package the job details..

	long counter;
	long number_of_jobs = image_asset_panel->ReturnGroupSize(GroupComboBox->GetCurrentSelection()); // how many images / movies in the selected group..

	bool ok_number_conversion;

	int number_of_processes;

	int current_asset_id;
	int parent_asset_id;
	int number_of_previous_estimations;

	wxString buffer_filename;

	std::string input_filename;
	bool        input_is_a_movie;
	int         number_of_frames_to_average;
	std::string output_diagnostic_filename;
	float 		pixel_size;
	float 		acceleration_voltage;
	float       spherical_aberration;
	float 		amplitude_contrast;
	int         box_size;
	float 		minimum_resolution;
	float       maximum_resolution;
	float       minimum_defocus;
	float       maximum_defocus;
	float       defocus_search_step;
	float       astigmatism_tolerance;
	bool       	find_additional_phase_shift;
	float  		minimum_additional_phase_shift;
	float		maximum_additional_phase_shift;
	float		additional_phase_shift_search_step;

	// allocate space for the buffered results..

	buffered_results = new JobResult[number_of_jobs];

	// read the options form the gui..

	if (MovieRadioButton->GetValue() == true) input_is_a_movie = true;
	else input_is_a_movie = false;

	number_of_frames_to_average = NoFramesToAverageSpinCtrl->GetValue();
	amplitude_contrast = AmplitudeContrastNumericCtrl->ReturnValue();
	box_size = BoxSizeSpinCtrl->GetValue();
	minimum_resolution = MinResNumericCtrl->ReturnValue();
	maximum_resolution = MaxResNumericCtrl->ReturnValue();
	minimum_defocus = LowDefocusNumericCtrl->ReturnValue();
	maximum_defocus = HighDefocusNumericCtrl->ReturnValue();
	defocus_search_step = DefocusStepNumericCtrl->ReturnValue();

	if (RestrainAstigmatismCheckBox->IsChecked() == false) astigmatism_tolerance = -100.0f;
	else astigmatism_tolerance = ToleratedAstigmatismNumericCtrl->ReturnValue();

	if (AdditionalPhaseShiftCheckBox->IsChecked() == true)
	{
		find_additional_phase_shift = true;
		minimum_additional_phase_shift = MinPhaseShiftNumericCtrl->ReturnValue();
		maximum_additional_phase_shift = MaxPhaseShiftNumericCtrl->ReturnValue();
		additional_phase_shift_search_step = PhaseShiftStepNumericCtrl->ReturnValue();
	}
	else
	{
		find_additional_phase_shift = false;
		minimum_additional_phase_shift = 0.0f;
		maximum_additional_phase_shift = 0.0f;
		additional_phase_shift_search_step = 0.0f;
	}


	my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()], "ctffind", number_of_jobs);

	for (counter = 0; counter < number_of_jobs; counter++)
	{
		// job is :-
		//
		// input_filename (string);
		// input_is_a_movie (bool);
		// number_of_frames_to_average (int);
		// output_diagnostic_filename (string);
		// pixel_size (float);
		// acceleration_voltage (float);
		// spherical_aberration (float);
		// amplitude_contrast (float);
		// box_size (int);
		// minimum_resolution (float);
		// maximum_resolution (float);
		// minimum_defocus (float);
		// maximum_defocus (float);
		// defocus_search_step (float);
		// astigmatism_tolerance (float);
		// find_additional_phase_shift (bool);
		// minimum_additional_phase_shift (float);
		// maximum_additional_phase_shift (float);
		///additional_phase_shift_search_step (float);

		if (input_is_a_movie == true)
		{
			parent_asset_id = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter))->parent_id;
			input_filename = movie_asset_panel->ReturnAssetPointer(movie_asset_panel->ReturnArrayPositionFromAssetID(parent_asset_id))->filename.GetFullPath().ToStdString();
		}
		else
		{
			input_filename = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter))->filename.GetFullPath().ToStdString();
		}

		pixel_size = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter))->pixel_size;
		acceleration_voltage = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter))->microscope_voltage;
		spherical_aberration = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter))->spherical_aberration;


		//output_filename = movie_asset_panel->ReturnAssetLongFilename(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter));
		//output_filename.Replace(".mrc", "_ali.mrc", false);

		current_asset_id = image_asset_panel->ReturnAssetID(image_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter));
		buffer_filename = image_asset_panel->ReturnAssetShortFilename(image_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter));
		number_of_previous_estimations =  main_frame->current_project.database.ReturnNumberOfPreviousCTFEstimationsByAssetID(current_asset_id);

		buffer_filename = main_frame->current_project.ctf_asset_directory.GetFullPath();
		buffer_filename += wxString::Format("/%s_CTF_%i.mrc", image_asset_panel->ReturnAssetShortFilename(image_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter)), current_asset_id, number_of_previous_estimations);

		output_diagnostic_filename = buffer_filename.ToStdString();


		my_job_package.AddJob("sbisffffiffffffbfff",	input_filename,
														input_is_a_movie,
														number_of_frames_to_average,
														output_diagnostic_filename,
														pixel_size,
														acceleration_voltage,
														spherical_aberration,
														amplitude_contrast,
														box_size,
														minimum_resolution,
														maximum_resolution,
														minimum_defocus,
														maximum_defocus,
														defocus_search_step,
														astigmatism_tolerance,
														find_additional_phase_shift,
														minimum_additional_phase_shift,
														maximum_additional_phase_shift,
														additional_phase_shift_search_step);
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
		ResultsPanel->Show(true);

		ExpertToggleButton->Enable(false);
		GroupComboBox->Enable(false);
		Layout();

		running_job = true;
		my_job_tracker.StartTracking(my_job_package.number_of_jobs);

	}
	ProgressBar->Pulse();

}

void MyFindCTFPanel::FinishButtonClick( wxCommandEvent& event )
{
	ProgressBar->SetValue(0);
	TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
	FinishButton->Show(false);

	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	output_textctrl->Clear();
	ResultsPanel->Show(false);
	//graph_is_hidden = true;
	InfoPanel->Show(true);

	if (ExpertToggleButton->GetValue() == true) ExpertPanel->Show(true);
	else ExpertPanel->Show(false);
	running_job = false;
	Layout();



}

void MyFindCTFPanel::TerminateButtonClick( wxCommandEvent& event )
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



/*


void MyAlignMoviesPanel::Refresh()
{
	FillGroupComboBox();
	FillRunProfileComboBox();
}



*/
void MyFindCTFPanel::WriteInfoText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void MyFindCTFPanel::WriteErrorText(wxString text_to_write)
{
	 output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
	 output_textctrl->AppendText(text_to_write);

	 if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}




void MyFindCTFPanel::OnJobSocketEvent(wxSocketEvent& event)
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
		  {/*
			 WriteInfoText("All Jobs have finished.");
			 ProgressBar->SetValue(100);
			 TimeRemainingText->SetLabel("Time Remaining : All Done!");
			 CancelAlignmentButton->Show(false);
			 FinishButton->Show(true);
			 ProgressPanel->Layout();
			 running_job = false;
*/
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

void  MyFindCTFPanel::ProcessResult(JobResult *result_to_process) // this will have to be overidden in the parent clas when i make it.
{
	int number_of_frames;
	int frame_counter;

	long current_time = time(NULL);


	wxPrintf ("Got a results DF1 = %f, DF2 = %f, AA = %f, PS = %f, SC = %f, TF = %f, AL = %f\n");
	// results should be ..

	// Defocus 1 (Angstroms)
	// Defocus 2 (Angstroms)
	// Astigmatism Angle (degrees)
	// Additional phase shift (e.g. from phase plate) radians
	// Score
	// Resolution (Angstroms) to which Thon rings are well fit by the CTF
	// Reolution (Angstroms) at which aliasing was detected

	if (current_time - time_of_last_result_update > 1)
	{

//		GraphPanel->Draw();

		//if (graph_is_hidden == true)
		{
	//		GraphPanel->Show(true);
	//		Layout();
	//		graph_is_hidden = false;
		}

		time_of_last_result_update = current_time;


	}

	my_job_tracker.MarkJobFinished();
//	if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();

	// store the results..

	buffered_results[result_to_process->job_number] = result_to_process;

	if (my_job_tracker.total_number_of_finished_jobs == my_job_tracker.total_number_of_jobs)
	{
		// job has really finished, so we can write to the database...

	//	WriteResultToDataBase();

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

/*


void MyAlignMoviesPanel::WriteResultToDataBase()
{

	long counter;
	int frame_counter;
	int array_location;
	int parent_id;
	bool have_errors = false;
	wxString current_table_name;
	ImageAsset temp_asset;

	// find the current highest alignment number in the database, then increment by one

	int starting_alignment_id = main_frame->current_project.database.ReturnHighestAlignmentID();
	int alignment_id = starting_alignment_id + 1;
	int alignment_job_id =  main_frame->current_project.database.ReturnHighestAlignmentJobID() + 1;

	// loop over all the jobs, and add them..

	main_frame->current_project.database.BeginBatchInsert("MOVIE_ALIGNMENT_LIST", 19, "ALIGNMENT_ID", "DATETIME_OF_RUN", "ALIGNMENT_JOB_ID", "MOVIE_ASSET_ID", "OUTPUT_FILE", "VOLTAGE", "PIXEL_SIZE", "EXPOSURE_PER_FRAME", "PRE_EXPOSURE_AMOUNT", "MIN_SHIFT", "MAX_SHIFT", "SHOULD_DOSE_FILTER", "SHOULD_RESTORE_POWER", "TERMINATION_THRESHOLD", "MAX_ITERATIONS", "BFACTOR", "SHOULD_MASK_CENTRAL_CROSS", "HORIZONTAL_MASK", "VERTICAL_MASK" );

	wxDateTime now = wxDateTime::Now();
	for (counter = 0; counter < my_job_tracker.total_number_of_jobs; counter++)
	{
		main_frame->current_project.database.AddToBatchInsert("iliitrrrrrriiriiiii", alignment_id,
				                                                                    (long int) now.GetAsDOS(),
																					alignment_job_id,
																					movie_asset_panel->ReturnAssetID(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter)),
																					my_job_package.jobs[counter].arguments[1].ReturnStringArgument(), // output_filename
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
																					int(my_job_package.jobs[counter].arguments[9].ReturnFloatArgument()), // bfactor
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

	// now we need to add the resulting image files as image assets..

	// for adding to the database..
	main_frame->current_project.database.BeginImageAssetInsert();

	MyErrorDialog *my_error = new MyErrorDialog(this);
	alignment_id = starting_alignment_id + 1;

	for (counter = 0; counter < my_job_tracker.total_number_of_jobs; counter++)
	{
			temp_asset.Update(my_job_package.jobs[counter].arguments[1].ReturnStringArgument());

			if (temp_asset.is_valid == true)
			{
				parent_id = movie_asset_panel->ReturnAssetID(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter));
				array_location = image_asset_panel->ReturnArrayPositionFromParentID(parent_id);

				// is this image (or a previous version) already an asset?

				if (array_location == -1) // we don't already have an asset from this movie..
				{
					temp_asset.asset_id = image_asset_panel->current_asset_number;
					temp_asset.parent_id = parent_id;
					temp_asset.alignment_id = alignment_id;
					temp_asset.microscope_voltage = my_job_package.jobs[counter].arguments[13].ReturnFloatArgument();
					temp_asset.pixel_size = my_job_package.jobs[counter].arguments[2].ReturnFloatArgument();
					temp_asset.position_in_stack = 1;
					temp_asset.spherical_aberration = movie_asset_panel->ReturnAssetSphericalAbberation(movie_asset_panel->ReturnArrayPositionFromAssetID(parent_id));
					image_asset_panel->AddAsset(&temp_asset);
					main_frame->current_project.database.AddNextImageAsset(temp_asset.asset_id, temp_asset.filename.GetFullPath(), temp_asset.position_in_stack, temp_asset.parent_id, alignment_id, temp_asset.x_size, temp_asset.y_size, temp_asset.microscope_voltage, temp_asset.pixel_size, temp_asset.spherical_aberration);


				}
				else
				{// TODO:: Rewrite this to use return asset pointer..//
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].filename = my_job_package.jobs[counter].arguments[1].ReturnStringArgument();
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].parent_id = parent_id;
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].alignment_id = alignment_id;
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].microscope_voltage = my_job_package.jobs[counter].arguments[13].ReturnFloatArgument();
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].pixel_size = my_job_package.jobs[counter].arguments[2].ReturnFloatArgument();
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].position_in_stack = 1;
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].spherical_aberration = movie_asset_panel->ReturnAssetSphericalAbberation(movie_asset_panel->ReturnArrayPositionFromAssetID(parent_id));
					main_frame->current_project.database.AddNextImageAsset(reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].asset_id,
							 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	            my_job_package.jobs[counter].arguments[1].ReturnStringArgument(),
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].position_in_stack,
																											parent_id,
																											alignment_id,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].x_size,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].y_size,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].microscope_voltage,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].pixel_size,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].spherical_aberration);
				}


				image_asset_panel->current_asset_number++;
			}
			else
			{
				my_error->ErrorText->AppendText(wxString::Format(wxT("%s is not a valid MRC file, skipping\n"), temp_asset.ReturnFullPathString()));
				have_errors = true;
			}

			alignment_id++;
	}

	main_frame->current_project.database.EndImageAssetInsert();

	image_asset_panel->is_dirty = true;
	movie_results_panel->is_dirty = true;


	if (have_errors == true)
	{
		my_error->ShowModal();
	}

	my_error->Destroy();


}


void MyAlignMoviesPanel::UpdateProgressBar()
{
	TimeRemaining time_left = my_job_tracker.ReturnRemainingTime();
	ProgressBar->SetValue(my_job_tracker.ReturnPercentCompleted());

	TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_left.hours, time_left.minutes, time_left.seconds));
}

*/
