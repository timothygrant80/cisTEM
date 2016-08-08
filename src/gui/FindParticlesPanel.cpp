//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMovieAssetPanel *movie_asset_panel;
extern MyImageAssetPanel *image_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyMainFrame *main_frame;
extern MyFindCTFResultsPanel *ctf_results_panel;
extern MyParticlePositionAssetPanel *particle_position_asset_panel;
extern MyPickingResultsPanel *picking_results_panel;


MyFindParticlesPanel::MyFindParticlesPanel( wxWindow* parent )
:
FindParticlesPanel( parent )
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
	FillPickingAlgorithmComboBox();

	wxSize input_size;
	int input_size_x = std::max(InputSizer->GetMinSize().x,ExpertInputSizer->GetMinSize().x);
	input_size.x = input_size_x + wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
	input_size.y = -1;

	PickingParametersPanel->SetMinSize(input_size);
	PickingParametersPanel->SetSize(input_size);

	/*
	AmplitudeContrastNumericCtrl->SetMinMaxValue(0.0f, 1.0f);
	MinResNumericCtrl->SetMinMaxValue(0.0f, 50.0f);
	MaxResNumericCtrl->SetMinMaxValue(0.0f, 50.0f);
	DefocusStepNumericCtrl->SetMinMaxValue(1.0f, FLT_MAX);
	ToleratedAstigmatismNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
	MinPhaseShiftNumericCtrl->SetMinMaxValue(-3.15, 3.15);
	MaxPhaseShiftNumericCtrl->SetMinMaxValue(-3.15, 3.15);
	PhaseShiftStepNumericCtrl->SetMinMaxValue(0.001, 3.15);
	*/

	//result_bitmap.Create(1,1, 24);
	time_of_last_result_update = time(NULL);

}


void MyFindParticlesPanel::OnInfoURL(wxTextUrlEvent& event)
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


void MyFindParticlesPanel::SetInfo()
{

	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("Particle picking"));
	InfoText->EndFontSize();
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("Individual particles need to be located in each micrograph so that they may be used to compute a 3D reconstruction later. Ideally one would find all the particles and not make any erroneous selections."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->WriteText(wxT("In the absence of a pre-existing 3D model, one can either select (click on) each particle manually, or use the 'ab initio' mode. In this mode, a template is genated internally, which consists of a cosine-shaped blob and then matched against each micrographs. This works reasonably well to find globular protein complexes, even though it is less accurate and more error-prone than template-based search strategy."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->WriteText(wxT(""));
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("Program Options"));
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Input Group : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The group of image assets to estimate the CTF for."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Picking algorithm : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Choice of method for picking particles."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Run Profile : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The selected run profile will be used to run the job.  The run profile describes how the job should be run (e.g. how many processors should be used, and on which different computers).  Run profiles are set in the Run Profile panel, located under settings."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Maximum radius : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("In Angstroms, the maximum radius of the particles to be found. This also determines the minimum distance between picks."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Threshold peak height : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Particle coordinates will be defined as the coordinates of any peak in the search function which exceeds this threshold. In numbers of standard deviations above expected noise variations in the scoring function. See Sigworth (2004) for definition."));
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("Program Options: ab initio mode"));
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Characteristic particle radius : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("In Angstroms, the radius within which most of the density is enclosed. This defines the radius at which the cosine-edge template reaches 0.5 (it is 1.0 at the origin and 0.0 at 1.5 * characteristic radius). A good default value might be half of the maximum radius."));
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
	InfoText->WriteText(wxT("Expert Program Options"));
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Highest resolution used in picking : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The template and micrograph will be resampled (by Fourier cropping) to a pixel size of half the resolution given here. Note that the information int the 'corners' of the Fourier transforms remains intact, so that there is some small risk of bias beyond this resolution."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Minimum distance from edges : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("No particle shall be picked closer than this distance from the edges of the micrograph. In pixels."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Avoid high-variance areas : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Avoid areas with abnormally high local variance. This can be effective in avoiding edges of support films or contamination."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Avoid areas with abnormal local means : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Avoid areas with abnormally low or high local mean. This can be effective to avoid picking from, e.g., contaminating ice crystals, support film."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Show estimated background spectrum : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Show 1D plot of the estimated background spectrum, which is used to build the whitening filter (see Sigworth, 2004, for details)"));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Show positions of background boxes : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Plot the position of areas the algorithm selected as background. For optimal performance, none of these boxes should contain any particles."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Number of background boxes : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Number of background areas to use in estimating the background spectrum. The larger the number of boxes, the more accurate the estimate should be, provided that none of the background boxes contain any particles to be picked."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Algorithm to find background areas : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Testing so far suggests that areas of lowest variance in experimental micrographs should be used to estimate the background spectrum. However, when using synthetic micrographs this can lead to bias in the spectrum estimation and the alternative (areas with local variances near the mean of the distribution of local variances) seems to perform better"));
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
	InfoText->WriteText(wxT("Sigworth F.J."));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2004. Classical detection theory and the cryo-EM particle selection problem. J. Struct. Biol. 192, 216–221. "));
	InfoText->BeginURL("http://dx.doi.org/10.1016/j.jsb.2003.10.025");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.1016/j.jsb.2003.10.025"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->EndSuppressUndo();


}

void MyFindParticlesPanel::FillGroupComboBox()
{

	GroupComboBox->Freeze();
	GroupComboBox->Clear();

	int number_of_groups = image_asset_panel->ReturnNumberOfGroups();

	// Reallocate the array that keeps track for each group of whether it can be picked


	for (long counter = 0; counter < number_of_groups; counter++)
	{
		GroupComboBox->Append(image_asset_panel->ReturnGroupName(counter) +  " (" + wxString::Format(wxT("%li"), image_asset_panel->ReturnGroupSize(counter)) + ")");
		// Check whether this group is ready to be picked

	}

	GroupComboBox->SetSelection(0);

	GroupComboBox->Thaw();
}

wxString MyFindParticlesPanel::ReturnNameOfPickingAlgorithm( const int wanted_algorithm )
{

	wxString string_to_return;

	switch(wanted_algorithm)
	{
	case(ab_initio):
			string_to_return = "ab initio";
			break;
	default:
			string_to_return = "unknown";
	}

	return string_to_return;
}

int MyFindParticlesPanel::ReturnNumberOfJobsCurrentlyRunning()
{
	return my_job_tracker.total_running_processes;
}

void MyFindParticlesPanel::FillPickingAlgorithmComboBox()
{

	PickingAlgorithmComboBox->Freeze();
	PickingAlgorithmComboBox->Clear();

	for ( int counter = 0; counter < number_of_picking_algorithms; counter ++ )
	{
		PickingAlgorithmComboBox->Append(ReturnNameOfPickingAlgorithm(counter).Capitalize());
	}

	PickingAlgorithmComboBox->SetSelection(-1);

	PickingAlgorithmComboBox->Thaw();
}

// When the user selects a new picking algorithm
void MyFindParticlesPanel::OnPickingAlgorithmComboBox( wxCommandEvent& event )
{
	switch(PickingAlgorithmComboBox->GetCurrentSelection())
	{
	case(-1) :
			// No algorithm selected
			PickingParametersPanel->Hide();
			ExpertToggleButton->Enable(false);
			break;
	case(0):
			// Ab initio
			PickingParametersPanel->Show(true);
			ExpertToggleButton->Enable(true);
			break;
	default:
			// This algorithm not implemented yet
			PickingParametersPanel->Hide();
			ExpertToggleButton->Enable(false);
	}

	Layout();
}

void MyFindParticlesPanel::OnAutoPickRefreshCheckBox( wxCommandEvent& event )
{
	if (AutoPickRefreshCheckBox->GetValue())
	{
		// Auto pick refresh is enabled
		TestOnCurrentMicrographButton->Enable(false);
	}
	else
	{
		// Auto pick refresh is disabled
		TestOnCurrentMicrographButton->Enable(true);
	}
}

void MyFindParticlesPanel::OnSetMinimumDistanceFromEdgesCheckBox( wxCommandEvent & event )
{
	if ( SetMinimumDistanceFromEdgesCheckBox->GetValue())
	{
		MinimumDistanceFromEdgesSpinCtrl->Enable(true);
	}
	else
	{
		MinimumDistanceFromEdgesSpinCtrl->Enable(false);
	}
	Layout();
}

void MyFindParticlesPanel::FillRunProfileComboBox()
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

void MyFindParticlesPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	// are there enough members in the selected group.
	if (main_frame->current_project.is_open == false)
	{
		RunProfileComboBox->Enable(false);
		GroupComboBox->Enable(false);
		PickingAlgorithmComboBox->Enable(false);
		ExpertToggleButton->Enable(false);
		StartPickingButton->Enable(false);
	}
	else
	{
		//Enable(true);

		if (group_combo_is_dirty == true)
		{
			FillGroupComboBox();
			CheckWhetherGroupsCanBePicked();
			group_combo_is_dirty = false;
		}

		if (run_profiles_are_dirty == true)
		{
			FillRunProfileComboBox();
			run_profiles_are_dirty = false;
		}

		if (running_job == false)
		{
			RunProfileComboBox->Enable(true);
			GroupComboBox->Enable(true);
			PickingAlgorithmComboBox->Enable(true);
			if (image_asset_panel->all_groups_list->groups[GroupComboBox->GetCurrentSelection()].can_be_picked)
			{
				if (PleaseEstimateCTFStaticText->IsShown())
				{
					PleaseEstimateCTFStaticText->Show(false);
					Layout();
				}
			}
			else
			{
				if (!PleaseEstimateCTFStaticText->IsShown())
				{
					PleaseEstimateCTFStaticText->Show(true);
					Layout();
				}
			}
			if (PickingAlgorithmComboBox->GetCurrentSelection() >= 0)
			{
				ExpertToggleButton->Enable(true);
			}
			if (RunProfileComboBox->GetCount() > 0)
			{
				if (image_asset_panel->ReturnGroupSize(GroupComboBox->GetCurrentSelection()) > 0 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(RunProfileComboBox->GetSelection()) > 1 && PickingAlgorithmComboBox->GetCurrentSelection() >= 0 && image_asset_panel->all_groups_list->groups[GroupComboBox->GetCurrentSelection()].can_be_picked)
				{
					StartPickingButton->Enable(true);
				}
				else StartPickingButton->Enable(false);
			}
			else
			{
				StartPickingButton->Enable(false);
			}
		}
		else
		{
			ExpertToggleButton->Enable(false);
			GroupComboBox->Enable(false);
			PickingAlgorithmComboBox->Enable(false);
			RunProfileComboBox->Enable(false);
			//StartAlignmentButton->SetLabel("Stop Job");
			//StartAlignmentButton->Enable(true);
		}

	}




}

void MyFindParticlesPanel::OnExpertOptionsToggle( wxCommandEvent& event )
{
	ExpertOptionsPanel->Show(ExpertToggleButton->GetValue());
	Layout();
}


void MyFindParticlesPanel::StartPickingClick( wxCommandEvent& event )
{

	wxPrintf("start picking clicked\n");
	MyDebugAssertTrue(buffered_results == NULL, "Error: buffered results not null")


	// Package the job details..

	long counter;
	long number_of_jobs = image_asset_panel->ReturnGroupSize(GroupComboBox->GetCurrentSelection()); // how many images / movies in the selected group..

	bool ok_number_conversion;

	int number_of_processes;

	int current_asset_id;
	int parent_asset_id;
	int number_of_previous_estimations;

	ImageAsset * current_image_asset;

	wxString buffer_filename;

	std::string input_filename;
	float		pixel_size;
	double 		acceleration_voltage;
	double      spherical_aberration;
	double 		amplitude_contrast;
	double		defocus_1;
	double		defocus_2;
	double		astigmatism_angle;
	double		additional_phase_shift;

	bool		already_have_templates = false;
	std::string	templates_filename = "no_templates.mrcs";
	bool		average_templates_radially = true;
	int			number_of_template_rotations = 1;
	float		typical_radius = CharacteristicParticleRadiusNumericCtrl->ReturnValue();
	float		maximum_radius = MaximumParticleRadiusNumericCtrl->ReturnValue();
	float		highest_resolution_to_use = HighestResolutionNumericCtrl->ReturnValue();
	std::string	output_stack_filename;
	int			output_stack_box_size = 0;
	int			minimum_distance_from_edges = 128;
	float		picking_threshold = ThresholdPeakHeightNumericCtrl->ReturnValue();
	int			number_of_previous_picks;
	bool		avoid_high_variance_areas;
	bool 		avoid_high_low_mean_areas;
	int			algorithm_to_find_background;
	int			number_of_background_boxes;

	// allocate space for the buffered results..

	buffered_results = new JobResult[number_of_jobs];

	// read the options form the gui..

	switch(PickingAlgorithmComboBox->GetSelection())
	{
	case(ab_initio) :
		already_have_templates = false;
		templates_filename = "no_templates.mrc";
		average_templates_radially = false;
		number_of_template_rotations = 1;
		typical_radius = CharacteristicParticleRadiusNumericCtrl->ReturnValue();
		maximum_radius = MaximumParticleRadiusNumericCtrl->ReturnValue();
	break;
	default :
		MyDebugAssertTrue(false,"Oops, uknown picking algorithm: %i\n",PickingAlgorithmComboBox->GetSelection());
	}

	highest_resolution_to_use = HighestResolutionNumericCtrl->ReturnValue();
	minimum_distance_from_edges = MinimumDistanceFromEdgesSpinCtrl->GetValue();
	picking_threshold = ThresholdPeakHeightNumericCtrl->ReturnValue();
	avoid_high_variance_areas = AvoidHighVarianceAreasCheckBox->GetValue();
	avoid_high_low_mean_areas = AvoidAbnormalLocalMeanAreasCheckBox->GetValue();
	algorithm_to_find_background = AlgorithmToFindBackgroundChoice->GetSelection();
	number_of_background_boxes = NumberOfBackgroundBoxesSpinCtrl->GetValue();


	my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()], "find_particles", number_of_jobs);

	for (counter = 0; counter < number_of_jobs; counter++)
	{

		current_image_asset = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter));

		input_filename 			=	current_image_asset->filename.GetFullPath().ToStdString();
		pixel_size				=	current_image_asset->pixel_size;
		acceleration_voltage	=	current_image_asset->microscope_voltage;
		spherical_aberration	=	current_image_asset->spherical_aberration;

		main_frame->current_project.database.GetCTFParameters(current_image_asset->ctf_estimation_id,acceleration_voltage,spherical_aberration,amplitude_contrast,defocus_1,defocus_2,astigmatism_angle,additional_phase_shift);


		number_of_previous_picks = main_frame->current_project.database.ReturnNumberOfPreviousParticlePicksByAssetID(current_image_asset->asset_id);

		output_stack_filename = main_frame->current_project.particle_position_asset_directory.GetFullPath();
		output_stack_filename += wxString::Format("/%s_COOS_%i.mrc", wxFileName::StripExtension(current_image_asset->ReturnShortNameString()),number_of_previous_picks);


		my_job_package.AddJob("sffffffffbsbifffsiifbbii",	input_filename.c_str(), // 0
															pixel_size,
															acceleration_voltage,
															spherical_aberration,
															amplitude_contrast,
															additional_phase_shift, // 5
															defocus_1,
															defocus_2,
															astigmatism_angle,
															already_have_templates,
															templates_filename.c_str(),
															average_templates_radially,
															number_of_template_rotations,
															typical_radius,
															maximum_radius, // 14
															highest_resolution_to_use,
															output_stack_filename.c_str(),
															output_stack_box_size,
															minimum_distance_from_edges,
															picking_threshold,
															avoid_high_variance_areas,
															avoid_high_low_mean_areas,
															algorithm_to_find_background,
															number_of_background_boxes
															);
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


		PickingParametersPanel->Show(false);
		ExpertOptionsPanel->Show(false);
		InfoPanel->Show(false);
		OutputTextPanel->Show(true);
		PickingResultsPanel->Show(true);

		ExpertToggleButton->Enable(false);
		GroupComboBox->Enable(false);
		PickingAlgorithmComboBox->Enable(false);
		Layout();

		running_job = true;
		my_job_tracker.StartTracking(my_job_package.number_of_jobs);

	}
	ProgressBar->Pulse();

}

void MyFindParticlesPanel::FinishButtonClick( wxCommandEvent& event )
{
	ProgressBar->SetValue(0);
	TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
	FinishButton->Show(false);

	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	output_textctrl->Clear();
	PickingResultsPanel->Show(false);
	//graph_is_hidden = true;
	InfoPanel->Show(true);

	if (PickingAlgorithmComboBox->GetCurrentSelection() >= 0)
	{
		PickingParametersPanel->Show(true);

		if (ExpertToggleButton->GetValue() == true) ExpertOptionsPanel->Show(true);
		else ExpertOptionsPanel->Show(false);
	}
	else
	{
		ExpertToggleButton->Enable(false);
	}


	running_job = false;
	Layout();

	//CTFResultsPanel->CTF2DResultsPanel->should_show = false;
	//CTFResultsPanel->CTF2DResultsPanel->Refresh();



}

void MyFindParticlesPanel::TerminateButtonClick( wxCommandEvent& event )
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
void MyFindParticlesPanel::WriteInfoText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void MyFindParticlesPanel::WriteErrorText(wxString text_to_write)
{
	 output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
	 output_textctrl->AppendText(text_to_write);

	 if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}


// Go over all groups and for each one check whether it's ready to be picked (e.g. that all its images have CTF estimates)
void MyFindParticlesPanel::CheckWhetherGroupsCanBePicked()
{
	int number_of_images_with_ctf_estimates;
	int number_of_images_in_group;
	bool current_image_has_estimate;
	int current_image_id;


	number_of_images_with_ctf_estimates = main_frame->current_project.database.ReturnNumberOfImageAssetsWithCTFEstimates();
	if (number_of_images_with_ctf_estimates > 0)
		{
		int ids_of_images_with_ctf_estimates [number_of_images_with_ctf_estimates];

		main_frame->current_project.database.GetUniqueIDsOfImagesWithCTFEstimations(ids_of_images_with_ctf_estimates,number_of_images_with_ctf_estimates);


		for (int group_counter = 0; group_counter < image_asset_panel->ReturnNumberOfGroups(); group_counter ++ )
		{
			// We start by assuming the current group can be picked
			image_asset_panel->all_groups_list->groups[group_counter].can_be_picked = true;

			number_of_images_in_group = image_asset_panel->ReturnGroupSize(group_counter);

			for ( int counter_in_group = 0; counter_in_group < number_of_images_in_group; counter_in_group ++ )
			{
				current_image_id = image_asset_panel->ReturnGroupMemberID(group_counter,counter_in_group);
				current_image_has_estimate = false;
				for ( int counter_in_estimates = 0; counter_in_estimates < number_of_images_with_ctf_estimates; counter_in_estimates ++ )
				{
					if (ids_of_images_with_ctf_estimates[counter_in_estimates] == current_image_id) current_image_has_estimate = true;
				}
				if (! current_image_has_estimate) {
					// Current group cannot be picked
					image_asset_panel->all_groups_list->groups[group_counter].can_be_picked = false;
				}
			}
		}
	}
	else
	{
		// No images have CTF estimates yet
		for (int group_counter = 0; group_counter < image_asset_panel->ReturnNumberOfGroups(); group_counter ++ )
		{
			image_asset_panel->all_groups_list->groups[group_counter].can_be_picked = false;
		}
	}
}




void MyFindParticlesPanel::OnJobSocketEvent(wxSocketEvent& event)
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
	    	  /*
				// which job is finished?

				int finished_job;
				sock->ReadMsg(&finished_job, 4);
				my_job_tracker.MarkJobFinished();
	    	   */

	    	  // which job is finished, and how big is the result?
	    	  char job_number_and_result_size[8];
	    	  sock->ReadMsg(&job_number_and_result_size,8);

	    	  int job_number;
	    	  int result_size;
	    	  unsigned char *byte_pointer;

	    	  byte_pointer = (unsigned char*) & job_number;

	    	  byte_pointer[0] = job_number_and_result_size[0];
	    	  byte_pointer[1] = job_number_and_result_size[1];
	    	  byte_pointer[2] = job_number_and_result_size[2];
	    	  byte_pointer[3] = job_number_and_result_size[3];

	    	  byte_pointer = (unsigned char*) & result_size;

			  byte_pointer[0] = job_number_and_result_size[4];
			  byte_pointer[1] = job_number_and_result_size[5];
			  byte_pointer[2] = job_number_and_result_size[6];
			  byte_pointer[3] = job_number_and_result_size[7];

	    	  // Here, we should call ProcessResult if the job result_size is zero, since in that case no results will be sent
	    	  if (result_size == 0)
	    	  {
	    		  ProcessResult(NULL);
	    	  }

	    	  //	 		 if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();
	    	  WriteInfoText(wxString::Format("Job %i has finished (result size = %i)!", job_number, result_size));
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
		  {
			 WriteInfoText("All Jobs have finished.");
			  /*
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

void  MyFindParticlesPanel::ProcessResult(JobResult *result_to_process) // this will have to be overidden in the parent clas when i make it.
{
	int number_of_frames;
	int frame_counter;

	long current_time = time(NULL);
	//wxString bitmap_string;
	//wxString plot_string;


	if (current_time - time_of_last_result_update > 5)
	{
		wxString image_filename = my_job_package.jobs[result_to_process->job_number].arguments[0].ReturnStringArgument();
		ArrayOfParticlePositionAssets array_of_assets = ParticlePositionsFromJobResults(result_to_process,image_asset_panel->ReturnGroupMemberID(GroupComboBox->GetSelection(),result_to_process->job_number),1,1,1);
		float radius_in_angstroms = my_job_package.jobs[result_to_process->job_number].arguments[14].ReturnFloatArgument();
		float pixel_size_in_angstroms = my_job_package.jobs[result_to_process->job_number].arguments[1].ReturnFloatArgument();
		PickingResultsPanel->PickingResultsImagePanel->allow_editing_of_coordinates = false;
		PickingResultsPanel->Draw(image_filename, array_of_assets, radius_in_angstroms, pixel_size_in_angstroms);

		time_of_last_result_update = time(NULL);
	}


	my_job_tracker.MarkJobFinished();
//	if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();

	// store the results..

	if (result_to_process) buffered_results[result_to_process->job_number] = result_to_process;

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




void MyFindParticlesPanel::WriteResultToDataBase()
{

	long counter;
	int frame_counter;
	int array_location;
	bool have_errors = false;
	int current_asset;
	bool restrain_astigmatism;
	bool find_additional_phase_shift;
	float min_phase_shift;
	float max_phase_shift;
	float phase_shift_step;
	float tolerated_astigmatism;
	wxString current_table_name;
	wxDateTime now = wxDateTime::Now();


	// find the current highest alignment number in the database, then increment by one

	int starting_picking_id = main_frame->current_project.database.ReturnHighestPickingID();
	int picking_id = starting_picking_id + 1;
	int picking_job_id =  main_frame->current_project.database.ReturnHighestPickingJobID() + 1;


	// Record the parameters we used to pick
	main_frame->current_project.database.BeginBatchInsert("PARTICLE_PICKING_LIST",14,
																						"PICKING_ID",
																						"PICKING_JOB_ID",
																						"DATETIME_OF_RUN",
																						"PARENT_IMAGE_ASSET_ID",
																						"PICKING_ALGORITHM",
																						"CHARACTERISTIC_RADIUS",
																						"MAXIMUM_RADIUS",
																						"THRESHOLD_PEAK_HEIGHT",
																						"HIGHEST_RESOLUTION_USED_IN_PICKING",
																						"MIN_DIST_FROM_EDGES",
																						"AVOID_HIGH_VARIANCE",
																						"AVOID_HIGH_LOW_MEAN",
																						"NUM_BACKGROUND_BOXES",
																						"MANUAL_EDIT");
	picking_id = starting_picking_id + 1;
	for (int counter = 0; counter < my_job_tracker.total_number_of_jobs; counter ++ )
	{
		main_frame->current_project.database.AddToBatchInsert("iiliirrrriiiii", 		picking_id,
																						picking_job_id,
																						(long int) now.GetAsDOS(),
																						image_asset_panel->ReturnGroupMemberID(GroupComboBox->GetCurrentSelection(),counter),
																						PickingAlgorithmComboBox->GetSelection(),
																						CharacteristicParticleRadiusNumericCtrl->ReturnValue(),
																						MaximumParticleRadiusNumericCtrl->ReturnValue(),
																						ThresholdPeakHeightNumericCtrl->ReturnValue(),
																						HighestResolutionNumericCtrl->ReturnValue(),
																						MinimumDistanceFromEdgesSpinCtrl->GetValue(),
																						AvoidHighVarianceAreasCheckBox->GetValue(),
																						AvoidAbnormalLocalMeanAreasCheckBox->GetValue(),
																						NumberOfBackgroundBoxesSpinCtrl->GetValue(),
																						0);
		picking_id ++;
	}
	main_frame->current_project.database.EndBatchInsert();

	// Remove group members and assets from the database, one group at a time
	int parent_id;
	wxString sql_command;
	for (int group_counter = 1; group_counter < particle_position_asset_panel->all_groups_list->number_of_groups; group_counter ++)
	{
		for (int job_counter = 0; job_counter < my_job_tracker.total_number_of_jobs; job_counter ++ )
		{
			parent_id = image_asset_panel->ReturnGroupMemberID(GroupComboBox->GetSelection(),job_counter);
			main_frame->current_project.database.RemoveParticlePositionsWithGivenParentImageIDFromGroup(group_counter,parent_id);
		}
	}

	// Remove from particle_position_assets assets which have a parent_id which is from picking_job_id that we've just done
	main_frame->current_project.database.RemoveParticlePositionAssetsPickedFromImagesAlsoPickedByGivenPickingJobID(picking_job_id);


	// Grab the results and build an array of particle position assets
	ArrayOfParticlePositionAssets array_of_assets;
	ArrayOfParticlePositionAssets temp_array_of_assets;
	int starting_asset_id = 0;
	if (starting_picking_id > 0) starting_asset_id = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(wxString::Format("SELECT MAX(POSITION_ID) FROM PARTICLE_PICKING_RESULTS_%i",picking_job_id-1));
	picking_id = starting_picking_id + 1;
	for (int counter = 0; counter < my_job_tracker.total_number_of_jobs; counter ++ )
	{
		temp_array_of_assets = ParticlePositionsFromJobResults(&buffered_results[counter],image_asset_panel->ReturnGroupMemberID(GroupComboBox->GetSelection(),counter),picking_job_id,picking_id,starting_asset_id);
		WX_APPEND_ARRAY(array_of_assets,temp_array_of_assets);
		starting_asset_id += temp_array_of_assets.GetCount();
		picking_id++;
	}



	// Now that we have our array of assets, let's add them to the database
	main_frame->current_project.database.CreateParticlePickingResultsTable(picking_job_id);
	main_frame->current_project.database.AddArrayOfParticlePositionAssetsToResultsTable(picking_job_id,&array_of_assets);
	main_frame->current_project.database.AddArrayOfParticlePositionAssetsToAssetsTable(&array_of_assets);



	// At this point, the database should be up-to-date
	particle_position_asset_panel->ImportAllFromDatabase();


	particle_position_asset_panel->is_dirty = true;
	picking_results_panel->is_dirty = true;

}

ArrayOfParticlePositionAssets MyFindParticlesPanel::ParticlePositionsFromJobResults(JobResult *job_result, const int &parent_image_id, const int &picking_job_id, const int &picking_id, const int &starting_asset_id)
{
	ParticlePositionAsset temp_asset;
	ArrayOfParticlePositionAssets array_of_assets;
	int address_within_results = 0;


	temp_asset.pick_job_id = picking_job_id;
	temp_asset.asset_id = starting_asset_id;

	temp_asset.parent_id = parent_image_id;
	temp_asset.picking_id = picking_id;
	// Loop over picked coordinates
	for (int particle_counter = 0; particle_counter < job_result->result_size / 5; particle_counter ++ )
	{
		address_within_results = particle_counter * 5;
		// Finish setting up the asset. We use an ID that hasn't been used for any other position asset previously.
		temp_asset.asset_id ++;
		temp_asset.x_position = job_result->result_data[address_within_results + 0];
		temp_asset.y_position = job_result->result_data[address_within_results + 1];
		temp_asset.peak_height = job_result->result_data[address_within_results + 2];

		//
		array_of_assets.Add(temp_asset);
	}

	return array_of_assets;
}
