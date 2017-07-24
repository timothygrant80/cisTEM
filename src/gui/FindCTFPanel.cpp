//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMovieAssetPanel *movie_asset_panel;
extern MyImageAssetPanel *image_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyMainFrame *main_frame;
extern MyFindCTFResultsPanel *ctf_results_panel;

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
	MinPhaseShiftNumericCtrl->SetMinMaxValue(-190, 190);
	MaxPhaseShiftNumericCtrl->SetMinMaxValue(-190, 190);
	PhaseShiftStepNumericCtrl->SetMinMaxValue(0.001, 190);

	EnableMovieProcessingIfAppropriate();

	result_bitmap.Create(1,1, 24);
	time_of_last_result_update = time(NULL);

}


void MyFindCTFPanel::EnableMovieProcessingIfAppropriate()
{
	// Check whether all members of the group have movie parents. If not, make sure we only allow image processing
	MovieRadioButton->Enable(true);
	NoMovieFramesStaticText->Enable(true);
	NoFramesToAverageSpinCtrl->Enable(true);
	for (int counter = 0; counter < image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection()); counter ++ )
	{
		if (image_asset_panel->all_assets_list->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(),counter))->parent_id < 0)
		{
			MovieRadioButton->SetValue(false);
			MovieRadioButton->Enable(false);
			NoMovieFramesStaticText->Enable(false);
			NoFramesToAverageSpinCtrl->Enable(false);
			ImageRadioButton->SetValue(true);
		}
	}
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

	wxLogNull *suppress_png_warnings = new wxLogNull;
	wxBitmap definitions_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_definitions);
	wxBitmap diagnostic_image_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_diagnostic_image);
	wxBitmap example_1dfit_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_example_1dfit);
	delete suppress_png_warnings;

	InfoText->GetCaret()->Hide();

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
	GroupComboBox->FillComboBox(true);
}

void MyFindCTFPanel::FillRunProfileComboBox()
{
	RunProfileComboBox->FillWithRunProfiles();
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
				if (image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection()) > 0 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(RunProfileComboBox->GetSelection()) > 1)
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
			EnableMovieProcessingIfAppropriate();
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
	Freeze();
	NoFramesToAverageSpinCtrl->Enable(true);
	NoMovieFramesStaticText->Enable(true);
	Thaw();
}

void MyFindCTFPanel::OnImageRadioButton(wxCommandEvent& event )
{
	Freeze();
	NoFramesToAverageSpinCtrl->Enable(false);
	NoMovieFramesStaticText->Enable(false);
	Thaw();
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
	long number_of_jobs = image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection()); // how many images / movies in the selected group..

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
	bool 		astigmatism_is_known;
	float		known_astigmatism;
	float		known_astigmatism_angle;
	bool		resample_if_pixel_too_small;
	bool		large_astigmatism_expected;

	wxString	current_gain_filename;
	bool 		movie_is_gain_corrected;

	bool        correct_movie_mag_distortion = false;
	float       movie_mag_distortion_angle = 0.0;
	float       movie_mag_distortion_major_scale = 1.0;
	float       movie_mag_distortion_minor_scale = 1.0;


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
	large_astigmatism_expected = LargeAstigmatismExpectedCheckBox->IsChecked();

	if (RestrainAstigmatismCheckBox->IsChecked() == false) astigmatism_tolerance = -100.0f;
	else astigmatism_tolerance = ToleratedAstigmatismNumericCtrl->ReturnValue();

	if (AdditionalPhaseShiftCheckBox->IsChecked() == true)
	{
		find_additional_phase_shift = true;
		minimum_additional_phase_shift = deg_2_rad(MinPhaseShiftNumericCtrl->ReturnValue());
		maximum_additional_phase_shift = deg_2_rad(MaxPhaseShiftNumericCtrl->ReturnValue());
		additional_phase_shift_search_step = deg_2_rad(PhaseShiftStepNumericCtrl->ReturnValue());
	}
	else
	{
		find_additional_phase_shift = false;
		minimum_additional_phase_shift = 0.0f;
		maximum_additional_phase_shift = 0.0f;
		additional_phase_shift_search_step = 0.0f;
	}


	OneSecondProgressDialog *my_progress_dialog = new OneSecondProgressDialog ("Preparing Job", "Preparing Job...", number_of_jobs, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE| wxPD_APP_MODAL);
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
		// large_asgitmatism_expected (bool);
		// astigmatism_tolerance (float);
		// find_additional_phase_shift (bool);
		// minimum_additional_phase_shift (float);
		// maximum_additional_phase_shift (float);
		///additional_phase_shift_search_step (float);
		// astigmatism_is_known (bool);
		// known_astigmatism (float);
		// known_astigmatism_angle (float);
		// resample_if_pixel_too_small (bool);

		if (input_is_a_movie == true)
		{
			parent_asset_id = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter))->parent_id;
			input_filename = movie_asset_panel->ReturnAssetPointer(movie_asset_panel->ReturnArrayPositionFromAssetID(parent_asset_id))->filename.GetFullPath().ToStdString();
			pixel_size = movie_asset_panel->ReturnAssetPointer(movie_asset_panel->ReturnArrayPositionFromAssetID(parent_asset_id))->pixel_size;
		}
		else
		{
			input_filename = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter))->filename.GetFullPath().ToStdString();
			pixel_size = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter))->pixel_size;
		}

		acceleration_voltage = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter))->microscope_voltage;
		spherical_aberration = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter))->spherical_aberration;


		//output_filename = movie_asset_panel->ReturnAssetLongFilename(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter));
		//output_filename.Replace(".mrc", "_ali.mrc", false);

		current_asset_id = image_asset_panel->ReturnAssetID(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter));
		buffer_filename = image_asset_panel->ReturnAssetShortFilename(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter));
		number_of_previous_estimations =  main_frame->current_project.database.ReturnNumberOfPreviousCTFEstimationsByAssetID(current_asset_id);

		buffer_filename = main_frame->current_project.ctf_asset_directory.GetFullPath();
		buffer_filename += wxString::Format("/%s_CTF_%i.mrc", wxFileName::StripExtension(image_asset_panel->ReturnAssetShortFilename(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter))), number_of_previous_estimations);

		output_diagnostic_filename = buffer_filename.ToStdString();

		// These parameters are not presented in the GUI (yet?)
		astigmatism_is_known = false;
		known_astigmatism = 0.0;
		known_astigmatism_angle = 0.0;
		//astigmatism_is_known = true;
		//known_astigmatism = 385.0;
		//known_astigmatism_angle = 35.0;
		resample_if_pixel_too_small = true;

		if (input_is_a_movie)
		{
			parent_asset_id = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter))->parent_id;
			MovieAsset *current_movie = movie_asset_panel->ReturnAssetPointer(movie_asset_panel->ReturnArrayPositionFromAssetID(parent_asset_id));
			current_gain_filename = current_movie->gain_filename;
			movie_is_gain_corrected = current_gain_filename.IsEmpty();

			correct_movie_mag_distortion = current_movie->correct_mag_distortion;

			if (correct_movie_mag_distortion == true)
			{
				movie_mag_distortion_angle = current_movie->mag_distortion_angle;
				movie_mag_distortion_major_scale = current_movie->mag_distortion_major_scale;
				movie_mag_distortion_minor_scale = current_movie->mag_distortion_minor_scale;
			}
			else
			{
				movie_mag_distortion_angle = 0.0;
				movie_mag_distortion_major_scale = 1.0;
				movie_mag_distortion_minor_scale = 1.0;
            }
        }
		else
		{
			current_gain_filename = "";
			movie_is_gain_corrected = true;
		}


		my_job_package.AddJob("sbisffffifffffbfbfffbffbbsbfff",	input_filename.c_str(), // 0
															input_is_a_movie, // 1
															number_of_frames_to_average, //2
															output_diagnostic_filename.c_str(), // 3
															pixel_size, // 4
															acceleration_voltage, // 5
															spherical_aberration, // 6
															amplitude_contrast, // 7
															box_size, // 8
															minimum_resolution, // 9
															maximum_resolution, // 10
															minimum_defocus, // 11
															maximum_defocus, // 12
															defocus_search_step, // 13
															large_astigmatism_expected, // 14
															astigmatism_tolerance, // 15
															find_additional_phase_shift, // 16
															minimum_additional_phase_shift, // 17
															maximum_additional_phase_shift, // 18
															additional_phase_shift_search_step, // 19
															astigmatism_is_known, // 20
															known_astigmatism, // 21
															known_astigmatism_angle, // 22
															resample_if_pixel_too_small,// 23
															movie_is_gain_corrected,
															current_gain_filename.ToStdString().c_str(),
															correct_movie_mag_distortion,
															movie_mag_distortion_angle,
															movie_mag_distortion_major_scale,
															movie_mag_distortion_minor_scale);

		my_progress_dialog->Update(counter + 1);
	}

	my_progress_dialog->Destroy();

	// launch a controller

	my_job_id = main_frame->job_controller.AddJob(this, run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()].gui_address);

	if (my_job_id != -1)
	{
		if (my_job_package.number_of_jobs + 1 < my_job_package.my_profile.ReturnTotalJobs()) number_of_processes = my_job_package.number_of_jobs + 1;
		else number_of_processes =  my_job_package.my_profile.ReturnTotalJobs();

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
		CTFResultsPanel->Show(true);

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
	CTFResultsPanel->Show(false);
	//graph_is_hidden = true;
	InfoPanel->Show(true);

	if (ExpertToggleButton->GetValue() == true) ExpertPanel->Show(true);
	else ExpertPanel->Show(false);
	running_job = false;
	Layout();

	CTFResultsPanel->CTF2DResultsPanel->should_show = false;
	CTFResultsPanel->CTF2DResultsPanel->Refresh();



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
	  sock->SetFlags(wxSOCKET_BLOCK | wxSOCKET_WAITALL);

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
			  ReadFromSocket(sock, &number_of_connections, 4);

              my_job_tracker.AddConnection();

    //          if (graph_is_hidden == true) ProgressBar->Pulse();

              //WriteInfoText(wxString::Format("There are now %i connections\n", number_of_connections));

              // send the info to the gui
              int total_processes;

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

	extern MyFindParticlesPanel *findparticles_panel;

	int number_of_frames;
	int frame_counter;

	long current_time = time(NULL);
	wxString bitmap_string;
	wxString plot_string;

	// results should be ..

	// Defocus 1 (Angstroms)
	// Defocus 2 (Angstroms)
	// Astigmatism Angle (degrees)
	// Additional phase shift (e.g. from phase plate) radians
	// Score
	// Resolution (Angstroms) to which Thon rings are well fit by the CTF
	// Reolution (Angstroms) at which aliasing was detected



	if (current_time - time_of_last_result_update > 5)
	{
		// we need the filename of the image..

		wxString image_filename = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), result_to_process->job_number))->filename.GetFullPath();

		CTFResultsPanel->Draw(my_job_package.jobs[result_to_process->job_number].arguments[3].ReturnStringArgument(), my_job_package.jobs[result_to_process->job_number].arguments[16].ReturnBoolArgument(), result_to_process->result_data[0], result_to_process->result_data[1], result_to_process->result_data[2], result_to_process->result_data[3], result_to_process->result_data[4], result_to_process->result_data[5], result_to_process->result_data[6], image_filename);
		time_of_last_result_update = time(NULL);
	}

	my_job_tracker.MarkJobFinished();
	if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();

	// store the results..

	buffered_results[result_to_process->job_number] = result_to_process;

	if (my_job_tracker.total_number_of_finished_jobs == my_job_tracker.total_number_of_jobs)
	{
		// job has really finished, so we can write to the database...

		WriteResultToDataBase();

		// let the FindParticles panel check whether any of the groups are now ready to be picked
		findparticles_panel->CheckWhetherGroupsCanBePicked();

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




void MyFindCTFPanel::WriteResultToDataBase()
{

	long counter;
	int frame_counter;
	int array_location;
	bool have_errors = false;
	int image_asset_id;
	int current_asset;
	bool restrain_astigmatism;
	bool find_additional_phase_shift;
	float min_phase_shift;
	float max_phase_shift;
	float phase_shift_step;
	float tolerated_astigmatism;
	wxString current_table_name;


	// find the current highest alignment number in the database, then increment by one

	int starting_ctf_estimation_id = main_frame->current_project.database.ReturnHighestFindCTFID();
	int ctf_estimation_id = starting_ctf_estimation_id + 1;
	int ctf_estimation_job_id =  main_frame->current_project.database.ReturnHighestFindCTFJobID() + 1;

	OneSecondProgressDialog *my_progress_dialog = new OneSecondProgressDialog ("Write Results", "Writing results to the database...", my_job_tracker.total_number_of_jobs * 2, this, wxPD_APP_MODAL);

	// global begin

	main_frame->current_project.database.Begin();

	// loop over all the jobs, and add them..
	main_frame->current_project.database.BeginBatchInsert("ESTIMATED_CTF_PARAMETERS", 31,
			                                                                              "CTF_ESTIMATION_ID",
																						  "CTF_ESTIMATION_JOB_ID",
																						  "DATETIME_OF_RUN",
																						  "IMAGE_ASSET_ID",
																						  "ESTIMATED_ON_MOVIE_FRAMES",
																						  "VOLTAGE",
																						  "SPHERICAL_ABERRATION",
																						  "PIXEL_SIZE",
																						  "AMPLITUDE_CONTRAST",
																						  "BOX_SIZE",
																						  "MIN_RESOLUTION",
																						  "MAX_RESOLUTION",
																						  "MIN_DEFOCUS",
																						  "MAX_DEFOCUS",
																						  "DEFOCUS_STEP",
																						  "RESTRAIN_ASTIGMATISM",
																						  "TOLERATED_ASTIGMATISM",
																						  "FIND_ADDITIONAL_PHASE_SHIFT",
																						  "MIN_PHASE_SHIFT",
																						  "MAX_PHASE_SHIFT",
																						  "PHASE_SHIFT_STEP",
																						  "DEFOCUS1",
																						  "DEFOCUS2",
																						  "DEFOCUS_ANGLE",
																						  "ADDITIONAL_PHASE_SHIFT",
																						  "SCORE",
																						  "DETECTED_RING_RESOLUTION",
																						  "DETECTED_ALIAS_RESOLUTION",
																						  "OUTPUT_DIAGNOSTIC_FILE",
																						  "NUMBER_OF_FRAMES_AVERAGED",
																						  "LARGE_ASTIGMATISM_EXPECTED");



	wxDateTime now = wxDateTime::Now();

	for (counter = 0; counter < my_job_tracker.total_number_of_jobs; counter++)
	{
		image_asset_id = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter))->asset_id;


		if (my_job_package.jobs[counter].arguments[15].ReturnFloatArgument() < 0)
		{
			restrain_astigmatism = false;
			tolerated_astigmatism = 0;
		}
		else
		{
			restrain_astigmatism = true;
			tolerated_astigmatism = my_job_package.jobs[counter].arguments[15].ReturnFloatArgument();
		}

		if ( my_job_package.jobs[counter].arguments[16].ReturnBoolArgument())
		{
			find_additional_phase_shift = true;
			min_phase_shift = my_job_package.jobs[counter].arguments[17].ReturnFloatArgument();
			max_phase_shift = my_job_package.jobs[counter].arguments[18].ReturnFloatArgument();
			phase_shift_step = my_job_package.jobs[counter].arguments[19].ReturnFloatArgument();
		}
		else
		{
			find_additional_phase_shift = false;
			min_phase_shift = 0;
			max_phase_shift = 0;
			phase_shift_step = 0;
		}


		main_frame->current_project.database.AddToBatchInsert("iiliirrrrirrrrririrrrrrrrrrrtii", ctf_estimation_id,
																					 ctf_estimation_job_id,
																					 (long int) now.GetAsDOS(),
																					 image_asset_id,
																					 my_job_package.jobs[counter].arguments[1].ReturnBoolArgument(), // input_is_a_movie
																					 my_job_package.jobs[counter].arguments[5].ReturnFloatArgument(), // voltage
																					 my_job_package.jobs[counter].arguments[6].ReturnFloatArgument(), // spherical_aberration
																					 my_job_package.jobs[counter].arguments[4].ReturnFloatArgument(), // pixel_size
																					 my_job_package.jobs[counter].arguments[7].ReturnFloatArgument(), // amplitude contrast
																					 my_job_package.jobs[counter].arguments[8].ReturnIntegerArgument(), // box_size
																					 my_job_package.jobs[counter].arguments[9].ReturnFloatArgument(), // min resolution
																					 my_job_package.jobs[counter].arguments[10].ReturnFloatArgument(),  // max resolution
																					 my_job_package.jobs[counter].arguments[11].ReturnFloatArgument(), // min defocus
																					 my_job_package.jobs[counter].arguments[12].ReturnFloatArgument(), // max defocus
																					 my_job_package.jobs[counter].arguments[13].ReturnFloatArgument(), // defocus_step
																					 restrain_astigmatism,
																					 tolerated_astigmatism,
																					 find_additional_phase_shift,
																					 min_phase_shift,
																					 max_phase_shift,
																					 phase_shift_step,
																					 buffered_results[counter].result_data[0], // defocus1
																					 buffered_results[counter].result_data[1], // defocus2
																					 buffered_results[counter].result_data[2], // defocus angle
																					 buffered_results[counter].result_data[3], // additional phase shift
																					 buffered_results[counter].result_data[4], // score
																					 buffered_results[counter].result_data[5], // detected ring resolution
																					 buffered_results[counter].result_data[6], // detected aliasing resolution
																					 my_job_package.jobs[counter].arguments[3].ReturnStringArgument().c_str(), // output diagnostic filename
																					 my_job_package.jobs[counter].arguments[2].ReturnIntegerArgument(),  // number of movie frames averaged
																					 my_job_package.jobs[counter].arguments[14].ReturnBoolArgument()); // large astigmatism expected
		ctf_estimation_id++;
		my_progress_dialog->Update(counter + 1);

	}

	main_frame->current_project.database.EndBatchInsert();

	// we need to update the image assets with the correct CTF estimation number..

	ctf_estimation_id = starting_ctf_estimation_id + 1;
	main_frame->current_project.database.BeginImageAssetInsert();

	for (counter = 0; counter < my_job_tracker.total_number_of_jobs; counter++)
	{
		current_asset = image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter);

		main_frame->current_project.database.AddNextImageAsset(image_asset_panel->ReturnAssetPointer(current_asset)->asset_id,
															image_asset_panel->ReturnAssetPointer(current_asset)->asset_name,
															   image_asset_panel->ReturnAssetPointer(current_asset)->filename.GetFullPath(),
															   image_asset_panel->ReturnAssetPointer(current_asset)->position_in_stack,
															   image_asset_panel->ReturnAssetPointer(current_asset)->parent_id,
															   image_asset_panel->ReturnAssetPointer(current_asset)->alignment_id,
															   ctf_estimation_id,
															   image_asset_panel->ReturnAssetPointer(current_asset)->x_size,
															   image_asset_panel->ReturnAssetPointer(current_asset)->y_size,
															   image_asset_panel->ReturnAssetPointer(current_asset)->microscope_voltage,
															   image_asset_panel->ReturnAssetPointer(current_asset)->pixel_size,
															   image_asset_panel->ReturnAssetPointer(current_asset)->spherical_aberration);


		image_asset_panel->ReturnAssetPointer(current_asset)->ctf_estimation_id = ctf_estimation_id;

		ctf_estimation_id++;
		my_progress_dialog->Update(my_job_tracker.total_number_of_jobs + counter + 1);


	}

	main_frame->current_project.database.EndImageAssetInsert();

	// Global Commit

	main_frame->current_project.database.Commit();
	my_progress_dialog->Destroy();
	ctf_results_panel->is_dirty = true;

}


void MyFindCTFPanel::UpdateProgressBar()
{
	TimeRemaining time_left = my_job_tracker.ReturnRemainingTime();
	ProgressBar->SetValue(my_job_tracker.ReturnPercentCompleted());

	TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_left.hours, time_left.minutes, time_left.seconds));
}

