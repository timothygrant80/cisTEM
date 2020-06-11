//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

// extern MyMovieAssetPanel *movie_asset_panel;
extern MyImageAssetPanel *image_asset_panel;
extern MyVolumeAssetPanel *volume_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyMainFrame *main_frame;
extern MatchTemplateResultsPanel *match_template_results_panel;

MatchTemplatePanel::MatchTemplatePanel( wxWindow* parent )
:
MatchTemplateParentPanel( parent )
{
	// Set variables

	my_job_id = -1;
	running_job = false;

	group_combo_is_dirty = false;
	run_profiles_are_dirty = false;

#ifndef ENABLEGPU
	UseGpuCheckBox->Show(false);
#endif

	SetInfo();
	FillGroupComboBox();
	FillRunProfileComboBox();

	wxSize input_size = InputSizer->GetMinSize();
	input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
	input_size.y = -1;
	ExpertPanel->SetMinSize(input_size);
	ExpertPanel->SetSize(input_size);

	ResetDefaults();
//	EnableMovieProcessingIfAppropriate();

	result_bitmap.Create(1,1, 24);
	time_of_last_result_update = time(NULL);

	DefocusSearchRangeNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
	DefocusSearchStepNumericCtrl->SetMinMaxValue(1.0f, FLT_MAX);
	PixelSizeSearchRangeNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
	PixelSizeSearchStepNumericCtrl->SetMinMaxValue(0.01f, FLT_MAX);
	HighResolutionLimitNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);

	SymmetryComboBox->Clear();
  	SymmetryComboBox->Append("C1");
  	SymmetryComboBox->Append("C2");
  	SymmetryComboBox->Append("C3");
  	SymmetryComboBox->Append("C4");
  	SymmetryComboBox->Append("D2");
  	SymmetryComboBox->Append("D3");
  	SymmetryComboBox->Append("D4");
  	SymmetryComboBox->Append("I");
  	SymmetryComboBox->Append("I2");
  	SymmetryComboBox->Append("O");
  	SymmetryComboBox->Append("T");
  	SymmetryComboBox->Append("T2");
  	SymmetryComboBox->SetSelection(0);

  	GroupComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MatchTemplatePanel::OnGroupComboBox, this);

}

/*
void MatchTemplatePanel::EnableMovieProcessingIfAppropriate()
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
*/

void MatchTemplatePanel::ResetAllDefaultsClick( wxCommandEvent& event )
{
	ResetDefaults();
}

void MatchTemplatePanel::OnInfoURL(wxTextUrlEvent& event)
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

void MatchTemplatePanel::Reset()
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
	InputPanel->Show(true);
	//graph_is_hidden = true;
	InfoPanel->Show(true);

	ResultsPanel->Clear();

	if (running_job == true)
	{
		main_frame->job_controller.KillJob(my_job_id);
		cached_results.Clear();

		running_job = false;
	}


	ResetDefaults();
	Layout();
}

void MatchTemplatePanel::ResetDefaults()
{
	OutofPlaneStepNumericCtrl->ChangeValueFloat(2.5);
	InPlaneStepNumericCtrl->ChangeValueFloat(1.5);
	MinPeakRadiusNumericCtrl->ChangeValueFloat(10.0f);

	DefocusSearchYesRadio->SetValue(true);
	PixelSizeSearchNoRadio->SetValue(true);

	SymmetryComboBox->SetValue("C1");

#ifdef ENABLEGPU
  	UseGpuCheckBox->SetValue(true);
#else
  	UseGpuCheckBox->SetValue(false); // Already disabled, but also set to un-ticked for visual consistency.
#endif

	DefocusSearchRangeNumericCtrl->ChangeValueFloat(1200.0f);
	DefocusSearchStepNumericCtrl->ChangeValueFloat(200.0f);
	PixelSizeSearchRangeNumericCtrl->ChangeValueFloat(0.05f);
	PixelSizeSearchStepNumericCtrl->ChangeValueFloat(0.01f);

//	AssetGroup active_group;
	active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection()]);
	if (active_group.number_of_members > 0)
	{
		ImageAsset *current_image;
		current_image = image_asset_panel->ReturnAssetPointer(GroupComboBox->GetSelection());
		HighResolutionLimitNumericCtrl->ChangeValueFloat(2.0f * current_image->pixel_size);
	}
}

void MatchTemplatePanel::OnGroupComboBox(wxCommandEvent &event)
{
//	ResetDefaults();
//	AssetGroup active_group;

	active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection()]);

	if (active_group.number_of_members > 0)
	{

		ImageAsset *current_image;
		current_image = image_asset_panel->ReturnAssetPointer(active_group.members[0]);
		HighResolutionLimitNumericCtrl->ChangeValueFloat(2.0f * current_image->pixel_size);

	}


	if (GroupComboBox->GetCount() > 0 && main_frame->current_project.is_open == true) all_images_have_defocus_values = CheckGroupHasDefocusValues();

	if (all_images_have_defocus_values == true && PleaseEstimateCTFStaticText->IsShown() == true)
	{

		PleaseEstimateCTFStaticText->Show(false);
		Layout();
	}
	else
	if (all_images_have_defocus_values == false && PleaseEstimateCTFStaticText->IsShown() == false)
	{
		PleaseEstimateCTFStaticText->Show(true);
		Layout();
	}

}

void MatchTemplatePanel::SetInfo()
{
/*	#include "icons/ctffind_definitions.cpp"
	#include "icons/ctffind_diagnostic_image.cpp"
	#include "icons/ctffind_example_1dfit.cpp"

	wxLogNull *suppress_png_warnings = new wxLogNull;
	wxBitmap definitions_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_definitions);
	wxBitmap diagnostic_image_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_diagnostic_image);
	wxBitmap example_1dfit_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_example_1dfit);
	delete suppress_png_warnings;*/

	InfoText->GetCaret()->Hide();

	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("Match Templates"));
	InfoText->EndFontSize();
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("Blah Blah Blah - See (Rickgauer, 2017)."));
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
	InfoText->WriteText(wxT("The group of image assets to look for templates in"));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Reference Volume : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The volume that will used for the template search."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Run Profile : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The selected run profile will be used to run the job. The run profile describes how the job should be run (e.g. how many processors should be used, and on which different computers).  Run profiles are set in the Run Profile panel, located under settings."));
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
	InfoText->WriteText(wxT("Out of Plane Angular Step : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The angular step that should be used for the out of plane search.  Smaller values may increase accuracy, but will significantly increase the required processing time."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("In Plane Angular Step : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The angular step that should be used for the in plane search.  As with the out of plane angle, smaller values may increase accuracy, but will significantly increase the required processing time."));
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
	InfoText->WriteText(wxT("Rickgauer J.P., Grigorieff N., Denk W."));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2017. Single-protein detection in crowded molecular environments in cryo-EM images. Elife 6, e25648.. "));
	InfoText->BeginURL("http://doi.org/10.7554/eLife.25648");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.7554/eLife.25648"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->EndSuppressUndo();
}

void MatchTemplatePanel::FillGroupComboBox()
{
	GroupComboBox->FillComboBox(true);

	if (GroupComboBox->GetCount() > 0 && main_frame->current_project.is_open == true) all_images_have_defocus_values = CheckGroupHasDefocusValues();

	if (all_images_have_defocus_values == true && PleaseEstimateCTFStaticText->IsShown() == true)
	{
		PleaseEstimateCTFStaticText->Show(false);
		Layout();
	}
	else
	if (all_images_have_defocus_values == false && PleaseEstimateCTFStaticText->IsShown() == false)
	{
		PleaseEstimateCTFStaticText->Show(true);
		Layout();
	}
}

void MatchTemplatePanel::FillRunProfileComboBox()
{
	RunProfileComboBox->FillWithRunProfiles();
}

bool MatchTemplatePanel::CheckGroupHasDefocusValues()
{
	wxArrayLong images_with_defocus_values = main_frame->current_project.database.ReturnLongArrayFromSelectCommand("SELECT DISTINCT IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS");
	long current_image_id;
	int images_with_defocus_counter;
	bool image_was_found;

	for (int image_in_group_counter = 0; image_in_group_counter < image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection()); image_in_group_counter++ )
	{
		current_image_id = image_asset_panel->all_assets_list->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), image_in_group_counter))->asset_id;
		image_was_found = false;

		for (images_with_defocus_counter = 0; images_with_defocus_counter < images_with_defocus_values.GetCount(); images_with_defocus_counter++)
		{
			if (images_with_defocus_values[images_with_defocus_counter] == current_image_id)
			{
				image_was_found = true;
				break;
			}
		}

		if (image_was_found == false) return false;
	}

	return true;
}

void MatchTemplatePanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	// are there enough members in the selected group.
	if (main_frame->current_project.is_open == false)
	{
		RunProfileComboBox->Enable(false);
		GroupComboBox->Enable(false);
		StartEstimationButton->Enable(false);
		ReferenceSelectPanel->Enable(false);

	}
	else
	{
		//Enable(true);

		if (running_job == false)
		{
			RunProfileComboBox->Enable(true);
			GroupComboBox->Enable(true);
			ReferenceSelectPanel->Enable(true);
#ifdef ENABLEGPU
			UseGpuCheckBox->Enable(true);
#endif

			if (RunProfileComboBox->GetCount() > 0)
			{
				if (image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection()) > 0 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(RunProfileComboBox->GetSelection()) > 0 && all_images_have_defocus_values == true)
				{
					StartEstimationButton->Enable(true);
				}
				else StartEstimationButton->Enable(false);
			}
			else
			{
				StartEstimationButton->Enable(false);
			}

			if (DefocusSearchYesRadio->GetValue() == true)
			{
				DefocusRangeStaticText->Enable(true);
				DefocusSearchRangeNumericCtrl->Enable(true);
				DefocusStepStaticText->Enable(true);
				DefocusSearchStepNumericCtrl->Enable(true);
			}
			else
			{
				DefocusRangeStaticText->Enable(false);
				DefocusSearchRangeNumericCtrl->Enable(false);
				DefocusStepStaticText->Enable(false);
				DefocusSearchStepNumericCtrl->Enable(false);
			}

			if (PixelSizeSearchYesRadio->GetValue() == true)
			{
				PixelSizeRangeStaticText->Enable(true);
				PixelSizeSearchRangeNumericCtrl->Enable(true);
				PixelSizeStepStaticText->Enable(true);
				PixelSizeSearchStepNumericCtrl->Enable(true);
			}
			else
			{
				PixelSizeRangeStaticText->Enable(false);
				PixelSizeSearchRangeNumericCtrl->Enable(false);
				PixelSizeStepStaticText->Enable(false);
				PixelSizeSearchStepNumericCtrl->Enable(false);
			}
		}
		else
		{
			GroupComboBox->Enable(false);
			ReferenceSelectPanel->Enable(false);
			RunProfileComboBox->Enable(false);
			UseGpuCheckBox->Enable(false); // Doesn't matter if ENABLEGPU
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

		if (volumes_are_dirty == true)
		{
			ReferenceSelectPanel->FillComboBox();
			volumes_are_dirty = false;
		}
	}
}

void MatchTemplatePanel::StartEstimationClick( wxCommandEvent& event )
{

	active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection()]);

	float resolution_limit;
	float orientations_per_process;
	float current_orientation_counter;

	int job_counter;
	int number_of_rotations = 0;
	int number_of_defocus_positions;
	int number_of_pixel_size_positions;

	bool use_gpu;

	int image_number_for_gui;
	int number_of_jobs_per_image_in_gui;
	int number_of_jobs;

	double voltage_kV;
	double spherical_aberration_mm;
	double amplitude_contrast;
	double defocus1;
	double defocus2;
	double defocus_angle;
	double phase_shift;
	double iciness;

	input_image_filenames.Clear();
	cached_results.Clear();

	ResultsPanel->Clear();

	// Package the job details..

	EulerSearch	*current_image_euler_search;
	ImageAsset *current_image;
	VolumeAsset *current_volume;

	current_volume = volume_asset_panel->ReturnAssetPointer(ReferenceSelectPanel->GetSelection());
	ref_box_size_in_pixels = current_volume->x_size / current_volume->pixel_size;

	ParameterMap parameter_map;
	parameter_map.SetAllTrue();

	float wanted_out_of_plane_angular_step = OutofPlaneStepNumericCtrl->ReturnValue();
	float wanted_in_plane_angular_step = InPlaneStepNumericCtrl->ReturnValue();

	float defocus_search_range;
	float defocus_step;
	float pixel_size_search_range;
	float pixel_size_step;

	if (DefocusSearchYesRadio->GetValue() == true)
	{
		defocus_search_range = DefocusSearchRangeNumericCtrl->ReturnValue();
		defocus_step = DefocusSearchStepNumericCtrl->ReturnValue();
	}
	else
	{
		defocus_search_range = 0.0f;
		defocus_step = 0.0f;
	}

	if (PixelSizeSearchYesRadio->GetValue() == true)
	{

		pixel_size_search_range = PixelSizeSearchRangeNumericCtrl->ReturnValue();
		pixel_size_step = PixelSizeSearchStepNumericCtrl->ReturnValue();
	}
	else
	{
		pixel_size_search_range = 0.0f;
		pixel_size_step = 0.0f;
	}

	float min_peak_radius = MinPeakRadiusNumericCtrl->ReturnValue();

	if (UseGpuCheckBox->GetValue() == true)
	{
		use_gpu = true;
	}
	else
	{
		use_gpu = false;
	}

	wxString wanted_symmetry = SymmetryComboBox->GetValue();
	wanted_symmetry = SymmetryComboBox->GetValue().Upper();
	float high_resolution_limit = HighResolutionLimitNumericCtrl->ReturnValue();

	wxPrintf("\n\nWanted symmetry %s, Defocus Range %3.3f, Defocus Step %3.3f\n",wanted_symmetry,defocus_search_range,defocus_step);

	RunProfile active_refinement_run_profile = run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()];

	int number_of_processes = active_refinement_run_profile.ReturnTotalJobs();

	// how many jobs are there going to be..

	// get first image to make decisions about how many jobs.. .we assume this is representative.


	current_image = image_asset_panel->ReturnAssetPointer(active_group.members[0]);
	current_image_euler_search = new EulerSearch;
	// WARNING: resolution_limit below is used before its value is set
	current_image_euler_search->InitGrid(wanted_symmetry, wanted_out_of_plane_angular_step, 0.0, 0.0, 360.0, wanted_in_plane_angular_step, 0.0, current_image->pixel_size / resolution_limit, parameter_map, 1);

	if (wanted_symmetry.StartsWith("C1"))
	{
	if (current_image_euler_search->test_mirror == true) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
	{
		current_image_euler_search->theta_max = 180.0f;
	}
	}

	current_image_euler_search->CalculateGridSearchPositions(false);

	if (use_gpu)
	{
		//	number_of_jobs_per_image_in_gui = std::max((int)1,number_of_processes / 2); // Using two threads in each job
		number_of_jobs_per_image_in_gui = number_of_processes; // Using two threads in each job

		number_of_jobs =  number_of_jobs_per_image_in_gui * active_group.number_of_members;

		wxPrintf("In USEGPU:\n There are %d search positions\nThere are %d jobs per image\n", current_image_euler_search->number_of_search_positions, number_of_jobs_per_image_in_gui);
		delete current_image_euler_search;
	}
	else
	{
		if (active_group.number_of_members >= 5 || current_image_euler_search->number_of_search_positions < number_of_processes * 20) number_of_jobs_per_image_in_gui = number_of_processes;
		else
		if (current_image_euler_search->number_of_search_positions > number_of_processes * 250) number_of_jobs_per_image_in_gui = number_of_processes * 10;
		else number_of_jobs_per_image_in_gui = number_of_processes * 5;

		number_of_jobs = number_of_jobs_per_image_in_gui * active_group.number_of_members;

		delete current_image_euler_search;
	}

// Some settings for testing
//	float defocus_search_range = 1200.0f;
//	float defocus_step = 200.0f;

	// number of rotations

	for (float current_psi = 0.0f; current_psi <= 360.0f; current_psi += wanted_in_plane_angular_step)
	{
		number_of_rotations++;
	}

	current_job_package.Reset(active_refinement_run_profile, "match_template", number_of_jobs);

	expected_number_of_results = 0;
	number_of_received_results = 0;

	// loop over all images..

	OneSecondProgressDialog *my_progress_dialog = new OneSecondProgressDialog ("Preparing Job", "Preparing Job...", active_group.number_of_members, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE| wxPD_APP_MODAL);

	TemplateMatchJobResults temp_result;
	temp_result.input_job_id = -1;
	temp_result.job_type = TEMPLATE_MATCH_FULL_SEARCH;
	temp_result.mask_radius = 0.0f;
	temp_result.min_peak_radius = min_peak_radius;
	temp_result.exclude_above_xy_threshold = false;
	temp_result.xy_change_threshold = 0.0f;

	for (int image_counter = 0; image_counter < active_group.number_of_members; image_counter++)
	{
		image_number_for_gui = image_counter + 1;

		// current image asset

		current_image = image_asset_panel->ReturnAssetPointer(active_group.members[image_counter]);

		// setup the euler search for this image..
		// this needs to be changed when more parameters are added.
		// right now, the resolution is always Nyquist.

		resolution_limit = current_image->pixel_size * 2.0f;
		current_image_euler_search = new EulerSearch;
		current_image_euler_search->InitGrid(wanted_symmetry, wanted_out_of_plane_angular_step, 0.0, 0.0, 360.0, wanted_in_plane_angular_step, 0.0, current_image->pixel_size / resolution_limit, parameter_map, 1);
		if (wanted_symmetry.StartsWith("C1"))
		{
			if (current_image_euler_search->test_mirror == true) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
			{
				current_image_euler_search->theta_max = 180.0f;
			}
		}
		current_image_euler_search->CalculateGridSearchPositions(false);

		if (DefocusSearchYesRadio->GetValue() == true) number_of_defocus_positions = 2 * myround(float(defocus_search_range)/float(defocus_step)) + 1;
		else number_of_defocus_positions = 1;

		if (PixelSizeSearchYesRadio->GetValue() == true) number_of_pixel_size_positions = 2 * myround(float(pixel_size_search_range)/float(pixel_size_step)) + 1;
		else number_of_pixel_size_positions = 1;

		wxPrintf("For Image %li\nThere are %i search positions\nThere are %i jobs per image\n", active_group.members[image_counter],current_image_euler_search->number_of_search_positions, number_of_jobs_per_image_in_gui);
		wxPrintf("Calculating %i correlation maps\n", current_image_euler_search->number_of_search_positions * number_of_rotations * number_of_defocus_positions * number_of_pixel_size_positions);
		// how many orientations will each process do for this image..
		expected_number_of_results += current_image_euler_search->number_of_search_positions * number_of_rotations * number_of_defocus_positions * number_of_pixel_size_positions;
		orientations_per_process = float(current_image_euler_search->number_of_search_positions) / float(number_of_jobs_per_image_in_gui);
		if (orientations_per_process < 1) orientations_per_process = 1;

		int number_of_previous_template_matches =  main_frame->current_project.database.ReturnNumberOfPreviousTemplateMatchesByAssetID(current_image->asset_id);
		main_frame->current_project.database.GetCTFParameters(current_image->ctf_estimation_id,voltage_kV,spherical_aberration_mm,amplitude_contrast,defocus1,defocus2,defocus_angle,phase_shift, iciness);

		wxString mip_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath();
		mip_output_file += wxString::Format("/%s_mip_%i_%i.mrc", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);

		wxString best_psi_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath();
		best_psi_output_file += wxString::Format("/%s_psi_%i_%i.mrc", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);

		wxString best_theta_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath();
		best_theta_output_file += wxString::Format("/%s_theta_%i_%i.mrc", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);

		wxString best_phi_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath();
		best_phi_output_file += wxString::Format("/%s_phi_%i_%i.mrc", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);


		wxString best_defocus_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath();
		best_defocus_output_file += wxString::Format("/%s_defocus_%i_%i.mrc", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);

		wxString best_pixel_size_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath();
		best_pixel_size_output_file += wxString::Format("/%s_pixel_size_%i_%i.mrc", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);

		wxString scaled_mip_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath();
		scaled_mip_output_file += wxString::Format("/%s_scaled_mip_%i_%i.mrc", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);

		wxString output_histogram_file = main_frame->current_project.template_matching_asset_directory.GetFullPath();
		output_histogram_file += wxString::Format("/%s_histogram_%i_%i.txt", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);

		wxString output_result_file = main_frame->current_project.template_matching_asset_directory.GetFullPath();
		output_result_file += wxString::Format("/%s_plotted_result_%i_%i.mrc", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);

		wxString correlation_avg_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath();
		correlation_avg_output_file += wxString::Format("/%s_avg_%i_%i.mrc", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);

		wxString correlation_std_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath();
		correlation_std_output_file += wxString::Format("/%s_std_%i_%i.mrc", current_image->filename.GetName(), current_image->asset_id, number_of_previous_template_matches);

//		wxString correlation_std_output_file = "/dev/null";
		current_orientation_counter = 0;

		wxString 	input_search_image = current_image->filename.GetFullPath();
		wxString 	input_reconstruction = current_volume->filename.GetFullPath();
		float		pixel_size = current_image->pixel_size;

		input_image_filenames.Add(input_search_image);

		float low_resolution_limit = 300.0f; // FIXME set this somehwere that is not buried in the code!

		temp_result.image_asset_id = current_image->asset_id;
		temp_result.job_name = wxString::Format("Full search with %s", current_volume->filename.GetName());
		temp_result.ref_volume_asset_id = current_volume->asset_id;
		wxDateTime now = wxDateTime::Now();
		temp_result.datetime_of_run = (long int) now.GetAsDOS();
		temp_result.symmetry = wanted_symmetry;
		temp_result.pixel_size = pixel_size;
		temp_result.voltage = voltage_kV;
		temp_result.spherical_aberration = spherical_aberration_mm;
		temp_result.amplitude_contrast = amplitude_contrast;
		temp_result.defocus1 = defocus1;
		temp_result.defocus2 = defocus2;
		temp_result.defocus_angle = defocus_angle;
		temp_result.phase_shift = phase_shift;
		temp_result.low_res_limit = low_resolution_limit;
		temp_result.high_res_limit = high_resolution_limit;
		temp_result.out_of_plane_step = wanted_out_of_plane_angular_step;
		temp_result.in_plane_step = wanted_in_plane_angular_step;
		temp_result.defocus_search_range = defocus_search_range;
		temp_result.defocus_step = defocus_step;
		temp_result.pixel_size_search_range = pixel_size_search_range;
		temp_result.pixel_size_step = pixel_size_step;
		temp_result.reference_box_size_in_angstroms = ref_box_size_in_pixels * pixel_size;
		temp_result.mip_filename = mip_output_file;
		temp_result.scaled_mip_filename = scaled_mip_output_file;
		temp_result.psi_filename = best_psi_output_file;
		temp_result.theta_filename = best_theta_output_file;
		temp_result.phi_filename = best_phi_output_file;
		temp_result.defocus_filename = best_defocus_output_file;
		temp_result.pixel_size_filename = best_pixel_size_output_file;
		temp_result.histogram_filename = output_histogram_file;
		temp_result.projection_result_filename = output_result_file;
		temp_result.avg_filename = correlation_avg_output_file;
		temp_result.std_filename = correlation_std_output_file;

		cached_results.Add(temp_result);

		for (job_counter = 0; job_counter < number_of_jobs_per_image_in_gui; job_counter++)
		{


//			float high_resolution_limit = resolution_limit;
			int best_parameters_to_keep = 1;
//			float defocus_search_range = 0.0f;
//			float defocus_step = 0.0f;
			float padding = 1;
			bool ctf_refinement = false;
			float mask_radius_search = 0.0f; //current_volume->x_size; // this is actually not really used...

			wxPrintf("\n\tFor image %i, current_orientation_counter is %f\n",image_number_for_gui,current_orientation_counter);
			if (current_orientation_counter >= current_image_euler_search->number_of_search_positions) current_orientation_counter = current_image_euler_search->number_of_search_positions - 1;
			int first_search_position = myroundint(current_orientation_counter);
			current_orientation_counter += orientations_per_process;
			if (current_orientation_counter >= current_image_euler_search->number_of_search_positions || job_counter == number_of_jobs_per_image_in_gui - 1) current_orientation_counter = current_image_euler_search->number_of_search_positions - 1;
			int last_search_position = myroundint(current_orientation_counter);
			current_orientation_counter++;

			wxString directory_for_results = main_frame->current_project.image_asset_directory.GetFullPath();
//			wxString directory_for_results = main_frame->ReturnScratchDirectory();


			//wxPrintf("%i = %i - %i\n", job_counter, first_search_position, last_search_position);


			current_job_package.AddJob("ttffffffffffifffffbfftttttttttftiiiitttfb",	input_search_image.ToUTF8().data(),
																	input_reconstruction.ToUTF8().data(),
																	pixel_size,
																	voltage_kV,
																	spherical_aberration_mm,
																	amplitude_contrast,
																	defocus1,
																	defocus2,
																	defocus_angle,
																	low_resolution_limit,
																	high_resolution_limit,
																	wanted_out_of_plane_angular_step,
																	best_parameters_to_keep,
																	defocus_search_range,
																	defocus_step,
																	pixel_size_search_range,
																	pixel_size_step,
																	padding,
																	ctf_refinement,
																	mask_radius_search,
																	phase_shift,
																	mip_output_file.ToUTF8().data(),
																	best_psi_output_file.ToUTF8().data(),
																	best_theta_output_file.ToUTF8().data(),
																	best_phi_output_file.ToUTF8().data(),
																	best_defocus_output_file.ToUTF8().data(),
																	best_pixel_size_output_file.ToUTF8().data(),
																	scaled_mip_output_file.ToUTF8().data(),
																	correlation_avg_output_file.ToUTF8().data(),
																	wanted_symmetry.ToUTF8().data(),
																	wanted_in_plane_angular_step,
																	output_histogram_file.ToUTF8().data(),
																	first_search_position,
																	last_search_position,
																	image_number_for_gui,
																	number_of_jobs_per_image_in_gui,
																	correlation_std_output_file.ToUTF8().data(),
																	directory_for_results.ToUTF8().data(),
																	output_result_file.ToUTF8().data(),
																	min_peak_radius,
																	use_gpu);
		}

		delete current_image_euler_search;
		my_progress_dialog->Update(image_counter + 1);
	}


	my_progress_dialog->Destroy();

	// Get ID's from database for writing results as they come in..

	template_match_id = main_frame->current_project.database.ReturnHighestTemplateMatchID() + 1;
	template_match_job_id =  main_frame->current_project.database.ReturnHighestTemplateMatchJobID() + 1;

	// launch a controller

	my_job_id = main_frame->job_controller.AddJob(this, run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()].gui_address);

	if (my_job_id != -1)
	{
		SetNumberConnectedTextToZeroAndStartTracking();

		StartPanel->Show(false);
		ProgressPanel->Show(true);
		InputPanel->Show(false);

		ExpertPanel->Show(false);
		InfoPanel->Show(false);
		OutputTextPanel->Show(true);
		ResultsPanel->Show(true);

		GroupComboBox->Enable(false);
		Layout();
	}


	ProgressBar->Pulse();
}

void MatchTemplatePanel::HandleSocketTemplateMatchResultReady(wxSocketBase *connected_socket, int &image_number, float &threshold_used, ArrayOfTemplateMatchFoundPeakInfos &peak_infos, ArrayOfTemplateMatchFoundPeakInfos &peak_changes)
{
	// result is available for an image..

	cached_results[image_number - 1].found_peaks.Clear();
	cached_results[image_number - 1].found_peaks = peak_infos;
	cached_results[image_number - 1].used_threshold = threshold_used;

	ResultsPanel->SetActiveResult(cached_results[image_number - 1]);

	// write to database..

	main_frame->current_project.database.Begin();

	cached_results[image_number - 1].job_id = template_match_job_id;
	main_frame->current_project.database.AddTemplateMatchingResult(template_match_id, cached_results[image_number - 1]);
	template_match_id++;

	main_frame->current_project.database.SetActiveTemplateMatchJobForGivenImageAssetID(cached_results[image_number - 1].image_asset_id, template_match_job_id);
	main_frame->current_project.database.Commit();
	match_template_results_panel->is_dirty = true;
}

void MatchTemplatePanel::FinishButtonClick( wxCommandEvent& event )
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
	InputPanel->Show(true);

	ExpertPanel->Show(true);

	running_job = false;
	Layout();
}

void MatchTemplatePanel::TerminateButtonClick( wxCommandEvent& event )
{
	// kill the job, this will kill the socket to terminate downstream processes
	// - this will have to be improved when clever network failure is incorporated


	main_frame->job_controller.KillJob(my_job_id);

	WriteInfoText("Terminated Job");
	TimeRemainingText->SetLabel("Time Remaining : Terminated");
	CancelAlignmentButton->Show(false);
	FinishButton->Show(true);
	ProgressPanel->Layout();
	cached_results.Clear();

	//running_job = false;
}


void MatchTemplatePanel::WriteInfoText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void MatchTemplatePanel::WriteErrorText(wxString text_to_write)
{
	 output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
	 output_textctrl->AppendText(text_to_write);

	 if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void MatchTemplatePanel::OnSocketJobResultMsg(JobResult &received_result)
{
	if (received_result.result_size > 0)
	{
		ProcessResult(&received_result);
	}
}

void MatchTemplatePanel::OnSocketJobResultQueueMsg(ArrayofJobResults &received_queue)
{
	for (int counter = 0; counter < received_queue.GetCount(); counter++)
	{
		ProcessResult(&received_queue.Item(counter));
	}
}

void MatchTemplatePanel::SetNumberConnectedText(wxString wanted_text)
{
	NumberConnectedText->SetLabel(wanted_text);
}

void MatchTemplatePanel::SetTimeRemainingText(wxString wanted_text)
{
	TimeRemainingText->SetLabel(wanted_text);
}

void MatchTemplatePanel::OnSocketAllJobsFinished()
{
	ProcessAllJobsFinished();
}



void  MatchTemplatePanel::ProcessResult(JobResult *result_to_process) // this will have to be overidden in the parent clas when i make it.
{

	long current_time = time(NULL);
	wxString bitmap_string;
	wxString plot_string;

	number_of_received_results++;


	if (number_of_received_results == 1)
	{
		current_job_starttime = current_time;
		time_of_last_update = 0;
	}
	else
	if (current_time != time_of_last_update)
	{
		int current_percentage;
		current_percentage = myroundint(float(number_of_received_results) / float(expected_number_of_results) * 100.0f);

		time_of_last_update = current_time;
		if (current_percentage > 100) current_percentage = 100;
		ProgressBar->SetValue(current_percentage);

		long job_time = current_time - current_job_starttime;
		float seconds_per_job = float(job_time) / float(number_of_received_results - 1);

		long seconds_remaining;
		seconds_remaining = float(expected_number_of_results - number_of_received_results) * seconds_per_job;

		wxTimeSpan time_remaining = wxTimeSpan(0,0,seconds_remaining);
		TimeRemainingText->SetLabel(time_remaining.Format("Time Remaining : %Hh:%Mm:%Ss"));
	}


	// results should be ..

	// Defocus 1 (Angstroms)
	// Defocus 2 (Angstroms)
	// Astigmatism Angle (degrees)
	// Additional phase shift (e.g. from phase plate) radians
	// Score
	// Resolution (Angstroms) to which Thon rings are well fit by the CTF
	// Reolution (Angstroms) at which aliasing was detected

/*

	if (current_time - time_of_last_result_update > 5)
	{
		// we need the filename of the image..

		wxString image_filename = image_asset_panel->ReturnAssetPointer(active_group.members[result_to_process->job_number])->filename.GetFullPath();

		ResultsPanel->Draw(my_job_package.jobs[result_to_process->job_number].arguments[3].ReturnStringArgument(), my_job_package.jobs[result_to_process->job_number].arguments[16].ReturnBoolArgument(), result_to_process->result_data[0], result_to_process->result_data[1], result_to_process->result_data[2], result_to_process->result_data[3], result_to_process->result_data[4], result_to_process->result_data[5], result_to_process->result_data[6], image_filename);
		time_of_last_result_update = time(NULL);
	}
*/

//	my_job_tracker.MarkJobFinished();
//	if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();

	// store the results..
	//buffered_results[result_to_process->job_number] = result_to_process;

}


void  MatchTemplatePanel::ProcessAllJobsFinished()
{

	MyDebugAssertTrue(my_job_tracker.total_number_of_finished_jobs == my_job_tracker.total_number_of_jobs,"In ProcessAllJobsFinished, but total_number_of_finished_jobs != total_number_of_jobs. Oops.");

	// Update the GUI with project timings
	extern MyOverviewPanel *overview_panel;
	overview_panel->SetProjectInfo();

	//
	WriteResultToDataBase();
	match_template_results_panel->is_dirty = true;

	// let the FindParticles panel check whether any of the groups are now ready to be picked
	//extern MyFindParticlesPanel *findparticles_panel;
	//findparticles_panel->CheckWhetherGroupsCanBePicked();

	cached_results.Clear();

	// Kill the job (in case it isn't already dead)
	main_frame->job_controller.KillJob(my_job_id);

	WriteInfoText("All Jobs have finished.");
	ProgressBar->SetValue(100);
	TimeRemainingText->SetLabel("Time Remaining : All Done!");
	CancelAlignmentButton->Show(false);
	FinishButton->Show(true);
	ProgressPanel->Layout();
}


void MatchTemplatePanel::WriteResultToDataBase()
{
	// I have moved this to HandleSocketTemplateMatchResultReady so that things are done one result at at time.
	/*
	// find the current highest template match numbers in the database, then increment by one

	int template_match_id = main_frame->current_project.database.ReturnHighestTemplateMatchID() + 1;
	int template_match_job_id =  main_frame->current_project.database.ReturnHighestTemplateMatchJobID() + 1;
	main_frame->current_project.database.Begin();


	for (int counter = 0; counter < cached_results.GetCount(); counter++)
	{
		cached_results[counter].job_id = template_match_job_id;
		main_frame->current_project.database.AddTemplateMatchingResult(template_match_id, cached_results[counter]);
		template_match_id++;
	}

	for (int counter = 0; counter < cached_results.GetCount(); counter++)
	{
		main_frame->current_project.database.SetActiveTemplateMatchJobForGivenImageAssetID(cached_results[counter].image_asset_id, template_match_job_id);
	}

	main_frame->current_project.database.Commit();

	match_template_results_panel->is_dirty = true;
*/
}


void MatchTemplatePanel::UpdateProgressBar()
{
	ProgressBar->SetValue(my_job_tracker.ReturnPercentCompleted());
	TimeRemainingText->SetLabel(my_job_tracker.ReturnRemainingTime().Format("Time Remaining : %Hh:%Mm:%Ss"));
}
