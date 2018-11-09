//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMovieAssetPanel *movie_asset_panel;
extern MyImageAssetPanel *image_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyMovieAlignResultsPanel *movie_results_panel;
extern MyMainFrame *main_frame;

MyAlignMoviesPanel::MyAlignMoviesPanel( wxWindow* parent )
:
AlignMoviesPanel( parent )
{
	// Set variables

	buffered_results = NULL;
	show_expert_options = false;

	// Fill combo box..

	FillGroupComboBox();

	my_job_id = -1;
	running_job = false;

	graph_is_hidden = true;
	group_combo_is_dirty = false;
	run_profiles_are_dirty = false;

	GraphPanel->SpectraPanel->use_auto_contrast = false;

	wxSize input_size = InputSizer->GetMinSize();
	input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
	input_size.y = -1;
	ExpertPanel->SetMinSize(input_size);
	ExpertPanel->SetSize(input_size);

	ResetDefaults();

	SetInfo();

}

void MyAlignMoviesPanel::Reset()
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

	ExpertToggleButton->SetValue(false);
	show_expert_options = false;
	ExpertPanel->Show(false);

	if (running_job == true)
	{
		main_frame->job_controller.KillJob(my_job_id);

		if (buffered_results != NULL)
		{
			delete [] buffered_results;
			buffered_results = NULL;
		}

		running_job = false;
	}

	ResetDefaults();
	Layout();

}

void MyAlignMoviesPanel::ResetDefaults()
{
	minimum_shift_text->ChangeValue("2");
	maximum_shift_text->ChangeValue("40");
	dose_filter_checkbox->SetValue(true);
	restore_power_checkbox->SetValue(true);
	termination_threshold_text->ChangeValue("1");
	max_iterations_spinctrl->SetValue(20);
	bfactor_spinctrl->SetValue(1500);
	mask_central_cross_checkbox->SetValue(true);
	horizontal_mask_spinctrl->SetValue(1);
	vertical_mask_spinctrl->SetValue(1);
	include_all_frames_checkbox->SetValue(true);
	first_frame_spin_ctrl->SetValue(1);
	last_frame_spin_ctrl->SetValue(1);
	SaveScaledSumCheckbox->SetValue(true);
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

	wxLogNull *suppress_png_warnings = new wxLogNull;
	wxBitmap alignment_bmp = wxBITMAP_PNG_FROM_DATA(dlp_alignment);
	delete suppress_png_warnings;

	InfoText->GetCaret()->Hide();

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
	InfoText->WriteText(wxT("Additionally an exposure weighted sum can be calculated which attempts to maximize the signal-to-noise ratio in the final sums by taking into account the radiation damage the sample has suffered as the movie progresses.  This exposure weighting is described in (Grant and Grigorieff, 2015a).  During this step, any anisotropic magnification distortions present in the images can also be corrected (Grant and Grigorieff, 2015b), using the parameters that were entered during movie import. "));
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
	InfoText->WriteText(wxT("Mask Central Cross? : "));
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
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Include All Frames in Sum? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If selected, all of the input frames will be used to create the final sum.  If you wish to specify a subset (e.g. to ignore the first couple of frames), untick this option."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("First Frame to Sum : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If not summing all frames then this is the first frame that will be included in the sum."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Last Frame to Sum : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If not summing all frames then this is the last frame that will be included in the sum. Entering 0 will use the last frame of the movie."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Also Save Scaled Sum? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If selected a scaled version of the sum will be written to disk to use for faster display later in the project."));
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
	InfoText->WriteText(wxT(" 2014. Beam-induced motion correction for sub-megadalton cryo-EM particles. Elife 3, e03665. "));
	InfoText->BeginURL("http://dx.doi.org/10.7554/eLife.03665");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.7554/eLife.03665"));
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
	InfoText->WriteText(wxT("Grant, T., Grigorieff, N.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2015. Automatic estimation and correction of anisotropic magnification distortion in electron microscopes. J. Struct. Biol. 192, 204-208. "));
	InfoText->BeginURL("https://doi.org/10.1016/j.jsb.2015.08.006");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.1016/j.jsb.2015.08.006"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();
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
				if (movie_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection()) > 0 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(RunProfileComboBox->GetSelection()) > 1)
				{
					StartAlignmentButton->Enable(true);
				}
				else StartAlignmentButton->Enable(false);
			}
			else
			{
				StartAlignmentButton->Enable(false);
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

			if (mask_central_cross_checkbox->GetValue() == true)
			{
				horizontal_mask_static_text->Enable(true);
				vertical_mask_static_text->Enable(true);
				horizontal_mask_spinctrl->Enable(true);
				vertical_mask_spinctrl->Enable(true);
			}
			else
			{
				horizontal_mask_static_text->Enable(false);
				vertical_mask_static_text->Enable(false);
				horizontal_mask_spinctrl->Enable(false);
				vertical_mask_spinctrl->Enable(false);
			}

			if (include_all_frames_checkbox->GetValue() == false)
			{
				first_frame_static_text->Enable(true);
				last_frame_static_text->Enable(true);
				first_frame_spin_ctrl->Enable(true);
				last_frame_spin_ctrl->Enable(true);
			}
			else
			{
				first_frame_static_text->Enable(false);
				last_frame_static_text->Enable(false);
				first_frame_spin_ctrl->Enable(false);
				last_frame_spin_ctrl->Enable(false);
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

	GroupComboBox->FillComboBox(true);
}

void MyAlignMoviesPanel::Refresh()
{
	FillGroupComboBox();
	FillRunProfileComboBox();
}

void MyAlignMoviesPanel::FillRunProfileComboBox()
{
	RunProfileComboBox->FillWithRunProfiles();
}

void MyAlignMoviesPanel::StartAlignmentClick( wxCommandEvent& event )
{

	MyDebugAssertTrue(buffered_results == NULL, "Error: buffered results not null")

	active_group.CopyFrom(&movie_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection()]);

	long counter;
	long number_of_jobs = active_group.number_of_members;//movie_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection()); // how many movies in the selected group..

	bool ok_number_conversion;

	double minimum_shift;
	double maximum_shift;

	bool should_dose_filter;
	bool should_restore_power;

	double termination_threshold;
	int max_iterations;
	int bfactor;
	int number_of_processes;

	int first_frame;
	int last_frame;

	int current_asset_id;
	int number_of_previous_alignments;

	bool should_mask_central_cross;
	int horizontal_mask;
	int vertical_mask;

	float current_pixel_size;

	float current_acceleration_voltage;
	float current_dose_per_frame;
	float current_pre_exposure;

	float max_movie_size = 0.0f;
	float current_movie_size;
	float max_movie_size_in_gb;
	float min_memory_in_gb_if_run_on_one_machine;


	time_of_last_graph_update = 0;

	std::string current_filename;
	wxString current_gain_filename;
	wxString output_filename;
	wxFileName buffer_filename;

	wxString amplitude_spectrum_filename;
	bool write_out_amplitude_spectrum = true;

	bool write_out_small_sum_image = SaveScaledSumCheckbox->GetValue();
	wxString small_sum_image_filename;

	bool movie_is_gain_corrected;
	float output_binning_factor;

	bool correct_mag_distortion;
	float mag_distortion_angle;
	float mag_distortion_major_axis_scale;
	float mag_distortion_minor_axis_scale;


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

	// first_frame

	if (include_all_frames_checkbox->GetValue() == true)
	{
		first_frame = 1;
		last_frame = 0;
	}
	else
	{
		first_frame = first_frame_spin_ctrl->GetValue();
		last_frame = last_frame_spin_ctrl->GetValue();
	}

	// allocate space for the buffered results..

	buffered_results = new JobResult[number_of_jobs];

	my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()], "unblur", number_of_jobs);



	OneSecondProgressDialog *my_progress_dialog = new OneSecondProgressDialog ("Preparing Job", "Preparing Job...", number_of_jobs, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE| wxPD_APP_MODAL);

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

		//output_filename = movie_asset_panel->ReturnAssetLongFilename(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter));
		//output_filename.Replace(".mrc", "_ali.mrc", false);

		current_movie_size = (float(movie_asset_panel->ReturnAssetPointer(active_group.members[counter])->x_size) / movie_asset_panel->ReturnAssetBinningFactor(active_group.members[counter])) * (float(movie_asset_panel->ReturnAssetPointer(active_group.members[counter])->y_size) / movie_asset_panel->ReturnAssetBinningFactor(active_group.members[counter]));
		current_movie_size *= movie_asset_panel->ReturnAssetPointer(active_group.members[counter])->number_of_frames;

		max_movie_size = std::max(max_movie_size, current_movie_size);

		current_asset_id = movie_asset_panel->ReturnAssetID(active_group.members[counter]);//movie_asset_panel->ReturnAssetID(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), counter));
		buffer_filename = movie_asset_panel->ReturnAssetShortFilename(active_group.members[counter]);
		number_of_previous_alignments =  main_frame->current_project.database.ReturnNumberOfPreviousMovieAlignmentsByAssetID(current_asset_id);

		output_filename = main_frame->current_project.image_asset_directory.GetFullPath();
		output_filename += wxString::Format("/%s_%i_%i.mrc", buffer_filename.GetName(), current_asset_id, number_of_previous_alignments);

		amplitude_spectrum_filename = main_frame->current_project.image_asset_directory.GetFullPath();
		amplitude_spectrum_filename += wxString::Format("/Spectra/%s_%i_%i.mrc", buffer_filename.GetName(), current_asset_id, number_of_previous_alignments);

		if (write_out_small_sum_image == true)
		{
			small_sum_image_filename = main_frame->current_project.image_asset_directory.GetFullPath();
			small_sum_image_filename += wxString::Format("/Scaled/%s_%i_%i.mrc", buffer_filename.GetName(), current_asset_id, number_of_previous_alignments);
		}
		else
		{
			small_sum_image_filename = "/dev/null";
		}

		current_filename = movie_asset_panel->ReturnAssetLongFilename(active_group.members[counter]).ToStdString();
		current_pixel_size = movie_asset_panel->ReturnAssetPixelSize(active_group.members[counter]);
		current_acceleration_voltage = movie_asset_panel->ReturnAssetAccelerationVoltage(active_group.members[counter]);
		current_dose_per_frame = movie_asset_panel->ReturnAssetDosePerFrame(active_group.members[counter]);
		current_pre_exposure = movie_asset_panel->ReturnAssetPreExposureAmount(active_group.members[counter]);

		current_gain_filename = movie_asset_panel->ReturnAssetGainFilename(active_group.members[counter]);
		movie_is_gain_corrected = current_gain_filename.IsEmpty();

		output_binning_factor = movie_asset_panel->ReturnAssetBinningFactor(active_group.members[counter]);

		correct_mag_distortion = movie_asset_panel->ReturnCorrectMagDistortion(active_group.members[counter]);
		mag_distortion_angle = movie_asset_panel->ReturnMagDistortionAngle(active_group.members[counter]);
		mag_distortion_major_axis_scale = movie_asset_panel->ReturnMagDistortionMajorScale(active_group.members[counter]);
		mag_distortion_minor_axis_scale = movie_asset_panel->ReturnMagDistortionMinorScale(active_group.members[counter]);


		my_job_package.AddJob("ssfffbbfifbiifffbsfbfffbtbtii",current_filename.c_str(), //0
														output_filename.ToUTF8().data(),
														current_pixel_size,
														float(minimum_shift),
														float(maximum_shift),
														should_dose_filter, //5
														should_restore_power,
														float(termination_threshold),
														max_iterations,
														float(bfactor),
														should_mask_central_cross, //10
														horizontal_mask,
														vertical_mask,
														current_acceleration_voltage,
														current_dose_per_frame,
														current_pre_exposure, //15
														movie_is_gain_corrected,
														current_gain_filename.ToStdString().c_str(),
														output_binning_factor,
														correct_mag_distortion,
														mag_distortion_angle,
														mag_distortion_major_axis_scale,
														mag_distortion_minor_axis_scale,
														write_out_amplitude_spectrum,
														amplitude_spectrum_filename.ToStdString().c_str(),
														write_out_small_sum_image,
														small_sum_image_filename.ToStdString().c_str(),
														first_frame,
														last_frame);

		my_progress_dialog->Update(counter + 1);
	}

	my_progress_dialog->Destroy();

	// write out details about the max movies size

	max_movie_size_in_gb = float(max_movie_size) /1073741824.0f;
	max_movie_size_in_gb *= 4;

	min_memory_in_gb_if_run_on_one_machine = max_movie_size_in_gb * run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()].ReturnTotalJobs();

	output_textctrl->AppendText("Approx. memory for each process is ");
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLUE));
	output_textctrl->AppendText(wxString::Format(" %.2f GB", max_movie_size_in_gb));
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(".\nIf running on a single machine, that machine will need at least ");
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLUE));
	output_textctrl->AppendText(wxString::Format(" %.2f GB", min_memory_in_gb_if_run_on_one_machine));
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(" of memory.\nIf you do not have enough memory available you will have to use a run profile with fewer processes.\n");

/*	WriteInfoText(wxString::Format("Approx. memory needed per process is %.2f GB", max_movie_size_in_gb));
	WriteInfoText(wxString::Format("If running on a single machine, that machine will need at least %.2f GB of memory.", min_memory_in_gb_if_run_on_one_machine));
	WriteInfoText("");*/

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

				if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();
				//WriteInfoText(wxString::Format("Job %i has finished!", finished_job));
		}
		else
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

			if (graph_is_hidden == true) ProgressBar->Pulse();

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
		{
			// As soon as it sends us the message that all jobs are finished, the controller should also
			// send timing info - we need to remember this
			long timing_from_controller;
			ReadFromSocket(sock, &timing_from_controller, sizeof(long));
			MyDebugAssertTrue(main_frame->current_project.total_cpu_hours + timing_from_controller / 3600000.0 >= main_frame->current_project.total_cpu_hours,"Oops. Double overflow when summing hours spent on project. Total number before adding: %f. Timing from controller: %li",main_frame->current_project.total_cpu_hours,timing_from_controller);
			main_frame->current_project.total_cpu_hours += timing_from_controller / 3600000.0;
			MyDebugAssertTrue(main_frame->current_project.total_cpu_hours >= 0.0,"Negative total_cpu_hour: %f %li",main_frame->current_project.total_cpu_hours,timing_from_controller);
			main_frame->current_project.total_jobs_run += my_job_tracker.total_number_of_jobs;

			// Update project statistics in the database
			main_frame->current_project.WriteProjectStatisticsToDatabase();

			// Other stuff to do once all jobs finished
			ProcessAllJobsFinished();
			return;
		}


		// Enable input events again.

		sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
		break;
	}



	case wxSOCKET_LOST:
	{

		//MyDebugPrint("Socket Disconnected!!\n");
		//sock->Destroy();
		main_frame->job_controller.KillJobIfSocketExists(sock);
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
		GraphPanel->ClearGraph();
		wxFileName current_filename = wxString(my_job_package.jobs[result_to_process->job_number].arguments[0].ReturnStringArgument());
		wxFileName sum_filename = wxString(my_job_package.jobs[result_to_process->job_number].arguments[1].ReturnStringArgument());
		//GraphPanel->title->SetName(current_filename.GetFullName());

		float current_corrected_pixel_size = my_job_package.jobs[result_to_process->job_number].arguments[2].ReturnFloatArgument() / my_job_package.jobs[result_to_process->job_number].arguments[18].ReturnFloatArgument();
		// if we corrected a mag distortion, we have to adjust the pixel size appropriately.

		if (my_job_package.jobs[result_to_process->job_number].arguments[19].ReturnBoolArgument() == true) // correct mag distortion
		{
			current_corrected_pixel_size = ReturnMagDistortionCorrectedPixelSize(current_corrected_pixel_size, my_job_package.jobs[result_to_process->job_number].arguments[21].ReturnFloatArgument(), my_job_package.jobs[result_to_process->job_number].arguments[22].ReturnFloatArgument());
		}

		float current_nyquist;

		if (current_corrected_pixel_size < 1.4) current_nyquist = 2.8;
		else current_nyquist = current_corrected_pixel_size * 2.0;

		GraphPanel->SpectraNyquistStaticText->SetLabel(wxString::Format(wxT("%.2f Å)   "), current_nyquist));
		GraphPanel->FilenameStaticText->SetLabel(wxString::Format("(%s)"        , current_filename.GetFullName()));

		// draw..

		if (DoesFileExist(my_job_package.jobs[result_to_process->job_number].arguments[24].ReturnStringArgument()) == true)
		{
			GraphPanel->SpectraPanel->PanelImage.QuickAndDirtyReadSlice(my_job_package.jobs[result_to_process->job_number].arguments[24].ReturnStringArgument(), 1);
			GraphPanel->SpectraPanel->should_show = true;
			GraphPanel->SpectraPanel->Refresh();
		}


		for (frame_counter = 0; frame_counter < number_of_frames; frame_counter++)
		{
			GraphPanel->AddPoint(exposure_per_frame * frame_counter, result_to_process->result_data[frame_counter], result_to_process->result_data[frame_counter + number_of_frames]);
		}

		GraphPanel->Draw();

		if (DoesFileExist(my_job_package.jobs[result_to_process->job_number].arguments[26].ReturnStringArgument()) == true)
		{
			GraphPanel->ImageDisplayPanel->ChangeFile(my_job_package.jobs[result_to_process->job_number].arguments[26].ReturnStringArgument(), "");
		}
		else
		if (DoesFileExist(sum_filename.GetFullPath()) == true) GraphPanel->ImageDisplayPanel->ChangeFile(sum_filename.GetFullPath(), sum_filename.GetShortPath());

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

}

void MyAlignMoviesPanel::ProcessAllJobsFinished()
{

	MyDebugAssertTrue(my_job_tracker.total_number_of_finished_jobs == my_job_tracker.total_number_of_jobs,"In ProcessAllJobsFinished, but total_number_of_finished_jobs != total_number_of_jobs. Oops.");

	// Update the GUI with project timings
	extern MyOverviewPanel *overview_panel;
	overview_panel->SetProjectInfo();

	//
	WriteResultToDataBase();

	if (buffered_results != NULL)
	{
		delete [] buffered_results;
		buffered_results = NULL;
	}

	// Kill the job (in case it isn't already dead)
	main_frame->job_controller.KillJob(my_job_id);

	WriteInfoText("All Jobs have finished.");
	ProgressBar->SetValue(100);
	TimeRemainingText->SetLabel("Time Remaining : All Done!");
	CancelAlignmentButton->Show(false);
	FinishButton->Show(true);
	ProgressPanel->Layout();
}

void MyAlignMoviesPanel::WriteResultToDataBase()
{

	long counter;
	int frame_counter;
	int array_location;
	int parent_id;
	bool have_errors = false;

	float x_bin_factor;
	float y_bin_factor;
	float average_bin_factor;

	float corrected_pixel_size;


	wxString current_table_name;
	ImageAsset temp_asset;

	// find the current highest alignment number in the database, then increment by one

	int starting_alignment_id = main_frame->current_project.database.ReturnHighestAlignmentID();
	int alignment_id = starting_alignment_id + 1;
	int alignment_job_id =  main_frame->current_project.database.ReturnHighestAlignmentJobID() + 1;

	// global begin

	main_frame->current_project.database.Begin();

	// loop over all the jobs, and add them..

	main_frame->current_project.database.BeginBatchInsert("MOVIE_ALIGNMENT_LIST", 22, "ALIGNMENT_ID", "DATETIME_OF_RUN", "ALIGNMENT_JOB_ID", "MOVIE_ASSET_ID", "OUTPUT_FILE", "VOLTAGE", "PIXEL_SIZE", "EXPOSURE_PER_FRAME", "PRE_EXPOSURE_AMOUNT", "MIN_SHIFT", "MAX_SHIFT", "SHOULD_DOSE_FILTER", "SHOULD_RESTORE_POWER", "TERMINATION_THRESHOLD", "MAX_ITERATIONS", "BFACTOR", "SHOULD_MASK_CENTRAL_CROSS", "HORIZONTAL_MASK", "VERTICAL_MASK", "SHOULD_INCLUDE_ALL_FRAMES_IN_SUM", "FIRST_FRAME_TO_SUM", "LAST_FRAME_TO_SUM" );

	wxDateTime now = wxDateTime::Now();

	OneSecondProgressDialog *my_progress_dialog = new OneSecondProgressDialog ("Write Results", "Writing results to the database...", my_job_tracker.total_number_of_jobs * 3, this, wxPD_APP_MODAL);


	for (counter = 0; counter < my_job_tracker.total_number_of_jobs; counter++)
	{
		main_frame->current_project.database.AddToBatchInsert("iliitrrrrrriiriiiiiiii", alignment_id,
				                                                                    (long int) now.GetAsDOS(),
																					alignment_job_id,
																					movie_asset_panel->ReturnAssetID(active_group.members[counter]),
																					my_job_package.jobs[counter].arguments[1].ReturnStringArgument().c_str(), // output_filename
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
																					my_job_package.jobs[counter].arguments[12].ReturnIntegerArgument(), // vertical mask
																					include_all_frames_checkbox->GetValue(), // include all frames
																					my_job_package.jobs[counter].arguments[27].ReturnIntegerArgument(), // first_frame
																					my_job_package.jobs[counter].arguments[28].ReturnIntegerArgument() // last_frame
																					);

		alignment_id++;

		my_progress_dialog->Update(counter + 1);

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

		my_progress_dialog->Update(my_job_tracker.total_number_of_jobs + counter + 1);

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
				parent_id = movie_asset_panel->ReturnAssetID(active_group.members[counter]);

				// work out the corrected pixel size
				// so the bin amount, might not be exactly the specified amount, as it is fixed by integer resizing of the image.

				x_bin_factor = float(movie_asset_panel->all_assets_list->ReturnMovieAssetPointer(active_group.members[counter])->x_size) / float(temp_asset.x_size);
				y_bin_factor = float(movie_asset_panel->all_assets_list->ReturnMovieAssetPointer(active_group.members[counter])->y_size) / float(temp_asset.y_size);
				average_bin_factor = (x_bin_factor + y_bin_factor) / 2.0;

				corrected_pixel_size = my_job_package.jobs[counter].arguments[2].ReturnFloatArgument() * average_bin_factor;

				// if we corrected a mag distortion, we have to adjust the pixel size appropriately.

				if (my_job_package.jobs[counter].arguments[19].ReturnBoolArgument() == true) // correct mag distortion
				{
					corrected_pixel_size = ReturnMagDistortionCorrectedPixelSize(corrected_pixel_size, my_job_package.jobs[counter].arguments[21].ReturnFloatArgument(), my_job_package.jobs[counter].arguments[22].ReturnFloatArgument());
				}

				array_location = image_asset_panel->ReturnArrayPositionFromParentID(parent_id);

				// is this image (or a previous version) already an asset?

				if (array_location == -1) // we don't already have an asset from this movie..
				{
					temp_asset.asset_id = image_asset_panel->current_asset_number;
					temp_asset.asset_name = movie_asset_panel->ReturnAssetName(active_group.members[counter]) + "_aligned";
					temp_asset.parent_id = parent_id;
					temp_asset.alignment_id = alignment_id;
					temp_asset.microscope_voltage = my_job_package.jobs[counter].arguments[13].ReturnFloatArgument();

					temp_asset.pixel_size = corrected_pixel_size;

					temp_asset.position_in_stack = 1;
					temp_asset.spherical_aberration = movie_asset_panel->ReturnAssetSphericalAbberation(movie_asset_panel->ReturnArrayPositionFromAssetID(parent_id));
					temp_asset.protein_is_white = movie_asset_panel->ReturnAssetProteinIsWhite(movie_asset_panel->ReturnArrayPositionFromAssetID(parent_id));
					image_asset_panel->AddAsset(&temp_asset);
					main_frame->current_project.database.AddNextImageAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath(), temp_asset.position_in_stack, temp_asset.parent_id, alignment_id, -1, temp_asset.x_size, temp_asset.y_size, temp_asset.microscope_voltage, temp_asset.pixel_size, temp_asset.spherical_aberration, temp_asset.protein_is_white);


				}
				else
				{// TODO:: Rewrite this to use return asset pointer..//
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].filename = my_job_package.jobs[counter].arguments[1].ReturnStringArgument();
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].parent_id = parent_id;
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].alignment_id = alignment_id;
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].microscope_voltage = my_job_package.jobs[counter].arguments[13].ReturnFloatArgument();
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].pixel_size = corrected_pixel_size;
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].position_in_stack = 1;
					reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].spherical_aberration = movie_asset_panel->ReturnAssetSphericalAbberation(movie_asset_panel->ReturnArrayPositionFromAssetID(parent_id));

					main_frame->current_project.database.AddNextImageAsset(reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].asset_id,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].asset_name,
							 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	            my_job_package.jobs[counter].arguments[1].ReturnStringArgument(),
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].position_in_stack,
																											parent_id,
																											alignment_id,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].ctf_estimation_id,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].x_size,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].y_size,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].microscope_voltage,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].pixel_size,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].spherical_aberration,
																											reinterpret_cast <ImageAsset *> (image_asset_panel->all_assets_list->assets)[array_location].protein_is_white);

					image_asset_panel->current_asset_number++;
				}
			}
			else
			{
				my_error->ErrorText->AppendText(wxString::Format(wxT("%s is not a valid MRC file, skipping\n"), temp_asset.ReturnFullPathString()));
				have_errors = true;
			}

			alignment_id++;

			my_progress_dialog->Update(my_job_tracker.total_number_of_jobs + my_job_tracker.total_number_of_jobs + counter + 1);
	}

	my_progress_dialog->Destroy();

	main_frame->current_project.database.EndImageAssetInsert();

	// global commit..

	main_frame->current_project.database.Commit();

	image_asset_panel->is_dirty = true;
	movie_results_panel->is_dirty = true;
	main_frame->DirtyImageGroups();


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


