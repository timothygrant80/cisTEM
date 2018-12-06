//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

// extern MyMovieAssetPanel *movie_asset_panel;
extern MyImageAssetPanel *image_asset_panel;
extern MyVolumeAssetPanel *volume_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyMainFrame *main_frame;
// extern MyFindCTFResultsPanel *ctf_results_panel;

MatchTemplatePanel::MatchTemplatePanel( wxWindow* parent )
:
MatchTemplateParentPanel( parent )
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

	ResetDefaults();
//	EnableMovieProcessingIfAppropriate();

	result_bitmap.Create(1,1, 24);
	time_of_last_result_update = time(NULL);

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
	CTFResultsPanel->Show(false);
	//graph_is_hidden = true;
	InfoPanel->Show(true);

	ExpertToggleButton->SetValue(false);
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

	CTFResultsPanel->CTF2DResultsPanel->should_show = false;
	CTFResultsPanel->CTF2DResultsPanel->Refresh();

	ResetDefaults();
	Layout();
}

void MatchTemplatePanel::ResetDefaults()
{
	OutofPlaneStepNumericCtrl->ChangeValueFloat(2.5);
	InPlaneStepNumericCtrl->ChangeValueFloat(1.5);
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
}

void MatchTemplatePanel::FillRunProfileComboBox()
{
	RunProfileComboBox->FillWithRunProfiles();
}

void MatchTemplatePanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	// are there enough members in the selected group.
	if (main_frame->current_project.is_open == false)
	{
		RunProfileComboBox->Enable(false);
		GroupComboBox->Enable(false);
		ExpertToggleButton->Enable(false);
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
			ReferenceSelectPanel->Enable(false);
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

		if (volumes_are_dirty == true)
		{
			ReferenceSelectPanel->FillComboBox();
			volumes_are_dirty = false;
		}


	}




}

void MatchTemplatePanel::OnExpertOptionsToggle(wxCommandEvent& event )
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


void MatchTemplatePanel::StartEstimationClick( wxCommandEvent& event )
{

	MyDebugAssertTrue(buffered_results == NULL, "Error: buffered results not null")
	active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection()]);

	float resolution_limit;
	float orientations_per_process;
	float current_orientation_counter;

	int job_counter;
	int number_of_rotations = 0;
	int number_of_defocus_positions;

	int image_number_for_gui;
	int number_of_jobs_per_image_in_gui;


	// Package the job details..

	EulerSearch	*current_image_euler_search;
	ImageAsset *current_image;
	VolumeAsset *current_volume;

	current_volume = volume_asset_panel->ReturnAssetPointer(ReferenceSelectPanel->GetSelection());

	bool parameter_map[5]; // needed for euler search init
	for (int i = 0; i < 5; i++) {parameter_map[i] = true;}

	float wanted_out_of_plane_angular_step = OutofPlaneStepNumericCtrl->ReturnValue();
	float wanted_in_plane_angular_step = InPlaneStepNumericCtrl->ReturnValue();

	RunProfile active_refinement_run_profile = run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()];

	int number_of_processes = active_refinement_run_profile.ReturnTotalJobs() - 1;

	// how many jobs are there going to be..

	// get first image to make decisions about how many jobs.. .we assume this is representative.


	current_image = image_asset_panel->ReturnAssetPointer(active_group.members[0]);
	current_image_euler_search = new EulerSearch;
	current_image_euler_search->InitGrid("C1", wanted_out_of_plane_angular_step, 0.0, 0.0, 360.0, wanted_in_plane_angular_step, 0.0, current_image->pixel_size / resolution_limit, parameter_map, 1);

	if (current_image_euler_search->test_mirror == true) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
	{
		current_image_euler_search->theta_max = 180.0f;
	}

	current_image_euler_search->CalculateGridSearchPositions(false);


	if (active_group.number_of_members >= 5 || current_image_euler_search->number_of_search_positions < number_of_processes * 20) number_of_jobs_per_image_in_gui = number_of_processes;
	else
	if (current_image_euler_search->number_of_search_positions > number_of_processes * 250) number_of_jobs_per_image_in_gui = number_of_processes * 10;
	else number_of_jobs_per_image_in_gui = number_of_processes * 5;

	int number_of_jobs = number_of_jobs_per_image_in_gui * active_group.number_of_members;

	delete current_image_euler_search;

// Some settings for testing
	float defocus_search_range = 1200.0f;
	float defocus_step = 200.0f;

	// number of rotations

	for (float current_psi = 0.0f; current_psi <= 360.0f; current_psi += wanted_in_plane_angular_step)
	{
		number_of_rotations++;
	}

	my_job_package.Reset(active_refinement_run_profile, "match_template", number_of_jobs);

	expected_number_of_results = 0;
	number_of_received_results = 0;

	// loop over all images..

	OneSecondProgressDialog *my_progress_dialog = new OneSecondProgressDialog ("Preparing Job", "Preparing Job...", active_group.number_of_members, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE| wxPD_APP_MODAL);

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
		current_image_euler_search->InitGrid("C1", wanted_out_of_plane_angular_step, 0.0, 0.0, 360.0, wanted_in_plane_angular_step, 0.0, current_image->pixel_size / resolution_limit, parameter_map, 1);

		if (current_image_euler_search->test_mirror == true) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
		{
			current_image_euler_search->theta_max = 180.0f;
		}

		current_image_euler_search->CalculateGridSearchPositions(false);

		number_of_defocus_positions = 2 * myround(float(defocus_search_range)/float(defocus_step)) + 1;

		wxPrintf("There are %i search positions\nThere are %i jobs per image\n", current_image_euler_search->number_of_search_positions, number_of_jobs_per_image_in_gui);
		wxPrintf("Calculating %i correlation maps\n", current_image_euler_search->number_of_search_positions * number_of_rotations * number_of_defocus_positions);
		// how many orientations will each process do for this image..
		expected_number_of_results += current_image_euler_search->number_of_search_positions * number_of_rotations * number_of_defocus_positions;
		orientations_per_process = float(current_image_euler_search->number_of_search_positions - number_of_jobs_per_image_in_gui) / float(number_of_jobs_per_image_in_gui);

		current_orientation_counter = 0;

		for (job_counter = 0; job_counter < number_of_jobs_per_image_in_gui; job_counter++)
		{
			wxString 	input_search_images = current_image->filename.GetFullPath();
			wxString 	input_reconstruction = current_volume->filename.GetFullPath();
			float		pixel_size = current_image->pixel_size;

			double voltage_kV;
			double spherical_aberration_mm;
			double amplitude_contrast;
			double defocus1;
			double defocus2;
			double defocus_angle;
			double phase_shift;

			main_frame->current_project.database.GetCTFParameters(current_image->ctf_estimation_id,voltage_kV,spherical_aberration_mm,amplitude_contrast,defocus1,defocus2,defocus_angle,phase_shift);

			float low_resolution_limit = 300.0f;
			float high_resolution_limit = resolution_limit;
			float angular_step = wanted_out_of_plane_angular_step;
			int best_parameters_to_keep = 1;
//			float defocus_search_range = 0.0f;
//			float defocus_step = 0.0f;
			float padding = 1;
			bool ctf_refinement = false;
			float mask_radius_search = EstimatedParticleSizeTextCtrl->ReturnValue();
			wxString mip_output_file = "/dev/null";
			wxString best_psi_output_file = "/dev/null";
			wxString best_theta_output_file = "/dev/null";
			wxString best_phi_output_file = "/dev/null";
			wxString best_defocus_output_file = "/dev/null";
			wxString scaled_mip_output_file ="/dev/null";
			wxString correlation_variance_output_file = "/dev/null";
			wxString my_symmetry = "C1";
			float in_plane_angular_step = wanted_in_plane_angular_step;
			wxString output_histogram_file = "/dev/null";

			if (current_orientation_counter >= current_image_euler_search->number_of_search_positions) current_orientation_counter = current_image_euler_search->number_of_search_positions - 1;
			int first_search_position = myroundint(current_orientation_counter);
			current_orientation_counter += orientations_per_process;
			if (current_orientation_counter >= current_image_euler_search->number_of_search_positions || job_counter == number_of_jobs_per_image_in_gui - 1) current_orientation_counter = current_image_euler_search->number_of_search_positions - 1;
			int last_search_position = myroundint(current_orientation_counter);
			current_orientation_counter++;

			wxString directory_for_results = main_frame->ReturnScratchDirectory();


			//wxPrintf("%i = %i - %i\n", job_counter, first_search_position, last_search_position);


			my_job_package.AddJob("ttffffffffffifffbffttttttttftiiiitt",	input_search_images.ToUTF8().data(),
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
																	angular_step,
																	best_parameters_to_keep,
																	defocus_search_range,
																	defocus_step,
																	padding,
																	ctf_refinement,
																	mask_radius_search,
																	phase_shift,
																	mip_output_file.ToUTF8().data(),
																	best_psi_output_file.ToUTF8().data(),
																	best_theta_output_file.ToUTF8().data(),
																	best_phi_output_file.ToUTF8().data(),
																	best_defocus_output_file.ToUTF8().data(),
																	scaled_mip_output_file.ToUTF8().data(),
																	correlation_variance_output_file.ToUTF8().data(),
																	my_symmetry.ToUTF8().data(),
																	in_plane_angular_step,
																	output_histogram_file.ToUTF8().data(),
																	first_search_position,
																	last_search_position,
																	image_number_for_gui,
																	number_of_jobs_per_image_in_gui,
																	correlation_variance_output_file.ToUTF8().data(),
																	directory_for_results.ToUTF8().data());
		}

		delete current_image_euler_search;
		my_progress_dialog->Update(image_counter + 1);

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

	if (buffered_results != NULL)
	{
		delete [] buffered_results;
		buffered_results = NULL;
	}

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


void MatchTemplatePanel::OnJobSocketEvent(wxSocketEvent& event)
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

			if (temp_result.result_size > 0)
			{
				ProcessResult(&temp_result);
			}
		}
		else
		if (memcmp(socket_input_buffer, socket_job_result_queue, SOCKET_CODE_SIZE) == 0) // identification
		{
			ArrayofJobResults temp_queue;
			ReceiveResultQueueFromSocket(sock, temp_queue);

			for (int counter = 0; counter < temp_queue.GetCount(); counter++)
			{
				ProcessResult(&temp_queue.Item(counter));
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
		{
			// As soon as it sends us the message that all jobs are finished, the controller should also
			// send timing info - we need to remember this
			long timing_from_controller;
			ReadFromSocket(sock, &timing_from_controller, sizeof(long));
			MyDebugAssertTrue(main_frame->current_project.total_cpu_hours + timing_from_controller / 3600000.0 >= main_frame->current_project.total_cpu_hours,"Oops. Double overflow when summing hours spent on project.");
			main_frame->current_project.total_cpu_hours += timing_from_controller / 3600000.0;
			MyDebugAssertTrue(main_frame->current_project.total_cpu_hours >= 0.0,"Negative total_cpu_hour");
			main_frame->current_project.total_jobs_run += my_job_tracker.total_number_of_jobs;

			// Update project statistics in the database
			main_frame->current_project.WriteProjectStatisticsToDatabase();

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
		main_frame->job_controller.KillJobIfSocketExists(sock);
		break;
	}
	default: ;
	}

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

		TimeRemaining time_remaining;

		if (seconds_remaining > 3600) time_remaining.hours = seconds_remaining / 3600;
		else time_remaining.hours = 0;

		if (seconds_remaining > 60) time_remaining.minutes = (seconds_remaining / 60) - (time_remaining.hours * 60);
		else time_remaining.minutes = 0;

		time_remaining.seconds = seconds_remaining - ((time_remaining.hours * 60 + time_remaining.minutes) * 60);
		TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_remaining.hours, time_remaining.minutes, time_remaining.seconds));
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

		CTFResultsPanel->Draw(my_job_package.jobs[result_to_process->job_number].arguments[3].ReturnStringArgument(), my_job_package.jobs[result_to_process->job_number].arguments[16].ReturnBoolArgument(), result_to_process->result_data[0], result_to_process->result_data[1], result_to_process->result_data[2], result_to_process->result_data[3], result_to_process->result_data[4], result_to_process->result_data[5], result_to_process->result_data[6], image_filename);
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
	//WriteResultToDataBase();

	// let the FindParticles panel check whether any of the groups are now ready to be picked
	//extern MyFindParticlesPanel *findparticles_panel;
	//findparticles_panel->CheckWhetherGroupsCanBePicked();

	if (buffered_results != NULL)
	{
		delete [] buffered_results;
		buffered_results = NULL;
	}

	// Kill the job (in case it isn't already dead)
	//main_frame->job_controller.KillJob(my_job_id);

	//WriteInfoText("All Jobs have finished.");
	//ProgressBar->SetValue(100);
	//TimeRemainingText->SetLabel("Time Remaining : All Done!");
	//CancelAlignmentButton->Show(false);
	//FinishButton->Show(true);
	//ProgressPanel->Layout();
}


void MatchTemplatePanel::WriteResultToDataBase()
{
/*
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
	int counter_aliasing_within_fit_range;


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
																				ProcessJobRes		  "ADDITIONAL_PHASE_SHIFT",
																						  "SCORE",
																						  "DETECTED_RING_RESOLUTION",
																						  "DETECTED_ALIAS_RESOLUTION",
																						  "OUTPUT_DIAGNOSTIC_FILE",
																						  "NUMBER_OF_FRAMES_AVERAGED",
																						  "LARGE_ASTIGMATISM_EXPECTED");



	wxDateTime now = wxDateTime::Now();
	counter_aliasing_within_fit_range = 0;
	for (counter = 0; counter < my_job_tracker.total_number_of_jobs; counter++)
	{
		image_asset_id = image_asset_panel->ReturnAssetPointer(active_group.members[counter])->asset_id;

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

		if (buffered_results[counter].result_data[6] > MaxResNumericCtrl->ReturnValue()) counter_aliasing_within_fit_range ++;

	}

	main_frame->current_project.database.EndBatchInsert();

	if (counter_aliasing_within_fit_range > 0)
	{
		WriteInfoText(wxString::Format("For %i of %i micrographs, CTF aliasing was detected within the fit range. Aliasing may affect the detected fit resolution and/or the quality of the defocus estimates. To reduce aliasing, use a larger box size (current box size: %i)\n", counter_aliasing_within_fit_range, my_job_tracker.total_number_of_jobs,BoxSizeSpinCtrl->GetValue()));
	}

	// we need to update the image assets with the correct CTF estimation number..

	ctf_estimation_id = starting_ctf_estimation_id + 1;
	main_frame->current_project.database.BeginImageAssetInsert();

	for (counter = 0; counter < my_job_tracker.total_number_of_jobs; counter++)
	{
		current_asset = active_group.members[counter];

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
															   image_asset_panel->ReturnAssetPointer(current_asset)->spherical_aberration,
															   image_asset_panel->ReturnAssetPointer(current_asset)->protein_is_white);


		image_asset_panel->ReturnAssetPointer(current_asset)->ctf_estimation_id = ctf_estimation_id;

		ctf_estimation_id++;
		my_progress_dialog->Update(my_job_tracker.total_number_of_jobs + counter + 1);


	}

	main_frame->current_project.database.EndImageAssetInsert();

	// Global Commit
	main_frame->current_project.database.Commit();


	my_progress_dialog->Destroy();
	ctf_results_panel->is_dirty = true;

	*/

}


void MatchTemplatePanel::UpdateProgressBar()
{
	TimeRemaining time_left = my_job_tracker.ReturnRemainingTime();
	ProgressBar->SetValue(my_job_tracker.ReturnPercentCompleted());

	TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_left.hours, time_left.minutes, time_left.seconds));
}

