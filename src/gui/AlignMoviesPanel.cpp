#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMovieAssetPanel *movie_asset_panel;
extern MyMainFrame *main_frame;

MyAlignMoviesPanel::MyAlignMoviesPanel( wxWindow* parent )
:
AlignMoviesPanel( parent )
{

		// Create a mpFXYVector layer for the plot

		mpFXYVector* vectorLayer = new mpFXYVector((""));

		accumulated_dose_data.push_back(0);
		accumulated_dose_data.push_back(5);
		accumulated_dose_data.push_back(10);
		accumulated_dose_data.push_back(15);
		accumulated_dose_data.push_back(20);

		average_movement_data.push_back(10);
		average_movement_data.push_back(6);
		average_movement_data.push_back(4);
		average_movement_data.push_back(2);
		average_movement_data.push_back(1);

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

void MyAlignMoviesPanel::OnStartAlignmentButtonUpdateUI( wxUpdateUIEvent& event )
{
	// are there enough members in the selected group.

	if (movie_asset_panel->ReturnGroupSize(GroupComboBox->GetCurrentSelection()) > 0)
	{
		StartAlignmentButton->Enable(true);
	}
	else StartAlignmentButton->Enable(false);

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

	std::string current_filename;

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



	my_job_package.Reset("unblur", number_of_processes, number_of_jobs);

	for (counter = 0; counter < number_of_jobs; counter++)
	{
		// job is :-
		//
		// Filename (string)
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

		current_filename = movie_asset_panel->ReturnAssetLongFilename(movie_asset_panel->ReturnGroupMember(GroupComboBox->GetCurrentSelection(), counter)).ToStdString();
		my_job_package.AddJob("sffbbfifbii", current_filename.c_str(), float(minimum_shift), float(maximum_shift), should_dose_filter, should_restore_power, float(termination_threshold), max_iterations, float(bfactor), should_mask_central_cross, horizontal_mask, vertical_mask);
	}

	// launch a controller

	my_job_id = main_frame->job_controller.AddJob(this);

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

	    	  //MyDebugPrint("Sending Job Details...");
	    	  my_job_package.SendJobPackage(sock);

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
