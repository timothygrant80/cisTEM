#include "AlignMoviesPanel.h"

extern MyMovieAssetPanel *movie_asset_panel;

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
