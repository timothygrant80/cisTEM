#include "../core/gui_core_headers.h"

DisplayRefinementResultsPanel::DisplayRefinementResultsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
    : DisplayRefinementResultsPanelParent(parent, id, pos, size, style) {
    ShowOrthDisplayPanel->EnableStartWithFourierScaling( );
    ShowOrthDisplayPanel->EnableDoNotShowStatusBar( );
    ShowOrthDisplayPanel->Initialise( );
}

void DisplayRefinementResultsPanel::Clear( ) {
    ShowOrthDisplayPanel->Clear( );
    AngularPlotPanel->Clear( );
    FSCResultsPanel->Clear( );
}

DisplayCTFRefinementResultsPanel::DisplayCTFRefinementResultsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
    : DisplayCTFRefinementResultsPanelParent(parent, id, pos, size, style) {

    ShowOrthDisplayPanel->EnableStartWithFourierScaling( );
    ShowOrthDisplayPanel->EnableDoNotShowStatusBar( );
    ShowOrthDisplayPanel->Initialise( );
}

void DisplayCTFRefinementResultsPanel::Clear( ) {
    ShowOrthDisplayPanel->Clear( );
    DefocusHistorgramPlotPanel->Clear( );
    FSCResultsPanel->Clear( );
}
