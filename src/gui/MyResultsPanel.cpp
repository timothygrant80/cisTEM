#include "../core/gui_core_headers.h"

MyResultsPanel::MyResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
:
ResultsPanel( parent, id, pos, size, style )
{
	// Bind OnListBookPageChanged from
	Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler( MyResultsPanel::OnResultsBookPageChanged ), this);
}

// TODO: destructor

void MyResultsPanel::OnResultsBookPageChanged(wxBookCtrlEvent& event )
{
	extern MyPickingResultsPanel *picking_results_panel;
	// We we were editing the particle picking results, and we move away from Results, we may need to do some database stuff
	if ( event.GetOldSelection() == 2) picking_results_panel->UpdateResultsFromBitmapPanel();
}
