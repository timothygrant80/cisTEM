#include "../core/gui_core_headers.h"

MySettingsPanel::MySettingsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
:
SettingsPanel( parent, id, pos, size, style )
{
	// Bind OnListBookPageChanged from
	Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler( MySettingsPanel::OnSettingsBookPageChanged ), this);
}

// TODO: destructor

void MySettingsPanel::OnSettingsBookPageChanged(wxBookCtrlEvent& event )
{
	extern MyRunProfilesPanel *run_profiles_panel;

	// Necessary for MacOS to refresh the panels
	run_profiles_panel->Layout();
	run_profiles_panel->Refresh();
}
