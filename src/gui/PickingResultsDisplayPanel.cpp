#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

PickingResultsDisplayPanel::PickingResultsDisplayPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
: PickingResultsDisplayParentPanel(parent, id, pos, size, style)
{
	//Bind(wxEVT_COMBOBOX, &ShowPickingResultsPanel::OnFitTypeRadioButton, this);
	//Bind(wxEVT_COMBOBOX, &ShowPickingResultsPanel::OnFitTypeRadioButton, this);

	//CTF2DResultsPanel->font_size_multiplier = 1.5;



}

PickingResultsDisplayPanel::~PickingResultsDisplayPanel()
{
	//Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);
	//Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);

}

void PickingResultsDisplayPanel::Clear()
{
	PickingResultsImagePanel->should_show = false;
	Refresh();
}

void PickingResultsDisplayPanel::Draw()
{


	PickingResultsImagePanel->Clear();

	PickingResultsImagePanel->panel_text = wxString::Format(wxT("This will display the picking results really nicely"));

	Freeze();

	// Read image from disk etc.

	Thaw();

}
