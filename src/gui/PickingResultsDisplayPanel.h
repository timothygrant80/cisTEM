#ifndef __SHOWPICKING_RESULTS_PANEL_H__
#define __SHOWPICKING_RESULTS_PANEL_H__

#include <vector>
#include <wx/panel.h>
#include "../gui/job_panel.h"
#include "../gui/ProjectX_gui.h"
class
PickingResultsDisplayPanel : public PickingResultsDisplayParentPanel
{
public :

	PickingResultsDisplayPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
	~PickingResultsDisplayPanel();

	//void OnFitTypeRadioButton(wxCommandEvent& event);
	void Clear();
	void Draw();

};


#endif
