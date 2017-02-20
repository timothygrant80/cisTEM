#ifndef __Refine2DResultsPanel__
#define __Refine2DResultsPanel__

class
Refine2DResultsPanel : public Refine2DResultsPanelParent
{

public:
	Refine2DResultsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
	void OnButtonClick(wxCommandEvent &event);

};



#endif
