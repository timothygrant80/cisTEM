#ifndef _SRC_GUI_RESULTS_PANEL_SPA_H_
#define _SRC_GUI_RESULTS_PANEL_SPA_H_

// Temporary forward declaration
class ResultsPanelParent;

class ResultsPanelSpa : public ResultsPanelParent {
  public:
    ResultsPanelSpa(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    void OnResultsBookPageChanged(wxBookCtrlEvent& event);
};

#endif