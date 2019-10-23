#ifndef __SHOW_TEMPLATE_MATCH_RESULTS_PANEL_H__
#define __SHOW_TEMPLATE_MATCH_RESULTS_PANEL_H__

class
ShowTemplateMatchResultsPanel : public ShowTemplateMatchResultsParentPanel
{
public :

	TemplateMatchJobResults current_result;

	ShowTemplateMatchResultsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
	~ShowTemplateMatchResultsPanel();

	void Clear(bool show_peak_change_window = false);
	void DrawHistogram(wxString histogram_filename);
	void SetHistogramLabelText(wxString wanted_text) {SurvivalHistogramText->SetLabelText(wanted_text); }
	void SetPeakTableLabelText(wxString wanted_text) {PeakTableStaticText->SetLabelText(wanted_text);}
	void ClearPeakList();

	void OnPeakListSelectionChange(wxListEvent& event);
	void OnChangeListSelectionChange(wxListEvent& event);
	void OnImageLeftClick( wxMouseEvent& event );
	void OnSavePeaksClick( wxCommandEvent& event );
	void FillPeakInfoTable(float threshold_used);

	void SetActiveResult(TemplateMatchJobResults &result_to_show);

};


#endif
