#ifndef __SHOWCTF_RESULTS_PANEL_H__
#define __SHOWCTF_RESULTS_PANEL_H__

#include <vector>
#include "../gui/mathplot.h"
#include <wx/panel.h>
#include "../gui/job_panel.h"
#include "../gui/ProjectX_gui.h"
class
ShowCTFResultsPanel : public ShowCTFResultsPanelParent
{
public :

	ShowCTFResultsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
	~ShowCTFResultsPanel();

	void OnFitTypeRadioButton(wxCommandEvent& event);
	void Clear();
	void Draw(wxString diagnostic_filename, bool find_additional_phase_shift, float defocus1, float defocus2, float defocus_angle, float phase_shift, float score, float fit_res, float alias_res, float iciness, float tilt_angle, float tilt_axis, wxString ImageFile);

};


#endif
