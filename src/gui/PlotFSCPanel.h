#ifndef __PLOTFSC_PANEL_H__
#define __PLOTFSC_PANEL_H__

#include <vector>
#include "../gui/mathplot.h"
#include <wx/panel.h>

class
PlotFSCPanel : public wxPanel
{

	wxBoxSizer* GraphSizer;

	public:

	PlotFSCPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
	~PlotFSCPanel();

	void Clear();
	void AddPoint(double spatial_frequency, double FSC);
	void Draw(float nyquist);


	std::vector<double> current_spatial_frequency_data;
	std::vector<double> current_FSC_data;


	mpWindow        *current_plot_window;
	mpTitle         *title;
	mpScaleX * current_xaxis;
	mpScaleY * current_yaxis;

	mpFXYVector* FSC_vector_layer;


};


#endif
