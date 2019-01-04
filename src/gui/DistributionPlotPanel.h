#ifndef __DISTRIBUTION_PLOT_PANEL_H__
#define __DISTRIBUTION_PLOT_PANEL_H__

#include <wx/panel.h>

//WX_DECLARE_OBJARRAY(RefinementResult,ArrayOfRefinementResults);

class DistributionPlotPanel : public wxPanel
{
public :

	//bool draw_axis_overlay_instead_of_underlay;

	DistributionPlotPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
	~DistributionPlotPanel();

	wxBoxSizer * plot_panel_sizer;

	void OnSize( wxSizeEvent & event );
	void OnPaint(wxPaintEvent & evt);
	void OnEraseBackground(wxEraseEvent& event);
	virtual void Clear();

	/*
	 * wxMathPlot stuff
	 */
	void PlotUsingwxMathPlot(std::vector<double> &x_to_plot, std::vector<double> &y_to_plot);
	void SetupwxMathPlot();
	mpWindow * mp_window;
	mpScaleX * mp_xaxis;
	mpScaleY * mp_yaxis;
	mpFXYVector * plot_vector_layer;


};

#endif
