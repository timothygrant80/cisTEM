#ifndef __PLOTCURVE_PANEL_H__
#define __PLOTCURVE_PANEL_H__

#include <vector>
#include "../gui/mathplot.h"
#include <wx/panel.h>

class
PlotCurvePanel : public wxPanel
{

	wxBoxSizer* GraphSizer;

	ArrayofCurves curves_to_plot;

	wxString stored_x_axis_text;
	wxString stored_y_axis_text;

	mpTopInfoLegend *legend;
	bool legend_is_visible;
	bool should_draw_axis_ticks;

	public:

	PlotCurvePanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
	~PlotCurvePanel();


	void Initialise(wxString wanted_x_axis_text, wxString wanted_y_axis_text, bool show_legend = false, int wanted_top_margin = 20, int wanted_bottom_margin = 50, int wanted_left_margin = 60, int wanted_right_margin = 20, bool wanted_draw_axis_ticks = true);
	void Clear(bool update_display = true);
	void AddCurve(Curve &curve_to_add, wxColour wanted_plot_colour, wxString wanted_name = "");
	void Draw(float wanted_x_min, float wanted_x_max, float wanted_y_min, float wanted_y_max);
	void Draw();
	void SetupBaseLayers(wxString wanted_x_axis_text, wxString wanted_y_axis_text);

	mpWindow        *current_plot_window;
	mpTitle         *title;
	mpScaleX * current_xaxis;
	mpScaleY * current_yaxis;


};


#endif
