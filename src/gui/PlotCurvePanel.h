#ifndef __PLOTCURVE_PANEL_H__
#define __PLOTCURVE_PANEL_H__

#include <vector>
#include "../gui/mathplot.h"
#include <wx/panel.h>

class
PlotCurvePanel : public wxPanel
{

public:

	PlotCurvePanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
	~PlotCurvePanel();


	void Initialise(wxString wanted_x_axis_text, wxString wanted_y_axis_text, bool show_legend = false, bool show_coordinates = false, int wanted_top_margin = 20, int wanted_bottom_margin = 50, int wanted_left_margin = 60, int wanted_right_margin = 20, bool wanted_draw_x_axis_ticks = true, bool wanted_draw_y_axis_ticks = true, bool wanted_centre_x_axis = false);
	void Clear(bool update_display = true);
	void AddCurve(Curve &curve_to_add, wxColour wanted_plot_colour, wxString wanted_name = "", int line_width = 2, wxPenStyle wanted_pen_style = wxPENSTYLE_SOLID );
	void Draw(float wanted_x_min, float wanted_x_max, float wanted_y_min, float wanted_y_max);
	void Draw();
	void SaveScreenshot(const wxString &filename, int type=wxBITMAP_TYPE_BMP);

	void SetXAxisLabel(wxString wanted_label);
	void SetYAxisLabel(wxString wanted_label);

	void SetXAxisMinStep(double wanted_min_step);

private:

	void SetupBaseLayers(wxString wanted_x_axis_text, wxString wanted_y_axis_text);
	void OnEraseBackground(wxEraseEvent& event);

	wxBoxSizer* GraphSizer;

	ArrayofCurves curves_to_plot;

	wxString stored_x_axis_text;
	wxString stored_y_axis_text;
	double min_x_step_size;

	mpTopInfoLegend *legend;
	bool legend_is_visible;
	bool should_draw_x_axis_ticks;
	bool should_draw_y_axis_ticks;
	bool should_draw_coords_box;
	bool centre_x_axis;

	mpWindow        * current_plot_window;
	mpTitle         * title;
	mpScaleX 		* current_xaxis;
	mpScaleY 		* current_yaxis;
	mpInfoCoords 	* info_coords;


};


#endif
