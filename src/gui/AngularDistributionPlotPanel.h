#ifndef __ANGULAR_DISTRIBUTION_PLOT_PANEL_H__
#define __ANGULAR_DISTRIBUTION_PLOT_PANEL_H__

#include <wx/panel.h>

WX_DECLARE_OBJARRAY(RefinementResult,ArrayOfRefinementResults);

class AngularDistributionPlotPanel : public wxPanel
{
public :



	AngularDistributionPlotPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
	~AngularDistributionPlotPanel();

	void OnSize( wxSizeEvent & event );
	void OnPaint(wxPaintEvent & evt);
	void OnEraseBackground(wxEraseEvent& event);
	void Clear();

	void UpdateScalingAndDimensions();
	void UpdateProjCircleRadius();

	void SetupBitmap();


	void AddRefinementResult(RefinementResult *refinement_result_to_add);

	float ReturnRadiusFromTheta(const float theta);
	//void XYFromPhiTheta(const float phi, const float theta, int &x, int &y);
	//void SetSymmetry(wxString wanted_symmetry_symbol);
	void SetSymmetryAndNumber(wxString wanted_symmetry_symbol, long wanted_number_of_final_results);

	ArrayOfRefinementResults refinement_results_to_plot;
	wxBitmap buffer_bitmap;

	SymmetryMatrix symmetry_matrices;
	AnglesAndShifts angles_and_shifts;

	bool 	should_show;
	float 	font_size_multiplier;
	long number_of_final_results;

	float		circle_center_x;
	float		circle_center_y;
	float		circle_radius;
	float		major_tick_length;
	float		minor_tick_length;
	float		margin_between_major_ticks_and_labels;
	float		margin_between_circles_and_theta_labels;

	float		proj_circle_radius;

	int colour_change_step;




};




#endif
