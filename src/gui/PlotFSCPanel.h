#ifndef __PLOTFSC_PANEL_H__
#define __PLOTFSC_PANEL_H__

#include <vector>
#include "../gui/mathplot.h"
#include <wx/panel.h>

class RefinementLimit : public mpFY
{
	double spatial_frequency;
public:
	RefinementLimit(float wanted_spatial_frequency) : mpFY(wxT(" Refinement limit"),mpALIGN_TOP) { spatial_frequency = wanted_spatial_frequency; }
	void SetSpatialFrequency(float wanted_spatial_frequency) { spatial_frequency = wanted_spatial_frequency; }
	virtual double GetX( double y ) { return spatial_frequency; }
	virtual double GetMinX() { return -0.05; }
	virtual double GetMinY() { return 1.05; }
};

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
	float current_refinement_resolution_limit;


	mpWindow        *current_plot_window;
	mpTitle         *title;
	mpScaleX * current_xaxis;
	mpScaleY * current_yaxis;

	mpFXYVector* FSC_vector_layer;

	RefinementLimit * refinement_limit;


};


#endif
