#include "../core/gui_core_headers.h"

#include <wx/arrimpl.cpp>
//WX_DEFINE_OBJARRAY(ArrayOfRefinementResults);

DistributionPlotPanel::DistributionPlotPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
: wxPanel(parent, id, pos, size, style, name)
{

	plot_panel_sizer = new wxBoxSizer( wxVERTICAL );
	SetSizer(plot_panel_sizer);
	Layout();
	plot_panel_sizer->Fit(this);

	mp_window = NULL;
	mp_xaxis = NULL;
	mp_yaxis = NULL;
	plot_vector_layer = NULL;


	int client_x;
	int client_y;

	GetClientSize(&client_x, &client_y);

	//Bind(wxEVT_PAINT, &DistributionPlotPanel::OnPaint, this);
	//Bind(wxEVT_SIZE,  &DistributionPlotPanel::OnSize, this);

}

void DistributionPlotPanel::PlotUsingwxMathPlot(std::vector<double> &x_to_plot, std::vector<double> &y_to_plot)
{
	Freeze();
	//
	Layout();

	plot_vector_layer->Clear();
	plot_vector_layer->SetData(x_to_plot,y_to_plot);
	mp_window->Fit();
	mp_window->UpdateAll();
	//
	Thaw();
}

void DistributionPlotPanel::SetupwxMathPlot()
{
	mp_window = new mpWindow( this, -1, wxPoint(0,0), wxSize(100, 100), wxSUNKEN_BORDER );

	// Axes
	mp_xaxis = new mpScaleX(wxT("Value"), mpALIGN_BOTTOM, true, mpX_NORMAL);
	mp_yaxis = new mpScaleY(wxT("Count"), mpALIGN_LEFT, true);

	// Plot
	wxPen vectorpen(*wxBLUE, 2, wxSOLID);
	plot_vector_layer = new mpFXYVector(("Plot"));
	plot_vector_layer->ShowName(false);
	plot_vector_layer->SetContinuity(true);
	plot_vector_layer->SetPen(vectorpen);
	plot_vector_layer->SetDrawOutsideMargins(false);

	mp_window->AddLayer(mp_xaxis);
	mp_window->AddLayer(mp_yaxis);
	mp_window->AddLayer(plot_vector_layer);
	mp_window->EnableDoubleBuffer(true);

	plot_panel_sizer->Add(mp_window, 1, wxEXPAND);
}


DistributionPlotPanel::~DistributionPlotPanel()
{
	//Unbind(wxEVT_PAINT, &DistributionPlotPanel::OnPaint, this);
	//Unbind(wxEVT_SIZE,  &DistributionPlotPanel::OnSize, this);
}

void DistributionPlotPanel::Clear()
{
	Freeze();
    wxClientDC dc(this);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear();
    Thaw();

	int client_x;
	int client_y;

	GetClientSize(&client_x, &client_y);

	Refresh();

}

void DistributionPlotPanel::OnSize(wxSizeEvent & event)
{
	event.Skip();
}
