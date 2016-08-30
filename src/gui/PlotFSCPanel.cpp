//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

PlotFSCPanel::PlotFSCPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
: wxPanel(parent, id, pos, size, style, name)
{

	GraphSizer = new wxBoxSizer( wxVERTICAL );
	SetSizer( GraphSizer );
	Layout();
	GraphSizer->Fit( this );

	// Create a mpFXYVector layer for the plot

	FSC_vector_layer = new mpFXYVector(("X-Shift"));
		//average_shift_vector_layer = new mpFXYVector((""));

	FSC_vector_layer->ShowName(false);

	wxPen vectorpen(*wxBLUE, 2, wxSOLID);

	//current_x_shift_vector_layer->SetData(current_accumulated_dose_data, current_movement_data);
	FSC_vector_layer->SetContinuity(true);
	FSC_vector_layer->SetPen(vectorpen);
	FSC_vector_layer->SetDrawOutsideMargins(false);

	wxFont graphFont(11, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);

	current_plot_window = new mpWindow( this, -1, wxPoint(0,0), wxSize(100, 100), wxSUNKEN_BORDER );

	current_xaxis = new mpScaleX(wxT("Spatial Frequency  (1 / Å)"), mpALIGN_BOTTOM, true, mpX_NORMAL);
	current_yaxis = new mpScaleY(wxT("FSC"), mpALIGN_LEFT, true);

	title = new mpTitle("");

    current_xaxis->SetFont(graphFont);
    current_yaxis->SetFont(graphFont);
    current_xaxis->SetDrawOutsideMargins(false);
    current_yaxis->SetDrawOutsideMargins(false);

    current_plot_window->SetMargins(20, 20, 60, 60);

    current_plot_window->AddLayer(current_xaxis);
    current_plot_window->AddLayer(current_yaxis);
	current_plot_window->AddLayer(FSC_vector_layer);
	current_plot_window->AddLayer(title);

	GraphSizer->Add(current_plot_window, 1, wxEXPAND );
    current_plot_window->EnableDoubleBuffer(true);
    current_plot_window->EnableMousePanZoom(false);
  //  current_plot_window->Fit();

	current_plot_window->SetLayerVisible(0, false);
	current_plot_window->SetLayerVisible(1, false);
	current_plot_window->SetLayerVisible(2, false);
	current_plot_window->SetLayerVisible(3, false);


}

PlotFSCPanel::~PlotFSCPanel()
{
	delete current_plot_window;
	//delete legend;
	//delete title;
	//delete current_x_shift_vector_layer;
	//delete current_y_shift_vector_layer;
	//delete current_xaxis;
	//delete current_yaxis;
}

void PlotFSCPanel::Clear()
{
	current_plot_window->Freeze();

	FSC_vector_layer->Clear();

	current_spatial_frequency_data.clear();
	current_FSC_data.clear();


	current_plot_window->SetLayerVisible(0, false);
	current_plot_window->SetLayerVisible(1, false);
	current_plot_window->SetLayerVisible(2, false);
	current_plot_window->SetLayerVisible(3, false);
	current_plot_window->Thaw();

}

void PlotFSCPanel::Draw(float nyquist)
{
	current_plot_window->Freeze();
	FSC_vector_layer->SetData(current_spatial_frequency_data, current_FSC_data);
	current_plot_window->Fit(0, 1./ nyquist, -0.05, 1.05);


	current_plot_window->UpdateAll();

	current_plot_window->SetLayerVisible(0, true);
	current_plot_window->SetLayerVisible(1, true);
	current_plot_window->SetLayerVisible(2, true);
	current_plot_window->SetLayerVisible(3, true);

	current_plot_window->Thaw();
}

void PlotFSCPanel::AddPoint(double spatial_frequency, double FSC)
{
	current_spatial_frequency_data.push_back(spatial_frequency);
	current_FSC_data.push_back(FSC);
}
