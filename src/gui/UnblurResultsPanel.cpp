//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

UnblurResultsPanel::UnblurResultsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
: wxPanel(parent, id, pos, size, style, name)
{

	GraphSizer = new wxBoxSizer( wxVERTICAL );
	SetSizer( GraphSizer );
	Layout();
	GraphSizer->Fit( this );

	// Create a mpFXYVector layer for the plot

	current_x_shift_vector_layer = new mpFXYVector(("X-Shift"));
	current_y_shift_vector_layer = new mpFXYVector(("Y-Shift"));
	//average_shift_vector_layer = new mpFXYVector((""));

	current_x_shift_vector_layer->ShowName(false);
	current_y_shift_vector_layer->ShowName(false);

	wxPen vectorpen(*wxBLUE, 2, wxSOLID);
	wxPen redvectorpen(*wxRED, 2, wxSOLID);

	//current_x_shift_vector_layer->SetData(current_accumulated_dose_data, current_movement_data);
	current_x_shift_vector_layer->SetContinuity(true);
	current_x_shift_vector_layer->SetPen(vectorpen);
	current_x_shift_vector_layer->SetDrawOutsideMargins(false);

	current_y_shift_vector_layer->SetContinuity(true);
	current_y_shift_vector_layer->SetPen(redvectorpen);
	current_y_shift_vector_layer->SetDrawOutsideMargins(false);

	wxFont graphFont(11, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);

	current_plot_window = new mpWindow( this, -1, wxPoint(0,0), wxSize(100, 100), wxSUNKEN_BORDER );

	current_xaxis = new mpScaleX(wxT("Accumulated Exposure  (e¯/Å²)"), mpALIGN_BOTTOM, true, mpX_NORMAL);
	current_yaxis = new mpScaleY(wxT("Shifts (Å)"), mpALIGN_LEFT, true);

	legend = new mpBottomInfoLegend;
	title = new mpTitle("");

    current_xaxis->SetFont(graphFont);
    current_yaxis->SetFont(graphFont);
    current_xaxis->SetDrawOutsideMargins(false);
    current_yaxis->SetDrawOutsideMargins(false);

    current_plot_window->SetMargins(30, 30, 100, 80);

    current_plot_window->AddLayer(current_xaxis);
    current_plot_window->AddLayer(current_yaxis);
	current_plot_window->AddLayer(current_x_shift_vector_layer);
	current_plot_window->AddLayer(current_y_shift_vector_layer);
	current_plot_window->AddLayer(legend);
	current_plot_window->AddLayer(title);

	GraphSizer->Add(current_plot_window, 1, wxEXPAND );
    current_plot_window->EnableDoubleBuffer(true);
    current_plot_window->EnableMousePanZoom(false);
  //  current_plot_window->Fit();

	current_plot_window->SetLayerVisible(0, false);
	current_plot_window->SetLayerVisible(1, false);
	current_plot_window->SetLayerVisible(2, false);
	current_plot_window->SetLayerVisible(3, false);
	current_plot_window->SetLayerVisible(4, false);
	current_plot_window->SetLayerVisible(5, false);


}

UnblurResultsPanel::~UnblurResultsPanel()
{
	delete current_plot_window;
	//delete legend;
	//delete title;
	//delete current_x_shift_vector_layer;
	//delete current_y_shift_vector_layer;
	//delete current_xaxis;
	//delete current_yaxis;
}

void UnblurResultsPanel::Clear()
{
	current_plot_window->Freeze();

	current_x_shift_vector_layer->Clear();
	current_y_shift_vector_layer->Clear();

	current_accumulated_dose_data.clear();
	current_x_movement_data.clear();
	current_y_movement_data.clear();

	current_plot_window->SetLayerVisible(0, false);
	current_plot_window->SetLayerVisible(1, false);
	current_plot_window->SetLayerVisible(2, false);
	current_plot_window->SetLayerVisible(3, false);
	current_plot_window->SetLayerVisible(4, false);
	current_plot_window->SetLayerVisible(5, false);
	current_plot_window->Thaw();

}

void UnblurResultsPanel::Draw()
{
	current_plot_window->Freeze();
	current_x_shift_vector_layer->SetData(current_accumulated_dose_data, current_x_movement_data);
	current_y_shift_vector_layer->SetData(current_accumulated_dose_data, current_y_movement_data);
	current_plot_window->Fit();

	double y_axis_difference = fabs(current_plot_window->GetDesiredYmin() - current_plot_window->GetDesiredYmax());

	if (y_axis_difference < 6)
	{
		double half_extra = (6 - y_axis_difference) / 2;
		current_plot_window->Fit(current_plot_window->GetDesiredXmin(), current_plot_window->GetDesiredXmax(), current_plot_window->GetDesiredYmin() - half_extra,  current_plot_window->GetDesiredYmax() + half_extra);
	}
	current_plot_window->UpdateAll();

	current_plot_window->SetLayerVisible(0, true);
	current_plot_window->SetLayerVisible(1, true);
	current_plot_window->SetLayerVisible(2, true);
	current_plot_window->SetLayerVisible(3, true);
	current_plot_window->SetLayerVisible(4, true);
	current_plot_window->SetLayerVisible(5, true);

	current_plot_window->Thaw();
}

void UnblurResultsPanel::AddPoint(double dose, double x_movement, double y_movement)
{
	current_accumulated_dose_data.push_back(dose);
	current_x_movement_data.push_back(x_movement);
	current_y_movement_data.push_back(y_movement);
}
