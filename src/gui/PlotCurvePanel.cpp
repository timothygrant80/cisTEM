//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

PlotCurvePanel::PlotCurvePanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
: wxPanel(parent, id, pos, size, style, name)
{

	current_xaxis = NULL;
	current_yaxis = NULL;
	info_coords = NULL;
	title = NULL;
	legend = NULL;

	GraphSizer = new wxBoxSizer( wxVERTICAL );
	SetSizer( GraphSizer );
	Layout();
	GraphSizer->Fit( this );

	current_plot_window = new mpWindow( this, -1, wxPoint(0,0), wxSize(100, 100), wxSUNKEN_BORDER );


    GraphSizer->Add(current_plot_window, 1, wxEXPAND );
    current_plot_window->EnableDoubleBuffer(true);
    current_plot_window->EnableMousePanZoom(false);

    stored_x_axis_text = "";
    stored_y_axis_text = "";

    legend_is_visible = false;
    should_draw_x_axis_ticks = false;
    should_draw_y_axis_ticks = false;
    should_draw_coords_box = false;
}

void PlotCurvePanel::SetupBaseLayers(wxString wanted_x_axis_text, wxString wanted_y_axis_text)
{

	stored_x_axis_text = wanted_x_axis_text;
	stored_y_axis_text = wanted_y_axis_text;

	wxFont graphFont(10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);

	current_xaxis = new mpScaleX(wanted_x_axis_text, mpALIGN_BOTTOM, should_draw_x_axis_ticks, mpX_NORMAL);
	current_yaxis = new mpScaleY(wanted_y_axis_text, mpALIGN_LEFT, should_draw_y_axis_ticks);

	title = new mpTitle("");
	legend = new mpTopInfoLegend();
	info_coords = new mpInfoCoords();

	current_xaxis->SetFont(graphFont);
	current_yaxis->SetFont(graphFont);
	current_xaxis->SetDrawOutsideMargins(false);
	current_yaxis->SetDrawOutsideMargins(false);

    current_plot_window->AddLayer(current_xaxis);
    current_plot_window->AddLayer(current_yaxis);
	current_plot_window->AddLayer(title);
	current_plot_window->AddLayer(legend);
	current_plot_window->AddLayer(info_coords);

	current_plot_window->SetLayerVisible(0, false);
	current_plot_window->SetLayerVisible(1, false);
	current_plot_window->SetLayerVisible(2, false);
	current_plot_window->SetLayerVisible(3, legend_is_visible);
	current_plot_window->SetLayerVisible(4,	should_draw_coords_box);



}

PlotCurvePanel::~PlotCurvePanel()
{
	delete current_plot_window;
}

void PlotCurvePanel::Initialise(wxString wanted_x_axis_text, wxString wanted_y_axis_text, bool show_legend, bool show_coordinates, int wanted_top_margin, int wanted_bottom_margin, int wanted_left_margin, int wanted_right_margin, bool wanted_draw_x_axis_ticks, bool wanted_draw_y_axis_ticks)
{
	current_plot_window->SetMargins(20, 20, 50, 60);
	current_plot_window->SetMargins(wanted_top_margin, wanted_right_margin, wanted_bottom_margin, wanted_left_margin);
	legend_is_visible = show_legend;
	should_draw_x_axis_ticks = wanted_draw_x_axis_ticks;
	should_draw_y_axis_ticks = wanted_draw_y_axis_ticks;
	stored_x_axis_text = wanted_x_axis_text;
	stored_y_axis_text = wanted_y_axis_text;
	should_draw_coords_box = show_coordinates;

	Clear();

}

void PlotCurvePanel::Clear(bool update_display)
{
	current_plot_window->Freeze();
	current_plot_window->DelAllLayers( true, false );

	curves_to_plot.Clear();

	SetupBaseLayers(stored_x_axis_text, stored_y_axis_text);

	current_plot_window->UpdateAll();
	current_plot_window->Thaw();

}

void PlotCurvePanel::AddCurve(Curve &curve_to_add, wxColour wanted_plot_colour, wxString wanted_name)
{
	curves_to_plot.Add(curve_to_add);

	std::vector<double> current_x_data;
	std::vector<double> current_y_data;
	mpFXYVector* current_plot_vector_layer;

	// Create a mpFXYVector layer for the plot

	current_plot_vector_layer = new mpFXYVector(wanted_name);

	wxPen vectorpen(wanted_plot_colour, 2, wxSOLID);

	current_plot_vector_layer->SetContinuity(true);
	current_plot_vector_layer->SetPen(vectorpen);
	current_plot_vector_layer->SetDrawOutsideMargins(false);
	current_plot_vector_layer->ShowName(false);

	for (int point_counter = 0; point_counter < curve_to_add.number_of_points; point_counter++)
	{
		current_x_data.push_back(curve_to_add.data_x[point_counter]);
		current_y_data.push_back(curve_to_add.data_y[point_counter]);
	}

	current_plot_vector_layer->SetData(current_x_data, current_y_data);
	current_plot_window->AddLayer(current_plot_vector_layer);
}

void PlotCurvePanel::Draw(float wanted_x_min, float wanted_x_max, float wanted_y_min, float wanted_y_max)
{
	current_plot_window->Freeze();

	current_plot_window->SetLayerVisible(0, true);
	current_plot_window->SetLayerVisible(1, true);
	current_plot_window->SetLayerVisible(2, true);
	current_plot_window->SetLayerVisible(3, legend_is_visible);
	current_plot_window->SetLayerVisible(4, should_draw_coords_box);

	for (int layer_counter = 5; layer_counter <= 4 + curves_to_plot.GetCount(); layer_counter++)
	{
		current_plot_window->SetLayerVisible(layer_counter, true);
	}

	current_plot_window->Fit(wanted_x_min, wanted_x_max, wanted_y_min, wanted_y_max);

	current_plot_window->UpdateAll();

	current_plot_window->Thaw();

}

void PlotCurvePanel::Draw()
{
	float x_min, x_max;
	float y_min, y_max;

	float global_x_max = -FLT_MAX;
	float global_y_max = -FLT_MAX;
	float global_x_min = FLT_MAX;
	float global_y_min = FLT_MAX;

	for (int curve_counter = 0; curve_counter < curves_to_plot.GetCount(); curve_counter++)
	{
		curves_to_plot[curve_counter].GetXMinMax(x_min, x_max);
		curves_to_plot[curve_counter].GetYMinMax(y_min, y_max);

		global_x_min = std::min(global_x_min, x_min);
		global_y_min = std::min(global_y_min, y_min);
		global_x_max = std::max(global_x_max, x_max);
		global_y_max = std::max(global_y_max, y_max);
	}


	Draw(global_x_min, global_x_max, global_y_min, global_y_max);
}

void PlotCurvePanel::SetXAxisLabel(wxString wanted_label)
{
	current_xaxis->SetName(wanted_label);
}

void PlotCurvePanel::SetYAxisLabel(wxString wanted_label)
{
	current_yaxis->SetName(wanted_label);
}

void PlotCurvePanel::SaveScreenshot(const wxString & filename, int type)
{
	current_plot_window->SaveScreenshot(filename,type);
}



