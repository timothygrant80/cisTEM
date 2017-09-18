//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
//WX_DEFINE_ARRAY(ArrayofmpFXYVectors);

PlotFSCPanel::PlotFSCPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
: wxPanel(parent, id, pos, size, style, name)
{

	GraphSizer = new wxBoxSizer( wxVERTICAL );
	SetSizer( GraphSizer );
	Layout();
	GraphSizer->Fit( this );

	current_plot_window = new mpWindow( this, -1, wxPoint(0,0), wxSize(100, 100), wxSUNKEN_BORDER );
    current_plot_window->SetMargins(20, 20, 50, 60);

    GraphSizer->Add(current_plot_window, 1, wxEXPAND );
    current_plot_window->EnableDoubleBuffer(true);
    current_plot_window->EnableMousePanZoom(false);

	number_of_added_fscs = 0;
	current_nyquist = 0;

}

void PlotFSCPanel::SetupBaseLayers()
{

	wxFont graphFont(10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);

	current_xaxis = new mpScaleX(wxT("Spatial Frequency  (1 / Å)"), mpALIGN_BOTTOM, true, mpX_NORMAL);
	current_yaxis = new mpScaleY(wxT("FSC"), mpALIGN_LEFT, true);

	refinement_limit = new RefinementLimit(-100);
	refinement_limit->SetDrawOutsideMargins(false);
	refinement_limit->SetPen(*wxGREY_PEN);

	title = new mpTitle("");

	current_xaxis->SetFont(graphFont);
	current_yaxis->SetFont(graphFont);
	current_xaxis->SetDrawOutsideMargins(false);
	current_yaxis->SetDrawOutsideMargins(false);

    current_plot_window->AddLayer(current_xaxis);
    current_plot_window->AddLayer(current_yaxis);
	current_plot_window->AddLayer(refinement_limit);
	current_plot_window->AddLayer(title);

	current_plot_window->SetLayerVisible(0, false);
	current_plot_window->SetLayerVisible(1, false);
	current_plot_window->SetLayerVisible(2, false);
	current_plot_window->SetLayerVisible(3, false);


}

PlotFSCPanel::~PlotFSCPanel()
{
	delete current_plot_window;
}

void PlotFSCPanel::Clear(bool update_display)
{
	current_plot_window->Freeze();
	current_plot_window->DelAllLayers( true, false );
	FSC_Layers.Clear();
	SetupBaseLayers();

	number_of_added_fscs = 0;

	current_plot_window->UpdateAll();
	current_plot_window->Thaw();

}

void PlotFSCPanel::AddPartFSC(ResolutionStatistics *statistics_to_add, float wanted_nyquist)
{
	if (number_of_added_fscs < default_colormap.GetCount())
	{
		std::vector<double> current_spatial_frequency_data;
		std::vector<double> current_FSC_data;
		mpFXYVector* current_FSC_vector_layer;

		// Create a mpFXYVector layer for the plot

		current_FSC_vector_layer = new mpFXYVector(("FSC"));
		current_FSC_vector_layer->ShowName(false);

		wxPen vectorpen(default_colormap[number_of_added_fscs], 2, wxSOLID);

		current_FSC_vector_layer->SetContinuity(true);
		current_FSC_vector_layer->SetPen(vectorpen);
		current_FSC_vector_layer->SetDrawOutsideMargins(false);

		for (int point_counter = 1; point_counter < statistics_to_add->part_FSC.number_of_points - 1; point_counter++)
		{
			if (statistics_to_add->part_FSC.data_x[point_counter] >= wanted_nyquist && statistics_to_add->part_FSC.data_x[point_counter] < 1000)
			{
				current_spatial_frequency_data.push_back(1.0 / statistics_to_add->part_FSC.data_x[point_counter]);
				current_FSC_data.push_back(statistics_to_add->part_FSC.data_y[point_counter]);
			}
		}

		current_FSC_vector_layer->SetData(current_spatial_frequency_data, current_FSC_data);
		current_plot_window->AddLayer(current_FSC_vector_layer);
		FSC_Layers.Add(current_FSC_vector_layer);
		number_of_added_fscs++;
	}
}

void PlotFSCPanel::HighlightClass(int wanted_class) // first class is 0!
{
	current_plot_window->DelAllLayers( false, false );
	SetupBaseLayers();
	wxColour colour_buffer;

	for (int class_counter = 0; class_counter < number_of_added_fscs; class_counter++)
	{
		if (class_counter != wanted_class)
		{
			//colour_buffer = default_colormap[class_counter];//.ChangeLightness(150);
			colour_buffer.Set(default_colormap[class_counter].Red(), default_colormap[class_counter].Green(), default_colormap[class_counter].Blue(), 128);
			wxPen vectorpen(colour_buffer, 1, wxSOLID);
			FSC_Layers[class_counter]->SetPen(vectorpen);
			current_plot_window->AddLayer(FSC_Layers[class_counter]);
		}
	}

	//colour_buffer = default_colormap[wanted_class];
	colour_buffer.Set(default_colormap[wanted_class].Red(), default_colormap[wanted_class].Green(), default_colormap[wanted_class].Blue());
	wxPen vectorpen(colour_buffer, 2, wxSOLID);
	FSC_Layers[wanted_class]->SetPen(vectorpen);
	current_plot_window->AddLayer(FSC_Layers[wanted_class]);
	Draw(current_nyquist);
}



void PlotFSCPanel::Draw(float nyquist)
{
	current_plot_window->Freeze();
	current_plot_window->Fit(0, 1./ nyquist, 0, 1.05);
	current_nyquist = nyquist;

	refinement_limit->SetSpatialFrequency(current_refinement_resolution_limit);

	current_plot_window->SetLayerVisible(0, true);
	current_plot_window->SetLayerVisible(1, true);
	current_plot_window->SetLayerVisible(2, current_refinement_resolution_limit > 0.0  && current_refinement_resolution_limit < 100.0);
	current_plot_window->SetLayerVisible(3, true);

	for (int layer_counter = 4; layer_counter < 4 + number_of_added_fscs; layer_counter++)
	{
		current_plot_window->SetLayerVisible(layer_counter, true);
	}
	current_plot_window->UpdateAll();

	current_plot_window->Thaw();
}

