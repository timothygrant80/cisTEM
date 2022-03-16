//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"
#include <algorithm>

CTF1DPanel::CTF1DPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
    : wxPanel(parent, id, pos, size, style, name) {

    GraphSizer = new wxBoxSizer(wxVERTICAL);
    SetSizer(GraphSizer);
    Layout( );
    GraphSizer->Fit(this);

    // Create a mpFXYVector layer for the plot

    current_ctf_fit_vector_layer        = new mpFXYVector(("CTF Fit"));
    current_quality_of_fit_vector_layer = new mpFXYVector(("Quality of Fit"));
    current_amplitude_vector_layer      = new mpFXYVector(("Amplitude Spectrum"));

    wxPen bluevectorpen(*wxBLUE, 1, wxSOLID);
    wxPen redvectorpen(wxColor(232, 162, 12), 1, wxSOLID);
    wxPen greenvectorpen(*wxGREEN, 1, wxSOLID);

    //current_x_shift_vector_layer->SetData(current_accumulated_dose_data, current_movement_data);
    current_ctf_fit_vector_layer->SetContinuity(true);
    current_ctf_fit_vector_layer->SetPen(redvectorpen);
    current_ctf_fit_vector_layer->SetDrawOutsideMargins(false);
    current_ctf_fit_vector_layer->ShowName(false);

    current_quality_of_fit_vector_layer->SetContinuity(true);
    current_quality_of_fit_vector_layer->SetPen(bluevectorpen);
    current_quality_of_fit_vector_layer->SetDrawOutsideMargins(false);
    current_quality_of_fit_vector_layer->ShowName(false);

    current_amplitude_vector_layer->SetContinuity(true);
    current_amplitude_vector_layer->SetPen(greenvectorpen);
    current_amplitude_vector_layer->SetDrawOutsideMargins(false);
    current_amplitude_vector_layer->ShowName(false);

    wxFont graphFont(11, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);

    current_plot_window = new mpWindow(this, -1, wxPoint(0, 0), wxSize(100, 100), wxSUNKEN_BORDER);

    mpScaleX* current_xaxis = new mpScaleX(wxT("Spatial Frequency  (1/Å)"), mpALIGN_BOTTOM, true, mpX_NORMAL);
    mpScaleY* current_yaxis = new mpScaleY(wxT("Amplitude (or Cross Correlation)"), mpALIGN_LEFT, true);

    legend = new mpTopInfoLegend;
    //legend->my_line_width = 20;

    //title = new mpTitle;

    current_xaxis->SetFont(graphFont);
    current_yaxis->SetFont(graphFont);
    current_xaxis->SetDrawOutsideMargins(false);
    current_yaxis->SetDrawOutsideMargins(false);

    current_plot_window->SetMargins(30, 30, 60, 80);

    current_plot_window->AddLayer(current_xaxis);
    current_plot_window->AddLayer(current_yaxis);
    current_plot_window->AddLayer(current_ctf_fit_vector_layer);
    current_plot_window->AddLayer(current_quality_of_fit_vector_layer);
    current_plot_window->AddLayer(current_amplitude_vector_layer);
    current_plot_window->AddLayer(legend);
    //current_plot_window->AddLayer(title);

    GraphSizer->Add(current_plot_window, 1, wxEXPAND);
    current_plot_window->EnableDoubleBuffer(true);
    current_plot_window->EnableMousePanZoom(false);
    //  current_plot_window->Fit();

    current_plot_window->SetLayerVisible(0, false);
    current_plot_window->SetLayerVisible(1, false);
    current_plot_window->SetLayerVisible(2, false);
    current_plot_window->SetLayerVisible(3, false);
    current_plot_window->SetLayerVisible(4, false);
    current_plot_window->SetLayerVisible(5, false);
    current_plot_window->SetLayerVisible(6, false);
}

void CTF1DPanel::Clear( ) {
    current_plot_window->Freeze( );

    current_ctf_fit_vector_layer->Clear( );
    current_quality_of_fit_vector_layer->Clear( );
    current_amplitude_vector_layer->Clear( ); //

    current_spatial_frequency.clear( );
    current_ctf_fit.clear( );
    current_quality_of_fit.clear( );
    current_amplitude_spectrum.clear( );
    ;

    current_plot_window->SetLayerVisible(0, false);
    current_plot_window->SetLayerVisible(1, false);
    current_plot_window->SetLayerVisible(2, false);
    current_plot_window->SetLayerVisible(3, false);
    current_plot_window->SetLayerVisible(4, false);
    current_plot_window->SetLayerVisible(5, false);
    current_plot_window->SetLayerVisible(6, false);
    current_plot_window->Thaw( );
}

void CTF1DPanel::Draw( ) {
    current_plot_window->Freeze( );
    current_ctf_fit_vector_layer->SetData(current_spatial_frequency, current_ctf_fit);
    current_quality_of_fit_vector_layer->SetData(current_spatial_frequency, current_quality_of_fit);
    current_amplitude_vector_layer->SetData(current_spatial_frequency, current_amplitude_spectrum);

    //current_plot_window->Fit();
    current_plot_window->Fit(0, *std::max_element(current_spatial_frequency.begin( ), current_spatial_frequency.end( )), -0.1, 1.1);
    /*
	double y_axis_difference = fabs(current_plot_window->GetDesiredYmin() - current_plot_window->GetDesiredYmax());

	if (y_axis_difference < 6)
	{
		double half_extra = (6 - y_axis_difference) / 2;
		current_plot_window->Fit(current_plot_window->GetDesiredXmin(), current_plot_window->GetDesiredXmax(), current_plot_window->GetDesiredYmin() - half_extra,  current_plot_window->GetDesiredYmax() + half_extra);
	}
	*/
    current_plot_window->UpdateAll( );

    current_plot_window->SetLayerVisible(0, true);
    current_plot_window->SetLayerVisible(1, true);
    current_plot_window->SetLayerVisible(2, true);
    current_plot_window->SetLayerVisible(3, true);
    current_plot_window->SetLayerVisible(4, true);
    current_plot_window->SetLayerVisible(5, true);
    current_plot_window->SetLayerVisible(6, true);

    current_plot_window->Thaw( );
}

void CTF1DPanel::AddPoint(double spatial_frequency, double ctf_fit, double quality_of_fit, double amplitude_spectrum) {
    current_spatial_frequency.push_back(spatial_frequency);
    current_ctf_fit.push_back(ctf_fit);
    current_quality_of_fit.push_back(quality_of_fit);
    current_amplitude_spectrum.push_back(amplitude_spectrum);
}
