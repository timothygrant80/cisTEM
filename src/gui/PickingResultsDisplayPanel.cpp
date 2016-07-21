//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

PickingResultsDisplayPanel::PickingResultsDisplayPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
: PickingResultsDisplayParentPanel(parent, id, pos, size, style)
{
	//Bind(wxEVT_COMBOBOX, &ShowPickingResultsPanel::OnFitTypeRadioButton, this);
	//Bind(wxEVT_COMBOBOX, &ShowPickingResultsPanel::OnFitTypeRadioButton, this);

	//CTF2DResultsPanel->font_size_multiplier = 1.5;



}

PickingResultsDisplayPanel::~PickingResultsDisplayPanel()
{
	//Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);
	//Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);

}

void PickingResultsDisplayPanel::Clear()
{
	PickingResultsImagePanel->should_show = false;
	Refresh();
}

void PickingResultsDisplayPanel::Draw(const wxString &image_filename, const int &number_of_particles, const double *particle_coordinates_x_in_angstroms, const double *particle_coordinates_y_in_angstroms, const float particle_radius_in_angstroms, const float pixel_size_in_angstroms)
{


	// Don't do this - it deallocates the images
	//PickingResultsImagePanel->Clear();

	PickingResultsImagePanel->SetImageFilename(image_filename,pixel_size_in_angstroms);

	PickingResultsImagePanel->UpdateScalingAndDimensions();

	PickingResultsImagePanel->UpdateImageInBitmap();

	PickingResultsImagePanel->SetParticleCoordinatesAndRadius(number_of_particles, particle_coordinates_x_in_angstroms, particle_coordinates_y_in_angstroms, particle_radius_in_angstroms);

	PickingResultsImagePanel->should_show = true;

	PickingResultsImagePanel->Draw();

	//PickingResultsImagePanel->Refresh();


}
