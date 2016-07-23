#ifndef __SHOWPICKING_RESULTS_PANEL_H__
#define __SHOWPICKING_RESULTS_PANEL_H__

#include <vector>
#include <wx/panel.h>
#include "../gui/job_panel.h"
#include "../gui/ProjectX_gui.h"
class
PickingResultsDisplayPanel : public PickingResultsDisplayParentPanel
{
public :

	PickingResultsDisplayPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
	~PickingResultsDisplayPanel();

	//void OnFitTypeRadioButton(wxCommandEvent& event);
	void Clear();
	void Draw(const wxString &image_filename, const int &number_of_particles, const double *particle_coordinates_x_in_pixels, const double *particle_coordinates_y_pixels, const float particle_radius_in_pixels, const float pixel_size_in_angstroms);

	void OnCirclesAroundParticlesCheckBox(wxCommandEvent& event);
	void OnHighPassFilterCheckBox(wxCommandEvent& event);
	void OnScaleBarCheckBox(wxCommandEvent& event);




};


#endif
