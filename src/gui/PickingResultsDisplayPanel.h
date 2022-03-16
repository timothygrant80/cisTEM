#ifndef __SHOWPICKING_RESULTS_PANEL_H__
#define __SHOWPICKING_RESULTS_PANEL_H__

#include <vector>
#include <wx/panel.h>
#include "../gui/job_panel.h"
#include "../gui/ProjectX_gui.h"

class
        PickingResultsDisplayPanel : public PickingResultsDisplayPanelParent {
  public:
    PickingResultsDisplayPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~PickingResultsDisplayPanel( );

    //void OnFitTypeRadioButton(wxCommandEvent& event);
    void Clear( );
    void Draw(const wxString& image_filename, ArrayOfParticlePositionAssets& array_of_assets, const float particle_radius_in_angstroms, const float pixel_size_in_angstroms, CTF micrograph_ctf, int image_asset_id, float iciness);

    void OnCirclesAroundParticlesCheckBox(wxCommandEvent& event);
    void OnHighPassFilterCheckBox(wxCommandEvent& event);
    void OnLowPassFilterCheckBox(wxCommandEvent& event);
    void OnWienerFilterCheckBox(wxCommandEvent& event);
    void OnScaleBarCheckBox(wxCommandEvent& event);
    void OnUndoButtonClick(wxCommandEvent& event);
    void OnRedoButtonClick(wxCommandEvent& event);

    void OnLowPassEnter(wxCommandEvent& event);
    void OnLowPassKillFocus(wxFocusEvent& event);

    void SetNumberOfPickedCoordinates(int number_of_coordinates);
    void SetIciness(float iciness);
    void SetImageAssetID(int image_asset_id);
    void SetDefocus(float defocus_in_angstroms);
};

#endif
