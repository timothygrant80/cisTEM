#ifndef __UNBLUR_RESULTS_PANEL_H__
#define __UNBLUR_RESULTS_PANEL_H__

#include <vector>
#include "../gui/mathplot.h"
#include <wx/panel.h>

class
        UnblurResultsPanel : public UnblurResultsPanelParent {
  public:
    UnblurResultsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
    ~UnblurResultsPanel( );

    void Clear( );
    void ClearGraph( );
    void AddPoint(double dose, double x_movement, double y_movement);
    void Draw( );

    std::vector<double> current_accumulated_dose_data;
    std::vector<double> current_x_movement_data;
    std::vector<double> current_y_movement_data;

    mpWindow* current_plot_window;
    //mpBottomInfoLegend    *legend;
    mpTopInfoLegend* legend;
    mpTitle*         title;
    mpScaleX*        current_xaxis;
    mpScaleY*        current_yaxis;

    mpFXYVector* current_x_shift_vector_layer;
    mpFXYVector* current_y_shift_vector_layer;
};

#endif
