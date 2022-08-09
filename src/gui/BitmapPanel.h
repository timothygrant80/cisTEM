#ifndef __BITMAP_PANEL_H__
#define __BITMAP_PANEL_H__

#include <wx/panel.h>

class BitmapPanel : public wxPanel {
  public:
    Image PanelImage;
    //wxBitmap PanelBitmap; // buffer for the panel size
    wxString panel_text;
    wxString title_text;
    bool     use_auto_contrast;

    BitmapPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
    ~BitmapPanel( );

    void OnPaint(wxPaintEvent& evt);
    void OnEraseBackground(wxEraseEvent& event);
    //	void SetupPanelBitmap();
    void Clear( );

    bool  should_show;
    float font_size_multiplier;
};

#endif
