#ifndef __DISPLAYFRAME_H__
#define __DISPLAYFRAME_H__

// Forward declare DisplayPanel
class DisplayPanel;

class DisplayFrame : public DisplayFrameParent {
  public:
    // Constructor/destructor
    DisplayFrame(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style);
    ~DisplayFrame( );

    //Additional functions
    void DisableAllToolbarButtons( );
    void EnableAllToolbarButtons( );
    void OnServerOpenFile(wxCommandEvent& event);

    // GUI event functions
    void OnCharHook(wxKeyEvent& event);
    void OnUpdateUI(wxUpdateUIEvent& event);

    // File menu
    void OnFileOpenClick(wxCommandEvent& event);
	void OnSaveDisplayedImagesClick(wxCommandEvent& event);
    void OnCloseTabClick(wxCommandEvent& event);
    void OnExitClick(wxCommandEvent& event);

    // Label menu
    void OnLocationNumberClick(wxCommandEvent& event);

    // Select menu
    void OnImageSelectionModeClick(wxCommandEvent& event);
    void OnCoordsSelectionModeClick(wxCommandEvent& event);
    void OnOpenTxtClick(wxCommandEvent& event);
    void OnSaveTxtClick(wxCommandEvent& event);
    void OnSaveTxtAsClick(wxCommandEvent& event);
    void OnInvertSelectionClick(wxCommandEvent& event);
    void OnClearSelectionClick(wxCommandEvent& event);

    // Options Menu
    void OnSize3(wxCommandEvent& event);
    void OnSize5(wxCommandEvent& event);
    void OnSize7(wxCommandEvent& event);
    void OnSize10(wxCommandEvent& event);
    void OnSingleImageModeClick(wxCommandEvent& event);
    void OnShowSelectionDistancesClick(wxCommandEvent& event);
    void OnShowResolution(wxCommandEvent& event);

    // Help menu
    void OnDocumentationClick(wxCommandEvent& event);

  private:
    bool     is_fullscreen;
    wxString remember_path;
    bool     LoadCoords(wxString current_line, long& x, long& y, long& image_number);
    bool     LoadImageSelections(wxString current_line);
    void     ClearTextFileFromPanel( );
};

#endif