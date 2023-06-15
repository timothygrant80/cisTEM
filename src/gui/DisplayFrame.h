#ifndef __DISPLAYFRAME_H__
#define __DISPLAYFRAME_H__

class DisplayFrame : public DisplayFrameParent {
  public:
    // Constructor/destructor
    DisplayFrame(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style);
    ~DisplayFrame( );

    //Additional functions
    void OpenFile(wxString wanted_filename, wxString wanted_tab_title, wxArrayLong* wanted_included_image_numbers = NULL, bool keep_scale_and_location_if_possible = false, bool force_local_survey = false);
    void DisableAllToolbarButtons( );
    void EnableAllToolbarButtons( );

    // GUI event functions
    void OnCharHook(wxKeyEvent& event);
    void OnUpdateUI(wxUpdateUIEvent& event);

    // File menu
    void OnFileOpenClick(wxCommandEvent& event);
    void OnCloseTabClick(wxCommandEvent& event);
    void OnExitClick(wxCommandEvent& event);

    // Label menu
    void OnLocationNumberClick(wxCommandEvent& event);

    // Select menu
    void OnImageSelectionModeClick(wxCommandEvent& event);
    void OnCoordsSelectionModeClick(wxCommandEvent& event);
    void OnOpenPLTClick(wxCommandEvent& event);
    void OnSavePLTClick(wxCommandEvent& event);
    void OnSavePLTAsClick(wxCommandEvent& event);
    void OnInvertSelectionClick(wxCommandEvent& event);
    void OnClearSelectionClick(wxCommandEvent& event);

    // Options Menu
    void OnSetPointSizeClick(wxCommandEvent& event);
    void OnShowCrossHairClick(wxCommandEvent& event);
    void OnSingleImageModeClick(wxCommandEvent& event);
    void On7BitGreyValuesClick(wxCommandEvent& event);
    void OnShowSelectionDistancesClick(wxCommandEvent& event);

    // Help menu
    void OnDocumentationClick(wxCommandEvent& event);

  private:
    bool is_fullscreen;
    bool image_is_open;
};

#endif