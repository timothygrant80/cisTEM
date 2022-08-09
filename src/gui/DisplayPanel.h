#ifndef __DisplayPanel__
#define __DisplayPanel__

#include <wx/popupwin.h>

#define CAN_CHANGE_FILE 1 // 2^0, bit 0
#define CAN_CLOSE_TABS 2 // 2^1, bit 1
#define CAN_FFT 4 // 2^2, bit 2
#define START_WITH_INVERTED_CONTRAST 8
#define START_WITH_AUTO_CONTRAST 16
#define NO_NOTEBOOK 32
#define FIRST_LOCATION_ONLY 64
#define START_WITH_FOURIER_SCALING 128
#define DO_NOT_SHOW_STATUS_BAR 256
#define CAN_SELECT_IMAGES 1024
#define NO_POPUP 2048
#define START_WITH_NO_LABEL 4096
#define SKIP_LEFTCLICK_TO_PARENT 8192
#define CAN_MOVE_TABS 16384
#define DRAW_IMAGE_SEPARATOR 32768
#define KEEP_TABS_LINKED_IF_POSSIBLE 65536

#define LOCAL_GREYS 0
#define GLOBAL_GREYS 1
#define MANUAL_GREYS 2
#define AUTO_GREYS 3

class DisplayPopup;
class DisplayNotebook;
class DisplayNotebookPanel;

class
        DisplayPanel : public DisplayPanelParent {
    friend class DisplayNotebook;
    friend class DisplayNotebookPanel;

  protected:
    int style_flags;

    wxTextCtrl*   toolbar_location_text;
    wxStaticText* toolbar_number_of_locations_text;
    wxComboBox*   toolbar_scale_combo;

    int panel_counter;

  public:
    DisplayNotebook*      my_notebook;
    wxStaticText*         StatusText;
    bool                  popup_exists;
    DisplayPopup*         popup;
    DisplayNotebookPanel* no_notebook_panel;

    DisplayPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    void Initialise(int style_flags = 0);

    void OpenFile(wxString wanted_filename, wxString wanted_tab_title, wxArrayLong* wanted_included_image_numbers = NULL, bool keep_scale_and_location_if_possible = false, bool force_local_survey = false);
    void ChangeFile(wxString wanted_filename, wxString wanted_tab_title, wxArrayLong* wanted_included_image_numbers = NULL);
    void ChangeFileForTabNumber(int wanted_tab_number, wxString wanted_filename, wxString wanted_tab_title, wxArrayLong* wanted_included_image_numbers = NULL);

    void OpenImage(Image* image_to_view, wxString wanted_tab_title, bool take_ownership = false, wxArrayLong* wanted_included_image_numbers = NULL);
    void ChangeImage(Image* image_to_View, wxString wanted_tab_title, bool take_ownership = false, wxArrayLong* wanted_included_image_numbers = NULL);

    void UpdateToolbar( );
    void ChangeFocusToPanel(void);
    void ReDrawCurrentPanel( );

    void SetSelectionSquareLocation(long wanted_location);

    void SetImageSelected(long wanted_image, bool refresh = true);
    void SetImageNotSelected(long wanted_image, bool refresh = true);
    void ToggleImageSelected(long wanted_image, bool refresh = true);
    void ClearSelection(bool refresh = true);
    bool IsImageSelected(long wanted_image);

    void SetActiveTemplateMatchMarkerPostion(float wanted_x_pos, float wanted_y_pos, float wanted_radius);
    void ClearActiveTemplateMatchMarker( );

    void RefreshCurrentPanel( );

    void OnAuto(wxCommandEvent& event);
    void OnLocal(wxCommandEvent& event);
    void OnManual(wxCommandEvent& event);
    void OnGlobal(wxCommandEvent& event);
    void OnNext(wxCommandEvent& event);
    void OnHistogram(wxCommandEvent& event);
    void OnFFT(wxCommandEvent& event);
    void OnInvert(wxCommandEvent& event);
    void OnPrevious(wxCommandEvent& event);
    void ChangeLocation(wxCommandEvent& event);
    void ChangeScaling(wxCommandEvent& event);
    void OnHighQuality(wxCommandEvent& event);

    DisplayNotebookPanel* ReturnCurrentPanel( );

    void Clear( );
    void CloseAllTabs( );
};

class
        DisplayNotebook : public wxAuiNotebook {

  public:
    DisplayNotebook(wxWindow* parent, wxWindowID id, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxAUI_NB_DEFAULT_STYLE);

    DisplayPanel* parent_display_panel;

    void OnSelectionChange(wxAuiNotebookEvent& event);
    void OnDragEnd(wxAuiNotebookEvent& event);
    void OnClosed(wxAuiNotebookEvent& event);
    void ChildGotFocus(wxChildFocusEvent& event);
    void GotFocus(wxFocusEvent& event);
    void OnEraseBackground(wxEraseEvent& event);
    void OnPaint(wxPaintEvent& evt);

    /*
	void OnRightClick(wxMouseEvent& event);
		void CloseTab( wxCommandEvent& WXUNUSED( event ));
		void OnSelectionChange(wxAuiNotebookEvent& event);
		void OnDragEnd(wxAuiNotebookEvent& event);
		void OnClosed(wxAuiNotebookEvent& event);
		void OnPrevious( wxCommandEvent& WXUNUSED(event) );
		void OnNext( wxCommandEvent& WXUNUSED(event) );
		void ChangeLocation(wxCommandEvent& WXUNUSED(event));
		void ChangeScaling(wxCommandEvent& WXUNUSED(event));
		void ChangeFilamentNumber(wxCommandEvent& WXUNUSED(event));
		void OnPreviousFilament( wxCommandEvent& WXUNUSED(event) );
		void OnNextFilament( wxCommandEvent& WXUNUSED(event) );
		void OnFilamentOnlyCheck( wxCommandEvent& WXUNUSED(event) );
		void OnLocal( wxCommandEvent& WXUNUSED(event) );
		void OnAuto( wxCommandEvent& WXUNUSED(event) );
		void OnGlobal( wxCommandEvent& WXUNUSED(event) );
		void OnManual( wxCommandEvent& WXUNUSED(event) );
		void OnHistogram( wxCommandEvent& WXUNUSED(event) );
		void OnRefresh( wxCommandEvent& WXUNUSED(event) );
		void OnKeyPress( wxKeyEvent& event);
		void OnKeyDown( wxKeyEvent& event);
		void OnKeyUp( wxKeyEvent& event);

		bool AcceptsFocus() const;
		bool AcceptsFocusFromKeyboard() const;

		void ChildGotFocus(wxChildFocusEvent& event);

		*/
};

class
        DisplayNotebookPanel : public wxPanel {
  public:
    DisplayNotebookPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~DisplayNotebookPanel( );

    DisplayPanel* parent_display_panel;

    ImageFile my_file;
    wxString  filename;
    bool      input_is_a_file;

    Image* image_to_display;
    bool   do_i_have_image_ownership;

    bool use_unscaled_image_for_popup;

    wxImage* panel_image;

    wxString tab_title;

    wxArrayLong included_image_numbers;

    void ReDrawPanel( );

    Image* image_memory_buffer;
    Image* scaled_image_memory_buffer;

    int number_allocated_for_buffer;

    void SetImageSelected(long wanted_image, bool refresh = true);
    void SetImageNotSelected(long wanted_image, bool refresh = true);
    void ToggleImageSelected(long wanted_image, bool refresh = true);
    void ClearSelection(bool refresh = true);

    void OnPaint(wxPaintEvent& evt);
    void OnEraseBackground(wxEraseEvent& event);
    void OnSize(wxSizeEvent& event);
    void OnLeftDown(wxMouseEvent& event);
    void OnRightDown(wxMouseEvent& event);
    void OnRightUp(wxMouseEvent& event);
    void OnMotion(wxMouseEvent& event);
    void OnLeaveWindow(wxMouseEvent& event);

    void OnKeyDown(wxKeyEvent& event);
    void OnKeyUp(wxKeyEvent& event);

    void UpdateImageStatusInfo(int x_pos, int y_pos);

    inline int ReturnNumberofImages( ) {
        if ( input_is_a_file == true )
            return my_file.ReturnNumberOfSlices( );
        else
            return image_to_display->logical_z_dimension;
    };

    inline int ReturnImageXSize( ) {
        if ( input_is_a_file == true )
            return my_file.ReturnXSize( );
        else
            return image_to_display->logical_x_dimension;
    };

    inline int ReturnImageYSize( ) {
        if ( input_is_a_file == true )
            return my_file.ReturnYSize( );
        else
            return image_to_display->logical_y_dimension;
    };

    inline void LoadIntoImage(Image* image_to_fill, long input_position) {
        if ( input_is_a_file == true )
            image_to_fill->ReadSlice(&my_file, included_image_numbers.Item(input_position));
        else {
            Image buffer_image;
            buffer_image.AllocateAsPointingToSliceIn3D(image_to_display, included_image_numbers.Item(input_position));
            image_to_fill->CopyFrom(&buffer_image);
        }
    };

    inline void SetImageInMemoryBuffer(long buffer_position, long input_position) {
        if ( input_is_a_file == true )
            image_memory_buffer[buffer_position].ReadSlice(&my_file, included_image_numbers.Item(input_position));
        else
            image_memory_buffer[buffer_position].AllocateAsPointingToSliceIn3D(image_to_display, included_image_numbers.Item(input_position));
    };

    bool CheckFileStillValid( );
    bool SetGlobalGreys( );

    wxString short_image_filename;
    wxString short_plt_filename;
    wxString short_waypoints_filename;

    //		char Filename[420];
    //		char PLTFilename[420];
    //		char WaypointsFilename[420];

    //	TigrisImage first_header;
    //	TigrisImage image_memory_buffer[5000];

    //	long image_format;

    long original_number_of_images;
    long original_x_size;
    long original_y_size;

    long current_location;
    long blue_selection_square_location;

    long location_on_last_draw;
    long images_in_x_on_last_draw;
    long images_in_y_on_last_draw;

    int template_matching_marker_x_pos;
    int template_matching_marker_y_pos;
    int template_matching_marker_radius;

    double pixel_size;

    double desired_scale_factor;
    double actual_scale_factor;

    double global_low_grey;
    double global_high_grey;

    double manual_low_grey;
    double manual_high_grey;

    double low_grey_value;
    double high_grey_value;

    long   grey_values_decided_by;
    double selected_distance;

    double integrate_box_x_pos;
    double integrate_box_y_pos;
    double integrated_value;
    long   integrate_box_size;
    long   integrate_image;

    bool panel_image_has_correct_greys;
    bool panel_image_has_correct_scale;
    bool use_7bit_greys;
    bool show_selection_distances;
    bool resolution_instead_of_radius;
    bool should_refresh;
    bool use_fft;
    bool invert_contrast;
    bool suspend_overlays;

    bool use_fourier_scaling;

    long images_in_current_view;

    int images_in_x;
    int images_in_y;
    int current_x_size;
    int current_y_size;

    int  single_image_x;
    int  single_image_y;
    bool show_label;
    bool show_crosshair;
    bool single_image;

    bool plt_is_saved;
    bool have_plt_filename;

    bool waypoints_is_saved;
    bool have_waypoints_filename;

    bool something_is_being_grabbed;
    long selected_filament_number;
    bool only_show_selected_filament;

    int label_mode;
    int picking_mode;

    wxBitmap panel_bitmap;

    bool* image_is_selected;
    int   number_of_selections;

    long selected_point_size;

    //CoordTracker coord_tracker;

    /*

		void OnLeftDown(wxMouseEvent& event);
	    void OnLeftUp(wxMouseEvent& event);

		void OnMiddleUp(wxMouseEvent& event);

		void OnMouseWheel(wxMouseEvent& event);
		void LostMouseCapture(wxMouseCaptureLostEvent& event);
		void CalculateIntegration();


		void UpdateImageStatusInfo(long x_pos, long y_pos);

		*/
};

class
        DisplayPopup : public wxPopupWindow {

    DisplayPanel* parent_display_panel;

  public:
    DisplayPopup(wxWindow* parent, int flags = wxBORDER_NONE);

    void OnPaint(wxPaintEvent& evt);
    void OnEraseBackground(wxEraseEvent& event);

    int x_pos;
    int y_pos;

    float current_low_grey_value;
    float current_high_grey_value;
};

class DisplayManualDialog : public DisplayManualDialogParent {
  public:
    DisplayManualDialog(wxWindow* parent, int id, const wxString& title, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxDEFAULT_DIALOG_STYLE);

  private:
    wxTextCtrl*   toolbar_location_text;
    wxStaticText* toolbar_number_of_locations_text;

    DisplayPanel* my_parent;

    void OnPaint(wxPaintEvent& evt);

    void PaintHistogram( );
    void GetLocalHistogram( );
    bool GetGlobalHistogram( );

    void OnButtonCancel(wxCommandEvent& WXUNUSED(event));
    void OnClose(wxCloseEvent& event);
    void OnLeftDown(wxMouseEvent& event);
    void OnRightDown(wxMouseEvent& event);
    void OnMotion(wxMouseEvent& event);
    void OnLowChange(wxCommandEvent& WXUNUSED(event));
    void OnHighChange(wxCommandEvent& WXUNUSED(event));
    void OnButtonOK(wxCommandEvent& WXUNUSED(event));
    void OnImageChange(wxCommandEvent& WXUNUSED(event));
    void OnHistogramCheck(wxCommandEvent& WXUNUSED(event));
    void OnRealtimeCheck(wxCommandEvent& WXUNUSED(event));
    void OnNext(wxCommandEvent& event);
    void OnPrevious(wxCommandEvent& event);

    Image InputImage;

    float*   histogram;
    float*   global_histogram;
    wxBitmap histogram_bitmap;

    bool have_global_histogram;

    float min_grey_level;
    float max_grey_level;

    float grey_level_increment;
    float global_grey_level_increment;

    long current_location;

    int current_grey_method;
};

enum {
    Toolbar_Open,
    Toolbar_Location_Text,
    Toolbar_Previous,
    Toolbar_Next,
    //	Toolbar_Save,
    //	Toolbar_Previous,
    //	Toolbar_Next,
    //	Toolbar_Previous_Filament,
    //	Toolbar_Next_Filament,
    Toolbar_Local,
    Toolbar_Auto,
    Toolbar_Global,
    Toolbar_Manual,
    Toolbar_Scale_Combo_Control,
    Toolbar_Refresh,
    Toolbar_Histogram,
    Toolbar_FFT,
    Toolbar_High_Quality,
    Toolbar_Invert,
    //	Location_Text_Control,
    //	Filament_Number_Text_Control,
    //	Toolbar_Filament_Only_Check,
    Manual_Image_TextCtrl,
    Manual_Min_TextCtrl,
    Manual_Histogram_All_Check,
    Realtime_Update_Check,
    Manual_Max_TextCtrl

};

#endif
