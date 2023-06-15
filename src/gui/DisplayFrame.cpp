#include "../core/gui_core_headers.h"

DisplayFrame::DisplayFrame(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style)
    : DisplayFrameParent(NULL, wxID_ANY, title, pos, size, style) {

    is_fullscreen = false;
    image_is_open = false;

    this->cisTEMDisplayPanel->Initialise(CAN_CHANGE_FILE | CAN_CLOSE_TABS | CAN_MOVE_TABS | CAN_FFT | CAN_SELECT_IMAGES | CAN_SELECT_COORDS);

    int screen_x_size = wxSystemSettings::GetMetric(wxSYS_SCREEN_X);
    int screen_y_size = wxSystemSettings::GetMetric(wxSYS_SCREEN_Y);
    int x_offset;
    int y_offset;

    if ( screen_x_size > 1920 && screen_y_size > 1080 ) {
        x_offset = (screen_x_size - 1920) / 2;
        y_offset = (screen_y_size - 1080) / 2;

        if ( x_offset < 0 )
            x_offset = 0;
        if ( y_offset < 0 )
            y_offset = 0;

        SetSize(x_offset, y_offset, 1920, 1080);
    }
    else {
        Maximize(true);
    }

    Bind(wxEVT_CHAR_HOOK, &DisplayFrame::OnCharHook, this);
}

DisplayFrame::~DisplayFrame( ) {
}

void DisplayFrame::OnCharHook(wxKeyEvent& event) {
    if ( event.GetKeyCode( ) == WXK_F11 ) {
        if ( is_fullscreen == true ) {
            ShowFullScreen(false);
            is_fullscreen = false;
        }
        else {
            ShowFullScreen(true);
            is_fullscreen = true;
        }
    }
    event.Skip( );
}

void DisplayFrame::OnFileOpenClick(wxCommandEvent& event) {
    this->cisTEMDisplayPanel->OnOpen(event);
    this->image_is_open = true;
}

void DisplayFrame::OpenFile(wxString wanted_filename, wxString wanted_tab_title, wxArrayLong* wanted_included_image_numbers, bool keep_scale_and_location_if_possible, bool force_local_survey) {
    this->cisTEMDisplayPanel->OpenFile(wanted_filename, wanted_tab_title, wanted_included_image_numbers, keep_scale_and_location_if_possible, force_local_survey);
}

void DisplayFrame::OnCloseTabClick(wxCommandEvent& event) {
    if ( this->cisTEMDisplayPanel != NULL && cisTEMDisplayPanel->ReturnCurrentPanel( ) != NULL ) {
        this->cisTEMDisplayPanel->my_notebook->DeletePage(this->cisTEMDisplayPanel->my_notebook->GetSelection( ));
    }
    if ( this->cisTEMDisplayPanel->my_notebook->GetSelection( ) == wxNOT_FOUND ) {
        this->DisableAllToolbarButtons( );
        this->image_is_open = false;
    }
}

void DisplayFrame::OnExitClick(wxCommandEvent& event) {
    this->Destroy( );
}

void DisplayFrame::OnLocationNumberClick(wxCommandEvent& event) {
    if ( this->cisTEMDisplayPanel->ReturnCurrentPanel( )->show_label == true )
        this->cisTEMDisplayPanel->ReturnCurrentPanel( )->show_label = false;
    else
        this->cisTEMDisplayPanel->ReturnCurrentPanel( )->show_label = true;

    this->cisTEMDisplayPanel->ReturnCurrentPanel( )->ReDrawPanel( );
}

void DisplayFrame::OnImageSelectionModeClick(wxCommandEvent& event) {
    // if we are already in selections mode, we don't want to do anything, so
    // make a check.

    if ( this->SelectCoordsSelectionMode->IsChecked( ) ) {
        this->SelectCoordsSelectionMode->Check(false);
        this->SelectImageSelectionMode->Check(true);
    }

    if ( this->cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode != IMAGES_PICK ) {
        // First, get into image picking mode
        this->cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode = IMAGES_PICK;

        // Then change the check marks

        wxCommandEvent blank_event;

        // ok, so the user is asking to change to images mode, if they have
        // already made some selections in the current mode, we need to show
        // a dialog to make sure they want to change the mode, and lose
        // their current selections.

        if ( this->cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->number_of_coords > 0 ) {
            int dialog_result;

            wxMessageDialog question_dialog(this, wxT("By switching the selection mode, you will lose your current coordinates selections if they are unsaved!\nAre you sure you want to continue?"), wxT("Are you Sure?"), wxYES_NO | wxNO_DEFAULT | wxICON_EXCLAMATION);
            dialog_result = question_dialog.ShowModal( );

            if ( dialog_result == wxID_YES ) {
                this->cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode = IMAGES_PICK;
                OnClearSelectionClick(blank_event);
                this->cisTEMDisplayPanel->ReturnCurrentPanel( )->have_plt_filename = false;
            }
        }
        else {
            this->cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode = IMAGES_PICK;
            OnClearSelectionClick(blank_event);
            this->cisTEMDisplayPanel->ReturnCurrentPanel( )->have_plt_filename = false;
        }
    }
}

void DisplayFrame::OnCoordsSelectionModeClick(wxCommandEvent& event) {
    // Need to make sure the check mark switches
    if ( this->SelectImageSelectionMode->IsChecked( ) ) {
        this->SelectImageSelectionMode->Check(false);
        this->SelectCoordsSelectionMode->Check(true);
    }

    if ( this->cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode != COORDS_PICK ) {
        this->cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode = COORDS_PICK;

        wxMessageDialog question_dialog(this, wxT("By switching the selection mode, you will lose your current coordinates selections if they are unsaved!\nAre you sure you want to continue?"), wxT("Are you Sure?"), wxYES_NO | wxNO_DEFAULT | wxICON_EXCLAMATION);
        dialog_result = question_dialog.ShowModal( );
    }
}

void DisplayFrame::OnOpenPLTClick(wxCommandEvent& event) {
}

void DisplayFrame::OnSavePLTClick(wxCommandEvent& event) {
}

void DisplayFrame::OnSavePLTAsClick(wxCommandEvent& event) {
}

void DisplayFrame::OnInvertSelectionClick(wxCommandEvent& event) {
}

void DisplayFrame::OnClearSelectionClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode == IMAGES_PICK ) {
        cisTEMDisplayPanel->ReturnCurrentPanel( )->number_of_selections = 0;

        for ( int image_counter = 0; image_counter <= this->cisTEMDisplayPanel->number_of_frames; image_counter++ ) {
            cisTEMDisplayPanel->ReturnCurrentPanel( )->image_is_selected[image_counter] = false;
        }
    }
    else if ( cisTEMDisplayPanel->ReturnCurrentPanel( ) == COORDS_PICK ) {
        cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->Clear( );
    }

    //this->cisTEMDisplayPanel->SetTabNameSaved( );
    cisTEMDisplayPanel->RefreshCurrentPanel( );
}

void DisplayFrame::OnSetPointSizeClick(wxCommandEvent& event) {
}

void DisplayFrame::OnShowCrossHairClick(wxCommandEvent& event) {
}

void DisplayFrame::OnSingleImageModeClick(wxCommandEvent& event) {
}

void DisplayFrame::On7BitGreyValuesClick(wxCommandEvent& event) {
}

void DisplayFrame::OnShowSelectionDistancesClick(wxCommandEvent& event) {
}

void DisplayFrame::OnDocumentationClick(wxCommandEvent& event) {
    wxLaunchDefaultBrowser("http://www.cistem.org/documentation");
}

// This prevents using buttons when an image or stack is not open to act on
void DisplayFrame::DisableAllToolbarButtons( ) {

    // Open menu only needs close tab disabled
    this->DisplayCloseTab->Enable(false);

    // Label menu
    this->LabelLocationNumber->Enable(false);

    // Select menu
    this->SelectImageSelectionMode->Enable(false);
    this->SelectCoordsSelectionMode->Enable(false);
    this->SelectOpenPLT->Enable(false);
    this->SelectSavePLT->Enable(false);
    this->SelectSavePLTAs->Enable(false);
    this->SelectInvertSelection->Enable(false);
    this->SelectClearSelection->Enable(false);

    // Options menu
    this->OptionsSetPointSize->Enable(false);
    this->OptionsShowCrossHair->Enable(false);
    this->Options7BitGreyValues->Enable(false);
    this->OptionsSingleImageMode->Enable(false);
    this->OptionsShowSelectionDistances->Enable(false);
    this->OptionsShowResolution->Enable(false);
}

// Call when an image is opened to activate all toolbar buttons
void DisplayFrame::EnableAllToolbarButtons( ) {
    // Open menu only needs close tab disabled
    this->DisplayCloseTab->Enable( );

    // Label menu
    this->LabelLocationNumber->Enable( );

    // Select menu
    this->SelectImageSelectionMode->Enable(true);
    this->SelectCoordsSelectionMode->Enable(true);
    this->SelectOpenPLT->Enable(true);
    this->SelectSavePLT->Enable(true);
    this->SelectSavePLTAs->Enable(true);
    this->SelectInvertSelection->Enable(true);
    this->SelectClearSelection->Enable(true);

    // Options menu
    this->OptionsSetPointSize->Enable(true);
    this->OptionsShowCrossHair->Enable(true);
    this->Options7BitGreyValues->Enable(true);
    this->OptionsSingleImageMode->Enable(true);
    this->OptionsShowSelectionDistances->Enable(true);
    this->OptionsShowResolution->Enable(true);
}

void DisplayFrame::OnUpdateUI(wxUpdateUIEvent& event) {
    if ( this->cisTEMDisplayPanel->my_notebook->GetSelection( ) != wxNOT_FOUND ) {
        this->EnableAllToolbarButtons( );
    }
    else
        this->DisableAllToolbarButtons( );
}