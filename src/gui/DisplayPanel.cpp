#include "../core/gui_core_headers.h"

DisplayPanel::DisplayPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : DisplayPanelParent(parent, id, pos, size, style) {

    toolbar_location_text            = NULL;
    toolbar_number_of_locations_text = NULL;
    toolbar_scale_combo              = NULL;
    my_notebook                      = NULL;
    StatusText                       = NULL;
    no_notebook_panel                = NULL;

    panel_counter = 0;

    popup_exists = false;

    Bind(wxEVT_MENU, &DisplayPanel::OnAuto, this, Toolbar_Auto);
    Bind(wxEVT_MENU, &DisplayPanel::OnLocal, this, Toolbar_Local);
    Bind(wxEVT_MENU, &DisplayPanel::OnGlobal, this, Toolbar_Global);
    Bind(wxEVT_MENU, &DisplayPanel::OnPrevious, this, Toolbar_Previous);
    Bind(wxEVT_MENU, &DisplayPanel::OnNext, this, Toolbar_Next);
    Bind(wxEVT_MENU, &DisplayPanel::OnManual, this, Toolbar_Manual);
    Bind(wxEVT_MENU, &DisplayPanel::OnHistogram, this, Toolbar_Histogram);
    Bind(wxEVT_MENU, &DisplayPanel::OnFFT, this, Toolbar_FFT);
    Bind(wxEVT_MENU, &DisplayPanel::OnHighQuality, this, Toolbar_High_Quality);
    Bind(wxEVT_MENU, &DisplayPanel::OnInvert, this, Toolbar_Invert);
    Bind(wxEVT_MENU, &DisplayPanel::OnOpen, this, Toolbar_Open);

    Bind(wxEVT_TEXT_ENTER, &DisplayPanel::ChangeLocation, this, Toolbar_Location_Text);
    Bind(wxEVT_TEXT_ENTER, &DisplayPanel::ChangeScaling, this, Toolbar_Scale_Combo_Control);
    Bind(wxEVT_COMBOBOX, &DisplayPanel::ChangeScaling, this, Toolbar_Scale_Combo_Control);
}

void DisplayPanel::Initialise(int wanted_style_flags) {

#include "icons/display_open_icon.cpp"
#include "icons/display_previous_icon.cpp"
#include "icons/display_next_icon.cpp"
#include "icons/display_local_icon.cpp"
#include "icons/display_auto_icon.cpp"
#include "icons/display_global_icon.cpp"
#include "icons/display_manual_icon.cpp"
#include "icons/display_histogram_icon.cpp"
#include "icons/display_refresh_icon.cpp"
#include "icons/display_fft_icon.cpp"
#include "icons/display_invert_icon.cpp"
#include "icons/display_high_quality_icon.cpp"

    style_flags = wanted_style_flags;

    if ( my_notebook != NULL ) {
        delete my_notebook;
        my_notebook = NULL;
    }

    if ( no_notebook_panel != NULL ) {
        delete no_notebook_panel;
        no_notebook_panel = NULL;
    }

    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
    }
    else {
        long flags = wxAUI_NB_SCROLL_BUTTONS | wxAUI_NB_TOP | wxAUI_NB_WINDOWLIST_BUTTON;

        if ( (style_flags & CAN_CLOSE_TABS) == CAN_CLOSE_TABS )
            flags |= wxAUI_NB_CLOSE_ON_ACTIVE_TAB;
        if ( (style_flags & CAN_MOVE_TABS) == CAN_MOVE_TABS )
            flags |= wxAUI_NB_TAB_MOVE;

        my_notebook = new DisplayNotebook(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, flags);

        MainSizer->Add(my_notebook, 1, wxEXPAND | wxALL, 5);
    }

    if ( StatusText != NULL )
        delete StatusText;

    StatusText = new wxStaticText(this, wxID_ANY, wxT(""), wxDefaultPosition, wxDefaultSize, 0);
    StatusText->Wrap(-1);
    MainSizer->Add(StatusText, 0, wxALL | wxEXPAND, 5);

    Layout( );

    wxLogNull* suppress_png_warnings = new wxLogNull;

    // setup the toolbar..

    if ( (style_flags & CAN_CHANGE_FILE) == CAN_CHANGE_FILE ) {
        Toolbar->AddTool(Toolbar_Open, wxT("Open Image file"), wxBITMAP_PNG_FROM_DATA(display_open_icon), wxNullBitmap, wxITEM_NORMAL, wxT("Open Image File"), wxT("Open an new image file in a new tab"));
        Toolbar->AddSeparator( );
        Toolbar->EnableTool(Toolbar_Open, true);
    }

    if ( toolbar_location_text != NULL )
        delete toolbar_location_text;
    toolbar_location_text = new wxTextCtrl(Toolbar, Toolbar_Location_Text, wxT(""), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER | wxTE_CENTRE, wxDefaultValidator, wxTextCtrlNameStr);
    toolbar_location_text->Enable(false);

    if ( toolbar_number_of_locations_text != NULL )
        delete toolbar_number_of_locations_text;
    toolbar_number_of_locations_text = new wxStaticText(Toolbar, wxID_ANY, wxT(""), wxDefaultPosition, wxSize(-1, -1));

    if ( (style_flags & FIRST_LOCATION_ONLY) == FIRST_LOCATION_ONLY ) {
    }
    else {
        Toolbar->AddTool(Toolbar_Previous, wxT("Previous"), wxBITMAP_PNG_FROM_DATA(display_previous_icon), wxNullBitmap, wxITEM_NORMAL, wxT("Previous"), wxT("Move to the previous set of images"));
        Toolbar->EnableTool(Toolbar_Previous, false);

        Toolbar->AddControl(toolbar_location_text);
        Toolbar->AddControl(toolbar_number_of_locations_text);

        Toolbar->AddTool(Toolbar_Next, wxT("Next"), wxBITMAP_PNG_FROM_DATA(display_next_icon), wxNullBitmap, wxITEM_NORMAL, wxT("Next"), wxT("Move to the next set of images"));
        Toolbar->EnableTool(Toolbar_Next, false);
        Toolbar->AddSeparator( );
    }

    wxString combo_choices[9];
    combo_choices[0] = wxT("300%");
    combo_choices[1] = wxT("200%");
    combo_choices[2] = wxT("150%");
    combo_choices[3] = wxT("100%");
    combo_choices[4] = wxT("66%");
    combo_choices[5] = wxT("50%");
    combo_choices[6] = wxT("33%");
    combo_choices[7] = wxT("25%");
    combo_choices[8] = wxT("10%");

    if ( toolbar_scale_combo != NULL )
        delete toolbar_scale_combo;
    toolbar_scale_combo = new wxComboBox(Toolbar, Toolbar_Scale_Combo_Control, wxT(""), wxDefaultPosition, wxSize(100, -1), 9, combo_choices, wxTE_PROCESS_ENTER, wxDefaultValidator, wxT("comboBox"));
    toolbar_scale_combo->Enable(false);
    Toolbar->AddControl(toolbar_scale_combo);

    Toolbar->AddSeparator( );

    Toolbar->AddTool(Toolbar_Local, wxT("Local Greys"), wxBITMAP_PNG_FROM_DATA(display_local_icon), wxNullBitmap, wxITEM_RADIO, wxT("Local Greys"), wxT("Set the grey value to a local survey"));
    Toolbar->EnableTool(Toolbar_Local, false);

    Toolbar->AddTool(Toolbar_Auto, wxT("Auto Greys"), wxBITMAP_PNG_FROM_DATA(display_auto_icon), wxNullBitmap, wxITEM_RADIO, wxT("Auto Greys"), wxT("Set the grey values via automatic contrast"));
    Toolbar->EnableTool(Toolbar_Auto, false);

    Toolbar->AddTool(Toolbar_Global, wxT("Global Greys"), wxBITMAP_PNG_FROM_DATA(display_global_icon), wxNullBitmap, wxITEM_RADIO, wxT("Global Greys"), wxT("Set the grey value to a global survey"));
    Toolbar->EnableTool(Toolbar_Global, false);
    Toolbar->AddTool(Toolbar_Manual, wxT("Manual Greys"), wxBITMAP_PNG_FROM_DATA(display_manual_icon), wxNullBitmap, wxITEM_RADIO, wxT("Manual Greys"), wxT("Set the grey values to those you have manually chosen"));
    Toolbar->EnableTool(Toolbar_Manual, false);

    Toolbar->ToggleTool(Toolbar_Local, true);

    Toolbar->AddSeparator( );

    Toolbar->AddTool(Toolbar_Histogram, wxT("Grey Histogram"), wxBITMAP_PNG_FROM_DATA(display_histogram_icon), wxNullBitmap, wxITEM_NORMAL, wxT("Grey Histogram"), wxT("See Histogram and set the manual grey values"));
    Toolbar->EnableTool(Toolbar_Histogram, false);

    Toolbar->AddTool(Toolbar_Invert, wxT("Invert Contrast"), wxBITMAP_PNG_FROM_DATA(display_invert_icon), wxNullBitmap, wxITEM_CHECK, wxT("Invert Image Contrast"), wxT("Invert Image Contrast"));
    Toolbar->EnableTool(Toolbar_Invert, false);

    Toolbar->AddSeparator( );

    if ( (style_flags & CAN_FFT) == CAN_FFT ) {
        Toolbar->AddTool(Toolbar_FFT, wxT("FFT"), wxBITMAP_PNG_FROM_DATA(display_fft_icon), wxNullBitmap, wxITEM_CHECK, wxT("FFT Images"), wxT("Fourier Transform Images"));
        Toolbar->EnableTool(Toolbar_FFT, false);
    }

    Toolbar->AddTool(Toolbar_High_Quality, wxT("High Quality Scaling"), wxBITMAP_PNG_FROM_DATA(display_high_quality_icon), wxNullBitmap, wxITEM_CHECK, wxT("Use High Quality Scaling"), wxT("Use High Quality Scaling"));
    Toolbar->EnableTool(Toolbar_High_Quality, false);

    if ( (style_flags & CAN_CHANGE_FILE) == CAN_CHANGE_FILE ) {
        Toolbar->AddTool(Toolbar_Refresh, wxT("Refresh"), wxBITMAP_PNG_FROM_DATA(display_refresh_icon), wxNullBitmap, wxITEM_NORMAL, wxT("Refresh"), wxT("Refresh the current view"));
        Toolbar->EnableTool(Toolbar_Refresh, false);
    }

    delete suppress_png_warnings;

    Toolbar->Realize( );

    if ( (style_flags & DO_NOT_SHOW_STATUS_BAR) == DO_NOT_SHOW_STATUS_BAR ) {
        StatusText->Show(false);
    }
    else
        StatusText->Show(true);

    Layout( );
}

void DisplayPanel::ChangeLocation(wxCommandEvent& WXUNUSED(event)) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

    long set_location;

    // get what the current value is.. if it can be converted to a valid number then
    // change the current location to that number and redraw.. otherwise set the
    // value back to the previous value..

    wxString current_string = toolbar_location_text->GetValue( );
    bool     has_worked     = current_string.ToLong(&set_location);

    if ( has_worked == true ) {
        // is this number valid?

        if ( set_location > 0 && set_location <= current_panel->included_image_numbers.GetCount( ) ) {
            current_panel->current_location = set_location;

            current_panel->panel_image_has_correct_greys = false;
            current_panel->ReDrawPanel( );

            if ( (style_flags & KEEP_TABS_LINKED_IF_POSSIBLE) == KEEP_TABS_LINKED_IF_POSSIBLE ) {
                // set all tabs to this scaling
                long global_location = current_panel->current_location;

                if ( my_notebook != NULL ) {
                    for ( int page_counter = 0; page_counter < my_notebook->GetPageCount( ); page_counter++ ) {
                        current_panel = reinterpret_cast<DisplayNotebookPanel*>(my_notebook->GetPage(page_counter));
                        if ( current_panel->current_location != global_location ) {
                            if ( global_location > 0 && global_location <= current_panel->included_image_numbers.GetCount( ) ) {
                                current_panel->current_location              = global_location;
                                current_panel->panel_image_has_correct_greys = false;
                                //current_panel->ReDrawPanel();
                                current_panel->Refresh( );
                            }
                        }
                    }
                }
            }

            UpdateToolbar( );
        }
        else
            has_worked = false;
    }

    // if for some reason it hasn't worked - set it back to it's previous value..

    if ( has_worked == false ) {
        toolbar_location_text->SetValue(wxString::Format(wxT("%li"), current_panel->current_location));
        Refresh( );
        Update( );
    }

    ChangeFocusToPanel( );
}

void DisplayPanel::Clear( ) {
    CloseAllTabs( );
    UpdateToolbar( );
}

void DisplayPanel::SetActiveTemplateMatchMarkerPostion(float wanted_x_pos, float wanted_y_pos, float wanted_radius) {

    ((DisplayNotebookPanel*)my_notebook->GetPage(0))->template_matching_marker_x_pos  = wanted_x_pos;
    ((DisplayNotebookPanel*)my_notebook->GetPage(0))->template_matching_marker_y_pos  = wanted_y_pos;
    ((DisplayNotebookPanel*)my_notebook->GetPage(0))->template_matching_marker_radius = wanted_radius;

    ((DisplayNotebookPanel*)my_notebook->GetPage(1))->template_matching_marker_x_pos  = wanted_x_pos;
    ((DisplayNotebookPanel*)my_notebook->GetPage(1))->template_matching_marker_y_pos  = wanted_y_pos;
    ((DisplayNotebookPanel*)my_notebook->GetPage(1))->template_matching_marker_radius = 5;

    ((DisplayNotebookPanel*)my_notebook->GetPage(2))->template_matching_marker_x_pos  = wanted_x_pos;
    ((DisplayNotebookPanel*)my_notebook->GetPage(2))->template_matching_marker_y_pos  = wanted_y_pos;
    ((DisplayNotebookPanel*)my_notebook->GetPage(2))->template_matching_marker_radius = wanted_radius;

    RefreshCurrentPanel( );
}

void DisplayPanel::ClearActiveTemplateMatchMarker( ) {
    ((DisplayNotebookPanel*)my_notebook->GetPage(0))->template_matching_marker_x_pos  = -1;
    ((DisplayNotebookPanel*)my_notebook->GetPage(0))->template_matching_marker_y_pos  = -1;
    ((DisplayNotebookPanel*)my_notebook->GetPage(0))->template_matching_marker_radius = -1;

    ((DisplayNotebookPanel*)my_notebook->GetPage(1))->template_matching_marker_x_pos  = -1;
    ((DisplayNotebookPanel*)my_notebook->GetPage(1))->template_matching_marker_y_pos  = -1;
    ((DisplayNotebookPanel*)my_notebook->GetPage(1))->template_matching_marker_radius = -1;

    ((DisplayNotebookPanel*)my_notebook->GetPage(2))->template_matching_marker_x_pos  = -1;
    ((DisplayNotebookPanel*)my_notebook->GetPage(2))->template_matching_marker_y_pos  = -1;
    ((DisplayNotebookPanel*)my_notebook->GetPage(2))->template_matching_marker_radius = -1;

    RefreshCurrentPanel( );
}

void DisplayPanel::CloseAllTabs( ) {
    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
        // shall i delete the panel
        if ( no_notebook_panel != NULL ) {
            MainSizer->Detach(no_notebook_panel);
            no_notebook_panel->Destroy( );
            no_notebook_panel = NULL;
            //Layout();
        }
    }
    else {
        my_notebook->DeleteAllPages( );
    }
}

void DisplayPanel::ChangeScaling(wxCommandEvent& WXUNUSED(event)) {
    // need to get the value and strip it of it's % character if it has one..

    bool     convert_success = false;
    long     new_value;
    wxString combo_value = toolbar_scale_combo->GetValue( );

    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

    if ( combo_value.Right(1).IsSameAs(wxT("%")) == true ) {
        combo_value.RemoveLast( );
    }

    convert_success = combo_value.ToLong(&new_value);

    if ( convert_success == true && new_value > 0 && double(new_value) / 100. != current_panel->desired_scale_factor ) {
        current_panel->desired_scale_factor          = double(new_value) / 100.;
        current_panel->panel_image_has_correct_scale = false;
        current_panel->ReDrawPanel( );

        if ( (style_flags & KEEP_TABS_LINKED_IF_POSSIBLE) == KEEP_TABS_LINKED_IF_POSSIBLE ) {
            // set all tabs to this scaling
            double global_scale_factor = current_panel->desired_scale_factor;

            if ( my_notebook != NULL ) {
                for ( int page_counter = 0; page_counter < my_notebook->GetPageCount( ); page_counter++ ) {
                    current_panel = reinterpret_cast<DisplayNotebookPanel*>(my_notebook->GetPage(page_counter));
                    if ( current_panel->desired_scale_factor != global_scale_factor ) {
                        current_panel->desired_scale_factor          = global_scale_factor;
                        current_panel->panel_image_has_correct_scale = false;
                        //current_panel->ReDrawPanel();
                        current_panel->Refresh( );
                    }
                }
            }
        }

        UpdateToolbar( );
    }
    else {
        // something went wrong to replace the value with what it was before..
        wxString temp_string;
        temp_string = wxString::Format(wxT("%i"), int(myround(current_panel->desired_scale_factor * 100)));
        temp_string += wxT("%");
        toolbar_scale_combo->SetValue(temp_string);
        Refresh( );
        Update( );
    }

    //notebook->SetFocus();
    ChangeFocusToPanel( );
}

void DisplayPanel::OnFFT(wxCommandEvent& WXUNUSED(event)) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

    if ( current_panel->use_fft == true )
        current_panel->use_fft = false;
    else {
        current_panel->use_fft = true;
    }

    current_panel->should_refresh = true;
    current_panel->ReDrawPanel( );
}

void DisplayPanel::OnHighQuality(wxCommandEvent& WXUNUSED(event)) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

    if ( current_panel->use_fourier_scaling == true )
        current_panel->use_fourier_scaling = false;
    else {
        current_panel->use_fourier_scaling = true;
    }

    current_panel->should_refresh = true;
    current_panel->ReDrawPanel( );
}

void DisplayPanel::OnInvert(wxCommandEvent& WXUNUSED(event)) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

    if ( current_panel->invert_contrast == true )
        current_panel->invert_contrast = false;
    else {
        current_panel->invert_contrast = true;
    }

    current_panel->should_refresh = true;
    current_panel->ReDrawPanel( );
}

void DisplayPanel::OnOpen(wxCommandEvent& WXUNUSED(event)) {

    int  filename_length;
    char InputFilename[500];

    wxString caption = wxT("Choose Image File");
    //wxString wildcard = wxT("IMG files (*.img)|*.img");
    wxString wildcard        = wxT("IMG and MRC files (*.img;*.mrc)|*.img;*.mrc|All Files (*.*)|*.*");
    wxString remember_path   = wxGetCwd( );
    wxString defaultDir      = remember_path;
    wxString defaultFilename = wxEmptyString;

    wxFileDialog dialog(NULL, caption, defaultDir, defaultFilename, wildcard, wxFD_FILE_MUST_EXIST);
    if ( dialog.ShowModal( ) == wxID_OK ) {
        wxString path            = dialog.GetPath( );
        remember_path            = dialog.GetDirectory( );
        wxString   this_filename = dialog.GetFilename( );
        wxFileName filename      = this_filename;

        // ok, we got a filename - was it an imagic file?.

        wxString extension = this_filename.Mid(this_filename.Len( ) - 4);

        if ( extension.IsSameAs(wxT(".img")) ) {
            // we're opening an imagic file..

            this_filename.Truncate(this_filename.Len( ) - 4);
            filename_length = path.Len( );

            //	InputFilename = new char[filename_length];

            for ( int mycounter = 0; mycounter < filename_length - 4; mycounter++ ) {
                InputFilename[mycounter] = char(path.GetWritableChar(mycounter));
            }

            InputFilename[filename_length - 4] = 0;
            //	path.Truncate(path.Len() - 4);

            OpenFile(InputFilename, this_filename);
        }
        else {
            // Perhaps we are opening some kind of mrc file..

            filename_length = path.Len( );
            //InputFilename = new char[filename_length];

            for ( int mycounter = 0; mycounter < filename_length; mycounter++ ) {

                InputFilename[mycounter] = char(path.GetWritableChar(mycounter));
            }

            InputFilename[filename_length] = 0;

            // does it seem to be an MRC file? If so, try and open it
            bool is_valid;
            //wxStripExtension(&this_filename);
            if ( filename.GetExt( ).IsSameAs("mrc", false) || filename.GetExt( ).IsSameAs("mrcs", false) ) {
                is_valid = GetMRCDetails(path, x_size, y_size, number_of_frames); // This is being recognized as valid mrc now; good
                if ( is_valid ) {
                    OpenFile(InputFilename, this_filename);
                }
                else
                    wxMessageBox(wxT("This file is not a compatible type! Accepted types are mrc and mrcs."), wxT("Error"), wxOK | wxICON_INFORMATION);
            }
        }
        ReturnCurrentPanel( )->short_image_filename = this_filename;
        //delete [] InputFilename;
    }
}

void DisplayPanel::OnManual(wxCommandEvent& WXUNUSED(event)) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

    if ( (current_panel->manual_low_grey == 0 && current_panel->manual_high_grey == 0) || current_panel->grey_values_decided_by == MANUAL_GREYS ) {
        DisplayManualDialog* my_dialog = new DisplayManualDialog(this, wxID_ANY, wxT("Histogram /  Grey Settings"));
        my_dialog->ShowModal( );
    }
    else {
        current_panel->low_grey_value                = current_panel->manual_low_grey;
        current_panel->high_grey_value               = current_panel->manual_high_grey;
        current_panel->grey_values_decided_by        = MANUAL_GREYS;
        current_panel->panel_image_has_correct_greys = false;
        current_panel->ReDrawPanel( );
    }
}

void DisplayPanel::OnHistogram(wxCommandEvent& WXUNUSED(event)) {
    DisplayManualDialog* my_dialog = new DisplayManualDialog(this, wxID_ANY, wxT("Histogram /  Grey Settings"));
    my_dialog->ShowModal( );
}

void DisplayPanel::OnPrevious(wxCommandEvent& WXUNUSED(event)) {
    // if we are drawing a popup - or we are drawing a selection square - STOP.

    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

    if ( popup_exists == true ) {
        ReleaseMouse( );
        current_panel->SetCursor(wxCursor(wxCURSOR_CROSS));
        popup->Destroy( );
        popup_exists = false;
    }
    /*
	if (drawing_selection_square == true)
	{
		drawing_selection_square = false;
	}
*/
    current_panel->current_location -= current_panel->images_in_current_view;
    if ( current_panel->current_location < 1 )
        current_panel->current_location = 1;

    current_panel->panel_image_has_correct_greys = false;
    current_panel->ReDrawPanel( );

    if ( (style_flags & KEEP_TABS_LINKED_IF_POSSIBLE) == KEEP_TABS_LINKED_IF_POSSIBLE ) {
        // set all tabs to this scaling
        long global_location = current_panel->current_location;

        if ( my_notebook != NULL ) {
            for ( int page_counter = 0; page_counter < my_notebook->GetPageCount( ); page_counter++ ) {
                current_panel = reinterpret_cast<DisplayNotebookPanel*>(my_notebook->GetPage(page_counter));
                if ( current_panel->current_location != global_location ) {
                    if ( global_location > 0 && global_location <= current_panel->included_image_numbers.GetCount( ) ) {
                        current_panel->current_location              = global_location;
                        current_panel->panel_image_has_correct_greys = false;
                        //current_panel->ReDrawPanel();
                        current_panel->Refresh( );
                    }
                }
            }
        }
    }

    UpdateToolbar( );
}

DisplayNotebookPanel* DisplayPanel::ReturnCurrentPanel( ) {
    DisplayNotebookPanel* current_panel;
    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK )
        current_panel = no_notebook_panel;
    else
        current_panel = (DisplayNotebookPanel*)my_notebook->GetCurrentPage( );

    MyDebugAssertTrue(current_panel != NULL, "Current panel is NULL!");
    return current_panel;
}

void DisplayPanel::OnNext(wxCommandEvent& WXUNUSED(event)) {
    // if we are drawing a popup - or we are drawing a selection square - STOP.

    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

    if ( popup_exists == true ) {
        ReleaseMouse( );
        current_panel->SetCursor(wxCursor(wxCURSOR_CROSS));
        popup->Destroy( );
        popup_exists = false;
    }
    /*
	if (drawing_selection_square == true)
	{
		drawing_selection_square = false;
	}*/

    current_panel->current_location += current_panel->images_in_current_view;
    if ( current_panel->current_location > current_panel->included_image_numbers.GetCount( ) )
        current_panel->current_location = current_panel->included_image_numbers.GetCount( );

    current_panel->panel_image_has_correct_greys = false;
    current_panel->ReDrawPanel( );

    if ( (style_flags & KEEP_TABS_LINKED_IF_POSSIBLE) == KEEP_TABS_LINKED_IF_POSSIBLE ) {
        // set all tabs to this scaling
        long global_location = current_panel->current_location;

        if ( my_notebook != NULL ) {
            for ( int page_counter = 0; page_counter < my_notebook->GetPageCount( ); page_counter++ ) {
                current_panel = reinterpret_cast<DisplayNotebookPanel*>(my_notebook->GetPage(page_counter));
                if ( current_panel->current_location != global_location ) {
                    if ( global_location > 0 && global_location <= current_panel->included_image_numbers.GetCount( ) ) {
                        current_panel->current_location              = global_location;
                        current_panel->panel_image_has_correct_greys = false;
                        //current_panel->ReDrawPanel();
                        current_panel->Refresh( );
                    }
                }
            }
        }
    }

    UpdateToolbar( );
}

void DisplayPanel::OnAuto(wxCommandEvent& WXUNUSED(event)) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );
    if ( current_panel->grey_values_decided_by == AUTO_GREYS ) {
    }
    else {
        current_panel->grey_values_decided_by        = AUTO_GREYS;
        current_panel->panel_image_has_correct_greys = false;
        current_panel->ReDrawPanel( );
    }
}

void DisplayPanel::OnLocal(wxCommandEvent& WXUNUSED(event)) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

    if ( current_panel->grey_values_decided_by == LOCAL_GREYS ) {
    }
    else {
        current_panel->grey_values_decided_by        = LOCAL_GREYS;
        current_panel->low_grey_value                = 0;
        current_panel->high_grey_value               = 0;
        current_panel->panel_image_has_correct_greys = false;
        current_panel->ReDrawPanel( );
    }
}

void DisplayPanel::OnGlobal(wxCommandEvent& WXUNUSED(event)) {

    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

    // do we already have the global values?

    if ( current_panel->global_low_grey == 0 && current_panel->global_high_grey == 0 ) {
        bool success = current_panel->SetGlobalGreys( );

        if ( success == true ) {
            current_panel->low_grey_value                = current_panel->global_low_grey;
            current_panel->high_grey_value               = current_panel->global_high_grey;
            current_panel->grey_values_decided_by        = GLOBAL_GREYS;
            current_panel->panel_image_has_correct_greys = false;
            current_panel->ReDrawPanel( );
        }
    }
    else {
        current_panel->low_grey_value                = current_panel->global_low_grey;
        current_panel->high_grey_value               = current_panel->global_high_grey;
        current_panel->grey_values_decided_by        = GLOBAL_GREYS;
        current_panel->panel_image_has_correct_greys = false;
        current_panel->ReDrawPanel( );
    }
}

void DisplayPanel::ChangeFocusToPanel(void) {
    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
        if ( no_notebook_panel != NULL )
            no_notebook_panel->SetFocus( );
    }
    else {
        if ( my_notebook->GetPageCount( ) > 0 ) {
            DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );
            current_panel->SetFocus( );
        }
    }
}

void DisplayPanel::UpdateToolbar(void) {
    bool blank_toolbar = false;

    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
        if ( no_notebook_panel == NULL )
            blank_toolbar = true;
    }
    else if ( my_notebook->GetPageCount( ) == 0 )
        blank_toolbar = true;

    if ( blank_toolbar == true ) {
        toolbar_location_text->SetValue(wxT(""));
        toolbar_location_text->Disable( );
        toolbar_scale_combo->SetValue(wxT(""));
        toolbar_scale_combo->Disable( );

        if ( (style_flags & CAN_CHANGE_FILE) == CAN_CHANGE_FILE )
            Toolbar->EnableTool(Toolbar_Open, true);
        //Toolbar->EnableTool(Toolbar_Save, false);
        Toolbar->EnableTool(Toolbar_Previous, false);
        Toolbar->EnableTool(Toolbar_Next, false);
        Toolbar->EnableTool(Toolbar_Histogram, false);
        if ( (style_flags & CAN_CHANGE_FILE) == CAN_CHANGE_FILE )
            Toolbar->EnableTool(Toolbar_Refresh, false);
        toolbar_number_of_locations_text->SetLabel(wxT(""));
        Toolbar->EnableTool(Toolbar_Local, false);
        Toolbar->EnableTool(Toolbar_Auto, false);
        Toolbar->EnableTool(Toolbar_Global, false);
        Toolbar->EnableTool(Toolbar_Manual, false);
        Toolbar->EnableTool(Toolbar_Invert, false);

        Toolbar->EnableTool(Toolbar_High_Quality, false);

        if ( (style_flags & CAN_FFT) == CAN_FFT )
            Toolbar->EnableTool(Toolbar_FFT, false);
    }
    else {
        if ( (style_flags & CAN_CHANGE_FILE) == CAN_CHANGE_FILE )
            Toolbar->EnableTool(Toolbar_Open, true);

        // there must be a tab then, so get the selected displayPanel..

        DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

        if ( current_panel != NULL ) {

            /*
		    if (current_panel->have_txt_filename == true)
		    {
		    	toolbar->EnableTool(Toolbar_Save, true);
		    }
		    else
		    {
		    	toolbar->EnableTool(Toolbar_Save, false);
		    }
		    */

            if ( current_panel->current_location > 1 )
                Toolbar->EnableTool(Toolbar_Previous, true);
            else
                Toolbar->EnableTool(Toolbar_Previous, false);

            if ( current_panel->current_location + current_panel->images_in_current_view <= current_panel->included_image_numbers.GetCount( ) )
                Toolbar->EnableTool(Toolbar_Next, true);
            else
                Toolbar->EnableTool(Toolbar_Next, false);

            // Enable the Previous/Next filament buttons

            toolbar_location_text->Enable( );
            toolbar_location_text->SetValue(wxString::Format(wxT("%i"), int(current_panel->current_location)));
            wxString temp_string = wxT(" / ");
            temp_string += wxString::Format(wxT("%i"), int(current_panel->included_image_numbers.GetCount( )));

            toolbar_number_of_locations_text->SetLabel(temp_string);

            toolbar_scale_combo->Enable( );
            temp_string = wxString::Format(wxT("%i"), int(myround(current_panel->actual_scale_factor * 100)));
            temp_string += wxT("%");
            toolbar_scale_combo->SetValue(temp_string);

            Toolbar->EnableTool(Toolbar_Local, true);
            Toolbar->EnableTool(Toolbar_Auto, true);
            Toolbar->EnableTool(Toolbar_Global, true);
            Toolbar->EnableTool(Toolbar_Manual, true);

            if ( (style_flags & CAN_CHANGE_FILE) == CAN_CHANGE_FILE )
                Toolbar->EnableTool(Toolbar_Refresh, true);
            Toolbar->EnableTool(Toolbar_Histogram, true);

            if ( (style_flags & CAN_FFT) == CAN_FFT ) {
                Toolbar->EnableTool(Toolbar_FFT, true);

                if ( current_panel->use_fft == true ) {
                    Toolbar->ToggleTool(Toolbar_FFT, true);
                }
                else {
                    Toolbar->ToggleTool(Toolbar_FFT, false);
                }
            }

            Toolbar->EnableTool(Toolbar_High_Quality, true);

            if ( current_panel->use_fourier_scaling == true ) {
                Toolbar->ToggleTool(Toolbar_High_Quality, true);
            }
            else {
                Toolbar->ToggleTool(Toolbar_High_Quality, false);
            }

            Toolbar->EnableTool(Toolbar_Invert, true);

            if ( current_panel->invert_contrast == true ) {
                Toolbar->ToggleTool(Toolbar_Invert, true);
            }
            else {
                Toolbar->ToggleTool(Toolbar_Invert, false);
            }

            if ( current_panel->grey_values_decided_by == LOCAL_GREYS ) {
                Toolbar->ToggleTool(Toolbar_Local, true);
                Toolbar->EnableTool(Toolbar_Histogram, false);
            }
            else if ( current_panel->grey_values_decided_by == GLOBAL_GREYS ) {
                Toolbar->ToggleTool(Toolbar_Global, true);
                Toolbar->EnableTool(Toolbar_Histogram, false);
            }
            if ( current_panel->grey_values_decided_by == MANUAL_GREYS ) {
                Toolbar->ToggleTool(Toolbar_Manual, true);
                Toolbar->EnableTool(Toolbar_Histogram, true);
            }
            else if ( current_panel->grey_values_decided_by == AUTO_GREYS ) {
                Toolbar->ToggleTool(Toolbar_Auto, true);
                Toolbar->EnableTool(Toolbar_Histogram, false);
            }
        }
    }

    Toolbar->Layout( );
    Refresh( );
    Update( );

    ChangeFocusToPanel( );
}

void DisplayPanel::RefreshCurrentPanel( ) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );
    current_panel->Refresh( );
    current_panel->Update( );
}

void DisplayPanel::SetSelectionSquareLocation(long wanted_location) {
    DisplayNotebookPanel* current_panel           = ReturnCurrentPanel( );
    current_panel->blue_selection_square_location = wanted_location;
    current_panel->Refresh( );
    current_panel->Update( );
}

bool DisplayPanel::IsImageSelected(long wanted_image) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );
    return current_panel->image_is_selected[wanted_image];
}

void DisplayPanel::SetImageSelected(long wanted_image, bool refresh) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );
    current_panel->SetImageSelected(wanted_image, refresh);
}

void DisplayPanel::SetImageNotSelected(long wanted_image, bool refresh) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );
    current_panel->SetImageNotSelected(wanted_image, refresh);
}

void DisplayPanel::ToggleImageSelected(long wanted_image, bool refresh) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );
    current_panel->ToggleImageSelected(wanted_image, refresh);
}

void DisplayPanel::ClearSelection(bool refresh) {
    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );
    current_panel->ClearSelection(refresh);
}

void DisplayPanel::OpenFile(wxString wanted_filename, wxString wanted_tab_title, wxArrayLong* wanted_included_image_numbers, bool keep_scale_and_location_if_possible, bool force_local_survey) {
    double current_scale_factor;
    long   current_image_location;

    if ( keep_scale_and_location_if_possible == true ) {
        DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

        if ( current_panel != NULL ) {
            current_scale_factor   = current_panel->desired_scale_factor;
            current_image_location = current_panel->current_location;
        }
        else {
            current_scale_factor   = 1.0;
            current_image_location = 1;
        }
    }
    else {
        current_scale_factor   = 1.0;
        current_image_location = 1;
    }

    if ( DoesFileExist(wanted_filename) == false ) {
        wxMessageBox(wxString::Format("Error, File does not exist (%s)", wanted_filename), wxT("Error"), wxOK | wxICON_INFORMATION, this);
        return;
    }

    if ( popup_exists == true ) {
        DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

        if ( current_panel != NULL ) {
            current_panel->ReleaseMouse( );
            current_panel->SetCursor(wxCursor(wxCURSOR_CROSS));
            popup->Destroy( );
            popup_exists = false;
        }
    }

    DisplayNotebookPanel* my_panel;

    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
        if ( no_notebook_panel != NULL )
            no_notebook_panel->Destroy( );
        no_notebook_panel = new DisplayNotebookPanel(this, panel_counter);
        my_panel          = no_notebook_panel;

        if ( force_local_survey == true )
            no_notebook_panel->grey_values_decided_by = LOCAL_GREYS;
        MainSizer->Insert(1, no_notebook_panel, 1, wxEXPAND | wxALL, 5);
        Layout( );
    }
    else {
        panel_counter++;
        my_panel = new DisplayNotebookPanel(my_notebook, panel_counter);
        if ( force_local_survey == true )
            my_panel->grey_values_decided_by = LOCAL_GREYS;
    }

    my_panel->my_file.OpenFile(wanted_filename.ToStdString( ), false);

    if ( my_panel->my_file.IsOpen( ) == false ) {
        wxMessageBox(wxString::Format("Error, Cannot open file (%s)", wanted_filename), wxT("Error"), wxOK | wxICON_INFORMATION, this);
        return;
    }

    // which images are we including..

    if ( wanted_included_image_numbers == NULL ) {
        for ( long counter = 1; counter <= my_panel->my_file.ReturnNumberOfSlices( ); counter++ ) {
            my_panel->included_image_numbers.Add(counter);
        }
    }
    else {
        for ( long counter = 0; counter < wanted_included_image_numbers->GetCount( ); counter++ ) {
            MyDebugAssertTrue(wanted_included_image_numbers->Item(counter) > 0 && wanted_included_image_numbers->Item(counter) <= my_panel->my_file.ReturnNumberOfSlices( ), "trying to add image numbers that don't exist")
                    my_panel->included_image_numbers.Add(wanted_included_image_numbers->Item(counter));
        }
    }

    // setup the panel attributes...

    my_panel->filename        = wanted_filename;
    my_panel->input_is_a_file = true;

    /*my_panel->image_is_selected = new bool[my_panel->first_header.number_following + 2];

	for (int mycounter = 0; mycounter < my_panel->first_header.number_following + 2; mycounter++)
	{
		my_panel->image_is_selected[mycounter] = false;
	}*/

    // add the panel

    if ( my_panel->panel_image != NULL )
        delete my_panel->panel_image;
    my_panel->panel_image = new wxImage(int(my_panel->my_file.ReturnXSize( )), int(my_panel->my_file.ReturnYSize( )));
    my_panel->tab_title   = wanted_tab_title;

    my_panel->desired_scale_factor = current_scale_factor;
    if ( my_panel->included_image_numbers.GetCount( ) > current_image_location ) {
        my_panel->current_location = current_image_location;
    }

    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
    }
    else {
        my_notebook->Freeze( );
        my_notebook->AddPage(my_panel, wanted_tab_title, false);
        my_notebook->Thaw( );
        my_notebook->SetSelection(my_notebook->GetPageCount( ) - 1);
    }

    // This was moved here because calling ClearSelection before the panel was added
    // resulted in a crash from a debug assert -- the current page was still null
    if ( (style_flags & CAN_SELECT_IMAGES) == CAN_SELECT_IMAGES || ReturnCurrentPanel( )->picking_mode == IMAGES_PICK ) {
        my_panel->image_is_selected = new bool[my_panel->my_file.ReturnNumberOfSlices( ) + 1];
        ClearSelection(false);
    }

    // we have switched focus so update toolbar..

    UpdateToolbar( );

    // we can directly call the drawing now..

    Refresh( );
    Update( );

    // if there is only one image, then set single image mode to true by default

    if ( my_panel->included_image_numbers.GetCount( ) == 1 ) {
        //my_panel->single_image = true;
        //my_panel->picking_mode = COORDS_PICK;
    }

    my_panel->ReDrawPanel( );
    //notebook->SetFocus();
    ChangeFocusToPanel( );
}

void DisplayPanel::ChangeFileForTabNumber(int wanted_tab_number, wxString wanted_filename, wxString wanted_tab_title, wxArrayLong* wanted_included_image_numbers) {
    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
        if ( no_notebook_panel == NULL ) {
            OpenFile(wanted_filename, wanted_tab_title, wanted_included_image_numbers);
            return;
        }
    }

    DisplayNotebookPanel* current_panel;
    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK )
        current_panel = no_notebook_panel;
    else {

        MyDebugAssertTrue(wanted_tab_number < my_notebook->GetPageCount( ) && wanted_tab_number >= 0, "Asking for a tab that does not exist");

        if ( wanted_tab_number == my_notebook->GetSelection( ) && popup_exists == true ) {
            DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

            if ( current_panel != NULL ) {
                current_panel->ReleaseMouse( );
                current_panel->SetCursor(wxCursor(wxCURSOR_CROSS));
                popup->Destroy( );
                popup_exists = false;
            }
        }

        current_panel = (DisplayNotebookPanel*)my_notebook->GetPage(wanted_tab_number);
    }

    if ( current_panel == NULL )
        return;

    if ( DoesFileExist(wanted_filename) == false ) {
        wxMessageBox(wxString::Format("Error, File does not exist (%s)", wanted_filename), wxT("Error"), wxOK | wxICON_INFORMATION, this);
        return;
    }

    if ( current_panel->input_is_a_file == true ) {
        if ( current_panel->my_file.IsOpen( ) == true )
            current_panel->my_file.CloseFile( );
    }
    else {
        if ( current_panel->image_to_display != NULL && current_panel->do_i_have_image_ownership == true ) {
            delete current_panel->image_to_display;
        }
    }

    current_panel->my_file.OpenFile(wanted_filename.ToStdString( ), false);

    if ( current_panel->my_file.IsOpen( ) == false ) {
        wxMessageBox(wxString::Format("Error, Cannot open file (%s)", wanted_filename), wxT("Error"), wxOK | wxICON_INFORMATION, this);
        return;
    }
    // which images are we including..

    current_panel->included_image_numbers.Clear( );

    if ( wanted_included_image_numbers == NULL ) {
        for ( long counter = 1; counter <= current_panel->my_file.ReturnNumberOfSlices( ); counter++ ) {
            current_panel->included_image_numbers.Add(counter);
        }
    }
    else {
        for ( long counter = 0; counter < wanted_included_image_numbers->GetCount( ); counter++ ) {
            MyDebugAssertTrue(wanted_included_image_numbers->Item(counter) > 0 && wanted_included_image_numbers->Item(counter) <= current_panel->my_file.ReturnNumberOfSlices( ), "trying to add image numbers that don't exist")
                    current_panel->included_image_numbers.Add(wanted_included_image_numbers->Item(counter));
        }
    }

    // setup the panel attributes...

    current_panel->filename        = wanted_filename;
    current_panel->input_is_a_file = true;

    if ( current_panel->image_is_selected != NULL ) {
        delete[] current_panel->image_is_selected;
        current_panel->image_is_selected = NULL;
    }

    if ( (style_flags & CAN_SELECT_IMAGES) == CAN_SELECT_IMAGES || ReturnCurrentPanel( )->picking_mode == IMAGES_PICK ) {
        current_panel->image_is_selected = new bool[current_panel->my_file.ReturnNumberOfSlices( ) + 1];

        for ( int mycounter = 0; mycounter < current_panel->my_file.ReturnNumberOfSlices( ) + 1; mycounter++ ) {
            current_panel->image_is_selected[mycounter] = false;
        }

        current_panel->number_of_selections = 0;
    }

    // add the panel

    current_panel->current_location              = 1;
    current_panel->should_refresh                = true;
    current_panel->panel_image_has_correct_greys = false;
    current_panel->panel_image_has_correct_scale = false;

    if ( current_panel->panel_image != NULL )
        delete current_panel->panel_image;
    current_panel->panel_image = new wxImage(int(current_panel->my_file.ReturnXSize( )), int(current_panel->my_file.ReturnYSize( )));
    current_panel->tab_title   = wanted_tab_title;

    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
    }
    else {

        my_notebook->SetPageText(wanted_tab_number, wanted_tab_title);

        if ( wanted_tab_number == my_notebook->GetSelection( ) ) {
            // we have switched focus so update toolbar..

            UpdateToolbar( );

            current_panel->ReDrawPanel( );
            ChangeFocusToPanel( );
            //notebook->SetFocus(
        }
    }

    current_panel->Refresh( );
    current_panel->Update( );
}

void DisplayPanel::ChangeFile(wxString wanted_filename, wxString wanted_tab_title, wxArrayLong* wanted_included_image_numbers) {
    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
        ChangeFileForTabNumber(-1, wanted_filename, wanted_tab_title, wanted_included_image_numbers);
    }
    else
        ChangeFileForTabNumber(my_notebook->GetSelection( ), wanted_filename, wanted_tab_title, wanted_included_image_numbers);
}

void DisplayPanel::OpenImage(Image* image_to_view, wxString wanted_tab_title, bool take_ownership, wxArrayLong* wanted_included_image_numbers) {
    if ( image_to_view == NULL ) {
        wxMessageBox(wxString::Format("Error, image is null"), wxT("Error"), wxOK | wxICON_INFORMATION, this);
        return;
    }

    if ( popup_exists == true ) {
        DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

        if ( current_panel != NULL ) {
            current_panel->ReleaseMouse( );
            current_panel->SetCursor(wxCursor(wxCURSOR_CROSS));
            popup->Destroy( );
            popup_exists = false;
        }
    }

    DisplayNotebookPanel* my_panel;

    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
        if ( no_notebook_panel != NULL )
            no_notebook_panel->Destroy( );
        no_notebook_panel = new DisplayNotebookPanel(this, panel_counter);
        my_panel          = no_notebook_panel;
        MainSizer->Insert(1, no_notebook_panel, 1, wxEXPAND | wxALL, 5);
    }
    else {
        panel_counter++;
        my_panel = new DisplayNotebookPanel(my_notebook, panel_counter);
    }

    // which images are we including..

    if ( wanted_included_image_numbers == NULL ) {
        for ( long counter = 1; counter <= image_to_view->logical_z_dimension; counter++ ) {
            my_panel->included_image_numbers.Add(counter);
        }
    }
    else {
        for ( long counter = 0; counter < wanted_included_image_numbers->GetCount( ); counter++ ) {
            MyDebugAssertTrue(wanted_included_image_numbers->Item(counter) > 0 && wanted_included_image_numbers->Item(counter) <= image_to_view->logical_z_dimension, "trying to add image numbers that don't exist")
                    my_panel->included_image_numbers.Add(wanted_included_image_numbers->Item(counter));
        }
    }

    // setup the panel attributes...

    my_panel->image_to_display          = image_to_view;
    my_panel->filename                  = "";
    my_panel->input_is_a_file           = false;
    my_panel->do_i_have_image_ownership = take_ownership;

    if ( (style_flags & CAN_SELECT_IMAGES) == CAN_SELECT_IMAGES || ReturnCurrentPanel( )->picking_mode == IMAGES_PICK ) {
        my_panel->image_is_selected = new bool[image_to_view->logical_z_dimension + 1];
        ClearSelection(false);
    }

    // add the panel

    if ( my_panel->panel_image != NULL )
        delete my_panel->panel_image;
    my_panel->panel_image = new wxImage(image_to_view->logical_x_dimension, image_to_view->logical_y_dimension);
    my_panel->tab_title   = wanted_tab_title;

    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
    }
    else {
        my_notebook->Freeze( );
        my_notebook->AddPage(my_panel, wanted_tab_title, false);
        my_notebook->Thaw( );
        my_notebook->SetSelection(my_notebook->GetPageCount( ) - 1);
    }

    // we have switched focus so update toolbar..

    UpdateToolbar( );

    // we can directly call the drawing now..

    Refresh( );
    Update( );

    // if there is only one image, then set single image mode to true by default

    if ( my_panel->included_image_numbers.GetCount( ) == 1 ) {
        //my_panel->single_image = true;
        //my_panel->picking_mode = COORDS_PICK;
    }

    my_panel->ReDrawPanel( );
    //notebook->SetFocus();
    ChangeFocusToPanel( );
}

void DisplayPanel::ChangeImage(Image* image_to_view, wxString wanted_tab_title, bool take_ownership, wxArrayLong* wanted_included_image_numbers) {
    if ( popup_exists == true ) {
        DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );

        if ( current_panel != NULL ) {
            current_panel->ReleaseMouse( );
            current_panel->SetCursor(wxCursor(wxCURSOR_CROSS));
            popup->Destroy( );
            popup_exists = false;
        }
    }

    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
        if ( no_notebook_panel == NULL ) {
            OpenImage(image_to_view, wanted_tab_title, take_ownership, wanted_included_image_numbers);
            return;
        }
    }
    else {
        if ( my_notebook->GetPageCount( ) == 0 ) {
            OpenImage(image_to_view, wanted_tab_title, take_ownership, wanted_included_image_numbers);
            return;
        }
    }

    DisplayNotebookPanel* current_panel = ReturnCurrentPanel( );
    if ( current_panel == NULL )
        return;

    if ( image_to_view == NULL ) {
        wxMessageBox(wxString::Format("Error, image is null"), wxT("Error"), wxOK | wxICON_INFORMATION, this);
        return;
    }

    current_panel->included_image_numbers.Clear( );
    if ( wanted_included_image_numbers == NULL ) {
        for ( long counter = 1; counter <= image_to_view->logical_z_dimension; counter++ ) {
            current_panel->included_image_numbers.Add(counter);
        }
    }
    else {
        for ( long counter = 0; counter < wanted_included_image_numbers->GetCount( ); counter++ ) {
            MyDebugAssertTrue(wanted_included_image_numbers->Item(counter) > 0 && wanted_included_image_numbers->Item(counter) <= image_to_view->logical_z_dimension, "trying to add image numbers that don't exist")
                    current_panel->included_image_numbers.Add(wanted_included_image_numbers->Item(counter));
        }
    }

    if ( current_panel->input_is_a_file == true ) {
        current_panel->input_is_a_file           = false;
        current_panel->do_i_have_image_ownership = take_ownership;
        current_panel->image_to_display          = image_to_view;

        if ( current_panel->my_file.IsOpen( ) == true )
            current_panel->my_file.CloseFile( );
    }
    else {
        bool   delete_old_image = false;
        Image* old_image;

        if ( current_panel->image_to_display != NULL && current_panel->do_i_have_image_ownership == true ) {
            delete_old_image = true;
            old_image        = current_panel->image_to_display;
        }

        current_panel->input_is_a_file           = false;
        current_panel->do_i_have_image_ownership = take_ownership;
        current_panel->image_to_display          = image_to_view;

        if ( delete_old_image == true ) {
            delete old_image;
        }
    }

    // setup the panel attributes...

    current_panel->filename = "";

    if ( current_panel->image_is_selected != NULL ) {
        delete[] current_panel->image_is_selected;
        current_panel->image_is_selected = NULL;
    }

    // Image or stack just opened; set all images as not selected
    if ( (style_flags & CAN_SELECT_IMAGES) == CAN_SELECT_IMAGES || ReturnCurrentPanel( )->picking_mode == IMAGES_PICK ) {
        current_panel->image_is_selected = new bool[image_to_view->logical_z_dimension + 1];
        for ( int mycounter = 0; mycounter < image_to_view->logical_z_dimension + 1; mycounter++ ) {
            current_panel->image_is_selected[mycounter] = false;
        }

        current_panel->number_of_selections = 0;
    }

    // add the panel

    current_panel->current_location              = 1;
    current_panel->should_refresh                = true;
    current_panel->panel_image_has_correct_greys = false;
    current_panel->panel_image_has_correct_scale = false;

    if ( current_panel->panel_image != NULL )
        delete current_panel->panel_image;
    current_panel->panel_image = new wxImage(image_to_view->logical_x_dimension, image_to_view->logical_y_dimension);
    current_panel->tab_title   = wanted_tab_title;

    if ( (style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
    }
    else {
        my_notebook->SetPageText(my_notebook->GetSelection( ), wanted_tab_title);
    }

    // we have switched focus so update toolbar..

    UpdateToolbar( );

    // we can directly call the drawing now..

    Refresh( );
    Update( );

    current_panel->ReDrawPanel( );
    //notebook->SetFocus();
    ChangeFocusToPanel( );
}

DisplayNotebook::DisplayNotebook(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : wxAuiNotebook(parent, id, pos, size, style) {
    parent_display_panel = reinterpret_cast<DisplayPanel*>(parent);

    Bind(wxEVT_AUINOTEBOOK_PAGE_CHANGED, &DisplayNotebook::OnSelectionChange, this);
    Bind(wxEVT_AUINOTEBOOK_END_DRAG, &DisplayNotebook::OnDragEnd, this);
    Bind(wxEVT_AUINOTEBOOK_PAGE_CLOSED, &DisplayNotebook::OnClosed, this);
    Bind(wxEVT_CHILD_FOCUS, &DisplayNotebook::ChildGotFocus, this);
    Bind(wxEVT_SET_FOCUS, &DisplayNotebook::GotFocus, this);
}

void DisplayNotebook::OnSelectionChange(wxAuiNotebookEvent& event) {
    parent_display_panel->UpdateToolbar( );
    event.Skip( );
}

void DisplayNotebook::OnDragEnd(wxAuiNotebookEvent& event) {
    parent_display_panel->UpdateToolbar( );
    event.Skip( );
}

void DisplayNotebook::OnClosed(wxAuiNotebookEvent& event) {
    parent_display_panel->UpdateToolbar( );
    event.Skip( );
}

void DisplayNotebook::ChildGotFocus(wxChildFocusEvent& event) {
    //wxPrintf("Child focus\n");
    parent_display_panel->ChangeFocusToPanel( );
    event.Skip( );
}

void DisplayNotebook::GotFocus(wxFocusEvent& event) {
    parent_display_panel->ChangeFocusToPanel( );
    event.Skip( );
}

DisplayNotebookPanel::DisplayNotebookPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : wxPanel(parent, id, pos, size, style) {
    image_memory_buffer        = NULL;
    scaled_image_memory_buffer = NULL;
    image_is_selected          = NULL;
    image_to_display           = NULL;

    coord_tracker = new CoordTracker(this); // Create coord_tracker for use when switching to coords picking

    input_is_a_file = true;

    number_allocated_for_buffer = 0;
    panel_image                 = NULL;

    if ( parent->IsKindOf(wxCLASSINFO(DisplayNotebook)) )
        parent_display_panel = reinterpret_cast<DisplayNotebook*>(parent)->parent_display_panel;
    else
        parent_display_panel = reinterpret_cast<DisplayPanel*>(parent);

    Bind(wxEVT_PAINT, &DisplayNotebookPanel::OnPaint, this);
    Bind(wxEVT_ERASE_BACKGROUND, &DisplayNotebookPanel::OnEraseBackground, this);
    Bind(wxEVT_SIZE, &DisplayNotebookPanel::OnSize, this);
    Bind(wxEVT_RIGHT_DOWN, &DisplayNotebookPanel::OnRightDown, this);
    Bind(wxEVT_LEFT_DOWN, &DisplayNotebookPanel::OnLeftDown, this);
    Bind(wxEVT_RIGHT_UP, &DisplayNotebookPanel::OnRightUp, this);
    Bind(wxEVT_MOTION, &DisplayNotebookPanel::OnMotion, this);
    Bind(wxEVT_KEY_DOWN, &DisplayNotebookPanel::OnKeyDown, this);
    Bind(wxEVT_KEY_UP, &DisplayNotebookPanel::OnKeyUp, this);
    Bind(wxEVT_LEAVE_WINDOW, &DisplayNotebookPanel::OnLeaveWindow, this);

    ////////////////////////////

    show_crosshair               = false;
    single_image                 = false;
    use_7bit_greys               = false;
    show_selection_distances     = false;
    resolution_instead_of_radius = false;

    blue_selection_square_location = -1;

    template_matching_marker_x_pos  = -1.0;
    template_matching_marker_y_pos  = -1.0;
    template_matching_marker_radius = -1.0;

    use_unscaled_image_for_popup = false;

    if ( (parent_display_panel->style_flags & FIRST_LOCATION_ONLY) == FIRST_LOCATION_ONLY || (parent_display_panel->style_flags & START_WITH_NO_LABEL) == START_WITH_NO_LABEL ) {
        show_label = false;
    }
    else
        show_label = true;

    if ( (parent_display_panel->style_flags & START_WITH_INVERTED_CONTRAST) == START_WITH_INVERTED_CONTRAST ) {
        invert_contrast = true;
    }
    else
        invert_contrast = false;

    if ( (parent_display_panel->style_flags & START_WITH_FOURIER_SCALING) == START_WITH_FOURIER_SCALING ) {
        use_fourier_scaling = true;
    }
    else
        use_fourier_scaling = false;

    selected_point_size = 3;
    should_refresh      = false;
    use_fft             = false;

    pixel_size = 1;

    selected_distance = 0;

    single_image_x = 0.;
    single_image_y = 0.;

    number_of_selections = 0;

    suspend_overlays = false;

    //	label_mode = Menu_Label_Nothing;
    SetBackgroundColour(*wxBLACK);
    current_location     = 1;
    desired_scale_factor = 1;
    actual_scale_factor  = 1;
    low_grey_value       = 0;
    high_grey_value      = 0;
    global_low_grey      = 0;
    global_high_grey     = 0;
    manual_low_grey      = 0.;
    manual_high_grey     = 0.;

    if ( (parent_display_panel->style_flags & START_WITH_AUTO_CONTRAST) == START_WITH_AUTO_CONTRAST )
        grey_values_decided_by = AUTO_GREYS;
    else
        grey_values_decided_by = LOCAL_GREYS;

    panel_image_has_correct_greys = false;
    panel_image_has_correct_scale = false;

    location_on_last_draw    = 0;
    images_in_x_on_last_draw = 0;
    images_in_y_on_last_draw = 0;

    integrate_box_x_pos = -1;
    integrate_box_y_pos = -1;
    integrated_value    = -1;

    picking_mode             = IMAGES_PICK; // Do this by default, change when creating the frame.
    have_txt_filename        = false;
    txt_is_saved             = false;
    selected_filament_number = 1;

    int window_x_size;
    int window_y_size;

    // create panel..

    this->GetClientSize(&window_x_size, &window_y_size);

    // size the draw bitmap to the client size..

    this->panel_bitmap.Create(window_x_size, window_y_size);

    current_location = 1;

    images_in_current_view = 0;

    images_in_x    = 0;
    images_in_y    = 0;
    current_x_size = window_x_size;
    current_y_size = window_y_size;

    SetCursor(wxCursor(wxCURSOR_CROSS));
}

void DisplayNotebookPanel::UpdateImageStatusInfo(int x_pos, int y_pos) {
    long   max_x = images_in_x * current_x_size;
    long   max_y = images_in_y * current_y_size;
    double current_resolution;

    if ( single_image == true ) {
        int current_x_pos = single_image_x + (x_pos / actual_scale_factor); // - 1;
        int current_y_pos = single_image_y + (y_pos / actual_scale_factor); // - 1;

        current_y_pos = ReturnImageYSize( ) - 1 - current_y_pos;

        int current_image = current_location;

        int current_radius = sqrtf(pow(current_x_pos - (ReturnImageXSize( ) / 2), 2) + pow(current_y_pos - (ReturnImageYSize( ) / 2), 2));

        if ( ReturnImageXSize( ) > ReturnImageYSize( ) )
            current_resolution = (pixel_size * ReturnImageXSize( )) / current_radius;
        else
            current_resolution = (pixel_size * ReturnImageXSize( )) / current_radius;

        if ( current_x_pos >= 0 && current_x_pos < ReturnImageXSize( ) - 1 && current_y_pos >= 0 && current_y_pos < ReturnImageYSize( ) - 1 ) {
            float raw_pixel_value = image_memory_buffer[0].ReturnRealPixelFromPhysicalCoord(current_x_pos, current_y_pos, 0);

            wxString StatusText;
            StatusText = wxT("Image: ") + wxString::Format(wxT("%i"), current_image) + wxT("  - X=") + wxString::Format(wxT("%i"), current_x_pos) + wxT(", Y=") + wxString::Format(wxT("%i"), current_y_pos);

            if ( resolution_instead_of_radius == true )
                StatusText += wxT(", Res=") + wxString::Format(wxT("%.2f"), current_resolution) + wxT("Å");
            else
                StatusText += wxT(", Rad=") + wxString::Format(wxT("%i"), current_radius);

            StatusText += wxT(", Value=") + wxString::Format(wxT("%f"), raw_pixel_value);

            //if (selected_distance != 0 && show_selection_distances == true) StatusText += wxT(", Dist=") + wxString::Format(wxT("%f"), selected_distance);
            //if (picking_mode == INTEGRATE_PICK && integrate_box_x_pos != -1 && integrate_box_y_pos != -1) StatusText += wxT(", Integrated Value =") + wxString::Format(wxT("%f"), integrated_value);
            parent_display_panel->StatusText->SetLabel(StatusText);
        }
        else {
            wxString StatusText = wxT("");

            //if (selected_distance != 0 && show_selection_distances == true) StatusText += wxT("Dist=") + wxString::Format(wxT("%f"), selected_distance);
            //	if (picking_mode == INTEGRATE_PICK && integrate_box_x_pos != -1 && integrate_box_y_pos != -1) StatusText += wxT("Integrated Value =") + wxString::Format(wxT("%f"), integrated_value);
            parent_display_panel->StatusText->SetLabel(StatusText);
        }
    }
    else {
        if ( x_pos < max_x && y_pos < max_y && x_pos >= 0 && y_pos >= 0 ) {
            // turn the position into "co-ordinates"

            int image_x_coord = x_pos / current_x_size;
            int image_y_coord = y_pos / current_y_size;
            int current_image = (images_in_x * (image_y_coord)) + image_x_coord + current_location;
            int current_x_pos = (x_pos - (current_x_size * image_x_coord)) / actual_scale_factor;
            int current_y_pos = (y_pos - (current_y_size * image_y_coord)) / actual_scale_factor;

            current_y_pos = ReturnImageYSize( ) - 1 - current_y_pos;

            int current_radius = sqrt(pow(current_x_pos - (ReturnImageXSize( ) / 2), 2) + pow(current_y_pos - (ReturnImageYSize( ) / 2), 2));

            if ( ReturnImageXSize( ) > ReturnImageYSize( ) )
                current_resolution = (pixel_size * ReturnImageYSize( )) / current_radius;
            else
                current_resolution = (pixel_size * ReturnImageYSize( )) / current_radius;

            if ( current_image > 0 && current_image <= images_in_current_view + current_image && current_image <= included_image_numbers.GetCount( ) ) {
                float raw_pixel_value = image_memory_buffer[(images_in_x * (image_y_coord)) + image_x_coord].ReturnRealPixelFromPhysicalCoord(current_x_pos, current_y_pos, 0);

                wxString StatusText;
                StatusText = wxT("Image: ") + wxString::Format(wxT("%i"), current_image) + wxT("  - X=") + wxString::Format(wxT("%i"), current_x_pos) + wxT(", Y=") + wxString::Format(wxT("%i"), current_y_pos);

                if ( resolution_instead_of_radius == true )
                    StatusText += wxT(", Res=") + wxString::Format(wxT("%.2f"), current_resolution) + wxT("Å");
                else
                    StatusText += wxT(", Rad=") + wxString::Format(wxT("%i"), current_radius);

                StatusText += wxT(", Value=") + wxString::Format(wxT("%f"), raw_pixel_value);

                //	if (selected_distance != 0 && show_selection_distances == true) StatusText += wxT(", Dist=") + wxString::Format(wxT("%f"), selected_distance);
                //	if (picking_mode == INTEGRATE_PICK && integrate_box_x_pos != -1 && integrate_box_y_pos != -1) StatusText += wxT(", Integrated Value =") + wxString::Format(wxT("%f"), integrated_value);
                parent_display_panel->StatusText->SetLabel(StatusText);
            }
            else {
                wxString StatusText = wxT("");

                //	if (selected_distance != 0 && show_selection_distances == true) StatusText += wxT("Dist=") + wxString::Format(wxT("%f"), selected_distance);
                //	if (picking_mode == INTEGRATE_PICK && integrate_box_x_pos != -1 && integrate_box_y_pos != -1) StatusText += wxT("Integrated Value =") + wxString::Format(wxT("%f"), integrated_value);
                parent_display_panel->StatusText->SetLabel(StatusText);
            }
        }
        else {
            wxString StatusText = wxT("");

            // 		if (selected_distance != 0 && show_selection_distances == true) StatusText += wxT("Dist=") + wxString::Format(wxT("%f"), selected_distance);
            // 		if (picking_mode == INTEGRATE_PICK && integrate_box_x_pos != -1 && integrate_box_y_pos != -1) StatusText += wxT(" Integrated Value =") + wxString::Format(wxT("%f"), integrated_value);
            parent_display_panel->StatusText->SetLabel(StatusText);
        }
    }
}

void DisplayNotebookPanel::OnKeyDown(wxKeyEvent& event) {
    int current_keycode = event.GetKeyCode( );

    wxCommandEvent null_event;

    if ( current_keycode == WXK_LEFT ) {
        if ( (parent_display_panel->style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
        }
        else {
            int current_page = parent_display_panel->my_notebook->GetSelection( );
            current_page--;
            if ( current_page < 0 )
                current_page = parent_display_panel->my_notebook->GetPageCount( ) - 1;
            parent_display_panel->my_notebook->SetSelection(current_page);
        }
    }
    else if ( current_keycode == WXK_RIGHT ) {
        if ( (parent_display_panel->style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
        }
        else {
            int current_page = parent_display_panel->my_notebook->GetSelection( );
            current_page++;
            if ( current_page > parent_display_panel->my_notebook->GetPageCount( ) - 1 )
                current_page = 0;
            parent_display_panel->my_notebook->SetSelection(current_page);
        }
    }
    else if ( current_keycode == WXK_UP ) {
        parent_display_panel->OnNext(null_event);
    }
    else if ( current_keycode == WXK_DOWN ) {
        parent_display_panel->OnPrevious(null_event);
    }
    else if ( current_keycode == 44 ) {
        parent_display_panel->OnPrevious(null_event);
    }
    else if ( current_keycode == 46 ) {
        parent_display_panel->OnNext(null_event);
    }
    else
        event.Skip( );
}

void DisplayNotebookPanel::OnKeyUp(wxKeyEvent& event) {
    /*
	int current_keycode = event.GetKeyCode();

	displayPanel *current_panel = (displayPanel*) notebook->GetCurrentPage();

	wxCommandEvent null_event;

	if (current_keycode == WXK_SPACE && suspend_overlays == true)
	{
		suspend_overlays = false;

		current_panel->Refresh();
		current_panel->Update();
	}

	if (current_keycode == WXK_CONTROL && drawing_selection_square == true)
	{
		drawing_selection_square = false;
		current_panel->Refresh();
		current_panel->Update();
	}

*/
    event.Skip( );
}

bool DisplayNotebookPanel::SetGlobalGreys(void) {
    // so now we have to work out the global grey value..
    // best use a progress dialog..

    Image buffer_image;

    if ( CheckFileStillValid( ) == true ) {
        float global_min;
        float global_max;
        float current_min;
        float current_max;

        LoadIntoImage(&buffer_image, 0);
        buffer_image.GetMinMax(global_min, global_max);

        bool should_continue = true;

        wxProgressDialog my_progress(wxT("Calculating..."), wxT("Performing Global Survey"), included_image_numbers.GetCount( ), parent_display_panel, wxPD_AUTO_HIDE | wxPD_APP_MODAL | wxPD_CAN_ABORT);

        for ( long counter = 0; counter < included_image_numbers.GetCount( ); counter++ ) {
            LoadIntoImage(&buffer_image, counter);
            buffer_image.GetMinMax(current_min, current_max);

            if ( current_max > global_max )
                global_max = current_max;
            if ( current_min < global_min )
                global_min = current_min;

            should_continue = my_progress.Update(counter);

            if ( should_continue == false ) {
                continue;
            }
        }

        if ( should_continue == true ) {
            global_low_grey  = global_min;
            global_high_grey = global_max;
            return true;
        }
        else
            return false;

        return false;
    }
    else
        return false;
}

DisplayNotebookPanel::~DisplayNotebookPanel( ) {
    if ( image_memory_buffer != NULL )
        delete[] image_memory_buffer;
    if ( scaled_image_memory_buffer != NULL )
        delete[] scaled_image_memory_buffer;
    if ( input_is_a_file == false && do_i_have_image_ownership == true && image_to_display != NULL )
        delete image_to_display;
}

void DisplayNotebookPanel::OnSize(wxSizeEvent& event) {

    parent_display_panel->Refresh( );
    event.Skip( );
}

void DisplayNotebookPanel::OnRightDown(wxMouseEvent& event) {

    long x_pos;
    long y_pos;

    event.GetPosition(&x_pos, &y_pos);

    if ( (parent_display_panel->style_flags & NO_POPUP) == NO_POPUP ) {
        // work out which is the selected image, and put it into the event..

        int image_x_coord;
        int image_y_coord;
        int current_x_pos;
        int current_y_pos;
        int current_image;

        if ( single_image == true ) {
            current_x_pos = single_image_x + (x_pos / actual_scale_factor); //- 1;
            current_y_pos = single_image_y + (y_pos / actual_scale_factor); // - 1;
            current_image = current_location;
        }
        else {
            image_x_coord = x_pos / current_x_size;
            image_y_coord = y_pos / current_y_size;
            current_image = (images_in_x * (image_y_coord)) + image_x_coord + current_location;
            current_x_pos = (x_pos - (current_x_size * image_x_coord)) / actual_scale_factor;
            current_y_pos = (y_pos - (current_y_size * image_y_coord)) / actual_scale_factor;
        }

        long max_x = images_in_x * current_x_size;
        long max_y = images_in_y * current_y_size;

        if ( x_pos < max_x && y_pos < max_y ) {
            event.SetId(current_image);
        }
        else
            event.SetId(-1);

        event.ResumePropagation(2); // go up to the parent parent panel..
        event.Skip( );
    }
    else if ( parent_display_panel->popup_exists == false ) {
        int client_x = int(x_pos);
        int client_y = int(y_pos);

        DisplayNotebookPanel* current_panel = parent_display_panel->ReturnCurrentPanel( );

        current_panel->ClientToScreen(&client_x, &client_y);

        // At the time of writing, when the popupwindow goes off the size of screen
        // it's draw direction is reveresed.. For this reason i've included this
        // rather dodgy get around, of just adding the box size when the box goes
        // off the edge.. hopefully it will hold up.

        int screen_x_size = wxSystemSettings::GetMetric(wxSYS_SCREEN_X);
        int screen_y_size = wxSystemSettings::GetMetric(wxSYS_SCREEN_Y);

        if ( client_x + 256 > screen_x_size )
            client_x += 256;
        if ( client_y + 256 > screen_y_size )
            client_y += 256;

        SetCursor(wxCursor(wxCURSOR_BLANK));

        current_panel->CaptureMouse( );
        parent_display_panel->popup = new DisplayPopup(parent_display_panel);
        parent_display_panel->popup->SetClientSize(256, 256);
        parent_display_panel->popup->Position(wxPoint(client_x - 128, client_y - 128), wxSize(0, 0));
        parent_display_panel->popup->x_pos = x_pos - 128;
        parent_display_panel->popup->y_pos = y_pos - 128;
        parent_display_panel->popup->SetCursor(wxCursor(wxCURSOR_BLANK));
        parent_display_panel->popup->Show( );
        parent_display_panel->popup->Refresh( );
        parent_display_panel->popup->Update( );

        parent_display_panel->popup_exists = true;
    }

    event.Skip( );
}

void DisplayNotebookPanel::SetImageSelected(long wanted_image, bool refresh) {
    MyDebugAssertTrue((parent_display_panel->style_flags & CAN_SELECT_IMAGES) == CAN_SELECT_IMAGES || parent_display_panel->ReturnCurrentPanel( )->picking_mode == IMAGES_PICK, "Trying to select images, but selection style flag not set");
    MyDebugAssertTrue(wanted_image > 0 && wanted_image <= ReturnNumberofImages( ), "Trying to select an image that doesn't exist (%li)", wanted_image);

    if ( image_is_selected[wanted_image] == false )
        number_of_selections++;
    image_is_selected[wanted_image] = true;

    if ( refresh == true ) {
        Refresh( );
        Update( );
    }
}

void DisplayNotebookPanel::SetImageNotSelected(long wanted_image, bool refresh) {
    MyDebugAssertTrue((parent_display_panel->style_flags & CAN_SELECT_IMAGES) == CAN_SELECT_IMAGES || parent_display_panel->ReturnCurrentPanel( )->picking_mode == IMAGES_PICK, "Trying to select images, but selection style flag not set");
    MyDebugAssertTrue(wanted_image > 0 && wanted_image <= ReturnNumberofImages( ), "Trying to select an image that doesn't exist (%li)", wanted_image);

    if ( image_is_selected[wanted_image] == true )
        number_of_selections--;
    image_is_selected[wanted_image] = false;

    if ( refresh == true ) {
        Refresh( );
        Update( );
    }
}

void DisplayNotebookPanel::ToggleImageSelected(long wanted_image, bool refresh) {
    MyDebugAssertTrue((parent_display_panel->style_flags & CAN_SELECT_IMAGES) == CAN_SELECT_IMAGES || parent_display_panel->ReturnCurrentPanel( )->picking_mode == IMAGES_PICK, "Trying to select images, but selection style flag not set");
    MyDebugAssertTrue(wanted_image > 0 && wanted_image <= ReturnNumberofImages( ), "Trying to select an image that doesn't exist (%li)", wanted_image);

    if ( image_is_selected[wanted_image] == true ) {
        image_is_selected[wanted_image] = false;
        number_of_selections--;
    }
    else {
        image_is_selected[wanted_image] = true;
        number_of_selections++;
    }

    if ( refresh == true ) {
        Refresh( );
        Update( );
    }
}

void DisplayNotebookPanel::ClearSelection(bool refresh) {
    MyDebugAssertTrue((parent_display_panel->style_flags & CAN_SELECT_IMAGES) == CAN_SELECT_IMAGES || (parent_display_panel->ReturnCurrentPanel( )->picking_mode == IMAGES_PICK), "Trying to clear selection, but selection style flag not set");

    for ( long mycounter = 0; mycounter < ReturnNumberofImages( ) + 1; mycounter++ ) {
        image_is_selected[mycounter] = false;
    }

    number_of_selections = 0;

    if ( refresh == true ) {
        Refresh( );
        Update( );
    }
}

void DisplayNotebookPanel::OnLeftDown(wxMouseEvent& event) {
    if ( parent_display_panel->popup_exists == true ) {
        ReleaseMouse( );
        SetCursor(wxCursor(wxCURSOR_CROSS));
        parent_display_panel->popup->Destroy( );
        parent_display_panel->popup_exists = false;
    }

    long x_pos;
    long y_pos;

    long max_x = images_in_x * current_x_size;
    long max_y = images_in_y * current_y_size;

    event.GetPosition(&x_pos, &y_pos);

    // convert the coords into image info..

    int image_x_coord;
    int image_y_coord;
    int current_x_pos;
    int current_y_pos;
    int current_image;

    if ( single_image == true ) {
        current_x_pos = single_image_x + (x_pos / actual_scale_factor); //- 1;
        current_y_pos = single_image_y + (y_pos / actual_scale_factor); // - 1;
        current_image = current_location;
    }
    else {
        image_x_coord = x_pos / current_x_size;
        image_y_coord = y_pos / current_y_size;
        current_image = (images_in_x * (image_y_coord)) + image_x_coord + current_location;
        current_x_pos = (x_pos - (current_x_size * image_x_coord)) / actual_scale_factor;
        current_y_pos = (y_pos - (current_y_size * image_y_coord)) / actual_scale_factor;
    }

    if ( (parent_display_panel->style_flags & SKIP_LEFTCLICK_TO_PARENT) == SKIP_LEFTCLICK_TO_PARENT ) {
        if ( x_pos < max_x && y_pos < max_y ) {
            event.SetId(current_image);
        }
        else
            event.SetId(-1);

        event.ResumePropagation(2); // go up to the parent parent panel..
        event.Skip( );
        return;
    }

    // check if the CTRL key is down, if so we want to make a selection box...

    /*if ( event.ControlDown( ) == true && picking_mode == COORDS_PICK ) // we want to draw a selection box..
    {
        drawing_selection_square = true;

        selection_square_start_x   = x_pos;
        selection_square_start_y   = y_pos;
        selection_square_current_x = x_pos;
        selection_square_current_y = y_pos;
        selection_square_image     = current_image;
    }
    */
    //else // we are doing a coord or image select

    // work out what image (if any) us under the mouse and select / deselect it

    if ( single_image == true ) {
        // perform the relevant action..

        if ( current_x_pos < parent_display_panel->x_size && current_x_pos >= 0 && current_y_pos < parent_display_panel->y_size && current_y_pos >= 0 ) {
            if ( current_image <= parent_display_panel->number_of_frames ) {
                if ( picking_mode == IMAGES_PICK ) {
                    if ( image_is_selected[current_image] == true ) {
                        image_is_selected[current_image] = false;
                        number_of_selections--;
                    }
                    else {
                        image_is_selected[current_image] = true;
                        number_of_selections++;
                    }

                    parent_display_panel->SetTabNameUnsaved( );
                }
                else if ( picking_mode == COORDS_PICK ) {
                    coord_tracker->ToggleCoord(current_image, current_x_pos, current_y_pos);
                    parent_display_panel->SetTabNameUnsaved( );
                }
                Refresh( );
                Update( );
            }
        }
    }
    else if ( x_pos < max_x && y_pos < max_y ) {
        // perform the relevant action..
        if ( current_image <= parent_display_panel->number_of_frames ) {
            if ( picking_mode == IMAGES_PICK ) {
                ToggleImageSelected(current_image, false);
                parent_display_panel->SetTabNameUnsaved( );
            }

            // We must be in coords mode
            else {
                coord_tracker->ToggleCoord(current_image, current_x_pos, current_y_pos);
                parent_display_panel->SetTabNameUnsaved( );
            }
        }
        Refresh( );
        Update( );
    }

    // Refresh StatusInfo for completeness

    UpdateImageStatusInfo(x_pos, y_pos);

    event.Skip( );
}

void DisplayNotebookPanel::OnRightUp(wxMouseEvent& event) {
    if ( parent_display_panel->popup_exists == true ) {

        DisplayNotebookPanel* current_panel = parent_display_panel->ReturnCurrentPanel( );
        current_panel->ReleaseMouse( );
        SetCursor(wxCursor(wxCURSOR_CROSS));
        parent_display_panel->popup->Destroy( );
        parent_display_panel->popup_exists = false;
    }

    event.Skip( );
}

void DisplayNotebookPanel::OnLeaveWindow(wxMouseEvent& event) {
    parent_display_panel->StatusText->SetLabel("");
    event.Skip( );
}

void DisplayNotebookPanel::OnMotion(wxMouseEvent& event) {
    long x_pos;
    long y_pos;

    event.GetPosition(&x_pos, &y_pos);

    DisplayNotebookPanel* current_panel = parent_display_panel->ReturnCurrentPanel( );

    int client_x = int(x_pos);
    int client_y = int(y_pos);
    current_panel->ClientToScreen(&client_x, &client_y);

    // if left button is down, and we are drawing a selections square, update to coords..
    /*
	 if (event.m_leftDown == true && drawing_selection_square == true)
	 {
		 // in coordinate selection mode - do not let the square leave the initial image, as otherwise, it becomes too complicated..

		 long max_x = images_in_x * current_x_size;
		 long max_y = images_in_y * current_y_size;

		 // convert the coords into image info..

		 int image_x_coord;
		 int image_y_coord;
		 int current_x_pos;
		 int current_y_pos;
		 int current_image;

		 int max_image_x_pos;
		 int max_image_y_pos;

		 int min_image_x_pos;
		 int min_image_y_pos;

		 if (single_image == true)
		 {
			 current_x_pos = single_image_x + (x_pos / actual_scale_factor) - 1;
			 current_y_pos = single_image_y + (y_pos / actual_scale_factor) - 1;
			 current_image = current_location;

			 max_image_x_pos = (first_header.x_size - single_image_x) * actual_scale_factor;
			 max_image_y_pos = (first_header.y_size - single_image_y) * actual_scale_factor;

			 min_image_x_pos = 0;
			 min_image_y_pos = 0;
		 }
		 else
		 {
			 image_x_coord = selection_square_start_x /  current_x_size;
			 image_y_coord = selection_square_start_y /  current_y_size;
			// current_image = (images_in_x * (image_y_coord)) + image_x_coord + current_location;
			 //current_x_pos = (x_pos - (current_x_size * image_x_coord)) / actual_scale_factor;
			 //current_y_pos = (y_pos - (current_y_size * image_y_coord)) / actual_scale_factor;

			 min_image_x_pos = (image_x_coord) * current_x_size;
			 min_image_y_pos = (image_y_coord) * current_y_size;

			 max_image_x_pos = min_image_x_pos + current_x_size;
			 max_image_y_pos = min_image_y_pos + current_y_size;


		 }

		 selection_square_current_x = x_pos;
		 selection_square_current_y = y_pos;

		 if (selection_square_current_x > max_image_x_pos) selection_square_current_x = max_image_x_pos;
		 if (selection_square_current_y > max_image_y_pos) selection_square_current_y = max_image_y_pos;

		 if (selection_square_current_x < min_image_x_pos) selection_square_current_x = min_image_x_pos;
		 if (selection_square_current_y < min_image_y_pos) selection_square_current_y = min_image_y_pos;

		 Refresh();
		 Update();
	 }
	 else // if the right button is down and the popup exists move it..*/
    if ( event.m_rightDown == true && parent_display_panel->popup_exists == true ) {
        // At the time of writing, when the popupwindow goes off the size of screen
        // it's draw direction is reveresed.. For this reason i've included this
        // rather dodgy get around, of just adding the box size when the box goes
        // off the edge.. hopefully it will hold up.

        int screen_x_size = wxSystemSettings::GetMetric(wxSYS_SCREEN_X);
        int screen_y_size = wxSystemSettings::GetMetric(wxSYS_SCREEN_Y);

        if ( client_x + 256 > screen_x_size )
            client_x += 256;
        if ( client_y + 256 > screen_y_size )
            client_y += 256;

        parent_display_panel->popup->Position(wxPoint(client_x - 128, client_y - 128), wxSize(0, 0));
        parent_display_panel->popup->x_pos = x_pos - 128;
        parent_display_panel->popup->y_pos = y_pos - 128;
        parent_display_panel->popup->Show( );
        parent_display_panel->popup->Refresh( );
        parent_display_panel->popup->Update( );
        parent_display_panel->Update( );
    }
    /*	 else
     if (event.m_middleDown == true && current_panel->single_image == true)
	 {
		 if (old_mouse_x == -9999 || old_mouse_y == -9999)
		 {
			 old_mouse_x = x_pos;
			 old_mouse_y = y_pos;
		 }
		 else
		 {
			 current_panel->single_image_x += (old_mouse_x - x_pos) / actual_scale_factor;
			 current_panel->single_image_y += (old_mouse_y - y_pos) / actual_scale_factor;

			 if (current_panel->single_image_x < 0) current_panel->single_image_x = 0;
			 else
			 if (current_panel->single_image_x > current_panel->first_header.x_size - 1) current_panel->single_image_x = current_panel->first_header.x_size - 1;

			 if (current_panel->single_image_y < 0) current_panel->single_image_y = 0;
			 else
			 if (current_panel->single_image_y > current_panel->first_header.y_size - 1) current_panel->single_image_y = current_panel->first_header.y_size - 1;

			 old_mouse_x = x_pos;
			 old_mouse_y = y_pos;



			 current_panel->DrawPanel();
		 }
	 }
     else


	 // write the current position to the status bar..
	*/

    UpdateImageStatusInfo(x_pos, y_pos);
    event.Skip( );
}

bool DisplayNotebookPanel::CheckFileStillValid( ) {

    if ( input_is_a_file == true ) {
        ImageFile buffer_file;
        buffer_file.OpenFile(filename.ToStdString( ), false);

        if ( buffer_file.ReturnNumberOfSlices( ) == my_file.ReturnNumberOfSlices( ) && buffer_file.ReturnXSize( ) == my_file.ReturnXSize( ) && buffer_file.ReturnYSize( ) == my_file.ReturnYSize( ) ) {
            return true;
        }
        else {
            wxMessageBox(wxT("The current file is no longer accessible, or has changed number/size/type!!"), wxT("Error!!"), wxOK | wxICON_INFORMATION, this);

            if ( (parent_display_panel->style_flags & NO_NOTEBOOK) == NO_NOTEBOOK ) {
                // delete panel?
            }
            else {
                parent_display_panel->my_notebook->DeletePage(parent_display_panel->my_notebook->GetSelection( ));
                parent_display_panel->UpdateToolbar( );
            }

            return false;
        }
    }
    else
        return true;
}

void DisplayNotebookPanel::ReDrawPanel(void) {

    int window_x_size;
    int window_y_size;

    long scaled_x_size;
    long scaled_y_size;

    long counter;

    long red_counter;
    long blue_counter;
    long green_counter;
    long image_counter;
    long current_value;

    long cut_x_size;
    long cut_y_size;

    float current_low_grey_value  = low_grey_value;
    float current_high_grey_value = low_grey_value;

    double image_total;
    double image_total_squared;
    double image_average_density;
    double image_variance;
    double image_stdev;

    long  number_of_pixels;
    float range;

    int  i, j, k;
    long address = 0;

    wxImage   FrameImage;
    wxBitmap* DrawBitmap;

    wxPen red_dashed_pen(*wxRED, long(myround(desired_scale_factor)), wxLONG_DASH);

    wxString LabelText;

    // get the device context..

    wxImage ScaledImage;

    // Get ready for text writing..

    wxMemoryDC memDC;
    wxMemoryDC dc;

    // we need to know how many images we are going to read in.. first get the current size of the window..

    GetClientSize(&window_x_size, &window_y_size);

    if ( window_x_size > 10 && window_y_size > 10 ) {
        //wxBeginBusyCursor();
        //wxYield();

        // size the draw bitmap to the client size..

        if ( window_x_size != panel_bitmap.GetWidth( ) || window_y_size != panel_bitmap.GetHeight( ) ) {
            Refresh( );
        }

        if ( included_image_numbers.GetCount( ) - current_location < images_in_current_view )
            Refresh( );

        panel_bitmap.Create(window_x_size, window_y_size);

        dc.SelectObject(panel_bitmap);
        dc.SetBackground(*wxBLACK_BRUSH);
        dc.Clear( );
        dc.SelectObject(wxNullBitmap);

        //Update();

        // how big are the images going to be, and how many will fit..

        scaled_x_size = long(myround(ReturnImageXSize( ) * desired_scale_factor));
        scaled_y_size = long(myround(ReturnImageYSize( ) * desired_scale_factor));

        if ( single_image == true ) {
            actual_scale_factor = desired_scale_factor;
        }
        else // single image mode is false...
        {
            images_in_x = window_x_size / scaled_x_size;
            images_in_y = window_y_size / scaled_y_size;

            if ( images_in_x < 1 || images_in_y < 1 ) {
                // none fit, so we should scale it so that 1 image fits...
                // determine if the X or Y is the limiting factor..

                if ( double(window_x_size) / double(scaled_x_size) >= double(window_y_size) / double(scaled_y_size) ) {
                    // this means that the limiting dimension is the y.. so scale appropriately..
                    actual_scale_factor = double(window_y_size) / double(ReturnImageYSize( ));
                }
                else
                    actual_scale_factor = double(window_x_size) / double(ReturnImageXSize( ));

                scaled_x_size = long(myround(ReturnImageXSize( ) * actual_scale_factor));
                scaled_y_size = long(myround(ReturnImageYSize( ) * actual_scale_factor));

                images_in_x = 1;
                images_in_y = 1;
            }
            else
                actual_scale_factor = desired_scale_factor;
        }

        current_x_size = scaled_x_size;
        current_y_size = scaled_y_size;

        if ( single_image == true ) {
            images_in_current_view = 1;
            images_in_x            = 1;
            images_in_y            = 1;
        }
        else
            images_in_current_view = images_in_x * images_in_y;

        if ( current_location != location_on_last_draw || images_in_x != images_in_x_on_last_draw || images_in_y != images_in_y_on_last_draw ) {
            //dc.Clear();
            if ( CheckFileStillValid( ) == true ) {

                if ( number_allocated_for_buffer < images_in_current_view ) {
                    if ( image_memory_buffer != NULL )
                        delete[] image_memory_buffer;
                    image_memory_buffer = new Image[images_in_current_view];

                    if ( scaled_image_memory_buffer != NULL )
                        delete[] scaled_image_memory_buffer;
                    scaled_image_memory_buffer = new Image[images_in_current_view];

                    number_allocated_for_buffer = images_in_current_view;
                }

                for ( image_counter = 0; image_counter < images_in_current_view; image_counter++ ) {
                    if ( current_location + image_counter <= included_image_numbers.GetCount( ) ) {
                        SetImageInMemoryBuffer(image_counter, current_location + image_counter - 1);

                        if ( use_fft == true ) {
                            Image buffer_image;
                            buffer_image.CopyFrom(&image_memory_buffer[image_counter]);
                            buffer_image.ForwardFFT(false);
                            buffer_image.DivideByConstant(sqrt(buffer_image.number_of_real_space_pixels));
                            buffer_image.ComputeAmplitudeSpectrumFull2D(&image_memory_buffer[image_counter]);
                            image_memory_buffer[image_counter].ZeroCentralPixel( );
                        }
                    }
                }
            }
            else
                return;

            location_on_last_draw    = current_location;
            images_in_x_on_last_draw = images_in_x;
            images_in_y_on_last_draw = images_in_y;
        }

        if ( should_refresh == true ) {
            // check none of the file attributes have changed..

            if ( CheckFileStillValid( ) == true ) {
                // we should be safe to reload..

                //dc.Clear();

                if ( number_allocated_for_buffer < images_in_current_view ) {
                    if ( image_memory_buffer != NULL )
                        delete[] image_memory_buffer;
                    image_memory_buffer         = new Image[images_in_current_view];
                    number_allocated_for_buffer = images_in_current_view;
                }

                for ( image_counter = 0; image_counter < images_in_current_view; image_counter++ ) {
                    if ( this->current_location + image_counter <= included_image_numbers.GetCount( ) ) {

                        SetImageInMemoryBuffer(image_counter, current_location + image_counter - 1);

                        if ( use_fft == true ) {
                            Image buffer_image;
                            buffer_image.CopyFrom(&image_memory_buffer[image_counter]);
                            buffer_image.ForwardFFT(false);
                            buffer_image.DivideByConstant(sqrt(buffer_image.number_of_real_space_pixels));
                            buffer_image.ComputeAmplitudeSpectrumFull2D(&image_memory_buffer[image_counter]);
                            image_memory_buffer[image_counter].ZeroCentralPixel( );
                        }
                    }
                }

                location_on_last_draw    = current_location;
                images_in_x_on_last_draw = images_in_x;
                images_in_y_on_last_draw = images_in_y;

                panel_image_has_correct_greys = false;
            }
            else
                return;

            should_refresh = false;
        }

        image_counter = 0;

        for ( int big_y = 0; big_y < images_in_y; big_y++ ) {
            for ( int big_x = 0; big_x < images_in_x; big_x++ ) {

                if ( this->current_location + image_counter <= included_image_numbers.GetCount( ) ) {

                    if ( panel_image_has_correct_greys == false || panel_image_has_correct_scale == false || single_image == false ) {
                        // drawing speed is critical so we'll do it by directly writing the data,
                        // this gives us a pointer to the image data..

                        if ( use_fourier_scaling == true && (scaled_x_size != ReturnImageXSize( ) || scaled_y_size != ReturnImageYSize( )) ) {
                            scaled_image_memory_buffer[image_counter].CopyFrom(&image_memory_buffer[image_counter]);
                            scaled_image_memory_buffer[image_counter].ForwardFFT( );
                            scaled_image_memory_buffer[image_counter].Resize(scaled_x_size, scaled_y_size, 1);
                            scaled_image_memory_buffer[image_counter].BackwardFFT( );
                        }

                        if ( use_fourier_scaling == true && (scaled_x_size != ReturnImageXSize( ) || scaled_y_size != ReturnImageYSize( )) ) {
                            panel_image->Resize(wxSize(scaled_x_size, scaled_y_size), wxPoint(0, 0));
                        }
                        else {
                            panel_image->Resize(wxSize(ReturnImageXSize( ), ReturnImageYSize( )), wxPoint(0, 0));
                        }

                        unsigned char* image_data = panel_image->GetData( );

                        // are we locally scaling?

                        if ( this->grey_values_decided_by == LOCAL_GREYS ) {
                            if ( use_fourier_scaling == true && (scaled_x_size != ReturnImageXSize( ) || scaled_y_size != ReturnImageYSize( )) ) {
                                scaled_image_memory_buffer[image_counter].GetMinMax(current_low_grey_value, current_high_grey_value);
                            }
                            else
                                image_memory_buffer[image_counter].GetMinMax(current_low_grey_value, current_high_grey_value);
                        }
                        else if ( this->grey_values_decided_by == AUTO_GREYS ) // auto contrast
                        {
                            // work out mean, and stdev..

                            image_total         = 0;
                            image_total_squared = 0;
                            number_of_pixels    = 0;

                            address = 0;

                            if ( use_fourier_scaling == true && (scaled_x_size != ReturnImageXSize( ) || scaled_y_size != ReturnImageYSize( )) ) {
                                for ( j = 0; j < scaled_image_memory_buffer[image_counter].logical_y_dimension; j++ ) {
                                    for ( i = 0; i < scaled_image_memory_buffer[image_counter].logical_x_dimension; i++ ) {
                                        if ( scaled_image_memory_buffer[image_counter].real_values[address] != 0. ) {
                                            image_total += scaled_image_memory_buffer[image_counter].real_values[address];
                                            image_total_squared += pow(scaled_image_memory_buffer[image_counter].real_values[address], 2);
                                            number_of_pixels++;
                                        }

                                        address++;
                                    }

                                    address += scaled_image_memory_buffer[image_counter].padding_jump_value;
                                }
                            }
                            else {
                                for ( j = 0; j < image_memory_buffer[image_counter].logical_y_dimension; j++ ) {
                                    for ( i = 0; i < image_memory_buffer[image_counter].logical_x_dimension; i++ ) {
                                        if ( image_memory_buffer[image_counter].real_values[address] != 0. ) {
                                            image_total += image_memory_buffer[image_counter].real_values[address];
                                            image_total_squared += pow(image_memory_buffer[image_counter].real_values[address], 2);
                                            number_of_pixels++;
                                        }
                                        address++;
                                    }

                                    address += image_memory_buffer[image_counter].padding_jump_value;
                                }
                            }

                            if ( image_total_squared == 0 ) {
                                current_low_grey_value  = 0;
                                current_high_grey_value = 0;
                            }
                            else {
                                image_average_density = image_total / number_of_pixels;
                                image_variance        = (image_total_squared / number_of_pixels) - pow(image_average_density, 2);
                                image_stdev           = sqrt(image_variance);

                                if ( actual_scale_factor < 1 && use_fourier_scaling == false ) {
                                    image_stdev *= pow(actual_scale_factor, 2);
                                }

                                current_low_grey_value  = image_average_density - (image_stdev * 2.5);
                                current_high_grey_value = image_average_density + (image_stdev * 2.5);

                                /*
						   if (actual_scale_factor < 1 && use_fourier_scaling == false)
						   {
							   current_low_grey_value = image_average_density - ((image_stdev * 2.5) * pow(actual_scale_factor,2));
							   current_high_grey_value = image_average_density + ((image_stdev * 2.5) * pow(actual_scale_factor,2));
						   }
						   else
						   {
							   current_low_grey_value = image_average_density - (image_stdev * 2.5);
							   current_high_grey_value = image_average_density + (image_stdev * 2.5);
						   }*/
                            }
                        }
                        else {
                            //  we aren't scaling locally, so threshold to the global values..

                            current_low_grey_value  = this->low_grey_value;
                            current_high_grey_value = this->high_grey_value;

                            //this->first_header.Threshold(current_low_grey_value, current_high_grey_value);
                        }

                        // scale to grey values and draw..

                        if ( invert_contrast == true && use_fft == false ) {
                            float buffer            = current_low_grey_value;
                            current_low_grey_value  = current_high_grey_value;
                            current_high_grey_value = buffer;
                        }

                        range = current_high_grey_value - current_low_grey_value;
                        range /= 256.0;

                        counter       = 0;
                        red_counter   = 0;
                        green_counter = 1;
                        blue_counter  = 2;

                        if ( use_fourier_scaling == true && (scaled_x_size != ReturnImageXSize( ) || scaled_y_size != ReturnImageYSize( )) ) {
                            for ( j = 0; j < scaled_image_memory_buffer[image_counter].logical_y_dimension; j++ ) {
                                address = (scaled_image_memory_buffer[image_counter].logical_y_dimension - 1 - j) * (scaled_image_memory_buffer[image_counter].logical_x_dimension + scaled_image_memory_buffer[image_counter].padding_jump_value);

                                for ( i = 0; i < scaled_image_memory_buffer[image_counter].logical_x_dimension; i++ ) {
                                    if ( (parent_display_panel->style_flags & DRAW_IMAGE_SEPARATOR) == DRAW_IMAGE_SEPARATOR && (j == 0 || j == scaled_image_memory_buffer[image_counter].logical_y_dimension - 1 || i == 0 || i == scaled_image_memory_buffer[image_counter].logical_x_dimension - 1) ) {
                                        image_data[red_counter]   = 0;
                                        image_data[green_counter] = 0;
                                        image_data[blue_counter]  = 0;
                                    }
                                    else {
                                        if ( current_high_grey_value - current_low_grey_value == 0 )
                                            current_value = 128;
                                        else
                                            current_value = int(myround((scaled_image_memory_buffer[image_counter].real_values[address] - current_low_grey_value) / range));

                                        if ( current_value > 255 )
                                            current_value = 255;
                                        else if ( current_value < 0 )
                                            current_value = 0;

                                        image_data[red_counter]   = (unsigned char)(current_value);
                                        image_data[green_counter] = (unsigned char)(current_value);
                                        image_data[blue_counter]  = (unsigned char)(current_value);
                                    }

                                    red_counter += 3;
                                    green_counter += 3;
                                    blue_counter += 3;

                                    address++;
                                }
                            }
                        }
                        else {

                            for ( j = 0; j < image_memory_buffer[image_counter].logical_y_dimension; j++ ) {
                                address = (image_memory_buffer[image_counter].logical_y_dimension - 1 - j) * (image_memory_buffer[image_counter].logical_x_dimension + image_memory_buffer[image_counter].padding_jump_value);

                                for ( i = 0; i < image_memory_buffer[image_counter].logical_x_dimension; i++ ) {
                                    if ( (parent_display_panel->style_flags & DRAW_IMAGE_SEPARATOR) == DRAW_IMAGE_SEPARATOR && (float(j) <= 1.0f / desired_scale_factor || float(j) >= image_memory_buffer[image_counter].logical_y_dimension - 1.0f / desired_scale_factor || float(i) <= 1.0f / desired_scale_factor || float(i) >= image_memory_buffer[image_counter].logical_x_dimension - 1.0f / desired_scale_factor) ) {
                                        image_data[red_counter]   = 0;
                                        image_data[green_counter] = 0;
                                        image_data[blue_counter]  = 0;
                                    }
                                    else {
                                        if ( current_high_grey_value - current_low_grey_value == 0 )
                                            current_value = 128;
                                        else
                                            current_value = int(myround((image_memory_buffer[image_counter].real_values[address] - current_low_grey_value) / range));

                                        if ( current_value > 255 )
                                            current_value = 255;
                                        else if ( current_value < 0 )
                                            current_value = 0;

                                        image_data[red_counter]   = (unsigned char)(current_value);
                                        image_data[green_counter] = (unsigned char)(current_value);
                                        image_data[blue_counter]  = (unsigned char)(current_value);
                                    }

                                    red_counter += 3;
                                    green_counter += 3;
                                    blue_counter += 3;

                                    address++;
                                }
                            }

                            if ( scaled_x_size > ReturnImageXSize( ) )
                                panel_image->Rescale(scaled_x_size, scaled_y_size, wxIMAGE_QUALITY_NORMAL);
                            else
                                panel_image->Rescale(scaled_x_size, scaled_y_size, wxIMAGE_QUALITY_HIGH);
                        }

                        panel_image_has_correct_scale = true;
                        panel_image_has_correct_greys = true;
                    }

                    // Here we convert the image to a bitmap (after first scaling it if need be), then
                    // we write the appropriate text on it, and blit it to the memory bitmap, in the
                    // case of "single image mode" things are different as in this case we cut out a sub-bitmap
                    // and only blit that..

                    if ( single_image == true ) {
                        cut_x_size = window_x_size;
                        cut_y_size = window_y_size;

                        // cut out the appropriate section taking the scaling factor into account.
                        // prevent cutting outside the size of the image..

                        if ( single_image_x * actual_scale_factor + window_x_size > panel_image->GetWidth( ) )
                            cut_x_size = panel_image->GetWidth( ) - single_image_x * actual_scale_factor;
                        if ( single_image_y * actual_scale_factor + window_y_size > panel_image->GetHeight( ) )
                            cut_y_size = panel_image->GetHeight( ) - single_image_y * actual_scale_factor;

                        FrameImage = panel_image->GetSubImage(wxRect(single_image_x * actual_scale_factor, single_image_y * actual_scale_factor, cut_x_size, cut_y_size));
                    }
                    else {
                        FrameImage = panel_image->Copy( );
                        /*
					if (image_memory_buffer[image_counter].x_size != scaled_x_size || image_memory_buffer[image_counter].y_size != scaled_y_size)
					{
						// first copy it, then scale it appropriately..

						FrameImage.Rescale(scaled_x_size, scaled_y_size);

					}*/
                    }

                    //convert into a bitmap..

                    DrawBitmap = new wxBitmap(FrameImage);

                    // prepare the pen, prepare the label, then write it on..

                    memDC.SelectObject(*DrawBitmap);
                    memDC.SetPen(red_dashed_pen);
                    memDC.SetBackgroundMode(wxSOLID);
                    memDC.SetTextForeground(*wxWHITE);
                    memDC.SetTextBackground(*wxBLACK);

#ifndef __WXMSW__
                    memDC.SetFont(*wxSMALL_FONT);
#else
                    memDC.SetFont(*wxNORMAL_FONT);
#endif

                    if ( this->show_label == true ) {
                        LabelText = wxString::Format(wxT("%i"), int(this->current_location + image_counter));
                    }
                    else
                        LabelText = wxT("");

                    // write on the text..

                    memDC.DrawText(LabelText, 0, DrawBitmap->GetHeight( ) - memDC.GetCharHeight( ));

                    // draw on crosshair if desired..

                    /*
				if (show_crosshair == true)
				{
					memDC.CrossHair(DrawBitmap->GetWidth() / 2, DrawBitmap->GetHeight() / 2);
				}*/

                    //memDC.SetPen(wxNullPen);
                    memDC.SelectObject(wxNullBitmap);
                    // blit it on..

                    dc.SelectObject(panel_bitmap);
                    dc.DrawBitmap(*DrawBitmap, big_x * scaled_x_size, big_y * scaled_y_size);
                    dc.SelectObject(wxNullBitmap);

                    // invalidate only the freshly drawn section, the paint event will
                    // then only redraw that area which should be much faster..

                    wxRect DrawnArea(big_x * scaled_x_size, big_y * scaled_y_size, scaled_x_size, scaled_y_size);

                    this->Refresh(false, &DrawnArea);
                    Update( );

                    /*
				// if we scaled it, we need to set it back to it's original size and re-get the
				// data pointer..

				if (this->first_header.x_size != scaled_x_size || first_header.y_size != scaled_y_size)
				{
					BufferImage.Create(int(this->first_header.x_size), int(this->first_header.y_size), false);
					image_data = BufferImage.GetData();
				}
*/
                    image_counter++;

                    delete DrawBitmap;
                }
            }
        }

        // do a final refresh for neatness..

        parent_display_panel->UpdateToolbar( );
        Update( );

        //	wxEndBusyCursor();
        parent_display_panel->ChangeFocusToPanel( );
    }
}

void DisplayNotebookPanel::OnEraseBackground(wxEraseEvent& event) {
    //event.Skip();
}

void DisplayNotebookPanel::OnPaint(wxPaintEvent& evt) {
    wxPaintDC dc(this);

    int window_x_size;
    int window_y_size;
    //int point_size = 3 * actual_scale_factor;

    //if (point_size < 3) point_size = 3;

    int point_size = selected_point_size;

    long counter;
    long coord_counter;
    long waypoint_counter;

    wxCoord actual_x, actual_y;
    float   tangent_x, tangent_y;

    GetClientSize(&window_x_size, &window_y_size);

    // is our stored bitmap the correct size? if not redraw

    if ( window_x_size != panel_bitmap.GetWidth( ) || window_y_size != panel_bitmap.GetHeight( ) ) {
        ReDrawPanel( );
    }
    else if ( panel_image_has_correct_greys == false || panel_image_has_correct_scale == false ) {
        ReDrawPanel( );
    }
    //else
    {
        //just redraw the areas that have changed..

        int              vX, vY, vW, vH; // Dimensions of client area in pixels
        wxRegionIterator upd(GetUpdateRegion( )); // get the update rect list

        while ( upd ) {
            vX = upd.GetX( );
            vY = upd.GetY( );
            vW = upd.GetW( );
            vH = upd.GetH( );

            // Repaint this rectangle

            if ( vX + vW <= panel_bitmap.GetWidth( ) && vY + vH <= panel_bitmap.GetHeight( ) )
                dc.DrawBitmap(panel_bitmap.GetSubBitmap(wxRect(vX, vY, vW, vH)), vX, vY);

            upd++;
        }

        // if we are drawing a selection box - draw it..

        /*
		if (drawing_selection_square == true)
		{
			wxPen selection_pen(*wxRED, 1, wxSHORT_DASH);
			dc.SetPen(selection_pen);
			dc.SetBrush(wxBrush(*wxRED, wxTRANSPARENT));

			dc.DrawRectangle(selection_square_start_x, selection_square_start_y, selection_square_current_x - selection_square_start_x, selection_square_current_y - selection_square_start_y);
		}*/

        // check if any of the current images are selected, if they are then
        // put a funky little red box around them.. or if coords mode is
        // selected.. draw coords. -UNLESS SUSPENDED

        if ( suspend_overlays == false ) {

            dc.SetPen(*wxRED);
            if ( picking_mode == IMAGES_PICK )
                dc.SetBrush(wxBrush(*wxRED, wxTRANSPARENT));
            else if ( picking_mode == COORDS_PICK ) {
                dc.SetBrush(wxBrush(*wxRED, wxSOLID));
            }

            counter = current_location;

            for ( int y = 0; y < images_in_y; y++ ) {
                for ( int x = 0; x < images_in_x; x++ ) {
                    if ( ReturnNumberofImages( ) >= counter ) {
                        if ( (parent_display_panel->style_flags & CAN_SELECT_IMAGES) == CAN_SELECT_IMAGES || parent_display_panel->ReturnCurrentPanel( )->picking_mode == IMAGES_PICK ) {
                            if ( image_is_selected[counter] == true ) {
                                // draw a rectangle around the image..

                                if ( single_image == true ) {
                                    dc.DrawRoundedRectangle(0, 0, window_x_size, window_y_size, -.5);
                                }
                                else
                                    dc.DrawRoundedRectangle(x * current_x_size, y * current_y_size, current_x_size, current_y_size, -.5);
                            }
                        }
                        // Otherwise, we're picking coords
                        else {
                            // find all the coordinates for this image..
                            for ( coord_counter = 0; coord_counter < coord_tracker->number_of_coords; coord_counter++ ) {
                                if ( coord_tracker->coords[coord_counter].image_number == counter ) {
                                    // draw the point on..
                                    if ( single_image == true ) {
                                        // we need to check if the co-ordinate is inside the current view, and if so, draw it.

                                        if ( coord_tracker->coords[coord_counter].x_pos > single_image_x && coord_tracker->coords[coord_counter].x_pos < single_image_x + window_x_size / actual_scale_factor && coord_tracker->coords[coord_counter].y_pos > single_image_y && coord_tracker->coords[coord_counter].y_pos < single_image_y + window_y_size / actual_scale_factor ) {
                                            dc.DrawCircle((coord_tracker->coords[coord_counter].x_pos - single_image_x) * actual_scale_factor, (coord_tracker->coords[coord_counter].y_pos - single_image_y) * actual_scale_factor, point_size);
                                        }

                                        // if this coord is the last coord, and show selection distances is true, show the distance.

                                        if ( coord_counter == coord_tracker->number_of_coords - 1 && coord_tracker->number_of_coords > 1 && show_selection_distances == true && coord_tracker->coords[coord_counter].image_number == coord_tracker->coords[coord_counter - 1].image_number ) {
                                            dc.DrawLine((coord_tracker->coords[coord_counter].x_pos - single_image_x) * actual_scale_factor, (coord_tracker->coords[coord_counter].y_pos - single_image_y) * actual_scale_factor, (coord_tracker->coords[coord_counter - 1].x_pos - single_image_x) * actual_scale_factor, (coord_tracker->coords[coord_counter - 1].y_pos - single_image_y) * actual_scale_factor);
                                        }
                                    }
                                    else {
                                        dc.DrawCircle(x * current_x_size + (coord_tracker->coords[coord_counter].x_pos * actual_scale_factor), y * current_y_size + (coord_tracker->coords[coord_counter].y_pos * actual_scale_factor), point_size);

                                        if ( coord_counter == coord_tracker->number_of_coords - 1 && coord_tracker->number_of_coords > 1 && show_selection_distances == true && coord_tracker->coords[coord_counter].image_number == coord_tracker->coords[coord_counter - 1].image_number ) {
                                            dc.DrawLine(x * current_x_size + (coord_tracker->coords[coord_counter].x_pos * actual_scale_factor), y * current_y_size + (coord_tracker->coords[coord_counter].y_pos * actual_scale_factor), x * current_x_size + (coord_tracker->coords[coord_counter - 1].x_pos * actual_scale_factor), y * current_y_size + (coord_tracker->coords[coord_counter - 1].y_pos * actual_scale_factor));
                                        }
                                    }
                                }
                            }
                        }
                        counter++;
                    }
                }

                if ( template_matching_marker_x_pos != -1.0 && template_matching_marker_y_pos != -1.0 ) {
                    dc.SetPen(*wxRED);
                    dc.SetBrush(wxBrush(*wxRED, wxTRANSPARENT));
                    int radius = myroundint(template_matching_marker_radius * actual_scale_factor);
                    if ( radius < 5 )
                        radius = 5;
                    dc.DrawCircle(myroundint(template_matching_marker_x_pos * actual_scale_factor), current_y_size - myroundint(template_matching_marker_y_pos * actual_scale_factor) - 1, radius);
                }

                if ( current_location <= blue_selection_square_location && blue_selection_square_location <= (current_location + images_in_current_view) ) {
                    counter = current_location;

                    for ( int y = 0; y < images_in_y; y++ ) {
                        for ( int x = 0; x < images_in_x; x++ ) {
                            if ( ReturnNumberofImages( ) >= counter ) {
                                if ( counter == blue_selection_square_location ) {
                                    dc.SetPen(wxColor(38, 124, 181));
                                    // draw a rectangle around the image..

                                    if ( single_image == true ) {
                                        dc.DrawRectangle(0, 0, window_x_size, window_y_size);
                                    }
                                    else
                                        dc.DrawRectangle(x * current_x_size, y * current_y_size, current_x_size, current_y_size);
                                }
                            }

                            counter++;
                        }
                    }
                }
            }
        }
    }

    evt.Skip( );
}

DisplayPopup::DisplayPopup(wxWindow* parent, int flags)
    : wxPopupWindow(parent, flags) {
    Bind(wxEVT_PAINT, &DisplayPopup::OnPaint, this);
    Bind(wxEVT_ERASE_BACKGROUND, &DisplayPopup::OnEraseBackground, this);

    parent_display_panel = reinterpret_cast<DisplayPanel*>(parent);
    SetBackgroundColour(*wxBLACK);

    DisplayNotebookPanel* current_panel = parent_display_panel->ReturnCurrentPanel( );

    if ( current_panel->use_unscaled_image_for_popup == true ) {

        float high_grey;
        float low_grey;

        if ( current_panel->grey_values_decided_by == LOCAL_GREYS ) {
            current_panel->image_memory_buffer->GetMinMax(low_grey, high_grey);
        }
        else if ( current_panel->grey_values_decided_by == AUTO_GREYS ) // auto contrast
        {
            // work out mean, and stdev..

            float image_total         = 0;
            float image_total_squared = 0;
            long  number_of_pixels    = 0;
            int   i, j;

            long address = 0;

            for ( j = 0; j < current_panel->image_memory_buffer->logical_y_dimension; j++ ) {
                for ( i = 0; i < current_panel->image_memory_buffer->logical_x_dimension; i++ ) {
                    if ( current_panel->image_memory_buffer->real_values[address] != 0. ) {
                        image_total += current_panel->image_memory_buffer->real_values[address];
                        image_total_squared += powf(current_panel->image_memory_buffer->real_values[address], 2);
                        number_of_pixels++;
                    }
                    address++;
                }

                address += current_panel->image_memory_buffer->padding_jump_value;
            }

            if ( image_total_squared == 0.0f ) {
                low_grey  = 0.0f;
                high_grey = 0.0f;
            }
            else {
                float image_average_density = image_total / number_of_pixels;
                float image_variance        = (image_total_squared / number_of_pixels) - powf(image_average_density, 2);
                float image_stdev           = sqrt(image_variance);

                low_grey  = image_average_density - (image_stdev * 2.5);
                high_grey = image_average_density + (image_stdev * 2.5);
            }
        }
        else {
            //  we aren't scaling locally, so threshold to the global values..

            low_grey  = current_panel->low_grey_value;
            high_grey = current_panel->high_grey_value;
        }

        if ( current_panel->invert_contrast == true && current_panel->use_fft == false ) {
            current_low_grey_value  = high_grey;
            current_high_grey_value = low_grey;
        }
        else {
            current_low_grey_value  = low_grey;
            current_high_grey_value = high_grey;
        }
    }
}

void DisplayPopup::OnPaint(wxPaintEvent& evt) {
    wxPaintDC dc(this);

    DisplayNotebookPanel* current_panel = parent_display_panel->ReturnCurrentPanel( );

    int sub_bitmap_x_pos    = x_pos + 64;
    int sub_bitmap_y_pos    = y_pos + 64;
    int sub_bitmap_x_size   = 128;
    int sub_bitmap_y_size   = 128;
    int sub_bitmap_x_offset = 0;
    int sub_bitmap_y_offset = 0;

    if ( current_panel->use_unscaled_image_for_popup == true ) {
        // get the original image position..

        int original_x_pos = myroundint(double(x_pos + 128) / current_panel->actual_scale_factor);
        int original_y_pos = myroundint(double(y_pos + 128) / current_panel->actual_scale_factor);

        int x, y;
        int image_x_coord;
        int image_y_coord;

        int   current_value;
        float range;

        long counter;
        long red_counter;
        long green_counter;
        long blue_counter;

        wxImage        subimage(sub_bitmap_x_size, sub_bitmap_y_size);
        unsigned char* image_data = subimage.GetData( );

        range = current_high_grey_value - current_low_grey_value;
        range /= 256.0;

        counter       = 0;
        red_counter   = 0;
        green_counter = 1;
        blue_counter  = 2;

        for ( y = -63; y < 65; y++ ) {
            for ( x = -63; x < 65; x++ ) {

                if ( current_high_grey_value - current_low_grey_value == 0 )
                    current_value = 128;
                else {
                    image_x_coord = original_x_pos + x;
                    image_y_coord = original_y_pos + y;

                    if ( image_x_coord >= 0 && image_x_coord < current_panel->image_memory_buffer->logical_x_dimension && image_y_coord >= 0 && image_y_coord < current_panel->image_memory_buffer->logical_y_dimension ) {
                        current_value = int(myround((current_panel->image_memory_buffer->ReturnRealPixelFromPhysicalCoord(image_x_coord, current_panel->image_memory_buffer->logical_y_dimension - image_y_coord, 0) - current_low_grey_value) / range));
                    }
                    else
                        current_value = 0;
                }

                if ( current_value > 255 )
                    current_value = 255;
                else if ( current_value < 0 )
                    current_value = 0;

                image_data[red_counter]   = (unsigned char)(current_value);
                image_data[green_counter] = (unsigned char)(current_value);
                image_data[blue_counter]  = (unsigned char)(current_value);

                red_counter += 3;
                green_counter += 3;
                blue_counter += 3;
            }
        }

        subimage.Rescale(sub_bitmap_x_size * 2, sub_bitmap_y_size * 2);
        wxBitmap topaint(subimage);
        dc.DrawBitmap(topaint, sub_bitmap_x_offset * 2, sub_bitmap_y_offset * 2);
    }
    else {

        // We are going to grab the section of the panel bitmap which corresponds to
        // a 128x128 square under the panel.  It is made slightly more complicated by the
        // fact that if we request part of a bitmap which does not exist the entire square
        // will be blank, so we have to do some bounds checking..

        int sub_bitmap_x_pos    = x_pos + 64;
        int sub_bitmap_y_pos    = y_pos + 64;
        int sub_bitmap_x_size   = 128;
        int sub_bitmap_y_size   = 128;
        int sub_bitmap_x_offset = 0;
        int sub_bitmap_y_offset = 0;

        if ( sub_bitmap_x_pos < 0 ) {
            sub_bitmap_x_offset = abs(sub_bitmap_x_pos);
            sub_bitmap_x_size -= sub_bitmap_x_offset;
        }
        else if ( sub_bitmap_x_pos >= current_panel->panel_bitmap.GetWidth( ) - 128 && sub_bitmap_x_pos < current_panel->panel_bitmap.GetWidth( ) )
            sub_bitmap_x_size = current_panel->panel_bitmap.GetWidth( ) - sub_bitmap_x_pos;

        if ( sub_bitmap_y_pos < 0 && sub_bitmap_y_pos > -128 ) {
            sub_bitmap_y_offset = abs(sub_bitmap_y_pos);
            sub_bitmap_y_size -= sub_bitmap_y_offset;
        }
        else if ( sub_bitmap_y_pos >= current_panel->panel_bitmap.GetHeight( ) - 128 && sub_bitmap_y_pos < current_panel->panel_bitmap.GetHeight( ) )
            sub_bitmap_y_size = current_panel->panel_bitmap.GetHeight( ) - sub_bitmap_y_pos;

        // the following line is a whole host of checks designed to not grab a dodgy bit of bitmap

        if ( sub_bitmap_x_pos + sub_bitmap_x_offset >= 0 && sub_bitmap_y_pos + sub_bitmap_y_offset >= 0 && sub_bitmap_y_pos + sub_bitmap_y_offset < current_panel->panel_bitmap.GetHeight( ) && sub_bitmap_x_pos + sub_bitmap_x_offset < current_panel->panel_bitmap.GetWidth( ) && sub_bitmap_x_size > 0 && sub_bitmap_y_size > 0 ) {
            wxBitmap section    = current_panel->panel_bitmap.GetSubBitmap(wxRect(sub_bitmap_x_pos + sub_bitmap_x_offset, sub_bitmap_y_pos + sub_bitmap_y_offset, sub_bitmap_x_size, sub_bitmap_y_size));
            wxImage  paintimage = section.ConvertToImage( );
            paintimage.Rescale(section.GetWidth( ) * 2, section.GetHeight( ) * 2);
            wxBitmap topaint(paintimage);

            dc.DrawBitmap(topaint, sub_bitmap_x_offset * 2, sub_bitmap_y_offset * 2);
            dc.SetPen(wxPen(*wxRED, 2, wxLONG_DASH));
            dc.CrossHair(128, 128);
        }
    }

    evt.Skip( );
}

void DisplayPopup::OnEraseBackground(wxEraseEvent& event) {

    //event.Skip();
}

DisplayManualDialog::DisplayManualDialog(wxWindow* parent, int id, const wxString& title, const wxPoint& pos, const wxSize& size, long style) : DisplayManualDialogParent(parent, id, title, pos, size, wxDEFAULT_DIALOG_STYLE) {

#include "icons/display_previous_icon.cpp"
#include "icons/display_next_icon.cpp"

    current_location = 1;

    wxString temp_string;
    my_parent                           = (DisplayPanel*)parent;
    DisplayNotebookPanel* current_panel = my_parent->ReturnCurrentPanel( );

    current_grey_method = current_panel->grey_values_decided_by;

    temp_string             = wxString::Format(wxT("%i"), int(current_panel->current_location));
    wxStaticText* junk_text = new wxStaticText(Toolbar, wxID_ANY, "Image : ", wxDefaultPosition, wxSize(-1, -1));
    toolbar_location_text   = new wxTextCtrl(Toolbar, Toolbar_Location_Text, temp_string, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER | wxTE_CENTRE, wxDefaultValidator, wxTextCtrlNameStr);

    temp_string = wxString::Format(wxT(" / %i"), int(current_panel->included_image_numbers.GetCount( )));

    toolbar_number_of_locations_text = new wxStaticText(Toolbar, wxID_ANY, temp_string, wxDefaultPosition, wxSize(-1, -1));

    Toolbar->AddControl(junk_text);
    Toolbar->AddTool(Toolbar_Previous, wxT("Previous"), wxBITMAP_PNG_FROM_DATA(display_previous_icon), wxNullBitmap, wxITEM_NORMAL, wxT("Previous"), wxT("Move to the previous set of images"));
    Toolbar->EnableTool(Toolbar_Previous, false);

    Toolbar->AddControl(toolbar_location_text);
    Toolbar->AddControl(toolbar_number_of_locations_text);

    Toolbar->AddTool(Toolbar_Next, wxT("Next"), wxBITMAP_PNG_FROM_DATA(display_next_icon), wxNullBitmap, wxITEM_NORMAL, wxT("Next"), wxT("Move to the next set of images"));
    Toolbar->EnableTool(Toolbar_Next, true);

    // events

    Bind(wxEVT_TEXT_ENTER, &DisplayManualDialog::OnImageChange, this, Toolbar_Location_Text);
    Bind(wxEVT_MENU, &DisplayManualDialog::OnPrevious, this, Toolbar_Previous);
    Bind(wxEVT_MENU, &DisplayManualDialog::OnNext, this, Toolbar_Next);

    current_panel->LoadIntoImage(&InputImage, current_panel->current_location - 1);

    // if we have no manual values set, then min/max the input image - otherwise load them..

    if ( current_panel->manual_low_grey == 0 && current_panel->manual_high_grey == 0 ) {
        float min_density;
        float max_density;

        InputImage.GetMinMax(min_density, max_density);
        temp_string = wxString::Format(wxT("%f"), min_density);
        minimum_text_ctrl->SetValue(temp_string);
        temp_string = wxString::Format(wxT("%f"), max_density);
        maximum_text_ctrl->SetValue(temp_string);
    }
    else {
        temp_string = wxString::Format(wxT("%f"), float(current_panel->manual_low_grey));
        minimum_text_ctrl->SetValue(temp_string);
        temp_string = wxString::Format(wxT("%f"), float(current_panel->manual_high_grey));
        maximum_text_ctrl->SetValue(temp_string);
    }

    Layout( );
    MainSizer->Fit(this);

    int width_of_dialog;
    int height_of_dialog;

    have_global_histogram = false;

    GetClientSize(&width_of_dialog, &height_of_dialog);

    histogram        = new float[width_of_dialog];
    global_histogram = new float[width_of_dialog];
    histogram_bitmap.Create(width_of_dialog, 200);

    GetLocalHistogram( );
    PaintHistogram( );
}

void DisplayManualDialog::GetLocalHistogram(void) {

    long address = 0;
    int  i, j;

    float grey_level = 0;

    long grey_level_index;
    long counter;

    int width_of_dialog;
    int height_of_dialog;

    GetClientSize(&width_of_dialog, &height_of_dialog);

    for ( counter = 0; counter < width_of_dialog; counter++ ) {
        histogram[counter] = 0;
    }

    InputImage.GetMinMax(min_grey_level, max_grey_level);

    // Set-up grey level increment between bins
    grey_level_increment = (max_grey_level - min_grey_level) / double(width_of_dialog);

    for ( j = 0; j < InputImage.logical_y_dimension; j++ ) {
        for ( i = 0; i < InputImage.logical_x_dimension; i++ ) {
            grey_level = InputImage.real_values[address];

            if ( grey_level != 0 ) {
                grey_level_index = (int)floor(0.5 + ((grey_level - min_grey_level) / grey_level_increment));

                // Check for bounds
                if ( grey_level_index < 0 )
                    grey_level_index = 0;
                if ( grey_level_index >= width_of_dialog )
                    grey_level_index = width_of_dialog - 1;

                // Increment count
                histogram[grey_level_index] += 1.0;
            }
            address++;
        }

        address += InputImage.padding_jump_value;
    }

    Refresh( );
    Update( );
}

void DisplayManualDialog::PaintHistogram(void) {
    wxMemoryDC dc(histogram_bitmap);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear( );

    wxPen mypen;
    mypen.SetColour(0, 0, 0);
    dc.SetPen(mypen);

    int width_of_dialog;
    int height_of_dialog;

    float max_value = 0;

    GetClientSize(&width_of_dialog, &height_of_dialog);

    float* current_histogram = new float[width_of_dialog];

    for ( int counter = 0; counter < width_of_dialog; counter++ ) {
        if ( histogram_checkbox->IsChecked( ) == true )
            current_histogram[counter] = global_histogram[counter];
        else
            current_histogram[counter] = histogram[counter];
    }

    for ( int x = 0; x < width_of_dialog; x++ ) {
        if ( current_histogram[x] > max_value )
            max_value = current_histogram[x];
    }

    float scale_factor = 200. / max_value;

    for ( int y = 0; y < 200; y++ ) {
        for ( int x = 0; x < width_of_dialog; x++ ) {
            if ( current_histogram[x] * scale_factor > float(y) )
                dc.DrawPoint(x, 199 - y);
        }
    }

    dc.SelectObject(wxNullBitmap);

    Refresh( );
    Update( );

    delete[] current_histogram;
}

bool DisplayManualDialog::GetGlobalHistogram(void) {
    DisplayNotebookPanel* current_panel = my_parent->ReturnCurrentPanel( );

    long number_of_images = current_panel->included_image_numbers.GetCount( );
    long address;
    int  i, j;
    bool should_continue = false;

    float grey_level = 0;

    long grey_level_index;
    long counter;
    long image_counter;

    int width_of_dialog;
    int height_of_dialog;

    GetClientSize(&width_of_dialog, &height_of_dialog);

    for ( counter = 0; counter < width_of_dialog; counter++ ) {
        global_histogram[counter] = 0;
    }

    // if the global grey values are set we will just use them.. otherwise
    // we have to calculate them..

    if ( current_panel->global_low_grey == 0 && current_panel->global_high_grey == 0 ) {
        should_continue = current_panel->SetGlobalGreys( );
    }
    else
        should_continue = true;

    if ( should_continue == true ) {
        // Set-up grey level increment between bins
        global_grey_level_increment = (current_panel->global_high_grey - current_panel->global_low_grey) / double(width_of_dialog);

        // use a progress dialog..

        wxProgressDialog my_progress(wxT("Calculating..."), wxT("Calculating Global Histogram"), current_panel->included_image_numbers.GetCount( ), this, wxPD_AUTO_HIDE | wxPD_APP_MODAL | wxPD_CAN_ABORT);

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            current_panel->LoadIntoImage(&InputImage, image_counter);

            address = 0;

            for ( j = 0; j < InputImage.logical_y_dimension; j++ ) {
                for ( i = 0; i < InputImage.logical_x_dimension; i++ ) {
                    grey_level = InputImage.real_values[address];

                    if ( grey_level != 0. ) {
                        grey_level_index = (int)floor(0.5 + ((grey_level - current_panel->global_low_grey) / global_grey_level_increment));

                        // Check for bounds
                        if ( grey_level_index < 0 )
                            grey_level_index = 0;
                        if ( grey_level_index >= width_of_dialog )
                            grey_level_index = width_of_dialog - 1;

                        // Increment count
                        global_histogram[grey_level_index] += 1.0;
                    }

                    address++;
                }

                address += InputImage.padding_jump_value;
            }

            should_continue = my_progress.Update(image_counter);

            if ( should_continue == false ) {
                continue;
            }
        }

        // if it is true set

        if ( should_continue == true ) {
            have_global_histogram = true;
        }
    }

    return should_continue;
}

void DisplayManualDialog::OnPaint(wxPaintEvent& evt) {
    bool has_worked;
    long current_x;

    wxPaintDC dc(this);
    wxPen     mypen;

    double min_grey;
    double max_grey;

    int width_of_dialog;
    int height_of_dialog;

    GetClientSize(&width_of_dialog, &height_of_dialog);

    // draw on the histogram..

    dc.DrawBitmap(histogram_bitmap, 0, 0);

    // now draw on the bars to represent the grey values..

    //	wxTextCtrl *min_text = (wxTextCtrl*) FindWindow(Manual_Min_TextCtrl);
    //	wxTextCtrl *max_text = (wxTextCtrl*) FindWindow(Manual_Max_TextCtrl);
    //	wxCheckBox *current_box = (wxCheckBox*) FindWindow(Manual_Histogram_All_Check);
    DisplayNotebookPanel* current_panel = my_parent->ReturnCurrentPanel( );

    double this_grey_level_increment;
    double this_min_grey_level;

    if ( histogram_checkbox->IsChecked( ) == true ) {
        this_grey_level_increment = global_grey_level_increment;
        this_min_grey_level       = current_panel->global_low_grey;
    }
    else {

        this_grey_level_increment = grey_level_increment;
        this_min_grey_level       = min_grey_level;
    }

    wxString temp_string = minimum_text_ctrl->GetValue( );
    has_worked           = temp_string.ToDouble(&min_grey);

    if ( has_worked != false ) {
        temp_string = maximum_text_ctrl->GetValue( );
        has_worked  = temp_string.ToDouble(&max_grey);
    }

    // if they both worked draw on the lines if not - do nothing..

    if ( has_worked != false ) {
        // which x is the min_value on?

        current_x = (int)floor(0.5 + ((min_grey - this_min_grey_level) / this_grey_level_increment));

        // Check for bounds
        if ( current_x >= 0 && current_x <= width_of_dialog ) {
            mypen.SetColour(*wxGREEN);
            dc.SetPen(mypen);
            dc.DrawLine(current_x, 0, current_x, 201);
        }

        current_x = (int)floor(0.5 + ((max_grey - this_min_grey_level) / this_grey_level_increment));

        if ( current_x >= 0 && current_x <= width_of_dialog ) {
            mypen.SetColour(*wxRED);
            dc.SetPen(mypen);
            dc.DrawLine(current_x - 1, 0, current_x - 1, 201);
        }
    }
}

void DisplayManualDialog::OnLeftDown(wxMouseEvent& event) {
    long x_pos;
    long y_pos;

    event.GetPosition(&x_pos, &y_pos);

    if ( y_pos < 200 ) {

        DisplayNotebookPanel* current_panel = my_parent->ReturnCurrentPanel( );

        double new_grey = current_panel->manual_low_grey;

        double this_grey_level_increment;
        double this_min_grey_level;

        if ( histogram_checkbox->IsChecked( ) == true ) {
            this_grey_level_increment = global_grey_level_increment;
            this_min_grey_level       = current_panel->global_low_grey;
        }
        else {

            this_grey_level_increment = grey_level_increment;
            this_min_grey_level       = min_grey_level;
        }

        // so we need to work out the grey value of the current x
        // then we can set it..

        new_grey = this_min_grey_level + (x_pos * this_grey_level_increment);

        // set the appropriate textctrl - this will also send a
        // change notification which will repaint..

        minimum_text_ctrl->SetValue(wxString::Format(wxT("%f"), new_grey));

        if ( new_grey != current_panel->manual_low_grey && live_checkbox->IsChecked( ) == true ) {
            current_panel->low_grey_value                = new_grey;
            current_panel->panel_image_has_correct_greys = false;
            current_panel->grey_values_decided_by        = MANUAL_GREYS;
            current_panel->ReDrawPanel( );
        }

        Refresh( );
    }
}

void DisplayManualDialog::OnRightDown(wxMouseEvent& event) {

    long x_pos;
    long y_pos;

    event.GetPosition(&x_pos, &y_pos);

    if ( y_pos < 200 ) {

        DisplayNotebookPanel* current_panel = my_parent->ReturnCurrentPanel( );

        double new_grey = current_panel->manual_high_grey;

        double this_grey_level_increment;
        double this_min_grey_level;

        if ( histogram_checkbox->IsChecked( ) == true ) {
            this_grey_level_increment = global_grey_level_increment;
            this_min_grey_level       = current_panel->global_low_grey;
        }
        else {

            this_grey_level_increment = grey_level_increment;
            this_min_grey_level       = min_grey_level;
        }

        // so we need to work out the grey value of the current x
        // then we can set it..

        new_grey = this_min_grey_level + (x_pos * this_grey_level_increment);

        // set the appropriate textctrl - this will also send a
        // change notification which will repaint..

        maximum_text_ctrl->SetValue(wxString::Format(wxT("%f"), new_grey));

        if ( new_grey != current_panel->manual_high_grey && live_checkbox->IsChecked( ) == true ) {
            current_panel->high_grey_value               = new_grey;
            current_panel->grey_values_decided_by        = MANUAL_GREYS;
            current_panel->panel_image_has_correct_greys = false;
            current_panel->ReDrawPanel( );
        }

        Refresh( );
    }

    event.Skip( );
}

void DisplayManualDialog::OnMotion(wxMouseEvent& event) {
    // check if the mouse is in the histogram window..

    long x_pos;
    long y_pos;

    event.GetPosition(&x_pos, &y_pos);

    if ( y_pos < 200 ) {
        DisplayNotebookPanel* current_panel = my_parent->ReturnCurrentPanel( );

        double this_grey_level_increment;
        double this_min_grey_level;

        if ( histogram_checkbox->IsChecked( ) == true ) {
            this_grey_level_increment = global_grey_level_increment;
            this_min_grey_level       = current_panel->global_low_grey;
        }
        else {

            this_grey_level_increment = grey_level_increment;
            this_min_grey_level       = min_grey_level;
        }

        // yes it is.. so check if a button is down..

        if ( event.m_leftDown == true && event.m_rightDown == false ) {

            double new_grey = this_min_grey_level + (x_pos * this_grey_level_increment);

            // set the appropriate textctrl - this will also send a
            // change notification which will repaint..

            minimum_text_ctrl->SetValue(wxString::Format(wxT("%f"), new_grey));

            if ( new_grey != current_panel->manual_low_grey && live_checkbox->IsChecked( ) == true ) {
                current_panel->low_grey_value                = new_grey;
                current_panel->panel_image_has_correct_greys = false;
                current_panel->grey_values_decided_by        = MANUAL_GREYS;
                current_panel->ReDrawPanel( );
            }
        }
        else if ( event.m_rightDown == true && event.m_leftDown == false ) {
            double new_grey = this_min_grey_level + (x_pos * this_grey_level_increment);

            // set the appropriate textctrl - this will also send a
            // change notification which will repaint..

            maximum_text_ctrl->SetValue(wxString::Format(wxT("%f"), new_grey));

            if ( new_grey != current_panel->manual_high_grey && live_checkbox->IsChecked( ) == true ) {
                current_panel->high_grey_value               = new_grey;
                current_panel->panel_image_has_correct_greys = false;
                current_panel->grey_values_decided_by        = MANUAL_GREYS;
                current_panel->ReDrawPanel( );
            }
        }

        Refresh( );
    }
    event.Skip( );
}

void DisplayManualDialog::OnButtonOK(wxCommandEvent& WXUNUSED(event)) {
    bool has_worked;

    double wanted_min_grey;
    double wanted_max_grey;

    wxString temp_string = minimum_text_ctrl->GetValue( );
    has_worked           = temp_string.ToDouble(&wanted_min_grey);

    if ( has_worked != false ) {
        temp_string = maximum_text_ctrl->GetValue( );
        has_worked  = temp_string.ToDouble(&wanted_max_grey);
    }

    if ( has_worked != false ) {
        DisplayNotebookPanel* current_panel = my_parent->ReturnCurrentPanel( );

        if ( wanted_min_grey != current_panel->manual_low_grey || wanted_max_grey != current_panel->manual_high_grey ) {
            current_panel->manual_low_grey  = wanted_min_grey;
            current_panel->manual_high_grey = wanted_max_grey;

            current_panel->panel_image_has_correct_greys = false;

            Show(false);
            current_panel->low_grey_value         = wanted_min_grey;
            current_panel->high_grey_value        = wanted_max_grey;
            current_panel->grey_values_decided_by = MANUAL_GREYS;
            current_panel->ReDrawPanel( );
        }
    }

    Destroy( );
}

void DisplayManualDialog::OnHistogramCheck(wxCommandEvent& WXUNUSED(event)) {

    // find out if it is checked or not.. if it is we may need to calculate
    // if it isn't we can do nothing as the painting will sort it out..

    bool should_continue = true;

    if ( histogram_checkbox->IsChecked( ) == true ) {
        // first disable the image select box..

        toolbar_location_text->Disable( );

        // we have just checked for a global histogram
        // if it has already been calculated, then do nothing.
        // Otherwise we need to calculate it..

        if ( have_global_histogram == false ) {
            should_continue = GetGlobalHistogram( );

            if ( should_continue == false ) {
                // if should_continue is false, then the user cancelled so uncheck the box
                // and also re-enable the image select box..

                histogram_checkbox->SetValue(false);
                toolbar_location_text->Enable( );
            }
        }
    }
    else {
        // enable the image select box..
        toolbar_location_text->Enable( );
    }

    PaintHistogram( );
}

void DisplayManualDialog::OnRealtimeCheck(wxCommandEvent& WXUNUSED(event)) {
    bool   has_worked;
    double wanted_min_grey;
    double wanted_max_grey;

    wxString temp_string = minimum_text_ctrl->GetValue( );
    has_worked           = temp_string.ToDouble(&wanted_min_grey);

    if ( has_worked != false ) {
        temp_string = maximum_text_ctrl->GetValue( );
        has_worked  = temp_string.ToDouble(&wanted_max_grey);
    }

    if ( has_worked != false ) {
        DisplayNotebookPanel* current_panel = my_parent->ReturnCurrentPanel( );

        if ( wanted_min_grey != current_panel->manual_low_grey || wanted_max_grey != current_panel->manual_high_grey ) {
            current_panel->manual_low_grey  = wanted_min_grey;
            current_panel->manual_high_grey = wanted_max_grey;

            current_panel->panel_image_has_correct_greys = false;

            current_panel->low_grey_value         = wanted_min_grey;
            current_panel->high_grey_value        = wanted_max_grey;
            current_panel->grey_values_decided_by = MANUAL_GREYS;
            current_panel->ReDrawPanel( );
        }
    }
}

void DisplayManualDialog::OnClose(wxCloseEvent& event) {
    DisplayNotebookPanel* current_panel   = my_parent->ReturnCurrentPanel( );
    current_panel->grey_values_decided_by = current_grey_method;
    current_panel->ReDrawPanel( );

    event.Skip( );
}

void DisplayManualDialog::OnButtonCancel(wxCommandEvent& WXUNUSED(event)) {
    DisplayNotebookPanel* current_panel   = my_parent->ReturnCurrentPanel( );
    current_panel->grey_values_decided_by = current_grey_method;
    current_panel->ReDrawPanel( );
    Destroy( );
}

void DisplayManualDialog::OnLowChange(wxCommandEvent& WXUNUSED(event)) {
    Refresh( );
    Update( );
}

void DisplayManualDialog::OnImageChange(wxCommandEvent& WXUNUSED(event)) {

    DisplayNotebookPanel* current_panel = my_parent->ReturnCurrentPanel( );

    long set_location;

    // get what the current value is.. if it can be converted to a valid number then
    // change the current location to that number and redraw.. otherwise set the
    // value back to the previous value..

    wxString current_string = toolbar_location_text->GetValue( );
    bool     has_worked     = current_string.ToLong(&set_location);

    if ( has_worked == true ) {
        // is this number valid?

        if ( set_location > 0 && set_location <= current_panel->included_image_numbers.GetCount( ) ) {
            current_location = set_location;

            if ( current_location == 1 )
                Toolbar->EnableTool(Toolbar_Previous, false);
            else
                Toolbar->EnableTool(Toolbar_Previous, true);

            if ( current_location == current_panel->included_image_numbers.GetCount( ) )
                Toolbar->EnableTool(Toolbar_Next, false);
            else
                Toolbar->EnableTool(Toolbar_Next, true);

            current_panel->LoadIntoImage(&InputImage, set_location - 1);

            GetLocalHistogram( );
            PaintHistogram( );
        }
        else
            has_worked = false;
    }

    // if for some reason it hasn't worked - set it back to it's previous value..

    if ( has_worked == false ) {
        toolbar_location_text->SetValue(wxString::Format(wxT("%li"), current_location));
        Refresh( );
        Update( );
    }
}

void DisplayManualDialog::OnNext(wxCommandEvent& event) {

    DisplayNotebookPanel* current_panel = my_parent->ReturnCurrentPanel( );

    long set_location;

    // get what the current value is.. if it can be converted to a valid number then
    // change the current location to that number and redraw.. otherwise set the
    // value back to the previous value..

    wxString current_string = toolbar_location_text->GetValue( );
    bool     has_worked     = current_string.ToLong(&set_location);
    set_location++;

    if ( has_worked == true ) {
        // is this number valid?

        if ( set_location > 0 && set_location <= current_panel->included_image_numbers.GetCount( ) ) {
            toolbar_location_text->ChangeValue(wxString::Format(wxT("%li"), set_location));
            current_location = set_location;
            current_panel->LoadIntoImage(&InputImage, set_location - 1);

            if ( current_location == 1 )
                Toolbar->EnableTool(Toolbar_Previous, false);
            else
                Toolbar->EnableTool(Toolbar_Previous, true);

            if ( current_location == current_panel->included_image_numbers.GetCount( ) )
                Toolbar->EnableTool(Toolbar_Next, false);
            else
                Toolbar->EnableTool(Toolbar_Next, true);

            GetLocalHistogram( );
            PaintHistogram( );
        }
        else
            has_worked = false;
    }

    // if for some reason it hasn't worked - set it back to it's previous value..

    if ( has_worked == false ) {
        toolbar_location_text->SetValue(wxString::Format(wxT("%li"), current_location));
        Refresh( );
        Update( );
    }
}

void DisplayManualDialog::OnPrevious(wxCommandEvent& event) {
    DisplayNotebookPanel* current_panel = my_parent->ReturnCurrentPanel( );

    long set_location;

    // get what the current value is.. if it can be converted to a valid number then
    // change the current location to that number and redraw.. otherwise set the
    // value back to the previous value..

    wxString current_string = toolbar_location_text->GetValue( );
    bool     has_worked     = current_string.ToLong(&set_location);
    set_location--;

    if ( has_worked == true ) {
        // is this number valid?

        if ( set_location > 0 && set_location <= current_panel->included_image_numbers.GetCount( ) ) {
            toolbar_location_text->ChangeValue(wxString::Format(wxT("%li"), set_location));
            current_location = set_location;
            current_panel->LoadIntoImage(&InputImage, set_location - 1);

            if ( current_location == 1 )
                Toolbar->EnableTool(Toolbar_Previous, false);
            else
                Toolbar->EnableTool(Toolbar_Previous, true);

            if ( current_location == current_panel->included_image_numbers.GetCount( ) )
                Toolbar->EnableTool(Toolbar_Next, false);
            else
                Toolbar->EnableTool(Toolbar_Next, true);

            GetLocalHistogram( );
            PaintHistogram( );
        }
        else
            has_worked = false;
    }

    // if for some reason it hasn't worked - set it back to it's previous value..

    if ( has_worked == false ) {
        toolbar_location_text->SetValue(wxString::Format(wxT("%li"), current_location));
        Refresh( );
        Update( );
    }
}

void DisplayManualDialog::OnHighChange(wxCommandEvent& WXUNUSED(event)) {
    Refresh( );
    Update( );
}

void DisplayPanel::SetTabNameUnsaved( ) {
    ReturnCurrentPanel( )->txt_is_saved = false;
    RefreshTabName( );
}

void DisplayPanel::SetTabNameSaved( ) {
    ReturnCurrentPanel( )->txt_is_saved = true;
    RefreshTabName( );
}

void DisplayPanel::RefreshTabName( ) {
    int selected_tab = my_notebook->GetSelection( );

    wxString tab_text;

    if ( ! ReturnCurrentPanel( )->have_txt_filename && ! ReturnCurrentPanel( )->txt_is_saved ) {
        tab_text = wxT("*") + ReturnCurrentPanel( )->short_image_filename;
    }
    else
        tab_text = ReturnCurrentPanel( )->short_image_filename;

    if ( ReturnCurrentPanel( )->have_txt_filename ) {
        tab_text += wxT(" : ");
        if ( ! ReturnCurrentPanel( )->txt_is_saved )
            tab_text += wxT("*");
        tab_text += ReturnCurrentPanel( )->short_txt_filename;
    }

    my_notebook->SetPageText(selected_tab, tab_text);
}

CoordTracker::CoordTracker(wxWindow* parent) {
    parent_notebook = reinterpret_cast<DisplayNotebookPanel*>(parent);
    // start off with 1000 coords..

    number_allocated = 1000;

    coords           = new Coord[1000];
    number_of_coords = 0;
}

CoordTracker::~CoordTracker( ) {
    delete[] coords;
}

void CoordTracker::Clear( ) {
    number_of_coords = 0;

    if ( number_allocated > 1000 ) {
        delete[] coords;
        coords           = new Coord[1000];
        number_allocated = 1000;
    }
}

void CoordTracker::ToggleCoord(long wanted_image, long wanted_x, long wanted_y) {
    // first check to see if it is already there..

    bool was_found = false;

    for ( long counter = 0; counter < number_of_coords; counter++ ) {
        if ( wanted_image == coords[counter].image_number ) {
            // it is on the same image..

            if ( abs(wanted_x - coords[counter].x_pos) < parent_notebook->selected_point_size && abs(wanted_y - coords[counter].y_pos) < parent_notebook->selected_point_size ) {
                // we are assuming that these two coords, correspond. Thus we want to remove this coord.

                RemoveCoord(counter);

                was_found = true;
                counter   = number_of_coords;
            }
        }
    }

    // if it wasn't found, then add it.

    if ( was_found == false ) {
        AddCoord(wanted_image, wanted_x, wanted_y);
    }
    // we need to know the distance between the last two coords (if possible), so that if
    // the user selects show_selection_distance - the distance will be displayed.

    // default position is no distance
    parent_notebook->selected_distance = 0.;

    // first_check there are at least two coords..

    if ( number_of_coords > 1 ) {
        // now check the last two are on the same image..

        if ( coords[number_of_coords - 1].image_number == coords[number_of_coords - 2].image_number ) {
            // now calculate distance..

            parent_notebook->selected_distance = sqrt(pow(coords[number_of_coords - 1].x_pos - coords[number_of_coords - 2].x_pos, 2) + pow(coords[number_of_coords - 1].y_pos - coords[number_of_coords - 2].y_pos, 2));
        }
    }
}

void CoordTracker::AddCoord(long wanted_image, long wanted_x, long wanted_y) {
    long counter;

    number_of_coords++;
    if ( number_allocated < number_of_coords ) {
        // have to allocate more space..

        Coord* coord_buffer = new Coord[number_allocated];

        for ( counter = 0; counter < number_allocated; counter++ ) {
            coord_buffer[counter].image_number = coords[counter].image_number;
            coord_buffer[counter].x_pos        = coords[counter].x_pos;
            coord_buffer[counter].y_pos        = coords[counter].y_pos;
        }

        // now allocate more space and copy back

        delete[] coords;
        coords = new Coord[number_allocated * 2];

        for ( counter = 0; counter < number_allocated; counter++ ) {
            coords[counter].image_number = coord_buffer[counter].image_number;
            coords[counter].x_pos        = coord_buffer[counter].x_pos;
            coords[counter].y_pos        = coord_buffer[counter].y_pos;
        }

        delete[] coord_buffer;

        number_allocated *= 2;
    }

    coords[number_of_coords - 1].image_number = wanted_image;
    coords[number_of_coords - 1].x_pos        = wanted_x;
    coords[number_of_coords - 1].y_pos        = wanted_y;

    //parent_display_panel->SetTabNameUnsaved( );
}

void CoordTracker::RemoveCoord(long coord_to_remove) {
    // remove the coord, then move all the coords after it up in the list.

    long counter;
    long buffer_counter = 0;

    //if (coord_to_remove == number_of_coords - 1) number_of_coords--;
    //else
    {
        Coord* coord_buffer = new Coord[(number_of_coords - coord_to_remove) + 5];

        for ( counter = coord_to_remove + 1; counter < number_of_coords; counter++ ) {
            coord_buffer[buffer_counter].image_number = coords[counter].image_number;
            coord_buffer[buffer_counter].x_pos        = coords[counter].x_pos;
            coord_buffer[buffer_counter].y_pos        = coords[counter].y_pos;

            buffer_counter++;
        }

        // take one off..

        number_of_coords--;

        // copy back..

        buffer_counter = 0.;

        for ( counter = coord_to_remove; counter < number_of_coords; counter++ ) {
            coords[counter].image_number = coord_buffer[buffer_counter].image_number;
            coords[counter].x_pos        = coord_buffer[buffer_counter].x_pos;
            coords[counter].y_pos        = coord_buffer[buffer_counter].y_pos;

            buffer_counter++;
        }

        delete[] coord_buffer;
    }

    parent_notebook->parent_display_panel->SetTabNameUnsaved( );
}

void CoordTracker::RectangleRemoveCoord(long wanted_image, long start_x, long start_y, long end_x, long end_y) {
    long from_x;
    long to_x;

    long from_y;
    long to_y;

    bool removed_any_coords = false;

    if ( end_x < start_x ) {
        from_x = end_x;
        to_x   = start_x;
    }
    else {
        from_x = start_x;
        to_x   = end_x;
    }

    if ( end_y < start_y ) {
        from_y = end_y;
        to_y   = start_y;
    }
    else {
        from_y = start_y;
        to_y   = end_y;
    }

    for ( long counter = 0; counter < number_of_coords; counter++ ) {
        if ( wanted_image == coords[counter].image_number ) {
            // it is on the same image..

            if ( coords[counter].x_pos > from_x && coords[counter].x_pos < to_x && coords[counter].y_pos > from_y && coords[counter].y_pos < to_y ) {
                RemoveCoord(counter);
                counter--;
                removed_any_coords = true;
            }
        }
    }

    if ( removed_any_coords == true )
        parent_notebook->parent_display_panel->SetTabNameUnsaved( );
}