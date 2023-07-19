#include "../core/gui_core_headers.h"

DisplayFrame::DisplayFrame(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style)
    : DisplayFrameParent(NULL, wxID_ANY, title, pos, size, style) {

    is_fullscreen = false;
    image_is_open = false;

    this->cisTEMDisplayPanel->Initialise(CAN_CHANGE_FILE | CAN_CLOSE_TABS | CAN_MOVE_TABS | CAN_FFT);

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
    cisTEMDisplayPanel->OnOpen(event);
    this->image_is_open = true;
}

void DisplayFrame::OpenFile(wxString wanted_filename, wxString wanted_tab_title, wxArrayLong* wanted_included_image_numbers, bool keep_scale_and_location_if_possible, bool force_local_survey) {
    // Open the file
    cisTEMDisplayPanel->OpenFile(wanted_filename, wanted_tab_title, wanted_included_image_numbers, keep_scale_and_location_if_possible, force_local_survey);
    cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode = IMAGES_PICK; // Set the newly opened image to image picking by default
    image_is_open                                           = true;
}

void DisplayFrame::OnCloseTabClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel != NULL && cisTEMDisplayPanel->ReturnCurrentPanel( ) != NULL ) {
        cisTEMDisplayPanel->my_notebook->DeletePage(cisTEMDisplayPanel->my_notebook->GetSelection( ));
    }
    if ( cisTEMDisplayPanel->my_notebook->GetSelection( ) == wxNOT_FOUND ) {
        DisableAllToolbarButtons( );
        image_is_open = false;
    }
}

void DisplayFrame::OnExitClick(wxCommandEvent& event) {
    this->Destroy( );
}

void DisplayFrame::OnLocationNumberClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->show_label == true )
        cisTEMDisplayPanel->ReturnCurrentPanel( )->show_label = false;
    else
        cisTEMDisplayPanel->ReturnCurrentPanel( )->show_label = true;

    cisTEMDisplayPanel->ReturnCurrentPanel( )->ReDrawPanel( );
}

void DisplayFrame::OnImageSelectionModeClick(wxCommandEvent& event) {
    // if we are already in selections mode, we don't want to do anything, so
    // make a check.
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode != IMAGES_PICK ) {
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->number_of_coords > 0 ) {
            int dialog_result;

            wxMessageDialog question_dialog(this, "By switching the selection mode, you will lose your current coordinates selections if they are unsaved!\nAre you sure you want to continue?", "Are you Sure?", wxYES_NO | wxNO_DEFAULT | wxICON_EXCLAMATION);
            dialog_result = question_dialog.ShowModal( );

            if ( dialog_result == wxID_YES ) {
                cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->Clear( );
                cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode = IMAGES_PICK;

                // Change checkmark if coords mode selected
                if ( SelectCoordsSelectionMode->IsChecked( ) ) {
                    SelectCoordsSelectionMode->Check(false);
                    SelectImageSelectionMode->Check(true);
                }
                SelectInvertSelection->Enable(true);
            }

            // User does not want to switch; do nothing
            else
                return;
        }
        else
            cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode = IMAGES_PICK;

        ClearTextFileFromPanel( );
        Refresh( );
        Update( );
    }
}

void DisplayFrame::OnCoordsSelectionModeClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode != COORDS_PICK ) {
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->number_of_selections > 0 ) {
            wxMessageDialog question_dialog(this, "By switching the selection mode, you will lose your current image selections.\nAre you sure you want to continue?", "Switch Selection Modes?", wxYES_NO | wxNO_DEFAULT | wxICON_EXCLAMATION);
            if ( question_dialog.ShowModal( ) == wxID_YES ) {
                cisTEMDisplayPanel->ClearSelection(false);
                cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode = COORDS_PICK;

                // Need to make sure the check mark switches
                if ( SelectImageSelectionMode->IsChecked( ) ) {
                    SelectImageSelectionMode->Check(false);
                    SelectCoordsSelectionMode->Check(true); // First, get into image picking mode
                }
                SelectInvertSelection->Enable(false);
            }
            // User doesn't want to lose selections; do nothing
            else
                return;
        }

        // No selections
        else {
            cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode = COORDS_PICK;
            // Switch check marks
            if ( SelectImageSelectionMode->IsChecked( ) ) {
                SelectImageSelectionMode->Check(false);
                SelectCoordsSelectionMode->Check(true); // First, get into image picking mode
            }
        }
        ClearTextFileFromPanel( );
        Refresh( );
        Update( );
    }
}

void DisplayFrame::OnOpenTxtClick(wxCommandEvent& event) {
    // TODO: check the file to see if there's 3 values or not
    // If there are 2 whitespaces, we know it's a coords file
    // Also add a similar check when in images_pick mode
    bool     valid_file = true;
    wxString name_of_file;
    // We want to open image selections if we're in IMAGES_PICK mode
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode == IMAGES_PICK ) {
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->number_of_selections > 0 ) {
            wxMessageDialog open_without_saving_selections_dialog(this, "To open image selections, all current selections must be cleared. Do you want to continue without saving?", "Proceed without Saving Current Selections?", wxYES_NO | wxNO_DEFAULT | wxICON_EXCLAMATION);
            if ( open_without_saving_selections_dialog.ShowModal( ) == wxID_YES ) {
                cisTEMDisplayPanel->ClearSelection(true);
                ClearTextFileFromPanel( );
            }
            // User does not want to switch; do nothing
            else
                return;
        }

        wxString      caption          = wxT("Open selections from text file");
        wxString      wildcard         = wxT("TXT files (*.txt)|*.txt");
        wxString      remember_path    = wxGetCwd( );
        wxString      default_dir      = remember_path;
        wxString      default_filename = wxEmptyString;
        wxFileDialog* open_dialog      = new wxFileDialog(this, caption, default_dir, default_filename, wildcard, wxFD_OPEN);
        if ( open_dialog->ShowModal( ) == wxID_OK ) {
            //Start with setting up the file info
            wxString path            = open_dialog->GetPath( );
            remember_path            = open_dialog->GetDirectory( );
            name_of_file             = open_dialog->GetFilename( );
            wxFileName  filename     = name_of_file;
            wxTextFile* file_to_open = new wxTextFile(name_of_file);

            // Start reading from the file
            file_to_open->Open( );
            wxString current_line;

            // Continue reading until through the file
            size_t line_counter = 0;
            while ( valid_file && line_counter < file_to_open->GetLineCount( ) ) {
                current_line = file_to_open->GetLine(line_counter);
                valid_file   = LoadSelections(current_line);
                line_counter++;
            }
        }
    }
    // Otherwise, we're in coords mode
    else {
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->number_of_coords > 0 ) {
            wxMessageDialog question_dialog(this, "Opening a text file with coordinates selected will clear any currently selected coordinates, so if they are needed, it is recommended to save them first. Do you want to continue without saving?", "Clear Coordinates?", wxYES_NO | wxNO_DEFAULT | wxICON_EXCLAMATION);
            if ( question_dialog.ShowModal( ) == wxID_YES ) {
                cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->Clear( ); // Clear out any selections before adding new ones.
                ClearTextFileFromPanel( );
            }
            // User wants to save first, so do nothing
            else
                return;
        }

        // Now we're definitely in coords mode with no coords selected; let user select file
        wxString      caption          = wxT("Open coordinates from text file");
        wxString      wildcard         = wxT("TXT files (*.txt)|*.txt");
        wxString      remember_path    = wxGetCwd( );
        wxString      default_dir      = remember_path;
        wxString      default_filename = wxEmptyString;
        wxFileDialog* open_dialog      = new wxFileDialog(this, caption, default_dir, default_filename, wildcard, wxFD_OPEN);
        if ( open_dialog->ShowModal( ) == wxID_OK ) {
            //Start with setting up the file info
            wxString path            = open_dialog->GetPath( );
            remember_path            = open_dialog->GetDirectory( );
            name_of_file             = open_dialog->GetFilename( );
            wxFileName  filename     = name_of_file;
            wxTextFile* file_to_open = new wxTextFile(name_of_file);

            // Start reading from the file
            file_to_open->Open( );
            wxString current_line;
            long     x, y, image_number;

            // Continue reading until through the file
            size_t line_counter = 0;
            while ( valid_file && line_counter < file_to_open->GetLineCount( ) ) {
                current_line = file_to_open->GetLine(line_counter);
                valid_file   = LoadCoords(current_line, x, y, image_number);
                line_counter++;
            }
        }
    }
    if ( valid_file ) {
        cisTEMDisplayPanel->ReturnCurrentPanel( )->short_txt_filename = name_of_file;
        cisTEMDisplayPanel->ReturnCurrentPanel( )->have_txt_filename  = true;
        cisTEMDisplayPanel->SetTabNameSaved( );
    }

    Refresh( );
    Update( );
}

void DisplayFrame::OnSaveTxtClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->have_txt_filename && cisTEMDisplayPanel->ReturnCurrentPanel( )->txt_is_saved ) {
        cisTEMDisplayPanel->SetTabNameSaved( ); // Just make sure it's saved and tab name is up to date
        return;
    }
    else if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->have_txt_filename && ! cisTEMDisplayPanel->ReturnCurrentPanel( )->txt_is_saved ) {
        // We have a filename and it's not saved, which means there's been changes
        // to image or coords selections; we want to clear the extant file
        // and replace it with what's current

        wxTextFile extant_file(cisTEMDisplayPanel->ReturnCurrentPanel( )->short_txt_filename); // Get a wxTextFile from extant file
        if ( ! extant_file.Exists( ) ) {
            wxMessageDialog nonexistent_dialog(this, "The text file you're attempting to overwrite does not exist.", "Error: File to save does not exist.", wxOK | wxOK_DEFAULT | wxICON_EXCLAMATION);
            return;
        }
        else {
            extant_file.Open( );
            extant_file.Clear( );

            if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode == IMAGES_PICK ) {
                for ( long i = 0; i < cisTEMDisplayPanel->number_of_frames; i++ ) {
                    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->image_is_selected[i] )
                        extant_file.AddLine(wxString::Format("%li", i));
                }
            }

            // COORDS_PICK mode
            else {
                for ( int i = 0; i < cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->number_of_coords; i++ ) {
                    extant_file.AddLine(wxString::Format("%li %li %li", cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->coords[i].x_pos, cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->coords[i].y_pos, cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->coords[i].image_number));
                }
            }
            extant_file.Write( );
            extant_file.Close( );
        }
        cisTEMDisplayPanel->SetTabNameSaved( );
    }
}

void DisplayFrame::OnSaveTxtAsClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode == IMAGES_PICK ) {
        wxString   caption          = wxT("Save image selections as text file");
        wxString   wildcard         = wxT("TXT files (*.txt)|*.txt");
        wxString   remember_path    = wxGetCwd( );
        wxString   default_dir      = remember_path;
        wxFileName mrc_name         = cisTEMDisplayPanel->ReturnCurrentPanel( )->filename;
        wxString   default_filename = "selections_" + mrc_name.GetName( ) + ".txt";
        wxFileName temp_filename    = default_filename;
        int        temp_int         = 1;

        // If the filename already exists, apppend an integer to default name
        if ( temp_filename.Exists( ) ) {
            while ( temp_filename.Exists( ) ) {
                temp_filename = default_filename;
                temp_filename = wxString::Format("%i_" + default_filename, temp_int);
                temp_int++;
            }
        }
        default_filename = temp_filename.GetFullName( );

        wxFileDialog* save_dialog = new wxFileDialog(this, caption, default_dir, default_filename, wildcard, wxFD_SAVE);
        if ( save_dialog->ShowModal( ) == wxID_OK ) {
            default_filename                = save_dialog->GetFilename( );
            wxTextFile* new_selections_file = new wxTextFile(default_filename);
            for ( long i = 0; i < cisTEMDisplayPanel->number_of_frames; i++ ) {
                if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->image_is_selected[i] )
                    new_selections_file->AddLine(wxString::Format("%li", i));
            }
            new_selections_file->Write( );
            new_selections_file->Close( );
        }

        // Track the currently opened file for saving if user selects more images
        cisTEMDisplayPanel->ReturnCurrentPanel( )->short_txt_filename = default_filename;
        cisTEMDisplayPanel->ReturnCurrentPanel( )->have_txt_filename  = true;
        cisTEMDisplayPanel->SetTabNameSaved( );
    }

    else {
        wxString   caption          = wxT("Save coordinates as text file");
        wxString   wildcard         = wxT("TXT files (*.txt)|*.txt");
        wxString   remember_path    = wxGetCwd( );
        wxString   default_dir      = remember_path;
        wxFileName mrc_name         = cisTEMDisplayPanel->ReturnCurrentPanel( )->filename;
        wxString   default_filename = "coords_" + mrc_name.GetName( ) + ".txt";
        wxFileName temp_filename    = default_filename;
        int        temp_int         = 1;

        // If the filename already exists, apppend an integer to default name
        if ( temp_filename.Exists( ) ) {
            while ( temp_filename.Exists( ) ) {
                temp_filename = default_filename;
                temp_filename = wxString::Format("%i_" + default_filename, temp_int);
                temp_int++;
            }
        }
        default_filename = temp_filename.GetFullName( );

        // Now set up the file with the new name and then open the dialog for saving
        wxFileDialog* save_dialog = new wxFileDialog(NULL, caption, default_dir, default_filename, wildcard, wxFD_SAVE);
        if ( save_dialog->ShowModal( ) == wxID_OK ) {
            default_filename            = save_dialog->GetFilename( );
            wxTextFile* new_coords_file = new wxTextFile(default_filename);
            for ( int i = 0; i < cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->number_of_coords; i++ ) {
                new_coords_file->AddLine(wxString::Format("%li %li %li", cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->coords[i].x_pos, cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->coords[i].y_pos, cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->coords[i].image_number));
            }
            new_coords_file->Write( );
            new_coords_file->Close( );
        }

        // Track the currently opened file for saving if user selects more coords
        cisTEMDisplayPanel->ReturnCurrentPanel( )->short_txt_filename = default_filename;
        cisTEMDisplayPanel->ReturnCurrentPanel( )->have_txt_filename  = true;
        cisTEMDisplayPanel->SetTabNameSaved( );
    }
}

void DisplayFrame::OnInvertSelectionClick(wxCommandEvent& event) {
    for ( long image_counter = 1; image_counter <= cisTEMDisplayPanel->number_of_frames; image_counter++ ) {
        cisTEMDisplayPanel->ToggleImageSelected(image_counter, false);
    }
    cisTEMDisplayPanel->RefreshCurrentPanel( );
    cisTEMDisplayPanel->SetTabNameUnsaved( );
}

void DisplayFrame::OnClearSelectionClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode == IMAGES_PICK ) {
        for ( int image_counter = 0; image_counter < cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnNumberofImages( ); image_counter++ ) {
            cisTEMDisplayPanel->ClearSelection(false);
        }
    }
    else if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode == COORDS_PICK ) {
        cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->Clear( );
    }
    ClearTextFileFromPanel( );
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
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->show_selection_distances )
        cisTEMDisplayPanel->ReturnCurrentPanel( )->show_selection_distances = false;
    else {
    }
}

void DisplayFrame::OnShowResolution(wxCommandEvent& event) {
    double wanted_pixel_size;

    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->resolution_instead_of_radius )
        cisTEMDisplayPanel->ReturnCurrentPanel( )->resolution_instead_of_radius = false;
    else {
        wxTextEntryDialog text_dialog(this, wxT("Pixel Size (Angstroms"), wxT("Select Pixel Size"), wxString::Format(wxT("%.2f"), cisTEMDisplayPanel->ReturnCurrentPanel( )->pixel_size), wxOK | wxCANCEL | wxCENTRE, wxDefaultPosition);
        text_dialog.ShowModal( );

        wxString current_value = text_dialog.GetValue( );
        text_dialog.Destroy( );
        if ( current_value.ToDouble(&wanted_pixel_size) == true ) {
            cisTEMDisplayPanel->ReturnCurrentPanel( )->pixel_size                   = wanted_pixel_size;
            cisTEMDisplayPanel->ReturnCurrentPanel( )->resolution_instead_of_radius = true;
        }
    }
}

void DisplayFrame::OnDocumentationClick(wxCommandEvent& event) {
    wxLaunchDefaultBrowser("http://www.cistem.org/documentation");
}

// This prevents using buttons when an image or stack is not open to act on
void DisplayFrame::DisableAllToolbarButtons( ) {

    // Open menu only needs close tab disabled
    DisplayCloseTab->Enable(false);

    // Label menu
    LabelLocationNumber->Enable(false);

    // Select menu
    SelectImageSelectionMode->Enable(false);
    SelectCoordsSelectionMode->Enable(false);
    SelectOpenTxt->Enable(false);
    SelectSaveTxt->Enable(false);
    SelectSaveTxtAs->Enable(false);
    SelectInvertSelection->Enable(false);
    SelectClearSelection->Enable(false);

    // Options menu
    OptionsSetPointSize->Enable(false);
    OptionsShowCrossHair->Enable(false);
    Options7BitGreyValues->Enable(false);
    OptionsSingleImageMode->Enable(false);
    OptionsShowSelectionDistances->Enable(false);
    OptionsShowResolution->Enable(false);
}

// Call when an image is opened to activate all toolbar buttons
void DisplayFrame::EnableAllToolbarButtons( ) {
    // Open menu only needs close tab disabled
    DisplayCloseTab->Enable( );

    // Label menu
    LabelLocationNumber->Enable( );

    // Select menu
    SelectImageSelectionMode->Enable(true);
    SelectCoordsSelectionMode->Enable(true);
    SelectOpenTxt->Enable(true);
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->have_txt_filename )
        SelectSaveTxt->Enable(true);
    else
        SelectSaveTxt->Enable(false);

    SelectSaveTxtAs->Enable(true);
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->picking_mode == IMAGES_PICK )
        SelectInvertSelection->Enable(true);
    else
        SelectInvertSelection->Enable(false);
    SelectClearSelection->Enable(true);

    // Options menu
    OptionsSetPointSize->Enable(true);
    OptionsShowCrossHair->Enable(true);
    Options7BitGreyValues->Enable(true);
    OptionsSingleImageMode->Enable(true);
    OptionsShowSelectionDistances->Enable(true);
    OptionsShowResolution->Enable(true);
}

void DisplayFrame::OnUpdateUI(wxUpdateUIEvent& event) {
    if ( this->cisTEMDisplayPanel->my_notebook->GetSelection( ) != wxNOT_FOUND )
        this->EnableAllToolbarButtons( );
    else
        this->DisableAllToolbarButtons( );

    if ( image_is_open ) {
        // Check that there are coords selected
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->number_of_coords > 0 ) {
            SelectSaveTxtAs->Enable(true);
            SelectSaveTxt->Enable(true);
        }
        else {
            SelectSaveTxtAs->Enable(false);
            SelectSaveTxt->Enable(false);
        }
    }
}

bool DisplayFrame::LoadCoords(wxString current_line, long& x, long& y, long& image_number) {
    // Parse the string for x, y, and the image number
    size_t index_of_whitespace      = current_line.find(' ');
    size_t prev_whitespace_position = 0;
    if ( index_of_whitespace == wxNOT_FOUND ) {
        wxMessageDialog wrong_file_format(this, "Cannot open Image Selection text file in Coordinate Selection mode.", "Incorrect File Format", wxOK | wxOK_DEFAULT | wxICON_EXCLAMATION);
        wrong_file_format.ShowModal( );
        return false;
    }
    current_line.SubString(prev_whitespace_position, index_of_whitespace - 1).ToLong(&x);
    prev_whitespace_position = index_of_whitespace;
    index_of_whitespace      = current_line.find(' ', index_of_whitespace + 1);
    current_line.SubString(prev_whitespace_position + 1, index_of_whitespace - 1).ToLong(&y);
    prev_whitespace_position = index_of_whitespace;
    index_of_whitespace      = current_line.find('\n', index_of_whitespace + 1);
    current_line.SubString(prev_whitespace_position + 1, index_of_whitespace - 1).ToLong(&image_number);

    // First, check that all coordinates and image numbers are valid for the open image
    if ( x < cisTEMDisplayPanel->x_size && y < cisTEMDisplayPanel->y_size && image_number < cisTEMDisplayPanel->number_of_frames ) {
        cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->ToggleCoord(image_number, x, y);
        return true;
    }
    else {
        wxMessageDialog invalid_file_dialog(this, wxString::Format("The selected coordinates exceed the dimensions of the currently opened *.mrc file. Cannot open selected coordinates (Selected x: %li, selected y: %li, image num: %li for image(s) with dimensions x: %i, y: %i, num images: %i). Try checking the selection mode and/or the text file contents.", x, y, image_number, cisTEMDisplayPanel->x_size, cisTEMDisplayPanel->y_size, cisTEMDisplayPanel->number_of_frames), "Invalid Coordinates for Current Image(s)", wxOK | wxOK_DEFAULT | wxICON_EXCLAMATION);
        if ( invalid_file_dialog.ShowModal( ) == wxID_OK )
            cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->Clear( );
        return false;
    }
}

bool DisplayFrame::LoadSelections(wxString current_line) {
    // Quick check of file format
    size_t index_of_whitespace = current_line.find(' ');
    if ( index_of_whitespace != wxNOT_FOUND ) {
        wxMessageDialog wrong_file_format(this, "Cannot open Coordinate Selection text file in Image Selection mode.", "Incorrect File Format", wxOK | wxOK_DEFAULT | wxICON_EXCLAMATION);
        wrong_file_format.ShowModal( );
        return false;
    }

    // Get the value that's selected
    long image_number = -1;
    current_line.ToLong(&image_number);

    // If the value exceeds the possible dimensions don't try to access the index for setting selected
    if ( image_number < cisTEMDisplayPanel->number_of_frames ) {
        cisTEMDisplayPanel->ReturnCurrentPanel( )->SetImageSelected(image_number, false);
        return true;
    }
    else {
        wxMessageDialog invalid_file_dialog(this, wxString::Format("The file being opened contains selected images that exceed the number of images in the current file. Cannot open the selections.(Images in open file: %i. Image index sought: %li)", cisTEMDisplayPanel->number_of_frames, image_number), "Invalid Selection(s) for Current Image(s)", wxOK | wxOK_DEFAULT | wxICON_EXCLAMATION);
        cisTEMDisplayPanel->ClearSelection(false);
    }
}

void DisplayFrame::ClearTextFileFromPanel( ) {
    cisTEMDisplayPanel->ReturnCurrentPanel( )->have_txt_filename  = false;
    cisTEMDisplayPanel->ReturnCurrentPanel( )->short_txt_filename = wxEmptyString;
    cisTEMDisplayPanel->SetTabNameSaved( );
}