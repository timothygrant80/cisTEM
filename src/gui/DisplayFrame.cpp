#include "../core/gui_core_headers.h"
#include "../programs/cisTEM_display/DisplayServer.h" // includes wxEVT_SERVER_OPEN_FILE
#include <wx/pen.h>

DisplayFrame::DisplayFrame(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style)
    : DisplayFrameParent(NULL, wxID_ANY, title, pos, size, style) {

    is_fullscreen = false;
    remember_path = wxGetCwd( );

    cisTEMDisplayPanel->EnableCanChangeFile( );
    cisTEMDisplayPanel->EnableCanCloseTabs( );
    cisTEMDisplayPanel->EnableCanMoveTabs( );
    cisTEMDisplayPanel->EnableCanFFT( );
    cisTEMDisplayPanel->Initialise( );

    // Set this bool to true so that DisplayPanel knows that this frame's panel is from the cisTEM_display program
    this->cisTEMDisplayPanel->is_from_display_program = true;

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
    Bind(EVT_SERVER_OPEN_FILE, &DisplayFrame::OnServerOpenFile, this);
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
}

void DisplayFrame::OnSaveDisplayedImagesClick(wxCommandEvent& event) {
    // Mimics the logic ProperOverwriteCheckSaveDialog in my_controls.cpp
    wxFileDialog save_file_dialog(this, _("Save png image"), wxEmptyString, wxEmptyString, "PNG files (*.png)|*.png", wxFD_SAVE | wxFD_OVERWRITE_PROMPT, wxDefaultPosition, wxDefaultSize, wxFileDialogNameStr);

    wxString wanted_extension = ".png";
    wxString default_dir      = cisTEMDisplayPanel->ReturnCurrentPanel( )->filename;

    // Strip away the filename to get the directory
    default_dir = default_dir.BeforeLast('/');

    save_file_dialog.SetDirectory(default_dir);
    wxString extension_lowercase = wanted_extension.Lower( );
    wxString extension_uppercase = wanted_extension.Upper( );

    if ( save_file_dialog.ShowModal( ) == wxID_CANCEL ) {
        save_file_dialog.Destroy( );
        return;
    }

    // Crop out the blank space around the image: get the true width of the relevant area on the bitmap.
    wxBitmap sub_bitmap = CropImageForSaving( );

    sub_bitmap.SaveFile(save_file_dialog.GetPath( ), wxBITMAP_TYPE_PNG);
}

void DisplayFrame::OnSaveDisplayedImagesWithLegendClick(wxCommandEvent& event) {
    // Mimics the logic ProperOverwriteCheckSaveDialog in my_controls.cpp
    wxFileDialog save_file_dialog(this, _("Save png image with legend"), wxEmptyString, wxEmptyString, "PNG files (*.png)|*.png", wxFD_SAVE | wxFD_OVERWRITE_PROMPT, wxDefaultPosition, wxDefaultSize, wxFileDialogNameStr);

    wxString wanted_extension = ".png";
    wxString default_dir      = cisTEMDisplayPanel->ReturnCurrentPanel( )->filename;

    // Strip away the filename to get the directory
    default_dir = default_dir.BeforeLast('/');

    save_file_dialog.SetDirectory(default_dir);
    wxString extension_lowercase = wanted_extension.Lower( );
    wxString extension_uppercase = wanted_extension.Upper( );

    if ( save_file_dialog.ShowModal( ) == wxID_CANCEL ) {
        save_file_dialog.Destroy( );
        return;
    }

    // Crop out the blank space around the image: get the true width of the relevant area on the bitmap.
    wxBitmap sub_bitmap     = CropImageForSaving( );
    int      sub_bmp_width  = sub_bitmap.GetWidth( );
    int      sub_bmp_height = sub_bitmap.GetHeight( );

    // Create legend, width of 80 pixels
    int legend_width = 80;

    int     legend_height = sub_bmp_height;
    wxImage legend_img(legend_width, legend_height);

    // Draw color bar gradient; this method calculates a value for each row
    // of the legend and fills it in with a grayscale color by using the
    // proportional distance from the top (max) to the bottom (min).
    for ( int y = 0; y < legend_height; ++y ) {
        double t = 1.0 - double(y) / legend_height;

        // Simple grayscale: interpolate between min and max
        unsigned char val = static_cast<unsigned char>(255 * t);
        for ( int x = 0; x < legend_width; ++x ) {
            legend_img.SetRGB(x, y, val, val, val);
        }
    }

    // Draw min/max text
    // Note: wxImage does not support drawing directly, so we convert to wxBitmap for this step
    // and then convert back to wxImage
    wxBitmap   legend_bmp(legend_img);
    wxMemoryDC dc(legend_bmp);

    // Add a spacer between the image and the legend
    int     spacer_width = 15;
    wxImage spacer_img(spacer_width, sub_bmp_height);
    for ( int y = 0; y < sub_bmp_height; ++y ) {
        for ( int x = 0; x < spacer_width; ++x ) {
            spacer_img.SetRGB(x, y, 255, 255, 255);
        }
    }

    int     combined_width = sub_bmp_width + legend_width + spacer_width;
    int     white_space    = 200;
    wxImage background_img(combined_width + white_space, sub_bmp_height + white_space);
    for ( int i = 0; i < background_img.GetWidth( ); ++i ) {
        for ( int j = 0; j < background_img.GetHeight( ); ++j ) {
            background_img.SetRGB(i, j, 255, 255, 255);
        }
    }

    // Minimum tick spacing should be about 1/5 of the legend height to balance readability and clutter;
    // if there is not much space, only use 2 gradations (min and max)
    int min_tick_spacing = sub_bmp_height / 5;
    int num_gradations   = std::max(2, legend_height / min_tick_spacing);

    float min_pixel, max_pixel;
    cisTEMDisplayPanel->ReturnCurrentPanel( )->image_memory_buffer->GetMinMax(min_pixel, max_pixel);
    float pixel_range = max_pixel - min_pixel;

    // Convert to bitmap to be able to draw
    wxBitmap background_bmp(background_img);
    dc.SelectObject(background_bmp);
    dc.SetPen(wxPen(*wxBLACK, 2));

    for ( int i = 0; i < num_gradations; ++i ) {
        // Spread gradations only across the legend area
        int legend_top_y    = white_space / 2;
        int legend_bottom_y = legend_top_y + legend_height - 1;
        int y               = legend_top_y + int(i * (legend_height - 1) / (num_gradations - 1));

        // Calculate the value corresponding to this gradation by interpolating between min and max
        double value = max_pixel - (pixel_range * i) / (num_gradations - 1);

        int legend_right_x    = sub_bmp_width + spacer_width + white_space / 2 + legend_width;
        int gradation_start_x = legend_right_x;
        int gradation_end_x   = gradation_start_x + 10;

        dc.DrawLine(gradation_start_x, y, gradation_end_x, y);

        // Subtract 12 from y to better align text with gradation line, add 5 to starting point
        // to space out from the gradation line
        dc.DrawText(wxString::Format("%.2f", value), gradation_end_x + 5, y - 12);
    }

    dc.SelectObject(wxNullBitmap);

    // Combine all the images
    wxImage combined_img(background_img.GetWidth( ), background_img.GetHeight( ), true);
    combined_img.Paste(background_bmp.ConvertToImage( ), 0, 0);
    combined_img.Paste(sub_bitmap.ConvertToImage( ), white_space / 2, white_space / 2);
    combined_img.Paste(spacer_img, sub_bmp_width + white_space / 2, white_space / 2);
    combined_img.Paste(legend_bmp.ConvertToImage( ), sub_bmp_width + spacer_width + white_space / 2, white_space / 2);

    // Finally, draw a rectangle around the legend area to separate it from the background
    // This is done last to ensure the rectangle is on top of everything else and transparent
    // so there's nothing blocking the view of the legend, but the rectangle border is still
    // visible.
    wxBitmap combined_bmp(combined_img);
    dc.SelectObject(combined_bmp);
    dc.SetBrush(*wxTRANSPARENT_BRUSH);
    dc.DrawRectangle(sub_bmp_width + spacer_width + white_space / 2, white_space / 2, legend_width, legend_height);
    combined_img = combined_bmp.ConvertToImage( );
    dc.SelectObject(wxNullBitmap);

    combined_img.SaveFile(save_file_dialog.GetPath( ), wxBITMAP_TYPE_PNG);
}

void DisplayFrame::OnServerOpenFile(wxCommandEvent& event) {
    wxString filename = event.GetString( );
    if ( cisTEMDisplayPanel ) {
        cisTEMDisplayPanel->OpenFile(filename, filename);
        this->Raise( );
        this->SetFocus( );
    }
}

void DisplayFrame::OnCloseTabClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( ) != NULL ) {
        cisTEMDisplayPanel->my_notebook->DeletePage(cisTEMDisplayPanel->my_notebook->GetSelection( ));
    }
    if ( cisTEMDisplayPanel->my_notebook->GetSelection( ) == wxNOT_FOUND ) {
        DisableAllToolbarButtons( );
    }
}

void DisplayFrame::OnExitClick(wxCommandEvent& event) {
    this->Destroy( );
}

void DisplayFrame::OnLocationNumberClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->show_label )
        cisTEMDisplayPanel->ReturnCurrentPanel( )->show_label = false;
    else
        cisTEMDisplayPanel->ReturnCurrentPanel( )->show_label = true;

    cisTEMDisplayPanel->ReturnCurrentPanel( )->ReDrawPanel( );
}

void DisplayFrame::OnImageSelectionModeClick(wxCommandEvent& event) {
    // if we are already in selections mode, we don't want to do anything, so
    // make a check.
    if ( ! cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled ) {
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->number_of_coords > 0 ) {
            wxMessageDialog question_dialog(this, "By switching the selection mode, you will lose your current coordinates selections if they are unsaved.\nDo you want to continue?", "Swtich Selection Modes?", wxYES_NO | wxNO_DEFAULT | wxICON_EXCLAMATION);

            if ( question_dialog.ShowModal( ) == wxID_YES ) {
                cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->Clear( );
                cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled = true;
                SelectInvertSelection->Enable(true);
            }

            // User does not want to switch; do nothing
            else
                return;
        }
        else
            cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled = true;

        ClearTextFileFromPanel( );
        Refresh( );
        Update( );
    }
}

void DisplayFrame::OnCoordsSelectionModeClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled ) {
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->number_of_selections > 0 ) {
            wxMessageDialog question_dialog(this, "By switching the selection mode, you will lose your current image selections.\nDo you want to continue?", "Switch Selection Modes?", wxYES_NO | wxNO_DEFAULT | wxICON_EXCLAMATION);
            if ( question_dialog.ShowModal( ) == wxID_YES ) {
                cisTEMDisplayPanel->ClearSelection(false);
                cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled = false;

                SelectInvertSelection->Enable(false);
            }
            // User doesn't want to lose selections; do nothing
            else
                return;
        }

        // No selections
        else
            cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled = false;

        ClearTextFileFromPanel( );
        Refresh( );
        Update( );
    }
}

void DisplayFrame::OnOpenTxtClick(wxCommandEvent& event) {
    bool     valid_file = true;
    wxString name_of_file;
    wxString caption;
    wxString wildcard;
    wxString default_dir;
    wxString default_filename;
    wxString path;

    // We want to open image selections if we're in IMAGES_PICK mode
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled ) {
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

        caption                   = wxT("Open selections from text file");
        wildcard                  = wxT("TXT files (*.txt)|*.txt");
        default_dir               = remember_path;
        default_filename          = wxEmptyString;
        wxFileDialog* open_dialog = new wxFileDialog(this, caption, default_dir, default_filename, wildcard, wxFD_OPEN);
        if ( open_dialog->ShowModal( ) == wxID_OK ) {
            //Start with setting up the file info
            path                     = open_dialog->GetPath( );
            remember_path            = open_dialog->GetDirectory( );
            name_of_file             = open_dialog->GetFilename( );
            wxTextFile* file_to_open = new wxTextFile(path);

            // Start reading from the file
            file_to_open->Open( );
            wxString current_line;

            // Continue reading until through the file
            size_t line_counter = 0;
            while ( valid_file && line_counter < file_to_open->GetLineCount( ) ) {
                current_line = file_to_open->GetLine(line_counter);
                valid_file   = LoadImageSelections(current_line);
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
        caption                   = wxT("Open coordinates from text file");
        wildcard                  = wxT("TXT files (*.txt)|*.txt");
        default_dir               = remember_path;
        default_filename          = wxEmptyString;
        wxFileDialog* open_dialog = new wxFileDialog(this, caption, default_dir, default_filename, wildcard, wxFD_OPEN);
        if ( open_dialog->ShowModal( ) == wxID_OK ) {
            //Start with setting up the file info
            path                     = open_dialog->GetPath( );
            remember_path            = open_dialog->GetDirectory( );
            name_of_file             = open_dialog->GetFilename( );
            wxTextFile* file_to_open = new wxTextFile(path);

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
        cisTEMDisplayPanel->ReturnCurrentPanel( )->current_file_path  = path;
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
    // Have unsaved file; update the file with current selections
    else if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->have_txt_filename && ! cisTEMDisplayPanel->ReturnCurrentPanel( )->txt_is_saved ) {
        wxTextFile file_to_update(cisTEMDisplayPanel->ReturnCurrentPanel( )->current_file_path); // Get a wxTextFile from extant file
        if ( ! file_to_update.Exists( ) ) {
            wxMessageDialog nonexistent_dialog(this, "The text file you're attempting to overwrite does not exist.", "Error: File to save does not exist.", wxOK | wxOK_DEFAULT | wxICON_EXCLAMATION);
            return;
        }
        else {
            // Just open, clear, and re-fill
            file_to_update.Open( );
            file_to_update.Clear( );

            if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled ) {
                for ( long i = 0; i <= cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnNumberofImages( ); i++ ) {
                    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->image_is_selected[i] )
                        file_to_update.AddLine(wxString::Format("%li", i));
                }
            }

            // coords mode
            else {
                for ( int i = 0; i < cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->number_of_coords; i++ ) {
                    file_to_update.AddLine(wxString::Format("%li %li %li", cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->coords[i].x_pos, cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->coords[i].y_pos, cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->coords[i].image_number));
                }
            }
            file_to_update.Write( );
            file_to_update.Close( );
        }
        cisTEMDisplayPanel->SetTabNameSaved( );
    }
}

void DisplayFrame::OnSaveTxtAsClick(wxCommandEvent& event) {
    wxString   caption;
    wxString   wildcard;
    wxString   default_dir;
    wxString   default_filename;
    wxString   path;
    wxFileName mrc_name;
    wxFileName temp_filename;
    int        temp_int;

    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled ) {
        caption          = wxT("Save image selections as text file");
        wildcard         = wxT("TXT files (*.txt)|*.txt");
        default_dir      = remember_path;
        mrc_name         = cisTEMDisplayPanel->ReturnCurrentPanel( )->filename;
        default_filename = "selections_" + mrc_name.GetName( ) + ".txt";
        temp_filename    = default_filename;
        temp_int         = 1;

        // If the default filename already exists, apppend an integer to default name
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
            path                            = save_dialog->GetPath( );
            remember_path                   = save_dialog->GetDirectory( );
            wxTextFile* new_selections_file = new wxTextFile(path);
            for ( long i = 0; i <= cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnNumberofImages( ); i++ ) {
                if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->image_is_selected[i] )
                    new_selections_file->AddLine(wxString::Format("%li", i));
            }
            new_selections_file->Write( );
            new_selections_file->Close( );
        }
    }

    // coords mode
    else {
        caption          = wxT("Save coordinates as text file");
        wildcard         = wxT("TXT files (*.txt)|*.txt");
        default_dir      = remember_path;
        mrc_name         = cisTEMDisplayPanel->ReturnCurrentPanel( )->filename;
        default_filename = "coords_" + mrc_name.GetName( ) + ".txt";
        temp_filename    = default_filename;
        int temp_int     = 1;

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
            path                        = save_dialog->GetPath( );
            remember_path               = save_dialog->GetDirectory( );
            wxTextFile* new_coords_file = new wxTextFile(path);
            for ( int i = 0; i < cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->number_of_coords; i++ ) {
                new_coords_file->AddLine(wxString::Format("%li %li %li", cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->coords[i].x_pos, cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->coords[i].y_pos, cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->coords[i].image_number));
            }
            new_coords_file->Write( );
            new_coords_file->Close( );
        }
    }
    // Track the currently opened file for saving in case user makes further selections
    cisTEMDisplayPanel->ReturnCurrentPanel( )->short_txt_filename = default_filename;
    cisTEMDisplayPanel->ReturnCurrentPanel( )->current_file_path  = path;
    cisTEMDisplayPanel->ReturnCurrentPanel( )->have_txt_filename  = true;
    cisTEMDisplayPanel->SetTabNameSaved( );
}

void DisplayFrame::OnInvertSelectionClick(wxCommandEvent& event) {
    for ( long image_counter = 1; image_counter <= cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnNumberofImages( ); image_counter++ ) {
        cisTEMDisplayPanel->ToggleImageSelected(image_counter, false);
    }
    cisTEMDisplayPanel->RefreshCurrentPanel( );
    cisTEMDisplayPanel->SetTabNameUnsaved( );
}

void DisplayFrame::OnClearSelectionClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled ) {
        for ( int image_counter = 0; image_counter < cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnNumberofImages( ); image_counter++ ) {
            cisTEMDisplayPanel->ClearSelection(false);
        }
    }
    else if ( ! cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled ) {
        cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->Clear( );
    }
    ClearTextFileFromPanel( );
    cisTEMDisplayPanel->RefreshCurrentPanel( );
}

void DisplayFrame::OnSize3(wxCommandEvent& event) {
    cisTEMDisplayPanel->ReturnCurrentPanel( )->selected_point_size = 3;
    Refresh( );
    Update( );
}

void DisplayFrame::OnSize5(wxCommandEvent& event) {
    cisTEMDisplayPanel->ReturnCurrentPanel( )->selected_point_size = 5;
    Refresh( );
    Update( );
}

void DisplayFrame::OnSize7(wxCommandEvent& event) {
    cisTEMDisplayPanel->ReturnCurrentPanel( )->selected_point_size = 7;
    Refresh( );
    Update( );
}

void DisplayFrame::OnSize10(wxCommandEvent& event) {
    cisTEMDisplayPanel->ReturnCurrentPanel( )->selected_point_size = 10;
    Refresh( );
    Update( );
}

void DisplayFrame::OnSingleImageModeClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->single_image ) {
        cisTEMDisplayPanel->ReturnCurrentPanel( )->single_image = false;
    }
    else {
        cisTEMDisplayPanel->ReturnCurrentPanel( )->single_image               = true;
        cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled = false;
    }

    cisTEMDisplayPanel->ReturnCurrentPanel( )->panel_image_has_correct_greys = false;
    cisTEMDisplayPanel->ReturnCurrentPanel( )->ReDrawPanel( );
    Refresh( );
    Update( );
}

void DisplayFrame::OnShowSelectionDistancesClick(wxCommandEvent& event) {
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->show_selection_distances )
        cisTEMDisplayPanel->ReturnCurrentPanel( )->show_selection_distances = false;
    else
        cisTEMDisplayPanel->ReturnCurrentPanel( )->show_selection_distances = true;
    Refresh( );
    Update( );
}

void DisplayFrame::OnShowResolution(wxCommandEvent& event) {
    double wanted_pixel_size;

    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->resolution_instead_of_radius )
        cisTEMDisplayPanel->ReturnCurrentPanel( )->resolution_instead_of_radius = false;
    else {
        wxTextEntryDialog text_dialog(this, wxT("Pixel Size (Angstroms)"), wxT("Select Pixel Size"), wxString::Format(wxT("%.2f"), cisTEMDisplayPanel->ReturnCurrentPanel( )->pixel_size), wxOK | wxCANCEL | wxCENTRE, wxDefaultPosition);
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

void DisplayFrame::OnDisplayControlsClick(wxCommandEvent& event) {
    // 1. Create a simple scroll window dialog
    wxDialog* manual_dialog = new wxDialog(this, wxID_ANY, "cisTEM Display Manual", wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, "cisTEM Display Manual");

    // 2. Populate the dialog with text that explains the display controls
    wxScrolledWindow* scrolled_window = new wxScrolledWindow(manual_dialog, wxID_ANY);
    wxBoxSizer*       content_sizer   = new wxBoxSizer(wxVERTICAL);

    wxRichTextCtrl* text_ctrl = new wxRichTextCtrl(scrolled_window, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE | wxTE_READONLY);

    text_ctrl->BeginFontSize(16);
    text_ctrl->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
    text_ctrl->BeginBold( );
    text_ctrl->WriteText("cisTEM Display Manual\n\n");
    text_ctrl->EndBold( );
    text_ctrl->EndAlignment( );
    text_ctrl->EndFontSize( );

    text_ctrl->BeginFontSize(12);
    text_ctrl->BeginBold( );
    text_ctrl->WriteText("\nKeyboard Shortcuts\n\n");
    text_ctrl->EndBold( );
    text_ctrl->EndFontSize( );

    text_ctrl->BeginBold( );
    text_ctrl->WriteText("\n1. Left Arrow Key");
    text_ctrl->EndBold( );
    text_ctrl->WriteText(": Scroll to the previous open image tab.\n");
    text_ctrl->BeginBold( );
    text_ctrl->WriteText("2. Right Arrow Key");
    text_ctrl->EndBold( );
    text_ctrl->WriteText(": Scroll to the next open image tab.\n");
    text_ctrl->BeginBold( );
    text_ctrl->WriteText("3. Up Arrow Key");
    text_ctrl->EndBold( );
    text_ctrl->WriteText(": Scroll to the next section of images/slices that will fit within the current display window.\n");
    text_ctrl->BeginBold( );
    text_ctrl->WriteText("4. Down Arrow Key");
    text_ctrl->EndBold( );
    text_ctrl->WriteText(": Scroll to the previous section of images/slices that will fit within the current display window.\n");

    text_ctrl->BeginBold( );
    text_ctrl->BeginFontSize(12);
    text_ctrl->WriteText("\nMouse Controls\n\n");
    text_ctrl->EndFontSize( );
    text_ctrl->WriteText("\n1. Left Mouse Button");
    text_ctrl->EndBold( );
    text_ctrl->WriteText(": Select or deselect images or coordinates, depending on the current picking mode (found within the Select menu).\n");
    text_ctrl->BeginBold( );
    text_ctrl->WriteText("2. Right Mouse Button");
    text_ctrl->EndBold( );
    text_ctrl->WriteText(": Create a zoomed/upscaled subwindow that will display a more detailed view of the image contents below the mouse position. This can be dragged to view different areas of the image.\n");
    text_ctrl->BeginBold( );
    text_ctrl->WriteText("3. Middle Mouse Button");
    text_ctrl->EndBold( );
    text_ctrl->WriteText(": When in Single Image Mode (selected from the Options menu), dragging the mouse will shift the displayed window in the direction of the drag.");

    wxStdDialogButtonSizer* button_sizer = new wxStdDialogButtonSizer( );
    wxButton*               ok_button    = new wxButton(manual_dialog, wxID_OK);
    button_sizer->AddButton(ok_button);
    button_sizer->Realize( );

    content_sizer->Add(text_ctrl, 1, wxEXPAND | wxALL, 10);
    scrolled_window->SetSizer(content_sizer);
    scrolled_window->SetScrollRate(5, 5);
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);
    main_sizer->Add(scrolled_window, 1, wxEXPAND);
    main_sizer->Add(button_sizer, 0, wxALIGN_RIGHT | wxALL, 5);
    manual_dialog->SetSizerAndFit(main_sizer);
    manual_dialog->SetMinSize(wxSize(900, 500));

    manual_dialog->Layout( );
    manual_dialog->Show( );
}

// This prevents using buttons when an image or stack is not open to act on
void DisplayFrame::DisableAllToolbarButtons( ) {

    // Open menu only needs close tab disabled
    DisplayCloseTab->Enable(false);
    SaveDisplayedImages->Enable(false);
    SaveDisplayedImagesWithLegend->Enable(false);

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
    OptionsSingleImageMode->Enable(false);
    OptionsShowSelectionDistances->Enable(false);
    OptionsShowResolution->Enable(false);
}

// Call when an image is opened to activate all toolbar buttons
void DisplayFrame::EnableAllToolbarButtons( ) {
    // Open menu only needs close tab disabled
    DisplayCloseTab->Enable( );
    SaveDisplayedImages->Enable( );
    SaveDisplayedImagesWithLegend->Enable( );

    // Label menu
    LabelLocationNumber->Enable( );

    // Select menu
    SelectImageSelectionMode->Enable(true);
    SelectCoordsSelectionMode->Enable(true);
    SelectOpenTxt->Enable(true);
    SelectSaveTxt->Enable(true);
    SelectSaveTxtAs->Enable(true);
    SelectInvertSelection->Enable(true);
    SelectClearSelection->Enable(true);

    // Options menu
    OptionsSingleImageMode->Enable(true);
    OptionsShowSelectionDistances->Enable(true);
    OptionsShowResolution->Enable(true);
}

void DisplayFrame::OnUpdateUI(wxUpdateUIEvent& event) {
    // First, do we have an image open?
    if ( cisTEMDisplayPanel->my_notebook->GetSelection( ) != wxNOT_FOUND ) {
        EnableAllToolbarButtons( );

        // Check that there are coords selected
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->number_of_coords > 0 || cisTEMDisplayPanel->ReturnCurrentPanel( )->number_of_selections > 0 ) {
            SelectSaveTxtAs->Enable(true);
            if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->have_txt_filename )
                SelectSaveTxt->Enable(true);
            else
                SelectSaveTxt->Enable(false);
        }
        else {
            SelectSaveTxtAs->Enable(false);
            SelectSaveTxt->Enable(false);
        }

        // Keep picking mode radio buttons visually current
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->image_picking_mode_enabled ) {
            SelectImageSelectionMode->Check(true);
            SelectInvertSelection->Enable(true);
        }
        else {
            SelectCoordsSelectionMode->Check(true);
            SelectInvertSelection->Enable(false);
        }

        // Make sure correct radio is checked for point size selection submenu
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->selected_point_size == 3 )
            CoordSize3->Check(true);
        else if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->selected_point_size == 5 )
            CoordSize5->Check(true);
        else if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->selected_point_size == 7 )
            CoordSize7->Check(true);
        else if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->selected_point_size == 10 )
            CoordSize10->Check(true);

        // Make sure single image mode is checked/unchecked based on current panel
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->single_image ) {
            if ( ! OptionsSingleImageMode->IsChecked( ) )
                OptionsSingleImageMode->Check(true);
            SelectImageSelectionMode->Enable(false);
        }
        else if ( ! cisTEMDisplayPanel->ReturnCurrentPanel( )->single_image ) {
            if ( OptionsSingleImageMode->IsChecked( ) )
                OptionsSingleImageMode->Check(false);
            SelectImageSelectionMode->Enable(true);
        }

        // Repeat above for res instead of radius
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->resolution_instead_of_radius ) {
            if ( ! OptionsShowResolution->IsChecked( ) ) {
                OptionsShowResolution->Check(true);
            }
        }
        else if ( ! cisTEMDisplayPanel->ReturnCurrentPanel( )->resolution_instead_of_radius ) {
            if ( OptionsShowResolution->IsChecked( ) ) {
                OptionsShowResolution->Check(false);
            }
        }

        // Repeat again for selection distance option
        if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->show_selection_distances ) {
            if ( ! OptionsShowSelectionDistances->IsChecked( ) ) {
                OptionsShowSelectionDistances->Check(true);
            }
        }
        else if ( ! cisTEMDisplayPanel->ReturnCurrentPanel( )->show_selection_distances ) {
            if ( OptionsShowSelectionDistances->IsChecked( ) ) {
                OptionsShowSelectionDistances->Check(false);
            }
        }
    }
    // No image -- don't want buttons active
    else
        DisableAllToolbarButtons( );
}

bool DisplayFrame::LoadCoords(wxString current_line, long& x, long& y, long& image_number) {
    // Parse the string for x, y, and the image number
    int index_of_whitespace      = current_line.find(' ');
    int prev_whitespace_position = 0;
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
    if ( x < cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnImageXSize( ) && y < cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnImageYSize( ) && image_number <= cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnNumberofImages( ) ) {
        cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->ToggleCoord(image_number, x, y);
        return true;
    }
    else {
        wxMessageDialog invalid_file_dialog(this, wxString::Format("The selected coordinates exceed the dimensions of the currently opened *.mrc file. Cannot open selected coordinates.\nSelected x: %li, selected y: %li, image num: %li for image(s) with dimensions x: %i, y: %i, num images: %i).\nTry checking the selection mode and/or the text file contents.", x, y, image_number, cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnImageXSize( ), cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnImageYSize( ), cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnNumberofImages( )), "Invalid Coordinates for Current Image(s)", wxOK | wxOK_DEFAULT | wxICON_EXCLAMATION);
        if ( invalid_file_dialog.ShowModal( ) == wxID_OK )
            cisTEMDisplayPanel->ReturnCurrentPanel( )->coord_tracker->Clear( );
        return false;
    }
}

bool DisplayFrame::LoadImageSelections(wxString current_line) {
    // Quick check of file format
    int index_of_whitespace = current_line.find(' ');
    if ( index_of_whitespace != wxNOT_FOUND ) {
        wxMessageDialog wrong_file_format(this, "Cannot open Coordinate Selection text file in Image Selection mode.", "Incorrect File Format", wxOK | wxOK_DEFAULT | wxICON_EXCLAMATION);
        wrong_file_format.ShowModal( );
        return false;
    }

    // Get the value that's selected
    long image_number;
    current_line.ToLong(&image_number);

    if ( image_number <= cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnNumberofImages( ) ) {
        cisTEMDisplayPanel->ReturnCurrentPanel( )->SetImageSelected(image_number, false);
        return true;
    }
    // If the value exceeds the possible dimensions don't try to access the index for setting selected
    else {
        wxMessageDialog invalid_file_dialog(this, wxString::Format("The file being opened contains selected images that exceed the number of images in the current file. Cannot open the selections.(Images in open file: %i. Image index sought: %li)", cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnNumberofImages( ), image_number), "Invalid Selection(s) for Current Image(s)", wxOK | wxOK_DEFAULT | wxICON_EXCLAMATION);
        cisTEMDisplayPanel->ClearSelection(false);
        return false;
    }
}

void DisplayFrame::ClearTextFileFromPanel( ) {
    cisTEMDisplayPanel->ReturnCurrentPanel( )->have_txt_filename  = false;
    cisTEMDisplayPanel->ReturnCurrentPanel( )->short_txt_filename = wxEmptyString;
    cisTEMDisplayPanel->ReturnCurrentPanel( )->current_file_path  = wxEmptyString;
    cisTEMDisplayPanel->SetTabNameSaved( );
}

/**
 * @brief Crops the current image at the borders to remove excess blank space aroudn the image(s) being displayed.
 * 
 * @return wxBitmap The cropped bitmap ready for saving.
 */
wxBitmap DisplayFrame::CropImageForSaving( ) {

    // TODO: must also account for the case of a single image being displayed but not being in single image mode;
    // failure to do so causes the saved image to have excessively large legend (speicfically in terms of legend height)
    // because the image is small but the legend is sized for the full panel.
    wxBitmap sub_bitmap;
    int      sub_bmp_width;
    int      sub_bmp_height;
    int      single_image_x = cisTEMDisplayPanel->ReturnCurrentPanel( )->single_image_x;
    int      single_image_y = cisTEMDisplayPanel->ReturnCurrentPanel( )->single_image_y;
    float    scale_factor   = cisTEMDisplayPanel->ReturnCurrentPanel( )->actual_scale_factor;
    if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->single_image ) {
        cisTEMDisplayPanel->ReturnCurrentPanel( )->GetClientSize(&sub_bmp_width, &sub_bmp_height);
        if ( single_image_x * scale_factor + sub_bmp_width > cisTEMDisplayPanel->ReturnCurrentPanel( )->panel_image->GetWidth( ) ) {
            sub_bmp_width = cisTEMDisplayPanel->ReturnCurrentPanel( )->panel_image->GetWidth( ) - single_image_x * scale_factor;
        }
        if ( single_image_y * scale_factor + sub_bmp_height > cisTEMDisplayPanel->ReturnCurrentPanel( )->panel_image->GetHeight( ) ) {
            sub_bmp_height = cisTEMDisplayPanel->ReturnCurrentPanel( )->panel_image->GetHeight( ) - single_image_y * scale_factor;
        }
        wxRect  sub_bmp_dims(single_image_x * scale_factor, single_image_y * scale_factor, sub_bmp_width, sub_bmp_height);
        wxImage tmp_sub_img(cisTEMDisplayPanel->ReturnCurrentPanel( )->panel_image->GetSubImage(sub_bmp_dims));
        sub_bitmap = wxBitmap(tmp_sub_img);
    }
    else {
        int num_rows_with_imgs = cisTEMDisplayPanel->ReturnCurrentPanel( )->images_in_current_view / cisTEMDisplayPanel->ReturnCurrentPanel( )->images_in_x;

        // if columns_in_x is 0, then we have less than one full row of images, the number of rows shown is 1
        // if columns_in_x is > 0, then we have at least one full row of images, and the number of rows shown is either 1 or more;
        // we can check if it's more than one by using modulus; if it's 0, then all rows are filled, otherwise we have a partial row
        // and must increment by 1.
        if ( num_rows_with_imgs > 0 ) {
            // We have a partial row, so increment filled_rows by 1
            if ( cisTEMDisplayPanel->ReturnCurrentPanel( )->images_in_current_view % cisTEMDisplayPanel->ReturnCurrentPanel( )->images_in_x != 0 ) {
                num_rows_with_imgs++;
            }
        }
        else {
            num_rows_with_imgs = 1;
        }

        sub_bmp_width  = cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnImageXSize( ) * cisTEMDisplayPanel->ReturnCurrentPanel( )->actual_scale_factor * cisTEMDisplayPanel->ReturnCurrentPanel( )->images_in_x;
        sub_bmp_height = cisTEMDisplayPanel->ReturnCurrentPanel( )->ReturnImageYSize( ) * cisTEMDisplayPanel->ReturnCurrentPanel( )->actual_scale_factor * num_rows_with_imgs;
        wxRect sub_bmp_dims(single_image_x * scale_factor, single_image_y * scale_factor, sub_bmp_width, sub_bmp_height);
        sub_bitmap = cisTEMDisplayPanel->ReturnCurrentPanel( )->panel_bitmap.GetSubBitmap(sub_bmp_dims);
    }
    return sub_bitmap;
}