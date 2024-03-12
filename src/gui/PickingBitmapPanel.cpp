#include "../core/gui_core_headers.h"

int compare_particle_position_asset_using_x(ParticlePositionAsset** first, ParticlePositionAsset** second) {
    if ( first[0]->x_position < second[0]->x_position ) {
        return -1;
    }
    else {
        if ( first[0]->x_position > second[0]->x_position )
            return 1;
        return 0;
    }
}

#include <wx/arrimpl.cpp>
WX_DEFINE_OBJARRAY(ArrayOfCoordinatesHistory);

PickingBitmapPanel::PickingBitmapPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
    : wxPanel(parent, id, pos, size, style, name) {

    // create a 1 pixel white bitmap

    PanelBitmap.Create(1, 1, 24);
    panel_text = "";

    should_show          = false;
    size_is_dirty        = true;
    font_size_multiplier = 1.0;
    Bind(wxEVT_PAINT, &PickingBitmapPanel::OnPaint, this);
    Bind(wxEVT_ERASE_BACKGROUND, &PickingBitmapPanel::OnEraseBackground, this);
    Bind(wxEVT_SIZE, &PickingBitmapPanel::OnSize, this);
    Bind(wxEVT_LEFT_DOWN, &PickingBitmapPanel::OnLeftDown, this);
    Bind(wxEVT_LEFT_UP, &PickingBitmapPanel::OnLeftUp, this);
    Bind(wxEVT_RIGHT_DOWN, &PickingBitmapPanel::OnRightDown, this);
    Bind(wxEVT_RIGHT_UP, &PickingBitmapPanel::OnRightUp, this);
    Bind(wxEVT_MIDDLE_UP, &PickingBitmapPanel::OnMiddleUp, this);
    Bind(wxEVT_MOTION, &PickingBitmapPanel::OnMotion, this);

    image_in_bitmap_filename       = "";
    image_in_bitmap_pixel_size     = 0.0;
    image_in_bitmap_scaling_factor = 0.0;

    image_in_memory_filename   = "";
    image_in_memory_pixel_size = 0.0;

    draw_circles_around_particles = true;
    //number_of_particles = 0;
    radius_of_circles_around_particles_in_angstroms         = 0.0;
    squared_radius_of_circles_around_particles_in_angstroms = 0.0;

    bitmap_height   = 0;
    bitmap_width    = 0;
    bitmap_x_offset = 0;
    bitmap_y_offset = 0;

    draw_scale_bar       = true;
    should_high_pass     = true;
    should_low_pass      = false;
    should_wiener_filter = true;
    popup_exists         = false;

    allow_editing_of_coordinates = true;

    draw_selection_rectangle                  = false;
    doing_shift_delete                        = false;
    selection_rectangle_current_x             = 0;
    selection_rectangle_current_y             = 0;
    selection_rectangle_start_x               = 0;
    selection_rectangle_start_y               = 0;
    selection_rectangle_start_x_in_angstroms  = 0.0;
    selection_rectangle_start_y_in_angstroms  = 0.0;
    selection_rectangle_finish_x_in_angstroms = 0.0;
    selection_rectangle_finish_y_in_angstroms = 0.0;
    clicked_point_x                           = 0;
    clicked_point_y                           = 0;
    clicked_point_x_in_angstroms              = 0.0;
    clicked_point_y_in_angstroms              = 0.0;
    current_step_in_history                   = 0;
    scale_factor                              = 1.0;

    low_res_filter_value  = -1.0;
    high_res_filter_value = -1.0;
}

PickingBitmapPanel::~PickingBitmapPanel( ) {
    Unbind(wxEVT_PAINT, &PickingBitmapPanel::OnPaint, this);
    Unbind(wxEVT_ERASE_BACKGROUND, &PickingBitmapPanel::OnEraseBackground, this);
    Unbind(wxEVT_SIZE, &PickingBitmapPanel::OnSize, this);
    Unbind(wxEVT_LEFT_DOWN, &PickingBitmapPanel::OnLeftDown, this);
    Unbind(wxEVT_LEFT_UP, &PickingBitmapPanel::OnLeftUp, this);
    Unbind(wxEVT_RIGHT_UP, &PickingBitmapPanel::OnRightDown, this);
    Unbind(wxEVT_RIGHT_DOWN, &PickingBitmapPanel::OnRightUp, this);
    Unbind(wxEVT_MIDDLE_UP, &PickingBitmapPanel::OnMiddleUp, this);
    Unbind(wxEVT_MOTION, &PickingBitmapPanel::OnMotion, this);
}

void PickingBitmapPanel::Clear( ) {
    Freeze( );
    wxClientDC dc(this);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear( );
    image_in_memory.Deallocate( );
    image_in_bitmap.Deallocate( );
    particle_coordinates_in_angstroms.Empty( );
    EmptyHistoryOfParticleCoordinates( );
    Thaw( );
}

void PickingBitmapPanel::EmptyHistoryOfParticleCoordinates( ) {
    extern MyPickingResultsPanel* picking_results_panel;

    for ( size_t counter = 0; counter < particle_coordinates_in_angstroms_history.GetCount( ); counter++ ) {
        particle_coordinates_in_angstroms_history.Item(counter).Empty( );
    }
    particle_coordinates_in_angstroms_history.Empty( );
    picking_results_panel->ResultDisplayPanel->UndoButton->Enable(false);
    picking_results_panel->ResultDisplayPanel->RedoButton->Enable(false);
    picking_results_panel->ResultDisplayPanel->SetNumberOfPickedCoordinates(particle_coordinates_in_angstroms.Count( ));
}

void PickingBitmapPanel::OnEraseBackground(wxEraseEvent& event) {
}

void PickingBitmapPanel::ResetHistory( ) {
    // Empty the history
    EmptyHistoryOfParticleCoordinates( );
    particle_coordinates_in_angstroms_history.Add(particle_coordinates_in_angstroms);
    current_step_in_history = 0;
}

void PickingBitmapPanel::SetParticleCoordinatesAndRadius(const ArrayOfParticlePositionAssets& array_of_assets, const float wanted_radius_in_angstroms) {
    extern MyPickingResultsPanel* picking_results_panel;

    particle_coordinates_in_angstroms                       = array_of_assets;
    radius_of_circles_around_particles_in_angstroms         = wanted_radius_in_angstroms;
    squared_radius_of_circles_around_particles_in_angstroms = powf(wanted_radius_in_angstroms, 2);
    picking_results_panel->ResultDisplayPanel->SetNumberOfPickedCoordinates(particle_coordinates_in_angstroms.Count( ));

    ResetHistory( );
}

int PickingBitmapPanel::RemoveParticleCoordinatesWithinRectangleOrNearClickedPoint( ) {
    extern MyPickingResultsPanel* picking_results_panel;
    Freeze( );
    SetCurrentAsLastStepInHistoryOfParticleCoordinates( );
    int number_of_removed_particles = 0;
    if ( particle_coordinates_in_angstroms.GetCount( ) > 0 ) {
        size_t counter = particle_coordinates_in_angstroms.GetCount( ) - 1;
        if ( draw_selection_rectangle ) {
            while ( true ) {
                if ( ParticleCoordinatesAreWithinRectangle(particle_coordinates_in_angstroms.Item(counter)) ) {
                    particle_coordinates_in_angstroms.RemoveAt(counter);
                    number_of_removed_particles++;
                }
                if ( counter == 0 )
                    break;
                counter--;
            }
        }
        else {
            while ( true ) {
                if ( ParticleCoordinatesAreNearClickedPoint(particle_coordinates_in_angstroms.Item(counter)) ) {
                    particle_coordinates_in_angstroms.RemoveAt(counter);
                    number_of_removed_particles++;
                }
                if ( counter == 0 )
                    break;
                counter--;
            }
        }
    }
    if ( number_of_removed_particles > 0 ) {
        particle_coordinates_in_angstroms_history.Add(particle_coordinates_in_angstroms);
        particle_coordinates_in_angstroms_history.Last( ).Shrink( );
        current_step_in_history++;
        picking_results_panel->ResultDisplayPanel->UndoButton->Enable(true);
    }
    picking_results_panel->ResultDisplayPanel->SetNumberOfPickedCoordinates(particle_coordinates_in_angstroms.Count( ));
    Thaw( );
    return number_of_removed_particles;
}

bool PickingBitmapPanel::ParticleCoordinatesAreWithinRectangle(const ParticlePositionAsset& particle_coordinates_to_check) {
    return particle_coordinates_to_check.x_position >= selection_rectangle_start_x_in_angstroms &&
           particle_coordinates_to_check.x_position <= selection_rectangle_finish_x_in_angstroms &&
           particle_coordinates_to_check.y_position >= selection_rectangle_start_y_in_angstroms &&
           particle_coordinates_to_check.y_position <= selection_rectangle_finish_y_in_angstroms;
}

bool PickingBitmapPanel::ParticleCoordinatesAreNearClickedPoint(const ParticlePositionAsset& particle_coordinates_to_check) {
    return squared_radius_of_circles_around_particles_in_angstroms >= powf(particle_coordinates_to_check.x_position - clicked_point_x_in_angstroms, 2) + powf(particle_coordinates_to_check.y_position - clicked_point_y_in_angstroms, 2);
}

void PickingBitmapPanel::SetCurrentAsLastStepInHistoryOfParticleCoordinates( ) {
    extern MyPickingResultsPanel* picking_results_panel;
    if ( current_step_in_history < particle_coordinates_in_angstroms_history.GetCount( ) - 1 ) {
        particle_coordinates_in_angstroms_history.RemoveAt(current_step_in_history + 1, particle_coordinates_in_angstroms_history.GetCount( ) - current_step_in_history - 1);
    }
    // Disable the redo button here
    picking_results_panel->ResultDisplayPanel->RedoButton->Enable(false);
}

void PickingBitmapPanel::StepForwardInHistoryOfParticleCoordinates( ) {
    extern MyPickingResultsPanel* picking_results_panel;
    current_step_in_history++;
    particle_coordinates_in_angstroms = particle_coordinates_in_angstroms_history.Item(current_step_in_history);
    if ( current_step_in_history == particle_coordinates_in_angstroms_history.GetCount( ) - 1 ) {
        picking_results_panel->ResultDisplayPanel->RedoButton->Enable(false);
    }
    picking_results_panel->ResultDisplayPanel->UndoButton->Enable(true);
    picking_results_panel->ResultDisplayPanel->SetNumberOfPickedCoordinates(particle_coordinates_in_angstroms.Count( ));
    Refresh( );
    Update( );
}

void PickingBitmapPanel::StepBackwardInHistoryOfParticleCoordinates( ) {
    extern MyPickingResultsPanel* picking_results_panel;
    MyDebugAssertTrue(current_step_in_history > 0, "Ooops, cannot step back in history when at step %i\n", current_step_in_history);
    current_step_in_history--;
    particle_coordinates_in_angstroms = particle_coordinates_in_angstroms_history.Item(current_step_in_history);
    if ( current_step_in_history == 0 ) {
        picking_results_panel->ResultDisplayPanel->UndoButton->Enable(false);
    }
    picking_results_panel->ResultDisplayPanel->RedoButton->Enable(true);
    picking_results_panel->ResultDisplayPanel->SetNumberOfPickedCoordinates(particle_coordinates_in_angstroms.Count( ));
    Refresh( );
    Update( );
}

void PickingBitmapPanel::SetImageFilename(wxString wanted_filename, const float& pixel_size, CTF ctf_of_image) {
    if ( ! wanted_filename.IsSameAs(image_in_memory_filename) ) {
        image_in_memory.QuickAndDirtyReadSlice(wanted_filename.ToStdString( ), 1);
        image_in_memory_pixel_size = pixel_size;
        image_in_memory_filename   = wanted_filename;
        SetCTFOfImageInMemory(ctf_of_image);
    }
}

void PickingBitmapPanel::SetCTFOfImageInMemory(CTF ctf_to_copy) {
    image_in_memory_ctf.CopyFrom(ctf_to_copy);
}

void PickingBitmapPanel::SetCTFOfImageInBitmap(CTF ctf_to_copy) {
    image_in_bitmap_ctf.CopyFrom(ctf_to_copy);
}

// EXPLANATION: This function pre-determines the size of the image to be 95% of the size of the allowed dimension (whichever is smaller)
// Since the dimension can vary, it opts for whichever scaling factor is smaller
void PickingBitmapPanel::UpdateScalingAndDimensions( ) {
    wxPrintf("Entered PickingBitmapPanel::UpdateScalingAndDimensions\n");
    if ( ! image_in_memory_filename.IsEmpty( ) ) {
        int panel_dim_x, panel_dim_y;
        GetClientSize(&panel_dim_x, &panel_dim_y);

        // TODO: is this where to add the scaling? Mutliply panel_dims * .95 by the scale factor
        float target_scaling_x = float(panel_dim_x) * 0.95 * scale_factor / float(image_in_memory.logical_x_dimension);
        float target_scaling_y = float(panel_dim_y) * 0.95 * scale_factor / float(image_in_memory.logical_y_dimension);
        float scaling_factor   = std::min(target_scaling_x, target_scaling_y);
        wxPrintf("scaling_factor = %f\n", scaling_factor);

        int new_x_dimension = std::max(int(float(image_in_memory.logical_x_dimension) * scaling_factor), 1);
        int new_y_dimension = std::max(int(float(image_in_memory.logical_y_dimension) * scaling_factor), 1);
        wxPrintf("new_x_dimension = %i, new_y_dimension = %i\n", new_x_dimension, new_y_dimension);

        // TODO: choose dimensions that are more favorable to FFT

        if ( ! image_in_bitmap.is_in_memory || new_x_dimension != image_in_bitmap.logical_x_dimension || new_y_dimension != image_in_bitmap.logical_y_dimension ) {
            image_in_bitmap.Allocate(new_x_dimension, new_y_dimension, true);
            image_in_bitmap_scaling_factor = scaling_factor;
            image_in_bitmap_pixel_size     = image_in_memory_pixel_size / scaling_factor;
            image_in_bitmap_ctf.CopyFrom(image_in_memory_ctf);
            image_in_bitmap_ctf.ChangePixelSize(image_in_memory_pixel_size, image_in_bitmap_pixel_size);
        }
    }
}

// FIXME: This is probably where I need to add the scaling changes code
// Might be better to start over at this point, and get a better understanding.
void PickingBitmapPanel::UpdateImageInBitmap(bool force_reload) {
    wxPrintf("Entering PickingBitmapPanel::UpdateImageInBitmap\n");
    if ( ! image_in_memory_filename.IsEmpty( ) ) {
        if ( force_reload || ! image_in_bitmap_filename.IsSameAs(image_in_memory_filename) || PanelBitmap.GetWidth( ) != image_in_bitmap.logical_x_dimension || PanelBitmap.GetHeight( ) != image_in_bitmap.logical_y_dimension ) {
            if ( image_in_memory.is_in_real_space )
                image_in_memory.ForwardFFT( );
            image_in_bitmap.is_in_real_space = false;
            image_in_memory.ClipInto(&image_in_bitmap);

            //const float filter_edge_width = 0.2;//image_in_bitmap_pixel_size / (2.0 * radius_of_circles_around_particles_in_angstroms);

            float high_pass_radius;
            float low_pass_radius;

            if ( high_res_filter_value < 0 )
                high_pass_radius = 8.0 / float(image_in_bitmap.logical_x_dimension);
            else
                high_pass_radius = image_in_bitmap_pixel_size / high_res_filter_value;

            if ( low_res_filter_value < 0 )
                low_pass_radius = image_in_bitmap_pixel_size / 20; //(0.5 * radius_of_circles_around_particles_in_angstroms);
            else
                low_pass_radius = image_in_bitmap_pixel_size / low_res_filter_value;

            if ( should_high_pass ) {
                image_in_bitmap.BackwardFFT( );
                image_in_bitmap.TaperEdges( );
                image_in_bitmap.ForwardFFT( );
                image_in_bitmap.CosineMask(high_pass_radius, high_pass_radius * 2.0, true);
            }

            if ( should_low_pass ) {
                image_in_bitmap.GaussianLowPassFilter(low_pass_radius * sqrt(2.0));
            }

            if ( should_wiener_filter ) {
                image_in_bitmap.OptimalFilterWarp(image_in_bitmap_ctf, image_in_bitmap_pixel_size);
            }

            image_in_bitmap.BackwardFFT( );
            image_in_bitmap_filename = image_in_memory_filename;

                        //wxImage frame_image = new wxImage(image_in_bitmap);

            ConvertImageToBitmap(&image_in_bitmap, &PanelBitmap, true);
        }
    }
}

void PickingBitmapPanel::OnSize(wxSizeEvent& event) {
    size_is_dirty = true;
    event.Skip( );
}

void PickingBitmapPanel::OnPaint(wxPaintEvent& evt) {

    //Freeze();

    int window_x_size;
    int window_y_size;

    wxPaintDC dc(this);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear( );
    GetClientSize(&window_x_size, &window_y_size);
    dc.SetBrush(wxNullBrush);
    dc.DrawRectangle(0, 0, window_x_size, window_y_size);

    if ( should_show ) {

        if ( size_is_dirty ) {
            UpdateScalingAndDimensions( );
            UpdateImageInBitmap( );
            size_is_dirty = false;
        }

        wxFont current_font = dc.GetFont( );
        current_font.Scale(font_size_multiplier);
        dc.SetFont(current_font);

        int text_x_size;
        int text_y_size;

        int combined_width;
        int combined_height;

        int text_y_offset;

        //float scale_factor;

        int x_oversize;
        int y_oversize;

        GetClientSize(&window_x_size, &window_y_size);
        bitmap_width  = PanelBitmap.GetWidth( );
        bitmap_height = PanelBitmap.GetHeight( );

        if ( panel_text.IsEmpty( ) == true ) {
            text_x_size = 0;
            text_y_size = 0;
        }
        else
            GetMultilineTextExtent(&dc, panel_text, text_x_size, text_y_size);

        combined_width  = bitmap_width + text_x_size;
        combined_height = bitmap_height;

        // FIXME: We're changing this; the image does not always have to fit in the window
        //if ( combined_width > window_x_size || combined_height > window_y_size ) {
        //    MyDebugAssertTrue(false, "Oops, should not get here, because the image should always fit in\n");
        //}
        //else {
        bitmap_x_offset = (window_x_size - ((bitmap_width) + text_x_size)) / 2;
        bitmap_y_offset = (window_y_size - (bitmap_height)) / 2;

        // Test offset can't be less than 0:
        if ( bitmap_x_offset < 0 )
            bitmap_x_offset = 0;
        if ( bitmap_y_offset < 0 )
            bitmap_y_offset = 0;

        // DEBUG:
        wxPrintf("bitmap_x_offset = %i; bitmap_y_offset = %i\n", bitmap_x_offset, bitmap_y_offset);

        // Draw the image bitmap
        dc.DrawBitmap(PanelBitmap, bitmap_x_offset, bitmap_y_offset, false); // Here, bitmap_x_offset and bitmap_y_offset are the starting coords for the iamge when drawn into the bitmap...

        // Choose a pen thickness for drawing circles around particles
        int pen_thickness = std::min(bitmap_width, bitmap_height) / 512;
        if ( pen_thickness < 1 )
            pen_thickness = 1;
        if ( pen_thickness > 5 )
            pen_thickness = 5;

        // Draw circles around particles
        if ( draw_circles_around_particles ) {
            float x, y;
            dc.SetPen(wxPen(wxColor(255, 0, 0), pen_thickness));
            dc.SetBrush(wxNullBrush);
            for ( int counter = 0; counter < particle_coordinates_in_angstroms.GetCount( ); counter++ ) {
                x = particle_coordinates_in_angstroms.Item(counter).x_position;
                y = particle_coordinates_in_angstroms.Item(counter).y_position;
                dc.DrawCircle(bitmap_x_offset + x / image_in_bitmap_pixel_size, bitmap_y_offset + bitmap_height - y / image_in_bitmap_pixel_size, radius_of_circles_around_particles_in_angstroms / image_in_bitmap_pixel_size);
            }
        }

        // Draw scale bar
        if ( draw_scale_bar ) {
            wxPen scalebar_pen;
            scalebar_pen = wxPen(*wxWHITE);
            dc.SetPen(scalebar_pen);
            dc.SetBrush(*wxWHITE_BRUSH);
            int scalebar_length;
            {
                const float bar_must_be_multiple_of = 5.0; //nm
                float       ideal_length_in_pixels  = float(bitmap_width) * 0.1;
                float       ideal_length_in_nm      = ideal_length_in_pixels * image_in_bitmap_pixel_size * 0.1;
                ideal_length_in_nm                  = roundf(ideal_length_in_nm / bar_must_be_multiple_of) * bar_must_be_multiple_of;
                ideal_length_in_pixels              = ideal_length_in_nm * 10.0 / image_in_bitmap_pixel_size;
                scalebar_length                     = myroundint(ideal_length_in_pixels);
            }
            int scalebar_x_start   = int(float(bitmap_width) * 0.85);
            int scalebar_y_pos     = int(float(bitmap_height) * 0.95);
            int scalebar_thickness = int(float(bitmap_height) / 50.0);
            dc.DrawRectangle(bitmap_x_offset + scalebar_x_start, bitmap_y_offset + scalebar_y_pos, scalebar_length, scalebar_thickness);
            //dc.SetPen( *wxRED_PEN );
            dc.SetTextForeground(*wxWHITE);
            dc.SetFont(*wxNORMAL_FONT);
            dc.SetFont(wxFont(std::max(12, int(float(scalebar_thickness) * 0.75)), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));
            wxString scalebar_label = wxString::Format("%.0f nm", float(scalebar_length) * image_in_bitmap_pixel_size * 0.1);
            int      scalebar_label_width;
            int      scalebar_label_height;
            dc.GetTextExtent(scalebar_label, &scalebar_label_width, &scalebar_label_height);
            dc.DrawText(scalebar_label, bitmap_x_offset + scalebar_x_start + scalebar_length / 2 - scalebar_label_width / 2, bitmap_y_offset + scalebar_y_pos - scalebar_label_height - scalebar_thickness / 8);
        }

        // Draw selection retangle
        if ( draw_selection_rectangle ) {
            dc.SetPen(wxPen(wxColor(255, 0, 0), pen_thickness * 2, wxPENSTYLE_LONG_DASH));
            dc.SetBrush(wxNullBrush);
            dc.DrawRectangle(std::min(selection_rectangle_start_x, selection_rectangle_current_x), std::min(selection_rectangle_start_y, selection_rectangle_current_y), abs(selection_rectangle_current_x - selection_rectangle_start_x), abs(selection_rectangle_current_y - selection_rectangle_start_y));
        }

        // Draw text

        text_y_offset = ((bitmap_height)-text_y_size) / 2;
        if ( text_y_offset > 0 )
            dc.DrawText(panel_text, bitmap_width + bitmap_x_offset, text_y_offset + bitmap_y_offset);
        else
            dc.DrawText(panel_text, bitmap_width + bitmap_x_offset, bitmap_y_offset);
        //}
    }

    //	Thaw();
}

void PickingBitmapPanel::SetupPanelBitmap( ) {
    int window_x_size;
    int window_y_size;

    int x_padding;
    int y_padding;

    GetClientSize(&window_x_size, &window_y_size);

    if ( window_x_size != PanelBitmap.GetWidth( ) || window_y_size != PanelBitmap.GetHeight( ) ) {
        PanelBitmap.Create(window_x_size, window_y_size, 24);
    }
}

void PickingBitmapPanel::OnLeftDown(wxMouseEvent& event) {
    if ( should_show == true ) {
        int x_pos, y_pos;
        event.GetPosition(&x_pos, &y_pos);
        if ( event.ControlDown( ) ) {
            draw_selection_rectangle    = true;
            selection_rectangle_start_x = x_pos;
            selection_rectangle_start_y = y_pos;
        }
        else if ( event.ShiftDown( ) ) {
            doing_shift_delete = true;
        }
    }
    event.Skip( );
}

/*
 * Conversion to/from angstrom and bitmap pixels
 *
 * x_pix = x_offset + bitmap_width - x_ang / psize;
 * x_ang = (x_offset + bitmap_width - x_pix) * psize;
 *
 * y_pix = y_offset + y_ang / psize;
 * y_ang = (y_pix - y_offset) * psize;
 *
 */

void PickingBitmapPanel::OnLeftUp(wxMouseEvent& event) {
    extern MyPickingResultsPanel* picking_results_panel;

    if ( should_show == true ) {
        if ( doing_shift_delete == true ) {
            doing_shift_delete = false;
        }
        else if ( draw_selection_rectangle ) {
            // Convert begin and end coordinates to Angstroms
            selection_rectangle_start_x_in_angstroms  = PixelToAngstromX(std::min(selection_rectangle_start_x, selection_rectangle_current_x));
            selection_rectangle_finish_x_in_angstroms = PixelToAngstromX(std::max(selection_rectangle_start_x, selection_rectangle_current_x));
            selection_rectangle_start_y_in_angstroms  = PixelToAngstromY(std::max(selection_rectangle_start_y, selection_rectangle_current_y));
            selection_rectangle_finish_y_in_angstroms = PixelToAngstromY(std::min(selection_rectangle_start_y, selection_rectangle_current_y));

            if ( allow_editing_of_coordinates )
                RemoveParticleCoordinatesWithinRectangleOrNearClickedPoint( );
            draw_selection_rectangle = false;
        }
        else {
            event.GetPosition(&clicked_point_x, &clicked_point_y);
            if ( clicked_point_x > bitmap_x_offset && clicked_point_x < bitmap_x_offset + bitmap_width && clicked_point_y > bitmap_y_offset && clicked_point_y < bitmap_y_offset + bitmap_height ) {
                clicked_point_x_in_angstroms = PixelToAngstromX(clicked_point_x);
                clicked_point_y_in_angstroms = PixelToAngstromY(clicked_point_y);
                if ( allow_editing_of_coordinates ) {
                    int number_of_removed_coordinates = RemoveParticleCoordinatesWithinRectangleOrNearClickedPoint( );
                    if ( number_of_removed_coordinates == 0 ) {
                        // The user clicked to add a particle
                        particle_coordinates_in_angstroms.Add(ParticlePositionAsset(clicked_point_x_in_angstroms, clicked_point_y_in_angstroms));
                        SetCurrentAsLastStepInHistoryOfParticleCoordinates( );
                        particle_coordinates_in_angstroms_history.Add(particle_coordinates_in_angstroms);
                        current_step_in_history++;
                        picking_results_panel->ResultDisplayPanel->UndoButton->Enable(true);
                        picking_results_panel->ResultDisplayPanel->SetNumberOfPickedCoordinates(particle_coordinates_in_angstroms.Count( ));
                    }
                }
            }
        }
        Refresh( );
        Update( );
    }
    event.Skip( );
}

// This function doesn't actually do any of the hard stuff; we will always want scaled image here, and we don't
// have to worry specifically about auto or manual or global grays
// What may be important is the Wiener filter, other checkboxes to the right of the bitmap
PickingBitmapPanelPopup::PickingBitmapPanelPopup(wxWindow* parent, int flags) : wxPopupWindow(parent, flags) {
    Bind(wxEVT_PAINT, &PickingBitmapPanelPopup::OnPaint, this);
    Bind(wxEVT_ERASE_BACKGROUND, &PickingBitmapPanelPopup::OnEraseBackground, this);
    SetBackgroundColour(*wxBLACK);

    parent_picking_bitmap_panel = reinterpret_cast<PickingBitmapPanel*>(parent);

    /* In DisplayPanel.cpp, we have to select the specific notebook panel
    and focus on the appropriate image; here, we only have one image on the panel
    at any given time, so we can just focus on only that image for all zooming
    and creating the mini client
    */
}

void PickingBitmapPanelPopup::OnPaint(wxPaintEvent& event) {
    wxPaintDC dc(this);

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
    else if ( sub_bitmap_x_pos >= parent_picking_bitmap_panel->PanelBitmap.GetWidth( ) - 128 && sub_bitmap_x_pos < parent_picking_bitmap_panel->PanelBitmap.GetWidth( ) )
        sub_bitmap_x_size = parent_picking_bitmap_panel->PanelBitmap.GetWidth( ) - sub_bitmap_x_pos;

    if ( sub_bitmap_y_pos < 0 && sub_bitmap_y_pos > -128 ) {
        sub_bitmap_y_offset = abs(sub_bitmap_y_pos);
        sub_bitmap_y_size -= sub_bitmap_y_offset;
    }
    else if ( sub_bitmap_y_pos >= parent_picking_bitmap_panel->PanelBitmap.GetHeight( ) - 128 && sub_bitmap_y_pos < parent_picking_bitmap_panel->PanelBitmap.GetHeight( ) )
        sub_bitmap_y_size = parent_picking_bitmap_panel->PanelBitmap.GetHeight( ) - sub_bitmap_y_pos;

    // the following line is a whole host of checks designed to not grab a dodgy bit of bitmap

    if ( sub_bitmap_x_pos + sub_bitmap_x_offset >= 0 && sub_bitmap_y_pos + sub_bitmap_y_offset >= 0 && sub_bitmap_y_pos + sub_bitmap_y_offset < parent_picking_bitmap_panel->PanelBitmap.GetHeight( ) && sub_bitmap_x_pos + sub_bitmap_x_offset < parent_picking_bitmap_panel->PanelBitmap.GetWidth( ) && sub_bitmap_x_size > 0 && sub_bitmap_y_size > 0 ) {
        wxBitmap section    = parent_picking_bitmap_panel->PanelBitmap.GetSubBitmap(wxRect(sub_bitmap_x_pos + sub_bitmap_x_offset, sub_bitmap_y_pos + sub_bitmap_y_offset, sub_bitmap_x_size, sub_bitmap_y_size));
        wxImage  paintimage = section.ConvertToImage( );
        paintimage.Rescale(section.GetWidth( ) * 2, section.GetHeight( ) * 2);
        wxBitmap topaint(paintimage);

        dc.DrawBitmap(topaint, sub_bitmap_x_offset * 2, sub_bitmap_y_offset * 2);
        dc.SetPen(wxPen(*wxRED, 2, wxLONG_DASH));
        dc.CrossHair(128, 128);
    }

    event.Skip( );
}

void PickingBitmapPanelPopup::OnEraseBackground(wxEraseEvent& event) {
}

void PickingBitmapPanel::OnRightDown(wxMouseEvent& event) {
    long x_pos;
    long y_pos;
    event.GetPosition(&x_pos, &y_pos);
    if ( ! popup_exists ) {
        int client_x = int(x_pos);
        int client_y = int(y_pos);

        ClientToScreen(&client_x, &client_y);

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

        CaptureMouse( );
        popup = new PickingBitmapPanelPopup(this);
        popup->SetClientSize(256, 256);
        popup->Position(wxPoint(client_x - 128, client_y - 128), wxSize(0, 0));
        popup->x_pos = x_pos - 128 - bitmap_x_offset;
        popup->y_pos = y_pos - 128 - bitmap_y_offset;
        popup->SetCursor(wxCursor(wxCURSOR_BLANK));
        popup->Show( );
        popup->Refresh( );
        popup->Update( );
        popup_exists = true;
    }
    event.Skip( );
}

void PickingBitmapPanel::OnRightUp(wxMouseEvent& event) {
    if ( popup_exists ) {
        ReleaseMouse( );
        SetCursor(wxCursor(wxCURSOR_CROSS));
        popup->Destroy( );
        popup_exists = false;
    }
    event.Skip( );
}

void PickingBitmapPanel::OnMiddleUp(wxMouseEvent& event) {
    old_mouse_x = -9999;
    old_mouse_y = -9999;
}

float PickingBitmapPanel::PixelToAngstromX(const int& x_in_pixels) {
    return float(x_in_pixels - bitmap_x_offset) * image_in_bitmap_pixel_size;
}

float PickingBitmapPanel::PixelToAngstromY(const int& y_in_pixels) {
    return float(bitmap_y_offset + bitmap_height - y_in_pixels) * image_in_bitmap_pixel_size;
}

void PickingBitmapPanel::OnMotion(wxMouseEvent& event) {
    int x_pos, y_pos;
    event.GetPosition(&x_pos, &y_pos);
    if ( draw_selection_rectangle ) {
        selection_rectangle_current_x = x_pos;
        selection_rectangle_current_y = y_pos;
    }
    else if ( event.LeftIsDown( ) == true && event.ShiftDown( ) ) // is the left button and shift down
    {
        event.GetPosition(&clicked_point_x, &clicked_point_y);
        if ( clicked_point_x > bitmap_x_offset && clicked_point_x < bitmap_x_offset + bitmap_width && clicked_point_y > bitmap_y_offset && clicked_point_y < bitmap_y_offset + bitmap_height ) {
            clicked_point_x_in_angstroms = PixelToAngstromX(clicked_point_x);
            clicked_point_y_in_angstroms = PixelToAngstromY(clicked_point_y);
            if ( allow_editing_of_coordinates ) {
                RemoveParticleCoordinatesWithinRectangleOrNearClickedPoint( );
            }
        }
    }

    // FIXME: we need to add accounting for the image bitmap being bigger than what fits into the panel
    // And in this situation, we need to not display anything
    else if ( event.RightIsDown( ) && popup_exists ) {
        int client_x = x_pos;
        int client_y = y_pos;
        ClientToScreen(&client_x, &client_y);

        // This is the same popup window workaround that was used for the DisplayPanel
        // In addition, however, we subtract by the bitmap_x_offset/bitmap_y_offset
        // because of the white border surrounding the image in the bitmap. Failure
        // to do this causes the popup to display the scaled bitmap at a different position
        // than the mouse.

        int screen_x_size = wxSystemSettings::GetMetric(wxSYS_SCREEN_X);
        int screen_y_size = wxSystemSettings::GetMetric(wxSYS_SCREEN_Y);

        // Try to bound by the size of the panel for what displays...
        int window_size_x;
        int window_size_y;
        GetClientSize(&window_size_x, &window_size_y); // the actual size of the panel

        wxPrintf("window_size_x = %i, window_size_y = %i\n", window_size_x, window_size_y);

        // TODO: add bounding for top and bottom...
        if ( client_x + 256 > screen_x_size )
            client_x += 256;
        if ( client_y + 256 > screen_y_size )
            client_y += 256;

        // DEBUG:
        wxPrintf("client_x = %i; client_y = %i\n", client_x, client_y);
        popup->Position(wxPoint(client_x - 128, client_y - 128), wxSize(0, 0));
        popup->x_pos = x_pos - 128 - bitmap_x_offset;
        popup->y_pos = y_pos - 128 - bitmap_y_offset;
        popup->Show( );
        popup->Refresh( );
        popup->Update( );
        Update( );
    }
    else if ( event.MiddleIsDown( ) ) {
        if ( old_mouse_x == -9999 || old_mouse_y == -9999 ) {
            old_mouse_x = x_pos;
            old_mouse_y = y_pos;
        }
        // FIXME: account for offset of image in the bitmap
        else {
            // Account for scaling
            image_in_bitmap_x = (old_mouse_x - x_pos) / scale_factor;
            image_in_bitmap_y = (old_mouse_y - y_pos) / scale_factor;

            // Set bounds to prevent stupid dragging
            // Left bound
            if ( image_in_bitmap_x < 0 )
                image_in_bitmap_x = 0;

            // Right bound
            else if ( image_in_bitmap_x > image_in_bitmap.logical_x_dimension - 1 )
                image_in_bitmap_x > image_in_bitmap.logical_x_dimension - 1;

            // Bottom bound
            if ( image_in_bitmap_y < 0 )
                image_in_bitmap_y = 0;

            // Top bound
            else if ( image_in_bitmap_y > image_in_bitmap.logical_y_dimension - 1 )
                image_in_bitmap_y = image_in_bitmap.logical_y_dimension - 1;
        }
    }

    event.Skip( );
}

bool PickingBitmapPanel::UserHasEditedParticleCoordinates( ) {
    return current_step_in_history > 0;
}
