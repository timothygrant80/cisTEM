#include "../core/gui_core_headers.h"

int compare_particle_coordinate_using_x( particle_coordinate **first, particle_coordinate **second)
{
	if (first[0]->x < second[0]->x)
	{
		return -1;
	}
	else
	{
		if (first[0]->x > second[0]->x) return 1;
		return 0;
	}
}


#include <wx/arrimpl.cpp>
WX_DEFINE_OBJARRAY(ArrayOfCoordinates);

PickingBitmapPanel::PickingBitmapPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
: wxPanel(parent, id, pos, size, style, name)
{

	// create a 1 pixel white bitmap

	PanelBitmap.Create(1, 1, 24);
	panel_text = "";



	should_show = false;
	font_size_multiplier = 1.0;

	Bind(wxEVT_PAINT, &PickingBitmapPanel::OnPaint, this);
	Bind(wxEVT_ERASE_BACKGROUND, &PickingBitmapPanel::OnEraseBackground, this);
	Bind(wxEVT_SIZE, &PickingBitmapPanel::OnSize, this);
	Bind(wxEVT_LEFT_DOWN, &PickingBitmapPanel::OnLeftDown, this);
	Bind(wxEVT_LEFT_UP, &PickingBitmapPanel::OnLeftUp, this);
	Bind(wxEVT_MOTION, &PickingBitmapPanel::OnMotion, this);

	image_in_bitmap_filename = "";
	image_in_bitmap_pixel_size = 0.0;
	image_in_bitmap_scaling_factor = 0.0;

	image_in_memory_filename = "";
	image_in_memory_pixel_size = 0.0;

	draw_circles_around_particles = true;
	number_of_particles = 0;
	radius_of_circles_around_particles_in_angstroms = 0.0;
	//particle_coordinates_x_in_angstroms = NULL;
	//particle_coordinates_y_in_angstroms = NULL;
	//particle_coordinates_in_angstroms = NULL;

	draw_scale_bar = true;
	should_high_pass = true;

	allow_editing_of_coordinates = true;

	draw_selection_rectangle = false;
	selection_rectangle_current_x = 0;
	selection_rectangle_current_y = 0;
	selection_rectangle_start_x = 0;
	selection_rectangle_start_y = 0;
	selection_rectangle_start_x_in_angstroms = 0.0;
	selection_rectangle_start_y_in_angstroms = 0.0;
	selection_rectangle_finish_x_in_angstroms = 0.0;
	selection_rectangle_finish_y_in_angstroms = 0.0;


}

PickingBitmapPanel::~PickingBitmapPanel()
{
	Unbind(wxEVT_PAINT, &PickingBitmapPanel::OnPaint, this);
	Unbind(wxEVT_ERASE_BACKGROUND, &PickingBitmapPanel::OnEraseBackground, this);
	Unbind(wxEVT_SIZE, &PickingBitmapPanel::OnSize, this);
	Unbind(wxEVT_LEFT_DOWN, &PickingBitmapPanel::OnLeftDown, this);
	Unbind(wxEVT_LEFT_UP, &PickingBitmapPanel::OnLeftUp, this);
	Unbind(wxEVT_MOTION, &PickingBitmapPanel::OnMotion, this);

	particle_coordinates_in_angstroms.Empty();

}

void PickingBitmapPanel::Clear()
{
	Freeze();
    wxClientDC dc(this);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear();
    image_in_memory.Deallocate();
    image_in_bitmap.Deallocate();
    Thaw();
}

void PickingBitmapPanel::OnEraseBackground(wxEraseEvent& event)
{

}

void PickingBitmapPanel::AllocateMemoryForParticleCoordinates(const int wanted_number_of_particles)
{
	if (wanted_number_of_particles != number_of_particles)
	{
		//delete [] particle_coordinates_x_in_angstroms;
		//delete [] particle_coordinates_y_in_angstroms;
		//delete [] particle_coordinates_in_angstroms;
		//particle_coordinates_x_in_angstroms = new float [wanted_number_of_particles];
		//particle_coordinates_y_in_angstroms = new float [wanted_number_of_particles];
		//particle_coordinates_in_angstroms = new particle_coordinate[wanted_number_of_particles];
		particle_coordinates_in_angstroms.Alloc(wanted_number_of_particles);
		number_of_particles = wanted_number_of_particles;
	}

}

void PickingBitmapPanel::SetParticleCoordinatesAndRadius(const int wanted_number_of_particles, const double *wanted_x, const double *wanted_y, const float wanted_radius_in_angstroms)
{
	AllocateMemoryForParticleCoordinates(wanted_number_of_particles);

	for (int counter = 0; counter < wanted_number_of_particles; counter ++ )
	{
		//particle_coordinates_x_in_angstroms[counter] = wanted_x[counter];
		//particle_coordinates_y_in_angstroms[counter] = wanted_y[counter];
		//particle_coordinates_in_angstroms[counter].x = wanted_x[counter];
		//particle_coordinates_in_angstroms[counter].y = wanted_y[counter];
		//particle_coordinates_in_angstroms.insert(particle_coordinates_in_angstroms.end(), particle_coordinate(float(wanted_x[counter]),float(wanted_y[counter])));
		particle_coordinates_in_angstroms.Add(particle_coordinate(float(wanted_x[counter]),float(wanted_y[counter])));
	}
	particle_coordinates_in_angstroms.Sort(compare_particle_coordinate_using_x);
	radius_of_circles_around_particles_in_angstroms = wanted_radius_in_angstroms;
}

void PickingBitmapPanel::RemoveParticleCoordinatesWithinRectangle()
{

	size_t counter = particle_coordinates_in_angstroms.GetCount() - 1;
	while (true)
	{
		if (ParticleCoordinatesShouldBeRemoved(particle_coordinates_in_angstroms.Item(counter))) particle_coordinates_in_angstroms.RemoveAt(counter);
		if (counter == 0) break;
		counter --;
	}

	Refresh();
	Update();

}

bool PickingBitmapPanel::ParticleCoordinatesShouldBeRemoved(const particle_coordinate &particle_coordinates_to_check)
{
	return 	particle_coordinates_to_check.x >= selection_rectangle_start_x_in_angstroms &&
			particle_coordinates_to_check.x <= selection_rectangle_finish_x_in_angstroms &&
			particle_coordinates_to_check.y >= selection_rectangle_start_y_in_angstroms &&
			particle_coordinates_to_check.y <= selection_rectangle_finish_y_in_angstroms;
}

void PickingBitmapPanel::SetImageFilename(wxString wanted_filename, float pixel_size)
{
	if (!wanted_filename.IsSameAs(image_in_memory_filename))
	{
		image_in_memory.QuickAndDirtyReadSlice(wanted_filename.ToStdString(),1);
		image_in_memory_pixel_size = pixel_size;
		image_in_memory_filename = wanted_filename;
	}
}

void PickingBitmapPanel::UpdateScalingAndDimensions()
{
	if (!image_in_memory_filename.IsEmpty())
	{
		int panel_dim_x, panel_dim_y;
		GetClientSize(&panel_dim_x, &panel_dim_y);

		float target_scaling_x = float(panel_dim_x) * 0.95 /float(image_in_memory.logical_x_dimension);
		float target_scaling_y = float(panel_dim_y) * 0.95 /float(image_in_memory.logical_y_dimension);
		float scaling_factor = std::min(target_scaling_x,target_scaling_y);

		int new_x_dimension = int(float(image_in_memory.logical_x_dimension) * scaling_factor);
		int new_y_dimension = int(float(image_in_memory.logical_y_dimension) * scaling_factor);

		// TODO: choose dimensions that are more favorable to FFT

		if (!image_in_bitmap.is_in_memory || new_x_dimension != image_in_bitmap.logical_x_dimension || new_y_dimension != image_in_bitmap.logical_y_dimension )
		{
			image_in_bitmap.Allocate(new_x_dimension,new_y_dimension,true);
			image_in_bitmap_scaling_factor = scaling_factor;
			image_in_bitmap_pixel_size = image_in_memory_pixel_size / scaling_factor;

		}
	}
}

void PickingBitmapPanel::UpdateImageInBitmap(bool force_reload)
{
	if (!image_in_memory_filename.IsEmpty())
	{
		if (force_reload || !image_in_bitmap_filename.IsSameAs(image_in_memory_filename) || PanelBitmap.GetWidth() != image_in_bitmap.logical_x_dimension || PanelBitmap.GetHeight() != image_in_bitmap.logical_y_dimension)
		{
			if (image_in_memory.is_in_real_space) image_in_memory.ForwardFFT();
			image_in_bitmap.is_in_real_space = false;
			image_in_memory.ClipInto(&image_in_bitmap);
			if (should_high_pass)
			{
				image_in_bitmap.CosineMask(image_in_bitmap_pixel_size / (4.0 * radius_of_circles_around_particles_in_angstroms),image_in_bitmap_pixel_size / (2.0 * radius_of_circles_around_particles_in_angstroms),true);
			}
			image_in_bitmap.BackwardFFT();
			image_in_bitmap_filename = image_in_memory_filename;
			ConvertImageToBitmap(&image_in_bitmap,&PanelBitmap,true);
		}
	}
}


void PickingBitmapPanel::OnSize(wxSizeEvent & event)
{
	UpdateScalingAndDimensions();
	UpdateImageInBitmap();
	event.Skip();
}


void PickingBitmapPanel::OnPaint(wxPaintEvent & evt)
{

	Freeze();

	int window_x_size;
	int window_y_size;

    wxPaintDC dc(this);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear();
    GetClientSize(&window_x_size, &window_y_size);
    dc.SetBrush( wxNullBrush );
    dc.DrawRectangle(0, 0, window_x_size, window_y_size);

	if (should_show)
	{
		wxFont current_font = dc.GetFont();
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

		GetClientSize(&window_x_size,&window_y_size);
		bitmap_width = PanelBitmap.GetWidth();
		bitmap_height = PanelBitmap.GetHeight();

		if (panel_text.IsEmpty() == true)
		{
			text_x_size = 0;
			text_y_size = 0;
		}
		else GetMultilineTextExtent(&dc, panel_text, text_x_size, text_y_size);

		combined_width = bitmap_width + text_x_size;
		combined_height = bitmap_height;

		if (combined_width > window_x_size || combined_height > window_y_size)
		{
			MyDebugAssertTrue(false,"Oops, should not get here, because the image should always fit in\n");
		}
		else
		{
			bitmap_x_offset = (window_x_size - ((bitmap_width) + text_x_size)) / 2;
			bitmap_y_offset = (window_y_size - (bitmap_height)) / 2;

			// Draw the image bitmap
			dc.DrawBitmap( PanelBitmap, bitmap_x_offset, bitmap_y_offset, false );

			// Choose a pen thickness for drawing circles around particles
			int pen_thickness = std::min(bitmap_width,bitmap_height) / 512;
			if (pen_thickness < 1) pen_thickness = 1;
			if (pen_thickness > 5) pen_thickness = 5;

			// Draw circles around particles
			if (draw_circles_around_particles)
			{
				float x,y;
				dc.SetPen( wxPen(wxColor(255,0,0),pen_thickness) );
				dc.SetBrush( wxNullBrush );
				for (int counter = 0; counter < particle_coordinates_in_angstroms.GetCount(); counter ++ )
				{
					x = particle_coordinates_in_angstroms.Item(counter).x;
					y = particle_coordinates_in_angstroms.Item(counter).y;
					//dc.DrawCircle(x_offset + bitmap_width - particle_coordinates_x_in_angstroms[counter] / image_in_bitmap_pixel_size, y_offset + particle_coordinates_y_in_angstroms[counter] / image_in_bitmap_pixel_size,radius_of_circles_around_particles_in_angstroms / image_in_bitmap_pixel_size);
					dc.DrawCircle(bitmap_x_offset + bitmap_width - x / image_in_bitmap_pixel_size, bitmap_y_offset + y / image_in_bitmap_pixel_size,radius_of_circles_around_particles_in_angstroms / image_in_bitmap_pixel_size);
				}
			}

			// Draw scale bar
			if (draw_scale_bar)
			{
				wxPen scalebar_pen;
				scalebar_pen = wxPen( *wxWHITE );
				dc.SetPen ( scalebar_pen );
				dc.SetBrush( *wxWHITE_BRUSH );
				int scalebar_length = int(float(bitmap_width) * 0.1);
				int scalebar_x_start = int(float(bitmap_width) * 0.85);
				int scalebar_y_pos = int(float(bitmap_height)*0.95);
				int scalebar_thickness = int(float(bitmap_height) / 50.0);
				dc.DrawRectangle(bitmap_x_offset+scalebar_x_start,bitmap_y_offset+scalebar_y_pos,scalebar_length,scalebar_thickness);
				//dc.SetPen( *wxRED_PEN );
				dc.SetTextForeground( *wxWHITE );
				dc.SetFont( *wxNORMAL_FONT );
				dc.SetFont(wxFont(std::max(12,int(float(scalebar_thickness)*0.75)),wxFONTFAMILY_DEFAULT,wxFONTSTYLE_NORMAL,wxFONTWEIGHT_BOLD));
				wxString scalebar_label = wxString::Format("%.1f nm",float(scalebar_length) * image_in_bitmap_pixel_size * 0.1);
				int scalebar_label_width;
				int scalebar_label_height;
				dc.GetTextExtent(scalebar_label,&scalebar_label_width,&scalebar_label_height);
				dc.DrawText(scalebar_label,bitmap_x_offset + scalebar_x_start + scalebar_length/2 - scalebar_label_width/2,bitmap_y_offset + scalebar_y_pos - scalebar_label_height - scalebar_thickness/8);
			}

			// Draw selection retangle
			if (draw_selection_rectangle)
			{
				dc.SetPen( wxPen(wxColor(255,0,0),pen_thickness * 2, wxPENSTYLE_LONG_DASH) );
				dc.SetBrush( wxNullBrush );
				dc.DrawRectangle(std::min(selection_rectangle_start_x,selection_rectangle_current_x),std::min(selection_rectangle_start_y,selection_rectangle_current_y),abs(selection_rectangle_current_x - selection_rectangle_start_x),abs(selection_rectangle_current_y - selection_rectangle_start_y));
			}


			// Draw text

			text_y_offset = ((bitmap_height) - text_y_size) / 2;
			if (text_y_offset > 0) dc.DrawText(panel_text, bitmap_width + bitmap_x_offset, text_y_offset + bitmap_y_offset);
			else
			dc.DrawText(panel_text, bitmap_width + bitmap_x_offset, bitmap_y_offset);

		}
	}

	Thaw();


}

void PickingBitmapPanel::SetupPanelBitmap()
{
	int window_x_size;
	int window_y_size;

	int x_padding;
	int y_padding;

	GetClientSize(&window_x_size,&window_y_size);

	if (window_x_size != PanelBitmap.GetWidth() || window_y_size != PanelBitmap.GetHeight())
	{
		PanelBitmap.Create(window_x_size, window_y_size, 24);
	}

}

void PickingBitmapPanel::OnLeftDown(wxMouseEvent & event)
{
	int x_pos, y_pos;
	event.GetPosition(&x_pos,&y_pos);
	if (event.ControlDown())
	{
		draw_selection_rectangle = true;
		selection_rectangle_start_x = x_pos;
		selection_rectangle_start_y = y_pos;
	}
	event.Skip();
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

void PickingBitmapPanel::OnLeftUp(wxMouseEvent & event)
{

	if (draw_selection_rectangle)
	{
		// Convert begin and end coordinates to Angstroms
		//dc.DrawCircle(x_offset + bitmap_width - particle_coordinates_in_angstroms[counter].x / image_in_bitmap_pixel_size, y_offset + particle_coordinates_in_angstroms[counter].y / image_in_bitmap_pixel_size,radius_of_circles_around_particles_in_angstroms / image_in_bitmap_pixel_size);
		selection_rectangle_start_x_in_angstroms 	= float(bitmap_x_offset + bitmap_width - std::max(selection_rectangle_start_x,selection_rectangle_current_x)) * image_in_bitmap_pixel_size;
		selection_rectangle_finish_x_in_angstroms	= float(bitmap_x_offset + bitmap_width - std::min(selection_rectangle_start_x,selection_rectangle_current_x)) * image_in_bitmap_pixel_size;
		selection_rectangle_start_y_in_angstroms 	= float(std::min(selection_rectangle_start_y,selection_rectangle_current_y) - bitmap_y_offset) * image_in_bitmap_pixel_size;
		selection_rectangle_finish_y_in_angstroms 	= float(std::max(selection_rectangle_start_y,selection_rectangle_current_y) - bitmap_y_offset) * image_in_bitmap_pixel_size;
		RemoveParticleCoordinatesWithinRectangle();
		draw_selection_rectangle = false;
	}
	Refresh();
	Update();
	event.Skip();
}

void PickingBitmapPanel::OnMotion(wxMouseEvent & event)
{
	int x_pos, y_pos;
	event.GetPosition(&x_pos,&y_pos);
	if (draw_selection_rectangle)
	{
		selection_rectangle_current_x = x_pos;
		selection_rectangle_current_y = y_pos;
	}
	Refresh();
	Update();
	event.Skip();
}
