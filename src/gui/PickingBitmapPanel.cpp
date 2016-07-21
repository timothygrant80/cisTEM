#include "../core/gui_core_headers.h"


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

	image_in_bitmap_filename = "";
	image_in_bitmap_pixel_size = 0.0;
	image_in_bitmap_scaling_factor = 0.0;

	image_in_memory_filename = "";
	image_in_memory_pixel_size = 0.0;

	draw_circles_around_particles = true;
	number_of_particles = 0;
	radius_of_circles_around_particles_in_angstroms = 0.0;
	particle_coordinates_x_in_angstroms = NULL;
	particle_coordinates_y_in_angstroms = NULL;

	draw_scale_bar = true;


}

PickingBitmapPanel::~PickingBitmapPanel()
{
	Unbind(wxEVT_PAINT, &PickingBitmapPanel::OnPaint, this);
	Unbind(wxEVT_ERASE_BACKGROUND, &PickingBitmapPanel::OnEraseBackground, this);

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
		delete [] particle_coordinates_x_in_angstroms;
		delete [] particle_coordinates_y_in_angstroms;
		particle_coordinates_x_in_angstroms = new float [wanted_number_of_particles];
		particle_coordinates_y_in_angstroms = new float [wanted_number_of_particles];
		number_of_particles = wanted_number_of_particles;
	}

}

void PickingBitmapPanel::SetParticleCoordinatesAndRadius(const int wanted_number_of_particles, const double *wanted_x, const double *wanted_y, const float wanted_radius_in_angstroms)
{
	AllocateMemoryForParticleCoordinates(wanted_number_of_particles);

	for (int counter = 0; counter < wanted_number_of_particles; counter ++ )
	{
		particle_coordinates_x_in_angstroms[counter] = wanted_x[counter];
		particle_coordinates_y_in_angstroms[counter] = wanted_y[counter];
	}
	radius_of_circles_around_particles_in_angstroms = wanted_radius_in_angstroms;
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
	int panel_dim_x, panel_dim_y;
	GetSize(&panel_dim_x, &panel_dim_y);

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

void PickingBitmapPanel::UpdateImageInBitmap()
{
	if (!image_in_memory_filename.IsEmpty())
	{
		if (!image_in_bitmap_filename.IsSameAs(image_in_memory_filename) || PanelBitmap.GetWidth() != image_in_bitmap.logical_x_dimension || PanelBitmap.GetHeight() != image_in_bitmap.logical_y_dimension)
		{
			wxPrintf("Doing resampling\n");
			if (image_in_memory.is_in_real_space) image_in_memory.ForwardFFT();
			image_in_bitmap.is_in_real_space = false;
			image_in_memory.ClipInto(&image_in_bitmap);
			image_in_bitmap.BackwardFFT();
			image_in_bitmap_filename = image_in_memory_filename;
			ConvertImageToBitmap(&image_in_bitmap,&PanelBitmap,true);
		}
	}
}

void PickingBitmapPanel::OnPaint(wxPaintEvent & evt)
{
	UpdateScalingAndDimensions();
	UpdateImageInBitmap();
	Draw();
}

void PickingBitmapPanel::Draw()
{

	Freeze();

	int window_x_size;
	int window_y_size;

    wxPaintDC dc(this);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear();
    GetClientSize(&window_x_size, &window_y_size);
    dc.DrawRectangle(0, 0, window_x_size, window_y_size);

	if (should_show)
	{
		wxFont current_font = dc.GetFont();
		current_font.Scale(font_size_multiplier);
		dc.SetFont(current_font);

		int text_x_size;
		int text_y_size;

		int bitmap_width;
		int bitmap_height;

		int combined_width;
		int combined_height;

		int x_offset;
		int y_offset;

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
			x_offset = (window_x_size - ((bitmap_width) + text_x_size)) / 2;
			y_offset = (window_y_size - (bitmap_height)) / 2;

			// Draw the image bitmap
			dc.DrawBitmap( PanelBitmap, x_offset, y_offset, false );


			// Draw circles around particles
			if (draw_circles_around_particles)
			{
				dc.SetPen( wxPen(wxColor(255,0,0),2) );
				dc.SetBrush( wxNullBrush );
				for (int counter = 0; counter < number_of_particles; counter ++ )
				{
					dc.DrawCircle(x_offset + bitmap_width - particle_coordinates_x_in_angstroms[counter] / image_in_bitmap_pixel_size, y_offset + particle_coordinates_y_in_angstroms[counter] / image_in_bitmap_pixel_size,radius_of_circles_around_particles_in_angstroms / image_in_bitmap_pixel_size);
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
				wxPrintf("Length of scale bar = %i pixels\n",scalebar_length);
				dc.DrawRectangle(x_offset+scalebar_x_start,y_offset+scalebar_y_pos,scalebar_length,scalebar_thickness);
				//dc.SetPen( *wxRED_PEN );
				dc.SetTextForeground( *wxWHITE );
				dc.SetFont( *wxNORMAL_FONT );
				dc.SetFont(wxFont(std::max(12,int(float(scalebar_thickness)*0.75)),wxFONTFAMILY_DEFAULT,wxFONTSTYLE_NORMAL,wxFONTWEIGHT_BOLD));
				wxString scalebar_label = wxString::Format("%.1f nm",float(scalebar_length) * image_in_bitmap_pixel_size * 0.1);
				int scalebar_label_width;
				int scalebar_label_height;
				dc.GetTextExtent(scalebar_label,&scalebar_label_width,&scalebar_label_height);
				dc.DrawText(scalebar_label,x_offset + scalebar_x_start + scalebar_length/2 - scalebar_label_width/2,y_offset + scalebar_y_pos - scalebar_label_height - scalebar_thickness/8);
			}


			// Draw text

			text_y_offset = ((bitmap_height) - text_y_size) / 2;
			if (text_y_offset > 0) dc.DrawText(panel_text, bitmap_width + x_offset, text_y_offset + y_offset);
			else
			dc.DrawText(panel_text, bitmap_width + x_offset, y_offset);

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
