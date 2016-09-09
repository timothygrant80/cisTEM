//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

BitmapPanel::BitmapPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
: wxPanel(parent, id, pos, size, style, name)
{

	// create a 1 pixel white bitmap

	PanelBitmap.Create(1, 1, 24);
	panel_text = "";
	title_text = "";
/*	wxNativePixelData pixel_data(PanelBitmap);

	if ( !pixel_data )		{
	   MyPrintWithDetails("Can't access bitmap data");
	   abort();
	}

	wxNativePixelData::Iterator p(pixel_data);

	p.Red() = 255;
	p.Green() = 255;
	p.Blue() = 255;

//	SetupPanelBitmap();*/

	should_show = false;
	font_size_multiplier = 1.0;

	Bind(wxEVT_PAINT, &BitmapPanel::OnPaint, this);
	Bind(wxEVT_ERASE_BACKGROUND, &BitmapPanel::OnEraseBackground, this);




}

BitmapPanel::~BitmapPanel()
{
	Unbind(wxEVT_PAINT, &BitmapPanel::OnPaint, this);
	Unbind(wxEVT_ERASE_BACKGROUND, &BitmapPanel::OnEraseBackground, this);

}

void BitmapPanel::Clear()
{
	Freeze();
    wxClientDC dc(this);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear();
    Thaw();
}

void BitmapPanel::OnEraseBackground(wxEraseEvent& event)
{

}

void BitmapPanel::OnPaint(wxPaintEvent & evt)
{
	Freeze();

	int window_x_size;
	int window_y_size;

    wxPaintDC dc(this);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear();
    dc.GetSize(&window_x_size, &window_y_size);
    dc.DrawRectangle(0, 0, window_x_size, window_y_size);

	if (should_show == true)
	{
		wxFont current_font = dc.GetFont();
		current_font.Scale(font_size_multiplier);
		dc.SetFont(current_font);

		int text_x_size;
		int text_y_size;

		int title_x_size;
		int title_y_size;

		int bitmap_width;
		int bitmap_height;

		int combined_width;
		int combined_height;

		int x_offset;
		int y_offset;

		int text_y_offset;
		int title_x_offset;

		float scale_factor;

		int x_oversize;
		int y_oversize;

		const int title_y_pad = 4;

		GetClientSize(&window_x_size,&window_y_size);
		bitmap_width = PanelBitmap.GetWidth();
		bitmap_height = PanelBitmap.GetHeight();

		if (panel_text.IsEmpty() == true)
		{
			text_x_size = 0;
			text_y_size = 0;
		}
		else GetMultilineTextExtent(&dc, panel_text, text_x_size, text_y_size);

		if (title_text.IsEmpty())
		{
			title_x_size = 0;
			title_y_size = 0;
		}
		else GetMultilineTextExtent(&dc, title_text, title_x_size, title_y_size);

		combined_width = bitmap_width + text_x_size;
		combined_height = bitmap_height + title_y_size;

		if (combined_width > window_x_size || combined_height > window_y_size)
		{
			x_oversize = combined_width - window_x_size;
			y_oversize = combined_height - window_y_size;

			if (x_oversize > y_oversize)
			{
				// need to x_scale..
				scale_factor = float(window_x_size - text_x_size) / float(bitmap_width);

				if (scale_factor > 0)
				{

					x_offset = (window_x_size - ((bitmap_width  * scale_factor) + text_x_size )) / 2;
					y_offset = (window_y_size - ((bitmap_height * scale_factor) + title_y_size)) / 2;

					dc.DrawBitmap(wxBitmap(PanelBitmap.ConvertToImage().Scale(bitmap_width * scale_factor, bitmap_height * scale_factor)), x_offset, y_offset, false);



					text_y_offset = ((bitmap_height * scale_factor) - text_y_size) / 2;
					//wxPrintf("BH = %i, t_size = %i, text_offset = %i\n", int(bitmap_height * scale_factor), text_y_size, text_offset);
					if (text_y_offset > 0) dc.DrawText(panel_text, (bitmap_width * scale_factor) + x_offset, y_offset + text_y_offset);
					else
					                       dc.DrawText(panel_text, (bitmap_width * scale_factor) + x_offset, y_offset);


					title_x_offset = (bitmap_width * scale_factor - title_x_size) / 2;

					if (title_x_offset > 0) dc.DrawText(title_text, title_x_offset + x_offset, y_offset - title_y_size - title_y_pad);
					else
											dc.DrawText(title_text, 0, 0);

				}
				else
				{
					dc.DrawText(panel_text, 0, 0);
				}

			}
			else
			{
				scale_factor = float(window_y_size - title_y_size) / float(bitmap_height);

				if (scale_factor > 0)
				{
					x_offset = (window_x_size - ((bitmap_width  * scale_factor) + text_x_size )) / 2;
					y_offset = (window_y_size - ((bitmap_height * scale_factor) + title_y_size)) / 2;

					dc.DrawBitmap(wxBitmap(PanelBitmap.ConvertToImage().Scale(bitmap_width * scale_factor, bitmap_height * scale_factor)), x_offset, y_offset, false);


					text_y_offset = ((bitmap_height * scale_factor) - text_y_size) / 2;

//					wxPrintf("BH = %i, t_size = %i, text_offset = %i\n", int(bitmap_height * scale_factor), text_y_size, text_offset);

					if (text_y_offset > 0)  dc.DrawText(panel_text, (bitmap_width * scale_factor) + x_offset, y_offset + text_y_offset);
					else
					dc.DrawText(panel_text, (bitmap_width * scale_factor) + x_offset, y_offset);

					title_x_offset = (bitmap_width - title_x_size) / 2;

					if (title_x_offset > 0) dc.DrawText(title_text, title_x_offset + x_offset, y_offset - title_y_size - title_y_pad);
					else
											dc.DrawText(title_text, 0, 0);

				}
				else
				{
					dc.DrawText(panel_text, 0, 0);

				}

			}
		}
		else
		{
			x_offset = (window_x_size - ((bitmap_width)  + text_x_size )) / 2;
			y_offset = (window_y_size - ((bitmap_height) + title_y_size)) / 2;


			dc.DrawBitmap( PanelBitmap, x_offset, y_offset, false );

			text_y_offset = ((bitmap_height) - text_y_size) / 2;
			if (text_y_offset > 0) dc.DrawText(panel_text, bitmap_width + x_offset, text_y_offset + y_offset);
			else
			dc.DrawText(panel_text, bitmap_width + x_offset, y_offset);

			title_x_offset = (bitmap_width - title_x_size) / 2;

			if (title_x_offset > 0) dc.DrawText(title_text, title_x_offset + x_offset, y_offset - title_y_size - title_y_pad);
			else
									dc.DrawText(title_text, 0, 0);

		}
	}

	Thaw();

}

void BitmapPanel::SetupPanelBitmap()
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
