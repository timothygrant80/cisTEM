#include "../core/gui_core_headers.h"

#include <wx/arrimpl.cpp>
//WX_DEFINE_OBJARRAY(ArrayOfRefinementResults);

DistributionPlotPanel::DistributionPlotPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
: wxPanel(parent, id, pos, size, style, name)
{

	should_show = true;
	font_size_multiplier = 1.0;
	number_of_final_results = 10000;
	colour_change_step = 5;

	draw_axis_overlay_instead_of_underlay = false;

	int client_x;
	int client_y;

	GetClientSize(&client_x, &client_y);
	buffer_bitmap.Create(wxSize(client_x, client_y));
	UpdateScalingAndDimensions();
	SetupBitmap();

	Bind(wxEVT_PAINT, &DistributionPlotPanel::OnPaint, this);
	Bind(wxEVT_SIZE,  &DistributionPlotPanel::OnSize, this);


}


DistributionPlotPanel::~DistributionPlotPanel()
{
	Unbind(wxEVT_PAINT, &DistributionPlotPanel::OnPaint, this);
	Unbind(wxEVT_SIZE,  &DistributionPlotPanel::OnSize, this);
}

void DistributionPlotPanel::Clear()
{
	Freeze();
    wxClientDC dc(this);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear();
    Thaw();
    refinement_results_to_plot.Clear();

	int client_x;
	int client_y;

	GetClientSize(&client_x, &client_y);
	buffer_bitmap.Create(wxSize(client_x, client_y));

	UpdateScalingAndDimensions();
	SetupBitmap();
	Refresh();

}


void DistributionPlotPanel::OnSize(wxSizeEvent & event)
{
	UpdateScalingAndDimensions();
	SetupBitmap();
	event.Skip();
}

void DistributionPlotPanel::DrawAxisUnderlay()
{
	float tmp_x = 0.0;
	float tmp_y = 0.0;

	wxMemoryDC *memDC;
	memDC = new wxMemoryDC;
	memDC->SelectObject(buffer_bitmap);

	wxGraphicsContext *gc = wxGraphicsContext::Create( *memDC );

	// Draw small circles for each projection direction
	UpdateProjCircleRadius();
	gc->SetPen( wxNullPen );
	//gc->SetPen( wxPen(wxColor(255,0,0),2) );
	gc->SetBrush( wxBrush(wxColor(50,50,200,60)) );

	wxGraphicsPath path;

	// Draw intermediate circles
	path = gc->CreatePath();
	gc->SetPen( *wxBLACK_DASHED_PEN );
	//path.AddCircle(circle_center_x,circle_center_y,circle_radius * PI * 0.5 / 9.0); // 10 degrees
	path.AddCircle(circle_center_x,circle_center_y,circle_radius * 0.33333f); // 30 degrees
	path.AddCircle(circle_center_x,circle_center_y,circle_radius * 0.66666f); // 60 degrees
	gc->StrokePath(path);

	// Draw a large cicle for the outside of the plot
	gc->SetPen( *wxBLACK_PEN );
	path = gc->CreatePath();
	path.AddCircle(circle_center_x,circle_center_y,circle_radius);
	gc->StrokePath(path);

	// Draw axes
	path = gc->CreatePath();
	path.AddCircle(circle_center_x,circle_center_y,circle_radius);
	path.MoveToPoint(circle_center_x - circle_radius - major_tick_length, circle_center_y);
	path.AddLineToPoint(circle_center_x + circle_radius + major_tick_length, circle_center_y);
	path.MoveToPoint(circle_center_x,circle_center_y - circle_radius - major_tick_length);
	path.AddLineToPoint(circle_center_x,circle_center_y + circle_radius + major_tick_length);
	gc->StrokePath(path);

	// Write labels
	wxDouble label_width;
	wxDouble label_height;
	wxDouble label_descent;
	wxDouble label_externalLeading;
	wxString greek_theta = wxT("\u03B8");
	wxString greek_phi = wxT("\u03C6");
	wxString degree_symbol = wxT("\u00B0");
	gc->SetFont( wxFont(10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false) , *wxBLACK );
	wxString current_label;

	current_label = greek_phi+" = 0 "+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	gc->DrawText(current_label, circle_center_x + circle_radius + major_tick_length + margin_between_major_ticks_and_labels, circle_center_y - label_height * 0.5, 0.0);
	current_label = greek_phi+" = 90 "+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	gc->DrawText(current_label, circle_center_x - label_width * 0.5, circle_center_y - circle_radius - major_tick_length - margin_between_major_ticks_and_labels - label_height, 0.0);

	current_label = greek_phi+" = 180 "+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	gc->DrawText(current_label, circle_center_x - circle_radius - major_tick_length - margin_between_major_ticks_and_labels - label_width, circle_center_y - label_height * 0.5, 0.0);
	current_label = greek_phi+" = 270 "+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	gc->DrawText(current_label, circle_center_x - label_width * 0.5, circle_center_y + circle_radius + major_tick_length + margin_between_major_ticks_and_labels, 0.0);

	current_label = greek_theta+" = 90 "+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	tmp_x = 0.5 * sqrt(2.0) * (circle_radius + margin_between_circles_and_theta_labels);
	tmp_y = - 0.5 * sqrt(2.0) * (circle_radius + margin_between_circles_and_theta_labels) - label_height;
	gc->DrawText(current_label, circle_center_x + tmp_x, circle_center_y + tmp_y, 0.0);

	//current_label = greek_theta+" = 10 "+degree_symbol;
	//gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	//tmp_x = 0.5 * sqrt(2.0) * (circle_radius * sin(PI * 0.5 / 9.0) + margin_between_circles_and_theta_labels);
	//tmp_y = - 0.5 * sqrt(2.0) * (circle_radius * sin(PI * 0.5 / 9.0) + margin_between_circles_and_theta_labels) - label_height;
	//gc->DrawText(current_label, circle_center_x + tmp_x, circle_center_y + tmp_y, 0.0);

	current_label = greek_theta+" = 30 "+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	tmp_x = 0.5 * sqrt(2.0) * (circle_radius * 0.3333f + margin_between_circles_and_theta_labels);
	tmp_y = - 0.5 * sqrt(2.0) * (circle_radius * 0.3333f + margin_between_circles_and_theta_labels) - label_height;
	gc->DrawText(current_label, circle_center_x + tmp_x, circle_center_y + tmp_y, 0.0);

	current_label = greek_theta+" = 60 "+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	tmp_x = 0.5 * sqrt(2.0) * (circle_radius * 0.6666f + margin_between_circles_and_theta_labels);
	tmp_y = - 0.5 * sqrt(2.0) * (circle_radius * 0.6666f + margin_between_circles_and_theta_labels) - label_height;
	gc->DrawText(current_label, circle_center_x + tmp_x, circle_center_y + tmp_y, 0.0);

	gc->SetPen(wxNullPen);
	gc->SetBrush(wxNullBrush);

	delete gc;
	memDC->SelectObject(wxNullBitmap);
	delete memDC;
}

void DistributionPlotPanel::SetupBitmap()
{
	if (buffer_bitmap.GetHeight() > 5 && buffer_bitmap.GetWidth() > 5)
	{
		wxMemoryDC *memDC;

		int window_x_size;
		int window_y_size;

		memDC = new wxMemoryDC;
		memDC->SelectObject(buffer_bitmap);
		memDC->SetBackground(*wxWHITE_BRUSH);
		memDC->Clear();
		memDC->GetSize(&window_x_size, &window_y_size);
		memDC->DrawRectangle(0, 0, window_x_size, window_y_size);
		memDC->SelectObject(wxNullBitmap);

		if (draw_axis_overlay_instead_of_underlay == true)
		{
			float circle_radius_squared = powf(circle_radius, 2.0f);
			float current_radius_squared;
			float y_position_squared;

			int x_position;
			int y_position;

			wxNativePixelData data(buffer_bitmap);

			if ( !data )
			{
				//... raw access to bitmap data unavailable, do something else ...
				return;
			}
			wxNativePixelData::Iterator p(data);

			p.MoveTo(data, circle_center_x - circle_radius, circle_center_y - circle_radius);

			for (y_position = myroundint(-circle_radius); y_position <= myroundint(circle_radius); y_position++)
			{
				y_position_squared = powf(y_position, 2.0f);
				wxNativePixelData::Iterator rowStart = p;

				for (x_position = myroundint(-circle_radius); x_position <= myroundint(circle_radius); x_position++ )
				{
					// what angle does this represent?
					current_radius_squared = powf(x_position, 2.0f) + y_position_squared;

					if (current_radius_squared <= circle_radius_squared)
					{
						p.Red() = 0;
						p.Green() = 0;
						p.Blue() = 128;
					}

					p++;
				}

				p = rowStart;
				p.OffsetY(data, 1);
			}

		}

		delete memDC;

		if (draw_axis_overlay_instead_of_underlay == false) DrawAxisUnderlay();

		//wxPrintf("Plotting %li, initialised with %li\n", refinement_results_to_plot.Count(), number_of_final_results);
		for (size_t counter = 0; counter < refinement_results_to_plot.Count(); counter ++ )
		{
			DrawBlueDot(refinement_results_to_plot.Item(counter));
		}

		if (draw_axis_overlay_instead_of_underlay == true) DrawAxisOverlay(0, min_number_of_projections_to_be_red);
	}
}

void DistributionPlotPanel::DrawBlueDot(RefinementResult &refinement_result_to_draw)
{
	float proj_x;
	float proj_y;
	float proj_z;

	float north_pole_x = 0.0;
	float north_pole_y = 0.0;
	float north_pole_z = 1.0;

	float tmp_x = 0.0;
	float tmp_y = 0.0;
	float tmp_angle = 0.0;

	float sym_phi;
	float sym_theta;

	float theta_radius;
	float phi_radius;

	int current_red_value;
	int current_blue_value;
	int current_green_value;

	int start_x;
	int start_y;

	float circle_radius_squared = pow(circle_radius, 2);

	RotationMatrix temp_matrix;

	wxNativePixelData data(buffer_bitmap);

	if ( !data )
	{
	    // ... raw access to bitmap data unavailable, do something else ...
	    return;
	}
	wxNativePixelData::Iterator p(data);
	//UpdateProjCircleRadius();
	angles_and_shifts.Init(refinement_result_to_draw.phi,refinement_result_to_draw.theta, refinement_result_to_draw.psi,0.0,0.0);

	// paint to the bitmap..

	for (int sym_counter = 0; sym_counter < symmetry_matrices.number_of_matrices; sym_counter ++ )
	{
		// Get the rotation matrix for the current orientation and current symmetry-related view
		temp_matrix = symmetry_matrices.rot_mat[sym_counter] * angles_and_shifts.euler_matrix;

		// Rotate a vector which initially points at the north pole
		temp_matrix.RotateCoords(north_pole_x,north_pole_y,north_pole_z,proj_x,proj_y,proj_z);

		// If we are in the southern hemisphere, we will need to plot the equivalent projection in the northen hemisphere
		if (proj_z < 0.0)
		{
			proj_z = - proj_z;
			proj_y = - proj_y;
			proj_x = - proj_x;
		}

		// i'm going to work out the x,y here, so need to convert to angle and back
		sym_theta = deg_2_rad(ConvertProjectionXYToThetaInDegrees(proj_x, proj_y));
		sym_phi = deg_2_rad(ConvertXYToPhiInDegrees(proj_x, proj_y));

//		while (sym_phi > PI) sym_phi -= 2.0*PI;
	//	while (sym_phi < -PI) sym_phi += 2.0*PI;

		proj_x = cos(sym_phi) * sym_theta / 1.5707963f;
		proj_y = sin(sym_phi) * sym_theta / 1.5707963f;

		start_x = myroundint((proj_x * circle_radius) - proj_circle_radius);
		start_y = myroundint((proj_y * circle_radius) - proj_circle_radius);

		p.MoveTo(data, myroundint((circle_center_x + proj_x * circle_radius) - proj_circle_radius), myroundint((circle_center_y + proj_y * circle_radius) - proj_circle_radius));

		for ( int y = 0; y < myroundint(proj_circle_radius * 2); ++y )
		{
		    wxNativePixelData::Iterator rowStart = p;
		    for ( int x = 0; x < myroundint(proj_circle_radius * 2); ++x, ++p )
		    {
		    	if (pow(start_x + x, 2) + pow(start_y + y, 2) > circle_radius_squared) continue;
		    	    current_red_value = p.Red();
		    		current_blue_value = p.Blue();
		    		current_green_value = p.Green();

		    	/*	current_colour = GetColourBarValue(current_value, min_value, top_value);

		    		if (current_red_value < 255 && current_blue_value == 255 && current_green_value == 0)
		    		{
		    			current_red_value+=colour_change_step;
		    			if (current_red_value > 255) current_red_value = 255;
		    			p.Red() = current_red_value;
		    			p.Green() = 0;
		    			p.Blue() = 255;
		    		}
		    		else
		    		if (current_red_value == 255 && current_green_value == 0)
		    		{
		    			current_blue_value-=colour_change_step;
		    			if (current_blue_value < 0) current_blue_value = 0;
		    			p.Red() = 255;
		    			p.Green() = 0;
		    			p.Blue() = current_blue_value;
		    		}
		    		else
		    		{
		    			p.Red() = 0;
		    			p.Green() = 0;
		    			p.Blue() = 255;
		    		}
		    		*/
		    		

		    		if (current_red_value == 255 && current_blue_value == 255 && current_green_value == 255)
		    		{
		      			p.Red() = 0;
		    		    p.Green() = 0;
		    		    p.Blue() = 128;
		    		}
		    		else
		    		if (current_red_value == current_blue_value && current_red_value == current_green_value) // grey, must be label
		    		{
		    			p.Red() = 0;
		    			p.Green() = 0;
		    			p.Blue() = 128;

		    		}
		    		else
		    		if (current_red_value == 0 && current_green_value == 0 && current_blue_value < 255)
		    		{
		    			current_blue_value += colour_change_step;
		    			if (current_blue_value > 255)
		    			{
		    				current_green_value += current_blue_value - 255;
		    				current_blue_value = 255;
		    			}

		    			p.Red() = current_red_value;
		    			p.Green() = current_green_value;
		    			p.Blue() = current_blue_value;
		    		}
		    		else
		    		if (current_red_value == 0 && current_blue_value == 255 && current_green_value < 255)
		    		{
		    			current_green_value += colour_change_step;
		    			if (current_green_value  > 255)
		    			{
		    				current_red_value += current_green_value - 255;
		    				current_green_value = 255;
		    			}

		    			p.Red() = current_red_value;
		    			p.Green() = current_green_value;
		    			p.Blue() = current_blue_value;
		    		}
		    		else
		    		if (current_blue_value == 255 && current_green_value == 255 && current_red_value < 128)
		    		{
		    			current_red_value += colour_change_step;
		    			if (current_red_value > 128)
		    			{
		    				current_blue_value -= current_red_value - 128;
		    			}

		    			p.Red() = current_red_value;
		    			p.Green() = current_green_value;
		    			p.Blue() = current_blue_value;

		    		}
		    		else
		    		if (current_green_value == 255 && current_blue_value > 128 && current_red_value < 255)
		    		{
		    			current_red_value += colour_change_step;
		    			current_blue_value -= colour_change_step;

		    			if (current_red_value > 255)
		    			{
		    				current_blue_value -= current_red_value - 255;
		    				current_red_value = 255;
		    			}

		    			if (current_blue_value < 0)
		    			{
		    				current_green_value += current_blue_value;
		    				current_blue_value = 0;
		    			}

		    			p.Red() = current_red_value;
		    			p.Green() = current_green_value;
		    			p.Blue() = current_blue_value;
		    		}
		    		else
		    		if (current_green_value == 255 && current_red_value == 255 && current_blue_value > 0)
		    		{
		    			current_blue_value -= colour_change_step;
		    			if (current_blue_value < 0)
		    			{
		    				current_green_value += current_blue_value;
		    				current_blue_value = 0;
		    			}

		    			p.Red() = current_red_value;
		    			p.Green() = current_green_value;
		    			p.Blue() = current_blue_value;
		    		}
		    		else
		    		if (current_red_value == 255 && current_blue_value == 0 && current_green_value > 0)
		    		{
		    			current_green_value -= colour_change_step;

		    			if (current_green_value < 0) current_green_value = 0;

		    			p.Red() = current_red_value;
		    			p.Green() = current_green_value;
		    			p.Blue() = current_blue_value;
		    		}
		    		else
		    		if (current_red_value == 255 && current_blue_value == 0 && current_green_value == 0)
		    		{
		    			// already at max
		    		}
		    		else
		    		{
		    			// if we got here, we must have hit a label - so go blue
		    			p.Red() = 0;
		    			p.Green() = 0;
		    			p.Blue() = 128;

		    		}






		    }
		    p = rowStart;
		    p.OffsetY(data, 1);
		}
	}
}

void DistributionPlotPanel::UpdateScalingAndDimensions()
{
	int panel_dim_x, panel_dim_y;
	GetClientSize(&panel_dim_x, &panel_dim_y);

	buffer_bitmap.Create(wxSize(panel_dim_x, panel_dim_y));

	if (draw_axis_overlay_instead_of_underlay == false)
	{
		circle_radius = std::min(panel_dim_x / 2.0f, panel_dim_y / 2.0f) * 0.7;
		circle_center_x = panel_dim_x / 2;
		circle_center_y = panel_dim_y / 2;

		bar_x = circle_center_x + circle_radius + circle_radius * 0.5f;
		bar_y = circle_center_y - circle_radius;

		bar_width = circle_radius * 0.2;
		bar_height = circle_radius * 2.0f;
	}
	else
	{
		circle_radius = std::min(panel_dim_x * 0.4f,panel_dim_y / 2.0f) * 0.7;
		bar_width = circle_radius * 0.2;
		bar_height = circle_radius * 2.0f;

		circle_center_x = (panel_dim_x / 2) - bar_width / 2;
		circle_center_y = panel_dim_y / 2;

		bar_x = circle_center_x + circle_radius + circle_radius * 0.5f;
		bar_y = circle_center_y - circle_radius;

	}

	major_tick_length = circle_radius * 0.05;
	minor_tick_length = major_tick_length * 0.5;

	margin_between_major_ticks_and_labels = std::max(major_tick_length * 0.5,5.0);
	margin_between_circles_and_theta_labels = 2.0;

	UpdateProjCircleRadius();
	if (panel_dim_x > 0 && panel_dim_y > 0) SetupBitmap();

}

void DistributionPlotPanel::OnPaint(wxPaintEvent & evt)
{

	Freeze();
	if (should_show == true)
	{
		wxPaintDC dc(this);
		dc.DrawBitmap(buffer_bitmap, wxPoint(0,0));
	}

    Thaw();


}

void DistributionPlotPanel::AddRefinementResult(RefinementResult * refinement_result_to_add)
{
	//wxPrintf("Adding refinement result to the panel: theta = %f phi = %f\n",refinement_result_to_add->theta, refinement_result_to_add->phi);

	//if (refinement_result_to_add->image_is_active >= 0)
	{
		refinement_results_to_plot.Add(*refinement_result_to_add);

		int panel_dim_x, panel_dim_y;
		GetClientSize(&panel_dim_x, &panel_dim_y);

		if (buffer_bitmap.GetHeight() != panel_dim_y || buffer_bitmap.GetWidth() != panel_dim_x)
		{
			UpdateScalingAndDimensions();
		}

		DrawBlueDot(*refinement_result_to_add);
	}
}

void DistributionPlotPanel::UpdateProjCircleRadius()
{
	proj_circle_radius = circle_radius / 200.0f;
	if (proj_circle_radius < 0.5f) proj_circle_radius = 0.5f; // we can't do less than 1 pixel

	//wxPrintf("radius = %f, square = %i\n", proj_circle_radius, myroundint(float(proj_circle_radius) * 2.0f));

	float circle_area = PI * pow(circle_radius, 2.0f);
	float square_area = pow(myround(proj_circle_radius * 2.0f), 2);
	float filled_area = square_area * number_of_final_results;
	float average_number_expected = filled_area / circle_area;

	// ok, so we want it to go to red when there are 4.5 times the expected number in the pixel..

	float red_value = average_number_expected * 4.5f;
	min_number_of_projections_to_be_red = myroundint(red_value);

	// there are 1022 colours available to show..

	if (draw_axis_overlay_instead_of_underlay == true) colour_change_step = myroundint(1021.0f / (red_value));
	else colour_change_step = myroundint(1022.0f / (red_value));

	if (colour_change_step < 1) colour_change_step = 1; // can't be less than 1.
	//wxPrintf("colour_change_step = %i\n", colour_change_step);

}

void DistributionPlotPanel::SetSymmetryAndNumber(wxString wanted_symmetry_symbol, long wanted_number_of_final_results)
{
	symmetry_matrices.Init(wanted_symmetry_symbol);
	number_of_final_results = wanted_number_of_final_results  * symmetry_matrices.number_of_matrices;

	UpdateProjCircleRadius();
}

void DistributionPlotPanel::DrawAxisOverlay(int min_value, int max_value)
{
	wxDouble label_width;
	wxDouble label_height;
	wxDouble label_descent;
	wxDouble label_externalLeading;
	wxString greek_theta = wxT("\u03B8");
	wxString greek_phi = wxT("\u03C6");
	wxString degree_symbol = wxT("\u00B0");
	wxString current_label;

	wxColour current_colour;

	wxMemoryDC *memDC;

	float tmp_x = 0.0;
	float tmp_y = 0.0;

	int x_position;
	int y_position;

	{
		wxNativePixelData data(buffer_bitmap);
		if ( !data )
		{
			//... raw access to bitmap data unavailable, do something else ...
			return;
		}

		wxNativePixelData::Iterator p(data);

		p.MoveTo(data, myroundint(bar_x + 1), myroundint(bar_y + 1));

		for (y_position = 0; y_position < myroundint(bar_height) - 1; y_position++)
		{
			wxNativePixelData::Iterator rowStart = p;
			current_colour = GetColourBarValue(myroundint(bar_height) - y_position, 1, myroundint(bar_height - 2));

			for (x_position = 0; x_position < myroundint(bar_width) - 1; x_position++ )
			{
				p.Red() = current_colour.Red();
				p.Green() = current_colour.Green();
				p.Blue() = current_colour.Blue();

				p++;
			}

			p = rowStart;
			p.OffsetY(data, 1);
		}

	}

	memDC = new wxMemoryDC;
	memDC->SelectObject(buffer_bitmap);;
	wxGraphicsContext *gc = wxGraphicsContext::Create( *memDC );

	wxGraphicsPath path;
	wxGraphicsPath black_path;

	float line_width = circle_radius / 90.0f;
	if (line_width < 1.0f) line_width = 1.0f;

	black_path = gc->CreatePath();
	gc->SetPen(wxPen(*wxBLACK, line_width ));
	black_path.AddCircle(circle_center_x,circle_center_y, circle_radius);
	black_path.AddRectangle(myroundint(bar_x), myroundint(bar_y), myroundint(bar_width), myroundint(bar_height));
	gc->StrokePath(black_path);

	gc->SetPen(wxPen(*wxWHITE, line_width, wxPENSTYLE_SHORT_DASH));
	path = gc->CreatePath();

	path.AddCircle(circle_center_x,circle_center_y,circle_radius * 0.3333f);
	path.AddCircle(circle_center_x,circle_center_y,circle_radius * 0.6666f);

	gc->StrokePath(path);

	// Write labels
	gc->SetFont( wxFont(circle_radius / 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false) , *wxBLACK );
	current_label = greek_phi+"=0"+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	gc->DrawText(current_label, circle_center_x + circle_radius + major_tick_length + margin_between_major_ticks_and_labels, circle_center_y - label_height * 0.5, 0.0);

	current_label = greek_phi+"=45"+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	tmp_x = 0.5 * sqrt(2.0) * (circle_radius + margin_between_circles_and_theta_labels);
	tmp_y = - 0.5 * sqrt(2.0) * (circle_radius + margin_between_circles_and_theta_labels) - label_height;
	gc->DrawText(current_label, circle_center_x + tmp_x, circle_center_y + tmp_y, 0.0);

	current_label = greek_phi+"=90"+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	gc->DrawText(current_label, circle_center_x - label_width * 0.5, circle_center_y - circle_radius - major_tick_length - margin_between_major_ticks_and_labels - label_height, 0.0);


	current_label = greek_phi+"=135"+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	tmp_x = -0.5 * sqrt(2.0) * (circle_radius + margin_between_circles_and_theta_labels) - label_width;
	tmp_y = - 0.5 * sqrt(2.0) * (circle_radius + margin_between_circles_and_theta_labels) - label_height;
	gc->DrawText(current_label, circle_center_x + tmp_x, circle_center_y + tmp_y, 0.0);


	current_label = greek_phi+"=180"+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	gc->DrawText(current_label, circle_center_x - circle_radius - major_tick_length - margin_between_major_ticks_and_labels - label_width, circle_center_y - label_height * 0.5, 0.0);

	current_label = greek_phi+"=225"+degree_symbol;
	tmp_x = -0.5 * sqrt(2.0) * (circle_radius + margin_between_circles_and_theta_labels) - label_width;
	tmp_y = 0.5 * sqrt(2.0) * (circle_radius + margin_between_circles_and_theta_labels);
	gc->DrawText(current_label, circle_center_x + tmp_x, circle_center_y + tmp_y, 0.0);

	current_label = greek_phi+"=270"+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	gc->DrawText(current_label, circle_center_x - label_width * 0.5, circle_center_y + circle_radius + major_tick_length + margin_between_major_ticks_and_labels, 0.0);

	current_label = greek_phi+"=315"+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	tmp_x = 0.5 * sqrt(2.0) * (circle_radius + margin_between_circles_and_theta_labels);
	tmp_y = 0.5 * sqrt(2.0) * (circle_radius + margin_between_circles_and_theta_labels);
	gc->DrawText(current_label, circle_center_x + tmp_x, circle_center_y + tmp_y, 0.0);

	// draw bar values in black

	//distribution_histogram.GetDistributionStatistics(min_value, max_value, average_value, std_dev);

	//if (min_value > average_value * 0.5) min_value = average_value * 0.5;
	//if (average_value + std_dev * 4.0f < average_value * 2.0) top_value = average_value * 2.0;
	//else top_value = average_value + std_dev * 4.0f;

	//wxPrintf("min = %f, max = %f, avg. = %f\n", min_value, max_value, average_value);

	current_label = wxString::Format("%i", int(min_value));
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	gc->DrawText(current_label, bar_x + (bar_width / 2 - label_width / 2), bar_y + bar_height + margin_between_major_ticks_and_labels);

	current_label = wxString::Format("%i+", int(max_value));
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	gc->DrawText(current_label, bar_x + (bar_width / 2 - label_width / 2), bar_y - (label_height + margin_between_major_ticks_and_labels));

	gc->StrokePath(path);

	gc->SetFont( wxFont(circle_radius / 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false) , *wxWHITE );
	current_label = greek_theta+"=30"+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	gc->DrawText(current_label, circle_center_x - label_width * 0.5f, circle_center_y - (circle_radius * 0.3333f + label_height  + margin_between_circles_and_theta_labels), 0.0);

	current_label = greek_theta+"=60"+degree_symbol;
	gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
	gc->DrawText(current_label, circle_center_x - label_width * 0.5f, circle_center_y - (circle_radius * 0.6666f + label_height  + margin_between_circles_and_theta_labels), 0.0);
	gc->StrokePath(path);

	gc->SetPen(wxNullPen);
	gc->SetBrush(wxNullBrush);
	memDC->SelectObject(wxNullBitmap);
	delete memDC;
	delete gc;
}

