#include "../core/gui_core_headers.h"

#include <wx/arrimpl.cpp>
WX_DEFINE_OBJARRAY(ArrayOfRefinementResults);


AngularDistributionPlotPanel::AngularDistributionPlotPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
: wxPanel(parent, id, pos, size, style, name)
{



	should_show = true;
	font_size_multiplier = 1.0;

	Bind(wxEVT_PAINT, &AngularDistributionPlotPanel::OnPaint, this);
	Bind(wxEVT_SIZE,  &AngularDistributionPlotPanel::OnSize, this);


}

AngularDistributionPlotPanel::~AngularDistributionPlotPanel()
{
	Unbind(wxEVT_PAINT, &AngularDistributionPlotPanel::OnPaint, this);
	Unbind(wxEVT_SIZE,  &AngularDistributionPlotPanel::OnSize, this);
}

void AngularDistributionPlotPanel::Clear()
{
	Freeze();
    wxClientDC dc(this);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear();
    Thaw();
}


void AngularDistributionPlotPanel::OnSize(wxSizeEvent & event)
{
	UpdateScalingAndDimensions();
	event.Skip();
}

void AngularDistributionPlotPanel::UpdateScalingAndDimensions()
{
	int panel_dim_x, panel_dim_y;
	GetClientSize(&panel_dim_x, &panel_dim_y);

	circle_center_x = panel_dim_x / 2;
	circle_center_y = panel_dim_y / 2;
	circle_radius = std::min(circle_center_x,circle_center_y) * 0.7;
	major_tick_length = circle_radius * 0.05;
	minor_tick_length = major_tick_length * 0.5;

	margin_between_major_ticks_and_labels = std::max(major_tick_length * 0.5,5.0);
	margin_between_circles_and_theta_labels = 2.0;
}

void AngularDistributionPlotPanel::OnPaint(wxPaintEvent & evt)
{

	Freeze();

	int window_x_size;
	int window_y_size;

	float proj_x;
	float proj_y;
	float proj_z;

	float north_pole_x = 0.0;
	float north_pole_y = 0.0;
	float north_pole_z = 1.0;

	float tmp_x = 0.0;
	float tmp_y = 0.0;
	float tmp_angle = 0.0;

	wxString plot_title;

	RotationMatrix temp_matrix;

    wxPaintDC dc(this);

    wxGraphicsContext *gc = wxGraphicsContext::Create( dc );


    dc.SetBackground(*wxWHITE_BRUSH);
	dc.Clear();
	dc.GetSize(&window_x_size, &window_y_size);
	dc.DrawRectangle(0, 0, window_x_size, window_y_size);

	if (should_show)
	{
		dc.DrawText(wxString::Format("Plot of projection directions (%li projections)",refinement_results_to_plot.Count()),10,10);

		// Draw small circles for each projection direction
		UpdateProjCircleRadius();
		gc->SetPen( wxNullPen );
		//gc->SetPen( wxPen(wxColor(255,0,0),2) );
		gc->SetBrush( wxBrush(wxColor(50,50,200,60)) );
		for (size_t counter = 0; counter < refinement_results_to_plot.Count(); counter ++ )
		{

			// Setup a angles and shifts
			angles_and_shifts.Init(refinement_results_to_plot.Item(counter).phi,refinement_results_to_plot.Item(counter).theta,refinement_results_to_plot.Item(counter).psi,0.0,0.0);


			gc->BeginLayer(50.0);


			// Loop over symmetry-related views
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

				// Do the actual plotting
				wxGraphicsPath path = gc->CreatePath();
				path.AddCircle(circle_center_x + proj_x * circle_radius,circle_center_y + proj_y * circle_radius,proj_circle_radius);
				gc->DrawPath(path);

			}

			gc->EndLayer();
		}

		wxGraphicsPath path;

		// Draw intermediate circles
		path = gc->CreatePath();
		gc->SetPen( *wxBLACK_DASHED_PEN );
		path.AddCircle(circle_center_x,circle_center_y,circle_radius * sin(PI * 0.5 / 9.0)); // 10 degrees
		path.AddCircle(circle_center_x,circle_center_y,circle_radius * sin(PI * 0.25)); // 45 degrees
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
		gc->DrawText(current_label, circle_center_x - label_height * 0.5, circle_center_y - circle_radius - major_tick_length - margin_between_major_ticks_and_labels, PI * 0.5);

		current_label = greek_phi+" = 180 "+degree_symbol;
		gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
		gc->DrawText(current_label, circle_center_x - circle_radius - major_tick_length - margin_between_major_ticks_and_labels - label_width, circle_center_y - label_height * 0.5, 0.0);

		current_label = greek_phi+" = 270 "+degree_symbol;
		gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
		gc->DrawText(current_label, circle_center_x + label_height * 0.5, circle_center_y + circle_radius + major_tick_length + margin_between_major_ticks_and_labels, PI * 1.5);

		current_label = greek_theta+" = 90 "+degree_symbol;
		gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
		tmp_x = 0.5 * sqrt(2.0) * (circle_radius + margin_between_circles_and_theta_labels);
		tmp_y = - 0.5 * sqrt(2.0) * (circle_radius + margin_between_circles_and_theta_labels) - label_height;
		gc->DrawText(current_label, circle_center_x + tmp_x, circle_center_y + tmp_y, 0.0);

		current_label = greek_theta+" = 10 "+degree_symbol;
		gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
		tmp_x = 0.5 * sqrt(2.0) * (circle_radius * sin(PI * 0.5 / 9.0) + margin_between_circles_and_theta_labels);
		tmp_y = - 0.5 * sqrt(2.0) * (circle_radius * sin(PI * 0.5 / 9.0) + margin_between_circles_and_theta_labels) - label_height;
		gc->DrawText(current_label, circle_center_x + tmp_x, circle_center_y + tmp_y, 0.0);

		current_label = greek_theta+" = 45 "+degree_symbol;
		gc->GetTextExtent(current_label,&label_width,&label_height,&label_descent,&label_externalLeading);
		tmp_x = 0.5 * sqrt(2.0) * (circle_radius * sin(PI * 0.25) + margin_between_circles_and_theta_labels);
		tmp_y = - 0.5 * sqrt(2.0) * (circle_radius * sin(PI * 0.25) + margin_between_circles_and_theta_labels) - label_height;
		gc->DrawText(current_label, circle_center_x + tmp_x, circle_center_y + tmp_y, 0.0);



	}
	delete gc;

	Thaw();


}

/*
float AngularDistributionPlotPanel::ReturnRadiusFromTheta(const float theta)
{
	return sin(theta / 180.0 * PI) * float(circle_radius);
}


void AngularDistributionPlotPanel::XYFromPhiTheta(const float phi, const float theta, int &x, int &y)
{
	float radius = ReturnRadiusFromTheta(theta);
	float phi_rad = phi / 180.0 * PI;

	// check whether mod(theta,360) is greater than 90, and less than 270, in which case, we need to fold psi around

	// also, work out symetry-related views

	x = cos(phi_rad) * radius;
	y = sin(phi_rad) * radius;
}
*/

void AngularDistributionPlotPanel::AddRefinementResult(RefinementResult * refinement_result_to_add)
{
	//wxPrintf("Adding refinement result to the panel: theta = %f phi = %f\n",refinement_result_to_add->theta, refinement_result_to_add->phi);
	refinement_results_to_plot.Add(refinement_result_to_add);
}

void AngularDistributionPlotPanel::UpdateProjCircleRadius()
{
	const float	maximum_proj_circle_radius = 15.0;
	const float minimum_proj_circle_radius = 1.0;
	const float minimum_log = 1.0;
	const float maximum_log = 5.0;

	const float	log_num_of_projs = logf(float(refinement_results_to_plot.Count() * symmetry_matrices.number_of_matrices)) / logf(10);



	proj_circle_radius = maximum_proj_circle_radius - (maximum_proj_circle_radius - minimum_proj_circle_radius) * (log_num_of_projs - minimum_log) / (maximum_log - minimum_log);

	if (proj_circle_radius < minimum_proj_circle_radius) proj_circle_radius = minimum_proj_circle_radius;
	if (proj_circle_radius > maximum_proj_circle_radius) proj_circle_radius = maximum_proj_circle_radius;


	//wxPrintf("Number of projections (log): %li (%f). Radius = %f\n",refinement_results_to_plot.Count(),log_num_of_projs, proj_circle_radius);
}

void AngularDistributionPlotPanel::SetSymmetry(wxString wanted_symmetry_symbol)
{
	symmetry_matrices.Init(wanted_symmetry_symbol);
}


