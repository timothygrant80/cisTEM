#ifndef __PICKING_BITMAP_PANEL_H__
#define __PICKING_BITMAP_PANEL_H__

#include <wx/panel.h>
#include <vector>
#include <functional>
#include <algorithm>


struct particle_coordinate
{
	float x;
	float y;
	// Constructor
	particle_coordinate(const int &wanted_x, const int &wanted_y)
	{
		x = wanted_x;
		y = wanted_y;
	}
};

WX_DECLARE_OBJARRAY(particle_coordinate, ArrayOfCoordinates);


class PickingBitmapPanel : public wxPanel
{
public :
	wxBitmap PanelBitmap; // buffer for the panel size
	wxString panel_text;



	PickingBitmapPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
	~PickingBitmapPanel();

	void OnSize( wxSizeEvent & event );
	void OnPaint(wxPaintEvent & evt);
	void OnEraseBackground(wxEraseEvent& event);
	void SetupPanelBitmap();
	void Clear();



	void AllocateMemoryForParticleCoordinates(const int number_of_particles);
	void SetParticleCoordinatesAndRadius(const int number_of_particles, const double *wanted_x, const double *wanted_y, const float wanted_radius_in_angstroms);
	void RemoveParticleCoordinatesWithinRectangle();
	bool ParticleCoordinatesShouldBeRemoved(const particle_coordinate &particle_coordinates_to_check);

	void SetImageFilename(wxString wanted_filename, float pixel_size);
	void UpdateScalingAndDimensions();
	void UpdateImageInBitmap( bool force_reload = false );

	// Mouse event handles
	void OnLeftDown(wxMouseEvent & event);
	void OnLeftUp(wxMouseEvent & event);
	void OnMotion(wxMouseEvent & event);


	//
	bool 		should_show;
	float 		font_size_multiplier;
	bool 		draw_circles_around_particles;
	bool		should_high_pass;
	bool 		draw_scale_bar;
	bool		allow_editing_of_coordinates;

private:
	wxString 	image_in_bitmap_filename;
	Image		image_in_bitmap;
	float		image_in_bitmap_pixel_size;
	float		image_in_bitmap_scaling_factor;

	wxString	image_in_memory_filename;
	Image		image_in_memory;
	float		image_in_memory_pixel_size;

	float 										radius_of_circles_around_particles_in_angstroms;
	ArrayOfCoordinates							particle_coordinates_in_angstroms;
	int 										number_of_particles;

	//
	bool		draw_selection_rectangle;
	int			selection_rectangle_start_x;
	int			selection_rectangle_start_y;
	int			selection_rectangle_current_x;
	int			selection_rectangle_current_y;
	float		selection_rectangle_start_x_in_angstroms;
	float		selection_rectangle_start_y_in_angstroms;
	float		selection_rectangle_finish_x_in_angstroms;
	float		selection_rectangle_finish_y_in_angstroms;

	//
	int 		bitmap_x_offset;
	int 		bitmap_y_offset;
	int 		bitmap_width;
	int 		bitmap_height;




};


#endif
