#ifndef __PICKING_BITMAP_PANEL_H__
#define __PICKING_BITMAP_PANEL_H__

#include <wx/panel.h>


class PickingBitmapPanel : public wxPanel
{
public :
	wxBitmap PanelBitmap; // buffer for the panel size
	wxString panel_text;



	PickingBitmapPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
	~PickingBitmapPanel();

	void OnPaint(wxPaintEvent & evt);
	void OnEraseBackground(wxEraseEvent& event);
	void SetupPanelBitmap();
	void Clear();



	void AllocateMemoryForParticleCoordinates(const int number_of_particles);
	void SetParticleCoordinatesAndRadius(const int number_of_particles, const double *wanted_x, const double *wanted_y, const float wanted_radius_in_angstroms);
	void SetImageFilename(wxString wanted_filename, float pixel_size);
	void UpdateScalingAndDimensions();
	void UpdateImageInBitmap();
	void Draw();

	bool should_show;
	float font_size_multiplier;

private:
	wxString 	image_in_bitmap_filename;
	Image		image_in_bitmap;
	float		image_in_bitmap_pixel_size;
	float		image_in_bitmap_scaling_factor;

	wxString	image_in_memory_filename;
	Image		image_in_memory;
	float		image_in_memory_pixel_size;

	bool 		draw_circles_around_particles;
	float 		radius_of_circles_around_particles_in_angstroms;
	float 		*particle_coordinates_x_in_angstroms;
	float 		*particle_coordinates_y_in_angstroms;
	int 		number_of_particles;

	bool 		draw_scale_bar;


};




#endif
