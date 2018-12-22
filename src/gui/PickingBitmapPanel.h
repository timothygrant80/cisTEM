#ifndef __PICKING_BITMAP_PANEL_H__
#define __PICKING_BITMAP_PANEL_H__

#include <wx/panel.h>
#include <vector>
#include <functional>
#include <algorithm>

/*
struct particle_coordinate
{
	float x;
	float y;
	int id;
	// Constructor
	particle_coordinate(const float &wanted_x, const float &wanted_y, const int &wanted_id)
	{
		x = wanted_x;
		y = wanted_y;
		id = wanted_id;
	}
};
*/

WX_DECLARE_OBJARRAY(ArrayOfParticlePositionAssets,ArrayOfCoordinatesHistory);


class PickingBitmapPanel : public wxPanel
{
public :
	wxBitmap PanelBitmap; // buffer for the panel size
	wxString panel_text;

	ArrayOfParticlePositionAssets	particle_coordinates_in_angstroms;



	PickingBitmapPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
	~PickingBitmapPanel();

	void OnSize( wxSizeEvent & event );
	void OnPaint(wxPaintEvent & evt);
	void OnEraseBackground(wxEraseEvent& event);
	void SetupPanelBitmap();
	void Clear();



	void SetParticleCoordinatesAndRadius(const ArrayOfParticlePositionAssets &array_of_assets, const float wanted_radius_in_angstroms);
	int RemoveParticleCoordinatesWithinRectangleOrNearClickedPoint();
	bool ParticleCoordinatesAreWithinRectangle(const ParticlePositionAsset &particle_coordinates_to_check);
	bool ParticleCoordinatesAreNearClickedPoint(const ParticlePositionAsset &particle_coordinates_to_check);
	void EmptyHistoryOfParticleCoordinates();
	void StepForwardInHistoryOfParticleCoordinates();
	void StepBackwardInHistoryOfParticleCoordinates();
	void SetCurrentAsLastStepInHistoryOfParticleCoordinates();
	float PixelToAngstromX(const int &x_in_pixels);
	float PixelToAngstromY(const int &y_in_pixels);

	bool UserHasEditedParticleCoordinates();
	void ResetHistory();


	void SetImageFilename(wxString wanted_filename, const float &pixel_size, CTF ctf_of_image);
	void UpdateScalingAndDimensions();
	void UpdateImageInBitmap( bool force_reload = false );
	void SetCTFOfImageInMemory(CTF ctf_to_copy);
	void SetCTFOfImageInBitmap(CTF ctf_to_copy);

	// Mouse event handles
	void OnLeftDown(wxMouseEvent & event);
	void OnLeftUp(wxMouseEvent & event);
	void OnMotion(wxMouseEvent & event);
	wxCursor CreatePaintCursor();


	//
	bool 		should_show;
	float 		font_size_multiplier;
	bool		size_is_dirty;
	bool 		draw_circles_around_particles;
	bool		should_high_pass;
	bool 		should_low_pass;
	bool		should_wiener_filter;
	bool 		draw_scale_bar;
	bool		allow_editing_of_coordinates;

	float low_res_filter_value;
	float high_res_filter_value;

private:
	wxString 	image_in_bitmap_filename;
	Image		image_in_bitmap;
	float		image_in_bitmap_pixel_size;
	float		image_in_bitmap_scaling_factor;
	CTF			image_in_bitmap_ctf;

	wxString	image_in_memory_filename;
	Image		image_in_memory;
	float		image_in_memory_pixel_size;
	CTF			image_in_memory_ctf;

	float 										radius_of_circles_around_particles_in_angstroms;
	float 										squared_radius_of_circles_around_particles_in_angstroms;
	ArrayOfCoordinatesHistory					particle_coordinates_in_angstroms_history; // for storing undo history
	size_t										current_step_in_history;

	//
	bool		draw_selection_rectangle;
	bool        doing_shift_delete;
	int			selection_rectangle_start_x;
	int			selection_rectangle_start_y;
	int			selection_rectangle_current_x;
	int			selection_rectangle_current_y;
	float		selection_rectangle_start_x_in_angstroms;
	float		selection_rectangle_start_y_in_angstroms;
	float		selection_rectangle_finish_x_in_angstroms;
	float		selection_rectangle_finish_y_in_angstroms;
	int			clicked_point_x;
	int			clicked_point_y;
	float		clicked_point_x_in_angstroms;
	float		clicked_point_y_in_angstroms;

	//
	int 		bitmap_x_offset;
	int 		bitmap_y_offset;
	int 		bitmap_width;
	int 		bitmap_height;




};


#endif
