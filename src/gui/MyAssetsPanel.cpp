#include "../core/gui_core_headers.h"

MyAssetsPanel::MyAssetsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
:
AssetsPanel( parent, id, pos, size, style )
{
	// Bind OnListBookPageChanged from
	Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler( MyAssetsPanel::OnAssetsBookPageChanged ), this);
}

// TODO: destructor

void MyAssetsPanel::OnAssetsBookPageChanged(wxBookCtrlEvent& event )
{
	extern MyMovieAssetPanel *movie_asset_panel;
	extern MyImageAssetPanel *image_asset_panel;
	extern MyParticlePositionAssetPanel *particle_position_asset_panel;
	extern MyVolumeAssetPanel *volume_asset_panel;
	extern MyRefinementResultsPanel *refinement_package_asset_panel;


	// Necessary for MacOS to refresh the panels
	movie_asset_panel->Layout();
	movie_asset_panel->Refresh();

	image_asset_panel->Layout();
	image_asset_panel->Refresh();

	particle_position_asset_panel->Layout();
	particle_position_asset_panel->Refresh();

	volume_asset_panel->Layout();
	volume_asset_panel->Refresh();

	refinement_package_asset_panel->Layout();
	refinement_package_asset_panel->Refresh();
}
