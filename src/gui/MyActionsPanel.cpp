#include "../core/gui_core_headers.h"

MyActionsPanel::MyActionsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
:
ActionsPanel( parent, id, pos, size, style )
{
	// Bind OnListBookPageChanged from
	Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler( MyActionsPanel::OnActionsBookPageChanged ), this);
}

// TODO: destructor

void MyActionsPanel::OnActionsBookPageChanged(wxBookCtrlEvent& event )
{
	extern MyAlignMoviesPanel *align_movies_panel;
	extern MyFindCTFPanel *findctf_panel;
	extern MyFindParticlesPanel *findparticles_panel;
	extern MyRefine2DPanel *classification_panel;
	extern AbInitio3DPanel *ab_initio_3d_panel;
	extern AutoRefine3DPanel *auto_refine_3d_panel;
	extern MyRefine3DPanel *refine_3d_panel;
	extern RefineCTFPanel *refine_ctf_panel;
	extern Generate3DPanel *generate_3d_panel;
	extern Sharpen3DPanel *sharpen_3d_panel;

	// Necessary for MacOS to refresh the panels
	align_movies_panel->Layout();
	align_movies_panel->Refresh();

	findctf_panel->Layout();
	findctf_panel->Refresh();

	findparticles_panel->Layout();
	findparticles_panel->Refresh();

	classification_panel->Layout();
	classification_panel->Refresh();

	ab_initio_3d_panel->Layout();
	ab_initio_3d_panel->Refresh();

	auto_refine_3d_panel->Layout();
	auto_refine_3d_panel->Refresh();

	refine_3d_panel->Layout();
	refine_3d_panel->Refresh();

	refine_ctf_panel->Layout();
	refine_ctf_panel->Refresh();

	generate_3d_panel->Layout();
	generate_3d_panel->Refresh();

	sharpen_3d_panel->Layout();
	sharpen_3d_panel->Refresh();

}
