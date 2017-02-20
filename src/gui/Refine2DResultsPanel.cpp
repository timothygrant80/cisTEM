#include "../core/gui_core_headers.h"

Refine2DResultsPanel::Refine2DResultsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
: Refine2DResultsPanelParent(parent, id, pos, size, style)
{

	ClassumDisplayPanel->Initialise(CAN_FFT);
	ParticleDisplayPanel->Initialise(CAN_FFT | START_WITH_INVERTED_CONTRAST | START_WITH_AUTO_CONTRAST);

}

void Refine2DResultsPanel::OnButtonClick(wxCommandEvent &event)
{
	wxArrayLong wanted_images = main_frame->current_project.database.Return2DClassMembers(359, 3);

	ClassumDisplayPanel->OpenFile("/home/grantt/Apo/Assets/ClassAverages/class_averages_0359.mrc", "Class. #359");
	ParticleDisplayPanel->OpenFile("/home/grantt/Apo/Assets/ParticleStacks/particle_stack_1.mrc", "Class. #359 Members", &wanted_images);
}
