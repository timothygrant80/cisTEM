#include "../core/gui_core_headers.h"


Sharpen3DPanel::Sharpen3DPanel( wxWindow* parent )
:
Sharpen3DPanelParent( parent )
{
	ResultDisplayPanel->Initialise(START_WITH_FOURIER_SCALING | DO_NOT_SHOW_STATUS_BAR);
}


