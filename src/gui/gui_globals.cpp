#include "../core/gui_core_headers.h"

wxDEFINE_EVENT(RETURN_PROCESSED_IMAGE_EVT, ReturnProcessedImageEvent);
wxDEFINE_EVENT(RETURN_SHARPENING_RESULTS_EVT, ReturnSharpeningResultsEvent);

#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayofColors);

ArrayofColors default_colormap;
ArrayofColors default_colorbar;
