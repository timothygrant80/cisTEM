#include "../core/gui_core_headers.h"

wxDEFINE_EVENT(MY_ORTH_DRAW_EVENT, MyOrthDrawEvent);

#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayofColors);

ArrayofColors default_colormap;
