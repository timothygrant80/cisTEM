#include "Icons.h"
#include "../../gui/icons/overview_icon.cpp"
#include "../../gui/icons/assets_icon.cpp"
#include "../../gui/icons/action_icon.cpp"
#include "../../gui/icons/results_icon.cpp"
#include "../../gui/icons/settings_icon.cpp"
#include "../../gui/icons/experimental_icon.cpp"
//#include "../../gui/icons/settings_icon2.cpp"

#include "../../gui/icons/movie_icon.cpp"
#include "../../gui/icons/image_icon.cpp"
#include "../../gui/icons/particle_position_icon.cpp"
#include "../../gui/icons/virus_icon.cpp"
#include "../../gui/icons/refinement_package_icon.cpp"
//#include "../../gui/icons/ribosome_icon.cpp"

#include "../../gui/icons/movie_align_icon.cpp"
#include "../../gui/icons/ctf_icon.cpp"
#include "../../gui/icons/2d_classification_icon.cpp"
//    #include "../../gui/icons/tool_icon.cpp"
#include "../../gui/icons/abinitio_icon.cpp"
#include "../../gui/icons/growth.cpp"
#include "../../gui/icons/manual_refine_icon.cpp"
#include "../../gui/icons/refine_ctf_icon.cpp"
#include "../../gui/icons/generate3d_icon.cpp"
#include "../../gui/icons/sharpen_map_icon.cpp"

#include "../../gui/icons/run_profiles_icon.cpp"
#include "../../gui/icons/match_template_icon.cpp"
#include "../../gui/icons/refine_template_icon.cpp"

#include <wx/bitmap.h>

wxImageList* GetActionsSpaBookIconImages( ) {
    wxImageList* list = nullptr;
    if ( ! list ) {
        list = new wxImageList( );
        list->Add(wxBITMAP_PNG_FROM_DATA(movie_align_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(ctf_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(particle_position_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(classification_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(abinitio_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(growth));
        list->Add(wxBITMAP_PNG_FROM_DATA(manual_refine_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(refine_ctf_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(generate3d_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(sharpen_map_icon));
    }

    return list;
}

wxImageList* GetActionsTmBookIconImages( ) {
    wxImageList* list = nullptr;
    if ( ! list ) {
        list = new wxImageList( );
        list->Add(wxBITMAP_PNG_FROM_DATA(movie_align_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(ctf_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(match_template_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(refine_template_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(generate3d_icon));
        list->Add(wxBITMAP_PNG_FROM_DATA(sharpen_map_icon));
    }

    return list;
}