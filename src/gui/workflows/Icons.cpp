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

    /* It is theoretically possible to get a statically initialized wxImageList to avoid re-instantiating a new
	list every time the workflow is changed, but the tradeoff in added code and complexity is likely not worth 
	it. The number of icons that are being added to these lists are maximally in the tens, which makes such 
	lazy loading relatively inexpensive. If the number of images were in the hundreds to thousands, this may be 
	a worthy solution, but despite being non-optimal, for cisTEM's case this is a perfectly suitable solution. */

	wxImageList* spa_list = nullptr;

    if ( ! spa_list ) {
        spa_list = new wxImageList( );
        spa_list->Add(wxBITMAP_PNG_FROM_DATA(movie_align_icon));
        spa_list->Add(wxBITMAP_PNG_FROM_DATA(ctf_icon));
        spa_list->Add(wxBITMAP_PNG_FROM_DATA(particle_position_icon));
        spa_list->Add(wxBITMAP_PNG_FROM_DATA(classification_icon));
        spa_list->Add(wxBITMAP_PNG_FROM_DATA(abinitio_icon));
        spa_list->Add(wxBITMAP_PNG_FROM_DATA(growth));
        spa_list->Add(wxBITMAP_PNG_FROM_DATA(manual_refine_icon));
        spa_list->Add(wxBITMAP_PNG_FROM_DATA(refine_ctf_icon));
        spa_list->Add(wxBITMAP_PNG_FROM_DATA(generate3d_icon));
        spa_list->Add(wxBITMAP_PNG_FROM_DATA(sharpen_map_icon));
    }

    return spa_list;
}

wxImageList* GetActionsTmBookIconImages( ) {
    wxImageList* tm_list = nullptr;

    if ( ! tm_list ) {
        tm_list = new wxImageList( );
        tm_list->Add(wxBITMAP_PNG_FROM_DATA(movie_align_icon));
        tm_list->Add(wxBITMAP_PNG_FROM_DATA(ctf_icon));
        tm_list->Add(wxBITMAP_PNG_FROM_DATA(match_template_icon));
        tm_list->Add(wxBITMAP_PNG_FROM_DATA(refine_template_icon));
        tm_list->Add(wxBITMAP_PNG_FROM_DATA(generate3d_icon));
        tm_list->Add(wxBITMAP_PNG_FROM_DATA(sharpen_map_icon));
    }

    return tm_list;
}