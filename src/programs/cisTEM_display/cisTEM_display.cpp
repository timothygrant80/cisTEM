#include "../../core/gui_core_headers.h"

class DisplayApp : public wxApp {
  public:
    virtual bool OnInit( );
    virtual int  OnExit( );
};

IMPLEMENT_APP(DisplayApp)

MyMainFrame*  main_frame;
DisplayFrame* display_frame;

MyAlignMoviesPanel*   align_movies_panel;
MyFindCTFPanel*       findctf_panel;
MyFindParticlesPanel* findparticles_panel;
MyRefine2DPanel*      classification_panel;
AbInitio3DPanel*      ab_initio_3d_panel;
AutoRefine3DPanel*    auto_refine_3d_panel;
MyRefine3DPanel*      refine_3d_panel;
RefineCTFPanel*       refine_ctf_panel;
Generate3DPanel*      generate_3d_panel;
Sharpen3DPanel*       sharpen_3d_panel;

MyOverviewPanel*           overview_panel;
ActionsPanelSpa*           actions_panel_spa;
ActionsPanelTm*            actions_panel_tm;
AssetsPanel*               assets_panel;
MyResultsPanel*            results_panel;
SettingsPanel*             settings_panel;
MatchTemplatePanel*        match_template_panel;
MatchTemplateResultsPanel* match_template_results_panel;
RefineTemplatePanel*       refine_template_panel;

#ifdef EXPERIMENTAL
ExperimentalPanel*      experimental_panel;
RefineTemplateDevPanel* refine_template_dev_panel;
#endif

MyMovieAssetPanel*            movie_asset_panel;
MyImageAssetPanel*            image_asset_panel;
MyParticlePositionAssetPanel* particle_position_asset_panel;
MyVolumeAssetPanel*           volume_asset_panel;
#ifdef EXPERIMENTAL
AtomicCoordinatesAssetPanel* atomic_coordinates_asset_panel;
#endif
MyRefinementPackageAssetPanel* refinement_package_asset_panel;

MyMovieAlignResultsPanel* movie_results_panel;
MyFindCTFResultsPanel*    ctf_results_panel;
MyPickingResultsPanel*    picking_results_panel;
Refine2DResultsPanel*     refine2d_results_panel;
MyRefinementResultsPanel* refinement_results_panel;

MyRunProfilesPanel* run_profiles_panel;

wxImageList* MenuBookIconImages;
wxImageList* ActionsSpaBookIconImages;
wxImageList* ActionsTmBookIconImages;
wxImageList* AssetsBookIconImages;
wxImageList* ResultsBookIconImages;
wxImageList* SettingsBookIconImages;

bool DisplayApp::OnInit( ) {
    wxInitAllImageHandlers( );
    display_frame = new DisplayFrame(NULL, wxID_ANY, "cisTEM Display", wxPoint(-1, -1), wxSize(-1, -1), wxDEFAULT_FRAME_STYLE);
    display_frame->Layout( );
    display_frame->Show(true);

    return true;
}

int DisplayApp::OnExit( ) {
    return 0;
}
