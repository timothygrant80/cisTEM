//#include "../../core/core_headers.h"
#include "../../core/gui_core_headers.h"
#ifdef ENABLE_WEBVIEW
#include "wx/webview.h"
#include "../../gui/WebViewPanel.h"
#endif
class
        MyGuiApp : public wxApp {

  public:
    virtual bool OnInit( );
    virtual int  OnExit( );
};

IMPLEMENT_APP(MyGuiApp)

MyMainFrame* main_frame;

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

#ifdef ENABLE_WEBVIEW
WebViewPanel* web_view_panel;
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

wxConfig* cistem_config;
SETUP_SOCKET_CODES

bool MyGuiApp::OnInit( ) {

    // icons..

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

    wxImage::AddHandler(new wxPNGHandler);

    wxImageList* MenuBookIconImages;
    wxImageList* ActionsSpaBookIconImages;
    wxImageList* ActionsTmBookIconImages;
    wxImageList* AssetsBookIconImages;
    wxImageList* SettingsBookIconImages;

#ifdef EXPERIMENTAL
    wxImageList* ExperimentalBookIconImages;
#endif

    main_frame = new MyMainFrame((wxWindow*)NULL);

    // global config..
    cistem_config = new wxConfig("cisTEM", "TG");
    wxConfig::Set(cistem_config);

    SetupDefaultColorMap( );
    SetupDefaultColorBar( );

    // Left hand Panels
    overview_panel    = new MyOverviewPanel(main_frame->MenuBook, wxID_ANY);
    actions_panel_spa = new ActionsPanelSpa(main_frame->MenuBook, wxID_ANY);
    actions_panel_tm  = new ActionsPanelTm(main_frame->MenuBook, wxID_ANY);
    assets_panel      = new MyAssetsPanel(main_frame->MenuBook, wxID_ANY);
    results_panel     = new MyResultsPanel(main_frame->MenuBook, wxID_ANY);
    settings_panel    = new MySettingsPanel(main_frame->MenuBook, wxID_ANY);
#ifdef EXPERIMENTAL
    experimental_panel = new MyExperimentalPanel(main_frame->MenuBook, wxID_ANY);
#endif

    // Individual Panels
    run_profiles_panel = new MyRunProfilesPanel(settings_panel->SettingsBook);

    movie_asset_panel             = new MyMovieAssetPanel(assets_panel->AssetsBook);
    image_asset_panel             = new MyImageAssetPanel(assets_panel->AssetsBook);
    particle_position_asset_panel = new MyParticlePositionAssetPanel(assets_panel->AssetsBook);
    volume_asset_panel            = new MyVolumeAssetPanel(assets_panel->AssetsBook);
#ifdef EXPERIMENTAL
    atomic_coordinates_asset_panel = new AtomicCoordinatesAssetPanel(assets_panel->AssetsBook);
#endif
    refinement_package_asset_panel = new MyRefinementPackageAssetPanel(assets_panel->AssetsBook);
    align_movies_panel             = new MyAlignMoviesPanel(actions_panel_spa->ActionsBook);
    findctf_panel                  = new MyFindCTFPanel(actions_panel_spa->ActionsBook);
    findparticles_panel            = new MyFindParticlesPanel(actions_panel_spa->ActionsBook);
    classification_panel           = new MyRefine2DPanel(actions_panel_spa->ActionsBook);
    ab_initio_3d_panel             = new AbInitio3DPanel(actions_panel_spa->ActionsBook);
    auto_refine_3d_panel           = new AutoRefine3DPanel(actions_panel_spa->ActionsBook);
    refine_3d_panel                = new MyRefine3DPanel(actions_panel_spa->ActionsBook);
    refine_ctf_panel               = new RefineCTFPanel(actions_panel_spa->ActionsBook);
    generate_3d_panel              = new Generate3DPanel(actions_panel_spa->ActionsBook);
    sharpen_3d_panel               = new Sharpen3DPanel(actions_panel_spa->ActionsBook);

    movie_results_panel      = new MyMovieAlignResultsPanel(results_panel->ResultsBook);
    ctf_results_panel        = new MyFindCTFResultsPanel(results_panel->ResultsBook);
    picking_results_panel    = new MyPickingResultsPanel(results_panel->ResultsBook);
    refine2d_results_panel   = new Refine2DResultsPanel(results_panel->ResultsBook);
    refinement_results_panel = new MyRefinementResultsPanel(results_panel->ResultsBook);

    // The other panels will be "stolen" from actions_panel_spa by "Reparenting" when the menu
    // item is selected in MainFrame.cpp
    match_template_panel         = new MatchTemplatePanel(actions_panel_tm->ActionsBook);
    match_template_results_panel = new MatchTemplateResultsPanel(actions_panel_tm->ActionsBook);
    refine_template_panel        = new RefineTemplatePanel(actions_panel_tm->ActionsBook);
#ifdef EXPERIMENTAL
    refine_template_dev_panel = new RefineTemplateDevPanel(experimental_panel->ExperimentalBook);
#endif

#ifdef ENABLE_WEBVIEW
    web_view_panel = new WebViewPanel(experimental_panel->ExperimentalBook);
#endif

    // Setup list books
    // Since wx 3.1 the resolution of the images needs to be declared when creating the image list

    MenuBookIconImages       = new wxImageList(80, 80);
    ActionsSpaBookIconImages = new wxImageList(48, 48);
    ActionsTmBookIconImages  = new wxImageList(48, 48);
    AssetsBookIconImages     = new wxImageList(48, 48);
    ResultsBookIconImages    = new wxImageList(48, 48);
    SettingsBookIconImages   = new wxImageList(48, 48);

#ifdef EXPERIMENTAL
    ExperimentalBookIconImages = new wxImageList(48, 48);
#endif
    wxLogNull* suppress_png_warnings = new wxLogNull;

    wxBitmap overview_icon_bmp = wxBITMAP_PNG_FROM_DATA(overview_icon);
    wxBitmap assets_icon_bmp   = wxBITMAP_PNG_FROM_DATA(assets_icon);
    wxBitmap action_icon_bmp   = wxBITMAP_PNG_FROM_DATA(action_icon);
    wxBitmap results_icon_bmp  = wxBITMAP_PNG_FROM_DATA(results_icon);
    wxBitmap settings_icon_bmp = wxBITMAP_PNG_FROM_DATA(settings_icon);

#ifdef EXPERIMENTAL
    wxBitmap experimental_icon_bmp = wxBITMAP_PNG_FROM_DATA(experimental_icon);
#endif

    wxBitmap movie_icon_bmp              = wxBITMAP_PNG_FROM_DATA(movie_icon);
    wxBitmap image_icon_bmp              = wxBITMAP_PNG_FROM_DATA(image_icon);
    wxBitmap particle_position_icon_bmp  = wxBITMAP_PNG_FROM_DATA(particle_position_icon);
    wxBitmap virus_icon_bmp              = wxBITMAP_PNG_FROM_DATA(virus_icon);
    wxBitmap refinement_package_icon_bmp = wxBITMAP_PNG_FROM_DATA(refinement_package_icon);
    //wxBitmap ribosome_icon_bmp = wxBITMAP_PNG_FROM_DATA(ribosome_icon);

    wxBitmap movie_align_icon_bmp     = wxBITMAP_PNG_FROM_DATA(movie_align_icon);
    wxBitmap ctf_icon_bmp             = wxBITMAP_PNG_FROM_DATA(ctf_icon);
    wxBitmap find_particles_icon_bmp  = wxBITMAP_PNG_FROM_DATA(particle_position_icon);
    wxBitmap classification_icon_bmp  = wxBITMAP_PNG_FROM_DATA(classification_icon);
    wxBitmap ab_initio_3d_icon_bmp    = wxBITMAP_PNG_FROM_DATA(abinitio_icon);
    wxBitmap refine3d_icon_bmp        = wxBITMAP_PNG_FROM_DATA(growth);
    wxBitmap manual_refine3d_icon_bmp = wxBITMAP_PNG_FROM_DATA(manual_refine_icon);
    wxBitmap refine_ctf_icon_bmp      = wxBITMAP_PNG_FROM_DATA(refine_ctf_icon);
    wxBitmap generate3d_icon_bmp      = wxBITMAP_PNG_FROM_DATA(generate3d_icon);
    wxBitmap sharpen_map_icon_bmp     = wxBITMAP_PNG_FROM_DATA(sharpen_map_icon);

    wxBitmap run_profiles_icon_bmp = wxBITMAP_PNG_FROM_DATA(run_profiles_icon);

    wxBitmap match_template_icon_bmp  = wxBITMAP_PNG_FROM_DATA(match_template_icon);
    wxBitmap refine_template_icon_bmp = wxBITMAP_PNG_FROM_DATA(refine_template_icon);

    delete suppress_png_warnings;

    MenuBookIconImages->Add(overview_icon_bmp);
    MenuBookIconImages->Add(assets_icon_bmp);
    MenuBookIconImages->Add(action_icon_bmp);
    MenuBookIconImages->Add(results_icon_bmp);
    MenuBookIconImages->Add(settings_icon_bmp);

#ifdef EXPERIMENTAL
    MenuBookIconImages->Add(experimental_icon_bmp);
#endif

    ActionsSpaBookIconImages->Add(movie_align_icon_bmp);
    ActionsSpaBookIconImages->Add(ctf_icon_bmp);
    ActionsSpaBookIconImages->Add(find_particles_icon_bmp);
    ActionsSpaBookIconImages->Add(classification_icon_bmp);
    ActionsSpaBookIconImages->Add(ab_initio_3d_icon_bmp);
    ActionsSpaBookIconImages->Add(refine3d_icon_bmp);
    ActionsSpaBookIconImages->Add(manual_refine3d_icon_bmp);
    ActionsSpaBookIconImages->Add(refine_ctf_icon_bmp);
    ActionsSpaBookIconImages->Add(generate3d_icon_bmp);
    ActionsSpaBookIconImages->Add(sharpen_map_icon_bmp);

    AssetsBookIconImages->Add(movie_icon_bmp);
    AssetsBookIconImages->Add(image_icon_bmp);
    AssetsBookIconImages->Add(particle_position_icon_bmp);
    AssetsBookIconImages->Add(virus_icon_bmp);
    AssetsBookIconImages->Add(refinement_package_icon_bmp);
#ifdef EXPERIMENTAL
    AssetsBookIconImages->Add(generate3d_icon_bmp);
#endif

    ResultsBookIconImages->Add(movie_align_icon_bmp);
    ResultsBookIconImages->Add(ctf_icon_bmp);
    ResultsBookIconImages->Add(particle_position_icon_bmp);
    ResultsBookIconImages->Add(classification_icon_bmp);
    ResultsBookIconImages->Add(refine3d_icon_bmp);

    SettingsBookIconImages->Add(run_profiles_icon_bmp);

    ActionsTmBookIconImages->Add(movie_align_icon_bmp);
    ActionsTmBookIconImages->Add(ctf_icon_bmp);
    ActionsTmBookIconImages->Add(match_template_icon_bmp);
    ActionsTmBookIconImages->Add(refine_template_icon_bmp);
    ActionsTmBookIconImages->Add(generate3d_icon_bmp);
    ActionsTmBookIconImages->Add(sharpen_map_icon_bmp);

    main_frame->MenuBook->AssignImageList(MenuBookIconImages);
    actions_panel_spa->ActionsBook->AssignImageList(ActionsSpaBookIconImages);
    actions_panel_tm->ActionsBook->AssignImageList(ActionsTmBookIconImages);
    assets_panel->AssetsBook->AssignImageList(AssetsBookIconImages);
    results_panel->ResultsBook->AssignImageList(ResultsBookIconImages);
    settings_panel->SettingsBook->AssignImageList(SettingsBookIconImages);

#ifdef EXPERIMENTAL
    ExperimentalBookIconImages->Add(refine_template_icon_bmp);
    ExperimentalBookIconImages->Add(refine_template_icon_bmp);

    experimental_panel->ExperimentalBook->AssignImageList(ExperimentalBookIconImages);
#endif

    main_frame->MenuBook->AddPage(overview_panel, "Overview", true, 0);
    main_frame->MenuBook->AddPage(assets_panel, "Assets", false, 1);
    main_frame->MenuBook->AddPage(actions_panel_tm, "Actions", false, 2);
    main_frame->MenuBook->RemovePage(2);

    main_frame->MenuBook->AddPage(actions_panel_spa, "Actions", false, 2);
    main_frame->MenuBook->AddPage(results_panel, "Results", false, 3);
    main_frame->MenuBook->AddPage(settings_panel, "Settings", false, 4);
#ifdef EXPERIMENTAL
    main_frame->MenuBook->AddPage(experimental_panel, "Experimental", false, 5);
#endif

    //main_frame->MenuBook->AppendSeparator();

    assets_panel->AssetsBook->AddPage(movie_asset_panel, "Movies", true, 0);
    assets_panel->AssetsBook->AddPage(image_asset_panel, "Images", false, 1);
    assets_panel->AssetsBook->AddPage(particle_position_asset_panel, "Particle Positions", false, 2);
    assets_panel->AssetsBook->AddPage(volume_asset_panel, "3D Volumes", false, 3);
    assets_panel->AssetsBook->AddPage(refinement_package_asset_panel, "Refine Pkgs.", false, 4);
#ifdef EXPERIMENTAL
    assets_panel->AssetsBook->AddPage(atomic_coordinates_asset_panel, "Atomic Coordinates", false, 5);
#endif

    actions_panel_spa->ActionsBook->AddPage(align_movies_panel, "Align Movies", true, 0);
    actions_panel_spa->ActionsBook->AddPage(findctf_panel, "Find CTF", false, 1);
    actions_panel_spa->ActionsBook->AddPage(findparticles_panel, "Find Particles", false, 2);
    actions_panel_spa->ActionsBook->AddPage(classification_panel, "2D Classify", false, 3);
    actions_panel_spa->ActionsBook->AddPage(ab_initio_3d_panel, "Ab-Initio 3D", false, 4);
    actions_panel_spa->ActionsBook->AddPage(auto_refine_3d_panel, "Auto Refine", false, 5);
    actions_panel_spa->ActionsBook->AddPage(refine_3d_panel, "Manual Refine", false, 6);
    actions_panel_spa->ActionsBook->AddPage(refine_ctf_panel, "Refine CTF", false, 7);
    actions_panel_spa->ActionsBook->AddPage(generate_3d_panel, "Generate 3D", false, 8);
    actions_panel_spa->ActionsBook->AddPage(sharpen_3d_panel, "Sharpen 3D", false, 9);

    actions_panel_tm->ActionsBook->AddPage(align_movies_panel, "Align Movies", true, 0);
    actions_panel_tm->ActionsBook->AddPage(findctf_panel, "Find CTF", false, 1);
    actions_panel_tm->ActionsBook->AddPage(match_template_panel, "Match Templates", false, 2);
    actions_panel_tm->ActionsBook->AddPage(refine_template_panel, "Refine Template", false, 3);
    actions_panel_tm->ActionsBook->AddPage(match_template_results_panel, "MT Results", false, 2);
    actions_panel_tm->ActionsBook->AddPage(generate_3d_panel, "Generate 3D", false, 4);
    actions_panel_tm->ActionsBook->AddPage(sharpen_3d_panel, "Sharpen 3D", false, 5);

    results_panel->ResultsBook->AddPage(movie_results_panel, "Align Movies", true, 0);
    results_panel->ResultsBook->AddPage(ctf_results_panel, "Find CTF", false, 1);
    results_panel->ResultsBook->AddPage(picking_results_panel, "Find Particles", false, 2);
    results_panel->ResultsBook->AddPage(refine2d_results_panel, "2D Classify", false, 3);
    results_panel->ResultsBook->AddPage(refinement_results_panel, "3D Refinement", false, 4);

    settings_panel->SettingsBook->AddPage(run_profiles_panel, "Run Profiles", true, 0);

#ifdef EXPERIMENTAL
    experimental_panel->ExperimentalBook->AddPage(refine_template_dev_panel, "Refine Templates", true, 0);
#endif

#ifdef ENABLE_WEBVIEW
    experimental_panel->ExperimentalBook->AddPage(web_view_panel, "Web View", true, 0);
#endif
    // Setup Movie Panel

    //movie_asset_panel->GroupListBox->InsertItem(0, "All Movies", 0);
    //vie_asset_panel->GroupListBox->SetItemState(0, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);

    // Setup the Asset Tree.

    main_frame->RecalculateAssetBrowser( );

    //Particles_Branch = main_frame->AssetTree->AppendItem(Tree_Root, "Particles (0)");
    //Classes_Branch = main_frame->AssetTree->AppendItem(Tree_Root, "Classes (0)");
    //Maps_Branch = main_frame->AssetTree->AppendItem(Tree_Root, "3D Maps (0)");

    // welcome text..

    //main_frame->StatusText->AppendText(wxT("Welcome to Project...X!\n"));

    //GuiJob test_job;

    //test_job.parent_panel = align_movies_panel;
    //test_job.parent_panel->UpdateJobDetails("Hello\n");

    //delete MenuBookIconImages;
    //delete ActionsBookIconImages;
    //delete AssetsBookIconImages;
    //delete SettingsBookIconImages;

    main_frame->Show( );

    return true;
}

int MyGuiApp::OnExit( ) {
    delete cistem_config;
    /*    main_frame->Destroy();
    overview_panel->Destroy();
    actions_panel_spa->Destroy();
    assets_panel->Destroy();
    settings_panel->Destroy();

    // Individual Panels

    movie_asset_panel->Destroy();
    image_asset_panel->Destroy();
    align_movies_panel->Destroy();
    run_profiles_panel->Destroy();*/

    return 0;
}
