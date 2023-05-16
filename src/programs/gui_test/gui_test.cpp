#include "../../core/gui_core_headers.h"
#include "../../gui/BitmapPanel.h"
#include "./gui_test.h"

IMPLEMENT_APP(GuiTestApp)

// This is just here to make it compile
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

//

wxBEGIN_EVENT_TABLE(GuiTestMainFrame, wxFrame)
        EVT_TIMER(1, GuiTestMainFrame::OnTimer)
                wxEND_EVENT_TABLE( )

                        GuiTestMainFrame::GuiTestMainFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
    : wxFrame(NULL, wxID_ANY, title, pos, size), m_timer(this, 1) {

    GuiTestApp::TestCtfNodes(this);
    m_timer.Start(2000);
}

GuiTestMainFrame* gui_test_main_frame;

void GuiTestMainFrame::OnTimer(wxTimerEvent& event) {
    event.GetTimer( ).Stop( );
    //Create a DC for the whole screen area
    wxScreenDC dcScreen;
    //Get the size of the screen/DC
    wxCoord screenWidth;
    wxCoord screenHeight;
    wxRect  dimensions = GetScreenRect( );

    //Create a Bitmap that will later on hold the screenshot image
    //Note that the Bitmap must have a size big enough to hold the screenshot
    //-1 means using the current default colour depth
    wxBitmap screenshot(dimensions.width, dimensions.height, -1);

    //Create a memory DC that will be used for actually taking the screenshot
    wxMemoryDC memDC;
    //Tell the memory DC to use our Bitmap
    //all drawing action on the memory DC will go to the Bitmap now
    memDC.SelectObject(screenshot);
    //Blit (in this case copy) the actual screen on the memory DC
    //and thus the Bitmap
    memDC.Blit((wxCoord)0, //Copy to this X coordinate
               (wxCoord)0, //Copy to this Y coordinate
               dimensions.width, //Copy this width
               dimensions.height, //Copy this height
               (wxDC*)&dcScreen, //From where do we copy?
               dimensions.x, //What's the X offset in the original DC?
               dimensions.y //What's the Y offset in the original DC?
    );
    //Select the Bitmap out of the memory DC by selecting a new
    //uninitialized Bitmap
    memDC.SelectObject(wxNullBitmap);
    //Our Bitmap now has the screenshot, so let's save it :-)
    screenshot.SaveFile("screenshot.png", wxBITMAP_TYPE_PNG);
    // Test if app is in ci_mode
    if ( ci_mode ) {
        // Exit the app
        wxExit( );
    }
}

bool GuiTestApp::OnInit( ) {
    if ( ! wxApp::OnInit( ) )
        return false;
    gui_test_main_frame          = new GuiTestMainFrame("GUI Test", wxPoint(150, 150), wxSize(1024, 500));
    gui_test_main_frame->ci_mode = ci_mode;
    gui_test_main_frame->Show(true);
    gui_test_main_frame->Layout( );
    return true;
}

void GuiTestApp::OnInitCmdLine(wxCmdLineParser& parser) {
    parser.SetDesc(g_cmdLineDesc);
    // must refuse '/' as parameter starter or cannot use "/path" style paths
    parser.SetSwitchChars(wxT("-"));
}

bool GuiTestApp::OnCmdLineParsed(wxCmdLineParser& parser) {
    ci_mode = parser.Found(wxT("c"));

    return true;
}

int GuiTestApp::OnExit( ) {

    return 0;
}
