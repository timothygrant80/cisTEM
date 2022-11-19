#include "../../core/gui_core_headers.h"
#include "../../gui/BitmapPanel.h"

class
        GuiTestApp : public wxApp {

  public:
    virtual bool OnInit( );
    virtual int  OnExit( );
};

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

// I really need these

PlotCurvePanel* plot_panel;
BitmapPanel*    bitmap_panel;
BitmapPanel*    bitmap_panel2;

class GuiTestMainFrame : public wxFrame {
  public:
    GuiTestMainFrame(const wxString& title, const wxPoint& pos, const wxSize& size);

  private:
    //void OnHello(wxCommandEvent& event);
    //void OnExit(wxCommandEvent& event);
    //void OnAbout(wxCommandEvent& event);
    //wxDECLARE_EVENT_TABLE( );
};

GuiTestMainFrame::GuiTestMainFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
    : wxFrame(NULL, wxID_ANY, title, pos, size) {

    plot_panel = new PlotCurvePanel((wxWindow*)this);

    bitmap_panel  = new BitmapPanel((wxWindow*)this, wxID_ANY, wxDefaultPosition, wxSize(300, 300), wxBORDER_SUNKEN, "Powerspectrum");
    bitmap_panel2 = new BitmapPanel((wxWindow*)this, wxID_ANY, wxDefaultPosition, wxSize(300, 300), wxBORDER_SUNKEN, "Powerspectrum");

    wxBoxSizer* bSizer2;
    bSizer2 = new wxBoxSizer(wxHORIZONTAL);
    bSizer2->Add(plot_panel, 1, wxEXPAND | wxALL, 5);
    bSizer2->Add(bitmap_panel, 0, wxEXPAND | wxALL, 5);
    bSizer2->Add(bitmap_panel2, 0, wxEXPAND | wxALL, 5);
    this->SetSizer(bSizer2);
    this->Layout( );

    CTF ctf;
    // CTF with a sample thickness parameter of 100.0
    ctf.Init(300, 2.7, 0.07, 5000, 9000, 0, 1.0, 0.0, 100.0);

    Curve ctf_curve;
    ctf_curve.SetupXAxis(0.0, 0.5, 500);
    ctf_curve.SetYToConstant(1.0);

    ctf_curve.ApplyPowerspectrumWithThickness(ctf, 0.0);

    Curve ctf_curve3, ctf_curve1;
    ctf_curve3.SetupXAxis(0.0, 0.5, 500);
    ctf_curve1.SetupXAxis(0.0, 0.5, 500);
    ctf_curve1.SetYToConstant(0.0);

    int   counter = 0;
    CTF   ctf1;
    Image powerspectrum, temp_image;
    powerspectrum.Allocate(500, 500, 1);
    powerspectrum.SetToConstant(0.0);
    temp_image.Allocate(500, 500, 1);
    for ( float z_level = -495.0; z_level < 500.0; z_level = z_level + 10.0f ) {
        ctf1.Init(300, 2.7, 0.07, 5000 + z_level, 9000 + z_level, 0, 1.0, 0.0, 0.0);
        counter++;
        ctf_curve3.SetYToConstant(1.0);
        temp_image.SetToConstant(1.0);
        ctf_curve3.ApplyCTF(ctf1);
        temp_image.ApplyPowerspectrumWithThickness(ctf1);
        ctf_curve3.MultiplyBy(ctf_curve3);
        powerspectrum.AddImage(&temp_image);
        ctf_curve1.AddWith(&ctf_curve3);
    }
    ctf_curve1.MultiplyByConstant(1.0f / counter);
    powerspectrum.DivideByConstant(float(counter));

    plot_panel->Initialise("Resolution", "CTF", false, true);
    //Bit of offset for vis
    ctf_curve1.AddConstant(0.01);
    plot_panel->AddCurve(ctf_curve, *wxBLUE);
    plot_panel->AddCurve(ctf_curve1, *wxRED);

    plot_panel->Draw( );

    bitmap_panel->PanelImage.Allocate(500, 500, 1);
    bitmap_panel->PanelImage.SetToConstant(1.0);
    bitmap_panel->PanelImage.ApplyPowerspectrumWithThickness(ctf);
    bitmap_panel->should_show       = true;
    bitmap_panel->use_auto_contrast = true;
    bitmap_panel->Refresh( );

    bitmap_panel2->PanelImage.Allocate(500, 500, 1);
    bitmap_panel2->PanelImage.CopyFrom(&powerspectrum);
    bitmap_panel2->use_auto_contrast = true;

    bitmap_panel2->should_show = true;
    bitmap_panel2->Refresh( );

    CreateStatusBar( );
    SetStatusText("Test 001");
}

GuiTestMainFrame* gui_test_main_frame;

bool GuiTestApp::OnInit( ) {
    gui_test_main_frame = new GuiTestMainFrame("GUI Test", wxPoint(50, 50), wxSize(1024, 500));

    gui_test_main_frame->Show(true);
    wxMilliSleep(1000);
    //Create a DC for the whole screen area
    wxScreenDC dcScreen(gui_test_main_frame);
    //Get the size of the screen/DC
    wxCoord screenWidth  = 1024;
    wxCoord screenHeight = 500;
    dcScreen.GetSize(&screenWidth, &screenHeight);

    //Create a Bitmap that will later on hold the screenshot image
    //Note that the Bitmap must have a size big enough to hold the screenshot
    //-1 means using the current default colour depth
    wxBitmap screenshot(screenWidth, screenHeight, -1);

    //Create a memory DC that will be used for actually taking the screenshot
    wxMemoryDC memDC;
    //Tell the memory DC to use our Bitmap
    //all drawing action on the memory DC will go to the Bitmap now
    memDC.SelectObject(screenshot);
    //Blit (in this case copy) the actual screen on the memory DC
    //and thus the Bitmap
    memDC.Blit(0, //Copy to this X coordinate
               0, //Copy to this Y coordinate
               screenWidth, //Copy this width
               screenHeight, //Copy this height
               &dcScreen, //From where do we copy?
               0, //What's the X offset in the original DC?
               0 //What's the Y offset in the original DC?
    );
    //Select the Bitmap out of the memory DC by selecting a new
    //uninitialized Bitmap
    memDC.SelectObject(wxNullBitmap);

    //Our Bitmap now has the screenshot, so let's save it :-)
    screenshot.SaveFile("screenshot.png", wxBITMAP_TYPE_PNG);
    return true;
}

int GuiTestApp::OnExit( ) {

    return 0;
}
