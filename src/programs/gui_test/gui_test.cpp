#include "../../core/gui_core_headers.h"

class
        GuiTestApp : public wxApp {

  public:
    virtual bool OnInit( );
    virtual int  OnExit( );
};

IMPLEMENT_APP(GuiTestApp)

PlotCurvePanel* plot_panel;

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
    wxBoxSizer* bSizer2;
    bSizer2 = new wxBoxSizer(wxVERTICAL);
    bSizer2->Add(plot_panel, 1, wxEXPAND | wxALL, 5);
    this->SetSizer(bSizer2);
    this->Layout( );

    CTF ctf;
    // CTF with a sample thickness parameter of 100.0
    ctf.Init(300, 2.7, 0.07, 5000, 5000, 0, 1.0, 0.0, 100.0);

    Curve ctf_curve;
    ctf_curve.SetupXAxis(0.0, 0.5, 500);
    ctf_curve.SetYToConstant(1.0);

    ctf_curve.ApplyCTFWithThickness(ctf, 0.0);

    Curve ctf_curve3, ctf_curve1;
    ctf_curve3.SetupXAxis(0.0, 0.5, 500);
    ctf_curve1.SetupXAxis(0.0, 0.5, 500);
    ctf_curve1.SetYToConstant(0.0);

    int counter = 0;
    CTF ctf1;
    for ( float z_level = -495.0; z_level < 500.0; z_level = z_level + 10.0f ) {
        ctf1.Init(300, 2.7, 0.07, 5000 + z_level, 5000 + z_level, 0, 1.0, 0.0, 0.0);
        counter++;
        ctf_curve3.SetYToConstant(1.0);
        ctf_curve3.ApplyCTF(ctf1);
        ctf_curve3.MultiplyBy(ctf_curve3);
        ctf_curve1.AddWith(&ctf_curve3);
    }
    ctf_curve1.MultiplyByConstant(1.0f / counter);

    plot_panel->Initialise("Resolution", "CTF", false, true);
    //Bit of offset for vis
    ctf_curve1.AddConstant(0.01);
    plot_panel->AddCurve(ctf_curve, *wxBLUE);
    plot_panel->AddCurve(ctf_curve1, *wxRED);

    plot_panel->Draw( );

    CreateStatusBar( );
    SetStatusText("Test 001");
}

GuiTestMainFrame* gui_test_main_frame;

bool GuiTestApp::OnInit( ) {
    gui_test_main_frame = new GuiTestMainFrame("GUI Test", wxPoint(50, 50), wxSize(450, 340));

    gui_test_main_frame->Show(true);

    return true;
}

int GuiTestApp::OnExit( ) {

    return 0;
}
