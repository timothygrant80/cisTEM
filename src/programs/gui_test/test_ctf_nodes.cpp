#include "../../core/gui_core_headers.h"
#include "../../gui/BitmapPanel.h"
#include "./gui_test.h"

void GuiTestApp::TestCtfNodes(wxFrame* main_frame) {
    PlotCurvePanel* plot_panel;
    BitmapPanel*    bitmap_panel;
    BitmapPanel*    bitmap_panel2;

    plot_panel = new PlotCurvePanel((wxWindow*)main_frame);

    bitmap_panel  = new BitmapPanel((wxWindow*)main_frame, wxID_ANY, wxDefaultPosition, wxSize(300, 300), wxBORDER_SUNKEN, "Powerspectrum");
    bitmap_panel2 = new BitmapPanel((wxWindow*)main_frame, wxID_ANY, wxDefaultPosition, wxSize(300, 300), wxBORDER_SUNKEN, "Powerspectrum");

    wxBoxSizer* bSizer2;
    bSizer2 = new wxBoxSizer(wxHORIZONTAL);
    bSizer2->Add(plot_panel, 1, wxEXPAND | wxALL, 5);
    bSizer2->Add(bitmap_panel, 0, wxEXPAND | wxALL, 5);
    bSizer2->Add(bitmap_panel2, 0, wxEXPAND | wxALL, 5);
    main_frame->SetSizer(bSizer2);
    main_frame->Layout( );

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
    main_frame->CreateStatusBar( );
    main_frame->SetStatusText("Test CTF Nodes");
}