//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

ShowCTFResultsPanel::ShowCTFResultsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : ShowCTFResultsPanelParent(parent, id, pos, size, style) {
    Bind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);
    Bind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);

    CTF2DResultsPanel->font_size_multiplier = 1.5;
    CTF2DResultsPanel->use_auto_contrast    = false;

    ImageDisplayPanel->EnableCanFFT( );
    ImageDisplayPanel->EnableNoNotebook( );
    ImageDisplayPanel->EnableFirstLocationOnly( );
    ImageDisplayPanel->EnableStartWithAutoContrast( );
    ImageDisplayPanel->EnableStartWithFourierScaling( );
    ImageDisplayPanel->EnableDoNotShowStatusBar( );
    ImageDisplayPanel->Initialise( );
}

ShowCTFResultsPanel::~ShowCTFResultsPanel( ) {
    Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);
    Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);
}

void ShowCTFResultsPanel::OnFitTypeRadioButton(wxCommandEvent& event) {
    Freeze( );

    /*	if (FitType2DRadioButton->GetValue() == true)
	{
		CTF2DResultsPanel->Show(true);
		CTFPlotPanel->Show(false);
		Layout();
	}
	else
	{
		CTF2DResultsPanel->Show(false);
		CTFPlotPanel->Show(true);
		Layout();
	}
*/
    Thaw( );
}

void ShowCTFResultsPanel::Clear( ) {
    CTF2DResultsPanel->should_show = false;
    CTFPlotPanel->Clear( );

    Defocus1Text->SetLabel("");
    Defocus2Text->SetLabel("");
    AngleText->SetLabel("");

    PhaseShiftText->SetLabel("");
    ScoreText->SetLabel("");
    FitResText->SetLabel("");
    AliasResText->SetLabel("");
    ImageFileText->SetLabel("");

    Refresh( );
}

void ShowCTFResultsPanel::Draw(wxString diagnostic_filename, bool find_additional_phase_shift, float defocus1, float defocus2, float defocus_angle, float phase_shift, float score, float fit_res, float alias_res, float iciness, float tilt_angle, float tilt_axis, float sample_thickness, wxString ImageFile) {
    Image    result_image;
    wxString avrot_filename;

    wxFileName image_filename = diagnostic_filename;

    Freeze( );

    CTFPlotPanel->Clear( );

    //CTF2DResultsPanel->title_text = wxString::Format(wxT("%s\n"),image_filename.GetFullName());
    Defocus1Text->SetLabel(wxString::Format(wxT("%i Å"), myroundint(defocus1)));
    Defocus2Text->SetLabel(wxString::Format(wxT("%i Å"), myroundint(defocus2)));
    AngleText->SetLabel(wxString::Format(wxT("%.2f °"), defocus_angle));

    if ( find_additional_phase_shift == true )
        PhaseShiftText->SetLabel(wxString::Format(wxT("%.2f °"), rad_2_deg(phase_shift)));
    else
        PhaseShiftText->SetLabel(wxT("0.00 °"));

    ScoreText->SetLabel(wxString::Format(wxT("%.3f"), score));
    FitResText->SetLabel(wxString::Format(wxT("%.2f Å"), fit_res));

    if ( alias_res == 0.0 )
        AliasResText->SetLabel("None");
    else
        AliasResText->SetLabel(wxString::Format(wxT("%.2f Å"), alias_res));

    IcinessStaticText->SetLabel(wxString::Format(wxT("%.2f"), iciness));
    TiltAngleStaticText->SetLabel(wxString::Format(wxT("%.2f  °"), tilt_angle));
    TiltAxisStaticText->SetLabel(wxString::Format(wxT("%.2f  °"), tilt_axis));
    ThicknessStaticText->SetLabel(wxString::Format(wxT("%.2f  Å"), sample_thickness));
    if ( DoesFileExist(diagnostic_filename) == true ) {
        CTF2DResultsPanel->PanelImage.QuickAndDirtyReadSlice(diagnostic_filename.ToStdString( ), 1);
        CTF2DResultsPanel->should_show = true;
        CTF2DResultsPanel->Refresh( );
    }

    avrot_filename = wxFileName::StripExtension(diagnostic_filename);
    avrot_filename += "_avrot.txt";

    if ( DoesFileExist(avrot_filename) == true ) {
        NumericTextFile ctf_plot_file(avrot_filename, OPEN_TO_READ);

        int number_of_points = ctf_plot_file.records_per_line;

        float* spatial_frequency  = new float[number_of_points];
        float* ctf_fit            = new float[number_of_points];
        float* quality_of_fit     = new float[number_of_points];
        float* amplitude_spectrum = new float[number_of_points];

        ctf_plot_file.ReadLine(spatial_frequency);
        ctf_plot_file.ReadLine(amplitude_spectrum);
        ctf_plot_file.ReadLine(amplitude_spectrum);
        ctf_plot_file.ReadLine(ctf_fit);
        ctf_plot_file.ReadLine(quality_of_fit);

        // smooth the amplitude spectra

        Curve amplitude_plot;

        for ( int counter = 0; counter < number_of_points; counter++ ) {
            amplitude_plot.AddPoint(spatial_frequency[counter], amplitude_spectrum[counter]);
        }

        amplitude_plot.FitSavitzkyGolayToData(7, 3);

        for ( int counter = 0; counter < number_of_points; counter++ ) {
            //CTFPlotPanel->AddPoint(spatial_frequency[counter], ctf_fit[counter], quality_of_fit[counter], amplitude_spectrum[counter]);
            CTFPlotPanel->AddPoint(spatial_frequency[counter], ctf_fit[counter], quality_of_fit[counter], amplitude_plot.savitzky_golay_fit[counter]);
        }

        CTFPlotPanel->Draw( );

        delete[] spatial_frequency;
        delete[] ctf_fit;
        delete[] quality_of_fit;
        delete[] amplitude_spectrum;
    }

    ImageFileText->SetLabel(wxString::Format("(%s)", wxFileName(ImageFile).GetFullName( )));

    wxString small_image_filename = main_frame->current_project.image_asset_directory.GetFullPath( );
    ;
    small_image_filename += wxString::Format("/Scaled/%s", wxFileName(ImageFile).GetFullName( ));

    if ( DoesFileExist(small_image_filename) == true ) {
        ImageDisplayPanel->ChangeFile(small_image_filename, "");
    }
    else if ( DoesFileExist(ImageFile) == true ) {
        ImageDisplayPanel->ChangeFile(ImageFile, "");
    }

    Thaw( );
}
