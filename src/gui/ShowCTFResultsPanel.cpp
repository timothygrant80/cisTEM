//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

ShowCTFResultsPanel::ShowCTFResultsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
: ShowCTFResultsParentPanel(parent, id, pos, size, style)
{
	Bind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);
	Bind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);

	CTF2DResultsPanel->font_size_multiplier = 1.5;



}

ShowCTFResultsPanel::~ShowCTFResultsPanel()
{
	Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);
	Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);

}

void ShowCTFResultsPanel::OnFitTypeRadioButton(wxCommandEvent& event)
{
	Freeze();

	if (FitType2DRadioButton->GetValue() == true)
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

	Thaw();
}

void ShowCTFResultsPanel::Clear()
{
	CTF2DResultsPanel->should_show = false;
	CTFPlotPanel->Clear();
	Refresh();
}

void ShowCTFResultsPanel::Draw(wxString diagnostic_filename, bool find_additional_phase_shift, float defocus1, float defocus2, float defocus_angle, float phase_shift, float score, float fit_res, float alias_res)
{
	Image result_image;
	wxString avrot_filename;

	wxFileName image_filename = diagnostic_filename;

	CTFPlotPanel->Clear();

	CTF2DResultsPanel->title_text = wxString::Format(wxT("%s\n"),image_filename.GetFullName());

	if (find_additional_phase_shift == false) // find phase shift
	{
		if (alias_res == 0.0)
		{
			CTF2DResultsPanel->panel_text = wxString::Format(wxT("     Defocus 1 :\t%i Å \n     Defocus 2 :\t%i Å \n     Angle :\t\t%.2f ° \n     Score :\t\t%.3f\n     Fit Res. :    \t%.2f Å \n     Alias. Res. :\tNone "), myroundint(defocus1), myroundint(defocus2), defocus_angle, score, fit_res);
			CTFPlotPanel->title->SetName(wxString::Format(wxT("%s\nDefocus 1: %iÅ, Defocus 2: %iÅ, Angle: %.2f°, Score: %.3f, Fit Res: %.2fÅ, Alias Res: None"), image_filename.GetFullName(), myroundint(defocus1), myroundint(defocus2), defocus_angle, score, fit_res));
		}
		else
		{
			CTF2DResultsPanel->panel_text = wxString::Format(wxT("     Defocus 1 :\t%i Å \n     Defocus 2 :\t%i Å \n     Angle :\t\t%.2f ° \n     Score :\t\t%.3f\n     Fit Res. :    \t%.2f Å \n     Alias. Res. :\t%.2f Å "), myroundint(defocus1), myroundint(defocus2), defocus_angle, score, fit_res, alias_res );
			CTFPlotPanel->title->SetName(wxString::Format(wxT("%s\nDefocus 1: %iÅ, Defocus 2: %iÅ, Angle: %.2f°, Score: %.3f, Fit Res: %.2fÅ, Alias Res: %.2fÅ"), image_filename.GetFullName(), myroundint(defocus1), myroundint(defocus2), defocus_angle, score, fit_res, alias_res));
		}
	}
	else
	{
		if (alias_res == 0.0)
		{
			CTF2DResultsPanel->panel_text = wxString::Format(wxT("     Defocus 1 :\t%i Å \n     Defocus 2 :\t%i Å \n     Angle :\t\t%.2f ° \n     Phase Shift :\t%.2f rad \n     Score :\t\t%.3f\n     Fit Res. :    \t%.2f Å \n     Alias. Res. :\tNone "), myroundint(defocus1), myroundint(defocus2), defocus_angle, phase_shift, score, fit_res);
			CTFPlotPanel->title->SetName(wxString::Format(wxT("%s\nDefocus 1: %iÅ, Defocus 2: %iÅ, Angle: %.2f°, Phase Shift: %.2frad, Score: %.3f, Fit Res: %.2fÅ, Alias Res: None"), image_filename.GetFullName(), myroundint(defocus1), myroundint(defocus2), defocus_angle, phase_shift, score, fit_res));
		}
		else
		{
			CTF2DResultsPanel->panel_text = wxString::Format(wxT("     Defocus 1 :\t%i Å \n     Defocus 2 :\t%i Å \n     Angle :\t\t%.2f ° \n     Phase Shift :\t%.2f rad \n     Score :\t\t%.3f\n     Fit Res. :    \t%.2f Å \n     Alias. Res. :\tNone "), myroundint(defocus1), myroundint(defocus2), defocus_angle, phase_shift, score, fit_res, alias_res);
			CTFPlotPanel->title->SetName(wxString::Format(wxT("%s\nDefocus 1: %iÅ, Defocus 2: %iÅ, Angle: %.2f°, Phase Shift: %.2frad, Score: %.3f, Fit Res: %.2fÅ, Alias Res: None"), image_filename.GetFullName(), myroundint(defocus1), myroundint(defocus2), defocus_angle, phase_shift, score, fit_res, alias_res));
		}
	}

	Freeze();

	result_image.QuickAndDirtyReadSlice(diagnostic_filename.ToStdString(), 1); // diagnostic image..
	ConvertImageToBitmap(&result_image, &CTF2DResultsPanel->PanelBitmap, true);
	CTF2DResultsPanel->should_show = true;
	CTF2DResultsPanel->Refresh();

	avrot_filename = wxFileName::StripExtension(diagnostic_filename);
	avrot_filename += "_avrot.txt";

	NumericTextFile ctf_plot_file(avrot_filename, OPEN_TO_READ);

	int number_of_points = ctf_plot_file.records_per_line;

	float *spatial_frequency = new float[number_of_points];
	float *ctf_fit = new float[number_of_points];
	float *quality_of_fit = new float[number_of_points];
	float *amplitude_spectrum = new float[number_of_points];

	ctf_plot_file.ReadLine(spatial_frequency);
	ctf_plot_file.ReadLine(amplitude_spectrum);
	ctf_plot_file.ReadLine(amplitude_spectrum);
	ctf_plot_file.ReadLine(ctf_fit);
	ctf_plot_file.ReadLine(quality_of_fit);

	for (int counter = 0; counter < number_of_points; counter++)
	{
		CTFPlotPanel->AddPoint(spatial_frequency[counter], ctf_fit[counter], quality_of_fit[counter], amplitude_spectrum[counter]);
	}

	CTFPlotPanel->Draw();

	delete [] spatial_frequency;
	delete [] ctf_fit;
	delete [] quality_of_fit;
	delete [] amplitude_spectrum;

	Thaw();

}
