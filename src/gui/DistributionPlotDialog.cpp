#include "../core/gui_core_headers.h"


DistributionPlotDialog::DistributionPlotDialog (wxWindow *parent, wxWindowID id, const wxString &title, const wxPoint &pos, const wxSize &size, long style)
:
DistributionPlotDialogParent( parent, id, title, pos, size, style)
{
	number_of_data_series = 0;
	number_of_points_in_data_series = 0;
	data_series = NULL;
	data_series_titles = NULL;
	value_on_focus_float = 0.0;
	histogram_upper_bound = 0.0;
	histogram_lower_bound = 0.0;

	/*
	 * GUI stuff
	 */
	int frame_width;
	int frame_height;
	int frame_position_x;
	int frame_position_y;

	main_frame->GetClientSize(&frame_width, &frame_height);
	main_frame->GetPosition(&frame_position_x, &frame_position_y);

	SetSize(wxSize(frame_height, myroundint(float(frame_height * 0.9f))));

	// ok so how big is this dialog now?

	int new_x_pos = (frame_position_x + (frame_width / 2) - (frame_height / 2));
	int new_y_pos = (frame_position_y + (frame_height / 2) - myroundint(float(frame_height) * 0.9f / 2.0f));

	Move(new_x_pos, new_y_pos);

	/*
	 * Get ready for plotting
	 */
	PlotCurvePanelInstance->Initialise("Value","Number of images",false);
}

void DistributionPlotDialog::SetDataSeries(int which_data_series, double * wanted_data_series, int number_of_points_in_series, wxString wanted_title)
{
	MyDebugAssertTrue(which_data_series < number_of_data_series,"Bad number of data series");
	data_series[which_data_series] = wanted_data_series; // we're just copying a pointer
	data_series_titles[which_data_series] = wanted_title; // we're copying the title itself
	number_of_points_in_data_series[which_data_series] = number_of_points_in_series;

	// GUI
	DataSeriesToPlotChoice->Insert(wanted_title,which_data_series);
}

void DistributionPlotDialog::SetNumberOfDataSeries(int wanted_number_of_data_series)
{
	ClearDataSeries();
	//
	number_of_data_series = wanted_number_of_data_series;
	data_series = new double*[number_of_data_series];
	data_series_titles = new wxString[number_of_data_series];
	number_of_points_in_data_series = new int[number_of_data_series];
}

void DistributionPlotDialog::ClearDataSeries()
{
	if (number_of_data_series > 0)
	{
		delete [] data_series_titles;
		delete [] number_of_points_in_data_series;
		delete [] data_series; // we deallocate the array of pointers to the data, but we're not deallocating the actual data
	}
}

void DistributionPlotDialog::OnCopyButtonClick( wxCommandEvent& event )
{
	if (wxTheClipboard->Open())
	{
	  //  wxTheClipboard->SetData( new wxTextDataObject(OutputTextCtrl->GetValue()) );
		//wxTheClipboard->SetData( new wxBitmapDataObject( PlotCurvePanelInstance->) );
		MyDebugPrint("Copying not yet implemented");
	    wxTheClipboard->Close();
	}
}

void DistributionPlotDialog::OnSaveButtonClick(wxCommandEvent &event)
{
	ProperOverwriteCheckSaveDialog *saveFileDialog;
	saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save png image"), "PNG files (*.png)|*.png", ".png");
	if (saveFileDialog->ShowModal() == wxID_CANCEL)
	{
		saveFileDialog->Destroy();
		return;
	}

	// save the file then..
	PlotCurvePanelInstance->SaveScreenshot(saveFileDialog->GetFilename(),wxBITMAP_TYPE_PNG);

	saveFileDialog->Destroy();
}

void DistributionPlotDialog::OnCloseButtonClick(wxCommandEvent &event)
{
	EndModal(0);
}

void DistributionPlotDialog::OnDataSeriesToPlotChoice(wxCommandEvent &event)
{
	int data_series_index = DataSeriesToPlotChoice->GetCurrentSelection();

	MyDebugPrint("Will plot "+data_series_titles[data_series_index]);

	/*
	 * When the user switches data series, we work out the bounds of the histogram for them
	 */
	HistogramComputeAutoBounds(data_series[data_series_index],number_of_points_in_data_series[data_series_index],histogram_lower_bound,histogram_upper_bound);
	LowerBoundNumericCtrl->SetValue(wxString::Format("%f",histogram_lower_bound));
	UpperBoundNumericCtrl->SetValue(wxString::Format("%f",histogram_upper_bound));

	LowerBoundNumericCtrl->SetMinMaxValue(-FLT_MAX,histogram_upper_bound);
	UpperBoundNumericCtrl->SetMinMaxValue(histogram_lower_bound,FLT_MAX);

	ComputeHistogramAndPlotIt();
}

void DistributionPlotDialog::ComputeHistogramAndPlotIt()
{
	int data_series_index = DataSeriesToPlotChoice->GetCurrentSelection();
	const int number_of_bins = 100;

	/*
	 * Compute a histogram
	 */
	Curve my_curve = HistogramFromArray(data_series[data_series_index],number_of_points_in_data_series[data_series_index],number_of_bins,histogram_lower_bound,histogram_upper_bound);

	/*
	 * Redo the plotting
	 */
	PlotCurvePanelInstance->Clear();
	PlotCurvePanelInstance->AddCurve(my_curve,wxColour(0, 0, 255),"Histogram");
	PlotCurvePanelInstance->SetXAxisLabel(data_series_titles[data_series_index]);
	PlotCurvePanelInstance->Draw();
}


void DistributionPlotDialog::OnLowerBoundTextEnter( wxCommandEvent& event )
{
	OnNewLowerBound();
}
void DistributionPlotDialog::OnLowerBoundKillFocus( wxFocusEvent & event )
{
	OnNewLowerBound();
}
void DistributionPlotDialog::OnLowerBoundSetFocus( wxFocusEvent & event )
{
	value_on_focus_float = LowerBoundNumericCtrl->ReturnValue();
}

void DistributionPlotDialog::OnUpperBoundTextEnter( wxCommandEvent& event )
{
	OnNewUpperBound();
}
void DistributionPlotDialog::OnUpperBoundKillFocus( wxFocusEvent & event )
{
	OnNewUpperBound();
}
void DistributionPlotDialog::OnUpperBoundSetFocus( wxFocusEvent & event )
{
	value_on_focus_float = UpperBoundNumericCtrl->ReturnValue();
}

void DistributionPlotDialog::OnNewUpperBound()
{
	UpperBoundNumericCtrl->CheckValues();
	LowerBoundNumericCtrl->SetMinMaxValue(-FLT_MAX,UpperBoundNumericCtrl->ReturnValue());
	histogram_upper_bound = UpperBoundNumericCtrl->ReturnValue();
	ComputeHistogramAndPlotIt();
}

void DistributionPlotDialog::OnNewLowerBound()
{
	LowerBoundNumericCtrl->CheckValues();
	UpperBoundNumericCtrl->SetMinMaxValue(LowerBoundNumericCtrl->ReturnValue(),FLT_MAX);
	histogram_lower_bound = LowerBoundNumericCtrl->ReturnValue();
	ComputeHistogramAndPlotIt();
}

void HistogramFromArray(double *data, int number_of_data, int number_of_bins, double lower_bound, double upper_bound, std::vector<double> &bin_centers, std::vector<double> &histogram)
{

	HistogramComputer hist = HistogramComputer(lower_bound,upper_bound,number_of_bins);

	for (int i=0;i<number_of_data;i++) { hist.AddDatum(data[i]); }

	bin_centers.clear();
	histogram.clear();

	for (int i=0;i<number_of_bins;i++)
	{
		bin_centers.push_back(hist.ReturnCenterOfBin(i));
		histogram.push_back(double(hist.ReturnCountInBin(i)));
	}

	MyDebugAssertTrue(bin_centers.size() == number_of_bins,"Bad vector size");
}

/*
 * Compute reasonable bounds for a histogram
 */
void HistogramComputeAutoBounds(double *data, int &number_of_data, double &min, double &max)
{
	/*
	 * Find min,max
	 */
	min = DBL_MAX;
	max = -DBL_MAX;
	for (int i=0;i<number_of_data;i++)
	{
		if (data[i] > max) max = data[i];
		if (data[i] < min) min = data[i];
	}

	/*
	 * Upper bound is exclusive, so let's bump the max slightly
	 * so that we don't automatically exclude the highest datum
	 * from the histogram
	 */
	max += (max-min)*0.001;
}

Curve HistogramFromArray(double *data, int number_of_data, int number_of_bins, double lower_bound, double upper_bound)
{

	std::vector<double> bin_centers, histogram_values;

	/*
	 * If the user gave 0.0 for both lower and upper bound,
	 * we take this as a hint that we should work it out ourselves
	 */
	if (lower_bound == 0.0 && upper_bound == 0.0) HistogramComputeAutoBounds(data,number_of_data,lower_bound,upper_bound);

	HistogramFromArray(data,number_of_data,number_of_bins,lower_bound,upper_bound,bin_centers,histogram_values);

	Curve my_curve;

	for (int i=0;i<number_of_bins;i++)
	{
		my_curve.AddPoint(float(bin_centers.at(i)),float(histogram_values.at(i)));
	}

	return my_curve;
}

HistogramComputer::HistogramComputer(double min, double max, int numberOfBins)
{
	lower_bound = min;
	upper_bound = max;
	number_of_bins = numberOfBins;

	//
	bin_width = (upper_bound-lower_bound)/double(number_of_bins);
	lower_outlier_count = 0;
	upper_outlier_count = 0;
	count_per_bin = new int[number_of_bins];
	for (int i=0;i<number_of_bins;i++) {count_per_bin[i] = 0;}
}

HistogramComputer::~HistogramComputer()
{
	delete [] count_per_bin;
}

// Credit: https://codereview.stackexchange.com/a/38931
void HistogramComputer::AddDatum(double datum) {
    int bin = (int)((datum - lower_bound) / bin_width);
    if (bin < 0)
    {
        lower_outlier_count++;
    }
    else if (bin >= number_of_bins)
    {
        upper_outlier_count++;
    }
    else
    {
        count_per_bin[bin]++;
    }
}

double HistogramComputer::ReturnCenterOfBin(int bin)
{
	return lower_bound + double(bin) * bin_width + 0.5 * bin_width;
}

int HistogramComputer::ReturnCountInBin(int bin)
{
	return count_per_bin[bin];
}
