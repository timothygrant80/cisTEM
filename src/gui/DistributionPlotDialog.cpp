#include "../core/gui_core_headers.h"


DistributionPlotDialog::DistributionPlotDialog (wxWindow *parent, wxWindowID id, const wxString &title, const wxPoint &pos, const wxSize &size, long style)
:
DistributionPlotDialogParent( parent, id, title, pos, size, style)
{
	number_of_data_series = 0;
	number_of_points_in_data_series = 0;
	data_series_x = NULL;
	data_series_y = NULL;
	data_series_titles = NULL;
	data_series_x_label = NULL;
	data_series_y_label = NULL;
	plot_histogram_of_this_series = NULL;
	value_on_focus_float = 0.0;
	upper_bound_x = 0.0;
	lower_bound_x = 0.0;
	upper_bound_y = 0.0;
	lower_bound_y = 0.0;

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
	PlotCurvePanelInstance->Initialise("Value","Number of images",false,false,20,50,90,20);
}

void DistributionPlotDialog::SetDataSeries(int which_data_series, double * wanted_data_series_x, double * wanted_data_series_y, int number_of_points_in_series, bool should_plot_histogram, wxString wanted_title, wxString wanted_x_label, wxString wanted_y_label)
{
	MyDebugAssertTrue(which_data_series < number_of_data_series,"Bad number of data series");
	data_series_x[which_data_series] = wanted_data_series_x;
	data_series_y[which_data_series] = wanted_data_series_y; // we're just copying a pointer
	data_series_titles[which_data_series] = wanted_title; // we're copying the title itself
	data_series_x_label[which_data_series] = wanted_x_label;
	data_series_y_label[which_data_series] = wanted_y_label;

	plot_histogram_of_this_series[which_data_series] = should_plot_histogram;

	number_of_points_in_data_series[which_data_series] = number_of_points_in_series;

	// GUI
	DataSeriesToPlotChoice->Insert(wanted_title,which_data_series);
}

void DistributionPlotDialog::SetNumberOfDataSeries(int wanted_number_of_data_series)
{
	ClearDataSeries();
	//
	number_of_data_series = wanted_number_of_data_series;
	data_series_x = new double*[number_of_data_series];
	data_series_y = new double*[number_of_data_series];
	data_series_titles = new wxString[number_of_data_series];
	data_series_x_label = new wxString[number_of_data_series];
	data_series_y_label = new wxString[number_of_data_series];
	plot_histogram_of_this_series = new bool[number_of_data_series];
	number_of_points_in_data_series = new int[number_of_data_series];
}

DistributionPlotDialog::~DistributionPlotDialog()
{
	ClearDataSeries();
}


void DistributionPlotDialog::ClearDataSeries()
{
	if (number_of_data_series > 0)
	{
		delete [] data_series_titles;
		delete [] data_series_x_label;
		delete [] data_series_y_label;
		delete [] plot_histogram_of_this_series;
		delete [] number_of_points_in_data_series;
		delete [] data_series_x;
		delete [] data_series_y; // we deallocate the array of pointers to the data, but we're not deallocating the actual data
	}
}

void DistributionPlotDialog::SelectDataSeries(int which_data_series)
{
	DataSeriesToPlotChoice->Select(which_data_series);
	UpdateCurveToPlot(true);
	DoThePlotting();
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

void DistributionPlotDialog::OnSavePNGButtonClick(wxCommandEvent &event)
{
	ProperOverwriteCheckSaveDialog *saveFileDialog;
	//
	saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save png image"), "PNG files (*.png)|*.png", ".png");
	if (saveFileDialog->ShowModal() == wxID_CANCEL)
	{
		saveFileDialog->Destroy();
		return;
	}

	PlotCurvePanelInstance->SaveScreenshot(saveFileDialog->GetFilename(),wxBITMAP_TYPE_PNG);

	saveFileDialog->Destroy();
}

void DistributionPlotDialog::OnSaveTXTButtonClick(wxCommandEvent &event)
{
	int data_series_index = DataSeriesToPlotChoice->GetCurrentSelection();

	ProperOverwriteCheckSaveDialog *saveFileDialog;
	//
	saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save histogram data"), "TXT files (*.txt)|*.txt", ".txt");
	if (saveFileDialog->ShowModal() == wxID_CANCEL)
	{
		saveFileDialog->Destroy();
		return;
	}

	wxString x_title,y_title;
	x_title = data_series_x_label[data_series_index];
	y_title = data_series_y_label[data_series_index];
	x_title.Replace(" ","_",true);
	y_title.Replace(" ","_",true);

	curve_to_plot.WriteToFile(saveFileDialog->GetFilename(),"#   "+x_title+" "+y_title);

	saveFileDialog->Destroy();
}

void DistributionPlotDialog::OnCloseButtonClick(wxCommandEvent &event)
{
	EndModal(0);
}

void DistributionPlotDialog::OnDataSeriesToPlotChoice(wxCommandEvent &event)
{
	UpdateCurveToPlot(true);
	DoThePlotting();
}

void DistributionPlotDialog::ResetBounds()
{
	int data_series_index = DataSeriesToPlotChoice->GetCurrentSelection();

	/*
	 * Work out bounds
	 */
	{
		float min_x, min_y, max_x, max_y;
		if (!plot_histogram_of_this_series[data_series_index])
		{
			curve_to_plot.GetXMinMax(min_x,max_x);
			lower_bound_x = double(min_x);
			upper_bound_x = double(max_x);
		}
		curve_to_plot.GetYMinMax(min_y,max_y);
		lower_bound_y = double(min_y);
		upper_bound_y = double(max_y);
	}

	/*
	 * Update the GUI
	 */

	LowerBoundXNumericCtrl->SetValue(wxString::Format("%f",lower_bound_x));
	UpperBoundXNumericCtrl->SetValue(wxString::Format("%f",upper_bound_x));
	LowerBoundYNumericCtrl->SetValue(wxString::Format("%f",lower_bound_y));
	UpperBoundYNumericCtrl->SetValue(wxString::Format("%f",upper_bound_y));

	LowerBoundXNumericCtrl->SetMinMaxValue(-FLT_MAX,upper_bound_x);
	UpperBoundXNumericCtrl->SetMinMaxValue(lower_bound_x,FLT_MAX);
	LowerBoundYNumericCtrl->SetMinMaxValue(-FLT_MAX,upper_bound_y);
	UpperBoundYNumericCtrl->SetMinMaxValue(lower_bound_y,FLT_MAX);
}

/*
 * Update the curve object
 */
void DistributionPlotDialog::UpdateCurveToPlot(bool reset_bounds)
{
	int data_series_index = DataSeriesToPlotChoice->GetCurrentSelection();

	/*
	 * Compute a histogram if required. If not, just copy
	 * the data over (the X axis will just be the array index)
	 */
	if (plot_histogram_of_this_series[data_series_index])
	{
		const int number_of_bins = 100;
		if (reset_bounds) HistogramComputeAutoBounds(data_series_y[data_series_index],number_of_points_in_data_series[data_series_index],lower_bound_x,upper_bound_x);
		curve_to_plot = HistogramFromArray(data_series_y[data_series_index],number_of_points_in_data_series[data_series_index],number_of_bins,lower_bound_x,upper_bound_x);
	}
	else
	{
		curve_to_plot.CopyDataFromArrays(data_series_x[data_series_index],data_series_y[data_series_index],number_of_points_in_data_series[data_series_index]);
	}

	/*
	 * Now reset the bounds
	 */
	if (reset_bounds) ResetBounds();
}

void DistributionPlotDialog::DoThePlotting()
{
	int data_series_index = DataSeriesToPlotChoice->GetCurrentSelection();

	UpdateCurveToPlot(false);

	/*
	 * Redo the plotting
	 */
	PlotCurvePanelInstance->Clear();
	PlotCurvePanelInstance->AddCurve(curve_to_plot,wxColour(0, 0, 255),data_series_titles[data_series_index]);
	PlotCurvePanelInstance->SetXAxisLabel(data_series_x_label[data_series_index]);
	PlotCurvePanelInstance->SetYAxisLabel(data_series_y_label[data_series_index]);
	PlotCurvePanelInstance->Draw(lower_bound_x,upper_bound_x,lower_bound_y,upper_bound_y);
}


void DistributionPlotDialog::OnLowerBoundXTextEnter( wxCommandEvent& event )
{
	OnNewLowerBoundX();
}
void DistributionPlotDialog::OnLowerBoundXKillFocus( wxFocusEvent & event )
{
	OnNewLowerBoundX();
}
void DistributionPlotDialog::OnLowerBoundXSetFocus( wxFocusEvent & event )
{
	value_on_focus_float = LowerBoundXNumericCtrl->ReturnValue();
}

void DistributionPlotDialog::OnUpperBoundXTextEnter( wxCommandEvent& event )
{
	OnNewUpperBoundX();
}
void DistributionPlotDialog::OnUpperBoundXKillFocus( wxFocusEvent & event )
{
	OnNewUpperBoundX();
}
void DistributionPlotDialog::OnUpperBoundXSetFocus( wxFocusEvent & event )
{
	value_on_focus_float = UpperBoundXNumericCtrl->ReturnValue();
}

void DistributionPlotDialog::OnNewUpperBoundX()
{
	UpperBoundXNumericCtrl->CheckValues();
	LowerBoundXNumericCtrl->SetMinMaxValue(-FLT_MAX,UpperBoundXNumericCtrl->ReturnValue());
	upper_bound_x = UpperBoundXNumericCtrl->ReturnValue();
	DoThePlotting();
}

void DistributionPlotDialog::OnNewLowerBoundX()
{
	LowerBoundXNumericCtrl->CheckValues();
	UpperBoundXNumericCtrl->SetMinMaxValue(LowerBoundXNumericCtrl->ReturnValue(),FLT_MAX);
	lower_bound_x = LowerBoundXNumericCtrl->ReturnValue();
	DoThePlotting();
}

void DistributionPlotDialog::OnLowerBoundYTextEnter( wxCommandEvent& event )
{
	OnNewLowerBoundY();
}
void DistributionPlotDialog::OnLowerBoundYKillFocus( wxFocusEvent & event )
{
	OnNewLowerBoundY();
}
void DistributionPlotDialog::OnLowerBoundYSetFocus( wxFocusEvent & event )
{
	value_on_focus_float = LowerBoundYNumericCtrl->ReturnValue();
}

void DistributionPlotDialog::OnUpperBoundYTextEnter( wxCommandEvent& event )
{
	OnNewUpperBoundY();
}
void DistributionPlotDialog::OnUpperBoundYKillFocus( wxFocusEvent & event )
{
	OnNewUpperBoundY();
}
void DistributionPlotDialog::OnUpperBoundYSetFocus( wxFocusEvent & event )
{
	value_on_focus_float = UpperBoundYNumericCtrl->ReturnValue();
}

void DistributionPlotDialog::OnNewUpperBoundY()
{
	UpperBoundYNumericCtrl->CheckValues();
	LowerBoundYNumericCtrl->SetMinMaxValue(-FLT_MAX,UpperBoundYNumericCtrl->ReturnValue());
	upper_bound_y = UpperBoundYNumericCtrl->ReturnValue();
	DoThePlotting();
}

void DistributionPlotDialog::OnNewLowerBoundY()
{
	LowerBoundYNumericCtrl->CheckValues();
	UpperBoundYNumericCtrl->SetMinMaxValue(LowerBoundYNumericCtrl->ReturnValue(),FLT_MAX);
	lower_bound_y = LowerBoundYNumericCtrl->ReturnValue();
	DoThePlotting();
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
