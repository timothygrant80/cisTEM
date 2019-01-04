#include "../core/gui_core_headers.h"


DistributionPlotDialog::DistributionPlotDialog (wxWindow *parent, wxWindowID id, const wxString &title, const wxPoint &pos, const wxSize &size, long style)
:
DistributionPlotDialogParent( parent, id, title, pos, size, style)
{
	number_of_data_series = 0;
	number_of_points_in_data_series = 0;
	data_series = NULL;
	data_series_titles = NULL;

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
		wxTheClipboard->SetData( new wxBitmapDataObject( DistributionPlotPanelInstance->buffer_bitmap) );
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

	DistributionPlotPanelInstance->buffer_bitmap.SaveFile(saveFileDialog->ReturnProperPath(), wxBITMAP_TYPE_PNG);
	saveFileDialog->Destroy();
}

void DistributionPlotDialog::OnCloseButtonClick(wxCommandEvent &event)
{
	EndModal(0);
}

void DistributionPlotDialog::OnDataSeriesToPlotChoice(wxCommandEvent &event)
{
	int data_series_index = DataSeriesToPlotChoice->GetCurrentSelection();
	const int number_of_bins = 100;

	MyDebugPrint("Will plot "+data_series_titles[data_series_index]);

	/*
	 * Compute a histogram
	 */
	HistogramFromArray(data_series[data_series_index],number_of_points_in_data_series[data_series_index],number_of_bins,distribution_one_x,distribution_one_y);

	MyDebugPrint("Histogram ready to be plotted. %d",distribution_one_x[10]);

	/*
	 * Redo the plotting
	 */
}

void HistogramFromArray(double *data, int number_of_data, int number_of_bins, std::vector<double> &bin_centers, std::vector<double> &histogram)
{

	/*
	 * Find min,max
	 */
	double min = DBL_MAX;
	double max = -DBL_MAX;
	for (int i=0;i<number_of_data;i++)
	{
		if (data[i] > max) max = data[i];
		if (data[i] < min) min = data[i];
	}

	HistogramComputer hist = HistogramComputer(min,max,number_of_bins);

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
