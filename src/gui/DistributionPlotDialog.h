#ifndef __DistributionPlotDialog__
#define __DistributionPlotDialog__

class DistributionPlotDialog : public DistributionPlotDialogParent
{

public :

	DistributionPlotDialog (wxWindow *parent, wxWindowID id, const wxString &title, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=wxDEFAULT_DIALOG_STYLE);
	void OnCopyButtonClick( wxCommandEvent& event );
	void OnCloseButtonClick(wxCommandEvent &event);
	void OnSaveButtonClick(wxCommandEvent &event);
	void OnDataSeriesToPlotChoice(wxCommandEvent &event);

	void SetNumberOfDataSeries(int wanted_number_of_data_series);
	void SetDataSeries(int which_data_series, double * wanted_data_series, int number_of_points_in_series, wxString wanted_title);

private:
	int number_of_data_series;
	int * number_of_points_in_data_series;
	double ** data_series;
	wxString * data_series_titles;
	void ClearDataSeries();

	std::vector<double> distribution_one_x; // called 'one' because in future, we'll want to do 2D histograms
	std::vector<double> distribution_one_y;
};

/*
 * Histogram class
 * Modified from: https://codereview.stackexchange.com/a/38931
 */
class HistogramComputer {
  public:
    HistogramComputer(double min, double max, int numberOfBins);

    void AddDatum(double datum);
    int ReturnNumberOfBins() const;                    // Get the number of bins
    int ReturnCountInBin(int bin);            // Get the number of data points in some bin
    int ReturnNumberOfLowerOutliers() const;      // Get count of values below the minimum
    int ReturnNumberOfHigherOutliers() const;      // Get count of values at or above the maximum
    double ReturnLowerBoundOfBin(int bin) const;    // Get the inclusive lower bound of a bin
    double ReturnUpperBoundOfBin(int bin) const;    // Get the exclusive upper bound of a bin
    double ReturnCenterOfBin(int bin);

    ~HistogramComputer();

  private:
    double bin_width;
    double lower_bound;
    double upper_bound;
    int number_of_bins;
    int lower_outlier_count;
    int upper_outlier_count;
    int * count_per_bin;
};

void HistogramFromArray(double *data, int number_of_data, int number_of_bins, std::vector<double> &bin_centers, std::vector<double> &histogram);


#endif

