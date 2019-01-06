#ifndef __DistributionPlotDialog__
#define __DistributionPlotDialog__

class DistributionPlotDialog : public DistributionPlotDialogParent
{

public :

	DistributionPlotDialog (wxWindow *parent, wxWindowID id, const wxString &title, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=wxDEFAULT_DIALOG_STYLE);
	~DistributionPlotDialog();
	void OnCopyButtonClick( wxCommandEvent& event );
	void OnCloseButtonClick(wxCommandEvent &event);
	void OnSavePNGButtonClick(wxCommandEvent &event);
	void OnSaveTXTButtonClick(wxCommandEvent &event);
	void OnDataSeriesToPlotChoice(wxCommandEvent &event);

	void SetNumberOfDataSeries(int wanted_number_of_data_series);
	void SetDataSeries(int which_data_series, double * wanted_data_series, int number_of_points_in_series, bool should_plot_histogram, wxString wanted_title, wxString wanted_x_label, wxString wanted_y_label);
	void SelectDataSeries(int which_data_series);


	void OnLowerBoundXTextEnter( wxCommandEvent& event );
	void OnLowerBoundXKillFocus( wxFocusEvent & event );
	void OnLowerBoundXSetFocus( wxFocusEvent & event );

	void OnUpperBoundXTextEnter( wxCommandEvent& event );
	void OnUpperBoundXKillFocus( wxFocusEvent & event );
	void OnUpperBoundXSetFocus( wxFocusEvent & event );

	void OnLowerBoundYTextEnter( wxCommandEvent& event );
	void OnLowerBoundYKillFocus( wxFocusEvent & event );
	void OnLowerBoundYSetFocus( wxFocusEvent & event );

	void OnUpperBoundYTextEnter( wxCommandEvent& event );
	void OnUpperBoundYKillFocus( wxFocusEvent & event );
	void OnUpperBoundYSetFocus( wxFocusEvent & event );

private:
	int number_of_data_series;
	int * number_of_points_in_data_series;
	double ** data_series;
	wxString * data_series_titles;
	wxString * data_series_x_label;
	wxString * data_series_y_label;
	bool * plot_histogram_of_this_series;
	void ClearDataSeries();
	void DoThePlotting();
	void ResetBounds();
	void UpdateCurveToPlot(bool reset_bounds);
	void OnNewUpperBoundX();
	void OnNewLowerBoundX();
	void OnNewUpperBoundY();
	void OnNewLowerBoundY();
	float value_on_focus_float;

	Curve curve_to_plot;
	double lower_bound_x;
	double upper_bound_x;
	double lower_bound_y;
	double upper_bound_y;
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

void HistogramFromArray(double *data, int number_of_data, int number_of_bins, double lower_bound, double upper_bound, std::vector<double> &bin_centers, std::vector<double> &histogram, double wanted_lower_bound, double wanted_upper_bound);
void HistogramComputeAutoBounds(double *data, int &number_of_data, double &min, double &max);
Curve HistogramFromArray(double *data, int number_of_data, int number_of_bins, double lower_bound = 0.0, double upper_bound = 0.0);


#endif

