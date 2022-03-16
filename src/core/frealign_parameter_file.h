/*  \brief  FrealignParameterFile class */

class FrealignParameterFile {

  public:
    FILE*    parameter_file;
    wxString filename;
    int      access_type;
    int      records_per_line;
    int      number_of_lines;
    int      current_line;
    float*   parameter_cache;
    float    average_defocus;
    float    defocus_coeff_a;
    float    defocus_coeff_b;

    FrealignParameterFile( );
    ~FrealignParameterFile( );

    FrealignParameterFile(wxString wanted_filename, int wanted_access_type, int wanted_records_per_line = 17);
    void   Open(wxString wanted_filename, int wanted_access_type, int wanted_records_per_line = 17);
    void   Close( );
    void   WriteCommentLine(wxString comment_string);
    void   WriteLine(float* parameters, bool comment = false);
    int    ReadFile(bool exclude_negative_film_numbers = false, int particles_in_stack = -1);
    void   ReadLine(float* parameters, int wanted_line_number = -1);
    float  ReadParameter(int wanted_line_number, int wanted_index);
    void   UpdateParameter(int wanted_line_number, int wanted_index, float wanted_value);
    void   Rewind( );
    float  ReturnMin(int wanted_index, bool exclude_negative_film_numbers = false);
    float  ReturnMax(int wanted_index, bool exclude_negative_film_numbers = false);
    double ReturnAverage(int wanted_index, bool exclude_negative_film_numbers = false);
    void   RemoveOutliers(int wanted_index, float wanted_standard_deviation, bool exclude_negative_film_numbers = false, bool reciprocal_square = false);
    float  ReturnThreshold(float wanted_percentage, bool exclude_negative_film_numbers = false);
    void   CalculateDefocusDependence(bool exclude_negative_film_numbers = false);
    void   AdjustScores(bool exclude_negative_film_numbers = false);
    float  ReturnScoreAdjustment(float defocus);
    void   ReduceAngles( );
    float  ReturnDistributionMax(int wanted_index, int selector = 0, bool exclude_negative_film_numbers = false);
    float  ReturnDistributionSigma(int wanted_index, float distribution_max, int selector = 0, bool exclude_negative_film_numbers = false);
    void   SetParameters(int wanted_index, float wanted_value, float wanted_sigma = 0.0, int selector = 0, bool exclude_negative_film_numbers = false);
};
