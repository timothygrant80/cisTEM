class StarFileParameters {

  public:
    int      position_in_stack;
    float    phi;
    float    theta;
    float    psi;
    float    x_coordinate;
    float    y_coordinate;
    float    x_shift;
    float    y_shift;
    float    defocus1;
    float    defocus2;
    float    defocus_angle;
    float    phase_shift;
    std::string micrograph_name;
    std::string image_name;
    int      random_subset;

    StarFileParameters( );
};

typedef std::vector<StarFileParameters> ArrayofStarFileParameters;

class BasicStarFileReader {

    int current_position_in_stack;
    int current_column;

    int phi_column;
    int theta_column;
    int psi_column;
    int xcoordinate_column;
    int ycoordinate_column;
    int xshift_column;
    int yshift_column;
    int defocus1_column;
    int defocus2_column;
    int defocus_angle_column;
    int phase_shift_column;
    int micrograph_name_column;
    int image_name_column;
    int random_subset_column;

  public:
    std::string filename;

    // The whole file, read into memory a line at a time, with the current position that
    // GetNextLine()/Eof() used to keep for us.
    std::vector<std::string> input_file_lines;
    bool                     input_file_is_opened;
    long                     current_line_number;

    bool AtEndOfFile( ) const { return current_line_number == long(input_file_lines.size( )); }

    std::string ReturnNextLine( ) {
        current_line_number++;
        if ( current_line_number == long(input_file_lines.size( )) )
            return std::string( );
        return input_file_lines[current_line_number];
    }

    ArrayofStarFileParameters cached_parameters;

    bool x_shifts_are_in_angst;
    bool y_shifts_are_in_angst;

    BasicStarFileReader( );
    ~BasicStarFileReader( );

    BasicStarFileReader(std::string wanted_filename);
    void Open(std::string wanted_filename);
    void Close( );
    bool ReadFile(std::string wanted_filename, std::string* error_string = NULL);

    bool ExtractParametersFromLine(std::string& wanted_line, std::string* error_string = NULL);

    inline int ReturnPositionInStack(int line_number) { return cached_parameters[line_number].position_in_stack; }

    inline float ReturnPhi(int line_number) { return cached_parameters[line_number].phi; }

    inline float ReturnTheta(int line_number) { return cached_parameters[line_number].theta; }

    inline float ReturnPsi(int line_number) { return cached_parameters[line_number].psi; }

    inline float ReturnXCoordinate(int line_number) { return cached_parameters[line_number].x_coordinate; }

    inline float ReturnYCoordinate(int line_number) { return cached_parameters[line_number].y_coordinate; }

    inline float ReturnXShift(int line_number) { return cached_parameters[line_number].x_shift; }

    inline float ReturnYShift(int line_number) { return cached_parameters[line_number].y_shift; }

    inline float ReturnDefocus1(int line_number) { return cached_parameters[line_number].defocus1; }

    inline float ReturnDefocus2(int line_number) { return cached_parameters[line_number].defocus2; }

    inline float ReturnDefocusAngle(int line_number) { return cached_parameters[line_number].defocus_angle; }

    inline float ReturnPhaseShift(int line_number) { return cached_parameters[line_number].phase_shift; }

    inline float ReturnAssignedSubset(int line_number) { return cached_parameters[line_number].random_subset; }

    inline std::string ReturnMicrographName(int line_number) { return cached_parameters[line_number].micrograph_name; }

    inline std::string ReturnImageName(int line_number) { return cached_parameters[line_number].image_name; }

    inline int returnNumLines( ) { return cached_parameters.size(); }
};
