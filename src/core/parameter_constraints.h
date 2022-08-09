/*  \brief  AnglesAndShifts class (particle Euler angles and shifts) */

class ParameterConstraints {

  public:
    ParameterConstraints( );
    //	~ParameterConstraints();							// destructor

    void InitPhi(float wanted_phi_average, float wanted_phi_variance, float wanted_phi_noise_variance);
    void InitTheta(float wanted_theta_average, float wanted_theta_variance, float wanted_theta_noise_variance);
    void InitPsi(float wanted_psi_average, float wanted_psi_variance, float wanted_psi_noise_variance);
    void InitShiftX(float wanted_shift_x_average, float wanted_shift_x_variance, float wanted_shift_x_noise_variance);
    void InitShiftY(float wanted_shift_y_average, float wanted_shift_y_variance, float wanted_shift_y_noise_variance);

    inline float ReturnPhiAngleLogP(float phi) { return -powf(phi - phi_average, 2) / 2.0 / phi_variance; };

    inline float ReturnThetaAngleLogP(float theta) { return -powf(theta - theta_average, 2) / 2.0 / theta_variance; };

    inline float ReturnPsiAngleLogP(float psi) { return -powf(psi - psi_average, 2) / 2.0 / psi_variance; };

    inline float ReturnShiftXLogP(float shift_x) { return -powf(shift_x - shift_x_average, 2) / 2.0 / shift_x_variance; };

    inline float ReturnShiftYLogP(float shift_y) { return -powf(shift_y - shift_y_average, 2) / 2.0 / shift_y_variance; };

    inline float ReturnPhiAnglePenalty(float phi) { return -phi_noise_variance * ReturnPhiAngleLogP(phi); };

    inline float ReturnThetaAnglePenalty(float theta) { return -theta_noise_variance * ReturnThetaAngleLogP(theta); };

    inline float ReturnPsiAnglePenalty(float psi) { return -psi_noise_variance * ReturnPsiAngleLogP(psi); };

    inline float ReturnShiftXPenalty(float shift_x) { return -shift_x_noise_variance * ReturnShiftXLogP(shift_x); };

    inline float ReturnShiftYPenalty(float shift_y) { return -shift_y_noise_variance * ReturnShiftYLogP(shift_y); };

    //	inline void SetAverageShiftX(float wanted_shift_x_average) {shift_x_average = wanted_shift_x_average;};
    //	inline void SetAverageShiftY(float wanted_shift_y_average) {shift_y_average = wanted_shift_y_average;};

    //private:

    float phi_average; // in degrees
    float phi_variance; // in degrees
    float phi_noise_variance; // in density units
    float theta_average; // in degrees
    float theta_variance; // in degrees
    float theta_noise_variance; // in density units
    float psi_average; // in degrees
    float psi_variance; // in degrees
    float psi_noise_variance; // in density units
    float shift_x_average; // in Angstrom
    float shift_x_variance; // in Angstrom
    float shift_x_noise_variance; // in density units
    float shift_y_average; // in Angstrom
    float shift_y_variance; // in Angstrom
    float shift_y_noise_variance; // in density units
};
