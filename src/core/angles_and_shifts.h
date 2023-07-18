/*  \brief  AnglesAndShifts class (particle Euler angles and shifts) */

class AnglesAndShifts {

  public:
    // FIXME: This should probably be private as changing it will not update the euler angles stored
    RotationMatrix euler_matrix;

    AnglesAndShifts( );
    AnglesAndShifts(float wanted_euler_phi, float wanted_euler_theta, float wanted_euler_psi, float wanted_shift_x = 0.0, float wanted_shift_y = 0.0); // constructor with size
    //	~AnglesAndShifts();							// destructor

    void GenerateEulerMatrices(float wanted_euler_phi, float wanted_euler_theta, float wanted_euler_psi);
    void GenerateRotationMatrix2D(float wanted_rotation_angle_in_degrees);
    void Init(float wanted_euler_phi_in_degrees, float wanted_euler_theta_in_degrees, float wanted_euler_psi_in_degrees, float wanted_shift_x, float wanted_shift_y);

    inline float ReturnPhiAngle( ) { return euler_phi; };

    inline float ReturnThetaAngle( ) { return euler_theta; };

    inline float ReturnPsiAngle( ) { return euler_psi; };

    inline float ReturnShiftX( ) { return shift_x; };

    inline float ReturnShiftY( ) { return shift_y; };

  private:
    float euler_phi; // in degrees
    float euler_theta; // in degrees
    float euler_psi; // in degrees
    float shift_x; // in Angstrom
    float shift_y; // in Angstrom
};
