/*  \brief  AnglesAndShifts class (particle Euler angles and shifts) */

class AnglesAndShifts {

public:

	RotationMatrix	euler_matrix;

	AnglesAndShifts();
	AnglesAndShifts(float wanted_euler_phi, float wanted_euler_theta, float wanted_euler_psi, float wanted_shift_x = 0.0, float wanted_shift_y = 0.0);	// constructor with size
//	~AnglesAndShifts();							// destructor

	void GenerateEulerMatrices(float wanted_euler_phi, float wanted_euler_theta, float wanted_euler_psi);
	void Init(float wanted_euler_phi, float wanted_euler_theta, float wanted_euler_psi, float wanted_shift_x, float wanted_shift_y);

private:

	float 			euler_phi;		// in degrees
	float 			euler_theta;	// in degrees
	float 			euler_psi;		// in degrees
	float			shift_x;		// in Angstrom
	float			shift_y;		// in Angstrom
};
