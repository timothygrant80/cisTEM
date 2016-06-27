#include "core_headers.h"

AnglesAndShifts::AnglesAndShifts()
{
	euler_phi = 0.0;
	euler_theta = 0.0;
	euler_psi = 0.0;
	shift_x = 0.0;
	shift_y = 0.0;
	euler_matrix.SetToIdentity();
}

AnglesAndShifts::AnglesAndShifts(float wanted_euler_phi, float wanted_euler_theta, float wanted_euler_psi, float wanted_shift_x, float wanted_shift_y)
{
	Init(wanted_euler_phi, wanted_euler_theta, wanted_euler_psi, wanted_shift_x, wanted_shift_y);
}

void AnglesAndShifts::Init(float wanted_euler_phi_in_degrees, float wanted_euler_theta_in_degrees, float wanted_euler_psi_in_degrees, float wanted_shift_x, float wanted_shift_y)
{
	shift_x = wanted_shift_x;
	shift_y = wanted_shift_y;
	GenerateEulerMatrices(wanted_euler_phi_in_degrees, wanted_euler_theta_in_degrees, wanted_euler_psi_in_degrees);
}

void AnglesAndShifts::GenerateEulerMatrices(float wanted_euler_phi_in_degrees, float wanted_euler_theta_in_degrees, float wanted_euler_psi_in_degrees)
{
	float			cos_phi;
	float			sin_phi;
	float			cos_theta;
	float			sin_theta;
	float			cos_psi;
	float			sin_psi;

	euler_phi = wanted_euler_phi_in_degrees;
	euler_theta = wanted_euler_theta_in_degrees;
	euler_psi = wanted_euler_psi_in_degrees;
	cos_phi = cosf(deg_2_rad(euler_phi));
	sin_phi = sinf(deg_2_rad(euler_phi));
	cos_theta = cosf(deg_2_rad(euler_theta));
	sin_theta = sinf(deg_2_rad(euler_theta));
	cos_psi = cosf(deg_2_rad(euler_psi));
	sin_psi = sinf(deg_2_rad(euler_psi));
	euler_matrix.m[0][0] = cos_phi * cos_theta * cos_psi - sin_phi * sin_psi;
	euler_matrix.m[1][0] = sin_phi * cos_theta * cos_psi + cos_phi * sin_psi;
	euler_matrix.m[2][0] = -sin_theta * cos_psi;
	euler_matrix.m[0][1] = -cos_phi * cos_theta * sin_psi - sin_phi * cos_psi;
	euler_matrix.m[1][1] = -sin_phi * cos_theta * sin_psi + cos_phi * cos_psi;
	euler_matrix.m[2][1] = sin_theta * sin_psi;
	euler_matrix.m[0][2] = sin_theta * cos_phi;
	euler_matrix.m[1][2] = sin_theta * sin_phi;
	euler_matrix.m[2][2] = cos_theta;
}

void AnglesAndShifts::GenerateRotationMatrix2D(float wanted_rotation_angle_in_degrees)
{
	float			cos_phi;
	float			sin_phi;
	float			cos_theta;
	float			sin_theta;
	float			cos_psi;
	float			sin_psi;

	euler_psi = wanted_rotation_angle_in_degrees;
	cos_psi = cosf(deg_2_rad(euler_psi));
	sin_psi = sinf(deg_2_rad(euler_psi));
	euler_matrix.m[0][0] = cos_psi;
	euler_matrix.m[1][0] = sin_psi;
	euler_matrix.m[2][0] = 0.0;
	euler_matrix.m[0][1] = -sin_psi;
	euler_matrix.m[1][1] = cos_psi;
	euler_matrix.m[2][1] = 0.0;
	euler_matrix.m[0][2] = 0.0;
	euler_matrix.m[1][2] = 0.0;
	euler_matrix.m[2][2] = 1.0;
}
