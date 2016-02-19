#include "core_headers.h"

ParameterConstraints::ParameterConstraints()
{
	phi_average = 0.0;
	phi_variance = 1.0;
	phi_noise_variance = 0.0;
	theta_average = 0.0;
	theta_variance = 1.0;
	theta_noise_variance = 0.0;
	psi_average = 0.0;
	psi_variance = 1.0;
	psi_noise_variance = 0.0;
	shift_x_average = 0.0;
	shift_x_variance = 1.0;
	shift_x_noise_variance = 0.0;
	shift_y_average = 0.0;
	shift_y_variance = 1.0;
	shift_y_noise_variance = 0.0;
}

void ParameterConstraints::InitPhi(float wanted_phi_average, float wanted_phi_variance, float wanted_phi_noise_variance)
{
	if (wanted_phi_variance > 0.0)
	{
		phi_average = wanted_phi_average;
		phi_variance = wanted_phi_variance;
		phi_noise_variance = wanted_phi_noise_variance;
	}
}

void ParameterConstraints::InitTheta(float wanted_theta_average, float wanted_theta_variance, float wanted_theta_noise_variance)
{
	if (wanted_theta_variance > 0.0)
	{
		theta_average = wanted_theta_average;
		theta_variance = wanted_theta_variance;
		theta_noise_variance = wanted_theta_noise_variance;
	}
}

void ParameterConstraints::InitPsi(float wanted_psi_average, float wanted_psi_variance, float wanted_psi_noise_variance)
{
	if (wanted_psi_variance > 0.0)
	{
		psi_average = wanted_psi_average;
		psi_variance = wanted_psi_variance;
		psi_noise_variance = wanted_psi_noise_variance;
	}
}

void ParameterConstraints::InitShiftX(float wanted_shift_x_average, float wanted_shift_x_variance, float wanted_shift_x_noise_variance)
{
	if (wanted_shift_x_variance > 0.0)
	{
		shift_x_average = wanted_shift_x_average;
		shift_x_variance = wanted_shift_x_variance;
		shift_x_noise_variance = wanted_shift_x_noise_variance;
	}
}

void ParameterConstraints::InitShiftY(float wanted_shift_y_average, float wanted_shift_y_variance, float wanted_shift_y_noise_variance)
{
	if (wanted_shift_y_variance > 0.0)
	{
		shift_y_average = wanted_shift_y_average;
		shift_y_variance = wanted_shift_y_variance;
		shift_y_noise_variance = wanted_shift_y_noise_variance;
	}
}
