#include "core_headers.h"

// these are needed for the simple rotation matrix creation, which uses very optimized code from an old game programming library i wrote when i was about 12,
// think i got it from a book i had about game programming in c.. it's probably slower than what the compiler would do these days.

#define AL_PI        3.14159265358979323846
#define _AL_SINCOS(x, s, c)  __asm__ ("fsincos" : "=t" (c), "=u" (s) : "0" (x))
#define FLOATSINCOS(x, s, c)  _AL_SINCOS((x) * AL_PI / 128.0, s ,c)

#define MAKE_ROTATION_f(x, y, z)                \
    float sin_x, cos_x;				\
    float sin_y, cos_y;				\
    float sin_z, cos_z;				\
    float sinx_siny, cosx_siny;			\
 						\
    FLOATSINCOS(x, sin_x, cos_x);		\
    FLOATSINCOS(y, sin_y, cos_y);		\
    FLOATSINCOS(z, sin_z, cos_z);		\
 						\
    sinx_siny = sin_x * sin_y;			\
    cosx_siny = cos_x * sin_y;

#define R00_f (cos_y * cos_z)
#define R10_f ((sinx_siny * cos_z) - (cos_x * sin_z))
#define R20_f ((cosx_siny * cos_z) + (sin_x * sin_z))

#define R01_f (cos_y * sin_z)
#define R11_f ((sinx_siny * sin_z) + (cos_x * cos_z))
#define R21_f ((cosx_siny * sin_z) - (sin_x * cos_z))

#define R02_f (-sin_y)
#define R12_f (sin_x * cos_y)
#define R22_f (cos_x * cos_y)

RotationMatrix::RotationMatrix()
{
	SetToConstant(0.0);
}

RotationMatrix RotationMatrix::operator + (const RotationMatrix &other)
{
	RotationMatrix temp_matrix;

	temp_matrix.m[0][0] = this->m[0][0] + other.m[0][0];
	temp_matrix.m[0][1] = this->m[0][1] + other.m[0][1];
	temp_matrix.m[0][2] = this->m[0][2] + other.m[0][2];
	temp_matrix.m[1][0] = this->m[1][0] + other.m[1][0];
	temp_matrix.m[1][1] = this->m[1][1] + other.m[1][1];
	temp_matrix.m[1][2] = this->m[1][2] + other.m[1][2];
	temp_matrix.m[2][0] = this->m[2][0] + other.m[2][0];
	temp_matrix.m[2][1] = this->m[2][1] + other.m[2][1];
	temp_matrix.m[2][2] = this->m[2][2] + other.m[2][2];

    return temp_matrix;
}

RotationMatrix RotationMatrix::operator - (const RotationMatrix &other)
{
	RotationMatrix temp_matrix;

	temp_matrix.m[0][0] = this->m[0][0] - other.m[0][0];
	temp_matrix.m[0][1] = this->m[0][1] - other.m[0][1];
	temp_matrix.m[0][2] = this->m[0][2] - other.m[0][2];
	temp_matrix.m[1][0] = this->m[1][0] - other.m[1][0];
	temp_matrix.m[1][1] = this->m[1][1] - other.m[1][1];
	temp_matrix.m[1][2] = this->m[1][2] - other.m[1][2];
	temp_matrix.m[2][0] = this->m[2][0] - other.m[2][0];
	temp_matrix.m[2][1] = this->m[2][1] - other.m[2][1];
	temp_matrix.m[2][2] = this->m[2][2] - other.m[2][2];

    return temp_matrix;
}

RotationMatrix RotationMatrix::operator * (const RotationMatrix &other)
{
	RotationMatrix temp_matrix;

	temp_matrix.m[0][0] = this->m[0][0] * other.m[0][0] + this->m[0][1] * other.m[1][0] + this->m[0][2] * other.m[2][0];
	temp_matrix.m[0][1] = this->m[0][0] * other.m[0][1] + this->m[0][1] * other.m[1][1] + this->m[0][2] * other.m[2][1];
	temp_matrix.m[0][2] = this->m[0][0] * other.m[0][2] + this->m[0][1] * other.m[1][2] + this->m[0][2] * other.m[2][2];
	temp_matrix.m[1][0] = this->m[1][0] * other.m[0][0] + this->m[1][1] * other.m[1][0] + this->m[1][2] * other.m[2][0];
	temp_matrix.m[1][1] = this->m[1][0] * other.m[0][1] + this->m[1][1] * other.m[1][1] + this->m[1][2] * other.m[2][1];
	temp_matrix.m[1][2] = this->m[1][0] * other.m[0][2] + this->m[1][1] * other.m[1][2] + this->m[1][2] * other.m[2][2];
	temp_matrix.m[2][0] = this->m[2][0] * other.m[0][0] + this->m[2][1] * other.m[1][0] + this->m[2][2] * other.m[2][0];
	temp_matrix.m[2][1] = this->m[2][0] * other.m[0][1] + this->m[2][1] * other.m[1][1] + this->m[2][2] * other.m[2][1];
	temp_matrix.m[2][2] = this->m[2][0] * other.m[0][2] + this->m[2][1] * other.m[1][2] + this->m[2][2] * other.m[2][2];

    return temp_matrix;
}

RotationMatrix &RotationMatrix::operator = (const RotationMatrix &other)	// &other contains the address of the other matrix
{
	*this = &other;
	return *this;
}

RotationMatrix &RotationMatrix::operator = (const RotationMatrix *other)	// *other is a pointer to the other matrix
{
   // Check for self assignment
   if (this != other)
   {
	   this->m[0][0] = other->m[0][0];
	   this->m[0][1] = other->m[0][1];
	   this->m[0][2] = other->m[0][2];
	   this->m[1][0] = other->m[1][0];
	   this->m[1][1] = other->m[1][1];
	   this->m[1][2] = other->m[1][2];
	   this->m[2][0] = other->m[2][0];
	   this->m[2][1] = other->m[2][1];
	   this->m[2][2] = other->m[2][2];
   }

   return *this;
}

RotationMatrix &RotationMatrix::operator += (const RotationMatrix &other)
{
	*this += &other;
	return *this;
}

RotationMatrix &RotationMatrix::operator += (const RotationMatrix *other)
{
	this->m[0][0] += other->m[0][0];
	this->m[0][1] += other->m[0][1];
	this->m[0][2] += other->m[0][2];
	this->m[1][0] += other->m[1][0];
	this->m[1][1] += other->m[1][1];
	this->m[1][2] += other->m[1][2];
	this->m[2][0] += other->m[2][0];
	this->m[2][1] += other->m[2][1];
	this->m[2][2] += other->m[2][2];
	return *this;
}

RotationMatrix &RotationMatrix::operator -= (const RotationMatrix &other)
{
	*this -= &other;
	return *this;
}

RotationMatrix &RotationMatrix::operator -= (const RotationMatrix *other)
{
	this->m[0][0] -= other->m[0][0];
	this->m[0][1] -= other->m[0][1];
	this->m[0][2] -= other->m[0][2];
	this->m[1][0] -= other->m[1][0];
	this->m[1][1] -= other->m[1][1];
	this->m[1][2] -= other->m[1][2];
	this->m[2][0] -= other->m[2][0];
	this->m[2][1] -= other->m[2][1];
	this->m[2][2] -= other->m[2][2];
	return *this;
}

RotationMatrix &RotationMatrix::operator *= (const RotationMatrix &other)
{
	*this *= &other;
	return *this;
}

RotationMatrix &RotationMatrix::operator *= (const RotationMatrix *other)
{
	RotationMatrix temp_matrix;

	temp_matrix = *this * *other;
	*this = temp_matrix;

    return *this;
}

RotationMatrix RotationMatrix::ReturnTransposed()
{
	RotationMatrix temp_matrix;

	temp_matrix.m[0][0] = this->m[0][0];
	temp_matrix.m[0][1] = this->m[1][0];
	temp_matrix.m[0][2] = this->m[2][0];
	temp_matrix.m[1][0] = this->m[0][1];
	temp_matrix.m[1][1] = this->m[1][1];
	temp_matrix.m[1][2] = this->m[2][1];
	temp_matrix.m[2][0] = this->m[0][2];
	temp_matrix.m[2][1] = this->m[1][2];
	temp_matrix.m[2][2] = this->m[2][2];

    return temp_matrix;
}

void RotationMatrix::SetToIdentity()
{
	this->m[0][0] = 1.0;
	this->m[1][0] = 0.0;
	this->m[2][0] = 0.0;
	this->m[0][1] = 0.0;
	this->m[1][1] = 1.0;
	this->m[2][1] = 0.0;
	this->m[0][2] = 0.0;
	this->m[1][2] = 0.0;
	this->m[2][2] = 1.0;
}

void RotationMatrix::SetToRotation(float input_x, float input_y, float input_z)
{
	float x = (256.0 / 360.0) * input_x;
	float y = (256.0 / 360.0) * input_y;
	float z = (256.0 / 360.0) * input_z;

    MAKE_ROTATION_f(x, y, z);
/*
    m->v[0][0] = R00_f;
    m->v[0][1] = R01_f;
    m->v[0][2] = R02_f;

    m->v[1][0] = R10_f;
    m->v[1][1] = R11_f;
    m->v[1][2] = R12_f;

    m->v[2][0] = R20_f;
    m->v[2][1] = R21_f;
    m->v[2][2] = R22_f;

    m->t[0] = m->t[1] = m->t[2] = 0;*/

    this->m[0][0] = R00_f;
	this->m[1][0] = R01_f;
	this->m[2][0] = R02_f;
	this->m[0][1] = R10_f;
	this->m[1][1] = R11_f;
	this->m[2][1] = R12_f;
	this->m[0][2] = R20_f;
	this->m[1][2] = R21_f;
	this->m[2][2] = R22_f;
}

void RotationMatrix::ConvertToValidEulerAngles(float &output_phi_in_degrees, float &output_theta_in_degrees, float &output_psi_in_degrees)
{

	// taken from Richard Henderson's ROT2EUL function.

	float cos_theta;
	float sin_theta;
	float cos_phi;
	float sin_phi;
	float cos_psi;
	float sin_psi;

	// first get cos(theta),theta and sin(theta)

	cos_theta = std::max(-1.0f, std::min(1.0f, m[2][2]));
	sin_theta = sqrtf(std::max(0.0f, 1-powf(m[2][2], 2)));

	output_theta_in_degrees = rad_2_deg(acosf(cos_theta));

	//for theta not equal to 0 or 180, PHI, PSI are unique

	if (fabsf(fabsf(cos_theta) - 1.0f) > 0.00001)
	{
		cos_phi = m[0][2] / sin_theta;
		sin_phi = m[1][2] / sin_theta;
		cos_phi = std::max(-1.0f, std::min(1.0f, cos_phi));
		output_phi_in_degrees = acosf(cos_phi);
		if (sin_phi < 0.0f) output_phi_in_degrees = -output_phi_in_degrees;
		output_phi_in_degrees = rad_2_deg(output_phi_in_degrees);

		cos_psi = -m[2][0] / sin_theta;
		sin_psi = m[2][1] / sin_theta;
		cos_psi = std::max(-1.0f, std::min(1.0f, cos_psi));
		output_psi_in_degrees = acosf(cos_psi);
		if (sin_psi < 0.0f) output_psi_in_degrees = -output_psi_in_degrees;
		output_psi_in_degrees = rad_2_deg(output_psi_in_degrees);

	}
	else
	{
		// for THETA=0/180, PHI and PSI can have an infinite number of values, only
		// [PSI-PHI] is defined, so PHI can be set to zero without restriction

		output_phi_in_degrees = 0.0f;

		cos_psi = m[0][0];
		sin_psi = m[1][0];
		cos_psi = std::max(-1.0f, std::min(1.0f, cos_psi));
		output_psi_in_degrees = acosf(cos_psi);
		if (sin_psi <= 0.0f) output_psi_in_degrees = -output_psi_in_degrees;
		output_psi_in_degrees = rad_2_deg(output_psi_in_degrees);

	}


	// check we are close to original rotation matrix, if not refine..

	RotationMatrix test_matrix;
	test_matrix.SetToEulerRotation(output_phi_in_degrees, output_theta_in_degrees, output_psi_in_degrees);

	bool should_throw_error = false;

	if (fabsf(test_matrix.m[0][0] - this->m[0][0]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[1][0] - this->m[1][0]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[2][0] - this->m[2][0]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[0][1] - this->m[0][1]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[1][1] - this->m[1][1]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[2][1] - this->m[2][1]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[0][2] - this->m[0][2]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[1][2] - this->m[1][2]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[2][2] - this->m[2][2]) > 0.001) should_throw_error = true;

	if (should_throw_error == true) // the matrix is not right, this is probably an edge case, the theta will be close to right, but the phi can have significant error -  lets brute force around the found solution to refine it! (note this "improves" on a nobel prize winners work :) )
	{

		//wxPrintf("\nFound Angles before refinement= (%.2f, %.2f, %.2f)\n\n", output_phi_in_degrees, output_theta_in_degrees, output_psi_in_degrees);
		/*wxPrintf("Found Matrix before refinement :- \n\n%+.4f\t%+.4f\t%+.4f\n", test_matrix.m[0][0], test_matrix.m[1][0], test_matrix.m[2][0]);
		wxPrintf("%+.4f\t%+.4f\t%+.4f\n", test_matrix.m[0][1], test_matrix.m[1][1], test_matrix.m[2][1]);
		wxPrintf("%+.4f\t%+.4f\t%+.4f\n", test_matrix.m[0][2], test_matrix.m[1][2], test_matrix.m[2][2]);*/

		float old_best_phi = output_phi_in_degrees;
		float old_best_theta = output_theta_in_degrees;
		float old_best_psi = output_psi_in_degrees;
		float current_difference;
		float best_difference = FLT_MAX;
		float current_phi;
		float current_theta;
		float current_psi;
		float best_phi;
		float best_theta;
		float best_psi;

		for (current_phi = old_best_phi - 1.0f; current_phi < old_best_phi + 1.0f; current_phi += 1.0f)
		{
			for (current_theta = old_best_theta - 1.0f; current_theta < old_best_theta + 1.0f; current_theta += 1.0f)
			{
				for (current_psi = old_best_psi - 90.0f; current_psi < old_best_psi + 90.0f; current_psi += 5.0f)
				{

					test_matrix.SetToEulerRotation(current_phi, current_theta, current_psi);
					current_difference = 0.0f;

					current_difference += fabsf(test_matrix.m[0][0] - this->m[0][0]);
					current_difference += fabsf(test_matrix.m[1][0] - this->m[1][0]);
					current_difference += fabsf(test_matrix.m[2][0] - this->m[2][0]);
					current_difference += fabsf(test_matrix.m[0][1] - this->m[0][1]);
					current_difference += fabsf(test_matrix.m[1][1] - this->m[1][1]);
					current_difference += fabsf(test_matrix.m[2][1] - this->m[2][1]);
					current_difference += fabsf(test_matrix.m[0][2] - this->m[0][2]);
					current_difference += fabsf(test_matrix.m[1][2] - this->m[1][2]);
					current_difference += fabsf(test_matrix.m[2][2] - this->m[2][2]);

					if (current_difference < best_difference)
					{
						best_difference = current_difference;
						best_phi = current_phi;
						best_theta = current_theta;
						best_psi = current_psi;
					}

				}
			}

		}


		old_best_phi = best_phi;
		old_best_theta = best_theta;
		old_best_psi = best_psi;

		for (current_phi = best_phi - 1.0f; current_phi < old_best_phi + 1.0f; current_phi += 0.1f)
		{
			for (current_theta = best_theta - 1.0f; current_theta < old_best_theta + 1.0f; current_theta += 0.1f)
			{
				for (current_psi = best_psi - 5.0f; current_psi < old_best_psi + 5.0f; current_psi += 0.1f)
				{

					test_matrix.SetToEulerRotation(current_phi, current_theta, current_psi);
					current_difference = 0.0f;

					current_difference += fabsf(test_matrix.m[0][0] - this->m[0][0]);
					current_difference += fabsf(test_matrix.m[1][0] - this->m[1][0]);
					current_difference += fabsf(test_matrix.m[2][0] - this->m[2][0]);
					current_difference += fabsf(test_matrix.m[0][1] - this->m[0][1]);
					current_difference += fabsf(test_matrix.m[1][1] - this->m[1][1]);
					current_difference += fabsf(test_matrix.m[2][1] - this->m[2][1]);
					current_difference += fabsf(test_matrix.m[0][2] - this->m[0][2]);
					current_difference += fabsf(test_matrix.m[1][2] - this->m[1][2]);
					current_difference += fabsf(test_matrix.m[2][2] - this->m[2][2]);

					if (current_difference < best_difference)
					{
						best_difference = current_difference;
						best_phi = current_phi;
						best_theta = current_theta;
						best_psi = current_psi;
					}

				}
			}
		}


		old_best_phi = best_phi;
		old_best_theta = best_theta;
		old_best_psi = best_psi;

		for (current_phi = best_phi - 0.1f; current_phi < old_best_phi + 0.1f; current_phi += 0.01f)
		{
			for (current_theta = best_theta - 0.1f; current_theta < old_best_theta + 0.1f; current_theta += 0.01f)
			{
				for (current_psi = best_psi - 0.1f; current_psi < old_best_psi + 0.1f; current_psi += 0.01f)
				{

					test_matrix.SetToEulerRotation(current_phi, current_theta, current_psi);
					current_difference = 0.0f;

					current_difference += fabsf(test_matrix.m[0][0] - this->m[0][0]);
					current_difference += fabsf(test_matrix.m[1][0] - this->m[1][0]);
					current_difference += fabsf(test_matrix.m[2][0] - this->m[2][0]);
					current_difference += fabsf(test_matrix.m[0][1] - this->m[0][1]);
					current_difference += fabsf(test_matrix.m[1][1] - this->m[1][1]);
					current_difference += fabsf(test_matrix.m[2][1] - this->m[2][1]);
					current_difference += fabsf(test_matrix.m[0][2] - this->m[0][2]);
					current_difference += fabsf(test_matrix.m[1][2] - this->m[1][2]);
					current_difference += fabsf(test_matrix.m[2][2] - this->m[2][2]);

					if (current_difference < best_difference)
					{
						best_difference = current_difference;
						best_phi = current_phi;
						best_theta = current_theta;
						best_psi = current_psi;
					}

				}
			}
		}


		output_phi_in_degrees = best_phi;
		output_theta_in_degrees = best_theta;
		output_psi_in_degrees = best_psi;

		//wxPrintf("\nFound Angles = (%.2f, %.2f, %.2f)\n\n", output_phi_in_degrees, output_theta_in_degrees, output_psi_in_degrees);
	}

	// if in debug mode do a final check
/*#ifdef DEBUG
	test_matrix.SetToEulerRotation(output_phi_in_degrees, output_theta_in_degrees, output_psi_in_degrees);

	should_throw_error = false;

	if (fabsf(test_matrix.m[0][0] - this->m[0][0]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[1][0] - this->m[1][0]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[2][0] - this->m[2][0]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[0][1] - this->m[0][1]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[1][1] - this->m[1][1]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[2][1] - this->m[2][1]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[0][2] - this->m[0][2]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[1][2] - this->m[1][2]) > 0.001) should_throw_error = true;
	if (fabsf(test_matrix.m[2][2] - this->m[2][2]) > 0.001) should_throw_error = true;

	if (should_throw_error == true)
	{
		MyPrintWithDetails("Rotation Matrix to Euler angles conversion failed\n");

		wxPrintf("\nFound Angles = (%.2f, %.2f, %.2f)\n\n", output_phi_in_degrees, output_theta_in_degrees, output_psi_in_degrees);

		wxPrintf("Matrix :- \n\n%+.4f\t%+.4f\t%+.4f\n", m[0][0], m[1][0], m[2][0]);
		wxPrintf("%+.4f\t%+.4f\t%+.4f\n", m[0][1], m[1][1], m[2][1]);
		wxPrintf("%+.4f\t%+.4f\t%+.4f\n\n", m[0][2], m[1][2], m[2][2]);
		wxPrintf("cos_theta values = %.2f\n", fabsf(fabsf(cos_theta) - 1.0f));

		wxPrintf("Found Matrix :- \n\n%+.4f\t%+.4f\t%+.4f\n", test_matrix.m[0][0], test_matrix.m[1][0], test_matrix.m[2][0]);
		wxPrintf("%+.4f\t%+.4f\t%+.4f\n", test_matrix.m[0][1], test_matrix.m[1][1], test_matrix.m[2][1]);
		wxPrintf("%+.4f\t%+.4f\t%+.4f\n", test_matrix.m[0][2], test_matrix.m[1][2], test_matrix.m[2][2]);

		wxPrintf("Difference :- \n\n%+.4f\t%+.4f\t%+.4f\n", fabsf(m[0][0] - test_matrix.m[0][0]), fabsf(m[1][0] - test_matrix.m[1][0]), fabsf(m[2][0] - test_matrix.m[2][0]));
		wxPrintf("%+.4f\t%+.4f\t%+.4f\n", fabsf(m[0][1] - test_matrix.m[0][1]), fabsf(m[1][1] - test_matrix.m[1][1]), fabsf(m[2][1] - test_matrix.m[2][1]));
		wxPrintf("%+.4f\t%+.4f\t%+.4f\n\n", fabsf(m[0][2] - test_matrix.m[0][2]), fabsf(m[1][2] - test_matrix.m[1][2]), fabsf(m[2][2] - test_matrix.m[2][2]));

		DEBUG_ABORT;
	}


#endif*/


}

void RotationMatrix::SetToEulerRotation(float wanted_euler_phi_in_degrees, float wanted_euler_theta_in_degrees, float wanted_euler_psi_in_degrees)
{

	float			cos_phi;
	float			sin_phi;
	float			cos_theta;
	float			sin_theta;
	float			cos_psi;
	float			sin_psi;

	cos_phi = cosf(deg_2_rad(wanted_euler_phi_in_degrees));
	sin_phi = sinf(deg_2_rad(wanted_euler_phi_in_degrees));
	cos_theta = cosf(deg_2_rad(wanted_euler_theta_in_degrees));
	sin_theta = sinf(deg_2_rad(wanted_euler_theta_in_degrees));
	cos_psi = cosf(deg_2_rad(wanted_euler_psi_in_degrees));
	sin_psi = sinf(deg_2_rad(wanted_euler_psi_in_degrees));
	m[0][0] = cos_phi * cos_theta * cos_psi - sin_phi * sin_psi;
	m[1][0] = sin_phi * cos_theta * cos_psi + cos_phi * sin_psi;
	m[2][0] = -sin_theta * cos_psi;
	m[0][1] = -cos_phi * cos_theta * sin_psi - sin_phi * cos_psi;
	m[1][1] = -sin_phi * cos_theta * sin_psi + cos_phi * cos_psi;
	m[2][1] = sin_theta * sin_psi;
	m[0][2] = sin_theta * cos_phi;
	m[1][2] = sin_theta * sin_phi;
	m[2][2] = cos_theta;
}void ConvertToValidEulerAngles(float &output_phi, float &output_theta, float &output_psi);

void RotationMatrix::SetToConstant(float constant)
{
	this->m[0][0] = constant;
	this->m[1][0] = constant;
	this->m[2][0] = constant;
	this->m[0][1] = constant;
	this->m[1][1] = constant;
	this->m[2][1] = constant;
	this->m[0][2] = constant;
	this->m[1][2] = constant;
	this->m[2][2] = constant;
}

void RotationMatrix::SetToValues(float m00, float m10, float m20, float m01, float m11, float m21, float m02, float m12, float m22)
{
	this->m[0][0] = m00;
	this->m[1][0] = m10;
	this->m[2][0] = m20;
	this->m[0][1] = m01;
	this->m[1][1] = m11;
	this->m[2][1] = m21;
	this->m[0][2] = m02;
	this->m[1][2] = m12;
	this->m[2][2] = m22;
}

