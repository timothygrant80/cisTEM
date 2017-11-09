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

