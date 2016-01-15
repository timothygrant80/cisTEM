#include "core_headers.h"

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

