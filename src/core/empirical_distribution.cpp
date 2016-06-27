#include "core_headers.h"

// Initialise an empirical distribution
EmpiricalDistribution::EmpiricalDistribution()
{
	Reset();
}


EmpiricalDistribution::~EmpiricalDistribution()
{
	// Here, would deallocate sample_values array
}

void EmpiricalDistribution::Reset()
{
	sum_of_samples = 0.0;
	sum_of_squared_samples = 0.0;
	number_of_samples = 0;
	minimum = std::numeric_limits<float>::max();
	maximum = - std::numeric_limits<float>::max();
	is_constant = true;
	last_added_value = 0.0;
}

void EmpiricalDistribution::AddSampleValue(float sample_value)
{
	sum_of_samples += sample_value;
	sum_of_squared_samples += pow(sample_value,2);
	number_of_samples++;
	minimum = std::min(minimum,sample_value);
	maximum = std::max(maximum,sample_value);
	if (number_of_samples == 1)
	{
		is_constant = true;
	}
	else
	{
		is_constant = is_constant && last_added_value == sample_value;
	}
	last_added_value = sample_value;
}

bool EmpiricalDistribution::IsConstant()
{
	return is_constant;
}

float EmpiricalDistribution::GetSampleSumOfSquares()
{
	return sum_of_squared_samples;
}

float EmpiricalDistribution::GetNumberOfSamples()
{
	return number_of_samples;
}

float EmpiricalDistribution::GetSampleSum()
{
	return sum_of_samples;
}

float EmpiricalDistribution::GetSampleMean()
{
	if (number_of_samples > 0)
	{
		return sum_of_samples / float(number_of_samples);
	}
	else
	{
		return 0.0;
	}
}

float EmpiricalDistribution::GetSampleVariance()
{
	if (number_of_samples > 0)
	{
		return (sum_of_squared_samples / float(number_of_samples)) - pow(sum_of_samples/float(number_of_samples),2);
	}
	else
	{
		return 0.0;
	}
}

float EmpiricalDistribution::GetUnbiasedEstimateOfPopulationVariance()
{
	if (number_of_samples > 0)
	{
		return GetSampleVariance() * float(number_of_samples) / float(number_of_samples-1);
	}
	else
	{
		return 0.0;
	}
}
