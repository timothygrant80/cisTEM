#include "core_headers.h"

// Initialise an empirical distribution
EmpiricalDistribution::EmpiricalDistribution(bool should_keep_sample_values)
{
	keep_sample_values = should_keep_sample_values;
	MyDebugAssertFalse(keep_sample_values,"Keeping of samples values (to compute histograms) not yet implemented\n");
	// Here, would allocate sample_values array
	sample_values = NULL;
	Reset();
}

EmpiricalDistribution::EmpiricalDistribution()
{
	MyDebugAssertFalse(true,"Sorry, you must use the other constructor, which specifies whether to keep samples values. To be improved later.\n")
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
	MyDebugAssertFalse(keep_sample_values,"Keeping of samples values (to compute histograms) not yet implemented\n");
	minimum = std::numeric_limits<float>::max();
	maximum = - std::numeric_limits<float>::max();
}

void EmpiricalDistribution::AddSampleValue(float sample_value)
{
	sum_of_samples += sample_value;
	sum_of_squared_samples += pow(sample_value,2);
	number_of_samples++;
	minimum = - std::max(minimum,sample_value);
	maximum = std::max(maximum,sample_value);
	// We may need to record the value
	if (keep_sample_values)
	{
		MyPrintWithDetails("Keeping of sample values not yet implemented\n");
		abort();
	}
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

void EmpiricalDistribution::PopulateHistogram()
{
	if (keep_sample_values)
	{
		MyPrintWithDetails("Keeping of sample values not yet implemented\n");
		abort();
	}
	else
	{
		MyPrintWithDetails("Need to have kept samples to compute a histogram\n");
		abort();
	}
}
