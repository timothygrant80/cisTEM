#include "core_headers.h"

// Initialise an empirical distribution
template <class Accumulator_t>
EmpiricalDistribution<Accumulator_t>::EmpiricalDistribution( ) {
    static_assert(std::is_same_v<Accumulator_t, double> || std::is_same_v<Accumulator_t, float>, "Accumulator_t must be double or float");
    Reset( );
}

template <class Accumulator_t>
EmpiricalDistribution<Accumulator_t>::~EmpiricalDistribution( ) {
    // Here, would deallocate sample_values array
}

template <class Accumulator_t>
void EmpiricalDistribution<Accumulator_t>::Reset( ) {
    sum_of_samples                 = Accumulator_t{0.0};
    sum_of_squared_samples         = Accumulator_t{0.0};
    number_of_samples_NON_integral = Accumulator_t{0.0};
    mean_welford                   = Accumulator_t{0.0};
    var_times_n_minus_1_welford    = Accumulator_t{0.0};
    kahan_correction_sum           = Accumulator_t{0.0};
    kahan_correction_sum_sqs       = Accumulator_t{0.0};
    number_of_samples              = 0;
    minimum                        = std::numeric_limits<float>::max( );
    maximum                        = -std::numeric_limits<float>::max( );
    is_constant                    = true;
    last_added_value               = 0.0;
    is_welford                     = false;
    is_default                     = false;
}

template <class Accumulator_t>
void EmpiricalDistribution<Accumulator_t>::AddSampleValue(float sample_value) {
    MyDebugAssertFalse(ReturnThreadNumberOfCurrentThread( ) > 0, "This function is not thread safe  b/c of last_added_value.");
    MyDebugAssertFalse(is_welford, "Cannot use AddSampleValue with Welford accumulator.");
    is_default = true;

    sum_of_samples += sample_value;
    sum_of_squared_samples += pow(sample_value, 2);
    number_of_samples++;
    minimum = std::min(minimum, sample_value);
    maximum = std::max(maximum, sample_value);
    if ( number_of_samples == 1 ) {
        is_constant = true;
    }
    else {
        is_constant = is_constant && last_added_value == sample_value;
    }
    last_added_value = sample_value;
}

template <class Accumulator_t>
void EmpiricalDistribution<Accumulator_t>::AddSampleValueWithKahanCorrection(float sample_value) {
    MyDebugAssertFalse(ReturnThreadNumberOfCurrentThread( ) > 0, "This function is not thread safe  b/c of last_added_value.");
    MyDebugAssertFalse(is_welford, "Cannot use AddSampleValue with Welford accumulator.");
    is_default = true;

    Accumulator_t y      = sample_value - kahan_correction_sum;
    Accumulator_t t      = sum_of_samples + y;
    kahan_correction_sum = ((t - sum_of_samples) - y);
    sum_of_samples       = t;

    sample_value *= sample_value;
    y                        = sample_value - kahan_correction_sum_sqs;
    t                        = sum_of_squared_samples + y;
    kahan_correction_sum_sqs = ((t - sum_of_squared_samples) - y);

    sum_of_squared_samples = t;
    number_of_samples++;
    minimum = std::min(minimum, sample_value);
    maximum = std::max(maximum, sample_value);
    if ( number_of_samples == 1 ) {
        is_constant = true;
    }
    else {
        is_constant = is_constant && last_added_value == sample_value;
    }
    last_added_value = sample_value;
}

template <class Accumulator_t>
void EmpiricalDistribution<Accumulator_t>::AddSampleValueForWelford(float sample_value) {
    MyDebugAssertFalse(is_default, "Cannot use AddSampleValueForWelford with default accumulator.");
    is_welford = true;

    // I want to be able to use the same method without having an EmpiricalDistribution class initialized,
    // so I'm going to use the static method AccumulateWelford

    number_of_samples_NON_integral += Accumulator_t{1.0};
    Accumulator_t delta = sample_value - mean_welford;
    mean_welford += delta / number_of_samples_NON_integral;
    Accumulator_t delta2 = sample_value - mean_welford;
    var_times_n_minus_1_welford += delta * delta2;
}

template <class Accumulator_t>
void EmpiricalDistribution<Accumulator_t>::AddSampleValueForWelfordBatched(float sample_mean, float sample_var_times_n_minus_1_welford, Accumulator_t& n_this_batch) {
    MyDebugAssertFalse(is_default, "Cannot use AddSampleValueForWelford with default accumulator.");
    is_welford = true;

    // I want to be able to use the same method without having an EmpiricalDistribution class initialized,
    // so I'm going to use the static method AccumulateWelford
    Accumulator_t delta2 = number_of_samples_NON_integral;
    number_of_samples_NON_integral += n_this_batch;
    Accumulator_t delta = (sample_mean - mean_welford);
    mean_welford += delta * n_this_batch / number_of_samples_NON_integral;
    delta2 *= (delta * delta * n_this_batch / number_of_samples_NON_integral);
    var_times_n_minus_1_welford += (sample_var_times_n_minus_1_welford + delta2);
}

template <class Accumulator_t>
bool EmpiricalDistribution<Accumulator_t>::IsConstant( ) {
    MyDebugAssertFalse(is_welford, "Cannot use IsConstant with Welford accumulator.");
    return is_constant;
}

template <class Accumulator_t>
float EmpiricalDistribution<Accumulator_t>::GetSampleSumOfSquares( ) {
    if ( is_welford )
        return var_times_n_minus_1_welford;

    else
        return sum_of_squared_samples;
}

template <class Accumulator_t>
long EmpiricalDistribution<Accumulator_t>::GetNumberOfSamples( ) {
    if ( is_welford )
        return long(number_of_samples_NON_integral);
    else
        return number_of_samples;
}

template <class Accumulator_t>
float EmpiricalDistribution<Accumulator_t>::GetSampleSum( ) {
    if ( is_welford )
        return mean_welford * number_of_samples_NON_integral;
    else
        return sum_of_samples;
}

template <class Accumulator_t>
float EmpiricalDistribution<Accumulator_t>::GetSampleMean( ) {
    if ( number_of_samples > 0 || number_of_samples_NON_integral > Accumulator_t{0.0} ) {
        if ( is_welford )
            return mean_welford;
        else
            return sum_of_samples / float(number_of_samples);
    }
    else {
        return 0.0;
    }
}

template <class Accumulator_t>
float EmpiricalDistribution<Accumulator_t>::GetSampleVariance( ) {
    MyDebugAssertFalse(is_welford, "Cannot use GetSampleVariance with Welford accumulator.");
    if ( number_of_samples > 0 ) {
        return (sum_of_squared_samples / float(number_of_samples)) - pow(sum_of_samples / float(number_of_samples), 2);
    }
    else {
        return 0.0;
    }
}

template <class Accumulator_t>
float EmpiricalDistribution<Accumulator_t>::GetUnbiasedEstimateOfPopulationVariance( ) {
    if ( number_of_samples > 0 || number_of_samples_NON_integral > Accumulator_t{0.0} ) {
        if ( is_welford )
            return var_times_n_minus_1_welford / float(number_of_samples_NON_integral - 1);
        else
            return GetSampleVariance( ) * float(number_of_samples) / float(number_of_samples - 1);
    }
    else {
        return 0.0;
    }
}

// Explicit instantiation
template class EmpiricalDistribution<double>;
template class EmpiricalDistribution<float>;