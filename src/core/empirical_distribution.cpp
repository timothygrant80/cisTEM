#include "core_headers.h"

// Initialise an empirical distribution
template <class Accumulator_t>
EmpiricalDistribution<Accumulator_t>::EmpiricalDistribution( )
    : constructed_in_parallel_region(ReturnInParallelRegionBool( )) {
    static_assert(std::is_same_v<Accumulator_t, double> || std::is_same_v<Accumulator_t, float>, "Accumulator_t must be double or float");
    Reset( );
}

template <class Accumulator_t>
EmpiricalDistribution<Accumulator_t>::EmpiricalDistribution(const EmpiricalDistribution& other)
    : constructed_in_parallel_region(ReturnInParallelRegionBool( )) {
    static_assert(std::is_same_v<Accumulator_t, double> || std::is_same_v<Accumulator_t, float>, "Accumulator_t must be double or float");
    // Copy all member variables from other
    sum_of_samples                 = other.sum_of_samples;
    sum_of_squared_samples         = other.sum_of_squared_samples;
    number_of_samples_NON_integral = other.number_of_samples_NON_integral;
    kahan_correction_sum           = other.kahan_correction_sum;
    kahan_correction_sum_sqs       = other.kahan_correction_sum_sqs;
    number_of_samples              = other.number_of_samples;
    minimum                        = other.minimum;
    maximum                        = other.maximum;
    is_constant                    = other.is_constant;
    last_added_value               = other.last_added_value;
    mean_welford                   = other.mean_welford;
    var_times_n_minus_1_welford    = other.var_times_n_minus_1_welford;
    is_welford                     = other.is_welford;
    is_default                     = other.is_default;
}

template <class Accumulator_t>
EmpiricalDistribution<Accumulator_t>::EmpiricalDistribution(EmpiricalDistribution&& other) noexcept
    : constructed_in_parallel_region(ReturnInParallelRegionBool( )) {
    static_assert(std::is_same_v<Accumulator_t, double> || std::is_same_v<Accumulator_t, float>, "Accumulator_t must be double or float");
    // Move (bitwise copy suffices for these trivially movable members)
    sum_of_samples                 = other.sum_of_samples;
    sum_of_squared_samples         = other.sum_of_squared_samples;
    number_of_samples_NON_integral = other.number_of_samples_NON_integral;
    kahan_correction_sum           = other.kahan_correction_sum;
    kahan_correction_sum_sqs       = other.kahan_correction_sum_sqs;
    number_of_samples              = other.number_of_samples;
    minimum                        = other.minimum;
    maximum                        = other.maximum;
    is_constant                    = other.is_constant;
    last_added_value               = other.last_added_value;
    mean_welford                   = other.mean_welford;
    var_times_n_minus_1_welford    = other.var_times_n_minus_1_welford;
    is_welford                     = other.is_welford;
    is_default                     = other.is_default;
}

template <class Accumulator_t>
EmpiricalDistribution<Accumulator_t>& EmpiricalDistribution<Accumulator_t>::operator=(const EmpiricalDistribution& other) {
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
    static_assert(std::is_same_v<Accumulator_t, double> || std::is_same_v<Accumulator_t, float>, "Accumulator_t must be double or float");
    if ( this != &other ) {
        // constructed_in_parallel_region is const and intentionally not reassigned
        sum_of_samples                 = other.sum_of_samples;
        sum_of_squared_samples         = other.sum_of_squared_samples;
        number_of_samples_NON_integral = other.number_of_samples_NON_integral;
        kahan_correction_sum           = other.kahan_correction_sum;
        kahan_correction_sum_sqs       = other.kahan_correction_sum_sqs;
        number_of_samples              = other.number_of_samples;
        minimum                        = other.minimum;
        maximum                        = other.maximum;
        is_constant                    = other.is_constant;
        last_added_value               = other.last_added_value;
        mean_welford                   = other.mean_welford;
        var_times_n_minus_1_welford    = other.var_times_n_minus_1_welford;
        is_welford                     = other.is_welford;
        is_default                     = other.is_default;
    }
    return *this;
}

template <class Accumulator_t>
EmpiricalDistribution<Accumulator_t>& EmpiricalDistribution<Accumulator_t>::operator=(EmpiricalDistribution&& other) noexcept {
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
    if ( this != &other ) {
        // constructed_in_parallel_region is const and intentionally not reassigned
        sum_of_samples                 = other.sum_of_samples;
        sum_of_squared_samples         = other.sum_of_squared_samples;
        number_of_samples_NON_integral = other.number_of_samples_NON_integral;
        kahan_correction_sum           = other.kahan_correction_sum;
        kahan_correction_sum_sqs       = other.kahan_correction_sum_sqs;
        number_of_samples              = other.number_of_samples;
        minimum                        = other.minimum;
        maximum                        = other.maximum;
        is_constant                    = other.is_constant;
        last_added_value               = other.last_added_value;
        mean_welford                   = other.mean_welford;
        var_times_n_minus_1_welford    = other.var_times_n_minus_1_welford;
        is_welford                     = other.is_welford;
        is_default                     = other.is_default;
    }
    return *this;
}

template <class Accumulator_t>
EmpiricalDistribution<Accumulator_t>::~EmpiricalDistribution( ) {
    // Here, would deallocate sample_values array
}

template <class Accumulator_t>
void EmpiricalDistribution<Accumulator_t>::Reset( ) {
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
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
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
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
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
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
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
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
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
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
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
    MyDebugAssertFalse(is_welford, "Cannot use IsConstant with Welford accumulator.");
    return is_constant;
}

template <class Accumulator_t>
float EmpiricalDistribution<Accumulator_t>::GetSampleSumOfSquares( ) {
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
    if ( is_welford )
        return var_times_n_minus_1_welford;

    else
        return sum_of_squared_samples;
}

template <class Accumulator_t>
long EmpiricalDistribution<Accumulator_t>::GetNumberOfSamples( ) {
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
    if ( is_welford )
        return long(number_of_samples_NON_integral);
    else
        return number_of_samples;
}

template <class Accumulator_t>
float EmpiricalDistribution<Accumulator_t>::GetSampleSum( ) {
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
    if ( is_welford )
        return mean_welford * number_of_samples_NON_integral;
    else
        return sum_of_samples;
}

template <class Accumulator_t>
float EmpiricalDistribution<Accumulator_t>::GetSampleMean( ) {
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
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
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
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
    MyDebugWarnThreadSafety(GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ));
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