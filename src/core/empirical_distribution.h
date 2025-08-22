#ifndef _SRC_CORE_EMPIRICAL_DISTRIBUTION_H_
#define _SRC_CORE_EMPIRICAL_DISTRIBUTION_H_

template <class Accumulator_t>
class EmpiricalDistribution {

  private:
    Accumulator_t sum_of_samples;
    Accumulator_t sum_of_squared_samples;
    Accumulator_t number_of_samples_NON_integral;
    Accumulator_t kahan_correction_sum;
    Accumulator_t kahan_correction_sum_sqs;
    long          number_of_samples;
    float         minimum;
    float         maximum;
    bool          is_constant;
    float         last_added_value;

    bool const constructed_in_parallel_region;

    Accumulator_t mean_welford;
    Accumulator_t var_times_n_minus_1_welford;
    // We don't trax min/max/last_added_value  or is_constant for Welford, set this to prevent those methods
    // from being called.
    bool is_welford{ };
    bool is_default{ };

  public:
    // Constructors, destructors
    EmpiricalDistribution( );
    EmpiricalDistribution(const EmpiricalDistribution& other);
    EmpiricalDistribution(EmpiricalDistribution&& other) noexcept;
    ~EmpiricalDistribution( );

    EmpiricalDistribution& operator=(const EmpiricalDistribution& other);
    EmpiricalDistribution& operator=(EmpiricalDistribution&& other) noexcept;

    void  AddSampleValue(float sample_value);
    void  AddSampleValueWithKahanCorrection(float sample_value);
    void  AddSampleValueForWelford(float sample_value);
    void  AddSampleValueForWelfordBatched(float sample_value, float sample_var_times_n_minus_1_welford, Accumulator_t& n_this_batch);
    long  GetNumberOfSamples( );
    float GetSampleSum( );
    float GetSampleMean( );
    float GetSampleVariance( );
    float GetSampleSumOfSquares( );
    float GetUnbiasedEstimateOfPopulationVariance( );

    inline float GetMinimum( ) { return minimum; };

    inline float GetMaximum( ) { return maximum; };

    bool IsConstant( );

    inline bool GetConstructedInParallelRegion( ) const { return constructed_in_parallel_region; }

    void PopulateHistogram( );
    void Reset( );
};

#endif // _SRC_CORE_EMPIRICAL_DISTRIBUTION_H_