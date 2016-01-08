class EmpiricalDistribution {

private:
	double		sum_of_samples;
	double		sum_of_squared_samples;
	long		number_of_samples;
	float		minimum;
	float		maximum;
	bool		keep_sample_values;
	float		*sample_values;

public:

	// Constructors, destructors
	EmpiricalDistribution();
	EmpiricalDistribution(bool should_keep_sample_values);
	~EmpiricalDistribution();

	void AddSampleValue(float sample_value);
	float GetNumberOfSamples();
	float GetSampleSum();
	float GetSampleMean();
	float GetSampleVariance();
	float GetSampleSumOfSquares();
	float GetUnbiasedEstimateOfPopulationVariance();
	void PopulateHistogram();
	void Reset();

};
