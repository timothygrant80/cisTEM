class EmpiricalDistribution {

private:
	double		sum_of_samples;
	double		sum_of_squared_samples;
	long		number_of_samples;
	float		minimum;
	float		maximum;
	bool		is_constant;
	float		last_added_value;

public:

	// Constructors, destructors
	EmpiricalDistribution();
	~EmpiricalDistribution();

	void AddSampleValue(float sample_value);
	float GetNumberOfSamples();
	float GetSampleSum();
	float GetSampleMean();
	float GetSampleVariance();
	float GetSampleSumOfSquares();
	float GetUnbiasedEstimateOfPopulationVariance();
	inline float GetMinimum() {return minimum;};
	inline float GetMaximum() {return maximum;};
	bool IsConstant();
	void PopulateHistogram();
	void Reset();

};
