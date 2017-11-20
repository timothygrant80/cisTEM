class AngularDistributionHistogram
{

public:

	wxArrayFloat histogram_data;
	wxArrayFloat phi_boundaries;
	wxArrayFloat theta_boundaries;

	int number_of_theta_steps;
	int number_of_phi_steps;

	AngularDistributionHistogram();
	~AngularDistributionHistogram();

	void AddPosition(float theta, float phi);
	float GetHistogramValue(float theta, float phi);
	void Init(int wanted_number_of_theta_steps, int wanted_number_of_phi_steps);

	void GetMinMaxValues(float &min_value, float &max_value);
	void GetDistributionStatistics(float &min_value, float &max_value, float &average_value, float &std_dev);
	void PrintToTerminal();

	void Clear();


	inline int ReturnThetaBin(float theta)
	{
		//MyDebugAssertTrue(theta >= 0.0f && theta <= 90.0f, "Theta out of range here")

		int theta_bin;

		for (theta_bin = 0; theta_bin < theta_boundaries.GetCount(); theta_bin++)
		{
			if (theta < theta_boundaries[theta_bin])
			{
				return theta_bin;
			}
		}

		// if we get here must be last bin..

		return theta_boundaries.GetCount();
	}

	inline int ReturnPhiBin(float phi)
	{
		//MyDebugAssertTrue(phi >= 0.0f && phi <= 360.0f, "Phi out of range here")

		//put it in the range of 0-360;

		while (phi < 0.0f) phi+=360.0f;
		while (phi > 360.f) phi-=360.0f;

		int phi_bin;

		for (phi_bin = 0; phi_bin < phi_boundaries.GetCount(); phi_bin++)
		{
			if (phi < phi_boundaries[phi_bin])
			{
				return phi_bin;
			}
		}

		// if we got here, must be last bin..

		return phi_boundaries.GetCount();
	}
};

WX_DECLARE_OBJARRAY(AngularDistributionHistogram, ArrayofAngularDistributionHistograms);
