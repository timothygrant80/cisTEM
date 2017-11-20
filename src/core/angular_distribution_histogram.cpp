#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayofAngularDistributionHistograms);

AngularDistributionHistogram::AngularDistributionHistogram()
{
	number_of_theta_steps = 0;
	number_of_phi_steps = 0;
}

AngularDistributionHistogram::~AngularDistributionHistogram()
{

}

void AngularDistributionHistogram::Clear()
{
	histogram_data.Clear();
	phi_boundaries.Clear();
	theta_boundaries.Clear();
}

void AngularDistributionHistogram::Init(int wanted_number_of_theta_steps, int wanted_number_of_phi_steps)
{
	float theta_step;
	float phi_step;
	float cosine_angle;

	number_of_theta_steps = wanted_number_of_theta_steps;
	number_of_phi_steps = wanted_number_of_phi_steps;

	theta_step = 90.0 / number_of_theta_steps;
	phi_step = 360.0 / number_of_phi_steps;

	histogram_data.Clear();
	histogram_data.Add(0, number_of_theta_steps * number_of_phi_steps);

	theta_boundaries.Clear();
	phi_boundaries.Clear();

	for (float current_theta = 90.0f - theta_step; current_theta > 0.0 ; current_theta-= theta_step)
	{
		cosine_angle = rad_2_deg(acos(current_theta / 90.0f));
		theta_boundaries.Add(cosine_angle);
	}

	for (float current_phi = phi_step; current_phi < 360.0; current_phi += phi_step)
	{
		phi_boundaries.Add(current_phi);
	}

}

void AngularDistributionHistogram::AddPosition(float theta, float phi)
{
	// which bin is this going into?

	int theta_bin = ReturnThetaBin(theta);
	int phi_bin = ReturnPhiBin(phi);

	histogram_data[number_of_theta_steps * phi_bin + theta_bin]++;
}

float AngularDistributionHistogram::GetHistogramValue(float theta, float phi)
{

	int theta_bin = ReturnThetaBin(theta);
	int phi_bin = ReturnPhiBin(phi);

	return histogram_data[number_of_theta_steps * phi_bin + theta_bin];
}

void AngularDistributionHistogram::GetMinMaxValues(float &min_value, float &max_value)
{
	MyDebugAssertTrue(histogram_data.GetCount() > 0, "No DATA!");

	min_value = FLT_MAX;
	max_value = -FLT_MAX;

	for (int location_counter = 0; location_counter < histogram_data.GetCount(); location_counter++)
	{
		min_value = std::min(min_value, histogram_data[location_counter]);
		max_value = std::max(max_value, histogram_data[location_counter]);
	}
}

void AngularDistributionHistogram::GetDistributionStatistics(float &min_value, float &max_value, float &average_value, float &std_dev)
{
	MyDebugAssertTrue(histogram_data.GetCount() > 0, "No DATA!");

	float total = 0.0f;
	float total_squared = 0.0f;

	min_value = FLT_MAX;
	max_value = -FLT_MAX;
	average_value = 0.0f;
	std_dev = 0.0f;

	for (int location_counter = 0; location_counter < histogram_data.GetCount(); location_counter++)
	{
		min_value = std::min(min_value, histogram_data[location_counter]);
		max_value = std::max(max_value, histogram_data[location_counter]);

		total += histogram_data[location_counter];
		total_squared += powf(histogram_data[location_counter], 2);
	}

    if (total != 0.0f)
    {
    	average_value = total / float(histogram_data.GetCount());
	}

    if (total_squared != 0.0f)
    {
    	std_dev = sqrtf((total_squared / float(histogram_data.GetCount())) - pow(average_value, 2));
    }
}

void AngularDistributionHistogram::PrintToTerminal()
{
	wxPrintf("\n\nThere are %li Bins.\n\n", histogram_data.GetCount());

	for (int counter = 0; counter < histogram_data.GetCount(); counter++)
	{
		wxPrintf("Bin %i = %f\n", counter + 1, histogram_data[counter]);
	}

	wxPrintf("\n\n");
}

