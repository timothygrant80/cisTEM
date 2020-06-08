#include "./matplotlibcpp.h"

#include "../../core/core_headers.h"

class
CtfplotApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};


IMPLEMENT_APP(CtfplotApp)

// override the DoInteractiveUserInput

void CtfplotApp::DoInteractiveUserInput()
{


}

// override the do calculation method which will be what is actually run..

bool CtfplotApp::DoCalculation()
{

	/*
	 * Setup the parameters
	 */
	const float pixel_size_angstroms = 1.5;
	const float acceleration_voltage_kv = 300.0;
	const float spherical_aberration_mm = 2.7;
	const float defocus_1_angstroms = 20000.0;
	const float defocus_2_angstroms = 21000.0;
	const float defocus_3_angstroms = 20500.0;
	const float defocus_fit_angstroms = 21900.0;
	const float additional_phase_shift_radians = 0.0;
	const float high_res_for_plotting_angstroms = 4.0;
	CTF ctfone;
	CTF ctftwo;
	CTF ctfthree;
	CTF ctffit;

	// Initialize CTF
	ctfone.Init(acceleration_voltage_kv,spherical_aberration_mm,0.07,defocus_1_angstroms,defocus_1_angstroms,0.0,pixel_size_angstroms,additional_phase_shift_radians);
	ctftwo.Init(acceleration_voltage_kv,spherical_aberration_mm,0.07,defocus_2_angstroms,defocus_2_angstroms,0.0,pixel_size_angstroms,additional_phase_shift_radians);
	ctfthree.Init(acceleration_voltage_kv,spherical_aberration_mm,0.07,defocus_3_angstroms,defocus_3_angstroms,0.0,pixel_size_angstroms,additional_phase_shift_radians);
	ctffit.Init(acceleration_voltage_kv,spherical_aberration_mm,0.07,defocus_fit_angstroms,defocus_fit_angstroms,0.0,pixel_size_angstroms,additional_phase_shift_radians);

	/*
	 * Do the plotting
	 */

	namespace plt = matplotlibcpp;

	// Prepare data.
	int n = 5000;
	std::vector<double> sf(n), curve1(n), curve2(n), curve3(n);
	float current_sf;
	float current_weight_protein;
	float current_weight_digitonin;
	for(int i=0; i<n; ++i) {
		current_sf = float(i)/float(n-1) * pixel_size_angstroms / high_res_for_plotting_angstroms;
		sf.at(i) = current_sf;
		//
		current_weight_digitonin = 0.0;
		if (current_sf > 0.12 && current_sf < 0.21) current_weight_digitonin = (current_sf - 0.12)/(0.21-0.12);
		if (current_sf >= 0.21) current_weight_digitonin = 1.0;

		//
		curve1.at(i) = fabs(   ctfone.Evaluate(current_sf * current_sf, 0.0));
		curve2.at(i) = 0.5 * fabs(   ctfthree.Evaluate(current_sf * current_sf, 0.0) + ctftwo.Evaluate(current_sf * current_sf, 0.0));
		curve3.at(i) = curve2.at(i) / curve1.at(i);

		//curve2.at(i) = fabs(   ctfone.Evaluate(current_sf * current_sf, 0.0)
		//					 + (ctfthree.Evaluate(current_sf * current_sf,0.0) * current_weight_digitonin)) / (1.0 + current_weight_digitonin);
	}

	// Plot a line whose name will show up as "log(x)" in the legend.
	plt::named_plot("ctf1", sf, curve1);
	plt::named_plot("ctf1 + ctf2", sf, curve2);
	//plt::named_plot("ctf3", sf, ctf3);


	// Enable legend.
	plt::legend();

	plt::show();

	return true;
}
