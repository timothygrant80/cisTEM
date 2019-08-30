#include "core_headers.h"

ElectronDose::ElectronDose()
{
	acceleration_voltage = -1;

	critical_dose_a = 0.0;
	critical_dose_b = 0.0;
	critical_dose_c = 0.0;

	voltage_scaling_factor = 0.0;

	pixel_size = 0.0;
}

ElectronDose::ElectronDose(float wanted_acceleration_voltage, float wanted_pixel_size)
{
	acceleration_voltage = -1;

	critical_dose_a = 0.0;
	critical_dose_b = 0.0;
	critical_dose_c = 0.0;

	voltage_scaling_factor = 0.0;

	pixel_size = 0.0;

	Init(wanted_acceleration_voltage, wanted_pixel_size);

}

void ElectronDose::Init(float wanted_acceleration_voltage, float wanted_pixel_size)
{
	if (wanted_acceleration_voltage < 301 && wanted_acceleration_voltage > 299)
	{
		acceleration_voltage = 300.0;
		voltage_scaling_factor = 1.0;
	}
	else
	if (wanted_acceleration_voltage < 201 && wanted_acceleration_voltage > 199)
	{
		acceleration_voltage = 200.0;
		voltage_scaling_factor = 0.8; // if this is based on the ratio of the wavelengths, shouldn't it be 0.785?
	}
	else
	if (wanted_acceleration_voltage < 101 && wanted_acceleration_voltage >  99)
	{
		acceleration_voltage = 100.0;
		voltage_scaling_factor = 0.532;
	}
	else
	{
		wxPrintf("Error: Unsupported voltage (%f)\n\n", wanted_acceleration_voltage);
		DEBUG_ABORT;
	}


	pixel_size = wanted_pixel_size;

	critical_dose_a = 0.24499;
	critical_dose_b = -1.6649;
	critical_dose_c = 2.8141;

}

void ElectronDose::CalculateDoseFilterAs1DArray(Image *ref_image, float *filter_array, float dose_start, float dose_finish)
{

	MyDebugAssertTrue(ref_image->logical_z_dimension == 1, "Reference Image is a 3D!");

	int i;
	int j;

	float x;
	float y;

	float current_critical_dose;
	float current_optimal_dose;
	float current_filter_value;

	int array_counter = 0;

	for (j = 0; j < ref_image->logical_y_dimension; j++)
	{
		y = pow(ref_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * ref_image->fourier_voxel_size_y, 2);

		for (i = 0; i <= ref_image->physical_upper_bound_complex_x; i++)
		{
			if (i == 0 && j == 0) filter_array[array_counter] = 1;
			else
			{
				x = pow(i * ref_image->fourier_voxel_size_x, 2);

				current_critical_dose = ReturnCriticalDose(sqrt(x+y) / pixel_size);
				current_optimal_dose =  2.51284 * current_critical_dose;

				// if the starting dose is already above the optimal dose, set to 0, otherwise calculate the filter..

//				if (dose_start > current_optimal_dose) filter_array[array_counter] =  0.0;
//				else filter_array[array_counter] = ReturnDoseFilter(dose_finish, current_critical_dose);
				filter_array[array_counter] = ReturnDoseFilter(dose_finish, current_critical_dose);

			}

			array_counter++;
		}
	}

}
