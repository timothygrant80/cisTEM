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

//	MyDebugAssertTrue(ref_image->logical_z_dimension == 1, "Reference Image is a 3D!");

	int i;
	int j;
	int k;

	float x;
	float y;
	float z;

	int array_counter = 0;

	for (k = 0; k <= ref_image->physical_upper_bound_complex_z; k++)
	{
		z = ref_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * ref_image->fourier_voxel_size_z;
		z *= z;

		for (j = 0; j <= ref_image->physical_upper_bound_complex_y; j++)
		{
			y = ref_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * ref_image->fourier_voxel_size_y;
			y *= y;

			for (i = 0; i <= ref_image->physical_upper_bound_complex_x; i++)
			{
				if (i == 0 && j == 0) filter_array[array_counter] = 1;
				else
				{
					x = i * ref_image->fourier_voxel_size_x;
					filter_array[array_counter] = ReturnDoseFilter(dose_finish, ReturnCriticalDose(sqrtf(x*x + y + z) / pixel_size));
				}

				array_counter++;
			}
		}
	}

}

void ElectronDose::CalculateCummulativeDoseFilterAs1DArray(Image *ref_image, float *filter_array, float dose_start, float dose_finish, float shift_x = 0.0f, float shift_y = 0.0f)
{

//	MyDebugAssertTrue(ref_image->logical_z_dimension == 1, "Reference Image is a 3D!");
	// The exposure causes both radiation damage && specimen motion. Optionally include a shift in X/Y from movie alignment, which is a lower bound on the intraframe motion.
	// Approximate the envelope from sinc(pi * <q,d>) as exp(-pi/6 * <q,d>)

	int i;
	int j;
	int k;

	float x;
	float y;
	float z;

	int array_counter = 0;

	for (k = 0; k <= ref_image->physical_upper_bound_complex_z; k++)
	{
		z = ref_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * ref_image->fourier_voxel_size_z;
		z *= z;

		for (j = 0; j <= ref_image->physical_upper_bound_complex_y; j++)
		{
			y = ref_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * ref_image->fourier_voxel_size_y;
			y *= y;

			for (i = 0; i <= ref_image->physical_upper_bound_complex_x; i++)
			{
				if (i == 0 && j == 0 && k == 0) filter_array[array_counter] = 1;
				else
				{
					x = i * ref_image->fourier_voxel_size_x;
					filter_array[array_counter] = ReturnCummulativeDoseFilter(dose_start, dose_finish, ReturnCriticalDose(sqrtf(x*x + y + z) / pixel_size));
				}

				array_counter++;
			}
		}
	}

}
