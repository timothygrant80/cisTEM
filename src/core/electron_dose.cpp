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
  reduced_critical_dose_b = critical_dose_b / 2.f;

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

  const float reduced_fourier_voxel_size = ref_image->fourier_voxel_size_x / pixel_size;
  // The spatial frequency is calculated as sqrt(x^2 + y^2 + z^2)/pixel_size.
  // To remove the sqrt call, we square the pixel size to move it inside, and then absorb the sqrt i.e. () ^ 1/2 into the exponent critical_dose_b -> critical_dose_b / 2
  pixel_size *= pixel_size;
	for (k = 0; k <= ref_image->physical_upper_bound_complex_z; k++)
	{
		z = ref_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * ref_image->fourier_voxel_size_z;
		z = (z*z/pixel_size);

		for (j = 0; j <= ref_image->physical_upper_bound_complex_y; j++)
		{
			y = ref_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * ref_image->fourier_voxel_size_y;
			y = (y*y/pixel_size);

			for (i = 0; i <= ref_image->physical_upper_bound_complex_x; i++)
			{
        x = i * reduced_fourier_voxel_size;
        filter_array[array_counter] = ReturnDoseFilter(dose_finish, ReturnCriticalDose(x*x + y + z));
				array_counter++;
			}
		}
	}
  
  filter_array[0] = 1.0;
}

void ElectronDose::CalculateCummulativeDoseFilterAs1DArray(Image *ref_image, float *filter_array, float dose_start, float dose_finish)
{

//	MyDebugAssertTrue(ref_image->logical_z_dimension == 1, "Reference Image is a 3D!");

	int i;
	int j;
	int k;

	float x;
	float y;
	float z;

	int array_counter = 0;
  const float reduced_fourier_voxel_size = ref_image->fourier_voxel_size_x / pixel_size;
  // The spatial frequency is calculated as sqrt(x^2 + y^2 + z^2)/pixel_size.
  // To remove the sqrt call, we square the pixel size to move it inside, and then absorb the sqrt i.e. () ^ 1/2 into the exponent critical_dose_b -> critical_dose_b / 2
  pixel_size *= pixel_size;

	for (k = 0; k <= ref_image->physical_upper_bound_complex_z; k++)
	{
		z = ref_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * ref_image->fourier_voxel_size_z;
		z = (z*z/pixel_size);


		for (j = 0; j <= ref_image->physical_upper_bound_complex_y; j++)
		{
			y = ref_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * ref_image->fourier_voxel_size_y;
			y = (y*y/pixel_size);

			for (i = 0; i <= ref_image->physical_upper_bound_complex_x; i++)
			{
				if (i == 0 && j == 0 && k == 0) filter_array[array_counter] = 1;
				else
				{
          x = i * reduced_fourier_voxel_size;
					filter_array[array_counter] = ReturnCummulativeDoseFilter(dose_start, dose_finish, ReturnCriticalDose(x*x + y + z));
				}

				array_counter++;
			}
		}
	}

  filter_array[0] = 1.0;
}
