#include "core_headers.h"


const double SOLVENT_DENSITY = 0.94; // 0.94 +/- 0.02 Ghormley JA, Hochanadel CJ. 1971

Water::Water()
{

}

Water::Water(const PDB *current_specimen, int wanted_size_neighborhood, float wanted_pixel_size, float wanted_dose_per_frame, float max_tilt)
{

	//


	this->Init( current_specimen, wanted_size_neighborhood, wanted_pixel_size, wanted_dose_per_frame, max_tilt);



}


Water::~Water()
{
	// Destructor? TODO
}

void Water::Init(const PDB *current_specimen, int wanted_size_neighborhood, float wanted_pixel_size, float wanted_dose_per_frame, float max_tilt)
{
	this->number_of_waters = 0;
	this->size_neighborhood = wanted_size_neighborhood;
	this->pixel_size = wanted_pixel_size;
	this->dose_per_frame = wanted_dose_per_frame;

	// Copy over some values from the current specimen
	this->vol_angX = current_specimen->vol_angX;
	this->vol_angY = current_specimen->vol_angY;
	this->vol_angZ = current_specimen->vol_angZ;

	this->vol_nX = ReturnPaddingForTilt(max_tilt, current_specimen->vol_nX);
	this->vol_nY = current_specimen->vol_nY;
	this->vol_nZ = current_specimen->vol_nZ;


	this->vol_oX = floor(this->vol_nX / 2);
	this->vol_oY = floor(this->vol_nY / 2);
	this->vol_oZ = floor(this->vol_nZ / 2);
}

void Water::SeedWaters3d()
{

	// Volume in Ang / (ang^3/nm^3 * nm^3/nWaters) buffer by 10%

	double waters_per_angstrom_cubed = SOLVENT_DENSITY * 602.2140857 / (18.01528 * 1000);
	double n_waters_lower_bound = waters_per_angstrom_cubed*(this->vol_angX * this->vol_angY * this->vol_angZ);
	long n_waters_possible = (long)floor(1.05*n_waters_lower_bound);
	wxPrintf("specimen volume is %3.3e nm expecting %3.3e waters\n",(this->vol_angX * this->vol_angY * this->vol_angZ)/1000,n_waters_lower_bound);



    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0);

	const float random_sigma_cutoff = 1 - (n_waters_lower_bound/double((this->vol_nX - (this->size_neighborhood*this->pixel_size)) *
																	   (this->vol_nY - (this->size_neighborhood*this->pixel_size)) *
																	   (this->vol_nZ - (this->size_neighborhood*this->pixel_size))));
	const float random_sigma_negativo = -1*random_sigma_cutoff;
	float current_random;
	const float random_sigma = 0.5/this->pixel_size; // random shift in pixel values
    float thisRand;
    wxPrintf("cuttoff is %2.6e %2.6e %f %f\n",n_waters_lower_bound,double((this->vol_nX - this->size_neighborhood) *
    																	  (this->vol_nY - this->size_neighborhood) *
    		 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	  (this->vol_nZ - this->size_neighborhood)), random_sigma_cutoff, random_sigma_negativo);

	long padding_jump_value;
	if (IsEven(this->vol_nX) == true) padding_jump_value = 2;
	else padding_jump_value = 1;


	// TODO calculate a table suitable for a range of reasonable pixel values (or figure out an ~analytic formula)
	// TODO add option for Sodium & Chloride to be added (also requires the ionic term in the potential calculation)


	int i;
	int j;
	int k;



	// Allocate the memory
	this->Allocate(3*n_waters_possible,1,1,true);
	this->SetToConstant(0.0);


	for (k = 0 + this->size_neighborhood; k < this->vol_nZ - this->size_neighborhood; k++)
	{
		for (j = 0 + this->size_neighborhood; j < this->vol_nY - this->size_neighborhood; j++)
		{
			for (i = 0 + this->size_neighborhood; i < this->vol_nX - this->size_neighborhood; i++)
			{
				current_random = dis(gen);//(double)global_random_number_generator.GetUniformRandom();
				// taking the abs is craaazzyy slow compared to checking both constants
				if (  current_random > random_sigma_cutoff ) //|| current_random < random_sigma_negativo )
				{

					real_values[0 + 3*number_of_waters] = i;
					real_values[1 + 3*number_of_waters] = j;
					real_values[2 + 3*number_of_waters] = k;



					number_of_waters++;


				}


			}
		}

	//wxPrintf("waters added %ld\n",this->n_waters_added);
	}

	wxPrintf("waters added %ld\n",this->number_of_waters);

}

void Water::ShakeWaters3d(int number_of_threads)
{

	// Try just using the functions output, which may not be quite perfect in distribution, but close.
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()


	float azimuthal;
	float cos_polar;


	const float random_sigma = (dose_per_frame); // TODO check this. The point is to convert an rms displacement in 3D to 2D

    std::normal_distribution<float>  norm_dist_mag(0.0,random_sigma*1.5);

	wxPrintf("Using a rmsd of %f for water perturbation\n", random_sigma);


	// Private variables for parfor loop
	float dr, dx, dy, dz;
	float azimuthal_angle = 0;
	float polar_angle = 0;


	int local_threads;
	if (number_of_threads > 4)
	{
		local_threads = 4;
	}
	else
	{
		local_threads = number_of_threads;
	}

	#pragma omp parallel for num_threads(local_threads) private(dr,dx,dy,dz,norm_dist_mag)

	for (long iWater = 0; iWater < number_of_waters; iWater++)
	{


//		#pragma omp simd
		for (int iGen = 0; iGen < 3; iGen++)
		{
			real_values[iWater*3 + iGen] += norm_dist_mag(gen);
		}

		// TODO 2x check that the periodic shifts are doing what they should be.
		// Check boundaries
		if (real_values[iWater*3 + 0] < size_neighborhood + 1)
		{
			real_values[iWater*3 + 0] = real_values[iWater*3 + 0] - 1*size_neighborhood +vol_nX;
		}
		else if (real_values[iWater*3 + 0] > vol_nX - size_neighborhood)
		{
			real_values[iWater*3 + 0] = real_values[iWater*3 + 0] - vol_nX + 1*size_neighborhood;
		}

		// Check boundaries
		if (real_values[iWater*3 + 1] < size_neighborhood + 1)
		{
			real_values[iWater*3 + 1] = real_values[iWater*3 + 1] - 1*size_neighborhood +vol_nY;
		}
		else if (real_values[iWater*3 + 1] > vol_nY - size_neighborhood)
		{
			real_values[iWater*3 + 1] = real_values[iWater*3 + 1] - vol_nY + 1*size_neighborhood;
		}

		// Check boundaries
		if (real_values[iWater*3 + 2] < size_neighborhood + 1)
		{
			real_values[iWater*3 + 2] = real_values[iWater*3 + 2] - 1*size_neighborhood +vol_nZ;
		}
		else if (real_values[iWater*3 + 2] > vol_nZ - size_neighborhood)
		{
			real_values[iWater*3 + 2] = real_values[iWater*3 + 2] - vol_nZ + 1*size_neighborhood;
		}

		//wxPrintf("Water %ld ending at x %f y %f z %f \n",iWater,this->water_x[iWater] ,this->water_y[iWater] ,this->water_z[iWater]);

	}


}

int Water::ReturnPaddingForTilt(float max_tilt, int current_nX)
{
	// Assuming tilting only along the Y-axis
	// TODO consider rotations of the projection which will also bring new water into view

	MyDebugAssertTrue(max_tilt < 70.0,"maximum tilt angle supported is 70 degress")
    if (fabsf(max_tilt) < 1e-1) { return current_nX ;}

	int padded_nX;
	float cos_max_tilt = cos(max_tilt * (float)PI / 180.0f);
	padded_nX = myroundint((float)current_nX / cos_max_tilt);
	return padded_nX;


}


