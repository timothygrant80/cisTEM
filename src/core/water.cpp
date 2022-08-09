#include "core_headers.h"

const double SOLVENT_DENSITY = 0.94; // 0.94 +/- 0.02 Ghormley JA, Hochanadel CJ. 1971
const double CARBON_DENSITY  = 1.75; // 2.0; // NIST and Holography paper TODO add cite (using the lower density to match the Holography paper)
const double MW_WATER        = 18.01528;
const double MW_CARBON       = 12.0107;
const double CARBON_X_ANG    = 384.0;
const double CARBON_Y_ANG    = 384.0;

Water::Water(bool do_carbon) {
    this->simulate_phase_plate = do_carbon;
}

Water::Water(const PDB* current_specimen, int wanted_size_neighborhood, float wanted_pixel_size, float wanted_dose_per_frame, RotationMatrix max_rotation, float in_plane_rotation, int* padX, int* padY, int nThreads, bool pad_based_on_rotation, bool do_carbon) {

    //
    this->simulate_phase_plate = do_carbon;
    this->Init(current_specimen, wanted_size_neighborhood, wanted_pixel_size, wanted_dose_per_frame, max_rotation, in_plane_rotation, padX, padY, nThreads, pad_based_on_rotation);
}

Water::~Water( ) {
    if ( is_allocated_water_coords ) {
        delete[] water_coords;
    }
}

void Water::Init(const PDB* current_specimen, int wanted_size_neighborhood, float wanted_pixel_size, float wanted_dose_per_frame, RotationMatrix max_rotation, float in_plane_rotation, int* padX, int* padY, int nThreads, bool pad_based_on_rotation) {

    this->size_neighborhood = wanted_size_neighborhood;
    this->pixel_size        = wanted_pixel_size;
    this->dose_per_frame    = wanted_dose_per_frame;

    this->nThreads = nThreads;

    // This input values in padX and padY if > 0 are the wanted output dimensions + padding for taper. Make sure solvent is at least this big if provided.
    int check_min_paddingX = *padX;
    int check_min_paddingY = *padY;

    if ( this->simulate_phase_plate ) {
        this->vol_angX = CARBON_X_ANG;
        this->vol_angY = CARBON_Y_ANG;
        this->vol_angZ = max_rotation.m[0][0];

        this->vol_nX = myroundint(this->vol_angX / wanted_pixel_size);
        this->vol_nY = myroundint(this->vol_angY / wanted_pixel_size);
        this->vol_nZ = myroundint(this->vol_angZ / wanted_pixel_size);
        if ( IsEven(this->vol_nZ) == false )
            this->vol_nZ += 1;
    }
    else {

        int padZ = 0;
        vol_nZ   = current_specimen->vol_nZ;

        wxPrintf("size pre rot padding %d %d %f rot\n", current_specimen->vol_nX, current_specimen->vol_nY, in_plane_rotation);

        // The padding function should be renamed to something that reflects that it adds padding based on that needed for rotation.
        // The boolean param should be renamed to reflect this as well.
        if ( ! pad_based_on_rotation ) {
            ReturnPadding(max_rotation, in_plane_rotation, current_specimen->vol_nZ, current_specimen->vol_nX, current_specimen->vol_nY, padX, padY, &padZ);
        }

        vol_nX = current_specimen->vol_nX + (*padX); // + padZ; // This assumes the tilting is only around the Y-Axis which isn't correct FIXME
        vol_nY = current_specimen->vol_nY + (*padY);
        vol_nZ = current_specimen->vol_nZ; // + 2*padZ;

        //		wxPrintf("size post rot 1 padding %d %d %f rot\n", current_specimen->vol_nX, current_specimen->vol_nY, in_plane_rotation);
        wxPrintf("size post rot 1 padding %d %d %f rot\n", vol_nX, vol_nY, in_plane_rotation);

        //		if (check_min_paddingX > 0)
        //		{
        //			int x_diff = vol_nX - check_min_paddingX;
        //			wxPrintf("Xdiff is %d\n",x_diff);
        //			if (x_diff < 0)
        //			{
        //				vol_nX = check_min_paddingX;
        //				*padX = -x_diff;
        //			}
        //			else
        //			{
        //				vol_nX = current_specimen->vol_nX;
        //				*padX = 0;
        //			}
        //
        //		}
        //		if  (check_min_paddingY > 0)
        //		{
        //			int y_diff = vol_nY - check_min_paddingY;
        //			wxPrintf("yiff is %d\n",y_diff);
        //
        //			if (y_diff < 0)
        //			{
        //				vol_nY = check_min_paddingY;
        //				*padY = -y_diff;
        //			}
        //			else
        //			{
        //				vol_nY = current_specimen->vol_nY;
        //				*padY = 0;
        //			}
        //		}

        wxPrintf("size post rot 2 padding %d %d padX %d padY %d padZ %d rot\n", vol_nX, vol_nY, *padX, *padY, padZ);
        MyAssertTrue(current_specimen->pixel_size > 0.0f, "The pixel size for your PDB object is not yet set.");
        // Copy over some values from the current specimen - Do these need to be updated for tilts and rotations?
        this->vol_angX = vol_nX * current_specimen->pixel_size; //current_specimen->vol_angX;
        this->vol_angY = vol_nY * current_specimen->pixel_size;
        this->vol_angZ = vol_nZ * current_specimen->pixel_size;
    }

    // wxPrintf("vol dimension in Ang %2.2f x %2.2f y  %2.2f z\n", this->vol_angX , this->vol_angY , this->vol_angZ);

    this->vol_oX = floor(this->vol_nX / 2);
    this->vol_oY = floor(this->vol_nY / 2);
    this->vol_oZ = floor(this->vol_nZ / 2);
}

void Water::SeedWaters3d( ) {

    // Volume in Ang / (ang^3/nm^3 * nm^3/nWaters) buffer by 10%

    double waters_per_angstrom_cubed;

    if ( this->simulate_phase_plate ) {
        // g/cm^3 * molecules/mole * mole/grams * 1cm^3/10^24 angstrom^3
        waters_per_angstrom_cubed = CARBON_DENSITY * 0.6022140857 / MW_CARBON;
    }
    else {
        waters_per_angstrom_cubed = SOLVENT_DENSITY * 0.6022140857 / MW_WATER;
    }

    wxPrintf("Atoms per nm^3 %3.3f, vol (in Ang^3) %2.2f %2.2f %2.2f\n", waters_per_angstrom_cubed * 1000, this->vol_angX, this->vol_angY, this->vol_angZ);
    double n_waters_lower_bound = waters_per_angstrom_cubed * (this->vol_angX * this->vol_angY * this->vol_angZ);
    long   n_waters_possible    = (long)floor(1.1 * n_waters_lower_bound); // maybe make this a real vector so it is extensible.
    // wxPrintf("specimen volume is %3.3e nm expecting %3.3e waters\n",(this->vol_angX * this->vol_angY * this->vol_angZ)/1000,n_waters_lower_bound);

    RandomNumberGenerator my_rand(PIf);

    // FIXME is the multiplication by pixel size correct? I am not so sure.
    const float random_sigma_cutoff   = 1 - (n_waters_lower_bound / double((this->vol_nX - (this->size_neighborhood * this->pixel_size)) *
                                                                         (this->vol_nY - (this->size_neighborhood * this->pixel_size)) *
                                                                         (this->vol_nZ - (this->size_neighborhood * this->pixel_size))));
    const float random_sigma_negativo = -1 * random_sigma_cutoff;
    float       current_random;
    const float random_sigma = 0.5 / this->pixel_size; // random shift in pixel values
    float       thisRand;
    // wxPrintf("cuttoff is %2.6e %2.6e %f %f\n",n_waters_lower_bound,double((this->vol_nX - this->size_neighborhood) *
    // 																	  (this->vol_nY - this->size_neighborhood) *
    // 		 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	  (this->vol_nZ - this->size_neighborhood)), random_sigma_cutoff, random_sigma_negativo);

    water_coords              = new AtomPos[n_waters_possible];
    is_allocated_water_coords = true;

    //  There are millions to billions of waters. We want to schedule the threads in a way that avoids atomic collisions
    //  Since the updates are in a projected potential, this means we want a given thread to be assigned a block of waters that DO
    //  overlap in Z, which it will handle serially. This is why K is on the inner loop here. To further optimize this, we can also increment the x/y dimensions
    //  in multiples of the neigborhood.

    // Break up the x/y dims into ~ nThreads^2 thread blocks
    int incX = ceil(vol_nX / nThreads);
    int incY = ceil(vol_nY / nThreads);
    int iLower, iUpper, jLower, jUpper, xUpper, yUpper;

    xUpper = this->vol_nX - this->size_neighborhood;
    yUpper = this->vol_nY - this->size_neighborhood;

    for ( int i = 0; i < nThreads; i++ ) {
        iLower = i * incX + size_neighborhood;
        iUpper = (1 + i) * incX + size_neighborhood;
        for ( int j = 0; j < nThreads; j++ ) {
            jLower = j * incY + size_neighborhood;
            jUpper = (1 + j) * incY + size_neighborhood;

            //			for (int k = this->size_neighborhood; k < this->vol_nZ - this->size_neighborhood; k++)
            for ( int k = 0; k < this->vol_nZ; k++ )

            {
                for ( int iInner = iLower; iInner < iUpper; iInner++ ) {
                    //					if (iInner > xUpper) { continue; }
                    for ( int jInner = jLower; jInner < jUpper; jInner++ ) {
                        //						if (jInner > yUpper) { continue; }

                        if ( my_rand.GetUniformRandomSTD(0.0, 1.0) > random_sigma_cutoff ) {

                            water_coords[number_of_waters].x = (float)iInner;
                            water_coords[number_of_waters].y = (float)jInner;
                            water_coords[number_of_waters].z = (float)k;
                            number_of_waters++;
                        }
                    }
                }
            }
        }
    }

    wxPrintf("waters added %3.3e (%2.2f%)\n", (float)this->number_of_waters, 100.0f * (float)this->number_of_waters / n_waters_lower_bound);
}

void Water::ShakeWaters3d(int number_of_threads) {

    float azimuthal;
    float cos_polar;

    const float random_sigma = 1.5f * (dose_per_frame); // TODO check this. The point is to convert an rms displacement in 3D to 2D

    // Try just using the functions output, which may not be quite perfect in distribution, but close.
    //    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    //    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    //    std::normal_distribution<float>  norm_dist_mag(0.0,random_sigma*1.5);

    // wxPrintf("Using a rmsd of %f for water perturbation\n", random_sigma);

    // Private variables for parfor loop
    float dr, dx, dy, dz;
    float azimuthal_angle = 0;
    float polar_angle     = 0;

    // TODO benchmark this
    int local_threads;
    if ( number_of_threads > 4 ) {
        local_threads = 4;
    }
    else {
        local_threads = number_of_threads;
    }

    RandomNumberGenerator my_rand(local_threads);

    if ( (float)number_of_waters < (powf(2, 31) - 1) / 3.0f ) {
#pragma omp parallel for num_threads(local_threads) private(dr, dx, dy, dz, my_rand)
        for ( long iWater = 0; iWater < number_of_waters; iWater++ ) {

            water_coords[iWater].x += my_rand.GetNormalRandomSTD(0.0f, random_sigma);
            water_coords[iWater].y += my_rand.GetNormalRandomSTD(0.0f, random_sigma);
            water_coords[iWater].z += my_rand.GetNormalRandomSTD(0.0f, random_sigma);

            // TODO 2x check that the periodic shifts are doing what they should be.
            // Check boundaries
            if ( water_coords[iWater].x < size_neighborhood + 1 ) {
                water_coords[iWater].x = water_coords[iWater].x - 1 * size_neighborhood + vol_nX;
            }
            else if ( water_coords[iWater].x > vol_nX - size_neighborhood ) {
                water_coords[iWater].x = water_coords[iWater].x - vol_nX + 1 * size_neighborhood;
            }

            // Check boundaries
            if ( water_coords[iWater].y < size_neighborhood + 1 ) {
                water_coords[iWater].y = water_coords[iWater].y - 1 * size_neighborhood + vol_nY;
            }
            else if ( water_coords[iWater].y > vol_nY - size_neighborhood ) {
                water_coords[iWater].y = water_coords[iWater].y - vol_nY + 1 * size_neighborhood;
            }

            // Check boundaries
            if ( water_coords[iWater].z < size_neighborhood + 1 ) {
                water_coords[iWater].z = water_coords[iWater].z - 1 * size_neighborhood + vol_nZ;
            }
            else if ( water_coords[iWater].z > vol_nZ - size_neighborhood ) {
                water_coords[iWater].z = water_coords[iWater].z - vol_nZ + 1 * size_neighborhood;
            }
        }
    }
    else {
#pragma omp parallel for num_threads(local_threads) private(dr, dx, dy, dz, my_rand)
        for ( long iWater = 0; iWater < number_of_waters; iWater++ ) {

            water_coords[iWater].x += my_rand.GetNormalRandomSTD(0.0f, random_sigma);
            water_coords[iWater].y += my_rand.GetNormalRandomSTD(0.0f, random_sigma);
            water_coords[iWater].z += my_rand.GetNormalRandomSTD(0.0f, random_sigma);

            // TODO 2x check that the periodic shifts are doing what they should be.
            // Check boundaries
            if ( water_coords[iWater].x < size_neighborhood + 1 ) {
                water_coords[iWater].x = water_coords[iWater].x - 1 * size_neighborhood + vol_nX;
            }
            else if ( water_coords[iWater].x > vol_nX - size_neighborhood ) {
                water_coords[iWater].x = water_coords[iWater].x - vol_nX + 1 * size_neighborhood;
            }

            // Check boundaries
            if ( water_coords[iWater].y < size_neighborhood + 1 ) {
                water_coords[iWater].y = water_coords[iWater].y - 1 * size_neighborhood + vol_nY;
            }
            else if ( water_coords[iWater].y > vol_nY - size_neighborhood ) {
                water_coords[iWater].y = water_coords[iWater].y - vol_nY + 1 * size_neighborhood;
            }

            // Check boundaries
            if ( water_coords[iWater].z < size_neighborhood + 1 ) {
                water_coords[iWater].z = water_coords[iWater].z - 1 * size_neighborhood + vol_nZ;
            }
            else if ( water_coords[iWater].z > vol_nZ - size_neighborhood ) {
                water_coords[iWater].z = water_coords[iWater].z - vol_nZ + 1 * size_neighborhood;
            }
        }
    }
}

void Water::ReturnPadding(RotationMatrix max_rotation, float in_plane_rotation, int current_nZ, int current_nX, int current_nY, int* padX, int* padY, int* padZ) {

    // If these need to be re-enabled, change RotationMatrix to an AnglesAndShifts object which also retains the original euler angles as member variables.
    //	MyAssertTrue(max_rotation. < 70.01, "maximum tilt angle supported is 70 degrees")
    //

    const bool vector_padding = true;

    if ( vector_padding ) {
        // TODO just put the vector types into the main cisTEM core.
        // We don't care if existing water is rotated out of bounds with respect to the starting box. If vacuum is rotated in bounds, we want to prevent that.

        AtomPos origin;
        AtomPos max_padding;
        AtomPos padding;

        float x_in(0), y_in(0), z_in(0), x_out(0), y_out(0), z_out(0), x_back(0), y_back(0), z_back(0);

        origin.x = (float)current_nX / 2;
        origin.y = (float)current_nY / 2;
        origin.z = (float)current_nZ / 2;

        max_padding.x = 0.0f;
        max_padding.y = 0.0f;
        max_padding.z = 0.0f;

        int corner_counter = 0;
        for ( int k = 0; k < 2; k++ ) {
            for ( int j = 0; j < 2; j++ ) {
                for ( int i = 0; i < 2; i++ ) {

                    // When 0, return positive radius, when 1 return negative radius.
                    x_in = (float)(1 - i) * origin.x - (float)i * origin.x;
                    y_in = (float)(1 - j) * origin.y - (float)j * origin.y;
                    z_in = (float)(1 - k) * origin.z - (float)k * origin.z;

                    max_rotation.RotateCoords(x_in, y_in, z_in, x_out, y_out, z_out);

                    // wxPrintf("Coords before %f %f %f\nRotated %f %f %f\n",x_in, y_in, z_in, x_out, y_out, z_out);
                    // Both x & y must be in bounds for padding to be required
                    if ( x_out >= -origin.x && x_out <= origin.x && y_out >= -origin.y && y_out <= origin.y ) {
                        // Find the smallest value to get us to an edge
                        if ( x_out + origin.x > origin.x - x_out ) {
                            // x is in the + quadrant, so extend it to that edge
                            x_out = origin.x;
                        }
                        else {
                            // t is in the - quadrant, so extend it to that edge
                            x_out = -origin.x;
                        }

                        if ( y_out + origin.y > origin.y - y_out ) {
                            // x is in the + quadrant, so extend it to that edge
                            y_out = origin.y;
                        }
                        else {
                            // t is in the - quadrant, so extend it to that edge
                            y_out = -origin.y;
                        }

                        // Rotate back into the original frame
                        max_rotation.RotateCoords(x_out, y_out, z_out, x_back, y_back, z_back);

                        // wxPrintf("Coords out %f %f %f\nRotated back %f %f %f\n",x_out, y_out, z_out, x_back, y_back, z_back);

                        // Now check to see if the padded vector is larger than our current largest
                        if ( x_back - x_in > max_padding.x ) {
                            max_padding.x = x_back - x_in;
                        }
                        if ( y_back - y_in > max_padding.y ) {
                            max_padding.y = y_back - y_in;
                        }
                        if ( z_back - z_in > max_padding.z ) {
                            max_padding.z = y_back - z_in;
                        }
                    }
                }
            }
        }

        wxPrintf("Original dims = %d %d %d\nPadding = %f %f %f\n", current_nX, current_nY, current_nZ, max_padding.x, max_padding.y, max_padding.z);
        *padX = 2 * (int)ceil(max_padding.x);
        *padY = 2 * (int)ceil(max_padding.y);
        *padZ = (int)ceil(max_padding.z);
    }
    else {

        float v1, v2, v3;
        float t1(1.0f), t2(0.0f), t3(0.0f);

        max_rotation.RotateCoords(t1, t2, t3, v1, v2, v3);
        float max_tilt = fabsf(asinf(v3 / v1));

        float max_ip_ang = 45.0f;

        if ( in_plane_rotation > max_ip_ang + .01 ) {
            wxPrintf("\n\n\t\tWarning, you have requested a tilt-axis rotation of %3.3f degrees, which is greater than the recommended max of %2.2f\n\t\tthis will add a lot of waters\n\n", in_plane_rotation, max_ip_ang);
        }

        // Set the 0 position to be the radius

        *padX = myroundint(0.5f * (float)current_nY * fabsf(sinf(in_plane_rotation * PIf / 180.0f)));
        *padY = myroundint(0.5f * (float)current_nX * fabsf(sinf(in_plane_rotation * PIf / 180.0f)));

        // TODO start here, change to find the four corners of the box under both rotations, first confirming the order with a simple grid and exit
        // Then determine the padding needed in x and y as a function of this. Also bring in the FFT padding here so the FFT size is also padded with water, which
        // will also require more variables to pass info back to the main simulator.

        if ( fabsf(max_tilt) < 1e-1 ) {
            *padZ = 0;
        }
        else {

            float          x0 = 0.5f * ((float)current_nX + (float)*padX);
            float          y0 = 0.0f;
            float          z0 = -(float)current_nZ / 2.0f;
            float          xf, yf, zf;
            RotationMatrix rotmat;
            rotmat.SetToEulerRotation(0.0f, max_tilt, 0.0f);
            rotmat.RotateCoords(x0, y0, z0, xf, yf, zf);

            *padZ = myroundint(fabsf(2.0f * (xf - x0)));
            //    	*padZ = myroundint(1.0f*current_nZ*sinf(max_tilt * (float)PIf / 180.0f));
        }
    }
}
