#include "core_headers.h"

// Align rotationally a (stack) of image(s) against another image. Return the rotation angle that gives the best normalised cross-correlation.
float SpectrumImage::FindRotationalAlignmentBetweenTwoStacksOfImages(Image* other_image, int number_of_images, float search_half_range, float search_step_size, float minimum_radius, float maximum_radius) {
    MyDebugAssertTrue(this[0].is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(this[0].is_in_real_space, "Not in real space");
    MyDebugAssertTrue(this[0].logical_z_dimension == 1, "Meant for images, not volumes");
    MyDebugAssertTrue(other_image[0].is_in_memory, "Memory not allocated - other_image");
    MyDebugAssertTrue(other_image[0].is_in_real_space, "Not in real space - other_image");
    MyDebugAssertTrue(other_image[0].logical_z_dimension == 1, "Meant for images, not volumes - other_image");
    MyDebugAssertTrue(this[0].HasSameDimensionsAs(&other_image[0]), "Images and reference images do not have same dimensions.");

    // Local variables
    const float           minimum_radius_sq           = pow(minimum_radius, 2);
    const float           maximum_radius_sq           = pow(maximum_radius, 2);
    const float           inverse_logical_x_dimension = 1.0 / float(this[0].logical_x_dimension);
    const float           inverse_logical_y_dimension = 1.0 / float(this[0].logical_y_dimension);
    float                 best_cc                     = -std::numeric_limits<float>::max( );
    float                 best_rotation               = -std::numeric_limits<float>::max( );
    float                 current_rotation            = -search_half_range;
    float                 current_rotation_rad;
    EmpiricalDistribution cc_numerator_dist;
    EmpiricalDistribution cc_denom_self_dist;
    EmpiricalDistribution cc_denom_other_dist;
    int                   current_image;
    int                   i, i_logi;
    float                 i_logi_frac, ii_phys;
    int                   j, j_logi;
    float                 j_logi_frac, jj_phys;
    float                 current_interpolated_value;
    long                  address_in_other_image;
    float                 current_cc;

    // Loop over possible rotations
    while ( current_rotation < search_half_range + search_step_size ) {

        current_rotation_rad = current_rotation / 180.0 * PIf;
        cc_numerator_dist.Reset( );
        cc_denom_self_dist.Reset( );
        cc_denom_other_dist.Reset( );
        // Loop over the array of images
        for ( current_image = 0; current_image < number_of_images; current_image++ ) {
            // Loop over the other (reference) image
            address_in_other_image = 0;
            for ( j = 0; j < other_image[0].logical_y_dimension; j++ ) {
                j_logi      = j - other_image[0].physical_address_of_box_center_y;
                j_logi_frac = pow(j_logi * inverse_logical_y_dimension, 2);
                for ( i = 0; i < other_image[0].logical_x_dimension; i++ ) {
                    i_logi      = i - other_image[0].physical_address_of_box_center_x;
                    i_logi_frac = pow(i_logi * inverse_logical_x_dimension, 2) + j_logi_frac;

                    if ( i_logi_frac >= minimum_radius_sq && i_logi_frac <= maximum_radius_sq ) {
                        // We do ccw rotation to go from other_image (reference) to self (input image)
                        ii_phys = i_logi * cos(current_rotation_rad) - j_logi * sin(current_rotation_rad) + this[0].physical_address_of_box_center_x;
                        jj_phys = i_logi * sin(current_rotation_rad) + j_logi * cos(current_rotation_rad) + this[0].physical_address_of_box_center_y;
                        //
                        if ( int(ii_phys) > 0 && int(ii_phys) + 1 < this[0].logical_x_dimension && int(jj_phys) > 0 && int(jj_phys) + 1 < this[0].logical_y_dimension ) // potential optimization: we have to compute the floor and ceiling in the interpolation routine. Is it not worth doing the bounds checking in the interpolation routine somehow?
                        {
                            this[0].GetRealValueByLinearInterpolationNoBoundsCheckImage(ii_phys, jj_phys, current_interpolated_value);
                            //MyDebugPrint("%g %g\n",current_interpolated_value,other_image[0].real_values[address_in_other_image]);
                            cc_numerator_dist.AddSampleValue(current_interpolated_value * other_image[current_image].real_values[address_in_other_image]);
                            cc_denom_other_dist.AddSampleValue(pow(other_image[0].real_values[address_in_other_image], 2)); // potential optimization: since other_image is not being rotated, we should only need to compute this quantity once, not for every potential rotation
                            cc_denom_self_dist.AddSampleValue(pow(current_interpolated_value, 2));
                        }
                    }
                    address_in_other_image++;
                } // i
                address_in_other_image += other_image[0].padding_jump_value;
            } // end of loop over other (reference) image
        } // end of loop over array of images

        current_cc = cc_numerator_dist.GetSampleSum( ) / sqrt(cc_denom_other_dist.GetSampleSum( ) * cc_denom_self_dist.GetSampleSum( ));

        if ( current_cc > best_cc ) {
            best_cc       = current_cc;
            best_rotation = current_rotation;
        }

        // Increment the rotation
        current_rotation += search_step_size;

    } // end of loop over rotations

    return best_rotation;
}

// This function assumes the image is in real space and generates the power spectrum
void SpectrumImage::GeneratePowerspectrum(CTF ctf_to_apply) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "image not in real space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Volumes not supported");

    int         i, j;
    float       j_logi, j_logi_sq, i_logi, i_logi_sq;
    float       current_spatial_frequency_squared;
    long        address                     = 0;
    const float inverse_logical_x_dimension = 1.0 / float(logical_x_dimension);
    const float inverse_logical_y_dimension = 1.0 / float(logical_y_dimension);
    float       current_azimuth;
    float       current_ctf_value;

    for ( j = 0; j < logical_y_dimension; j++ ) {
        address   = j * (padding_jump_value + 2 * physical_address_of_box_center_x);
        j_logi    = float(j - physical_address_of_box_center_y) * inverse_logical_y_dimension;
        j_logi_sq = powf(j_logi, 2);
        for ( i = 0; i < logical_x_dimension; i++ ) {
            i_logi    = float(i - physical_address_of_box_center_x) * inverse_logical_x_dimension;
            i_logi_sq = powf(i_logi, 2);

            // Where are we?
            current_spatial_frequency_squared = j_logi_sq + i_logi_sq;

            current_azimuth   = atan2f(j_logi, i_logi);
            current_ctf_value = ctf_to_apply.EvaluatePowerspectrumWithThickness(current_spatial_frequency_squared, current_azimuth);
            // Apply powespectrum
            real_values[address + i] = current_ctf_value;
        }
    }
}
