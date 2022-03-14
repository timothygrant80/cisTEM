// BEGIN_FOR_STAND_ALONE_CTFFIND
#include "core_headers.h"

using namespace cistem;

wxMutex Image::s_mutexProtectingFFTW;
double  BeamTiltScoreFunctionForSimplex(void* pt2Object, double values[]);

void Image::SetupInitialValues() {
    logical_x_dimension = 0;
    logical_y_dimension = 0;
    logical_z_dimension = 0;

    is_in_real_space         = true;
    object_is_centred_in_box = true;

    physical_upper_bound_complex_x = 0;
    physical_upper_bound_complex_y = 0;
    physical_upper_bound_complex_z = 0;

    physical_address_of_box_center_x = 0;
    physical_address_of_box_center_y = 0;
    physical_address_of_box_center_z = 0;

    // physical_index_of_first_negative_frequency_x = 0;
    physical_index_of_first_negative_frequency_y = 0;
    physical_index_of_first_negative_frequency_z = 0;

    fourier_voxel_size_x = 0.0;
    fourier_voxel_size_y = 0.0;
    fourier_voxel_size_z = 0.0;

    logical_upper_bound_complex_x = 0;
    logical_upper_bound_complex_y = 0;
    logical_upper_bound_complex_z = 0;

    logical_lower_bound_complex_x = 0;
    logical_lower_bound_complex_y = 0;
    logical_lower_bound_complex_z = 0;

    logical_upper_bound_real_x = 0;
    logical_upper_bound_real_y = 0;
    logical_upper_bound_real_z = 0;

    logical_lower_bound_real_x = 0;
    logical_lower_bound_real_y = 0;
    logical_lower_bound_real_z = 0;

    insert_into_which_reconstruction = 0;
    real_values                      = NULL;
    complex_values                   = NULL;

    is_in_memory          = false;
    real_memory_allocated = 0;

    plan_fwd = NULL;
    plan_bwd = NULL;

    planned = false;

    padding_jump_value                     = 0;
    image_memory_should_not_be_deallocated = false;
}

Image::Image() { SetupInitialValues(); }

Image::Image(const Image& other_image) // copy constructor
{

    SetupInitialValues();
    // MyDebugPrint("Warning: copying an image object");
    *this = other_image;
}

Image::~Image() { Deallocate(); }

int Image::ReturnSmallestLogicalDimension() {
    if (logical_z_dimension == 1) {
        return std::min(logical_x_dimension, logical_y_dimension);
    }
    else {
        int temp_int;
        temp_int = std::min(logical_x_dimension, logical_y_dimension);
        return std::min(temp_int, logical_z_dimension);
    }
}

int Image::ReturnLargestLogicalDimension() {
    if (logical_z_dimension == 1) {
        return std::max(logical_x_dimension, logical_y_dimension);
    }
    else {
        int temp_int;
        temp_int = std::max(logical_x_dimension, logical_y_dimension);
        return std::max(temp_int, logical_z_dimension);
    }
}
// END_FOR_STAND_ALONE_CTFFIND

void Image::SampleFFT(Image& sampled_image, int sample_rate) {
    MyDebugAssertTrue(sample_rate > 0, "Invalid sample rate");
    MyDebugAssertTrue(!is_in_real_space, "Not in Fourier space");
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(sampled_image.is_in_memory, "Memory of sampled image not allocated");
    MyDebugAssertTrue(logical_x_dimension == sample_rate * sampled_image.logical_x_dimension,
                      "X dimensions incompatible");
    MyDebugAssertTrue(logical_y_dimension == sample_rate * sampled_image.logical_y_dimension,
                      "Y dimensions incompatible");
    MyDebugAssertTrue(logical_z_dimension == 1 ||
                          logical_z_dimension == sample_rate * sampled_image.logical_z_dimension,
                      "Z dimensions incompatible");
    MyDebugAssertTrue(IsEven(logical_x_dimension) && IsEven(logical_y_dimension) &&
                          (IsEven(logical_z_dimension) || logical_z_dimension == 1),
                      "Only works on even dimensions");

    int  i;
    int  j;
    int  k;
    long address_in    = 0;
    long address_out   = 0;
    long sample_rate_y = (logical_x_dimension / 2) * (sample_rate - 1);
    long sample_rate_z = (logical_x_dimension / 2) * logical_y_dimension * (sample_rate - 1);

    int  x_in;
    int  y_in;
    long address;

    sampled_image.is_in_real_space = false;
    //	sampled_image.SetToConstant(100000.0);

    for (k = 0; k <= physical_upper_bound_complex_z; k += sample_rate) {
        for (j = 0; j <= physical_upper_bound_complex_y; j += sample_rate) {
            for (i = 0; i <= physical_upper_bound_complex_x; i += sample_rate) {
                sampled_image.complex_values[address_out] = complex_values[address_in];
                //				x_in = ReturnFourierLogicalCoordGivenPhysicalCoord_X(i);
                //				y_in = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j);
                //				address = ReturnFourier1DAddressFromLogicalCoord(x_in, y_in, 0);
                //				address = sampled_image.ReturnFourier1DAddressFromLogicalCoord(x_in / sample_rate, y_in /
                //sample_rate, 0); 				wxPrintf("address_in, address_out, address = %li, %li, %li, x,y_in = %i, %i, value =
                //%g\n", address_in, address_out, address, x_in, y_in, cabsf(complex_values[address_in]));
                address_in += sample_rate;
                address_out++;
            }
            address_in += sample_rate_y;
        }
        address_in += sample_rate_z;
    }
    sampled_image.object_is_centred_in_box = object_is_centred_in_box;
}

float Image::ReturnSumOfRealValues() {
    MyDebugAssertTrue(is_in_memory, "memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "not in real space");
    double sum;
    int    i, j, k;
    long   address;
    sum     = 0.0;
    address = 0;
    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                sum += real_values[address];
                address++;
            }
            address += padding_jump_value;
        }
    }
    return float(sum);
}

float Image::ReturnSumOfSquares(float wanted_mask_radius, float wanted_center_x, float wanted_center_y,
                                float wanted_center_z, bool invert_mask) {
    //	result is too big
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    //	MyDebugAssertTrue(is_in_real_space, "Not in real space");

    int  i;
    int  j;
    int  k;
    int  jj;
    int  kk;
    long address          = 0;
    long number_of_pixels = 0;
    bool x_is_even        = IsEven(logical_x_dimension);
    //	bool y_is_even = IsEven(logical_y_dimension);
    //	bool z_is_even = IsEven(logical_z_dimension);
    float  x;
    float  y;
    float  z;
    float  distance_from_center_squared;
    float  mask_radius_squared;
    float  edge;
    float  center_x;
    float  center_y;
    float  center_z;
    double sum = 0.0;

    float weight = 0.0;

    if (is_in_real_space) {
        if (wanted_mask_radius > 0.0) {

            if (wanted_center_x == 0.0 && wanted_center_y == 0.0 && wanted_center_z == 0.0) {
                center_x = physical_address_of_box_center_x;
                center_y = physical_address_of_box_center_y;
                center_z = physical_address_of_box_center_z;
            }
            else {
                center_x = wanted_center_x;
                center_y = wanted_center_y;
                center_z = wanted_center_z;
            }
            mask_radius_squared = powf(wanted_mask_radius, 2);
            for (k = 0; k < logical_z_dimension; k++) {
                z = powf(k - center_z, 2);

                for (j = 0; j < logical_y_dimension; j++) {
                    y = powf(j - center_y, 2);

                    for (i = 0; i < logical_x_dimension; i++) {
                        x = powf(i - center_x, 2);

                        distance_from_center_squared = x + y + z;

                        if (invert_mask) {
                            if (distance_from_center_squared > mask_radius_squared) {
                                sum += pow(real_values[address], 2);
                                number_of_pixels++;
                            }
                        }
                        else {
                            if (distance_from_center_squared <= mask_radius_squared) {
                                sum += pow(real_values[address], 2);
                                number_of_pixels++;
                            }
                        }
                        address++;
                    }
                    address += padding_jump_value;
                }
            }
            if (number_of_pixels > 0) {
                return sum / number_of_pixels;
            }
            else {
                return 0.0;
            }
        }
        else {
            for (k = 0; k < logical_z_dimension; k++) {
                for (j = 0; j < logical_y_dimension; j++) {
                    for (i = 0; i < logical_x_dimension; i++) {
                        sum += pow(real_values[address], 2);
                        address++;
                    }
                    address += padding_jump_value;
                }
            }
            return sum / logical_x_dimension / logical_y_dimension / logical_z_dimension;
        }
    }
    else {
        for (k = 0; k <= physical_upper_bound_complex_z; k++) {
            kk = ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k);
            for (j = 0; j <= physical_upper_bound_complex_y; j++) {
                jj = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j);
                for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                    if ((i == 0 || (i == logical_upper_bound_complex_x && x_is_even)) &&
                        (jj == 0 || (jj == logical_lower_bound_complex_y && x_is_even)) &&
                        (kk == 0 || (kk == logical_lower_bound_complex_z && x_is_even)))
                        sum += pow(abs(complex_values[address]), 2) * 0.5;
                    else if ((i == 0 || (i == logical_upper_bound_complex_x && x_is_even)) && logical_z_dimension != 1)
                        sum += pow(abs(complex_values[address]), 2) * 0.25;
                    else if ((i != 0 && (i != logical_upper_bound_complex_x || !x_is_even)) || (jj >= 0 && kk >= 0))
                        sum += pow(abs(complex_values[address]), 2);
                    address++;
                }
            }
        }
        return 2.0 * sum;
    }
}

float Image::ReturnSigmaNoise() {
    MyDebugAssertTrue(is_in_memory, "particle_image memory not allocated");

    float  variance;
    Image* subsampled1 = new Image;
    subsampled1->Allocate(logical_x_dimension, logical_y_dimension, false);
    Image* subsampled2 = new Image;
    subsampled1->Allocate(logical_x_dimension, logical_y_dimension, false);

    SubSampleMask(subsampled1, subsampled2);
    subsampled1->ForwardFFT();
    subsampled2->ForwardFFT();
    subsampled1->Resize(logical_x_dimension / 2, logical_y_dimension / 2, 1);
    subsampled2->Resize(logical_x_dimension / 2, logical_y_dimension / 2, 1);
    subsampled1->BackwardFFT();
    subsampled2->BackwardFFT();
    subsampled1->SubtractImage(subsampled2);
    variance = subsampled1->ReturnSumOfSquares();

    delete subsampled1;
    delete subsampled2;

    // Divide variance by 2 because it relates to the difference of two noise images.
    // Multiply variance by 2 because of 2x2 pixel binning (only two pixels are non-zero)?
    // Multiply by 4 for FFT scaling due to different sizes.
    return sqrtf(4.0 * variance);
}

float Image::ReturnSigmaNoise(Image& matching_projection, float mask_radius) {
    float correlation_coefficient;
    float alpha;
    float sigma;
    float pixel_variance_image;
    float pixel_variance_projection;

    correlation_coefficient   = ReturnCorrelationCoefficientUnnormalized(matching_projection, mask_radius);
    pixel_variance_image      = ReturnVarianceOfRealValues(mask_radius);
    pixel_variance_projection = matching_projection.ReturnVarianceOfRealValues(mask_radius);
    alpha                     = fabsf(correlation_coefficient / pixel_variance_projection);
    //	sigma = sqrtf(pixel_variance_image - powf(alpha,2) * pixel_variance_projection);
    //	wxPrintf("var_input = %f, var_output = %g, alpha = %f, sigma_signal = %f, sigma_noise = %f\n",
    //pixel_variance_image, pixel_variance_projection, alpha, sqrtf(pixel_variance_image - powf(sigma,2)), sigma);

    return sqrtf(pixel_variance_image - powf(alpha, 2) * pixel_variance_projection);
}

float Image::ReturnImageScale(Image& matching_projection, float mask_radius) {
    float correlation_coefficient;
    //	float pixel_variance_image;
    float pixel_variance_projection;

    correlation_coefficient = ReturnCorrelationCoefficientUnnormalized(matching_projection, mask_radius);
    //	pixel_variance_image = ReturnVarianceOfRealValues(mask_radius);
    pixel_variance_projection = matching_projection.ReturnVarianceOfRealValues(mask_radius);
    //	pixel_variance_projection = matching_projection.ReturnSumOfSquares(mask_radius);

    return fabsf(correlation_coefficient / pixel_variance_projection);
}
// Returns the correlation coefficient of the floated images
float Image::ReturnCorrelationCoefficientUnnormalized(Image& other_image, float wanted_mask_radius) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(other_image.is_in_memory, "Other image memory not allocated");
    MyDebugAssertTrue(other_image.is_in_real_space == is_in_real_space, "Images not in the same space");

    int  i;
    int  j;
    int  k;
    long number_of_pixels = 0;

    float x;
    float y;
    float z;

    long pixel_counter = 0;

    float  distance_from_center_squared;
    float  mask_radius_squared;
    float  edge;
    double average1    = 0.0;
    double average2    = 0.0;
    double cross_terms = 0.0;

    if (wanted_mask_radius > 0.0) {
        mask_radius_squared = powf(wanted_mask_radius, 2);
        for (k = 0; k < logical_z_dimension; k++) {
            z = powf(k - physical_address_of_box_center_z, 2);

            for (j = 0; j < logical_y_dimension; j++) {
                y = powf(j - physical_address_of_box_center_y, 2);

                for (i = 0; i < logical_x_dimension; i++) {
                    x = powf(i - physical_address_of_box_center_x, 2);

                    distance_from_center_squared = x + y + z;

                    if (distance_from_center_squared <= mask_radius_squared) {
                        cross_terms += real_values[pixel_counter] * other_image.real_values[pixel_counter];
                        average1 += real_values[pixel_counter];
                        average2 += other_image.real_values[pixel_counter];
                        number_of_pixels++;
                    }
                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }
    else {
        for (k = 0; k < logical_z_dimension; k++) {
            for (j = 0; j < logical_y_dimension; j++) {
                for (i = 0; i < logical_x_dimension; i++) {
                    cross_terms += real_values[pixel_counter] * other_image.real_values[pixel_counter];
                    average1 += real_values[pixel_counter];
                    average2 += other_image.real_values[pixel_counter];
                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
        number_of_pixels = long(logical_x_dimension) * long(logical_y_dimension) * long(logical_z_dimension);
    }

    return float(cross_terms / number_of_pixels - average1 / number_of_pixels * average2 / number_of_pixels);
}

float Image::ReturnPixelWiseProduct(Image& other_image) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(other_image.is_in_memory, "Other image memory not allocated");
    MyDebugAssertTrue(other_image.is_in_real_space == is_in_real_space, "Images not in the same space");

    int    i;
    int    j;
    int    k;
    long   pixel_counter = 0;
    double cross_terms   = 0.0;

    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                cross_terms += real_values[pixel_counter] * other_image.real_values[pixel_counter];
                pixel_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }

    return float(cross_terms / number_of_real_space_pixels);
}

// Frealign weighted correlation coefficient
// float GetWeightedCorrelationWithImage(Image &projection_image, float low_limit, float high_limit, int &bins)
float Image::GetWeightedCorrelationWithImage(Image& projection_image, int* bins, float signed_CC_limit) {
    MyDebugAssertTrue(is_in_memory, "Image memory not allocated");
    MyDebugAssertTrue(projection_image.is_in_memory, "projection_image memory not allocated");
    MyDebugAssertTrue(!is_in_real_space, "Image not in Fourier space");
    MyDebugAssertTrue(!projection_image.is_in_real_space, "projection_image not in Fourier space");
    //	MyDebugAssertTrue(! projection_image.object_is_centred_in_box, "projection_image quadrants have not been
    //swapped");
    MyDebugAssertTrue(HasSameDimensionsAs(&projection_image), "Images do not have the same dimensions");
    MyDebugAssertTrue(bins != NULL, , "bin_index not calculated");
    MyDebugAssertFalse(HasNan(), "Image has one or more NaN pixels");
    MyDebugAssertFalse(projection_image.HasNan(), "Projection has one or more NaN pixels");

    int i;
    //	int j;
    //	int k;
    int bin;
    int bin_limit_signed;

    //	float x;
    //	float y;
    //	float z;
    //	float frequency;
    //	float frequency_squared;
    float score;
    float sum1;
    float sum2;
    float sum3;

    //	float low_limit2 = powf(low_limit,2);
    //	float high_limit2 = fminf(powf(high_limit,2),0.25);

    int number_of_bins  = ReturnLargestLogicalDimension() / 2 + 1;
    int number_of_bins2 = 2 * (number_of_bins - 1);

    double* sum_a       = new double[number_of_bins];
    double* sum_b       = new double[number_of_bins];
    double* cross_terms = new double[number_of_bins];

    long pixel_counter = 0;

    std::complex<float> temp_c;

    ZeroDoubleArray(sum_a, number_of_bins);
    ZeroDoubleArray(sum_b, number_of_bins);
    ZeroDoubleArray(cross_terms, number_of_bins);

    /*	for (k = 0; k <= physical_upper_bound_complex_z; k++)
        {
            z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

            for (j = 0; j <= physical_upper_bound_complex_y; j++)
            {
                y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

                for (i = 0; i <= physical_upper_bound_complex_x; i++)
                {
                    x = powf(i * fourier_voxel_size_x, 2);
                    frequency_squared = x + y + z;

                    if (frequency_squared >= low_limit2 && frequency_squared <= high_limit2)
                    {
                        bin = int(sqrtf(frequency_squared) * number_of_bins2);

                        temp_c = real(complex_values[pixel_counter] *
       conj(projection_image.complex_values[pixel_counter])); if (temp_c != 0.0)
                        {
                            sum_a[bin] += real(complex_values[pixel_counter] * conj(complex_values[pixel_counter]));
                            sum_b[bin] += real(projection_image.complex_values[pixel_counter] *
       conj(projection_image.complex_values[pixel_counter])); cross_terms[bin] += real(temp_c);
                        }
                    }
                    pixel_counter++;
                }
            }
        }
    */
    for (pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++) {
        bin = bins[pixel_counter];
        if (bin >= 1) {
            temp_c = complex_values[pixel_counter] * conj(projection_image.complex_values[pixel_counter]);
            if (temp_c != 0.0f) {
                sum_a[bin] += real(complex_values[pixel_counter] * conj(complex_values[pixel_counter]));
                sum_b[bin] += real(projection_image.complex_values[pixel_counter] *
                                   conj(projection_image.complex_values[pixel_counter]));
                // Since projection_image has been whitened, the sum in a resolution zone is simply the number of terms
                // that are added Not true in new code where projection could be non-white
                //				sum_b[bin] ++;
                cross_terms[bin] += real(temp_c);
            }
        }
    }

    sum1             = 0.0;
    sum2             = 0.0;
    sum3             = 0.0;
    bin_limit_signed = signed_CC_limit * number_of_bins2;
    // Exclude last resolution bin since it may contain some incompletely calculated terms
    for (i = 1; i < number_of_bins - 1; i++) {
        if (sum_b[i] != 0.0) {
            if (i <= bin_limit_signed) {
                sum3 += cross_terms[i];
                //				wxPrintf("i_s = %i, sum_a = %g, sum_b = %i, cross = %g\n", i, sum_a[i], sum_b[i],
                //cross_terms[i]); 				r = cross_terms[i] / sqrtf(sum_b[i]);
            }
            else {
                sum3 += fabsf(cross_terms[i]);
                //				wxPrintf("i_u = %i, sum_a = %g, sum_b = %i, cross = %g\n", i, sum_a[i], sum_b[i],
                //cross_terms[i]); 				r = fabsf(cross_terms[i] / sqrtf(sum_b[i]));
            }
            sum1 += sum_a[i];
            sum2 += sum_b[i];
            //			sum1 += 1.0;
            //			sum2 += sum_a[i];
            //			sum3 += r;
        }
    }
    sum1 *= sum2;
    if (sum1 != 0.0)
        sum3 /= sqrtf(sum1);

    delete[] sum_a;
    delete[] sum_b;
    delete[] cross_terms;

    return sum3;
}

void Image::PhaseFlipPixelWise(Image& phase_image) {
    MyDebugAssertTrue(is_in_memory, "Image memory not allocated");
    MyDebugAssertTrue(phase_image.is_in_memory, "Other image memory not allocated");
    MyDebugAssertTrue(is_in_real_space == phase_image.is_in_real_space, "Image and phase image not in same space");

    int  i;
    long pixel_counter;

    if (is_in_real_space) {
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
            if (phase_image.real_values[pixel_counter] < 0.0)
                real_values[pixel_counter] = -real_values[pixel_counter];
        }
    }
    else {
        float* real_a;
        float* real_b;
        float* real_r;
        real_a = real_values;
        real_b = real_values + 1;
        real_r = phase_image.real_values;

        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter += 2) {
            if (real_r[pixel_counter] < 0.0) {
                real_a[pixel_counter] = -real_a[pixel_counter];
                real_b[pixel_counter] = -real_b[pixel_counter];
            }
        }
    }
}

void Image::MultiplyPixelWiseReal(Image& other_image, bool absolute) {
    MyDebugAssertTrue(!is_in_real_space, "Image is in real space");
    MyDebugAssertTrue(!other_image.is_in_real_space, "Other image is in real space");
    MyDebugAssertTrue(is_in_memory, "Image memory not allocated");
    MyDebugAssertTrue(other_image.is_in_memory, "Other image memory not allocated");
    MyDebugAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");

    int  i;
    long pixel_counter;

    float* real_a;
    float* real_b;
    float* real_r;
    real_a = real_values;
    real_b = real_values + 1;
    real_r = other_image.real_values;

    if (absolute) {
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter += 2) {
            real_a[pixel_counter] *= fabsf(real_r[pixel_counter]);
        };
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter += 2) {
            real_b[pixel_counter] *= fabsf(real_r[pixel_counter]);
        };
    }
    else {
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter += 2) {
            real_a[pixel_counter] *= real_r[pixel_counter];
        };
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter += 2) {
            real_b[pixel_counter] *= real_r[pixel_counter];
        };
    }
}

void Image::ConjugateMultiplyPixelWise(Image& other_image) {
    MyDebugAssertTrue(is_in_memory, "Image memory not allocated");
    MyDebugAssertTrue(other_image.is_in_memory, "Other image memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image must be in Fourier space");
    MyDebugAssertFalse(other_image.is_in_real_space, "Other image must be in Fourier space");
    MyDebugAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");

    long pixel_counter;

#ifdef MKL
    // Use the MKL - not sure whether this can work in place
    vmcMulByConj(real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(complex_values),
                 reinterpret_cast<MKL_Complex8*>(other_image.complex_values),
                 reinterpret_cast<MKL_Complex8*>(complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    for (pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++) {
        complex_values[pixel_counter] *= conj(other_image.complex_values[pixel_counter]);
    }
#endif
}

void Image::ComputeFSCVectorized(Image* other_image, Image* work_this_image_squared, Image* work_other_image_squared,
                                 Image* work_cross_product_image, int number_of_shells, int* shell_number,
                                 float* computed_fsc, double* work_sum_of_squares, double* work_sum_of_other_squares,
                                 double* work_sum_of_cross_products) {
    MyDebugAssertTrue(is_in_memory, "Image memory not allocated");
    MyDebugAssertTrue(other_image->is_in_memory, "Other image memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image must be in Fourier space");
    MyDebugAssertFalse(other_image->is_in_real_space, "Other image must be in Fourier space");
    MyDebugAssertTrue(HasSameDimensionsAs(other_image), "Images have different dimensions");
    MyDebugAssertTrue(HasSameDimensionsAs(work_this_image_squared), "Images have different dimensions");
    MyDebugAssertTrue(HasSameDimensionsAs(work_other_image_squared), "Images have different dimensions");
    MyDebugAssertTrue(HasSameDimensionsAs(work_cross_product_image), "Images have different dimensions");

    /*
     * Prepare working memory
     */
    for (long pixel_counter = 0; pixel_counter < number_of_real_space_pixels; pixel_counter++) {
        work_this_image_squared->real_values[pixel_counter] = real_values[pixel_counter];
    }
    for (long pixel_counter = 0; pixel_counter < number_of_real_space_pixels; pixel_counter++) {
        work_cross_product_image->real_values[pixel_counter] = real_values[pixel_counter];
    }
    for (long pixel_counter = 0; pixel_counter < number_of_real_space_pixels; pixel_counter++) {
        work_other_image_squared->real_values[pixel_counter] = other_image->real_values[pixel_counter];
    }
    for (int shell_counter = 0; shell_counter < number_of_shells; shell_counter++) {
        computed_fsc[shell_counter]               = 0.0;
        work_sum_of_squares[shell_counter]        = 0.0;
        work_sum_of_other_squares[shell_counter]  = 0.0;
        work_sum_of_cross_products[shell_counter] = 0.0;
    }
    work_this_image_squared->is_in_real_space  = false;
    work_cross_product_image->is_in_real_space = false;
    work_other_image_squared->is_in_real_space = false;

    /*
     * Do the conjugate multiplications
     */
    work_this_image_squared->ConjugateMultiplyPixelWise(*work_this_image_squared);
    work_cross_product_image->ConjugateMultiplyPixelWise(*work_other_image_squared);
    work_other_image_squared->ConjugateMultiplyPixelWise(*work_other_image_squared);

    /*
     * Accumulate intermediate results
     */
    for (long pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++) {
        work_sum_of_other_squares[shell_number[pixel_counter]] +=
            real(work_other_image_squared->complex_values[pixel_counter]);
        work_sum_of_squares[shell_number[pixel_counter]] +=
            real(work_this_image_squared->complex_values[pixel_counter]);
        work_sum_of_cross_products[shell_number[pixel_counter]] +=
            real(work_cross_product_image->complex_values[pixel_counter]);
    }

    /*
     * Compute the FSC
     */
    float denominator;
    for (int shell_counter = 0; shell_counter < number_of_shells; shell_counter++) {
        // check whether the product is positive before doing sqrtf?
        denominator = sqrtf(work_sum_of_squares[shell_counter] * work_sum_of_other_squares[shell_counter]);
        if (denominator == 0.0) {
            computed_fsc[shell_counter] = 0.0;
        }
        else {
            computed_fsc[shell_counter] = work_sum_of_cross_products[shell_counter] / denominator;
        }
    }
}

void Image::ComputeFSC(Image* other_image, int number_of_shells, int* shell_number, float* computed_fsc,
                       double* work_sum_of_squares, double* work_sum_of_other_squares,
                       double* work_sum_of_cross_products) {
    MyDebugAssertTrue(is_in_memory, "Image memory not allocated");
    MyDebugAssertTrue(other_image->is_in_memory, "Other image memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image must be in Fourier space");
    MyDebugAssertFalse(other_image->is_in_real_space, "Other image must be in Fourier space");
    MyDebugAssertTrue(HasSameDimensionsAs(other_image), "Images have different dimensions");

    /*
     * Prepare working memory
     */
    for (int shell_counter = 0; shell_counter < number_of_shells; shell_counter++) {
        computed_fsc[shell_counter]               = 0.0;
        work_sum_of_squares[shell_counter]        = 0.0;
        work_sum_of_other_squares[shell_counter]  = 0.0;
        work_sum_of_cross_products[shell_counter] = 0.0;
    }

    /*
     * Accumulate intermediate results
     */
    for (long pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++) {
        work_sum_of_other_squares[shell_number[pixel_counter]] +=
            real(other_image->complex_values[pixel_counter] * conj(other_image->complex_values[pixel_counter]));
        work_sum_of_squares[shell_number[pixel_counter]] +=
            real(complex_values[pixel_counter] * conj(complex_values[pixel_counter]));
        work_sum_of_cross_products[shell_number[pixel_counter]] +=
            real(complex_values[pixel_counter] * conj(other_image->complex_values[pixel_counter]));
    }

    /*
     * Compute the FSC
     */
    float denominator;
    for (int shell_counter = 0; shell_counter < number_of_shells; shell_counter++) {
        // check whether the product is positive before doing sqrtf?
        denominator = sqrtf(work_sum_of_squares[shell_counter] * work_sum_of_other_squares[shell_counter]);
        if (denominator == 0.0) {
            computed_fsc[shell_counter] = 0.0;
        }
        else {
            computed_fsc[shell_counter] = work_sum_of_cross_products[shell_counter] / denominator;
        }
    }
}

// BEGIN_FOR_STAND_ALONE_CTFFIND

void Image::MultiplyPixelWise(Image& other_image) {
    MyDebugAssertTrue(is_in_memory, "Image memory not allocated");
    MyDebugAssertTrue(other_image.is_in_memory, "Other image memory not allocated");

    int  i;
    long pixel_counter;

    if (is_in_real_space) {
        MyDebugAssertTrue(other_image.is_in_real_space, "Other image needs to be in real space");
        MyDebugAssertTrue(HasSameDimensionsAs(&other_image), "Images do not have same dimensions");
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
            real_values[pixel_counter] *= other_image.real_values[pixel_counter];
        }
    }
    else {
        if (other_image.is_in_real_space) {
            /*
             * This image is in Fourier space, the other image in is in real space.
             * They better be the expected dimensions!
             */
            MyDebugAssertTrue(other_image.logical_x_dimension == logical_x_dimension / 2 + 1 &&
                                  other_image.logical_y_dimension == logical_y_dimension &&
                                  other_image.logical_z_dimension == logical_z_dimension,
                              "This image is in Fourier space (dimensions = %i,%i,%i), the other is in real space "
                              "(dimensions = %i,%i,%i), but it does not have expected dimensions",
                              logical_x_dimension, logical_y_dimension, logical_z_dimension,
                              other_image.logical_x_dimension, other_image.logical_y_dimension,
                              other_image.logical_z_dimension);
            long pixel_counter_this, pixel_counter_other;
            int  i_other, j_other, k_other;
            pixel_counter_this  = 0;
            pixel_counter_other = 0;
            for (k_other = 0; k_other < other_image.logical_z_dimension; k_other++) {
                for (j_other = 0; j_other < other_image.logical_y_dimension; j_other++) {
                    for (i_other = 0; i_other < other_image.logical_x_dimension; i_other++) {
                        complex_values[pixel_counter_this] *= other_image.real_values[pixel_counter_other];
                        // complex_values[pixel_counter_this] *= 0.5;

                        pixel_counter_other++;
                        pixel_counter_this++;
                    }
                    pixel_counter_other += other_image.padding_jump_value;
                }
            }
        }
        else {
            // Both images are in Fourier space
            MyDebugAssertTrue(HasSameDimensionsAs(&other_image), "Images do not have same dimensions");
            // TODO: add MKL implementation (see EulerSearch::Run for a similar example)
            for (pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++) {
                complex_values[pixel_counter] *= other_image.complex_values[pixel_counter];
            }
        }
    }
}

// END_FOR_STAND_ALONE_CTFFIND

void Image::DividePixelWise(Image& other_image) {
    MyDebugAssertTrue(is_in_memory, "Image memory not allocated");
    MyDebugAssertTrue(other_image.is_in_memory, "Other image memory not allocated");
    MyDebugAssertTrue(is_in_real_space == other_image.is_in_real_space, "Both images need to be in same space");

    int  i;
    long pixel_counter;

    if (is_in_real_space) {
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
            real_values[pixel_counter] /= other_image.real_values[pixel_counter];
        }
    }
    else {
        // TODO: add MKL implementation (see EulerSearch::Run for a similar example)
        for (pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++) {
            complex_values[pixel_counter] /= other_image.complex_values[pixel_counter];
        }
    }
}

void Image::AddGaussianNoise(float wanted_sigma_value, RandomNumberGenerator* provided_generator) {
    MyDebugAssertTrue(is_in_real_space == true, "Image must be in real space");

    RandomNumberGenerator* used_generator;

    if (provided_generator == NULL)
        used_generator = &global_random_number_generator;
    else
        used_generator = provided_generator;

    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] += wanted_sigma_value * used_generator->GetNormalRandom();
    }
}

void Image::AddNoise(NoiseType wanted_noise_type, float noise_param_1, float noise_param_2) {
    MyDebugAssertTrue(is_in_real_space == true, "Image must be in real space");
    // These could go down in the switch, but I think it is cleaner to keep DebugAsserts at the top of the method like
    // elsewhere in cisTEM.
    MyDebugAssertTrue(wanted_noise_type == POISSON ? noise_param_1 > 0 : true,
                      "Mean of a Poisson distribution must be positive");
    MyDebugAssertTrue(wanted_noise_type == EXPONENTIAL ? noise_param_1 > 0 : true,
                      "Mean of an Exponential distribution must be positive");
    MyDebugAssertTrue(wanted_noise_type == GAMMA ? (noise_param_1 > 0 && noise_param_2 > 0) : true,
                      "alpha and beta of a Gamma distribution must be positive");

    RandomNumberGenerator my_rng(PIf);

    switch (wanted_noise_type) {
        case GAUSSIAN:
            for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
                real_values[pixel_counter] += my_rng.GetNormalRandomSTD(noise_param_1, noise_param_2);
            }
            break;
        case POISSON:
            for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
                real_values[pixel_counter] += my_rng.GetPoissonRandomSTD(noise_param_1);
            }
            break;
        case UNIFORM:
            for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
                real_values[pixel_counter] += my_rng.GetUniformRandomSTD(noise_param_1, noise_param_2);
            }
            break;
        case EXPONENTIAL:
            for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
                real_values[pixel_counter] += my_rng.GetExponentialRandomSTD(noise_param_1);
            }
            break;
        case GAMMA:
            for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
                real_values[pixel_counter] += my_rng.GetGammaRandomSTD(noise_param_1, noise_param_2);
            }
            break;
        default:
            MyPrintWithDetails("Error: Unknown noise type %i\n", wanted_noise_type);
            DEBUG_ABORT;
            break; // shouldn't be needed with the DEBUG_ABORT, but just in case that breaks for whatever reason.
    }
}

long Image::ZeroFloat(float wanted_mask_radius, bool outside) {
    MyDebugAssertTrue(is_in_real_space == true, "Image must be in real space");

    EmpiricalDistribution my_distribution = ReturnDistributionOfRealValues(wanted_mask_radius, outside);
    AddConstant(-my_distribution.GetSampleMean());

    return my_distribution.GetNumberOfSamples();
}

long Image::ZeroFloatAndNormalize(float wanted_sigma_value, float wanted_mask_radius, bool outside) {
    MyDebugAssertTrue(is_in_real_space == true, "Image must be in real space");

    EmpiricalDistribution my_distribution = ReturnDistributionOfRealValues(wanted_mask_radius, outside);
    if (my_distribution.IsConstant()) {
        AddConstant(-my_distribution.GetSampleMean());
    }
    else {
        AddMultiplyConstant(-my_distribution.GetSampleMean(),
                            wanted_sigma_value / sqrtf(my_distribution.GetSampleVariance()));
    }
    return my_distribution.GetNumberOfSamples();
}

// Normalize without zero-floating
long Image::Normalize(float wanted_sigma_value, float wanted_mask_radius, bool outside) {
    MyDebugAssertTrue(is_in_real_space == true, "Image must be in real space");

    EmpiricalDistribution my_distribution = ReturnDistributionOfRealValues(wanted_mask_radius, outside);
    if (!my_distribution.IsConstant()) {
        AddMultiplyAddConstant(-my_distribution.GetSampleMean(),
                               wanted_sigma_value / sqrtf(my_distribution.GetSampleVariance()),
                               my_distribution.GetSampleMean());
    }
    return my_distribution.GetNumberOfSamples();
}

void Image::ZeroFloatOutside(float wanted_mask_radius, bool invert_mask) {
    MyDebugAssertTrue(is_in_real_space == true, "Image must be in real space");

    float average = ReturnAverageOfRealValues(wanted_mask_radius, !invert_mask);

    AddConstant(-average);
}

// Pixels with values greater than maximum_n_sigmas above the mean or less than maximum_n_sigmas below the mean will be
// replaced with the mean
void Image::ReplaceOutliersWithMean(float maximum_n_sigmas) {
    MyDebugAssertTrue(is_in_real_space, "Image must be in real space");

    float sigma = sqrtf(ReturnVarianceOfRealValues());
    float mean  = ReturnAverageOfRealValues();
    float max   = mean + maximum_n_sigmas * sigma;
    float min   = mean - maximum_n_sigmas * sigma;

    for (long address = 0; address < real_memory_allocated; address++) {
        if (real_values[address] > max) {
            real_values[address] = mean;
        }
        else if (real_values[address] < min) {
            real_values[address] = mean;
        }
    }
}

float Image::ReturnVarianceOfRealValues(float wanted_mask_radius, float wanted_center_x, float wanted_center_y,
                                        float wanted_center_z, bool invert_mask) {
    MyDebugAssertTrue(is_in_real_space == true, "Image must be in real space");
    MyDebugAssertTrue(logical_z_dimension != 1 || wanted_center_z == 0.0, "Requesting 3D mask coordinates in 2D image");

    int  i;
    int  j;
    int  k;
    long number_of_pixels = 0;

    float x;
    float y;
    float z;

    long pixel_counter = 0;

    float  distance_from_center_squared;
    float  mask_radius_squared;
    float  edge;
    float  center_x;
    float  center_y;
    float  center_z;
    double pixel_sum         = 0.0;
    double pixel_sum_squared = 0.0;

    if (wanted_center_x == 0.0 && wanted_center_y == 0.0 && wanted_center_z == 0.0) {
        center_x = physical_address_of_box_center_x;
        center_y = physical_address_of_box_center_y;
        center_z = physical_address_of_box_center_z;
    }
    else {
        center_x = wanted_center_x;
        center_y = wanted_center_y;
        center_z = wanted_center_z;
    }

    if (wanted_mask_radius > 0.0) {
        mask_radius_squared = powf(wanted_mask_radius, 2);
        for (k = 0; k < logical_z_dimension; k++) {
            z = powf(k - center_z, 2);

            for (j = 0; j < logical_y_dimension; j++) {
                y = powf(j - center_y, 2);

                for (i = 0; i < logical_x_dimension; i++) {
                    x = powf(i - center_x, 2);

                    distance_from_center_squared = x + y + z;

                    if (invert_mask) {
                        if (distance_from_center_squared > mask_radius_squared) {
                            pixel_sum += real_values[pixel_counter];
                            pixel_sum_squared += powf(real_values[pixel_counter], 2);
                            number_of_pixels++;
                        }
                    }
                    else {
                        if (distance_from_center_squared <= mask_radius_squared) {
                            pixel_sum += real_values[pixel_counter];
                            pixel_sum_squared += powf(real_values[pixel_counter], 2);
                            number_of_pixels++;
                        }
                    }
                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }
    else {
        for (k = 0; k < logical_z_dimension; k++) {
            for (j = 0; j < logical_y_dimension; j++) {
                for (i = 0; i < logical_x_dimension; i++) {
                    pixel_sum += real_values[pixel_counter];
                    pixel_sum_squared += powf(real_values[pixel_counter], 2);
                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
        number_of_pixels = long(logical_x_dimension) * long(logical_y_dimension) * long(logical_z_dimension);
    }

    if (number_of_pixels > 0) {
        return fabsf(float(pixel_sum_squared / number_of_pixels - powf(pixel_sum / number_of_pixels, 2)));
    }
    else {
        return 0.0;
    }
}

void Image::ApplySqrtNFilter() {
    MyDebugAssertTrue(is_in_memory, "Image memory not allocated");
    MyDebugAssertTrue(!is_in_real_space, "Image not in Fourier space");

    int i;
    int j;
    int k;
    int bin;

    float x;
    float y;
    float z;
    float frequency;
    float frequency_squared;

    long pixel_counter = 0;

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                x                 = powf(i * fourier_voxel_size_x, 2);
                frequency_squared = x + y + z;

                complex_values[pixel_counter] /= sqrtf(frequency_squared);
                pixel_counter++;
            }
        }
    }
}

void Image::Whiten(float resolution_limit, Curve* whitening_filter) {
    MyDebugAssertTrue(is_in_real_space == false, "Image to whiten is not in Fourier space");

    int i;
    int j;
    int k;
    int bin;

    float x;
    float y;
    float z;
    float frequency;
    float frequency_squared;
    float temp_float;
    float resolution_limit_pixel = resolution_limit * logical_x_dimension;
    //	float resolution_limit_pixel = std::min(float(0.5), resolution_limit) * logical_x_dimension - 1.0;

    int number_of_bins = ReturnLargestLogicalDimension() / 2 + 1;
    // Extend table to include corners in 3D Fourier space
    int     number_of_bins_extended = int(number_of_bins * sqrtf(3.0)) + 1;
    double* sum                     = new double[number_of_bins_extended];
    int     number_of_bins2         = 2 * (number_of_bins - 1);
    int*    non_zero_count          = new int[number_of_bins_extended];
    ZeroDoubleArray(sum, number_of_bins_extended);
    ZeroIntArray(non_zero_count, number_of_bins_extended);

    if (whitening_filter != NULL) // setup the curve
    {
        whitening_filter->ClearData();
    }

    long pixel_counter = 0;

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                temp_float = powf(abs(complex_values[pixel_counter]), 2);
                if (temp_float != 0.0) {
                    x                 = powf(i * fourier_voxel_size_x, 2);
                    frequency_squared = x + y + z;

                    // compute radius, in units of physical Fourier pixels
                    bin = int(sqrtf(frequency_squared) * number_of_bins2);
                    if (bin <= resolution_limit_pixel) {
                        sum[bin] += temp_float;
                        non_zero_count[bin] += 1;
                    }
                }
                pixel_counter++;
            }
        }
    }

    for (i = 0; i < number_of_bins_extended; i++) {
        if (non_zero_count[i] != 0)
            sum[i] = sqrtf(sum[i] / non_zero_count[i]);
    }

    pixel_counter = 0;
    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                x                 = powf(i * fourier_voxel_size_x, 2);
                frequency_squared = x + y + z;

                // compute radius, in units of physical Fourier pixels
                bin = int(sqrtf(frequency_squared) * number_of_bins2);

                if (bin <= resolution_limit_pixel) {
                    if (non_zero_count[bin] != 0.0) {
                        complex_values[pixel_counter] /= sum[bin];
                    }
                    else {
                        complex_values[pixel_counter] = 0.0f + I * 0.0f;
                    }
                }
                else {
                    complex_values[pixel_counter] = 0.0f + I * 0.0f;
                }
                pixel_counter++;
            }
        }
    }

    delete[] sum;
    delete[] non_zero_count;
}

void Image::MultiplyByWeightsCurve(Curve& weights, float scale_factor) {
    MyDebugAssertTrue(is_in_real_space == false, "Image to filter not in Fourier space");
    MyDebugAssertTrue(weights.number_of_points > 0, "weights curve not calculated");

    int i;
    int j;
    int k;
    int bin;

    float  x;
    float  y;
    float  z;
    float  frequency;
    float  frequency_squared;
    float* real_a;
    float* real_b;
    real_a = real_values;
    real_b = real_values + 1;

    int number_of_bins2 = ReturnLargestLogicalDimension();

    long pixel_counter = 0;

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                x                 = powf(i * fourier_voxel_size_x, 2);
                frequency_squared = x + y + z;
                // compute radius, in units of physical Fourier pixels
                bin = int(sqrtf(frequency_squared) * number_of_bins2);

                if ((frequency_squared <= 0.25) && (bin < weights.number_of_points)) {
                    //					if (power == 1.0)
                    //					{
                    real_a[pixel_counter] = real_a[pixel_counter] * fabsf(weights.data_y[bin]) * scale_factor;
                    real_b[pixel_counter] = real_b[pixel_counter] * fabsf(weights.data_y[bin]) * scale_factor;
                    //					}
                    //					else
                    //					{
                    //						real_a[pixel_counter] = (powf(real_a[pixel_counter], power) +
                    //powf(real_b[pixel_counter], power)) * fabsf(weights.data_y[bin]) * scale_factor;
                    //						real_b[pixel_counter] = 0.0;
                    //					}
                }
                else {
                    real_a[pixel_counter] = 0.0;
                    real_b[pixel_counter] = 0.0;
                }
                pixel_counter += 2;
            }
        }
    }
}

void Image::OptimalFilterBySNRImage(Image& SNR_image, int include_reference_weighting) {
    MyDebugAssertTrue(is_in_real_space == false, "Image to filter not in Fourier space");
    MyDebugAssertTrue(SNR_image.is_in_real_space == false, "SNR image not in Fourier space");

    int    i;
    float  weight;
    float* real_a;
    float* real_b;
    real_a = real_values;
    real_b = real_values + 1;
    float* real_SNR_a;
    //	float *real_SNR_b;
    real_SNR_a = SNR_image.real_values;
    //	real_SNR_b = SNR_image.real_values + 1;

    long pixel_counter = 0;

    for (i = 0; i < real_memory_allocated / 2; i++) {
        // After whitening, the noise variance should be 1/(1+snr).
        // To whiten the noise, we must divide by the sqrt of the
        // variance, i.e. multiply by sqrt(1+snr).
        if (include_reference_weighting == 1) {
            // The line below is what was derived for the optimal filter
            weight = sqrtf(real_SNR_a[pixel_counter] + powf(real_SNR_a[pixel_counter], 2));
            // The line below was used in earlier versions but is slightly incorrect
            //			weight = sqrtf(real_SNR_a[pixel_counter]) + real_SNR_a[pixel_counter];
        }
        else if (include_reference_weighting == 0) {
            weight = sqrtf(1.0 + real_SNR_a[pixel_counter]);
        }
        else {
            weight = sqrtf(real_SNR_a[pixel_counter]);
        }
        real_a[pixel_counter] *= weight;
        real_b[pixel_counter] *= weight;
        //		complex_values[pixel_counter] *= (1.0 + snr);
        //		complex_values[pixel_counter] *= snr / (1.0 + snr);
        pixel_counter += 2;
    }
}

void Image::WeightBySSNR(Image& ctf_image, float molecular_mass_kDa, float pixel_size, Curve& SSNR,
                         Image& projection_image, bool weight_particle_image, bool weight_projection_image) {
    MyDebugAssertTrue(is_in_memory, "image memory not allocated");
    MyDebugAssertTrue(projection_image.is_in_memory, "projection_image memory not allocated");
    MyDebugAssertTrue(ctf_image.is_in_memory, "CTF image memory not allocated");

    int   i;
    float particle_area_in_pixels =
        PI * powf(3.0 * (kDa_to_Angstrom3(molecular_mass_kDa) / powf(pixel_size, 3)) / 4.0 / PI, 2.0 / 3.0);
    float ssnr_scale_factor = particle_area_in_pixels / logical_x_dimension / logical_y_dimension;

    if (is_in_real_space)
        ForwardFFT();

    Image* snr_image = new Image;
    snr_image->Allocate(ctf_image.logical_x_dimension, ctf_image.logical_y_dimension, false);

    for (i = 0; i < ctf_image.real_memory_allocated / 2; i++) {
        snr_image->complex_values[i] = ctf_image.complex_values[i] * conj(ctf_image.complex_values[i]);
    }
    if (!weight_projection_image)
        projection_image.MultiplyPixelWiseReal(*snr_image, true);
    snr_image->MultiplyByWeightsCurve(SSNR, ssnr_scale_factor);
    if (weight_particle_image) {
        Whiten();
        OptimalFilterBySNRImage(*snr_image, 0);
    }
    if (weight_projection_image)
        projection_image.OptimalFilterBySNRImage(*snr_image, -1);

    delete snr_image;
}

/*
 * A Wiener-like filter proposed by Tegunov & Cramer
 * ("Real-time cryo-EM data pre-processing with Warp" 2018)
 *
 * The SSNR is modeled as an exponential falloff parametrised by two
 * parameters, modulated by the CTF. Fudge factors default to 1.0 and
 * are probably not needed.
 *
 */
void Image::OptimalFilterWarp(CTF ctf, float pixel_size_in_angstroms, float ssnr_falloff_fudge_factor,
                              float ssnr_scale_fudge_factor, float ssnr_falloff_frequency, float high_pass_frequency) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == false, "image not in Fourier space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Volumes not supported");

    int j;
    int i;

    long pixel_counter = 0;

    float y_coord_sq;
    float x_coord_sq;

    float y_coord;
    float x_coord;

    float frequency_squared;
    float frequency;
    float azimuth;
    float ctf_value;
    float hp_value;
    float ssnr_value_pre_ctf;
    float filter_value;

    const float ssnr_falloff_characteristic_spacing =
        ssnr_falloff_frequency / pixel_size_in_angstroms *
        ssnr_falloff_fudge_factor; // 100A was suggested by Tegunov, seems to work well
    const float ssnr_peak_scale_factor   = powf(10.0, 3.0 * ssnr_scale_fudge_factor);
    const float hp_radius                = 1.0 / high_pass_frequency * pixel_size_in_angstroms;
    const float hp_width                 = 1.0 * hp_radius;
    const float hp_radius_start_squared  = powf(hp_radius - 0.5 * hp_width, 2);
    const float hp_radius_finish_squared = powf(hp_radius + 0.5 * hp_width, 2);

    for (j = 0; j <= physical_upper_bound_complex_y; j++) {
        y_coord    = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y;
        y_coord_sq = powf(y_coord, 2.0);

        for (i = 0; i <= physical_upper_bound_complex_x; i++) {
            x_coord    = i * fourier_voxel_size_x;
            x_coord_sq = powf(x_coord, 2);

            // Compute the azimuth
            if (i == 0 && j == 0) {
                azimuth = 0.0;
            }
            else {
                azimuth = atan2f(y_coord, x_coord);
            }

            // Compute the square of the frequency
            frequency_squared = x_coord_sq + y_coord_sq;
            frequency         = sqrtf(frequency_squared);

            ctf_value = ctf.Evaluate(frequency_squared, azimuth);
            if (frequency_squared >= hp_radius_start_squared && frequency_squared <= hp_radius_finish_squared) {
                hp_value = 1.0 - ((1.0 + cosf(PI * (frequency - hp_radius) / hp_width)) / 2.0);
            }
            else if (frequency_squared < hp_radius_start_squared) {
                hp_value = 0.0;
            }
            { hp_value = 1.0; }
            ssnr_value_pre_ctf = exp(-frequency * ssnr_falloff_characteristic_spacing) * ssnr_peak_scale_factor;

            // Wiener filter
            filter_value = -ctf_value / (powf(ctf_value, 2) + 1.0 / (ssnr_value_pre_ctf * hp_value));

            /*
            if (j==0 && i < 300 && i > 0)
            {
                wxPrintf("sp = %f; ssnr = %f; hp = %f; ctf_value = %f; filter =
            %f\n",1.5/frequency,ssnr_value_pre_ctf,hp_value,ctf_value,filter_value);
            }
            */

            complex_values[pixel_counter] *= filter_value;
            pixel_counter++;
        }
    }
}

void Image::OptimalFilterSSNR(Curve& SSNR) {
    MyDebugAssertTrue(is_in_real_space == false, "Image to filter not in Fourier space");
    MyDebugAssertTrue(SSNR.number_of_points > 0, "SSNR curve not calculated");

    int i;
    int j;
    int k;
    int bin;
    int number_of_bins2;

    float x;
    float y;
    float z;
    float frequency;
    float frequency_squared;
    float snr;

    long pixel_counter = 0;

    //	number_of_bins2 = 2 * (SSNR.number_of_points - 1);
    number_of_bins2 = ReturnLargestLogicalDimension();

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                x                 = powf(i * fourier_voxel_size_x, 2);
                frequency_squared = x + y + z;
                // compute radius, in units of physical Fourier pixels
                bin = int(sqrtf(frequency_squared) * number_of_bins2);

                if ((frequency_squared <= 0.25) && (bin < SSNR.number_of_points)) {
                    snr = fabsf(SSNR.data_y[bin]);
                    // Should this be the same as in OptimalFilterBySNRImage?
                    //					complex_values[pixel_counter] *= sqrtf(1.0 + snr);
                    complex_values[pixel_counter] *= snr / (1.0 + snr);
                }
                else {
                    complex_values[pixel_counter] = 0.0f + I * 0.0f;
                }
                pixel_counter++;
            }
        }
    }
}

void Image::OptimalFilterFSC(Curve& FSC) {
    MyDebugAssertTrue(is_in_real_space == false, "Image to filter not in Fourier space");
    MyDebugAssertTrue(FSC.number_of_points > 0, "FSC curve not calculated");

    int i;
    int j;
    int k;
    int bin;
    int number_of_bins2;

    float x;
    float y;
    float z;
    float frequency;
    float frequency_squared;

    long pixel_counter = 0;

    //	number_of_bins2 = 2 * (FSC.number_of_points - 1);
    number_of_bins2 = ReturnLargestLogicalDimension();

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                x                 = powf(i * fourier_voxel_size_x, 2);
                frequency_squared = x + y + z;
                // compute radius, in units of physical Fourier pixels
                bin = int(sqrtf(frequency_squared) * number_of_bins2);

                if ((frequency_squared <= 0.25) && (bin < FSC.number_of_points)) {
                    //					if (FSC.data_y[bin] != 0.0) complex_values[pixel_counter] /= (1.0 + 0.5 * (1.0 -
                    //fabsf(FSC.data_y[bin])) / fabsf(FSC.data_y[bin]));
                    if (FSC.data_y[bin] != 0.0)
                        complex_values[pixel_counter] *= 2.0 * fabsf(FSC.data_y[bin]) / (1.0 + fabsf(FSC.data_y[bin]));
                    //					if (j == 0 && k == 0) wxPrintf("FSC, filt = %i %g %g %g\n", bin, x, FSC.data_y[bin], 2.0
                    //* fabsf(FSC.data_y[bin]) / (1.0 + fabsf(FSC.data_y[bin])));
                }
                else {
                    complex_values[pixel_counter] = 0.0f + I * 0.0f;
                }
                pixel_counter++;
            }
        }
    }
}
/*
float Image::Correct3D(float wanted_mask_radius)
{
    MyDebugAssertTrue(is_in_real_space == true, "reconstruction to correct not in real space");

    int i;
    int j;
    int k;
    int int_x_coordinate;
    int int_y_coordinate;
    int int_z_coordinate;

    float x;
    float y;
    float z;
    float distance_from_center_squared;
    float mask_radius = wanted_mask_radius;
    float mask_radius_squared;
    double pixel_sum = 0.0;

    long pixel_counter = 0;

    float weight;
    float weight_y;
    float weight_z;
    float scale_x = PI / logical_x_dimension;
    float scale_y = PI / logical_y_dimension;
    float scale_z = PI / logical_z_dimension;

    if (wanted_mask_radius <= 0.0) mask_radius = logical_x_dimension + logical_y_dimension + logical_z_dimension;
    mask_radius_squared = powf(mask_radius,2);

    for (k = 0; k < logical_z_dimension; k++)
    {
        int_z_coordinate = k - physical_address_of_box_center_z;
        z = powf(int_z_coordinate, 2);
        weight_z = sinc(float(int_z_coordinate) * scale_z);

        for (j = 0; j < logical_y_dimension; j++)
        {
            int_y_coordinate = j - physical_address_of_box_center_y;
            y = powf(int_y_coordinate, 2);
            weight_y = sinc(float(int_y_coordinate) * scale_y);

            for (i = 0; i < logical_x_dimension; i++)
            {
                int_x_coordinate = i - physical_address_of_box_center_x;
                x = powf(int_x_coordinate, 2);

                weight = powf(sinc(float(int_x_coordinate) * scale_x) * weight_y * weight_z,2);

                real_values[pixel_counter] /= weight;

                distance_from_center_squared = x + y + z;

                if (distance_from_center_squared <= mask_radius_squared)
                {
                    pixel_sum += powf(weight,2);
                }

                pixel_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }
    if (pixel_sum == 0.0) return 0.0;
    if (wanted_mask_radius <= 0.0) return pixel_sum / logical_x_dimension / logical_y_dimension / logical_z_dimension;
    return pixel_sum / (4.0 / 3.0 * PI * powf(mask_radius,3));
}*/

float Image::CorrectSinc(float wanted_mask_radius, float padding_factor, bool force_background_value,
                         float wanted_mask_value) {
    //	MyDebugAssertTrue(is_in_real_space == true, "reconstruction to correct not in real space");
    MyDebugAssertTrue(padding_factor >= 1, "invalid padding factor");

    int i;
    int j;
    int k;
    int int_x_coordinate;
    int int_y_coordinate;
    int int_z_coordinate;

    float  x;
    float  y;
    float  z;
    float  distance_from_center_squared;
    float  mask_radius = wanted_mask_radius;
    float  mask_radius_squared;
    float  average;
    double pixel_sum = 0.0;

    long pixel_counter = 0;

    float weight;
    float weight_outside;
    float weight_y;
    float weight_z;
    float scale_x = PI / int(logical_x_dimension * padding_factor + 0.5);
    float scale_y = PI / int(logical_y_dimension * padding_factor + 0.5);
    float scale_z = PI / int(logical_z_dimension * padding_factor + 0.5);

    if (is_in_real_space) {
        if (force_background_value)
            average = wanted_mask_value;
        else
            average = ReturnAverageOfRealValues(0.45 * logical_x_dimension, true);

        if (wanted_mask_radius <= 0.0)
            mask_radius = logical_x_dimension + logical_y_dimension + logical_z_dimension;
        mask_radius_squared = powf(mask_radius, 2);

        weight_outside = powf(sinc(mask_radius * scale_x), 2);

        for (k = 0; k < logical_z_dimension; k++) {
            int_z_coordinate = k - physical_address_of_box_center_z;
            z                = powf(int_z_coordinate, 2);
            if (z < mask_radius_squared)
                weight_z = sinc(float(int_z_coordinate) * scale_z);
            for (j = 0; j < logical_y_dimension; j++) {
                int_y_coordinate = j - physical_address_of_box_center_y;
                y                = powf(int_y_coordinate, 2);
                if (y < mask_radius_squared)
                    weight_y = sinc(float(int_y_coordinate) * scale_y);
                for (i = 0; i < logical_x_dimension; i++) {
                    int_x_coordinate             = i - physical_address_of_box_center_x;
                    x                            = powf(int_x_coordinate, 2);
                    distance_from_center_squared = x + y + z;
                    if (distance_from_center_squared < mask_radius_squared) {
                        weight = powf(sinc(float(int_x_coordinate) * scale_x) * weight_y * weight_z, 2);
                        real_values[pixel_counter] = (real_values[pixel_counter] - average) / weight + average;

                        //						distance_from_center_squared = x + y + z;

                        //						if (distance_from_center_squared <= mask_radius_squared)
                        //						{
                        pixel_sum += powf(weight, 2);
                        //						}
                    }
                    else
                        real_values[pixel_counter] = (real_values[pixel_counter] - average) / weight_outside + average;

                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
        //		if (pixel_sum == 0.0) return 0.0;
        return pixel_sum / logical_x_dimension / logical_y_dimension / logical_z_dimension;
        //		if (wanted_mask_radius <= 0.0) return pixel_sum / logical_x_dimension / logical_y_dimension /
        //logical_z_dimension; 		return pixel_sum / (4.0 / 3.0 * PI * powf(mask_radius,3));
    }
    else {
        for (k = 0; k <= physical_upper_bound_complex_z; k++) {
            weight_z = sinc(float(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k)) * scale_z);

            for (j = 0; j <= physical_upper_bound_complex_y; j++) {
                weight_y = sinc(float(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j)) * scale_y);

                for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                    weight = powf(sinc(float(i) * scale_x) * weight_y * weight_z, 2);

                    complex_values[pixel_counter] /= weight;

                    pixel_counter++;
                }
            }
        }
        return 0.0;
    }
}

void Image::MirrorXFourier2D(Image& mirrored_image) {
    MyDebugAssertTrue(mirrored_image.logical_z_dimension == 1, "Error: attempting to mirrored_image into 3D image");
    MyDebugAssertTrue(logical_z_dimension == 1, "Error: attempting to mirrored_image from 3D image");
    MyDebugAssertTrue(!is_in_real_space, "Image is in real space");
    MyDebugAssertTrue(mirrored_image.is_in_memory, "Memory not allocated for receiving image");
    MyDebugAssertTrue(mirrored_image.logical_x_dimension == logical_x_dimension &&
                          mirrored_image.logical_y_dimension == logical_y_dimension,
                      "Error: Images different sizes");

    int pixel_counter;
    int mirrored_counter;
    int x_counter;
    int y_counter;
    int fft_dim_x = physical_upper_bound_complex_x + 1;

    x_counter = 0;
    y_counter = 0;
    for (pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++) {
        mirrored_counter = y_counter * fft_dim_x + x_counter;
        //					correlation_map->complex_values[pixel_counter] = rotated_image->complex_values[mirrored_counter] *
        //conj(particle.particle_image->complex_values[pixel_counter]); 					correlation_map->complex_values[pixel_counter] =
        //conj(rotated_image->complex_values[pixel_counter]);
        mirrored_image.complex_values[pixel_counter] = complex_values[mirrored_counter];
        x_counter++;
        if (x_counter >= fft_dim_x) {
            x_counter -= fft_dim_x;
            y_counter--;
            if (y_counter < 0)
                y_counter += logical_y_dimension;
        }
    }
    mirrored_image.is_in_real_space = false;
}

void Image::MirrorYFourier2D(Image& mirrored_image) {
    MyDebugAssertTrue(mirrored_image.logical_z_dimension == 1, "Error: attempting to mirrored_image into 3D image");
    MyDebugAssertTrue(logical_z_dimension == 1, "Error: attempting to mirrored_image from 3D image");
    MyDebugAssertTrue(!is_in_real_space, "Image is in real space");
    MyDebugAssertTrue(mirrored_image.is_in_memory, "Memory not allocated for receiving image");
    MyDebugAssertTrue(mirrored_image.logical_x_dimension == logical_x_dimension &&
                          mirrored_image.logical_y_dimension == logical_y_dimension,
                      "Error: Images different sizes");

    int pixel_counter;
    int mirrored_counter;
    int x_counter;
    int y_counter;
    int fft_dim_x = physical_upper_bound_complex_x + 1;

    x_counter = 0;
    y_counter = 0;
    for (pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++) {
        mirrored_counter                             = y_counter * fft_dim_x + x_counter;
        mirrored_image.complex_values[pixel_counter] = conj(complex_values[mirrored_counter]);
        x_counter++;
        if (x_counter >= fft_dim_x) {
            x_counter -= fft_dim_x;
            y_counter--;
            if (y_counter < 0)
                y_counter += logical_y_dimension;
        }
    }
    mirrored_image.is_in_real_space = false;
}

void Image::RotateQuadrants(Image& rotated_image, int quad_i) {
    MyDebugAssertTrue(rotated_image.logical_z_dimension == 1, "Error: attempting to rotate into 3D image");
    MyDebugAssertTrue(logical_z_dimension == 1, "Error: attempting to rotate from 3D image");
    MyDebugAssertTrue(rotated_image.is_in_memory, "Memory not allocated for receiving image");
    //	MyDebugAssertTrue(! is_in_real_space, "Image is in real space");
    MyDebugAssertTrue(IsSquare(), "Image to rotate is not square");
    MyDebugAssertTrue(rotated_image.logical_x_dimension == logical_x_dimension &&
                          rotated_image.logical_y_dimension == logical_y_dimension,
                      "Error: Images different sizes");
    MyDebugAssertTrue(quad_i == 0 || quad_i == 90 || quad_i == 180 || quad_i == 270, "Selected rotation invalid");

    int i;
    int j;

    long pixel_counter;
    long pixel_counter2;

    rotated_image.object_is_centred_in_box = object_is_centred_in_box;
    rotated_image.is_in_real_space         = is_in_real_space;

    if (!is_in_real_space) {
        if (quad_i == 0) {
            for (i = 0; i < real_memory_allocated / 2; i++) {
                rotated_image.complex_values[i] = complex_values[i];
            };
            return;
        }

        if (quad_i == 180) {
            for (i = 0; i < real_memory_allocated / 2; i++) {
                rotated_image.complex_values[i] = conj(complex_values[i]);
            };
            return;
        }

        if (quad_i == 90) {
            for (j = logical_lower_bound_complex_y; j <= logical_upper_bound_complex_y; j++) {
                for (i = 0; i <= logical_upper_bound_complex_x; i++) {
                    pixel_counter = ReturnFourier1DAddressFromLogicalCoord(i, j, 0);

                    if (i <= logical_upper_bound_complex_y) {
                        pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(-j, i, 0);
                    }
                    else {
                        pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(-j, -i, 0);
                    }

                    if (j <= 0) {
                        rotated_image.complex_values[pixel_counter2] = conj(complex_values[pixel_counter]);
                    }
                    else {
                        rotated_image.complex_values[pixel_counter2] = complex_values[pixel_counter];
                    }
                }
            }
            return;
        }

        if (quad_i == 270) {
            for (j = logical_lower_bound_complex_y; j <= logical_upper_bound_complex_y; j++) {
                for (i = 0; i <= logical_upper_bound_complex_x; i++) {
                    pixel_counter = ReturnFourier1DAddressFromLogicalCoord(i, j, 0);

                    if (i <= logical_upper_bound_complex_y) {
                        pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(-j, i, 0);
                    }
                    else {
                        pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(-j, -i, 0);
                    }

                    if (j <= 0) {
                        rotated_image.complex_values[pixel_counter2] = complex_values[pixel_counter];
                    }
                    else {
                        rotated_image.complex_values[pixel_counter2] = conj(complex_values[pixel_counter]);
                    }
                }
            }
            return;
        }
        //	    rotated_image.is_in_real_space = false;
    }
    else {
        if (quad_i == 0) {
            for (i = 0; i < real_memory_allocated; i++) {
                rotated_image.real_values[i] = real_values[i];
            };
            return;
        }

        if (quad_i == 180) {
            pixel_counter  = 0;
            pixel_counter2 = real_memory_allocated;
            for (j = 0; j < logical_y_dimension; j++) {
                pixel_counter2 -= padding_jump_value;
                for (i = 0; i < logical_x_dimension; i++) {
                    pixel_counter2--;
                    rotated_image.real_values[pixel_counter2] = real_values[pixel_counter];
                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
            return;
        }

        if (quad_i == 90) {
            pixel_counter = 0;
            for (j = 1; j <= logical_y_dimension; j++) {
                for (i = 0; i < logical_x_dimension; i++) {
                    pixel_counter++;

                    pixel_counter2 = rotated_image.ReturnReal1DAddressFromPhysicalCoord(logical_y_dimension - j, i, 0);
                    rotated_image.real_values[pixel_counter2] = real_values[pixel_counter];
                }
                pixel_counter += padding_jump_value;
            }
            return;
        }

        if (quad_i == 270) {
            pixel_counter = 0;
            for (j = 0; j < logical_y_dimension; j++) {
                for (i = 1; i <= logical_x_dimension; i++) {
                    pixel_counter++;

                    pixel_counter2 = rotated_image.ReturnReal1DAddressFromPhysicalCoord(j, logical_x_dimension - i, 0);
                    rotated_image.real_values[pixel_counter2] = real_values[pixel_counter];
                }
                pixel_counter += padding_jump_value;
            }
            return;
        }
    }
}

void Image::GenerateReferenceProjections(Image* projections, EulerSearch& parameters, float resolution) {
    int             i;
    float           variance;
    float           effective_bfactor = 100;
    AnglesAndShifts angles;

    //	if (resolution != 0.0)
    //	{
    //		effective_bfactor = 2.0 * 4.0 * powf(1.0 / resolution,2);
    //	}
    //	else
    //	{
    //		effective_bfactor = 0.0;
    //	}

    for (i = 0; i < parameters.number_of_search_positions; i++) {
        angles.Init(parameters.list_of_search_parameters[i][0], parameters.list_of_search_parameters[i][1], 0.0, 0.0,
                    0.0);
        ExtractSlice(projections[i], angles, parameters.resolution_limit);
        //		projections[i].BackwardFFT();
        //		projections[i].ZeroFloatOutside(0.5 * projections[i].logical_x_dimension - 1.0);
        //		projections[i].ForwardFFT();
        //		variance = projections[i].ReturnSumOfSquares();
        projections[i].Whiten(resolution);
        projections[i].ApplyBFactor(effective_bfactor);
        variance = projections[i].ReturnSumOfSquares();
        if (variance > 0.0)
            projections[i].MultiplyByConstant(1.0 / sqrtf(variance));
    }
}

void Image::RotateFourier2DGenerateIndex(Kernel2D**& kernel_index, float psi_max, float psi_step, float psi_start,
                                         bool invert_angle) {
    int             psi_i;
    int             number_of_psi_positions;
    float           psi;
    AnglesAndShifts angles;

    number_of_psi_positions = myroundint(psi_max / psi_step);
    if (number_of_psi_positions < 1)
        number_of_psi_positions = 1;
    //	wxPrintf("psi_max = %f, psi_step = %f, number of psi positions = %i\n", psi_max, psi_step,
    //number_of_psi_positions);
    kernel_index = new Kernel2D*[number_of_psi_positions]; // dynamic array of pointers to float
    for (psi_i = 0; psi_i < number_of_psi_positions; psi_i++) {
        kernel_index[psi_i] =
            new Kernel2D[real_memory_allocated / 2]; // each i-th pointer is now pointing to dynamic array (size
                                                     // number_of_positions) of actual float values
    }

    for (psi_i = 0; psi_i < number_of_psi_positions; psi_i++) {
        if (invert_angle) {
            angles.GenerateRotationMatrix2D(-psi_i * psi_step - psi_start);
        }
        else {
            angles.GenerateRotationMatrix2D(psi_i * psi_step + psi_start);
        }
        RotateFourier2DIndex(kernel_index[psi_i], angles);
    }
}

void Image::RotateFourier2DDeleteIndex(Kernel2D**& kernel_index, float psi_max, float psi_step) {
    int psi_i;
    int number_of_psi_positions;

    number_of_psi_positions = myroundint(psi_max / psi_step);
    if (number_of_psi_positions < 1)
        number_of_psi_positions = 1;
    for (psi_i = 0; psi_i < number_of_psi_positions; psi_i++) {
        delete[] kernel_index[psi_i]; // delete inner arrays of floats
    }
    delete[] kernel_index; // delete array of pointers to float arrays

    kernel_index = NULL;
}

void Image::RotateFourier2DFromIndex(Image& rotated_image, Kernel2D* kernel_index) {
    MyDebugAssertTrue(rotated_image.logical_z_dimension == 1, "Error: attempting to rotate into 3D image");
    MyDebugAssertTrue(logical_z_dimension == 1, "Error: attempting to rotate from 3D image");
    MyDebugAssertTrue(rotated_image.is_in_memory, "Memory not allocated for receiving image");
    MyDebugAssertTrue(!is_in_real_space, "Image is in real space");
    MyDebugAssertTrue(IsSquare(), "Image to rotate is not square");
    MyDebugAssertTrue(rotated_image.logical_x_dimension == logical_x_dimension &&
                          rotated_image.logical_y_dimension == logical_y_dimension,
                      "Error: Images different sizes");
    MyDebugAssertTrue(!object_is_centred_in_box, "Image volume quadrants not swapped");

    int   i;
    int   index;
    float weight;
    int   pixel_counter;

    for (pixel_counter = 0; pixel_counter < rotated_image.real_memory_allocated / 2; pixel_counter++) {
        rotated_image.complex_values[pixel_counter] = 0.0f + I * 0.0f;
        if (kernel_index[pixel_counter].pixel_index[0] != kernel_index[pixel_counter].pixel_index[3]) {
            for (i = 0; i < 4; i++) {
                index  = kernel_index[pixel_counter].pixel_index[i];
                weight = kernel_index[pixel_counter].pixel_weight[i];
                if (weight < 0.0) {
                    rotated_image.complex_values[pixel_counter] -= conj(complex_values[index]) * weight;
                }
                else {
                    rotated_image.complex_values[pixel_counter] += complex_values[index] * weight;
                }
            }
        }
    }
    rotated_image.is_in_real_space = false;
}

void Image::RotateFourier2DIndex(Kernel2D* kernel_index, AnglesAndShifts& rotation_angle, float resolution_limit,
                                 float padding_factor) {
    MyDebugAssertTrue(IsSquare(), "Image to rotate is not square");

    int i;
    int j;

    int pixel_counter;
    int pixel_counter2;

    float x_coordinate_2d;
    float y_coordinate_2d;

    float x_coordinate_3d;
    float y_coordinate_3d;

    float resolution_limit_sq = powf(resolution_limit * logical_x_dimension, 2);
    float y_coord_sq;

    Kernel2D null_kernel;

    null_kernel.pixel_index[0]  = 0;
    null_kernel.pixel_index[1]  = 0;
    null_kernel.pixel_index[2]  = 0;
    null_kernel.pixel_index[3]  = 0;
    null_kernel.pixel_weight[0] = 0.0;
    null_kernel.pixel_weight[1] = 0.0;
    null_kernel.pixel_weight[2] = 0.0;
    null_kernel.pixel_weight[3] = 0.0;

    for (j = logical_lower_bound_complex_y; j <= logical_upper_bound_complex_y; j++) {
        y_coordinate_2d = j * padding_factor;
        y_coord_sq      = powf(y_coordinate_2d, 2);
        for (i = 1; i <= logical_upper_bound_complex_x; i++) {
            x_coordinate_2d = i * padding_factor;
            pixel_counter   = ReturnFourier1DAddressFromLogicalCoord(i, j, 0);
            if (powf(x_coordinate_2d, 2) + y_coord_sq <= resolution_limit_sq) {
                rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d,
                                                           y_coordinate_3d);
                //				rotated_image.complex_values[pixel_counter] =
                //ReturnLinearInterpolatedFourier2D(x_coordinate_3d, y_coordinate_3d);
                kernel_index[pixel_counter] = ReturnLinearInterpolatedFourierKernel2D(x_coordinate_3d, y_coordinate_3d);
            }
            else {
                //				rotated_image.complex_values[pixel_counter] = 0.0;
                kernel_index[pixel_counter] = null_kernel;
            }
        }
    }
    // Now deal with special case of i = 0
    for (j = 1; j <= logical_upper_bound_complex_y; j++) {
        y_coordinate_2d = j * padding_factor;
        x_coordinate_2d = 0;
        if (powf(y_coordinate_2d, 2) <= resolution_limit_sq) {
            rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d,
                                                       y_coordinate_3d);
            pixel_counter = ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
            //			rotated_image.complex_values[pixel_counter] = ReturnLinearInterpolatedFourier2D(x_coordinate_3d,
            //y_coordinate_3d);
            kernel_index[pixel_counter] = ReturnLinearInterpolatedFourierKernel2D(x_coordinate_3d, y_coordinate_3d);
            pixel_counter2              = ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
            //			rotated_image.complex_values[pixel_counter2] =
            //conj(rotated_image.complex_values[pixel_counter]);
            kernel_index[pixel_counter2]                 = kernel_index[pixel_counter];
            kernel_index[pixel_counter2].pixel_weight[0] = -kernel_index[pixel_counter2].pixel_weight[0];
            kernel_index[pixel_counter2].pixel_weight[1] = -kernel_index[pixel_counter2].pixel_weight[1];
            kernel_index[pixel_counter2].pixel_weight[2] = -kernel_index[pixel_counter2].pixel_weight[2];
            kernel_index[pixel_counter2].pixel_weight[3] = -kernel_index[pixel_counter2].pixel_weight[3];
        }
        else {
            pixel_counter = ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
            //			rotated_image.complex_values[pixel_counter] = 0.0;
            kernel_index[pixel_counter] = null_kernel;
            pixel_counter2              = ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
            //			rotated_image.complex_values[pixel_counter2] = 0.0;
            kernel_index[pixel_counter2] = null_kernel;
        }
    }
    // Deal with pixel at edge if image dimensions are even
    if (-logical_lower_bound_complex_y != logical_upper_bound_complex_y) {
        y_coordinate_2d = logical_lower_bound_complex_y * padding_factor;
        x_coordinate_2d = 0;
        rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
        pixel_counter = ReturnFourier1DAddressFromLogicalCoord(0, logical_lower_bound_complex_y, 0);
        //		rotated_image.complex_values[pixel_counter] = ReturnLinearInterpolatedFourier2D(x_coordinate_3d,
        //y_coordinate_3d);
        kernel_index[pixel_counter] = ReturnLinearInterpolatedFourierKernel2D(x_coordinate_3d, y_coordinate_3d);
    }

    // Set origin to zero to generate a projection with average set to zero
    //	rotated_image.complex_values[0] = 0.0;
    kernel_index[0] = null_kernel;
}

Kernel2D Image::ReturnLinearInterpolatedFourierKernel2D(float& x, float& y) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(!is_in_real_space, "Must be in Fourier space");

    //	fftwf_complex sum = 0.0;
    int      i;
    int      j;
    int      i_start;
    int      i_end;
    int      j_start;
    int      j_end;
    int      physical_y_address;
    int      jj;
    int      i_coeff = 0;
    Kernel2D kernel;

    float weight;
    float y_dist;

    kernel.pixel_index[0]  = 0;
    kernel.pixel_index[1]  = 0;
    kernel.pixel_index[2]  = 0;
    kernel.pixel_index[3]  = 0;
    kernel.pixel_weight[0] = 0.0;
    kernel.pixel_weight[1] = 0.0;
    kernel.pixel_weight[2] = 0.0;
    kernel.pixel_weight[3] = 0.0;

    if (x >= 0.0) {
        i_start = int(floorf(x));
        i_end   = i_start + 1;
        if (i_end > logical_upper_bound_complex_x)
            return kernel;

        j_start = int(floorf(y));
        if (j_start < logical_lower_bound_complex_y)
            return kernel;
        j_end = j_start + 1;
        if (j_end > logical_upper_bound_complex_y)
            return kernel;

        for (j = j_start; j <= j_end; j++) {
            if (j >= 0) {
                physical_y_address = j;
            }
            else {
                physical_y_address = logical_y_dimension + j;
            }
            jj     = (physical_upper_bound_complex_x + 1) * physical_y_address;
            y_dist = fabsf(y - float(j));
            for (i = i_start; i <= i_end; i++) {
                weight                       = (1.0 - fabsf(x - float(i))) * (1.0 - y_dist);
                kernel.pixel_index[i_coeff]  = jj + i;
                kernel.pixel_weight[i_coeff] = weight;
                i_coeff++;
                //				sum = sum + complex_values[jj + i] * weight;
            }
        }
    }
    else {
        i_start = int(floorf(x));
        if (i_start < logical_lower_bound_complex_x)
            return kernel;
        i_end = i_start + 1;

        j_start = int(floorf(y));
        if (j_start < logical_lower_bound_complex_y)
            return kernel;
        j_end = j_start + 1;
        if (j_end > logical_upper_bound_complex_y)
            return kernel;

        for (j = j_start; j <= j_end; j++) {
            if (j > 0) {
                physical_y_address = logical_y_dimension - j;
            }
            else {
                physical_y_address = -j;
            }
            jj     = (physical_upper_bound_complex_x + 1) * physical_y_address;
            y_dist = fabsf(y - float(j));
            for (i = i_start; i <= i_end; i++) {
                weight                       = (1.0 - fabsf(x - float(i))) * (1.0 - y_dist);
                weight                       = (1.0 - fabsf(x - float(i))) * (1.0 - y_dist);
                kernel.pixel_index[i_coeff]  = jj - i;
                kernel.pixel_weight[i_coeff] = -weight;
                i_coeff++;
                //				sum = sum + conj(complex_values[jj - i]) * weight;
            }
        }
    }
    return kernel;
}

void Image::RotateFourier2D(Image& rotated_image, AnglesAndShifts& rotation_angle,
                            float resolution_limit_in_reciprocal_pixels, bool use_nearest_neighbor) {
    MyDebugAssertTrue(rotated_image.logical_z_dimension == 1, "Error: attempting to rotate into 3D image");
    MyDebugAssertTrue(logical_z_dimension == 1, "Error: attempting to rotate from 3D image");
    MyDebugAssertTrue(rotated_image.is_in_memory, "Memory not allocated for receiving image");
    MyDebugAssertTrue(!is_in_real_space, "Image is in real space");
    MyDebugAssertTrue(IsSquare(), "Image to rotate is not square");
    MyDebugAssertTrue(rotated_image.logical_x_dimension == logical_x_dimension &&
                          rotated_image.logical_y_dimension == logical_y_dimension,
                      "Error: Images different sizes");
    MyDebugAssertTrue(!object_is_centred_in_box, "Image volume quadrants not swapped");

    int i;
    int j;

    long pixel_counter;
    long pixel_counter2;

    float x_coordinate_2d;
    float y_coordinate_2d;

    float x_coordinate_3d;
    float y_coordinate_3d;

    float resolution_limit_sq = powf(resolution_limit_in_reciprocal_pixels * logical_x_dimension, 2);
    float y_coord_sq;
    float padding_factor = logical_x_dimension / rotated_image.logical_x_dimension;

    rotated_image.object_is_centred_in_box = false;
    rotated_image.is_in_real_space         = false;
    //	image_to_extract.SetToConstant(0.0);

    if (use_nearest_neighbor) {
        for (j = rotated_image.logical_lower_bound_complex_y; j <= rotated_image.logical_upper_bound_complex_y; j++) {
            y_coordinate_2d = j * padding_factor;
            y_coord_sq      = powf(y_coordinate_2d, 2);
            for (i = 1; i <= rotated_image.logical_upper_bound_complex_x; i++) {
                x_coordinate_2d = i * padding_factor;
                pixel_counter   = rotated_image.ReturnFourier1DAddressFromLogicalCoord(i, j, 0);
                if (powf(x_coordinate_2d, 2) + y_coord_sq <= resolution_limit_sq) {
                    rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d,
                                                               y_coordinate_3d);
                    rotated_image.complex_values[pixel_counter] =
                        ReturnNearestFourier2D(x_coordinate_3d, y_coordinate_3d);
                }
                else {
                    rotated_image.complex_values[pixel_counter] = 0.0f + I * 0.0f;
                }
            }
        }
        // Now deal with special case of i = 0
        for (j = 1; j <= rotated_image.logical_upper_bound_complex_y; j++) {
            y_coordinate_2d = j * padding_factor;
            x_coordinate_2d = 0;
            if (powf(y_coordinate_2d, 2) <= resolution_limit_sq) {
                rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d,
                                                           y_coordinate_3d);
                pixel_counter = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
                rotated_image.complex_values[pixel_counter] = ReturnNearestFourier2D(x_coordinate_3d, y_coordinate_3d);
                pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
                rotated_image.complex_values[pixel_counter2] = conj(rotated_image.complex_values[pixel_counter]);
            }
            else {
                pixel_counter = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
                rotated_image.complex_values[pixel_counter] = 0.0f + I * 0.0f;
                pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
                rotated_image.complex_values[pixel_counter2] = 0.0f + I * 0.0f;
            }
        }
        // Deal with pixel at edge if image dimensions are even
        if (-rotated_image.logical_lower_bound_complex_y != rotated_image.logical_upper_bound_complex_y) {
            y_coordinate_2d = rotated_image.logical_lower_bound_complex_y * padding_factor;
            x_coordinate_2d = 0;
            rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d,
                                                       y_coordinate_3d);
            pixel_counter =
                rotated_image.ReturnFourier1DAddressFromLogicalCoord(0, rotated_image.logical_lower_bound_complex_y, 0);
            rotated_image.complex_values[pixel_counter] = ReturnNearestFourier2D(x_coordinate_3d, y_coordinate_3d);
        }
    }
    else {
        for (j = rotated_image.logical_lower_bound_complex_y; j <= rotated_image.logical_upper_bound_complex_y; j++) {
            y_coordinate_2d = j * padding_factor;
            y_coord_sq      = powf(y_coordinate_2d, 2);
            for (i = 1; i <= rotated_image.logical_upper_bound_complex_x; i++) {
                x_coordinate_2d = i * padding_factor;
                pixel_counter   = rotated_image.ReturnFourier1DAddressFromLogicalCoord(i, j, 0);
                if (powf(x_coordinate_2d, 2) + y_coord_sq <= resolution_limit_sq) {
                    rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d,
                                                               y_coordinate_3d);
                    rotated_image.complex_values[pixel_counter] =
                        ReturnLinearInterpolatedFourier2D(x_coordinate_3d, y_coordinate_3d);
                }
                else {
                    rotated_image.complex_values[pixel_counter] = 0.0f + I * 0.0f;
                }
            }
        }
        // Now deal with special case of i = 0
        for (j = 1; j <= rotated_image.logical_upper_bound_complex_y; j++) {
            y_coordinate_2d = j * padding_factor;
            x_coordinate_2d = 0;
            if (powf(y_coordinate_2d, 2) <= resolution_limit_sq) {
                rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d,
                                                           y_coordinate_3d);
                pixel_counter = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
                rotated_image.complex_values[pixel_counter] =
                    ReturnLinearInterpolatedFourier2D(x_coordinate_3d, y_coordinate_3d);
                pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
                rotated_image.complex_values[pixel_counter2] = conj(rotated_image.complex_values[pixel_counter]);
            }
            else {
                pixel_counter = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
                rotated_image.complex_values[pixel_counter] = 0.0f + I * 0.0f;
                pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
                rotated_image.complex_values[pixel_counter2] = 0.0f + I * 0.0f;
            }
        }
        // Deal with pixel at edge if image dimensions are even
        if (-rotated_image.logical_lower_bound_complex_y != rotated_image.logical_upper_bound_complex_y) {
            y_coordinate_2d = rotated_image.logical_lower_bound_complex_y * padding_factor;
            x_coordinate_2d = 0;
            rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d,
                                                       y_coordinate_3d);
            pixel_counter =
                rotated_image.ReturnFourier1DAddressFromLogicalCoord(0, rotated_image.logical_lower_bound_complex_y, 0);
            rotated_image.complex_values[pixel_counter] =
                ReturnLinearInterpolatedFourier2D(x_coordinate_3d, y_coordinate_3d);
        }
    }

    // Set origin to zero to generate a projection with average set to zero
    rotated_image.complex_values[0] = 0.0f + I * 0.0f;

    rotated_image.is_in_real_space         = false;
    rotated_image.object_is_centred_in_box = false;
}

void Image::ExtractSlice(Image& image_to_extract, AnglesAndShifts& angles_and_shifts_of_image, float resolution_limit,
                         bool apply_resolution_limit) {
    //	MyDebugAssertTrue(image_to_extract.logical_x_dimension == logical_x_dimension &&
    //image_to_extract.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
    MyDebugAssertTrue(image_to_extract.logical_z_dimension == 1,
                      "Error: attempting to extract 3D image from 3D reconstruction");
    MyDebugAssertTrue(image_to_extract.is_in_memory, "Memory not allocated for receiving image");
    MyDebugAssertTrue(IsCubic(), "Image volume to project is not cubic");
    MyDebugAssertFalse(object_is_centred_in_box, "Image volume quadrants not swapped");

    int i;
    int j;

    long pixel_counter;
    long pixel_counter2;

    float x_coordinate_2d;
    float y_coordinate_2d;
    float z_coordinate_2d = 0.0;

    float x_coordinate_3d;
    float y_coordinate_3d;
    float z_coordinate_3d;

    float resolution_limit_sq = powf(resolution_limit * logical_x_dimension, 2);
    float y_coord_sq;

    bool padding = false;

    Image* temp_image;

    if (image_to_extract.logical_x_dimension != logical_x_dimension ||
        image_to_extract.logical_y_dimension != logical_y_dimension)
        padding = true;

    if (!padding) {
        temp_image = &image_to_extract;
    }
    else {
        temp_image = new Image;
        temp_image->Allocate(logical_x_dimension, logical_y_dimension, false);
    }

    temp_image->object_is_centred_in_box = false;
    temp_image->is_in_real_space         = false;
    //	temp_image->SetToConstant(0.0);

    if (apply_resolution_limit) {
        for (j = temp_image->logical_lower_bound_complex_y; j <= temp_image->logical_upper_bound_complex_y; j++) {
            y_coordinate_2d = j;
            y_coord_sq      = powf(y_coordinate_2d, 2);
            for (i = 1; i <= temp_image->logical_upper_bound_complex_x; i++) {
                x_coordinate_2d = i;
                pixel_counter   = temp_image->ReturnFourier1DAddressFromLogicalCoord(i, j, 0);
                if (powf(x_coordinate_2d, 2) + y_coord_sq <= resolution_limit_sq) {
                    angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d,
                                                                         z_coordinate_2d, x_coordinate_3d,
                                                                         y_coordinate_3d, z_coordinate_3d);
                    temp_image->complex_values[pixel_counter] =
                        ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
                }
                else {
                    temp_image->complex_values[pixel_counter] = 0.0f + I * 0.0f;
                }
            }
        }
        // Now deal with special case of i = 0
        for (j = 1; j <= temp_image->logical_upper_bound_complex_y; j++) {
            y_coordinate_2d = j;
            x_coordinate_2d = 0;
            if (powf(y_coordinate_2d, 2) <= resolution_limit_sq) {
                angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d,
                                                                     x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
                pixel_counter = temp_image->ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
                temp_image->complex_values[pixel_counter] =
                    ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
                pixel_counter2 = temp_image->ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
                temp_image->complex_values[pixel_counter2] = conj(temp_image->complex_values[pixel_counter]);
            }
            else {
                pixel_counter                             = temp_image->ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
                temp_image->complex_values[pixel_counter] = 0.0f + I * 0.0f;
                pixel_counter2 = temp_image->ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
                temp_image->complex_values[pixel_counter2] = 0.0f + I * 0.0f;
            }
        }
        // Deal with pixel at edge if image dimensions are even
        if (-temp_image->logical_lower_bound_complex_y != temp_image->logical_upper_bound_complex_y) {
            y_coordinate_2d = temp_image->logical_lower_bound_complex_y;
            x_coordinate_2d = 0;
            if (powf(y_coordinate_2d, 2) <= resolution_limit_sq) {
                angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d,
                                                                     x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
                pixel_counter =
                    temp_image->ReturnFourier1DAddressFromLogicalCoord(0, temp_image->logical_lower_bound_complex_y, 0);
                temp_image->complex_values[pixel_counter] =
                    ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
            }
            else {
                pixel_counter =
                    temp_image->ReturnFourier1DAddressFromLogicalCoord(0, temp_image->logical_lower_bound_complex_y, 0);
                temp_image->complex_values[pixel_counter] = 0.0f + I * 0.0f;
            }
        }
    }
    else {
        for (j = temp_image->logical_lower_bound_complex_y; j <= temp_image->logical_upper_bound_complex_y; j++) {
            y_coordinate_2d = j;
            for (i = 1; i <= temp_image->logical_upper_bound_complex_x; i++) {
                x_coordinate_2d = i;
                pixel_counter   = temp_image->ReturnFourier1DAddressFromLogicalCoord(i, j, 0);
                angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d,
                                                                     x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
                temp_image->complex_values[pixel_counter] =
                    ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
            }
        }
        // Now deal with special case of i = 0
        for (j = 1; j <= temp_image->logical_upper_bound_complex_y; j++) {
            y_coordinate_2d = j;
            x_coordinate_2d = 0;
            angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d,
                                                                 x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
            pixel_counter = temp_image->ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
            temp_image->complex_values[pixel_counter] =
                ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
            pixel_counter2                             = temp_image->ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
            temp_image->complex_values[pixel_counter2] = conj(temp_image->complex_values[pixel_counter]);
        }
        // Deal with pixel at edge if image dimensions are even
        if (-temp_image->logical_lower_bound_complex_y != temp_image->logical_upper_bound_complex_y) {
            y_coordinate_2d = temp_image->logical_lower_bound_complex_y;
            x_coordinate_2d = 0;
            angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d,
                                                                 x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
            pixel_counter =
                temp_image->ReturnFourier1DAddressFromLogicalCoord(0, temp_image->logical_lower_bound_complex_y, 0);
            temp_image->complex_values[pixel_counter] =
                ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
        }
    }

    // Set origin to zero to generate a projection with average set to zero
    temp_image->complex_values[0] = 0.0f + I * 0.0f;
    // This was changed to make projections compatible with ML algorithm
    //	temp_image->complex_values[0] = complex_values[0];

    temp_image->is_in_real_space = false;

    if (padding) {
        temp_image->SwapRealSpaceQuadrants();
        temp_image->BackwardFFT();
        if (temp_image->logical_x_dimension < image_to_extract.logical_x_dimension &&
            temp_image->logical_y_dimension < image_to_extract.logical_y_dimension)
            temp_image->ClipIntoLargerRealSpace2D(&image_to_extract);
        else
            temp_image->ClipInto(&image_to_extract);
        image_to_extract.ForwardFFT();
        image_to_extract.SwapRealSpaceQuadrants();
        delete temp_image;
    }
}

void Image::ExtractSliceByRotMatrix(Image& image_to_extract, RotationMatrix& wanted_matrix, float resolution_limit,
                                    bool apply_resolution_limit) {
    //	MyDebugAssertTrue(image_to_extract.logical_x_dimension == logical_x_dimension &&
    //image_to_extract.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
    MyDebugAssertTrue(image_to_extract.logical_z_dimension == 1,
                      "Error: attempting to extract 3D image from 3D reconstruction");
    MyDebugAssertTrue(image_to_extract.is_in_memory, "Memory not allocated for receiving image");
    MyDebugAssertTrue(IsCubic(), "Image volume to project is not cubic");
    MyDebugAssertTrue(!object_is_centred_in_box, "Image volume quadrants not swapped");

    int i;
    int j;

    long pixel_counter;
    long pixel_counter2;

    float x_coordinate_2d;
    float y_coordinate_2d;
    float z_coordinate_2d = 0.0;

    float x_coordinate_3d;
    float y_coordinate_3d;
    float z_coordinate_3d;

    float resolution_limit_sq = powf(resolution_limit * logical_x_dimension, 2);
    float y_coord_sq;

    bool padding = false;

    Image* temp_image;

    if (image_to_extract.logical_x_dimension != logical_x_dimension ||
        image_to_extract.logical_y_dimension != logical_y_dimension)
        padding = true;

    if (!padding) {
        temp_image = &image_to_extract;
    }
    else {
        temp_image = new Image;
        temp_image->Allocate(logical_x_dimension, logical_y_dimension, false);
    }

    temp_image->object_is_centred_in_box = false;
    temp_image->is_in_real_space         = false;
    //	temp_image->SetToConstant(0.0);

    if (apply_resolution_limit) {
        for (j = temp_image->logical_lower_bound_complex_y; j <= temp_image->logical_upper_bound_complex_y; j++) {
            y_coordinate_2d = j;
            y_coord_sq      = powf(y_coordinate_2d, 2);
            for (i = 1; i <= temp_image->logical_upper_bound_complex_x; i++) {
                x_coordinate_2d = i;
                pixel_counter   = temp_image->ReturnFourier1DAddressFromLogicalCoord(i, j, 0);
                if (powf(x_coordinate_2d, 2) + y_coord_sq <= resolution_limit_sq) {
                    wanted_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d,
                                               y_coordinate_3d, z_coordinate_3d);
                    temp_image->complex_values[pixel_counter] =
                        ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
                }
                else {
                    temp_image->complex_values[pixel_counter] = 0.0f + I * 0.0f;
                }
            }
        }
        // Now deal with special case of i = 0
        for (j = 1; j <= temp_image->logical_upper_bound_complex_y; j++) {
            y_coordinate_2d = j;
            x_coordinate_2d = 0;
            if (powf(y_coordinate_2d, 2) <= resolution_limit_sq) {
                wanted_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d,
                                           y_coordinate_3d, z_coordinate_3d);
                pixel_counter = temp_image->ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
                temp_image->complex_values[pixel_counter] =
                    ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
                pixel_counter2 = temp_image->ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
                temp_image->complex_values[pixel_counter2] = conj(temp_image->complex_values[pixel_counter]);
            }
            else {
                pixel_counter                             = temp_image->ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
                temp_image->complex_values[pixel_counter] = 0.0f + I * 0.0f;
                pixel_counter2 = temp_image->ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
                temp_image->complex_values[pixel_counter2] = 0.0f + I * 0.0f;
            }
        }
        // Deal with pixel at edge if image dimensions are even
        if (-temp_image->logical_lower_bound_complex_y != temp_image->logical_upper_bound_complex_y) {
            y_coordinate_2d = temp_image->logical_lower_bound_complex_y;
            x_coordinate_2d = 0;
            if (powf(y_coordinate_2d, 2) <= resolution_limit_sq) {
                wanted_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d,
                                           y_coordinate_3d, z_coordinate_3d);
                pixel_counter =
                    temp_image->ReturnFourier1DAddressFromLogicalCoord(0, temp_image->logical_lower_bound_complex_y, 0);
                temp_image->complex_values[pixel_counter] =
                    ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
            }
            else {
                pixel_counter =
                    temp_image->ReturnFourier1DAddressFromLogicalCoord(0, temp_image->logical_lower_bound_complex_y, 0);
                temp_image->complex_values[pixel_counter] = 0.0f + I * 0.0f;
            }
        }
    }
    else {
        for (j = temp_image->logical_lower_bound_complex_y; j <= temp_image->logical_upper_bound_complex_y; j++) {
            y_coordinate_2d = j;
            for (i = 1; i <= temp_image->logical_upper_bound_complex_x; i++) {
                x_coordinate_2d = i;
                pixel_counter   = temp_image->ReturnFourier1DAddressFromLogicalCoord(i, j, 0);
                wanted_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d,
                                           y_coordinate_3d, z_coordinate_3d);
                temp_image->complex_values[pixel_counter] =
                    ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
            }
        }
        // Now deal with special case of i = 0
        for (j = 1; j <= temp_image->logical_upper_bound_complex_y; j++) {
            y_coordinate_2d = j;
            x_coordinate_2d = 0;
            wanted_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d,
                                       y_coordinate_3d, z_coordinate_3d);
            pixel_counter = temp_image->ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
            temp_image->complex_values[pixel_counter] =
                ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
            pixel_counter2                             = temp_image->ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
            temp_image->complex_values[pixel_counter2] = conj(temp_image->complex_values[pixel_counter]);
        }
        // Deal with pixel at edge if image dimensions are even
        if (-temp_image->logical_lower_bound_complex_y != temp_image->logical_upper_bound_complex_y) {
            y_coordinate_2d = temp_image->logical_lower_bound_complex_y;
            x_coordinate_2d = 0;
            wanted_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d,
                                       y_coordinate_3d, z_coordinate_3d);
            pixel_counter =
                temp_image->ReturnFourier1DAddressFromLogicalCoord(0, temp_image->logical_lower_bound_complex_y, 0);
            temp_image->complex_values[pixel_counter] =
                ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
        }
    }

    // Set origin to zero to generate a projection with average set to zero
    temp_image->complex_values[0] = 0.0f + I * 0.0f;
    // This was changed to make projections compatible with ML algorithm
    //	temp_image->complex_values[0] = complex_values[0];

    temp_image->is_in_real_space = false;

    if (padding) {
        temp_image->SwapRealSpaceQuadrants();
        temp_image->BackwardFFT();
        if (temp_image->logical_x_dimension < image_to_extract.logical_x_dimension &&
            temp_image->logical_y_dimension < image_to_extract.logical_y_dimension)
            temp_image->ClipIntoLargerRealSpace2D(&image_to_extract);
        else
            temp_image->ClipInto(&image_to_extract);
        image_to_extract.ForwardFFT();
        image_to_extract.SwapRealSpaceQuadrants();
        delete temp_image;
    }
}

std::complex<float> Image::ReturnNearestFourier2D(float& x, float& y) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(!is_in_real_space, "Must be in Fourier space");

    int i_nearest;
    int j_nearest;
    int physical_y_address;

    if (x >= 0.0) {
        i_nearest = myroundint(x);
        if (i_nearest > logical_upper_bound_complex_x)
            return 0.0f + I * 0.0f;

        j_nearest = myroundint(y);
        if (j_nearest < logical_lower_bound_complex_y)
            return 0.0f + I * 0.0f;
        if (j_nearest > logical_upper_bound_complex_y)
            return 0.0f + I * 0.0f;

        if (j_nearest >= 0) {
            physical_y_address = j_nearest;
        }
        else {
            physical_y_address = logical_y_dimension + j_nearest;
        }
        return complex_values[(physical_upper_bound_complex_x + 1) * physical_y_address + i_nearest];
    }
    else {
        i_nearest = myroundint(x);
        if (i_nearest < logical_lower_bound_complex_x)
            return 0.0f + I * 0.0f;

        j_nearest = myroundint(y);
        if (j_nearest < logical_lower_bound_complex_y)
            return 0.0f + I * 0.0f;
        if (j_nearest > logical_upper_bound_complex_y)
            return 0.0f + I * 0.0f;

        if (j_nearest > 0) {
            physical_y_address = logical_y_dimension - j_nearest;
        }
        else {
            physical_y_address = -j_nearest;
        }
        return conj(complex_values[(physical_upper_bound_complex_x + 1) * physical_y_address - i_nearest]);
    }
}

// Implementation of Frealign's ainterpo3ds
std::complex<float> Image::ReturnLinearInterpolatedFourier2D(float& x, float& y) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(!is_in_real_space, "Must be in Fourier space");

    std::complex<float> sum = 0.0f + I * 0.0f;
    int                 i;
    int                 j;
    int                 i_start;
    int                 i_end;
    int                 j_start;
    int                 j_end;
    int                 physical_y_address;
    int                 jj;

    float weight;
    float y_dist;

    if (x >= 0.0) {
        i_start = int(floorf(x));
        i_end   = i_start + 1;
        if (i_end > logical_upper_bound_complex_x)
            return 0.0f + I * 0.0f;

        j_start = int(floorf(y));
        if (j_start < logical_lower_bound_complex_y)
            return 0.0f + I * 0.0f;
        j_end = j_start + 1;
        if (j_end > logical_upper_bound_complex_y)
            return 0.0f + I * 0.0f;

        for (j = j_start; j <= j_end; j++) {
            if (j >= 0) {
                physical_y_address = j;
            }
            else {
                physical_y_address = logical_y_dimension + j;
            }
            jj     = (physical_upper_bound_complex_x + 1) * physical_y_address;
            y_dist = fabsf(y - float(j));
            for (i = i_start; i <= i_end; i++) {
                weight = (1.0 - fabsf(x - float(i))) * (1.0 - y_dist);
                sum    = sum + complex_values[jj + i] * weight;
            }
        }
    }
    else {
        i_start = int(floorf(x));
        if (i_start < logical_lower_bound_complex_x)
            return 0.0f + I * 0.0f;
        i_end = i_start + 1;

        j_start = int(floorf(y));
        if (j_start < logical_lower_bound_complex_y)
            return 0.0f + I * 0.0f;
        j_end = j_start + 1;
        if (j_end > logical_upper_bound_complex_y)
            return 0.0f + I * 0.0f;

        for (j = j_start; j <= j_end; j++) {
            if (j > 0) {
                physical_y_address = logical_y_dimension - j;
            }
            else {
                physical_y_address = -j;
            }
            jj     = (physical_upper_bound_complex_x + 1) * physical_y_address;
            y_dist = fabsf(y - float(j));
            for (i = i_start; i <= i_end; i++) {
                weight = (1.0 - fabsf(x - float(i))) * (1.0 - y_dist);
                sum    = sum + conj(complex_values[jj - i]) * weight;
            }
        }
    }
    return sum;
}

std::complex<float> Image::ReturnLinearInterpolatedFourier(float& x, float& y, float& z) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(!is_in_real_space, "Must be in Fourier space");

    std::complex<float> sum = 0.0f + I * 0.0f;
    int                 i;
    int                 j;
    int                 k;
    int                 i_start;
    int                 i_end;
    int                 j_start;
    int                 j_end;
    int                 k_start;
    int                 k_end;
    int                 physical_y_address;
    int                 physical_z_address;
    int                 jj;
    int                 kk;

    float weight;
    float y_dist;
    float z_dist;

    if (x >= 0.0) {
        i_start = int(floorf(x));
        i_end   = i_start + 1;
        if (i_end > logical_upper_bound_complex_x)
            return 0.0f + I * 0.0f;

        j_start = int(floorf(y));
        if (j_start < logical_lower_bound_complex_y)
            return 0.0f + I * 0.0f;
        j_end = j_start + 1;
        if (j_end > logical_upper_bound_complex_y)
            return 0.0f + I * 0.0f;

        k_start = int(floorf(z));
        if (k_start < logical_lower_bound_complex_z)
            return 0.0f + I * 0.0f;
        k_end = k_start + 1;
        if (k_end > logical_upper_bound_complex_z)
            return 0.0f + I * 0.0f;

        for (k = k_start; k <= k_end; k++) {
            if (k >= 0) {
                physical_z_address = k;
            }
            else {
                physical_z_address = logical_z_dimension + k;
            }
            kk     = (physical_upper_bound_complex_y + 1) * physical_z_address;
            z_dist = fabsf(z - float(k));
            for (j = j_start; j <= j_end; j++) {
                if (j >= 0) {
                    physical_y_address = j;
                }
                else {
                    physical_y_address = logical_y_dimension + j;
                }
                jj     = (physical_upper_bound_complex_x + 1) * (kk + physical_y_address);
                y_dist = fabsf(y - float(j));
                for (i = i_start; i <= i_end; i++) {
                    weight = (1.0 - fabsf(x - float(i))) * (1.0 - y_dist) * (1.0 - z_dist);
                    sum    = sum + complex_values[jj + i] * weight;
                }
            }
        }
    }
    else {
        i_start = int(floorf(x));
        if (i_start < logical_lower_bound_complex_x)
            return 0.0f + I * 0.0f;
        i_end = i_start + 1;

        j_start = int(floorf(y));
        if (j_start < logical_lower_bound_complex_y)
            return 0.0f + I * 0.0f;
        j_end = j_start + 1;
        if (j_end > logical_upper_bound_complex_y)
            return 0.0f + I * 0.0f;

        k_start = int(floorf(z));
        if (k_start < logical_lower_bound_complex_z)
            return 0.0f + I * 0.0f;
        k_end = k_start + 1;
        if (k_end > logical_upper_bound_complex_z)
            return 0.0f + I * 0.0f;

        for (k = k_start; k <= k_end; k++) {
            if (k > 0) {
                physical_z_address = logical_z_dimension - k;
            }
            else {
                physical_z_address = -k;
            }
            kk     = (physical_upper_bound_complex_y + 1) * physical_z_address;
            z_dist = fabsf(z - float(k));
            for (j = j_start; j <= j_end; j++) {
                if (j > 0) {
                    physical_y_address = logical_y_dimension - j;
                }
                else {
                    physical_y_address = -j;
                }
                jj     = (physical_upper_bound_complex_x + 1) * (kk + physical_y_address);
                y_dist = fabsf(y - float(j));
                for (i = i_start; i <= i_end; i++) {
                    weight = (1.0 - fabsf(x - float(i))) * (1.0 - y_dist) * (1.0 - z_dist);
                    sum    = sum + conj(complex_values[jj - i]) * weight;
                }
            }
        }
    }
    return sum;
}

void Image::AddByLinearInterpolationReal(float& wanted_physical_x_coordinate, float& wanted_physical_y_coordinate,
                                         float& wanted_physical_z_coordinate, float& wanted_value) {
    int i;
    int j;
    int k;
    int int_x_coordinate;
    int int_y_coordinate;
    int int_z_coordinate;

    long physical_coord;

    float weight_x;
    float weight_y;
    float weight_z;

    MyDebugAssertTrue(is_in_real_space == true, "Error: attempting REAL insertion into COMPLEX image");
    MyDebugAssertTrue(wanted_physical_x_coordinate >= 0 && wanted_physical_x_coordinate < logical_x_dimension,
                      "Error: attempting insertion outside image X boundaries (%.2f)", wanted_physical_x_coordinate);
    MyDebugAssertTrue(wanted_physical_y_coordinate >= 0 && wanted_physical_y_coordinate < logical_y_dimension,
                      "Error: attempting insertion outside image Y boundaries (%.2f)", wanted_physical_y_coordinate);
    MyDebugAssertTrue(wanted_physical_z_coordinate >= 0 && wanted_physical_z_coordinate < logical_z_dimension,
                      "Error: attempting insertion outside image Z boundaries (%.2f)", wanted_physical_z_coordinate);

    int_x_coordinate = int(wanted_physical_x_coordinate);
    int_y_coordinate = int(wanted_physical_y_coordinate);
    int_z_coordinate = int(wanted_physical_z_coordinate);

    for (k = int_z_coordinate; k <= int_z_coordinate + 1; k++) {
        weight_z = (1.0 - fabsf(wanted_physical_z_coordinate - k));
        for (j = int_y_coordinate; j <= int_y_coordinate + 1; j++) {
            weight_y = (1.0 - fabsf(wanted_physical_y_coordinate - j));
            for (i = int_x_coordinate; i <= int_x_coordinate + 1; i++) {
                weight_x       = (1.0 - fabsf(wanted_physical_x_coordinate - i));
                physical_coord = ReturnReal1DAddressFromPhysicalCoord(i, j, k);
                real_values[physical_coord] =
                    real_values[physical_coord] + wanted_value * weight_x * weight_y * weight_z;
            }
        }
    }
}

void Image::AddByLinearInterpolationFourier2D(float& wanted_logical_x_coordinate, float& wanted_logical_y_coordinate,
                                              std::complex<float>& wanted_value) {
    int i;
    int j;
    int int_x_coordinate;
    int int_y_coordinate;

    long physical_coord;

    float weight_y;
    float weight;

    std::complex<float> conjugate;

    MyDebugAssertTrue(is_in_real_space != true, "Error: attempting COMPLEX insertion into REAL image");

    int_x_coordinate = int(floor(wanted_logical_x_coordinate));
    int_y_coordinate = int(floor(wanted_logical_y_coordinate));

    for (j = int_y_coordinate; j <= int_y_coordinate + 1; j++) {
        weight_y = (1.0 - fabsf(wanted_logical_y_coordinate - j));
        for (i = int_x_coordinate; i <= int_x_coordinate + 1; i++) {
            if (i >= logical_lower_bound_complex_x && i <= logical_upper_bound_complex_x &&
                j >= logical_lower_bound_complex_y && j <= logical_upper_bound_complex_y) {
                weight         = weight_y * (1.0 - fabsf(wanted_logical_x_coordinate - i));
                physical_coord = ReturnFourier1DAddressFromLogicalCoord(i, j, 0);
                if (i < 0) {
                    conjugate                      = conj(wanted_value);
                    complex_values[physical_coord] = complex_values[physical_coord] + conjugate * weight;
                }
                else if (i == 0 && j != logical_lower_bound_complex_y && j != 0) {
                    complex_values[physical_coord] = complex_values[physical_coord] + wanted_value * weight;
                    physical_coord                 = ReturnFourier1DAddressFromLogicalCoord(i, -j, 0);
                    conjugate                      = conj(wanted_value);
                    complex_values[physical_coord] = complex_values[physical_coord] + conjugate * weight;
                }
                else {
                    complex_values[physical_coord] = complex_values[physical_coord] + wanted_value * weight;
                }
            }
        }
    }
}

void Image::CalculateCTFImage(CTF& ctf_of_image, bool calculate_complex_ctf, bool apply_coherence_envelope) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated for CTF image");
    if (apply_coherence_envelope) {
        MyDebugAssertFalse(calculate_complex_ctf,
                           "calculating a complex CTF and a coherence envelope is not supported.");
    }
    //	MyDebugAssertTrue(is_in_real_space == false, "CTF image not in Fourier space");

    int i;
    int j;

    long pixel_counter = 0;

    float x_coordinate_2d;
    float y_coordinate_2d;

    float y_coord_sq;

    float frequency_squared;
    float azimuth;

    for (j = 0; j <= physical_upper_bound_complex_y; j++) {
        y_coordinate_2d = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y;
        y_coord_sq      = powf(y_coordinate_2d, 2);

        for (i = 0; i <= physical_upper_bound_complex_x; i++) {
            x_coordinate_2d = i * fourier_voxel_size_x;
            // Compute the azimuth
            if (i == 0 && j == 0) {
                azimuth = 0.0;
            }
            else {
                azimuth = atan2f(y_coordinate_2d, x_coordinate_2d);
            }
            // Compute the square of the frequency
            frequency_squared = powf(x_coordinate_2d, 2) + y_coord_sq;

            if (calculate_complex_ctf) {
                complex_values[pixel_counter] = ctf_of_image.EvaluateComplex(frequency_squared, azimuth);
            }
            else {
                if (apply_coherence_envelope)
                    complex_values[pixel_counter] =
                        ctf_of_image.EvaluateWithEnvelope(frequency_squared, azimuth) + I * 0.0f;
                else
                    complex_values[pixel_counter] = ctf_of_image.Evaluate(frequency_squared, azimuth) + I * 0.0f;
            }
            pixel_counter++;
        }
    }

    is_in_real_space = false;
}

void Image::CalculateBeamTiltImage(CTF& ctf_of_image, bool output_phase_shifts) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated for beam tilt image");
    //	MyDebugAssertTrue(is_in_real_space == false, "beam tilt image not in Fourier space");

    int i;
    int j;

    long pixel_counter = 0;

    float x_coordinate_2d;
    float y_coordinate_2d;

    float y_coord_sq;

    float frequency_squared;
    float azimuth;
    float phase_shift;

    for (j = 0; j <= physical_upper_bound_complex_y; j++) {
        y_coordinate_2d = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y;
        y_coord_sq      = powf(y_coordinate_2d, 2);

        for (i = 0; i <= physical_upper_bound_complex_x; i++) {
            x_coordinate_2d = i * fourier_voxel_size_x;
            // Compute the azimuth
            if (i == 0 && j == 0)
                azimuth = 0.0;
            else
                azimuth = atan2f(y_coordinate_2d, x_coordinate_2d);
            // Compute the square of the frequency
            frequency_squared = powf(x_coordinate_2d, 2) + y_coord_sq;

            //			phase_shift = 2.0f * PI * (x_coordinate_2d * image_shift_x + y_coordinate_2d * image_shift_y);

            if (output_phase_shifts) {
                phase_shift = ctf_of_image.PhaseShiftGivenBeamTiltAndShift(
                    frequency_squared, ctf_of_image.BeamTiltGivenAzimuth(azimuth),
                    ctf_of_image.ParticleShiftGivenAzimuth(azimuth));
                complex_values[pixel_counter] = phase_shift + I * 0.0f;
            }
            else
                complex_values[pixel_counter] = ctf_of_image.EvaluateBeamTiltPhaseShift(frequency_squared, azimuth);
            //			else complex_values[pixel_counter] = ctf_of_image.EvaluateBeamTiltPhaseShift(frequency_squared,
            //azimuth) * (cosf( phase_shift ) + I * sinf( phase_shift ));

            pixel_counter++;
        }
    }

    is_in_real_space = false;
}

// Apply a cosine-edge mask. By default, pixels on the outside of the mask radius are flattened. If invert=true, the
// pixels near the center are flattened. This does not currently work when quadrants are swapped.
float Image::CosineRingMask(float wanted_inner_radius, float wanted_outer_radius, float wanted_mask_edge) {
    //	MyDebugAssertTrue(! is_in_real_space || object_is_centred_in_box, "Image in real space but not centered");
    MyDebugAssertTrue(wanted_inner_radius <= wanted_outer_radius, "Inner radius larger than outer radius");

    // If wanted_inner_radius < 0 only mask the outside.
    bool do_inner_masking = true;
    if (wanted_inner_radius < 0.0f) {
        do_inner_masking = false;
    }

    int  i;
    int  j;
    int  k;
    int  ii;
    int  jj;
    int  kk;
    long outer_number_of_pixels;
    long inner_number_of_pixels;

    float x;
    float y;
    float z;

    long pixel_counter = 0;

    float  distance_from_center;
    float  distance_from_center_squared;
    float  outer_mask_radius;
    float  outer_mask_radius_squared;
    float  outer_mask_radius_plus_edge;
    float  outer_mask_radius_plus_edge_squared;
    float  inner_mask_radius;
    float  inner_mask_radius_squared;
    float  inner_mask_radius_minus_edge;
    float  inner_mask_radius_minus_edge_squared;
    float  edge;
    double outer_pixel_sum;
    double inner_pixel_sum;

    float frequency;
    float frequency_squared;

    double mask_volume = 0.0;

    outer_mask_radius = wanted_outer_radius - wanted_mask_edge / 2;
    if (outer_mask_radius < 0.0)
        outer_mask_radius = 0.0;
    outer_mask_radius_plus_edge = outer_mask_radius + wanted_mask_edge;

    outer_mask_radius_squared           = powf(outer_mask_radius, 2);
    outer_mask_radius_plus_edge_squared = powf(outer_mask_radius_plus_edge, 2);

    inner_mask_radius                    = wanted_inner_radius + wanted_mask_edge / 2;
    inner_mask_radius_minus_edge         = inner_mask_radius - wanted_mask_edge;
    inner_mask_radius_squared            = powf(inner_mask_radius, 2);
    inner_mask_radius_minus_edge_squared = powf(inner_mask_radius_minus_edge, 2);

    outer_pixel_sum        = 0.0;
    outer_number_of_pixels = 0;
    inner_pixel_sum        = 0.0;
    inner_number_of_pixels = 0;
    if (is_in_real_space && object_is_centred_in_box) {
        MyDebugAssertTrue(wanted_mask_edge > 1, "Edge width too small: %f\n", wanted_mask_edge);
        for (k = 0; k < logical_z_dimension; k++) {
            z = powf(k - physical_address_of_box_center_z, 2);

            for (j = 0; j < logical_y_dimension; j++) {
                y = powf(j - physical_address_of_box_center_y, 2);

                for (i = 0; i < logical_x_dimension; i++) {
                    x = powf(i - physical_address_of_box_center_x, 2);

                    distance_from_center_squared = x + y + z;

                    if (distance_from_center_squared >= outer_mask_radius_squared &&
                        distance_from_center_squared <= outer_mask_radius_plus_edge_squared) {
                        outer_pixel_sum += real_values[pixel_counter];
                        outer_number_of_pixels++;
                    }
                    if (distance_from_center_squared <= inner_mask_radius_squared &&
                        distance_from_center_squared >= inner_mask_radius_minus_edge_squared) {
                        inner_pixel_sum += real_values[pixel_counter];
                        inner_number_of_pixels++;
                    }

                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
        outer_pixel_sum /= outer_number_of_pixels;
        inner_pixel_sum /= inner_number_of_pixels;

        pixel_counter = 0.0;
        for (k = 0; k < logical_z_dimension; k++) {
            z = powf(k - physical_address_of_box_center_z, 2);

            for (j = 0; j < logical_y_dimension; j++) {
                y = powf(j - physical_address_of_box_center_y, 2);

                for (i = 0; i < logical_x_dimension; i++) {
                    x = powf(i - physical_address_of_box_center_x, 2);

                    distance_from_center_squared = x + y + z;

                    if (distance_from_center_squared >= outer_mask_radius_squared &&
                        distance_from_center_squared <= outer_mask_radius_plus_edge_squared) {
                        distance_from_center = sqrtf(distance_from_center_squared);
                        edge = (1.0 + cosf(PI * (distance_from_center - outer_mask_radius) / wanted_mask_edge)) / 2.0;
                        real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * outer_pixel_sum;
                        mask_volume += powf(edge, 2);
                    }
                    else if (distance_from_center_squared >= outer_mask_radius_plus_edge_squared) {
                        real_values[pixel_counter] = outer_pixel_sum;
                    }
                    else if (distance_from_center_squared <= inner_mask_radius_squared &&
                             distance_from_center_squared >= inner_mask_radius_minus_edge_squared &&
                             wanted_inner_radius > 0.0) {
                        distance_from_center = sqrtf(distance_from_center_squared);
                        edge = (1.0 + cosf(PI * (inner_mask_radius - distance_from_center) / wanted_mask_edge)) / 2.0;
                        real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * inner_pixel_sum;
                        mask_volume += powf(edge, 2);
                    }
                    else if (distance_from_center_squared <= inner_mask_radius_minus_edge_squared &&
                             wanted_inner_radius > 0.0) {
                        real_values[pixel_counter] = inner_pixel_sum;
                    }
                    else {
                        mask_volume += 1.0;
                    }

                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }
    else if (is_in_real_space) {
        MyDebugAssertTrue(wanted_mask_edge > 1, "Edge width too small: %f\n", wanted_mask_edge);
        for (k = 0; k < logical_z_dimension; k++) {
            kk = k;
            if (kk >= physical_address_of_box_center_z)
                kk -= logical_z_dimension;
            z = powf(kk, 2);

            for (j = 0; j < logical_y_dimension; j++) {
                jj = j;
                if (jj >= physical_address_of_box_center_y)
                    jj -= logical_y_dimension;
                y = powf(jj, 2);

                for (i = 0; i < logical_x_dimension; i++) {
                    ii = i;
                    if (ii >= physical_address_of_box_center_x)
                        ii -= logical_x_dimension;
                    x = powf(ii, 2);

                    distance_from_center_squared = x + y + z;

                    if (distance_from_center_squared >= outer_mask_radius_squared &&
                        distance_from_center_squared <= outer_mask_radius_plus_edge_squared) {
                        outer_pixel_sum += real_values[pixel_counter];
                        outer_number_of_pixels++;
                    }
                    if (distance_from_center_squared <= inner_mask_radius_squared &&
                        distance_from_center_squared >= inner_mask_radius_minus_edge_squared) {
                        inner_pixel_sum += real_values[pixel_counter];
                        inner_number_of_pixels++;
                    }

                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
        outer_pixel_sum /= outer_number_of_pixels;
        inner_pixel_sum /= inner_number_of_pixels;

        pixel_counter = 0.0;
        for (k = 0; k < logical_z_dimension; k++) {
            kk = k;
            if (kk >= physical_address_of_box_center_z)
                kk -= logical_z_dimension;
            z = powf(kk, 2);

            for (j = 0; j < logical_y_dimension; j++) {
                jj = j;
                if (jj >= physical_address_of_box_center_y)
                    jj -= logical_y_dimension;
                y = powf(jj, 2);

                for (i = 0; i < logical_x_dimension; i++) {
                    ii = i;
                    if (ii >= physical_address_of_box_center_x)
                        ii -= logical_x_dimension;
                    x = powf(ii, 2);

                    distance_from_center_squared = x + y + z;

                    if (distance_from_center_squared >= outer_mask_radius_squared &&
                        distance_from_center_squared <= outer_mask_radius_plus_edge_squared) {
                        distance_from_center = sqrtf(distance_from_center_squared);
                        edge = (1.0 + cosf(PI * (distance_from_center - outer_mask_radius) / wanted_mask_edge)) / 2.0;
                        real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * outer_pixel_sum;
                        mask_volume += powf(edge, 2);
                    }
                    else if (distance_from_center_squared >= outer_mask_radius_plus_edge_squared) {
                        real_values[pixel_counter] = outer_pixel_sum;
                    }
                    else if (distance_from_center_squared <= inner_mask_radius_squared &&
                             distance_from_center_squared >= inner_mask_radius_minus_edge_squared &&
                             wanted_inner_radius > 0.0) {
                        distance_from_center = sqrtf(distance_from_center_squared);
                        edge = (1.0 + cosf(PI * (inner_mask_radius - distance_from_center) / wanted_mask_edge)) / 2.0;
                        real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * inner_pixel_sum;
                        mask_volume += powf(edge, 2);
                    }
                    else if (distance_from_center_squared <= inner_mask_radius_minus_edge_squared &&
                             wanted_inner_radius > 0.0) {
                        real_values[pixel_counter] = inner_pixel_sum;
                    }
                    else {
                        mask_volume += 1.0;
                    }

                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }
    else {
        for (k = 0; k <= physical_upper_bound_complex_z; k++) {
            z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

            for (j = 0; j <= physical_upper_bound_complex_y; j++) {
                y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

                for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                    x = powf(i * fourier_voxel_size_x, 2);

                    // compute squared radius, in units of reciprocal pixels

                    frequency_squared = x + y + z;

                    if (frequency_squared >= outer_mask_radius_squared &&
                        frequency_squared <= outer_mask_radius_plus_edge_squared) {
                        frequency = sqrtf(frequency_squared);
                        edge      = (1.0 + cosf(PI * (frequency - outer_mask_radius) / wanted_mask_edge)) / 2.0;
                        complex_values[pixel_counter] *= edge;
                    }
                    if (do_inner_masking) {
                        if (frequency_squared <= inner_mask_radius_squared &&
                            frequency_squared >= inner_mask_radius_minus_edge_squared) {
                            frequency = sqrtf(frequency_squared);
                            edge      = (1.0 + cosf(PI * (outer_mask_radius - frequency) / wanted_mask_edge)) / 2.0;
                            complex_values[pixel_counter] *= edge;
                        }
                        if (frequency_squared <= inner_mask_radius_minus_edge_squared)
                            complex_values[pixel_counter] = 0.0f + I * 0.0f;
                    }

                    if (frequency_squared >= outer_mask_radius_plus_edge_squared)
                        complex_values[pixel_counter] = 0.0f + I * 0.0f;

                    pixel_counter++;
                }
            }
        }
    }

    return float(mask_volume);
}

// BEGIN_FOR_STAND_ALONE_CTFFIND

void Image::CircleMask(float wanted_mask_radius, bool invert) {
    MyDebugAssertTrue(is_in_real_space, "Image not in real space");
    MyDebugAssertTrue(object_is_centred_in_box, "Object not centered in box");

    long        pixel_counter;
    int         i, j, k;
    float       x, y, z;
    float       distance_from_center_squared;
    const float wanted_mask_radius_squared = powf(wanted_mask_radius, 2);
    double      average_value              = 0.0;
    long        number_of_pixels           = 0;

    pixel_counter = 0;
    for (k = 0; k < logical_z_dimension; k++) {
        z = powf(k - physical_address_of_box_center_z, 2);

        for (j = 0; j < logical_y_dimension; j++) {
            y = powf(j - physical_address_of_box_center_y, 2);

            for (i = 0; i < logical_x_dimension; i++) {
                x = powf(i - physical_address_of_box_center_x, 2);

                distance_from_center_squared = x + y + z;

                if (abs(distance_from_center_squared - wanted_mask_radius_squared) <= 4.0) {
                    number_of_pixels++;
                    average_value += real_values[pixel_counter];
                }

                pixel_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }

    // Now we know what value to mask with
    average_value /= float(number_of_pixels);

    // Let's mask
    pixel_counter = 0;
    for (k = 0; k < logical_z_dimension; k++) {
        z = powf(k - physical_address_of_box_center_z, 2);

        for (j = 0; j < logical_y_dimension; j++) {
            y = powf(j - physical_address_of_box_center_y, 2);

            for (i = 0; i < logical_x_dimension; i++) {
                x = powf(i - physical_address_of_box_center_x, 2);

                distance_from_center_squared = x + y + z;

                if (invert) {
                    if (distance_from_center_squared <= wanted_mask_radius_squared) {
                        real_values[pixel_counter] = average_value;
                    }
                }
                else {
                    if (distance_from_center_squared > wanted_mask_radius_squared) {
                        real_values[pixel_counter] = average_value;
                    }
                }

                pixel_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }
}

// END_FOR_STAND_ALONE_CTFFIND

void Image::CircleMaskWithValue(float wanted_mask_radius, float wanted_mask_value, bool invert) {
    MyDebugAssertTrue(is_in_real_space, "Image not in real space");
    MyDebugAssertTrue(object_is_centred_in_box, "Object not centered in box");

    long        pixel_counter;
    int         i, j, k;
    float       x, y, z;
    float       distance_from_center_squared;
    const float wanted_mask_radius_squared = powf(wanted_mask_radius, 2);
    long        number_of_pixels           = 0;

    // Let's mask
    pixel_counter = 0;
    for (k = 0; k < logical_z_dimension; k++) {
        z = powf(k - physical_address_of_box_center_z, 2);

        for (j = 0; j < logical_y_dimension; j++) {
            y = powf(j - physical_address_of_box_center_y, 2);

            for (i = 0; i < logical_x_dimension; i++) {
                x = powf(i - physical_address_of_box_center_x, 2);

                distance_from_center_squared = x + y + z;

                if (invert) {
                    if (distance_from_center_squared <= wanted_mask_radius_squared) {
                        real_values[pixel_counter] = wanted_mask_value;
                    }
                }
                else {
                    if (distance_from_center_squared > wanted_mask_radius_squared) {
                        real_values[pixel_counter] = wanted_mask_value;
                    }
                }

                pixel_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }
}

void Image::TriangleMask(float wanted_triangle_half_base_length) {
    MyDebugAssertTrue(is_in_real_space, "Image not in real space");
    MyDebugAssertTrue(object_is_centred_in_box, "Object not centered in box");

    long  pixel_counter;
    int   i, j, k;
    float x, y, z;
    long  number_of_pixels = 0;

    // Let's mask
    pixel_counter = 0;
    for (k = 0; k < logical_z_dimension; k++) {
        z = 1.0 - (abs(k - physical_address_of_box_center_z) / wanted_triangle_half_base_length);
        if (z < 0.0)
            z = 0.0;

        for (j = 0; j < logical_y_dimension; j++) {
            y = 1.0 - (abs(j - physical_address_of_box_center_y) / wanted_triangle_half_base_length);
            if (y < 0.0)
                y = 0.0;

            for (i = 0; i < logical_x_dimension; i++) {
                x = 1.0 - (abs(i - physical_address_of_box_center_x) / wanted_triangle_half_base_length);
                if (x < 0.0)
                    x = 0.0;

                real_values[pixel_counter] *= x * y * z;

                pixel_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }
}

void Image::SquareMaskWithValue(float wanted_mask_dim, float wanted_mask_value, bool invert, int wanted_center_x,
                                int wanted_center_y, int wanted_center_z) {
    MyDebugAssertTrue(is_in_real_space, "Image not in real space");
    MyDebugAssertTrue(object_is_centred_in_box, "Object not centered in box");

    long  pixel_counter;
    int   i, j, k;
    float x, y, z;

    if (wanted_center_x == 0 && wanted_center_y == 0 && wanted_center_z == 0) {
        wanted_center_x = physical_address_of_box_center_x;
        wanted_center_y = physical_address_of_box_center_y;
        wanted_center_z = physical_address_of_box_center_z;
    }

    const int i_min = wanted_center_x - wanted_mask_dim / 2;
    const int i_max = wanted_center_x + (wanted_mask_dim - wanted_mask_dim / 2 - 1);
    const int j_min = wanted_center_y - wanted_mask_dim / 2;
    const int j_max = wanted_center_y + (wanted_mask_dim - wanted_mask_dim / 2 - 1);
    const int k_min = wanted_center_z - wanted_mask_dim / 2;
    const int k_max = wanted_center_z + (wanted_mask_dim - wanted_mask_dim / 2 - 1);

    // Let's mask
    pixel_counter = 0;
    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {

                if (invert) {
                    if (i >= i_min && i <= i_max && j >= j_min && j <= j_max && k >= k_min && k <= k_max) {
                        real_values[pixel_counter] = wanted_mask_value;
                    }
                }
                else {
                    if (i < i_min || i > i_max || j < j_min || j > j_max || k < k_min || k > k_max) {
                        real_values[pixel_counter] = wanted_mask_value;
                    }
                }

                pixel_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }
}

void Image::GaussianLowPassFilter(float sigma) {

    int i;
    int j;
    int k;

    float x;
    float y;
    float z;

    long  pixel_counter;
    float one_over_two_sigma_squared = 0.5 / powf(sigma, 2);

    if (is_in_real_space) {
        /*
         * Real space (masking)
         */
        float distance_from_center_squared;

        pixel_counter = 0;
        for (k = 0; k < logical_z_dimension; k++) {
            z = powf(k - physical_address_of_box_center_z, 2);

            for (j = 0; j < logical_y_dimension; j++) {
                y = powf(j - physical_address_of_box_center_y, 2);

                for (i = 0; i < logical_x_dimension; i++) {
                    x = powf(i - physical_address_of_box_center_x, 2);

                    distance_from_center_squared = x + y + z;

                    real_values[pixel_counter] *= expf(-distance_from_center_squared * one_over_two_sigma_squared);

                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }

    else {
        /*
         * Fourier space (filtering)
         */
        float frequency_squared;

        pixel_counter = 0;
        for (k = 0; k <= physical_upper_bound_complex_z; k++) {
            z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

            for (j = 0; j <= physical_upper_bound_complex_y; j++) {
                y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

                for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                    x = powf(i * fourier_voxel_size_x, 2);

                    // compute squared radius, in units of reciprocal pixels
                    frequency_squared = x + y + z;
                    complex_values[pixel_counter] *= expf(-frequency_squared * one_over_two_sigma_squared);
                    pixel_counter++;
                }
            }
        }
    }
}

void Image::GaussianHighPassFilter(float sigma) {
    MyDebugAssertTrue(is_in_real_space == false, "Image Must Be Complex");

    int i;
    int j;
    int k;

    float x;
    float y;
    float z;

    long pixel_counter = 0;

    float frequency_squared;
    float frequency;
    float one_over_two_sigma_squared = 0.5 / powf(sigma, 2);

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                x = powf(i * fourier_voxel_size_x, 2);

                // compute squared radius, in units of reciprocal pixels
                frequency_squared = x + y + z;
                complex_values[pixel_counter] *= 1.0 - expf(-frequency_squared * one_over_two_sigma_squared);
                pixel_counter++;
            }
        }
    }
}
void Image::RandomisePhases(float wanted_radius_in_reciprocal_pixels) {
    bool need_to_fft = false;

    int   i;
    int   j;
    int   k;
    float x;
    float y;
    float z;

    float pixel_real;
    float pixel_imag;

    float current_amplitude;
    float current_phase;

    float frequency_squared;
    wanted_radius_in_reciprocal_pixels = powf(wanted_radius_in_reciprocal_pixels, 2);

    long pixel_counter = 0;

    if (is_in_real_space == true) {
        ForwardFFT();
        need_to_fft = true;
    }

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                x = powf(i * fourier_voxel_size_x, 2);

                // compute squared radius, in units of reciprocal pixels

                frequency_squared = x + y + z;

                if (frequency_squared >= wanted_radius_in_reciprocal_pixels) {
                    pixel_real = real(complex_values[pixel_counter]);
                    pixel_imag = imag(complex_values[pixel_counter]);

                    current_amplitude = sqrtf(powf(pixel_real, 2) + powf(pixel_imag, 2));
                    current_phase     = global_random_number_generator.GetUniformRandom() * PI;

                    complex_values[pixel_counter] =
                        (current_amplitude * cosf(current_phase)) + I * (current_amplitude * sinf(current_phase));
                }

                pixel_counter++;
            }
        }
    }

    if (need_to_fft == true)
        BackwardFFT();
}

// BEGIN_FOR_STAND_ALONE_CTFFIND
/*
 * Raise cosine-edged mask. The first argument gives the radius of the midpoint of the cosine (where cos = 0.5)
 */
float Image::CosineMask(float wanted_mask_radius, float wanted_mask_edge, bool invert, bool force_mask_value,
                        float wanted_mask_value) {
    //	MyDebugAssertTrue(! is_in_real_space || object_is_centred_in_box, "Image in real space but not centered");

    /*if (is_in_real_space)
    {
        MyDebugAssertTrue(wanted_mask_edge >= 1.0, "Edge width too small");
    }
    else
    {
        MyDebugAssertTrue(wanted_mask_edge > 0.0, "Edge width too small");
    }
*/
    int  i;
    int  j;
    int  k;
    int  ii;
    int  jj;
    int  kk;
    long number_of_pixels;

    float x;
    float y;
    float z;

    long pixel_counter = 0;

    float  distance_from_center;
    float  mask_radius_plus_edge;
    float  distance_from_center_squared;
    float  mask_radius;
    float  mask_radius_squared;
    float  mask_radius_plus_edge_squared;
    float  edge;
    double pixel_sum;

    float frequency;
    float frequency_squared;

    double mask_volume = 0.0;

    mask_radius = wanted_mask_radius - wanted_mask_edge * 0.5;
    if (mask_radius < 0.0)
        mask_radius = 0.0;
    mask_radius_plus_edge = mask_radius + wanted_mask_edge;

    mask_radius_squared           = powf(mask_radius, 2);
    mask_radius_plus_edge_squared = powf(mask_radius_plus_edge, 2);

    pixel_sum        = 0.0;
    number_of_pixels = 0;
    if (is_in_real_space && object_is_centred_in_box) {
        if (force_mask_value) {
            pixel_sum = wanted_mask_value;
        }
        else {
            for (k = 0; k < logical_z_dimension; k++) {
                z = powf(k - physical_address_of_box_center_z, 2);

                for (j = 0; j < logical_y_dimension; j++) {
                    y = powf(j - physical_address_of_box_center_y, 2);

                    for (i = 0; i < logical_x_dimension; i++) {
                        x = powf(i - physical_address_of_box_center_x, 2);

                        distance_from_center_squared = x + y + z;

                        if (distance_from_center_squared >= mask_radius_squared &&
                            distance_from_center_squared <= mask_radius_plus_edge_squared) {
                            pixel_sum += real_values[pixel_counter];
                            number_of_pixels++;
                        }
                        pixel_counter++;
                    }
                    pixel_counter += padding_jump_value;
                }
            }
            MyDebugAssertTrue(number_of_pixels > 0, "Oops, did not find any pixels to average over");
            pixel_sum /= number_of_pixels;
        }

        pixel_counter = 0;
        for (k = 0; k < logical_z_dimension; k++) {
            z = powf(k - physical_address_of_box_center_z, 2);

            for (j = 0; j < logical_y_dimension; j++) {
                y = powf(j - physical_address_of_box_center_y, 2);

                for (i = 0; i < logical_x_dimension; i++) {
                    x = powf(i - physical_address_of_box_center_x, 2);

                    distance_from_center_squared = x + y + z;

                    if (distance_from_center_squared >= mask_radius_squared &&
                        distance_from_center_squared <= mask_radius_plus_edge_squared) {
                        distance_from_center = sqrtf(distance_from_center_squared);
                        edge = (1.0 + cosf(PI * (distance_from_center - mask_radius) / wanted_mask_edge)) / 2.0;
                        if (invert) {
                            real_values[pixel_counter] = real_values[pixel_counter] * (1.0 - edge) + edge * pixel_sum;
                            mask_volume += powf(1.0 - edge, 2);
                        }
                        else {
                            real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * pixel_sum;
                            mask_volume += powf(edge, 2);
                        }
                    }
                    else if (invert) {
                        if (distance_from_center_squared <= mask_radius_squared) {
                            real_values[pixel_counter] = pixel_sum;
                        }
                        else {
                            mask_volume += 1.0;
                        }
                    }
                    else {
                        if (distance_from_center_squared >= mask_radius_plus_edge_squared) {
                            real_values[pixel_counter] = pixel_sum;
                        }
                        else {
                            mask_volume += 1.0;
                        }
                    }

                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }
    else if (is_in_real_space) {
        if (force_mask_value) {
            pixel_sum = wanted_mask_value;
        }
        else {
            for (k = 0; k < logical_z_dimension; k++) {
                kk = k;
                if (kk >= physical_address_of_box_center_z)
                    kk -= logical_z_dimension;
                z = powf(kk, 2);

                for (j = 0; j < logical_y_dimension; j++) {
                    jj = j;
                    if (jj >= physical_address_of_box_center_y)
                        jj -= logical_y_dimension;
                    y = powf(jj, 2);

                    for (i = 0; i < logical_x_dimension; i++) {
                        ii = i;
                        if (ii >= physical_address_of_box_center_x)
                            ii -= logical_x_dimension;
                        x = powf(ii, 2);

                        distance_from_center_squared = x + y + z;

                        if (distance_from_center_squared >= mask_radius_squared &&
                            distance_from_center_squared <= mask_radius_plus_edge_squared) {
                            pixel_sum += real_values[pixel_counter];
                            number_of_pixels++;
                        }
                        pixel_counter++;
                    }
                    pixel_counter += padding_jump_value;
                }
            }
            pixel_sum /= number_of_pixels;
        }

        pixel_counter = 0.0;
        for (k = 0; k < logical_z_dimension; k++) {
            kk = k;
            if (kk >= physical_address_of_box_center_z)
                kk -= logical_z_dimension;
            z = powf(kk, 2);

            for (j = 0; j < logical_y_dimension; j++) {
                jj = j;
                if (jj >= physical_address_of_box_center_y)
                    jj -= logical_y_dimension;
                y = powf(jj, 2);

                for (i = 0; i < logical_x_dimension; i++) {
                    ii = i;
                    if (ii >= physical_address_of_box_center_x)
                        ii -= logical_x_dimension;
                    x = powf(ii, 2);

                    distance_from_center_squared = x + y + z;

                    if (distance_from_center_squared >= mask_radius_squared &&
                        distance_from_center_squared <= mask_radius_plus_edge_squared) {
                        distance_from_center = sqrtf(distance_from_center_squared);
                        edge = (1.0 + cosf(PI * (distance_from_center - mask_radius) / wanted_mask_edge)) / 2.0;
                        real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * pixel_sum;
                        mask_volume += powf(edge, 2);
                    }
                    else if (distance_from_center_squared >= mask_radius_plus_edge_squared)
                        real_values[pixel_counter] = pixel_sum;
                    else {
                        mask_volume += 1.0;
                    }

                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }
    else {
        for (k = 0; k <= physical_upper_bound_complex_z; k++) {
            z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

            for (j = 0; j <= physical_upper_bound_complex_y; j++) {
                y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

                for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                    x = powf(i * fourier_voxel_size_x, 2);

                    // compute squared radius, in units of reciprocal pixels

                    frequency_squared = x + y + z;

                    if (frequency_squared >= mask_radius_squared &&
                        frequency_squared <= mask_radius_plus_edge_squared) {
                        frequency = sqrtf(frequency_squared);
                        edge      = (1.0 + cosf(PI * (frequency - mask_radius) / wanted_mask_edge)) / 2.0;
                        if (invert) {
                            complex_values[pixel_counter] *= (1.0 - edge);
                        }
                        else {
                            complex_values[pixel_counter] *= edge;
                        }
                    }
                    if (invert) {
                        if (frequency_squared <= mask_radius_squared)
                            complex_values[pixel_counter] = 0.0f + I * 0.0f;
                    }
                    else {
                        if (frequency_squared >= mask_radius_plus_edge_squared)
                            complex_values[pixel_counter] = 0.0f + I * 0.0f;
                    }

                    pixel_counter++;
                }
            }
        }
    }

    return float(mask_volume);
}

float Image::CosineRectangularMask(float wanted_mask_radius_x, float wanted_mask_radius_y, float wanted_mask_radius_z,
                                   float wanted_mask_edge, bool invert, bool force_mask_value,
                                   float wanted_mask_value) {
    //	MyDebugAssertTrue(! is_in_real_space || object_is_centred_in_box, "Image in real space but not centered");
    if (is_in_real_space) {
        MyDebugAssertTrue(wanted_mask_edge >= 1.0f, "Edge width too small");
    }
    else {
        MyDebugAssertTrue(wanted_mask_edge > 0.0f, "Edge width too small");
    }

    int  i;
    int  j;
    int  k;
    int  ii;
    int  jj;
    int  kk;
    long number_of_pixels;

    int   x;
    int   y;
    int   z;
    float float_x;
    float float_y;
    float float_z;

    long pixel_counter = 0;

    float  distance_from_center;
    float  mask_radius_plus_edge_x;
    float  mask_radius_plus_edge_y;
    float  mask_radius_plus_edge_z;
    float  mask_radius_x;
    float  mask_radius_y;
    float  mask_radius_z;
    float  edge;
    double pixel_sum;

    double mask_volume = 0.0f;

    mask_radius_x = wanted_mask_radius_x - wanted_mask_edge * 0.5f;
    mask_radius_y = wanted_mask_radius_y - wanted_mask_edge * 0.5f;
    mask_radius_z = wanted_mask_radius_z - wanted_mask_edge * 0.5f;
    if (mask_radius_x < 0.0f)
        mask_radius_x = 0.0f;
    if (mask_radius_y < 0.0f)
        mask_radius_y = 0.0f;
    if (mask_radius_z < 0.0f)
        mask_radius_z = 0.0f;
    mask_radius_plus_edge_x = mask_radius_x + wanted_mask_edge;
    mask_radius_plus_edge_y = mask_radius_y + wanted_mask_edge;
    mask_radius_plus_edge_z = mask_radius_z + wanted_mask_edge;

    pixel_sum        = 0.0f;
    number_of_pixels = 0;
    if (is_in_real_space && object_is_centred_in_box) {
        if (force_mask_value) {
            pixel_sum = wanted_mask_value;
        }
        else {
            for (k = 0; k < logical_z_dimension; k++) {
                z = abs(k - physical_address_of_box_center_z);

                for (j = 0; j < logical_y_dimension; j++) {
                    y = abs(j - physical_address_of_box_center_y);

                    for (i = 0; i < logical_x_dimension; i++) {
                        x = abs(i - physical_address_of_box_center_x);

                        if (!(x <= mask_radius_x && y <= mask_radius_y && z <= mask_radius_z) &&
                            (x <= mask_radius_plus_edge_x && y <= mask_radius_plus_edge_y &&
                             z <= mask_radius_plus_edge_z)) {
                            pixel_sum += real_values[pixel_counter];
                            number_of_pixels++;
                        }
                        pixel_counter++;
                    }
                    pixel_counter += padding_jump_value;
                }
            }
            MyDebugAssertTrue(number_of_pixels > 0, "Oops, did not find any pixels to average over");
            pixel_sum /= number_of_pixels;
        }

        pixel_counter = 0.0f;
        for (k = 0; k < logical_z_dimension; k++) {
            z = abs(k - physical_address_of_box_center_z);

            for (j = 0; j < logical_y_dimension; j++) {
                y = abs(j - physical_address_of_box_center_y);

                for (i = 0; i < logical_x_dimension; i++) {
                    x = abs(i - physical_address_of_box_center_x);

                    if (!(x <= mask_radius_x && y <= mask_radius_y && z <= mask_radius_z) &&
                        (x < mask_radius_plus_edge_x && y < mask_radius_plus_edge_y && z < mask_radius_plus_edge_z)) {
                        edge = 1.0f;
                        if (x > mask_radius_x && x < mask_radius_plus_edge_x)
                            edge *= (1.0f + cosf(PI * (x - mask_radius_x) / wanted_mask_edge)) / 2.0f;
                        if (y > mask_radius_y && y < mask_radius_plus_edge_y)
                            edge *= (1.0f + cosf(PI * (y - mask_radius_y) / wanted_mask_edge)) / 2.0f;
                        if (z > mask_radius_z && z < mask_radius_plus_edge_z)
                            edge *= (1.0f + cosf(PI * (z - mask_radius_z) / wanted_mask_edge)) / 2.0f;
                        if (invert) {
                            real_values[pixel_counter] = real_values[pixel_counter] * (1.0f - edge) + edge * pixel_sum;
                            mask_volume += powf(1.0f - edge, 2);
                        }
                        else {
                            real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0f - edge) * pixel_sum;
                            mask_volume += powf(edge, 2);
                        }
                    }
                    else if (invert) {
                        if (x <= mask_radius_x && y <= mask_radius_y && z <= mask_radius_z) {
                            real_values[pixel_counter] = pixel_sum;
                        }
                        else {
                            mask_volume += 1.0f;
                        }
                    }
                    else {
                        if (x >= mask_radius_plus_edge_x || y >= mask_radius_plus_edge_y ||
                            z >= mask_radius_plus_edge_z)
                            real_values[pixel_counter] = pixel_sum;
                        else {
                            mask_volume += 1.0f;
                        }
                    }

                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }
    else if (is_in_real_space) {
        if (force_mask_value) {
            pixel_sum = wanted_mask_value;
        }
        else {
            for (k = 0; k < logical_z_dimension; k++) {
                kk = k;
                if (kk >= physical_address_of_box_center_z)
                    kk -= logical_z_dimension;
                z = abs(kk);

                for (j = 0; j < logical_y_dimension; j++) {
                    jj = j;
                    if (jj >= physical_address_of_box_center_y)
                        jj -= logical_y_dimension;
                    y = abs(jj);

                    for (i = 0; i < logical_x_dimension; i++) {
                        ii = i;
                        if (ii >= physical_address_of_box_center_x)
                            ii -= logical_x_dimension;
                        x = abs(ii);

                        if (!(x <= mask_radius_x && y <= mask_radius_y && z <= mask_radius_z) &&
                            (x <= mask_radius_plus_edge_x && y <= mask_radius_plus_edge_y &&
                             z <= mask_radius_plus_edge_z)) {
                            pixel_sum += real_values[pixel_counter];
                            number_of_pixels++;
                        }
                        pixel_counter++;
                    }
                    pixel_counter += padding_jump_value;
                }
            }
            pixel_sum /= number_of_pixels;
        }

        pixel_counter = 0.0f;
        for (k = 0; k < logical_z_dimension; k++) {
            kk = k;
            if (kk >= physical_address_of_box_center_z)
                kk -= logical_z_dimension;
            z = abs(kk);

            for (j = 0; j < logical_y_dimension; j++) {
                jj = j;
                if (jj >= physical_address_of_box_center_y)
                    jj -= logical_y_dimension;
                y = abs(jj);

                for (i = 0; i < logical_x_dimension; i++) {
                    ii = i;
                    if (ii >= physical_address_of_box_center_x)
                        ii -= logical_x_dimension;
                    x = abs(ii);

                    if (!(x <= mask_radius_x && y <= mask_radius_y && z <= mask_radius_z) &&
                        (x < mask_radius_plus_edge_x && y < mask_radius_plus_edge_y && z < mask_radius_plus_edge_z)) {
                        edge = 1.0f;
                        if (x > mask_radius_x && x < mask_radius_plus_edge_x)
                            edge *= (1.0f + cosf(PI * (x - mask_radius_x) / wanted_mask_edge)) / 2.0;
                        if (y > mask_radius_y && y < mask_radius_plus_edge_y)
                            edge *= (1.0f + cosf(PI * (y - mask_radius_y) / wanted_mask_edge)) / 2.0;
                        if (z > mask_radius_z && z < mask_radius_plus_edge_z)
                            edge *= (1.0f + cosf(PI * (z - mask_radius_z) / wanted_mask_edge)) / 2.0;
                        real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0f - edge) * pixel_sum;
                        mask_volume += powf(edge, 2);
                    }
                    else if (x >= mask_radius_plus_edge_x || y >= mask_radius_plus_edge_y ||
                             z >= mask_radius_plus_edge_z)
                        real_values[pixel_counter] = pixel_sum;
                    else {
                        mask_volume += 1.0f;
                    }

                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }
    else {
        for (k = 0; k <= physical_upper_bound_complex_z; k++) {
            float_z = fabsf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z);

            for (j = 0; j <= physical_upper_bound_complex_y; j++) {
                float_y = fabsf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y);

                for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                    float_x = i * fourier_voxel_size_x;

                    // compute squared radius, in units of reciprocal pixels

                    if (!(float_x <= mask_radius_x && float_y <= mask_radius_y && float_z <= mask_radius_z) &&
                        (float_x < mask_radius_plus_edge_x && float_y < mask_radius_plus_edge_y &&
                         float_z < mask_radius_plus_edge_z)) {
                        edge = 1.0f;
                        if (float_x > mask_radius_x && float_x < mask_radius_plus_edge_x)
                            edge *= (1.0f + cosf(PI * (float_x - mask_radius_x) / wanted_mask_edge)) / 2.0f;
                        if (float_y > mask_radius_y && float_y < mask_radius_plus_edge_y)
                            edge *= (1.0f + cosf(PI * (float_y - mask_radius_y) / wanted_mask_edge)) / 2.0f;
                        if (float_z > mask_radius_z && float_z < mask_radius_plus_edge_z)
                            edge *= (1.0f + cosf(PI * (float_z - mask_radius_z) / wanted_mask_edge)) / 2.0f;
                        if (invert) {
                            complex_values[pixel_counter] *= (1.0f - edge);
                        }
                        else {
                            complex_values[pixel_counter] *= edge;
                        }
                    }
                    if (invert) {
                        if (float_x <= mask_radius_x && float_y <= mask_radius_y && float_z <= mask_radius_z)
                            complex_values[pixel_counter] = 0.0f + I * 0.0f;
                    }
                    else {
                        if (float_x >= mask_radius_plus_edge_x || float_y >= mask_radius_plus_edge_y ||
                            float_z >= mask_radius_plus_edge_z)
                            complex_values[pixel_counter] = 0.0f + I * 0.0f;
                    }

                    pixel_counter++;
                }
            }
        }
    }

    return float(mask_volume);
}

void Image::LocalResSignificanceFilter(float pixel_size, float starting_resolution, float mask_radius_in_angstroms) {
    Image sharp3d;
    Image automask;
    Image output3d;
    Image buffer3d;
    float start_spatial_frequency = (pixel_size / starting_resolution) + 0.03f;
    if (start_spatial_frequency > 0.5f)
        start_spatial_frequency = 0.5f;
    float end_spatial_frequency = pixel_size / 30.0f;
    float step_size             = (start_spatial_frequency - end_spatial_frequency) / 6.0f;
    float window                = (start_spatial_frequency - end_spatial_frequency) / 20.0f;
    float threshold_value;
    long  pixel_counter;
    float filter_frequency;
    int   i, j, k;

    int pixels_above_threshold;
    int pixels_added;

    EmpiricalDistribution distro;

    automask.CopyFrom(this);
    sharp3d.CopyFrom(this);

    output3d.Allocate(logical_x_dimension, logical_y_dimension, logical_z_dimension);
    output3d.SetToConstant(0.0f);

    automask.ConvertToAutoMask(pixel_size, mask_radius_in_angstroms, 10.0f, 0.05);
    int counter = 1;
    for (float current_spatial_frequency = start_spatial_frequency; current_spatial_frequency >= end_spatial_frequency;
         current_spatial_frequency -= step_size) {
        buffer3d.CopyFrom(this);
        sharp3d.CopyFrom(this);

        buffer3d.ForwardFFT();
        filter_frequency = current_spatial_frequency + window;
        if (filter_frequency > 0.5)
            filter_frequency = 0.5;
        buffer3d.CosineMask(filter_frequency, window);
        buffer3d.BackwardFFT();
        buffer3d.QuickAndDirtyWriteSlices(wxString::Format("/tmp/filter_%i.mrc", counter).ToStdString(), 1,
                                          buffer3d.logical_z_dimension);

        sharp3d.SharpenMap(pixel_size, pixel_size / (current_spatial_frequency), false, 0.0f, mask_radius_in_angstroms,
                           50.0f, 0.0f, 0.0f, 500.0f, false, &automask);

        distro.Reset();
        pixel_counter = 0;

        for (k = 0; k < sharp3d.logical_z_dimension; k++) {
            for (j = 0; j < sharp3d.logical_y_dimension; j++) {
                for (i = 0; i < sharp3d.logical_x_dimension; i++) {
                    distro.AddSampleValue(sharp3d.real_values[pixel_counter]);
                    pixel_counter++;
                }

                pixel_counter += sharp3d.padding_jump_value;
            }
        }
        // threshold

        // threshold_value = sharp3d.ReturnAverageOfMaxN(50) * 0.3;

        // threshold_value = sharp3d.ReturnAverageOfRealValues() + ((sharp3d.ReturnAverageOfMaxN(50) -
        // sharp3d.ReturnAverageOfRealValues()) * 0.5f); threshold_value =
        // sharp3d.ReturnAverageOfRealValuesAtRadius(sharp3d.logical_x_dimension / 2.0f) + (6.0f *
        // sqrtf(sharp3d.ReturnVarianceOfRealValues(sharp3d.logical_x_dimension / 2.0f))); threshold_value =
        // distro.GetSampleMean() + (8.0f * sqrtf(distro.GetSampleVariance())); threshold_value = (8.0f *
        // sqrtf(distro.GetSampleVariance())); sharp3d.QuickAndDirtyWriteSlices("/tmp/sharp.mrc", 1,
        // sharp3d.logical_z_dimension); wxPrintf("Threshold = %f\n", threshold_value);
        sharp3d.QuickAndDirtyWriteSlices(wxString::Format("/tmp/sharp_%i.mrc", counter).ToStdString(), 1,
                                         buffer3d.logical_z_dimension);

        float original_average_value = sharp3d.ReturnAverageOfRealValues(mask_radius_in_angstroms / pixel_size, true);
        sharp3d.SetMinimumValue(original_average_value);
        float average_value  = sharp3d.ReturnAverageOfRealValues(mask_radius_in_angstroms / pixel_size, true);
        float average_of_max = sharp3d.ReturnAverageOfMaxN(50, mask_radius_in_angstroms / pixel_size);
        threshold_value      = average_value + ((average_of_max - average_value) * 0.25);

        sharp3d.Binarise(threshold_value);
        sharp3d.QuickAndDirtyWriteSlices("/tmp/bin.mrc", 1, sharp3d.logical_z_dimension);
        sharp3d.QuickAndDirtyWriteSlices(wxString::Format("/tmp/bin_%i.mrc", counter).ToStdString(), 1,
                                         buffer3d.logical_z_dimension);
        counter++;
        pixels_above_threshold = 0;
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
            if (sharp3d.real_values[pixel_counter] == 1.0f)
                pixels_above_threshold++;
        }

        //	sharp3d.QuickAndDirtyWriteSlices("sharp.mrc", 1, sharp3d.logical_z_dimension);
        pixels_added = 0;
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
            if (sharp3d.real_values[pixel_counter] == 1.0f && output3d.real_values[pixel_counter] == 0.0f) {
                output3d.real_values[pixel_counter] = buffer3d.real_values[pixel_counter];
                pixels_added++;
            }
        }

        wxPrintf("Resolution %.2f has %i pixels above threshold (this added %i pixels to the output)\n",
                 pixel_size / current_spatial_frequency, pixels_above_threshold, pixels_added);
    }

    for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        if (output3d.real_values[pixel_counter] == 0.0f)
            output3d.real_values[pixel_counter] = buffer3d.real_values[pixel_counter];
    }

    //	output3d.QuickAndDirtyWriteSlices("filtered.mrc", 1, sharp3d.logical_z_dimension);
    Consume(&output3d);
}

void Image::ConvertToAutoMask(float pixel_size, float outer_mask_radius_in_angstroms,
                              float filter_resolution_in_angstroms, float rebin_value,
                              bool auto_estimate_initial_bin_value, float wanted_initial_bin_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Image must be in real space");

    Image buffer_image;

    int original_x_size;
    int original_y_size;
    int original_z_size;

    int binned_x_size;
    int binned_y_size;
    int binned_z_size;

    float original_average_value;
    float average_value;
    float average_of_max;
    float threshold_value;
    float binning_factor = filter_resolution_in_angstroms / 2.0f / pixel_size;

    int number_of_top_pixels_to_use;

    original_x_size = logical_x_dimension;
    original_y_size = logical_y_dimension;
    original_z_size = logical_z_dimension;

    binned_x_size = original_x_size / binning_factor + 0.5f;
    binned_y_size = original_y_size / binning_factor + 0.5f;
    binned_z_size = original_z_size / binning_factor + 0.5f;

    if (IsOdd(binned_x_size) == true)
        binned_x_size++;
    if (IsOdd(binned_y_size) == true)
        binned_y_size++;
    if (IsOdd(binned_z_size) == true)
        binned_z_size++;

    if (binned_x_size > original_x_size || binned_y_size > original_y_size || binned_z_size > original_z_size) {
        binned_x_size = original_x_size;
        binned_y_size = original_y_size;
        binned_z_size = original_z_size;
    }

    if (binned_x_size != original_x_size) {
        ForwardFFT();
        // DivideByConstant(sqrtf(number_of_real_space_pixels));
        Resize(binned_x_size, binned_y_size, binned_z_size);
        // DivideByConstant(sqrtf(number_of_real_space_pixels));
        BackwardFFT();
    }

    //	QuickAndDirtyWriteSlices("/tmp/binning_input.mrc", 1, logical_z_dimension, true);

    if (auto_estimate_initial_bin_value == false)
        threshold_value = wanted_initial_bin_value;
    else {
        original_average_value =
            ReturnAverageOfRealValues(outer_mask_radius_in_angstroms / binning_factor / pixel_size, true);
        SetMinimumValue(original_average_value);
        average_value = ReturnAverageOfRealValues(outer_mask_radius_in_angstroms / binning_factor / pixel_size, true);

        number_of_top_pixels_to_use = number_of_real_space_pixels * 0.000005;
        if (number_of_top_pixels_to_use < 5)
            number_of_top_pixels_to_use = 5;

        average_of_max  = ReturnAverageOfMaxN(number_of_top_pixels_to_use,
                                             outer_mask_radius_in_angstroms / binning_factor / pixel_size);
        threshold_value = average_value + ((average_of_max - average_value) * 0.05);
    }

    CosineMask(outer_mask_radius_in_angstroms / binning_factor / pixel_size, 1.0, false, true, -FLT_MAX);

    Binarise(threshold_value);
    //	QuickAndDirtyWriteSlices("/tmp/binned.mrc", 1, logical_z_dimension, true);

    // rle3d my_rle3d(*this);
    // my_rle3d.ConnectedSizeDecodeTo(*this);
    // Binarise(ReturnMaximumValue() - 1.0f);

    buffer_image.CopyFrom(this);
    ccl3d my_ccl3d(buffer_image);
    my_ccl3d.GetLargestConnectedDensityMask(buffer_image, *this);

    ForwardFFT();
    GaussianLowPassFilter(binning_factor / pixel_size / 16.0f);
    Resize(original_x_size, original_y_size, original_z_size, 0.0);
    BackwardFFT();

    DivideByConstant(ReturnMaximumValue());
    Binarise(rebin_value);
}

Image& Image::operator=(const Image& other_image) {
    *this = &other_image;
    return *this;
}

Image& Image::operator=(const Image* other_image) {
    // Check for self assignment
    if (this != other_image) {
        MyDebugAssertTrue(other_image->is_in_memory, "Other image Memory not allocated");

        if (is_in_memory == true) {

            if (logical_x_dimension != other_image->logical_x_dimension ||
                logical_y_dimension != other_image->logical_y_dimension ||
                logical_z_dimension != other_image->logical_z_dimension) {
                Deallocate();
                Allocate(other_image->logical_x_dimension, other_image->logical_y_dimension,
                         other_image->logical_z_dimension, other_image->is_in_real_space);
            }
        }
        else {
            Allocate(other_image->logical_x_dimension, other_image->logical_y_dimension,
                     other_image->logical_z_dimension, other_image->is_in_real_space);
        }

        // by here the memory allocation should be ok..

        is_in_real_space         = other_image->is_in_real_space;
        object_is_centred_in_box = other_image->object_is_centred_in_box;

        for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
            real_values[pixel_counter] = other_image->real_values[pixel_counter];
        }
    }

    return *this;
}

//!>  \brief  Deallocate all the memory.  The idea is to keep this safe in the case that something isn't
//    allocated, so it can always be safely called.  I.e. it should check whatever it is deallocating is
//    in fact allocated.
//

void Image::Deallocate() {
    if (is_in_memory == true && image_memory_should_not_be_deallocated == false) {
        fftwf_free(real_values);
        is_in_memory = false;
    }

    if (planned == true) {
        wxMutexLocker lock(s_mutexProtectingFFTW); // the mutex will be unlocked when this object is destroyed (when it
                                                   // goes out of scope)
        MyDebugAssertTrue(lock.IsOk(), "Mute locking failed");
        fftwf_destroy_plan(plan_fwd);
        fftwf_destroy_plan(plan_bwd);
        planned = false;
    }
}

//!>  \brief  Allocate memory for the Image object.
//
//  If the object is already allocated with correct dimensions, nothing happens. Otherwise, object is deallocated first.

void Image::Allocate(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool should_be_in_real_space,
                     bool do_fft_planning) {

    MyDebugAssertTrue(wanted_x_size > 0 && wanted_y_size > 0 && wanted_z_size > 0, "Bad dimensions: %i %i %i\n",
                      wanted_x_size, wanted_y_size, wanted_z_size);

    // check to see if we need to do anything?

    if (is_in_memory == true) {
        is_in_real_space = should_be_in_real_space;

        if (wanted_x_size == logical_x_dimension && wanted_y_size == logical_y_dimension &&
            wanted_z_size == logical_z_dimension) {
            // everything is already done..

            is_in_real_space = should_be_in_real_space;
            //			wxPrintf("returning\n");

            return;
        }
        else
            Deallocate();
    }

    // if we got here we need to do allocation..

    SetLogicalDimensions(wanted_x_size, wanted_y_size, wanted_z_size);
    is_in_real_space = should_be_in_real_space;

    // first_x_dimension
    if (IsEven(wanted_x_size) == true)
        real_memory_allocated = wanted_x_size / 2 + 1;
    else
        real_memory_allocated = (wanted_x_size - 1) / 2 + 1;

    real_memory_allocated *= wanted_y_size * wanted_z_size; // other dimensions
    real_memory_allocated *= 2;                             // room for complex

    real_values = (float*)fftwf_malloc(sizeof(float) * real_memory_allocated);
    complex_values =
        (std::complex<float>*)real_values; // Set the complex_values to point at the newly allocated real values;

    is_in_memory = true;

    // Update addresses etc..

    UpdateLoopingAndAddressing();

    // Prepare the plans for FFTW

    if (planned == false && do_fft_planning == true) // skip fft planning at your peril!
    {
        wxMutexLocker lock(s_mutexProtectingFFTW); // the mutex will be unlocked when this object is destroyed (when it
                                                   // goes out of scope)
        MyDebugAssertTrue(lock.IsOk(), "Mute locking failed");
        if (logical_z_dimension > 1) {
            plan_fwd = fftwf_plan_dft_r2c_3d(logical_z_dimension, logical_y_dimension, logical_x_dimension, real_values,
                                             reinterpret_cast<fftwf_complex*>(complex_values), FFTW_ESTIMATE);
            plan_bwd =
                fftwf_plan_dft_c2r_3d(logical_z_dimension, logical_y_dimension, logical_x_dimension,
                                      reinterpret_cast<fftwf_complex*>(complex_values), real_values, FFTW_ESTIMATE);
        }
        else {
            plan_fwd = fftwf_plan_dft_r2c_2d(logical_y_dimension, logical_x_dimension, real_values,
                                             reinterpret_cast<fftwf_complex*>(complex_values), FFTW_ESTIMATE);
            plan_bwd =
                fftwf_plan_dft_c2r_2d(logical_y_dimension, logical_x_dimension,
                                      reinterpret_cast<fftwf_complex*>(complex_values), real_values, FFTW_ESTIMATE);
        }

        if (plan_fwd == NULL || plan_bwd == NULL) {
            MyPrintWithDetails("Error in FFT Planning...");
            DEBUG_ABORT;
        }

        planned = true;
    }

    // set the loop junk value..

    if (IsEven(logical_x_dimension) == true)
        padding_jump_value = 2;
    else
        padding_jump_value = 1;

    //

    number_of_real_space_pixels = long(logical_x_dimension) * long(logical_y_dimension) * long(logical_z_dimension);
    ft_normalization_factor     = 1.0 / sqrtf(float(number_of_real_space_pixels));
}

//!>  \brief  Allocate memory for the Image object.
//
//  Overloaded version of allocate to cover the supplying just 2 dimensions along with the should_be_in_real_space bool.

void Image::Allocate(int wanted_x_size, int wanted_y_size, bool should_be_in_real_space) {
    Allocate(wanted_x_size, wanted_y_size, 1, should_be_in_real_space);
}

//!>  \brief  Allocate memory for the Image object.
//
//  Overloaded version allowing you to supply a reference 3d - the data is NOT copied.

void Image::Allocate(Image* image_to_copy_size_and_space_from) {
    Allocate(
        image_to_copy_size_and_space_from->logical_x_dimension, image_to_copy_size_and_space_from->logical_y_dimension,
        image_to_copy_size_and_space_from->logical_z_dimension, image_to_copy_size_and_space_from->is_in_real_space);
}

void Image::AllocateAsPointingToSliceIn3D(Image* wanted3d, long wanted_slice) {
    Deallocate();
    is_in_real_space = wanted3d->is_in_real_space;

    // if we got here we need to do allocation..

    SetLogicalDimensions(wanted3d->logical_x_dimension, wanted3d->logical_y_dimension, 1);

    // we are not actually allocating, we are pointing..

    long bytes_in_slice = wanted3d->real_memory_allocated / wanted3d->logical_z_dimension;

    image_memory_should_not_be_deallocated = true;
    is_in_memory                           = true;           // kind of a lie
    real_memory_allocated                  = bytes_in_slice; // kind of a lie

    real_values = wanted3d->real_values + (bytes_in_slice * (wanted_slice - 1)); // point to the 3d..
    complex_values =
        (std::complex<float>*)real_values; // Set the complex_values to point at the newly allocated real values;

    // Update addresses etc..

    UpdateLoopingAndAddressing();

    // Prepare the plans for FFTW

    if (planned == false) {
        wxMutexLocker lock(s_mutexProtectingFFTW); // the mutex will be unlocked when this object is destroyed (when it
                                                   // goes out of scope)
        MyDebugAssertTrue(lock.IsOk(), "Mute locking failed");

        if (logical_z_dimension > 1) {
            plan_fwd = fftwf_plan_dft_r2c_3d(logical_z_dimension, logical_y_dimension, logical_x_dimension, real_values,
                                             reinterpret_cast<fftwf_complex*>(complex_values), FFTW_ESTIMATE);
            plan_bwd =
                fftwf_plan_dft_c2r_3d(logical_z_dimension, logical_y_dimension, logical_x_dimension,
                                      reinterpret_cast<fftwf_complex*>(complex_values), real_values, FFTW_ESTIMATE);
        }
        else {
            plan_fwd = fftwf_plan_dft_r2c_2d(logical_y_dimension, logical_x_dimension, real_values,
                                             reinterpret_cast<fftwf_complex*>(complex_values), FFTW_ESTIMATE);
            plan_bwd =
                fftwf_plan_dft_c2r_2d(logical_y_dimension, logical_x_dimension,
                                      reinterpret_cast<fftwf_complex*>(complex_values), real_values, FFTW_ESTIMATE);
        }

        planned = true;
    }

    // set the loop jump value..

    if (IsEven(logical_x_dimension) == true)
        padding_jump_value = 2;
    else
        padding_jump_value = 1;

    //
    number_of_real_space_pixels = long(logical_x_dimension) * long(logical_y_dimension) * long(logical_z_dimension);
    ft_normalization_factor     = 1.0 / sqrtf(float(number_of_real_space_pixels));
}

//!>  \brief  Change the logical dimensions of an image and update all related values

void Image::SetLogicalDimensions(int wanted_x_size, int wanted_y_size, int wanted_z_size) {
    logical_x_dimension = wanted_x_size;
    logical_y_dimension = wanted_y_size;
    logical_z_dimension = wanted_z_size;
}

//!>  \brief  Update all properties related to looping & addressing in real & Fourier space, given the current logical
//!dimensions.

void Image::UpdateLoopingAndAddressing() {

    physical_upper_bound_complex_x = logical_x_dimension / 2;
    physical_upper_bound_complex_y = logical_y_dimension - 1;
    physical_upper_bound_complex_z = logical_z_dimension - 1;

    UpdatePhysicalAddressOfBoxCenter();

    // physical_index_of_first_negative_frequency_x = logical_x_dimension / 2 + 1;
    if (IsEven(logical_y_dimension) == true) {
        physical_index_of_first_negative_frequency_y = logical_y_dimension / 2;
    }
    else {
        physical_index_of_first_negative_frequency_y = logical_y_dimension / 2 + 1;
    }

    if (IsEven(logical_z_dimension) == true) {
        physical_index_of_first_negative_frequency_z = logical_z_dimension / 2;
    }
    else {
        physical_index_of_first_negative_frequency_z = logical_z_dimension / 2 + 1;
    }

    // Update the Fourier voxel size

    fourier_voxel_size_x = 1.0 / double(logical_x_dimension);
    fourier_voxel_size_y = 1.0 / double(logical_y_dimension);
    fourier_voxel_size_z = 1.0 / double(logical_z_dimension);

    // Logical bounds
    if (IsEven(logical_x_dimension) == true) {
        logical_lower_bound_complex_x = -logical_x_dimension / 2;
        logical_upper_bound_complex_x = logical_x_dimension / 2;
        logical_lower_bound_real_x    = -logical_x_dimension / 2;
        logical_upper_bound_real_x    = logical_x_dimension / 2 - 1;
    }
    else {
        logical_lower_bound_complex_x = -(logical_x_dimension - 1) / 2;
        logical_upper_bound_complex_x = (logical_x_dimension - 1) / 2;
        logical_lower_bound_real_x    = -(logical_x_dimension - 1) / 2;
        logical_upper_bound_real_x    = (logical_x_dimension - 1) / 2;
    }

    if (IsEven(logical_y_dimension) == true) {
        logical_lower_bound_complex_y = -logical_y_dimension / 2;
        logical_upper_bound_complex_y = logical_y_dimension / 2 - 1;
        logical_lower_bound_real_y    = -logical_y_dimension / 2;
        logical_upper_bound_real_y    = logical_y_dimension / 2 - 1;
    }
    else {
        logical_lower_bound_complex_y = -(logical_y_dimension - 1) / 2;
        logical_upper_bound_complex_y = (logical_y_dimension - 1) / 2;
        logical_lower_bound_real_y    = -(logical_y_dimension - 1) / 2;
        logical_upper_bound_real_y    = (logical_y_dimension - 1) / 2;
    }

    if (IsEven(logical_z_dimension) == true) {
        logical_lower_bound_complex_z = -logical_z_dimension / 2;
        logical_upper_bound_complex_z = logical_z_dimension / 2 - 1;
        logical_lower_bound_real_z    = -logical_z_dimension / 2;
        logical_upper_bound_real_z    = logical_z_dimension / 2 - 1;
    }
    else {
        logical_lower_bound_complex_z = -(logical_z_dimension - 1) / 2;
        logical_upper_bound_complex_z = (logical_z_dimension - 1) / 2;
        logical_lower_bound_real_z    = -(logical_z_dimension - 1) / 2;
        logical_upper_bound_real_z    = (logical_z_dimension - 1) / 2;
    }
}

//!>  \brief  Returns the physical address of the image origin

void Image::UpdatePhysicalAddressOfBoxCenter() {
    /*
    if (IsEven(logical_x_dimension)) physical_address_of_box_center_x = logical_x_dimension / 2;
    else physical_address_of_box_center_x = (logical_x_dimension - 1) / 2;

    if (IsEven(logical_y_dimension)) physical_address_of_box_center_y = logical_y_dimension / 2;
    else physical_address_of_box_center_y = (logical_y_dimension - 1) / 2;

    if (IsEven(logical_z_dimension)) physical_address_of_box_center_z = logical_z_dimension / 2;
    else physical_address_of_box_center_z = (logical_z_dimension - 1) / 2;
*/
    physical_address_of_box_center_x = logical_x_dimension / 2;
    physical_address_of_box_center_y = logical_y_dimension / 2;
    physical_address_of_box_center_z = logical_z_dimension / 2;
}

// Work out whether a given Fourier component has a Hermitian mate which is also described explicitely by the FFTW
bool Image::FourierComponentHasExplicitHermitianMate(int physical_index_x, int physical_index_y, int physical_index_z) {
    bool explicit_mate;

    explicit_mate = physical_index_x == 0 && !(physical_index_y == 0 && physical_index_z == 0);

    // We assume that the Y dimension is the non-flat one
    if (IsEven(logical_y_dimension)) {
        explicit_mate = explicit_mate && physical_index_y != physical_index_of_first_negative_frequency_y - 1;
    }

    if (logical_z_dimension > 1) {
        if (IsEven(logical_z_dimension)) {
            explicit_mate = explicit_mate && physical_index_z != physical_index_of_first_negative_frequency_z - 1;
        }
    }

    return explicit_mate;
}

// Work out whether a given Fourier component is a (redundant) Hermitian mate which is described explicitely by the FFTW
// but shouldn't be counted in statistics as an independent Fourier component
bool Image::FourierComponentIsExplicitHermitianMate(int physical_index_x, int physical_index_y, int physical_index_z) {
    bool explicit_mate = physical_index_x == 0 && (physical_index_y >= physical_index_of_first_negative_frequency_y ||
                                                   physical_index_z >= physical_index_of_first_negative_frequency_z);

    return explicit_mate;
}

//!> \brief   Apply a forward FT to the Image object. The FT is scaled.
//   The DC component is at (self%DIM(1)/2+1,self%DIM(2)/2+1,self%DIM(3)/2+1) (assuming even dimensions) or at (1,1,1)
//   by default.
//
//
//   For details on FFTW, see http://www.fftw.org/
//   A helpful page for understanding the output format: http://www.dsprelated.com/showmessage/102675/1.php
//   A helpful page to learn about vectorization and FFTW benchmarking:
//   http://www.dsprelated.com/showmessage/76779/1.php \todo   Check this:
//   http://objectmix.com/fortran/371439-ifort-openmp-fftw-problem.html
//
//
//
// 	A note on scaling: by default, we divide by N, the number of pixels. This ensures that after we do an inverse FT
// (without further scaling),
//	we will return to our original values. However, it means that while in Fourier space, the amplitudes are too high,
//by a factor of sqrt(N),
//  such that, for example, Parserval's theorem is not satisfied.
void Image::ForwardFFT(bool should_scale) {

    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Image already in Fourier space");
    MyDebugAssertTrue(planned, "FFT's not planned");

    fftwf_execute_dft_r2c(plan_fwd, real_values, reinterpret_cast<fftwf_complex*>(complex_values));

    if (should_scale) {
        DivideByConstant(float(number_of_real_space_pixels));
    }

    // Set the image type

    is_in_real_space = false;
}

void Image::BackwardFFT() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image already in real space");

    fftwf_execute_dft_c2r(plan_bwd, reinterpret_cast<fftwf_complex*>(complex_values), real_values);

    // Set the image type

    is_in_real_space = true;
}

//!> \brief Divide all voxels by a constant value (this is actually done as a multiplication by the inverse)

void Image::DivideByConstant(float constant_to_divide_by) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    float inverse = 1. / constant_to_divide_by;
    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] *= inverse;
    }
}

void Image::MultiplyAddConstant(float constant_to_multiply_by, float constant_to_add) {
    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] = real_values[pixel_counter] * constant_to_multiply_by + constant_to_add;
    }
}

void Image::AddMultiplyConstant(float constant_to_add, float constant_to_multiply_by) {
    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] = (real_values[pixel_counter] + constant_to_add) * constant_to_multiply_by;
    }
}

void Image::AddMultiplyAddConstant(float first_constant_to_add, float constant_to_multiply_by,
                                   float second_constant_to_add) {
    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] =
            (real_values[pixel_counter] + first_constant_to_add) * constant_to_multiply_by + second_constant_to_add;
    }
}

//!> \brief Multiply all voxels by a constant value

// inline
void Image::MultiplyByConstant(float constant_to_multiply_by) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] *= constant_to_multiply_by;
    }
}

void Image::TakeReciprocalRealValues(float zeros_become) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        if (real_values[pixel_counter] != 0.0)
            real_values[pixel_counter] = 1.0 / real_values[pixel_counter];
        else
            real_values[pixel_counter] = zeros_become;
    }
}

void Image::InvertRealValues() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] = -real_values[pixel_counter];
    }
}

void Image::SquareRealValues() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Must be in real space to square real values");
    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] *= real_values[pixel_counter];
    }
}

void Image::ExponentiateRealValues() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Must be in real space to square real values");
    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] = exp(real_values[pixel_counter]);
    }
}

void Image::SquareRootRealValues() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Must be in real space to square real values");
    MyDebugAssertFalse(HasNegativeRealValue(), "Image has negative value(s). Cannot compute square root.\n");
    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] = sqrtf(real_values[pixel_counter]);
    }
}

bool Image::IsConstant(bool compare_to_constant, float constant_to_compare) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Only makes sense for images in real space") MyDebugAssertTrue(
        !compare_to_constant || is_in_real_space, "compare_to_constant not implemented for Fourier transforms");

    long pixel_counter;

    if (is_in_real_space == true) {
        float constant;
        if (constant_to_compare)
            constant = constant_to_compare;
        else
            constant = real_values[0];

        pixel_counter = 0;
        for (int k = 0; k < logical_z_dimension; k++) {
            for (int j = 0; j < logical_y_dimension; j++) {
                for (int i = 0; i < logical_x_dimension; i++) {
                    if (real_values[pixel_counter] != constant)
                        return false;
                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }
    else {
        for (pixel_counter = 1; pixel_counter < real_memory_allocated / 2; pixel_counter++) {
            if (complex_values[pixel_counter] != complex_values[0])
                return false;
        }
    }

    return true;
}

bool Image::IsBinary() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Only makes sense for images in real space")

        long pixel_counter = 0;

    for (int k = 0; k < logical_z_dimension; k++) {
        for (int j = 0; j < logical_y_dimension; j++) {
            for (int i = 0; i < logical_x_dimension; i++) {
                if (real_values[pixel_counter] != 0 && real_values[pixel_counter] != 1)
                    return false;
                pixel_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }

    return true;
}

bool Image::HasNan() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    if (is_in_real_space == true) {
        long pixel_counter = 0;
        for (int k = 0; k < logical_z_dimension; k++) {
            for (int j = 0; j < logical_y_dimension; j++) {
                for (int i = 0; i < logical_x_dimension; i++) {
                    if (std::isnan(real_values[pixel_counter]))
                        return true;
                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }
    else {
        for (long pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++) {
            if (std::isnan(abs(complex_values[pixel_counter])))
                return true;
        }
    }
    return false;
}

bool Image::HasNegativeRealValue() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    long pixel_counter = 0;
    for (int k = 0; k < logical_z_dimension; k++) {
        for (int j = 0; j < logical_y_dimension; j++) {
            for (int i = 0; i < logical_x_dimension; i++) {
                if (real_values[pixel_counter] < 0.0)
                    return true;
                pixel_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }
    return false;
}

void Image::ReadSlices(ImageFile* input_file, long start_slice, long end_slice) {
    MyDebugAssertTrue(start_slice <= end_slice, "Start slice larger than end slice!");
    MyDebugAssertTrue(start_slice > 0, "Start slice is less than 0, the first slice is 1!");
    MyDebugAssertTrue(end_slice <= input_file->ReturnNumberOfSlices(),
                      "End slice is greater than number of slices in the file!");
    MyDebugAssertTrue(input_file->IsOpen(), "Image file is not open!");

    // check the allocations..

    int number_of_slices = (end_slice - start_slice) + 1;

    if (logical_x_dimension != input_file->ReturnXSize() || logical_y_dimension != input_file->ReturnYSize() ||
        logical_z_dimension != number_of_slices || is_in_memory == false) {
        Deallocate();
        Allocate(input_file->ReturnXSize(), input_file->ReturnYSize(), number_of_slices);
    }

    // We should be set up - so read in the correct values.
    // AT THE MOMENT, WE CAN ONLY READ REAL SPACE IMAGES, SO OVERRIDE THIS!!!!!

    is_in_real_space         = true;
    object_is_centred_in_box = true;

    input_file->ReadSlicesFromDisk(start_slice, end_slice, real_values);

    // we need to respace this to take into account the FFTW padding..

    AddFFTWPadding();
}

//!> \brief Read a set of slices from disk (FFTW padding is done automatically)

void Image::ReadSlices(MRCFile* input_file, long start_slice, long end_slice) {

    MyDebugAssertTrue(start_slice <= end_slice, "Start slice larger than end slice!");
    MyDebugAssertTrue(start_slice > 0, "Start slice is less than 0, the first slice is 1!");
    MyDebugAssertTrue(end_slice <= input_file->ReturnNumberOfSlices(),
                      "End slice is greater than number of slices in the file!");
    MyDebugAssertTrue(input_file->my_file->is_open(), "MRCFile not open!");

    // check the allocations..

    int number_of_slices = (end_slice - start_slice) + 1;

    if (logical_x_dimension != input_file->ReturnXSize() || logical_y_dimension != input_file->ReturnYSize() ||
        logical_z_dimension != number_of_slices || is_in_memory == false) {
        Deallocate();
        Allocate(input_file->ReturnXSize(), input_file->ReturnYSize(), number_of_slices);
    }

    // We should be set up - so read in the correct values.
    // AT THE MOMENT, WE CAN ONLY READ REAL SPACE IMAGES, SO OVERRIDE THIS!!!!!

    is_in_real_space         = true;
    object_is_centred_in_box = true;

    input_file->ReadSlicesFromDisk(start_slice, end_slice, real_values);

    // we need to respace this to take into account the FFTW padding..

    AddFFTWPadding();
}

void Image::ReadSlices(EerFile* input_file, long start_slice, long end_slice) {

    MyDebugAssertTrue(start_slice <= end_slice, "Start slice larger than end slice!");
    MyDebugAssertTrue(start_slice > 0, "Start slice is less than 0, the first slice is 1!");
    MyDebugAssertTrue(end_slice <= input_file->ReturnNumberOfSlices(),
                      "End slice is greater than number of slices in the file!");
    MyDebugAssertTrue(input_file->IsOpen(), "EER file is not open!");

    // check the allocations..

    int number_of_slices = (end_slice - start_slice) + 1;

    if (logical_x_dimension != input_file->ReturnXSize() || logical_y_dimension != input_file->ReturnYSize() ||
        logical_z_dimension != number_of_slices || is_in_memory == false) {
        Deallocate();
        Allocate(input_file->ReturnXSize(), input_file->ReturnYSize(),
                 number_of_slices); // ReturnXSize returns the super-res dimension, if appropriate
    }

    is_in_real_space         = true;
    object_is_centred_in_box = true;

    input_file->ReadSlicesFromDisk(start_slice, end_slice, real_values);

    // we need to respace this to take into account the FFTW padding..

    AddFFTWPadding();
}

//!> \brief Read a set of slices from disk (FFTW padding is done automatically)

void Image::ReadSlices(DMFile* input_file, long start_slice, long end_slice) {

    MyDebugAssertTrue(start_slice <= end_slice, "Start slice larger than end slice!");
    MyDebugAssertTrue(start_slice > 0, "Start slice is less than 0, the first slice is 1!");
    MyDebugAssertTrue(end_slice <= input_file->ReturnNumberOfSlices(),
                      "End slice is greater than number of slices in the file!");
    MyDebugAssertTrue(start_slice == end_slice, "Can only read one slice at a time from DM files. Sorry.")

        // check the allocations..

        int number_of_slices = (end_slice - start_slice) + 1;

    if (logical_x_dimension != input_file->ReturnXSize() || logical_y_dimension != input_file->ReturnYSize() ||
        logical_z_dimension != number_of_slices || is_in_memory == false) {
        Deallocate();
        Allocate(input_file->ReturnXSize(), input_file->ReturnYSize(), number_of_slices);
    }

    // We should be set up - so read in the correct values.
    // AT THE MOMENT, WE CAN ONLY READ REAL SPACE IMAGES, SO OVERRIDE THIS!!!!!

    is_in_real_space         = true;
    object_is_centred_in_box = true;

    input_file->ReadSliceFromDisk(start_slice - 1, real_values); // DM indexes slices starting at 0

    // we need to respace this to take into account the FFTW padding..

    AddFFTWPadding();
}

//!> \brief Write a set of slices from disk (FFTW padding is done automatically)

void Image::WriteSlices(MRCFile* input_file, long start_slice, long end_slice) {

    if (input_file->do_nothing)
        return; // this happens when the user gave /dev/null as a filename

    MyDebugAssertTrue(start_slice <= end_slice, "Start slice larger than end slice!");
    MyDebugAssertTrue(input_file->IsOpen(), "MRCFile not open!");

    // THIS PROBABLY NEEDS ATTENTION..

    if (start_slice == 1) // if the start slice is one, we set the header to match the image
    {
        input_file->my_header.SetDimensionsImage(logical_x_dimension, logical_y_dimension);

        if (end_slice > input_file->ReturnNumberOfSlices()) {
            input_file->my_header.SetNumberOfImages(end_slice);
        }

        // input_file->WriteHeader();
        input_file->rewrite_header_on_close = true;
    }
    else // if the last slice is bigger than the current max number of slices, increase the max number of slices
    {
        if (end_slice > input_file->ReturnNumberOfSlices()) {
            input_file->my_header.SetNumberOfImages(end_slice);
        }

        input_file->rewrite_header_on_close = true;
    }

    MyDebugAssertTrue(logical_x_dimension == input_file->ReturnXSize() ||
                          logical_y_dimension == input_file->ReturnYSize(),
                      "Image dimensions (%i, %i) and file dimensions (%i, %i) differ!", logical_x_dimension,
                      logical_y_dimension, input_file->ReturnXSize(), input_file->ReturnYSize());

    // if the image is complex.. make a temp image and transform it..

    int number_of_slices = (end_slice - start_slice) + 1;

    if (is_in_real_space == false) {
        Image temp_image;
        temp_image.CopyFrom(this);
        temp_image.BackwardFFT();
        temp_image.RemoveFFTWPadding();
        input_file->WriteSlicesToDisk(start_slice, end_slice, temp_image.real_values);
    }
    else // real space
    {
        RemoveFFTWPadding();
        input_file->WriteSlicesToDisk(start_slice, end_slice, real_values);
        AddFFTWPadding(); // to go back
    }
}

void Image::WriteSlicesAndFillHeader(std::string wanted_filename, float wanted_pixel_size) {
    MRCFile output_file;
    output_file.OpenFile(wanted_filename, true);
    WriteSlices(&output_file, 1, logical_z_dimension);
    output_file.SetPixelSize(wanted_pixel_size);
    EmpiricalDistribution density_distribution;
    UpdateDistributionOfRealValues(&density_distribution);
    output_file.SetDensityStatistics(density_distribution.GetMinimum(), density_distribution.GetMaximum(),
                                     density_distribution.GetSampleMean(),
                                     sqrtf(density_distribution.GetSampleVariance()));
    output_file.CloseFile();
}

void Image::QuickAndDirtyWriteSlices(std::string filename, long first_slice_to_write, long last_slice_to_write,
                                     bool overwrite, float pixel_size) {
    MyDebugAssertTrue(first_slice_to_write > 0, "Slice is less than 1, first slice is 1");
    MRCFile output_file(filename, overwrite);
    WriteSlices(&output_file, first_slice_to_write, last_slice_to_write);
    if (pixel_size <= 0.0f)
        pixel_size = 1.0;
    output_file.SetPixelSize(pixel_size);
    output_file.WriteHeader();
}

void Image::QuickAndDirtyWriteSlice(std::string filename, long slice_to_write, bool overwrite, float pixel_size) {
    MyDebugAssertTrue(slice_to_write > 0, "Slice is less than 1, first slice is 1");
    MRCFile output_file(filename, overwrite);
    WriteSlice(&output_file, slice_to_write);
    if (pixel_size <= 0.0f)
        pixel_size = 1.0;
    output_file.SetPixelSize(pixel_size);
    output_file.WriteHeader();
}

void Image::QuickAndDirtyReadSlice(std::string filename, long slice_to_read) {
    ImageFile image_file(filename);
    ReadSlice(&image_file, slice_to_read);
}

void Image::QuickAndDirtyReadSlices(std::string filename, int first_slice_to_read, int last_slice_to_read) {
    ImageFile image_file(filename);
    ReadSlices(&image_file, first_slice_to_read, last_slice_to_read);
}
//!> \brief Take a contiguous set of values, and add the FFTW padding.

void Image::AddFFTWPadding() {
    MyDebugAssertTrue(is_in_memory, "Image not allocated!");

    int x, y, z;

    long current_write_position = real_memory_allocated - (1 + padding_jump_value);
    long current_read_position =
        (long(logical_x_dimension) * long(logical_y_dimension) * long(logical_z_dimension)) - 1;

    for (z = 0; z < logical_z_dimension; z++) {
        for (y = 0; y < logical_y_dimension; y++) {
            for (x = 0; x < logical_x_dimension; x++) {
                real_values[current_write_position] = real_values[current_read_position];
                current_write_position--;
                current_read_position--;
            }

            current_write_position -= padding_jump_value;
        }
    }
}

//!> \brief Take a set of FFTW padded values, and remove the padding.

void Image::RemoveFFTWPadding() {
    MyDebugAssertTrue(is_in_memory, "Image not allocated!");

    int x, y, z;

    long current_write_position = 0;
    long current_read_position  = 0;

    for (z = 0; z < logical_z_dimension; z++) {
        for (y = 0; y < logical_y_dimension; y++) {
            for (x = 0; x < logical_x_dimension; x++) {
                real_values[current_write_position] = real_values[current_read_position];
                current_write_position++;
                current_read_position++;
            }

            current_read_position += padding_jump_value;
        }
    }
}

void Image::SetToConstant(float wanted_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] = wanted_value;
    }
}

void Image::AddConstant(float wanted_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] += wanted_value;
    }
}

void Image::SetMaximumValue(float new_maximum_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    for (long address = 0; address < real_memory_allocated; address++) {
        real_values[address] = std::min(new_maximum_value, real_values[address]);
    }
}

void Image::SetMinimumValue(float new_minimum_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    for (long address = 0; address < real_memory_allocated; address++) {
        real_values[address] = std::max(new_minimum_value, real_values[address]);
    }
}

void Image::Binarise(float threshold_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    for (long address = 0; address < real_memory_allocated; address++) {
        if (real_values[address] >= threshold_value)
            real_values[address] = 1.0f;
        else
            real_values[address] = 0.0f;
    }
}

void Image::BinariseInverse(float threshold_value) {
    // Set values below the threshold to 1
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    for (long address = 0; address < real_memory_allocated; address++) {
        if (real_values[address] <= threshold_value)
            real_values[address] = 1.0f;
        else
            real_values[address] = 0.0f;
    }
}

void Image::SetMinimumAndMaximumValues(float new_minimum_value, float new_maximum_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    for (long address = 0; address < real_memory_allocated; address++) {
        real_values[address] = std::max(std::min(new_maximum_value, real_values[address]), new_minimum_value);
    }
}

float Image::ReturnMaximumDiagonalRadius() {
    if (is_in_real_space) {
        return sqrt(pow(physical_address_of_box_center_x, 2) + pow(physical_address_of_box_center_y, 2) +
                    pow(physical_address_of_box_center_z, 2));
    }
    else {
        return sqrt(pow(logical_lower_bound_complex_x * fourier_voxel_size_x, 2) +
                    pow(logical_lower_bound_complex_y * fourier_voxel_size_y, 2) +
                    pow(logical_lower_bound_complex_z * fourier_voxel_size_z, 2));
    }
}

void Image::GetMinMax(float& min_value, float& max_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Only real space supported");

    min_value = FLT_MAX;
    max_value = -FLT_MAX;

    int  i, j, k;
    long address = 0;

    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                if (real_values[address] < min_value)
                    min_value = real_values[address];
                if (real_values[address] > max_value)
                    max_value = real_values[address];

                address++;
            }
            address += padding_jump_value;
        }
    }
}

// END_FOR_STAND_ALONE_CTFFIND

/*
!>  \brief  return the maximum radius possible along a diagonal
pure function GetMaximumDiagonalRadius(self)  result(maximum_radius)

    ! Arguments

    class(image),   intent(in)  ::  self

    ! Variables

    real                        ::  maximum_radius  !<  In pixels (real-space) or reciprocal pixels (Fourier space)

    ! Start work
    if (self%IsInRealSpace()) then
        maximum_radius = sqrt(real(sum((self%physical_address_of_box_center-1)**2)))
    else
        maximum_radius = sqrt(  (self%logical_upper_bound_complex(1)*self%fourier_voxel_size(1))**2 &
                            +   (self%logical_upper_bound_complex(2)*self%fourier_voxel_size(2))**2 &
                            +   (self%logical_upper_bound_complex(3)*self%fourier_voxel_size(3))**2)
    endif


end function GetMaximumDiagonalRadius
*/

long Image::ReturnNumberofNonZeroPixels() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "Image not in Real Space");

    int  i, j, k;
    long number_of_non_zero_pixels = 0;
    long address                   = 0;

    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                if (real_values[address] != 0.0f)
                    number_of_non_zero_pixels++;

                address++;
            }
            address += padding_jump_value;
        }
    }

    return number_of_non_zero_pixels;
}

float Image::ReturnSigmaOfFourierValuesOnEdges() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == false, "Image not in Fourier Space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Only 2D currently supported");

    long   number_of_pixels = 0;
    double total            = 0.0;
    double total_squared    = 0.0;
    float  sigma;
    float  average_density;
    long   counter = 0;

    int x_counter;
    int y_counter;
    int logical_x;
    int logical_y;

    for (y_counter = 0; y_counter <= physical_upper_bound_complex_y; y_counter++) {
        logical_y = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(y_counter);

        for (x_counter = 0; x_counter <= physical_upper_bound_complex_x; x_counter++) {
            logical_x = ReturnFourierLogicalCoordGivenPhysicalCoord_X(x_counter);

            if (logical_x == logical_lower_bound_complex_x || y_counter == logical_lower_bound_complex_y ||
                x_counter == logical_upper_bound_complex_x || y_counter == logical_upper_bound_complex_y) {
                total += real(complex_values[counter]);
                total += imag(complex_values[counter]);

                total_squared += powf(real(complex_values[counter]), 2);
                total_squared += powf(imag(complex_values[counter]), 2);
                number_of_pixels += 2;
            }

            counter++;
        }
    }

    average_density = total / double(number_of_pixels);
    sigma           = sqrtf((total_squared / double(number_of_pixels)) - pow(average_density, 2));
    // wxPrintf("Average = %f, Sigma = %f\n", average_density, sigma);
    return sigma;
}

float Image::ReturnSigmaOfFourierValuesOnEdgesAndCorners() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == false, "Image not in Fourier Space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Only 2D currently supported");

    long   number_of_pixels = 0;
    double total            = 0.0;
    double total_squared    = 0.0;
    float  sigma;
    float  average_density;
    float  frequency_squared;
    long   counter = 0;

    int k;
    int j;
    int i;

    float x;
    float y;
    float z;

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                x = powf(i * fourier_voxel_size_x, 2);

                // compute squared radius, in units of reciprocal pixels

                frequency_squared = x + y + z;

                if (frequency_squared >= 0.249) {
                    total += real(complex_values[counter]);
                    total += imag(complex_values[counter]);

                    total_squared += powf(real(complex_values[counter]), 2);
                    total_squared += powf(imag(complex_values[counter]), 2);
                    number_of_pixels += 2;
                }

                counter++;
            }
        }
    }

    average_density = total / double(number_of_pixels);
    sigma           = sqrtf((total_squared / double(number_of_pixels)) - pow(average_density, 2));
    // wxPrintf("Average = %f, Sigma = %f\n", average_density, sigma);
    return sigma;
}

// BEGIN_FOR_STAND_ALONE_CTFFIND

float Image::ReturnAverageOfRealValuesOnEdges() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    double sum;
    long   number_of_pixels;
    int    pixel_counter;
    int    line_counter;
    int    plane_counter;
    long   address;

    sum              = 0.0;
    number_of_pixels = 0;
    address          = 0;

    if (logical_z_dimension == 1) {
        // Two-dimensional image

        // First line
        for (pixel_counter = 0; pixel_counter < logical_x_dimension; pixel_counter++) {
            sum += real_values[address];
            address++;
        }
        number_of_pixels += logical_x_dimension;
        address += padding_jump_value;
        // Other lines
        for (line_counter = 1; line_counter < logical_y_dimension - 1; line_counter++) {
            sum += real_values[address];
            address += logical_x_dimension - 1;
            sum += real_values[address];
            address += padding_jump_value + 1;
            number_of_pixels += 2;
        }
        // Last line
        for (pixel_counter = 0; pixel_counter < logical_x_dimension; pixel_counter++) {
            sum += real_values[address];
            address++;
        }
        number_of_pixels += logical_x_dimension;
    }
    else {
        // Three-dimensional volume

        // First plane
        for (line_counter = 0; line_counter < logical_y_dimension; line_counter++) {
            for (pixel_counter = 0; pixel_counter < logical_x_dimension; pixel_counter++) {
                sum += real_values[address];
                address++;
            }
            address += padding_jump_value;
        }
        number_of_pixels += logical_x_dimension * logical_y_dimension;
        // Other planes
        for (plane_counter = 1; plane_counter < logical_z_dimension - 1; plane_counter++) {
            for (line_counter = 0; line_counter < logical_y_dimension; line_counter++) {
                if (line_counter == 0 || line_counter == logical_y_dimension - 1) {
                    // First and last line of that section
                    for (pixel_counter = 0; pixel_counter < logical_x_dimension; pixel_counter++) {
                        sum += real_values[address];
                        address++;
                    }
                    address += padding_jump_value;
                    number_of_pixels += logical_x_dimension;
                }
                else {
                    // All other lines (only count first and last pixel)
                    sum += real_values[address];
                    address += logical_x_dimension - 1;
                    sum += real_values[address];
                    address += padding_jump_value + 1;
                    number_of_pixels += 2;
                }
            }
        }
        // Last plane
        for (line_counter = 0; line_counter < logical_y_dimension; line_counter++) {
            for (pixel_counter = 0; pixel_counter < logical_x_dimension; pixel_counter++) {
                sum += real_values[address];
                address++;
            }
            address += padding_jump_value;
        }
        number_of_pixels += logical_x_dimension * logical_y_dimension;
    }
    return sum / float(number_of_pixels);
}

float Image::ReturnAverageOfRealValuesInRing(float wanted_inner_radius, float wanted_outer_radius) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(wanted_outer_radius > wanted_inner_radius, "Radii don't make sense");

    double sum              = 0.0;
    long   address          = 0;
    long   number_of_pixels = 0;
    int    i;
    int    j;
    int    k;
    float  x;
    float  y;
    float  z;
    float  inner_radius_squared;
    float  outer_radius_squared;
    float  distance_from_center_squared;

    inner_radius_squared = powf(wanted_inner_radius, 2);
    outer_radius_squared = powf(wanted_outer_radius, 2);
    number_of_pixels     = 0;
    for (k = 0; k < logical_z_dimension; k++) {
        z = powf(k - physical_address_of_box_center_z, 2);

        for (j = 0; j < logical_y_dimension; j++) {
            y = powf(j - physical_address_of_box_center_y, 2);

            for (i = 0; i < logical_x_dimension; i++) {
                x = powf(i - physical_address_of_box_center_x, 2);

                distance_from_center_squared = x + y + z;

                if (distance_from_center_squared >= inner_radius_squared &&
                    distance_from_center_squared <= outer_radius_squared) {
                    sum += real_values[address];
                    number_of_pixels++;
                }
                address++;
            }
            address += padding_jump_value;
        }
    }
    if (number_of_pixels > 0) {
        return float(sum / number_of_pixels);
    }
    else {
        return 0.0;
    }
}

float Image::ReturnAverageOfRealValuesAtRadius(float wanted_mask_radius) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    double sum              = 0.0;
    long   address          = 0;
    long   number_of_pixels = 0;
    int    i;
    int    j;
    int    k;
    float  x;
    float  y;
    float  z;
    float  mask_radius_squared;
    float  distance_from_center_squared;

    mask_radius_squared = powf(wanted_mask_radius, 2);
    number_of_pixels    = 0;
    for (k = 0; k < logical_z_dimension; k++) {
        z = powf(k - physical_address_of_box_center_z, 2);

        for (j = 0; j < logical_y_dimension; j++) {
            y = powf(j - physical_address_of_box_center_y, 2);

            for (i = 0; i < logical_x_dimension; i++) {
                x = powf(i - physical_address_of_box_center_x, 2);

                distance_from_center_squared = x + y + z;

                if (fabsf(distance_from_center_squared - mask_radius_squared) < 4.0) {
                    sum += real_values[address];
                    number_of_pixels++;
                }
                address++;
            }
            address += padding_jump_value;
        }
    }
    if (number_of_pixels > 0) {
        return float(sum / number_of_pixels);
    }
    else {
        return 0.0;
    }
}

// Find the largest voxel value, only considering voxels which are at least a certain distance from the center and from
// the edge in each dimension
float Image::ReturnMaximumValue(float minimum_distance_from_center, float minimum_distance_from_edge) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    int       i, j, k;
    int       i_dist_from_center, j_dist_from_center, k_dist_from_center;
    float     maximum_value             = -std::numeric_limits<float>::max();
    const int last_acceptable_address_x = logical_x_dimension - minimum_distance_from_edge - 1;
    const int last_acceptable_address_y = logical_y_dimension - minimum_distance_from_edge - 1;
    const int last_acceptable_address_z = logical_z_dimension - minimum_distance_from_edge - 1;
    long      address                   = 0;

    for (k = 0; k < logical_z_dimension; k++) {
        if (logical_z_dimension > 1) {
            k_dist_from_center = abs(k - physical_address_of_box_center_z);
            if (k_dist_from_center < minimum_distance_from_center || k < minimum_distance_from_edge ||
                k > last_acceptable_address_z) {
                address += logical_y_dimension * (logical_x_dimension + padding_jump_value);
                continue;
            }
        }
        for (j = 0; j < logical_y_dimension; j++) {
            j_dist_from_center = abs(j - physical_address_of_box_center_y);
            if (j_dist_from_center < minimum_distance_from_center || j < minimum_distance_from_edge ||
                j > last_acceptable_address_y) {
                address += logical_x_dimension + padding_jump_value;
                continue;
            }
            for (i = 0; i < logical_x_dimension; i++) {
                i_dist_from_center = abs(i - physical_address_of_box_center_x);
                if (i_dist_from_center < minimum_distance_from_center || i < minimum_distance_from_edge ||
                    i > last_acceptable_address_x) {
                    address++;
                    continue;
                }

                maximum_value = std::max(maximum_value, real_values[address]);
                address++;
            }
            address += padding_jump_value;
        }
    }

    return maximum_value;
}

// Find the largest voxel value, only considering voxels which are at least a certain distance from the center and from
// the edge in each dimension
float Image::ReturnMinimumValue(float minimum_distance_from_center, float minimum_distance_from_edge) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    int       i, j, k;
    int       i_dist_from_center, j_dist_from_center, k_dist_from_center;
    float     minimum_value             = std::numeric_limits<float>::max();
    const int last_acceptable_address_x = logical_x_dimension - minimum_distance_from_edge - 1;
    const int last_acceptable_address_y = logical_y_dimension - minimum_distance_from_edge - 1;
    const int last_acceptable_address_z = logical_z_dimension - minimum_distance_from_edge - 1;
    long      address                   = 0;

    for (k = 0; k < logical_z_dimension; k++) {
        if (logical_z_dimension > 1) {
            k_dist_from_center = abs(k - physical_address_of_box_center_z);
            if (k_dist_from_center < minimum_distance_from_center || k < minimum_distance_from_edge ||
                k > last_acceptable_address_z) {
                address += logical_y_dimension * (logical_x_dimension + padding_jump_value);
                continue;
            }
        }
        for (j = 0; j < logical_y_dimension; j++) {
            j_dist_from_center = abs(j - physical_address_of_box_center_y);
            if (j_dist_from_center < minimum_distance_from_center || j < minimum_distance_from_edge ||
                j > last_acceptable_address_y) {
                address += logical_x_dimension + padding_jump_value;
                continue;
            }
            for (i = 0; i < logical_x_dimension; i++) {
                i_dist_from_center = abs(i - physical_address_of_box_center_x);
                if (i_dist_from_center < minimum_distance_from_center || i < minimum_distance_from_edge ||
                    i > last_acceptable_address_x) {
                    address++;
                    continue;
                }

                minimum_value = std::min(minimum_value, real_values[address]);
                address++;
            }
            address += padding_jump_value;
        }
    }

    return minimum_value;
}

// TODO: consolidate (reduce code duplication) by using an Empirical distribution object
float Image::ReturnMedianOfRealValues() {
    long   number_of_voxels = logical_x_dimension * logical_y_dimension * logical_z_dimension;
    float* buffer_array     = new float[number_of_voxels];

    float median_value;

    long address        = 0;
    long buffer_counter = 0;

    int i;
    int j;
    int k;

    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                buffer_array[buffer_counter] = real_values[address];

                buffer_counter++;
                address++;
            }

            address += padding_jump_value;
        }
    }

    std::sort(buffer_array, buffer_array + number_of_voxels - 1);
    median_value = buffer_array[number_of_voxels / 2];
    delete[] buffer_array;

    return median_value;
}

// TODO: consolidate (reduce code duplication) by using an Empirical distribution object
float Image::ReturnAverageOfRealValues(float wanted_mask_radius, bool invert_mask) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    double sum              = 0.0;
    long   address          = 0;
    long   number_of_pixels = 0;
    int    i;
    int    j;
    int    k;
    float  x;
    float  y;
    float  z;
    float  mask_radius_squared;
    float  distance_from_center_squared;

    if (wanted_mask_radius > 0.0) {
        mask_radius_squared = powf(wanted_mask_radius, 2);
        number_of_pixels    = 0;
        for (k = 0; k < logical_z_dimension; k++) {
            z = powf(k - physical_address_of_box_center_z, 2);

            for (j = 0; j < logical_y_dimension; j++) {
                y = powf(j - physical_address_of_box_center_y, 2);

                for (i = 0; i < logical_x_dimension; i++) {
                    x = powf(i - physical_address_of_box_center_x, 2);

                    distance_from_center_squared = x + y + z;

                    if (invert_mask) {
                        if (distance_from_center_squared > mask_radius_squared) {
                            sum += real_values[address];
                            number_of_pixels++;
                        }
                    }
                    else {
                        if (distance_from_center_squared <= mask_radius_squared) {
                            sum += real_values[address];
                            number_of_pixels++;
                        }
                    }
                    address++;
                }
                address += padding_jump_value;
            }
        }
        if (number_of_pixels > 0) {
            return float(sum / number_of_pixels);
        }
        else {
            return 0.0;
        }
    }
    else {
        for (k = 0; k < logical_z_dimension; k++) {
            for (j = 0; j < logical_y_dimension; j++) {
                for (i = 0; i < logical_x_dimension; i++) {
                    sum += real_values[address];
                    address++;
                }
                address += padding_jump_value;
            }
        }
    }
    return float(sum / (long(logical_x_dimension) * long(logical_y_dimension) * long(logical_z_dimension)));
}

void Image::UpdateDistributionOfRealValues(EmpiricalDistribution* my_distribution, float wanted_mask_radius,
                                           bool outside, float wanted_center_x, float wanted_center_y,
                                           float wanted_center_z) {

    MyDebugAssertTrue(is_in_real_space, "Image must be in real space");

    int i;
    int j;
    int k;

    float x;
    float y;
    float z;

    long pixel_counter = 0;

    float distance_from_center_squared;
    float mask_radius_squared;
    float edge;
    float center_x;
    float center_y;
    float center_z;

    if (wanted_center_x == 0.0 && wanted_center_y == 0.0 && wanted_center_z == 0.0) {
        center_x = physical_address_of_box_center_x;
        center_y = physical_address_of_box_center_y;
        center_z = physical_address_of_box_center_z;
    }
    else {
        center_x = wanted_center_x;
        center_y = wanted_center_y;
        center_z = wanted_center_z;
    }

    if (wanted_mask_radius > 0.0) {
        mask_radius_squared = powf(wanted_mask_radius, 2);
        for (k = 0; k < logical_z_dimension; k++) {
            z = powf(k - center_z, 2);

            for (j = 0; j < logical_y_dimension; j++) {
                y = powf(j - center_y, 2);

                for (i = 0; i < logical_x_dimension; i++) {
                    x = powf(i - center_x, 2);

                    distance_from_center_squared = x + y + z;

                    if (outside) {
                        if (distance_from_center_squared > mask_radius_squared) {
                            my_distribution->AddSampleValue(real_values[pixel_counter]);
                        }
                    }
                    else {
                        if (distance_from_center_squared <= mask_radius_squared) {
                            my_distribution->AddSampleValue(real_values[pixel_counter]);
                        }
                    }
                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }
    else {
        for (k = 0; k < logical_z_dimension; k++) {
            for (j = 0; j < logical_y_dimension; j++) {
                for (i = 0; i < logical_x_dimension; i++) {
                    my_distribution->AddSampleValue(real_values[pixel_counter]);
                    pixel_counter++;
                }
                pixel_counter += padding_jump_value;
            }
        }
    }
}

EmpiricalDistribution Image::ReturnDistributionOfRealValues(float wanted_mask_radius, bool outside,
                                                            float wanted_center_x, float wanted_center_y,
                                                            float wanted_center_z) {
    MyDebugAssertTrue(is_in_real_space, "Image must be in real space");

    EmpiricalDistribution my_distribution;

    UpdateDistributionOfRealValues(&my_distribution, wanted_mask_radius, outside, wanted_center_x, wanted_center_y,
                                   wanted_center_z);

    return my_distribution;
}

void Image::ComputeAverageAndSigmaOfValuesInSpectrum(float minimum_radius, float maximum_radius, float& average,
                                                     float& sigma, int cross_half_width) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(maximum_radius > minimum_radius, "Maximum radius must be greater than minimum radius");
    MyDebugAssertTrue(logical_z_dimension == 1, "Meant for images, not volumes");

    // Private variables
    int                   i, j;
    float                 x_sq, y_sq, rad_sq;
    EmpiricalDistribution my_distribution;
    const float           min_rad_sq          = powf(minimum_radius, 2);
    const float           max_rad_sq          = powf(maximum_radius, 2);
    const float           cross_half_width_sq = powf(cross_half_width, 2);
    long                  address             = -1;

    for (j = 0; j < logical_y_dimension; j++) {
        y_sq = powf(j - physical_address_of_box_center_y, 2);
        if (y_sq <= cross_half_width_sq) {
            address += logical_x_dimension + padding_jump_value;
            continue;
        }
        for (i = 0; i < logical_x_dimension; i++) {
            address++;
            x_sq = powf(i - physical_address_of_box_center_x, 2);
            if (x_sq <= cross_half_width_sq)
                continue;
            rad_sq = x_sq + y_sq;
            if (rad_sq > min_rad_sq && rad_sq < max_rad_sq) {
                my_distribution.AddSampleValue(real_values[address]);
            }
        }
        address += padding_jump_value;
    }
    average = my_distribution.GetSampleMean();
    sigma   = sqrtf(my_distribution.GetSampleVariance());
}

void Image::ZeroCentralPixel() {
    if (is_in_real_space == false) {
        complex_values[0] = 0.0f * I + 0.0f;
    }
    else {
        MyDebugAssertTrue(logical_z_dimension == 1, "Meant for images, not volumes");

        int  i, j;
        long address = 0;

        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                if (j == physical_address_of_box_center_y && i == physical_address_of_box_center_x) {
                    real_values[address] = 0;
                }
                address++;
            }
            address += padding_jump_value;
        }
    }
}

void Image::SetMaximumValueOnCentralCross(float maximum_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Meant for images, not volumes");

    int  i, j;
    long address = 0;

    for (j = 0; j < logical_y_dimension; j++) {
        for (i = 0; i < logical_x_dimension; i++) {
            if (j == physical_address_of_box_center_y || i == physical_address_of_box_center_x) {
                real_values[address] = std::min(maximum_value, real_values[address]);
            }
            address++;
        }
        address += padding_jump_value;
    }
}

/*
 * The iciness of an amplitude spectrum is defined in the FOCUS package
 * (Biyani et al ,2018):
 * "
 * The introduced measure, iciness, is the ratio of the intensity in the
 * resolution band between 3.5 and 3.9 A and the intensity in the
 * resolution band between 30 and  6  A,  which  gives  a  good  estimate
 * of  the  ice  crystal  content  of  the  vitreous  specimen.
 * Cryo-EM  images  of  correctly  vitrified  specimens  usually
 * have  an  iciness  value  below 1.0. Iciness values higher than
 * 1.5 in most cases indicate images that are unusable to pick single
 * particles.
 * "
 */
float Image::ReturnIcinessOfSpectrum(float pixel_size_in_Angstroms) {
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(is_in_memory, "Not in memory");
    MyDebugAssertTrue(logical_x_dimension == logical_y_dimension, "Not square");

    const float control_band_low_in_angstroms  = 30.0;
    const float control_band_high_in_angstroms = 6.0;
    const float test_band_low_in_angstroms     = 3.9;
    const float test_band_high_in_angstroms    = 3.5;

    float control_band_low  = float(logical_x_dimension) * pixel_size_in_Angstroms / control_band_low_in_angstroms;
    float control_band_high = float(logical_x_dimension) * pixel_size_in_Angstroms / control_band_high_in_angstroms;

    float test_band_low  = float(logical_x_dimension) * pixel_size_in_Angstroms / test_band_low_in_angstroms;
    float test_band_high = float(logical_x_dimension) * pixel_size_in_Angstroms / test_band_high_in_angstroms;

    float intensity_in_control_band = ReturnAverageOfRealValuesInRing(control_band_low, control_band_high);
    float intensity_in_test_band    = ReturnAverageOfRealValuesInRing(test_band_low, test_band_high);

    float iciness;

    if (intensity_in_control_band <= 0.0) {
        iciness = 0.0;
    }
    else {
        iciness = powf(intensity_in_test_band, 2) / powf(intensity_in_control_band, 2);
    }

    return iciness;
}

// The image is assumed to be an amplitude spectrum, which we want to correlate with a set of CTF parameters
float Image::GetCorrelationWithCTF(CTF ctf) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Meant for images, not volumes");
    MyDebugAssertTrue(ctf.GetLowestFrequencyForFitting() > 0, "Will not work with lowest frequency for fitting of 0.");

    // Local variables
    double      cross_product    = 0.0;
    double      norm_image       = 0.0;
    double      norm_ctf         = 0.0;
    long        number_of_values = 0;
    int         i, j;
    float       i_logi, j_logi;
    float       i_logi_sq, j_logi_sq;
    const float inverse_logical_x_dimension = 1.0 / float(logical_x_dimension);
    const float inverse_logical_y_dimension = 1.0 / float(logical_y_dimension);
    float       current_spatial_frequency_squared;
    const float lowest_freq  = powf(ctf.GetLowestFrequencyForFitting(), 2);
    const float highest_freq = powf(ctf.GetHighestFrequencyForFitting(), 2);
    long        address      = 0;
    float       current_azimuth;
    float       current_ctf_value;
    const int   central_cross_half_width = 10;
    float       astigmatism_penalty;

    // Loop over half of the image (ignore Friedel mates)
    for (j = 0; j < logical_y_dimension; j++) {
        // DNM: Moved test on j out of inner loop, loop i only as far as needed, use an address computed for each line
        // (speeds up ~13%)
        if (j < physical_address_of_box_center_y - central_cross_half_width ||
            j > physical_address_of_box_center_y + central_cross_half_width) {
            address   = j * (padding_jump_value + 2 * physical_address_of_box_center_x);
            j_logi    = float(j - physical_address_of_box_center_y) * inverse_logical_y_dimension;
            j_logi_sq = powf(j_logi, 2);
            for (i = 0; i < physical_address_of_box_center_x - central_cross_half_width; i++) {
                i_logi    = float(i - physical_address_of_box_center_x) * inverse_logical_x_dimension;
                i_logi_sq = powf(i_logi, 2);

                // Where are we?
                current_spatial_frequency_squared = j_logi_sq + i_logi_sq;

                if (current_spatial_frequency_squared > lowest_freq &&
                    current_spatial_frequency_squared < highest_freq) {
                    current_azimuth   = atan2f(j_logi, i_logi);
                    current_ctf_value = fabsf(ctf.Evaluate(current_spatial_frequency_squared, current_azimuth));
                    // accumulate results
                    number_of_values++;
                    cross_product += real_values[address + i] * current_ctf_value;
                    norm_image += pow(real_values[address + i], 2);
                    norm_ctf += pow(current_ctf_value, 2);

                } // end of test whether within min,max frequency range
            }
        }
    }

    // Compute the penalty due to astigmatism
    if (ctf.GetAstigmatismTolerance() > 0.0) {
        astigmatism_penalty =
            powf(ctf.GetAstigmatism(), 2) * 0.5 / powf(ctf.GetAstigmatismTolerance(), 2) / float(number_of_values);
    }
    else {
        astigmatism_penalty = 0.0;
    }

    // The final score
    return cross_product / sqrt(norm_image * norm_ctf) - astigmatism_penalty;
}

// DNM: Set up for doing correlations with a CTF by making tables of all the pixels that are included, and their
// frequencies and azimuths When called with addresses NULL, it simply returns the number of values needed in the arrays
// When called with proper addresses, it fills the array and computes norm_image and image_mean, which are constant
void Image::SetupQuickCorrelationWithCTF(CTF ctf, int& number_of_values, double& norm_image, double& image_mean,
                                         int* addresses, float* spatial_frequency_squared, float* azimuth) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Meant for images, not volumes");
    MyDebugAssertTrue(ctf.GetLowestFrequencyForFitting() > 0, "Will not work with lowest frequency for fitting of 0.");

    // Local variables
    int         i, j;
    float       i_logi, j_logi;
    float       i_logi_sq, j_logi_sq;
    const float inverse_logical_x_dimension = 1.0 / float(logical_x_dimension);
    const float inverse_logical_y_dimension = 1.0 / float(logical_y_dimension);
    float       current_spatial_frequency_squared;

    const float lowest_freq  = powf(ctf.GetLowestFrequencyForFitting(), 2);
    const float highest_freq = powf(ctf.GetHighestFrequencyForFitting(), 2);
    int         address      = 0;
    float       current_azimuth;
    const int   central_cross_half_width = 10;
    double      image_sum                = 0.;

    number_of_values = 0;
    norm_image       = 0;
    image_mean       = 0.;

    // Loop over half of the image (ignore Friedel mates)
    for (j = 0; j < logical_y_dimension; j++) {
        if (j < physical_address_of_box_center_y - central_cross_half_width ||
            j > physical_address_of_box_center_y + central_cross_half_width) {
            address   = j * (padding_jump_value + 2 * physical_address_of_box_center_x);
            j_logi    = float(j - physical_address_of_box_center_y) * inverse_logical_y_dimension;
            j_logi_sq = powf(j_logi, 2);
            for (i = 0; i < physical_address_of_box_center_x - central_cross_half_width; i++) {
                i_logi    = float(i - physical_address_of_box_center_x) * inverse_logical_x_dimension;
                i_logi_sq = powf(i_logi, 2);

                // Where are we?
                current_spatial_frequency_squared = j_logi_sq + i_logi_sq;

                if (current_spatial_frequency_squared > lowest_freq &&
                    current_spatial_frequency_squared < highest_freq) {
                    current_azimuth = atan2f(j_logi, i_logi);
                    if (addresses) {
                        addresses[number_of_values]                 = address + i;
                        spatial_frequency_squared[number_of_values] = current_spatial_frequency_squared;
                        azimuth[number_of_values]                   = current_azimuth;
                        image_sum += real_values[address + i];
                    }
                    number_of_values++;
                } // end of test whether within min,max frequency range
            }
        }
    }

    // Now get sum of squared deviations from mean, more accurate than using raw cross-products
    if (addresses) {
        image_mean = image_sum / number_of_values;
        for (i = 0; i < number_of_values; i++)
            norm_image += pow(real_values[addresses[i]] - image_mean, 2);
    }
}

// DNM: Computes correlation with the current CTF estimate given the pixel indexes, frequency and azimuth values
// It is about 30% faster than original and now returns true correlation coefficient
float Image::QuickCorrelationWithCTF(CTF ctf, int number_of_values, double norm_image, double image_mean,
                                     int* addresses, float* spatial_frequency_squared, float* azimuth) {

    // Local variables
    int    i, j;
    double cross_product = 0.0;
    double norm_ctf      = 0.0;
    double ctf_sum       = 0.;
    float  current_ctf_value;
    float  astigmatism_penalty;

    for (i = 0; i < number_of_values; i++) {
        j = addresses[i];
        current_ctf_value =
            fabsf(-sin(ctf.PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(spatial_frequency_squared[i], azimuth[i])));
        cross_product += real_values[j] * current_ctf_value;
        norm_ctf += pow(current_ctf_value, 2);
        ctf_sum += current_ctf_value;
    }

    // Compute the penalty due to astigmatism
    if (ctf.GetAstigmatismTolerance() > 0.0) {
        astigmatism_penalty =
            powf(ctf.GetAstigmatism(), 2) * 0.5 / powf(ctf.GetAstigmatismTolerance(), 2) / float(number_of_values);
    }
    else {
        astigmatism_penalty = 0.0;
    }

    // The final score: norm_image is already a sum of squared deviations from mean; norm_ctf requires adjustment to
    // give true CC
    return (cross_product - image_mean * ctf_sum) /
               sqrt(norm_image * (norm_ctf - ctf_sum * ctf_sum / number_of_values)) -
           astigmatism_penalty;
}

void Image::ApplyMirrorAlongY() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Meant for images, not volumes");

    int   i, j;
    long  address = logical_x_dimension + padding_jump_value;
    int   j_dist;
    float temp_value;

    for (j = 1; j < physical_address_of_box_center_y; j++) {
        j_dist = 2 * (physical_address_of_box_center_y - j) * (logical_x_dimension + padding_jump_value);

        for (i = 0; i < logical_x_dimension; i++) {
            temp_value                    = real_values[address];
            real_values[address]          = real_values[address + j_dist];
            real_values[address + j_dist] = temp_value;
            address++;
        }
        address += padding_jump_value;
    }

    // The column j=0 is undefined, we set it to the average of the values that were there before the mirror operation
    // was applied
    temp_value = 0;
    for (i = 0; i < logical_x_dimension; i++) {
        temp_value += real_values[i];
    }
    temp_value /= float(logical_x_dimension);
    for (i = 0; i < logical_x_dimension; i++) {
        real_values[i] = temp_value;
    }
}

void Image::InvertPixelOrder() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Meant for images, not volumes");

    Image buffer_image;
    buffer_image.Allocate(logical_x_dimension, logical_y_dimension, 1);
    buffer_image.SetToConstant(-1.0f);
    int  i, j;
    long start_address = 0;
    long end_address   = real_memory_allocated - 1 - padding_jump_value;

    for (j = 0; j < logical_y_dimension; j++) {
        for (i = 0; i < logical_x_dimension; i++) {
            buffer_image.real_values[start_address] = real_values[end_address];
            start_address++;
            end_address--;
        }

        start_address += padding_jump_value;
        end_address -= padding_jump_value;
    }

    Consume(&buffer_image);
}

void Image::AddImage(Image* other_image) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(logical_x_dimension == other_image->logical_x_dimension &&
                          logical_y_dimension == other_image->logical_y_dimension &&
                          logical_z_dimension == other_image->logical_z_dimension,
                      "Image dimensions do not match, Image  %d, %d, %d \nImage to be added %d, %d, %d",
                      logical_x_dimension, logical_y_dimension, logical_z_dimension, other_image->logical_x_dimension,
                      other_image->logical_y_dimension, other_image->logical_z_dimension);
    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] += other_image->real_values[pixel_counter];
    }
}

void Image::SubtractImage(Image* other_image) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(HasSameDimensionsAs(other_image),
                      "Images should have same dimensions, but they don't: %i %i %i        %i %i %i",
                      logical_x_dimension, logical_y_dimension, logical_z_dimension, other_image->logical_x_dimension,
                      other_image->logical_y_dimension, other_image->logical_z_dimension);

    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] -= other_image->real_values[pixel_counter];
    }
}

void Image::SubtractSquaredImage(Image* other_image) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
        real_values[pixel_counter] -= powf(other_image->real_values[pixel_counter], 2);
    }
}

int Image::ReturnFourierLogicalCoordGivenPhysicalCoord_X(int physical_index) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(physical_index <= physical_upper_bound_complex_x, "index out of bounds");

    // if (physical_index >= physical_index_of_first_negative_frequency_x)
    if (physical_index > physical_address_of_box_center_x) {
        return physical_index - logical_x_dimension;
    }
    else
        return physical_index;
}

int Image::ReturnFourierLogicalCoordGivenPhysicalCoord_Y(int physical_index) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(physical_index <= physical_upper_bound_complex_y, "index out of bounds");

    if (physical_index >= physical_index_of_first_negative_frequency_y) {
        return physical_index - logical_y_dimension;
    }
    else
        return physical_index;
}

int Image::ReturnFourierLogicalCoordGivenPhysicalCoord_Z(int physical_index) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(physical_index <= physical_upper_bound_complex_z, "index out of bounds");

    if (physical_index >= physical_index_of_first_negative_frequency_z) {
        return physical_index - logical_z_dimension;
    }
    else
        return physical_index;
}

// END_FOR_STAND_ALONE_CTFFIND

// Pixel values in the image are replaced with the radial average from the image
void Image::AverageRadially() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    int   i;
    int   j;
    int   k;
    int   i_logi;
    int   j_logi;
    int   k_logi;
    float rad;
    long  address;

    Curve average;
    Curve number_of_values;

    if (is_in_real_space) {
        average.SetupXAxis(0.0, ReturnMaximumDiagonalRadius(), logical_x_dimension);
    }
    else {
        average.SetupXAxis(0.0, sqrt(2.0) * 0.5, logical_x_dimension);
    }
    number_of_values = average;

    // Compute the average curve
    Compute1DRotationalAverage(average, number_of_values);

    //
    address = 0;
    if (is_in_real_space) {
        for (k = 0; k < logical_z_dimension; k++) {
            k_logi = pow((k - physical_address_of_box_center_z), 2);
            for (j = 0; j < logical_y_dimension; j++) {
                j_logi = pow((j - physical_address_of_box_center_y), 2) + k_logi;
                for (i = 0; i < logical_x_dimension; i++) {
                    i_logi = pow((i - physical_address_of_box_center_x), 2) + j_logi;
                    //
                    rad = sqrt(float(i_logi));
                    //
                    real_values[address] = average.ReturnLinearInterpolationFromX(rad);

                    // Increment the address
                    address++;
                }
                // End of the line in real space
                address += padding_jump_value;
            }
        }
    }
    else {
        for (k = 0; k < logical_z_dimension; k++) {
            k_logi = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);
            for (j = 0; j < logical_y_dimension; j++) {
                j_logi = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2) + k_logi;
                for (i = 0; i < physical_upper_bound_complex_x; i++) {
                    i_logi = pow(i * fourier_voxel_size_x, 2) + j_logi;
                    //
                    if (FourierComponentIsExplicitHermitianMate(i, j, k)) {
                        address++;
                        continue;
                    }
                    rad = sqrt(float(i_logi));
                    //

                    complex_values[address] = (average.ReturnLinearInterpolationFromX(rad), 0.0f) + I * 0.0f;

                    // Increment the address
                    address++;
                }
            }
        }
    }
}

// BEGIN_FOR_STAND_ALONE_CTFFIND

//  \brief  Compute the 1D rotational average
//			The first element will be the value at the center/origin of the image.
//			It is assumed the X axis of the Curve object has been setup already. It should run from 0.0 to the maximum
//value 			possible, which is approximately sqrt(2)*0.5 in Fourier space or sqrt(2)*0.5*logical_dimension in real space 			(to
//compute this properly, use ReturnMaximumDiagonalRadius * fourier_voxel_size). To use 			The Fourier space radius
//convention in real space, give fractional_radius_in_real_space
void Image::Compute1DRotationalAverage(Curve& average, Curve& number_of_values, bool fractional_radius_in_real_space,
                                       bool average_real_parts) {

    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(average.number_of_points == number_of_values.number_of_points,
                      "Curves do not have the same number of points");
    MyDebugAssertTrue((is_in_real_space && !average_real_parts) || !is_in_real_space,
                      "average_real_part not possible for real space image");

    int   i;
    int   j;
    int   k;
    float rad;
    long  address;

    // Initialise
    average.ZeroYData();
    number_of_values.ZeroYData();
    address = 0;

    //
    if (is_in_real_space && !fractional_radius_in_real_space) {
        int i_logi, j_logi, k_logi;

        for (k = 0; k < logical_z_dimension; k++) {
            k_logi = pow((k - physical_address_of_box_center_z), 2);
            for (j = 0; j < logical_y_dimension; j++) {
                j_logi = pow((j - physical_address_of_box_center_y), 2) + k_logi;
                for (i = 0; i < logical_x_dimension; i++) {
                    i_logi = pow((i - physical_address_of_box_center_x), 2) + j_logi;
                    //
                    rad = sqrt(float(i_logi));
                    //
                    average.AddValueAtXUsingLinearInterpolation(rad, real_values[address], true);
                    number_of_values.AddValueAtXUsingLinearInterpolation(rad, 1.0, true);

                    // Increment the address
                    address++;
                }
                // End of the line in real space
                address += padding_jump_value;
            }
        }
    }
    else {
        float i_logi, j_logi, k_logi;

        if (is_in_real_space && fractional_radius_in_real_space) {
            for (k = 0; k < logical_z_dimension; k++) {
                k_logi = pow((k - physical_address_of_box_center_z) * fourier_voxel_size_z, 2);
                for (j = 0; j < logical_y_dimension; j++) {
                    j_logi = pow((j - physical_address_of_box_center_y) * fourier_voxel_size_y, 2) + k_logi;
                    for (i = 0; i < logical_x_dimension; i++) {
                        i_logi = pow((i - physical_address_of_box_center_x) * fourier_voxel_size_x, 2) + j_logi;
                        //
                        rad = sqrt(float(i_logi));
                        //
                        average.AddValueAtXUsingLinearInterpolation(rad, real_values[address], true);
                        number_of_values.AddValueAtXUsingLinearInterpolation(rad, 1.0, true);

                        // Increment the address
                        address++;
                    }
                    // End of the line in real space
                    address += padding_jump_value;
                }
            }
        }
        else {
            for (k = 0; k <= physical_upper_bound_complex_z; k++) {
                k_logi = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);
                for (j = 0; j <= physical_upper_bound_complex_y; j++) {
                    j_logi = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2) + k_logi;
                    for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                        i_logi = pow(i * fourier_voxel_size_x, 2) + j_logi;
                        //
                        if (FourierComponentIsExplicitHermitianMate(i, j, k)) {
                            address++;
                            continue;
                        }
                        rad = sqrt(float(i_logi));
                        //
                        if (average_real_parts)
                            average.AddValueAtXUsingLinearInterpolation(rad, real(complex_values[address]), true);
                        else
                            average.AddValueAtXUsingLinearInterpolation(rad, abs(complex_values[address]), true);
                        number_of_values.AddValueAtXUsingLinearInterpolation(rad, 1.0, true);

                        // Increment the address
                        address++;
                    }
                }
            }
        }
    }

    // Do the actual averaging
    for (int counter = 0; counter < average.number_of_points; counter++) {
        if (number_of_values.data_y[counter] != 0.0)
            average.data_y[counter] /= number_of_values.data_y[counter];
    }
}

void Image::Compute1DAmplitudeSpectrumCurve(Curve* curve_with_average_power, Curve* curve_with_number_of_values) {
    bool average_amplitudes_not_intensities = true;
    Compute1DPowerSpectrumCurve(curve_with_average_power, curve_with_number_of_values,
                                average_amplitudes_not_intensities);
}
// It is assumed the curve objects are already setup with an X axis in reciprocal pixels (i.e. origin is 0.0, Nyquist is
// 0.5) This function is called by Compute1DAmplitudeSpectrumCurve with average_amplitudes_not_intensities=true.
void Image::Compute1DPowerSpectrumCurve(Curve* curve_with_average_power, Curve* curve_with_number_of_values,
                                        bool average_amplitudes_not_intensities) {

    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image not in Fourier space");
    MyDebugAssertTrue(curve_with_average_power->number_of_points > 0, "Curve not setup");
    MyDebugAssertTrue(curve_with_average_power->data_x[0] == 0.0, "Curve does not start at x = 0\n");
    MyDebugAssertTrue(curve_with_average_power->data_x[curve_with_average_power->number_of_points - 1] >= 0.5,
                      "Curve does not go to at least x = 0.5 (it goes to %f)\n",
                      curve_with_average_power->data_x[curve_with_average_power->number_of_points - 1]);
    MyDebugAssertTrue(curve_with_average_power->number_of_points == curve_with_number_of_values->number_of_points,
                      "Curves need to have the same number of points");
    MyDebugAssertTrue(curve_with_average_power->data_x[0] == curve_with_number_of_values->data_x[0],
                      "Curves need to have the same starting point");
    MyDebugAssertTrue(curve_with_average_power->data_x[curve_with_average_power->number_of_points - 1] ==
                          curve_with_number_of_values->data_x[curve_with_number_of_values->number_of_points - 1],
                      "Curves need to have the same ending point");

    int   i, j, k;
    float sq_dist_x, sq_dist_y, sq_dist_z;
    int   counter;
    long  address;
    float spatial_frequency;
    int   number_of_hermitian_mates = 0;

    // Make sure the curves are clean
    curve_with_average_power->ZeroYData();
    curve_with_number_of_values->ZeroYData();

    // Get amplitudes and sum them into the curve object
    address = 0;
    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        sq_dist_z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);
        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            sq_dist_y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);
            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                if (FourierComponentIsExplicitHermitianMate(i, j, k)) {
                    number_of_hermitian_mates++;
                    address++;
                    continue;
                }
                else {
                    sq_dist_x         = powf(i * fourier_voxel_size_x, 2);
                    spatial_frequency = sqrtf(sq_dist_x + sq_dist_y + sq_dist_z);

                    // TODO: this could be made faster by doing both interpolations in one go, so one wouldn't have to
                    // work out twice between which points the interpolation will happen
                    if (average_amplitudes_not_intensities) {
                        curve_with_average_power->AddValueAtXUsingLinearInterpolation(
                            spatial_frequency,
                            sqrtf(real(complex_values[address]) * real(complex_values[address]) +
                                  imag(complex_values[address]) * imag(complex_values[address])),
                            true);
                    }
                    else {
                        curve_with_average_power->AddValueAtXUsingLinearInterpolation(
                            spatial_frequency,
                            real(complex_values[address]) * real(complex_values[address]) +
                                imag(complex_values[address]) * imag(complex_values[address]),
                            true);
                    }
                    curve_with_number_of_values->AddValueAtXUsingLinearInterpolation(spatial_frequency, 1.0, true);

                    address++;
                }
            }
        }
    }

    // Do the actual averaging
    for (counter = 0; counter < curve_with_average_power->number_of_points; counter++) {
        if (curve_with_number_of_values->data_y[counter] > 0.0) {
            curve_with_average_power->data_y[counter] /= curve_with_number_of_values->data_y[counter];
        }
        else {
            curve_with_average_power->data_y[counter] = 0.0;
        }
    }
}

// END_FOR_STAND_ALONE_CTFFIND

/*
 * Replace every voxel value (in Fourier space) with its spatial frequency (in reciprocal pixels, i.e. where Nyquist =
 * 0.5)
 */
void Image::ComputeSpatialFrequencyAtEveryVoxel() {
    int   xi, yi, zi;
    float x, y, z;
    float frequency;
    long  pixel_counter = 0;

    for (int k = 0; k <= physical_upper_bound_complex_z; k++) {
        zi = ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k);
        z  = powf(zi * fourier_voxel_size_z, 2);

        for (int j = 0; j <= physical_upper_bound_complex_y; j++) {
            yi = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j);
            y  = powf(yi * fourier_voxel_size_y, 2);

            for (int i = 0; i <= physical_upper_bound_complex_x; i++) {
                x = powf(i * fourier_voxel_size_x, 2);

                frequency                     = sqrtf(x + y + z);
                complex_values[pixel_counter] = frequency;

                pixel_counter++;
            }
        }
    }
}

// Return a histogram as a curve object.
void Image::ComputeHistogramOfRealValuesCurve(Curve* histogram_curve) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Image is in Fourier space");

    // Decide on the min/max and number of bins
    float     min_value, max_value;
    const int number_of_bins = 100; // TODO: better choice of number_of_bins, particularly in cases of very small images
    GetMinMax(min_value, max_value);

    histogram_curve->SetupXAxis(min_value, max_value, number_of_bins);
    histogram_curve->ZeroYData();

    // Loop over image
    int  current_bin;
    long address = 0;
    for (int k = 0; k < logical_z_dimension; k++) {
        for (int j = 0; j < logical_y_dimension; j++) {
            for (int i = 0; i < logical_x_dimension; i++) {
                current_bin = histogram_curve->ReturnIndexOfNearestPointFromX(real_values[address]);
                histogram_curve->data_y[current_bin] += 1.0;
                address++;
            }
            address += padding_jump_value;
        }
    }
}

// Apply an arbitrary filter to an image.
// The curve object should have X values in reciprocal pixels (Nyquist is 0.5, corner is ~sqrt(2.0)/2).
void Image::ApplyCurveFilter(Curve* filter_to_apply, float resolution_limit) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image not in Fourier space");

    int   i, j, k;
    float sq_dist_x, sq_dist_y, sq_dist_z;
    int   counter;
    long  pixel_counter = 0;
    float spatial_frequency;
    //	float resolution_limit_sq = powf(resolution_limit, 2);
    //	float resolution_limit_pixel = resolution_limit * logical_x_dimension;

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        sq_dist_z = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);
        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            sq_dist_y = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);
            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                sq_dist_x         = pow(i * fourier_voxel_size_x, 2);
                spatial_frequency = sqrt(sq_dist_x + sq_dist_y + sq_dist_z);

                if (spatial_frequency <= resolution_limit) {
                    complex_values[pixel_counter] *= filter_to_apply->ReturnLinearInterpolationFromX(spatial_frequency);
                }
                else
                    complex_values[pixel_counter] = 0.0f + I * 0.0f;
                pixel_counter++;
            }
        }
    }
}

// This assumes an X axis set up with
// SetupXAxis(0.0f, 0.5f * sqrtf(dimension_of_image), int((scale / 2.0f + 1.0f) * sqrtf(dimension_of_image) + 1.0f))
void Image::ApplyCurveFilterUninterpolated(Curve* filter_to_apply, float resolution_limit, float scale) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image not in Fourier space");

    int   i, j, k;
    float sq_dist_x, sq_dist_y, sq_dist_z;
    int   counter;
    int   index;
    long  pixel_counter = 0;
    float spatial_frequency;
    //	float resolution_limit_sq = powf(resolution_limit, 2);
    //	float resolution_limit_pixel = resolution_limit * logical_x_dimension;
    if (scale == 0.0f)
        scale = logical_x_dimension;

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        sq_dist_z = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);
        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            sq_dist_y = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);
            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                sq_dist_x         = pow(i * fourier_voxel_size_x, 2);
                spatial_frequency = sqrt(sq_dist_x + sq_dist_y + sq_dist_z);

                if (spatial_frequency <= resolution_limit) {
                    index = myroundint(spatial_frequency * scale);
                    complex_values[pixel_counter] *= filter_to_apply->data_y[index];
                }
                else
                    complex_values[pixel_counter] = 0.0f + I * 0.0f;
                pixel_counter++;
            }
        }
    }
}

void Image::ApplyLocalResolutionFilter(Image& local_resolution_map, float pixel_size, int wanted_number_of_levels) {
    MyDebugAssertTrue(HasSameDimensionsAs(&local_resolution_map),
                      "Local resolution map does not have expected dimensions");
    MyDebugAssertTrue(wanted_number_of_levels > 2, "Number of levels must be > 2");

    // Algorithm parameters
    const bool small_step_size = true;

    // Get some stats about the local resolution volume
    EmpiricalDistribution distribution_of_local_resolutions;
    local_resolution_map.UpdateDistributionOfRealValues(&distribution_of_local_resolutions,
                                                        float(logical_x_dimension) * 0.45);
    float min_res_Angstroms = distribution_of_local_resolutions.GetMinimum();
    float max_res_Angstroms = distribution_of_local_resolutions.GetMaximum();
    MyDebugPrint("Local res min = %f max = %f\n", min_res_Angstroms, max_res_Angstroms);

    // Prepare a copy of the input volume
    Image lp_volume;
    lp_volume = this;

    /*
     *  Filter parameters
     */
    float cosine_falloff_width = 10.0 / float(logical_x_dimension); // 5 Fourier voxels
    float filter_freq_step_size;
    if (small_step_size) {
        filter_freq_step_size = 1.0 / float(logical_x_dimension);
    }
    else {
        filter_freq_step_size = cosine_falloff_width;
    }

    /*
     * Setup a curve object to act as filter
     */
    Curve current_filter;
    current_filter.SetupXAxis(0.0, 1.0, logical_x_dimension);
    current_filter.SetYToConstant(1.0);

    /*
     * Loop over spatial frequencies, starting at high resolution
     */
    int   filter_counter       = 0;
    bool  on_lowest_resolution = false;
    float current_filter_freq  = pixel_size / min_res_Angstroms;
    float previous_filter_freq = -1.0;
    while (current_filter_freq >= pixel_size / max_res_Angstroms - filter_freq_step_size - 0.01) {
        filter_counter++;
        if (current_filter_freq < pixel_size / max_res_Angstroms)
            current_filter_freq = pixel_size / max_res_Angstroms;
        on_lowest_resolution = current_filter_freq <= pixel_size / max_res_Angstroms;
        wxPrintf("current filter freq = %f; pixel_size/max_res_Angstroms = %f; filter_freq_step_size = %f\n",
                 current_filter_freq, pixel_size / max_res_Angstroms, filter_freq_step_size);
        if (on_lowest_resolution)
            wxPrintf("On lowest resolution\n");

        // Apply filter
        lp_volume.ForwardFFT();
        MyDebugPrint("Filter #%i: resolution = %f\n", filter_counter, pixel_size / current_filter_freq);
        if (small_step_size) {
            // let's build the filter again
            current_filter.SetYToConstant(1.0);
            // undo effect of filter from previous iteration
            if (filter_counter > 1)
                current_filter.ApplyCosineMask(previous_filter_freq - 0.75 * cosine_falloff_width, cosine_falloff_width,
                                               true);
            // apply this iteration's filter
            current_filter.ApplyCosineMask(current_filter_freq - 0.75 * cosine_falloff_width, cosine_falloff_width);
            // Now apply this filter to the image
            lp_volume.ApplyCurveFilter(&current_filter);
        }
        else {
            lp_volume.CosineMask(current_filter_freq, cosine_falloff_width);
        }
        lp_volume.BackwardFFT();
        /*
#ifdef DEBUG
        lp_volume.QuickAndDirtyWriteSlices(wxString::Format("dbg_fil_%02i.mrc",filter_counter).ToStdString(), 1,
lp_volume.logical_z_dimension); #endif
        */

        /*
         * Loop over the image and copy relevant voxels over
         * (note we are doing dumb "nearest-neighbour" interpolation between resolution levels.
         * This makes the end map look a bit funky, with abrupt changes. I think that's OK for
         * usage within the refinement loop, but for final map post-processing, we probably
         * want a linear interpolation step here to make things look smoother.
         *
         */
        if (false) // NEAREST
        {
            int   i, j, k;
            long  pixel_counter;
            float res_of_interest_min = pixel_size / (current_filter_freq + 0.5 * filter_freq_step_size);
            float res_of_interest_max = pixel_size / (current_filter_freq - 0.5 * filter_freq_step_size);
            if (on_lowest_resolution)
                res_of_interest_max = 99999.9;
            MyDebugPrint("Looking for areas with resolution between %f and %f\n", res_of_interest_min,
                         res_of_interest_max);

            pixel_counter = 0;
            for (k = 0; k < logical_z_dimension; k++) {
                for (j = 0; j < logical_y_dimension; j++) {
                    for (i = 0; i < logical_x_dimension; i++) {
                        if (local_resolution_map.real_values[pixel_counter] >= res_of_interest_min &&
                            local_resolution_map.real_values[pixel_counter] < res_of_interest_max) {
                            real_values[pixel_counter] = lp_volume.real_values[pixel_counter];
                        }
                        pixel_counter++;
                    }
                    pixel_counter += padding_jump_value;
                }
            }
        }
        else // LINEAR
        {
            int   i, j, k;
            long  pixel_counter;
            float freq_of_interest_min = current_filter_freq - filter_freq_step_size;
            float freq_of_interest_max = current_filter_freq + filter_freq_step_size;
            if (on_lowest_resolution)
                freq_of_interest_min = 0.00001;
            float current_res_freq;
            float weight;
            float inverse_filter_freq_step_size =
                0.01; // ALR 200606: this was not set and icpc gave a warning. Not sure what value is reasonable.
            MyDebugPrint("Looking for areas with resolution between %f and %f\n", pixel_size / freq_of_interest_min,
                         pixel_size / freq_of_interest_max);

            pixel_counter = 0;
            for (k = 0; k < logical_z_dimension; k++) {
                for (j = 0; j < logical_y_dimension; j++) {
                    for (i = 0; i < logical_x_dimension; i++) {
                        current_res_freq = pixel_size / local_resolution_map.real_values[pixel_counter];

                        if (current_res_freq >= freq_of_interest_min && current_res_freq < freq_of_interest_max) {
                            weight = 1.0 - abs(current_res_freq - current_filter_freq) * inverse_filter_freq_step_size;
                            real_values[pixel_counter] = lp_volume.real_values[pixel_counter] * weight;
                        }
                        pixel_counter++;
                    }
                    pixel_counter += padding_jump_value;
                }
            }
        }

        if (on_lowest_resolution)
            break;

        // Decrement the filter frequency
        previous_filter_freq = current_filter_freq;
        current_filter_freq -= filter_freq_step_size;
    }
}

// The output image will be allocated to the correct dimensions (half-volume, a la FFTW)
void Image::ComputeAmplitudeSpectrum(Image* amplitude_spectrum, bool signed_values) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image not in Fourier space");

    //
    int       i, j, k;
    long      address_in_amplitude_spectrum = 0;
    long      address_in_self               = 0;
    const int spectrum_logical_dim_x        = physical_upper_bound_complex_x + 1;
    const int spectrum_logical_dim_y        = physical_upper_bound_complex_y + 1;
    const int spectrum_logical_dim_z        = physical_upper_bound_complex_z + 1;

    //
    if (logical_z_dimension > 1) {
        amplitude_spectrum->Allocate(spectrum_logical_dim_x, spectrum_logical_dim_y, spectrum_logical_dim_z, true);
    }
    else {
        amplitude_spectrum->Allocate(spectrum_logical_dim_x, spectrum_logical_dim_y, true);
    }

    // Loop over the amplitude spectrum
    for (k = 0; k < amplitude_spectrum->logical_z_dimension; k++) {
        for (j = 0; j < amplitude_spectrum->logical_y_dimension; j++) {
            for (i = 0; i < amplitude_spectrum->logical_x_dimension; i++) {
                address_in_self = ReturnFourier1DAddressFromPhysicalCoord(i, j, k);
                if (!signed_values) {
                    amplitude_spectrum->real_values[address_in_amplitude_spectrum] =
                        abs(complex_values[address_in_self]);
                }
                else {
                    if (real(complex_values[address_in_self]) >= 0.0f)
                        amplitude_spectrum->real_values[address_in_amplitude_spectrum] =
                            abs(complex_values[address_in_self]);
                    else
                        amplitude_spectrum->real_values[address_in_amplitude_spectrum] =
                            -abs(complex_values[address_in_self]);
                }
                address_in_amplitude_spectrum++;
            }
            address_in_amplitude_spectrum += amplitude_spectrum->padding_jump_value;
        }
    }
}

// BEGIN_FOR_STAND_ALONE_CTFFIND

void Image::ComputeAmplitudeSpectrumFull2D(Image* amplitude_spectrum, bool calculate_phases, float phase_multiplier) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(amplitude_spectrum->is_in_memory, "Other image memory not allocated");
    MyDebugAssertTrue(HasSameDimensionsAs(amplitude_spectrum), "Images do not have same dimensions");
    MyDebugAssertFalse(is_in_real_space, "Image not in Fourier space");

    int ampl_addr_i;
    int ampl_addr_j;
    int image_addr_i;
    int image_addr_j;
    int i_mate;
    int j_mate;

    long address_in_amplitude_spectrum = 0;
    long address_in_self;

    float amplitude;
    float phase;

    // Loop over the amplitude spectrum
    for (ampl_addr_j = 0; ampl_addr_j < amplitude_spectrum->logical_y_dimension; ampl_addr_j++) {
        for (ampl_addr_i = 0; ampl_addr_i < amplitude_spectrum->logical_x_dimension; ampl_addr_i++) {
            address_in_self = ReturnFourier1DAddressFromLogicalCoord(
                ampl_addr_i - amplitude_spectrum->physical_address_of_box_center_x,
                ampl_addr_j - amplitude_spectrum->physical_address_of_box_center_y, 0);
            amplitude = abs(complex_values[address_in_self]);
            if (!calculate_phases) {
                amplitude_spectrum->real_values[address_in_amplitude_spectrum] = amplitude;
            }
            else {
                if (amplitude != 0.0f) {
                    if (ampl_addr_i >= amplitude_spectrum->physical_address_of_box_center_x)
                        phase = std::arg(complex_values[address_in_self]);
                    else
                        phase = std::arg(conj(complex_values[address_in_self]));
                }
                else
                    phase = 0.0f;
                phase *= phase_multiplier;
                phase = fmodf(phase, 2.0f * (float)PI);
                if (phase > PI)
                    phase -= 2.0f * PI;
                if (phase <= -PI)
                    phase += 2.0f * PI;
                amplitude_spectrum->real_values[address_in_amplitude_spectrum] = phase;
            }
            address_in_amplitude_spectrum++;
        }
        address_in_amplitude_spectrum += amplitude_spectrum->padding_jump_value;
    }
    // Done
    amplitude_spectrum->is_in_real_space         = true;
    amplitude_spectrum->object_is_centred_in_box = true;
}

void Image::ComputeFilteredAmplitudeSpectrumFull2D(Image* average_spectrum_masked, Image* current_power_spectrum,
                                                   float& average, float& sigma, float minimum_resolution,
                                                   float maximum_resolution, float pixel_size_for_fitting) {

    // This is pulled from the ctffind.cpp App - this filtering is used also in
    // wave_function_propagtor::ReturnImageContrast. Any changes here should affect both programs identically. The only
    // change I (Ben) made is to change amplitud_spectrum-> to this->

    int convolution_box_size;

    // Try to weaken cross artefacts
    this->ComputeAverageAndSigmaOfValuesInSpectrum(float(this->logical_x_dimension) * pixel_size_for_fitting /
                                                       minimum_resolution,
                                                   float(this->logical_x_dimension), average, sigma, 12);
    this->DivideByConstant(sigma);
    this->SetMaximumValueOnCentralCross(average / sigma + 10.0);
    //			average_spectrum_masked->CopyFrom(this);

    // this->QuickAndDirtyWriteSlice("dbg_average_spectrum_before_conv.mrc",1);

    // Compute low-pass filtered version of the spectrum
    convolution_box_size =
        int(float(this->logical_x_dimension) * pixel_size_for_fitting / minimum_resolution * sqrt(2.0));
    if (IsEven(convolution_box_size))
        convolution_box_size++;
    current_power_spectrum->Allocate(this->logical_x_dimension, this->logical_y_dimension, true);
    current_power_spectrum->SetToConstant(0.0); // According to valgrind, this avoid potential problems later on.
    this->SpectrumBoxConvolution(current_power_spectrum, convolution_box_size,
                                 float(this->logical_x_dimension) * pixel_size_for_fitting / minimum_resolution);

    // current_power_spectrum->QuickAndDirtyWriteSlice("dbg_spec_convoluted.mrc",1);

    // Subtract low-pass-filtered spectrum from the spectrum. This should remove the background slope.
    this->SubtractImage(current_power_spectrum);

    //			this->QuickAndDirtyWriteSlice("dbg_spec_before_thresh.mrc",1);

    // Threshold high values
    this->SetMaximumValue(this->ReturnMaximumValue(3, 3));

    //			convolution_box_size = int( float(this->logical_x_dimension) * pixel_size_for_fitting / minimum_resolution /
    //sqrt(2.0) ); 			if (IsEven(convolution_box_size)) convolution_box_size++; 			current_power_spectrum->SetToConstant(0.0);
    //// According to valgrind, this avoid potential problems later on.
    //			average_spectrum_masked->SpectrumBoxConvolution(current_power_spectrum,convolution_box_size,float(average_spectrum_masked->logical_x_dimension)*pixel_size_for_fitting/minimum_resolution);
    //			average_spectrum_masked->SubtractImage(current_power_spectrum);
    //			average_spectrum_masked->SetMaximumValue(average_spectrum_masked->ReturnMaximumValue(3,3));

    average_spectrum_masked->CopyFrom(this);
    average_spectrum_masked->CosineMask(float(average_spectrum_masked->logical_x_dimension) * pixel_size_for_fitting /
                                            std::max(maximum_resolution, 8.0f),
                                        float(average_spectrum_masked->logical_x_dimension) * pixel_size_for_fitting /
                                            std::max(maximum_resolution, 4.0f),
                                        true);
    //			average_spectrum_masked->QuickAndDirtyWriteSlice("dbg_spec_before_thresh.mrc",1);
    //			average_spectrum_masked->CorrectSinc();
    //			average_spectrum_masked->CorrectSinc(float(average_spectrum_masked->logical_x_dimension)*pixel_size_for_fitting/std::max(maximum_resolution,
    //8.0f), 0.5, true, 0.0);
}

// END_FOR_STAND_ALONE_CTFFIND

// Compute the local mean and variance of the image at every point. The mask image must have the same dimensions as the
// image itself. Typically, the mask image would be 0.0 everywhere, except 1.0 in a central disk. This defines the area
// over which the local statistics will be computed. See van Heel 1987 and Roseman 2003 for more details.
void Image::ComputeLocalMeanAndVarianceMaps(Image* local_mean_map, Image* local_variance_map, Image* mask,
                                            long number_of_pixels_within_mask) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(local_mean_map->is_in_memory, "Mean map memory not allocated");
    MyDebugAssertTrue(local_variance_map->is_in_memory, "Variance map image memory not allocated");
    MyDebugAssertTrue(mask->is_in_memory, "Other image memory not allocated");
    MyDebugAssertTrue(HasSameDimensionsAs(local_mean_map), "Local mean map does not have same dimensions");
    MyDebugAssertTrue(HasSameDimensionsAs(local_variance_map), "Local variance map does not have same dimensions");
    MyDebugAssertTrue(HasSameDimensionsAs(mask), "Mask does not have the same dimensions");

    // For now, we assume all images start off in real space,
    // such that we can control normalization internally
    MyDebugAssertTrue(is_in_real_space, "Image should be in real space");
    MyDebugAssertTrue(local_mean_map->is_in_real_space, "Mean map should be in real space");
    MyDebugAssertTrue(local_variance_map->is_in_real_space, "Variance map should be in real space");
    MyDebugAssertTrue(mask->is_in_real_space, "Mask image should be in real space");

    //
    // Let's compute the mean first
    //

    // Make a couple of copies of the input image
    local_mean_map->CopyFrom(this);

    // Compute the local average in the micrograph, which is the convolution of
    // the micrograph with the mask
    // (because we will multiply with the mask FT later on, we do not need to normalize both FTs)
    mask->ForwardFFT(false);
    local_mean_map->ForwardFFT(true);
    local_mean_map->MultiplyPixelWise(*mask);
    local_mean_map->SwapRealSpaceQuadrants();
    local_mean_map->BackwardFFT();

    // Now divide by the number of pixels within the mask, and calculate the mean at the same time
    // (we will need it later
    float  inverse_number_of_pixels_within_mask = 1.0 / float(number_of_pixels_within_mask);
    long   address                              = 0;
    double local_mean_average                   = 0.0;
    for (int k = 0; k < logical_z_dimension; k++) {
        for (int j = 0; j < logical_y_dimension; j++) {
            for (int i = 0; i < logical_x_dimension; i++) {
                local_mean_average += local_mean_map->real_values[address];
                local_mean_map->real_values[address] *= inverse_number_of_pixels_within_mask;
                address++;
            }
            address += padding_jump_value;
        }
    }
    local_mean_average /= float(number_of_real_space_pixels * long(number_of_pixels_within_mask));

    //
    // Now let's compute the local variance
    // To avoid numeric instability and catastrophic cancellation (liable to occur for
    // example when the micrograph has a large mean and a small variance), we will
    // first subtract from the micrograph values the mean (it doesn't need to be exactly
    // the mean, just any value between min and max would do)
    //
    address = 0;
    for (int k = 0; k < logical_z_dimension; k++) {
        for (int j = 0; j < logical_y_dimension; j++) {
            for (int i = 0; i < logical_x_dimension; i++) {
                local_variance_map->real_values[address] = powf(real_values[address] - local_mean_average, 2);
                address++;
            }
            address += padding_jump_value;
        }
    }

    // Fourier transforms
    // (because we will multiply with the mask FT later on, we do not need to normalize both FTs)
    local_variance_map->ForwardFFT(true);

    // Convolute the squared micrograph with the mask image
    local_variance_map->MultiplyPixelWise(*mask);
    local_variance_map->SwapRealSpaceQuadrants();
    local_variance_map->BackwardFFT();

    // Compute the local variance (Eqn 10 in Roseman 2003)
    for (long address = 0; address < real_memory_allocated; address++) {
        local_variance_map->real_values[address] =
            (local_variance_map->real_values[address] * inverse_number_of_pixels_within_mask) -
            powf(local_mean_map->real_values[address] - local_mean_average, 2);
    }

    local_mean_map->object_is_centred_in_box     = true;
    local_variance_map->object_is_centred_in_box = true;

    MyDebugAssertFalse(local_mean_map->HasNan(), "Local mean map has NaN value(s)\n");
    MyDebugAssertFalse(local_variance_map->HasNan(), "Local variance map has NaN value(s)\n");
    // MyDebugAssertFalse(local_variance_map->HasNegativeRealValue(),"Local variance map has negative value(s)\n");
}

// BEGIN_FOR_STAND_ALONE_CTFFIND

/*
 * Real-space box convolution meant for 2D amplitude spectra
 *
 * This is adapted from the MSMOOTH subroutine from CTFFIND3, with a different wrap-around behaviour.
 * Also, in this version, we loop over the full 2D, rather than just half - this runs faster because less logic within
 * the loop DNM rewrote this to be vastly faster
 */
void Image::SpectrumBoxConvolution(Image* output_image, int box_size, float minimum_radius) {
    MyDebugAssertTrue(IsEven(box_size) == false, "Box size must be odd");
    MyDebugAssertTrue(logical_z_dimension == 1, "Volumes not supported");
    MyDebugAssertTrue(output_image->is_in_memory == true, "Output image not allocated");
    MyDebugAssertTrue(HasSameDimensionsAs(output_image), "Output image does not have same dimensions as image");

    // Variables
    const int   half_box_size              = (box_size - 1) / 2;
    const int   cross_half_width_to_ignore = 1;
    int         i;
    int         i_sq;
    int         ii;
    int         j;
    int         j_sq;
    int         jj;
    int         num_voxels;
    int         m;
    int         l;
    const float minimum_radius_sq = pow(minimum_radius, 2);
    float       radius_sq;
    const int   first_i_to_ignore = physical_address_of_box_center_x - cross_half_width_to_ignore;
    const int   last_i_to_ignore  = physical_address_of_box_center_x + cross_half_width_to_ignore;
    const int   first_j_to_ignore = physical_address_of_box_center_y - cross_half_width_to_ignore;
    const int   last_j_to_ignore  = physical_address_of_box_center_y + cross_half_width_to_ignore;

    // Addresses
    long address_within_output = 0;
    long address_within_input;

    // Starting and ending x indexes of one or two loops for each line
    int*   x1start      = new int[logical_x_dimension];
    int*   x1end        = new int[logical_x_dimension];
    int*   x2start      = new int[logical_x_dimension];
    int*   x2end        = new int[logical_x_dimension];
    int*   numInLineSum = new int[logical_x_dimension];
    float* lineSums     = new float[logical_x_dimension * logical_y_dimension];
    float  sum;
    int    ybase;

    // Get the limits for one or two loops for making line sums at each X position
    for (i = 0; i < logical_x_dimension; i++) {
        x1start[i] = i - half_box_size;
        x1end[i]   = i + half_box_size;
        x2start[i] = 0;
        x2end[i]   = -1;

        // Wrap around left edge
        if (x1start[i] < 0) {
            x2start[i] = x1start[i] + logical_x_dimension;
            x2end[i]   = logical_x_dimension - 1;
            x1start[i] = 0;
        }

        // Or wrap around right edge
        else if (x1end[i] >= logical_x_dimension) {
            x2end[i]   = x1end[i] - logical_x_dimension;
            x2start[i] = 0;
            x1end[i]   = logical_x_dimension - 1;
        }

        // Or handle intersection with the central cross by trimming or splitting into two loops
        else if (x1start[i] <= last_i_to_ignore && x1end[i] >= first_i_to_ignore) {
            if (x1start[i] >= first_i_to_ignore)
                x1start[i] = last_i_to_ignore + 1;
            else if (x1end[i] <= last_i_to_ignore)
                x1end[i] = first_i_to_ignore - 1;
            else {
                x2end[i]   = x1end[i];
                x2start[i] = last_i_to_ignore + 1;
                x1end[i]   = first_i_to_ignore - 1;
            }
        }
        numInLineSum[i] = x1end[i] + 1 - x1start[i];
        if (x2end[i] >= x2start[i])
            numInLineSum[i] += x2end[i] + 1 - x2start[i];
    }

    // Loop over Y positions for line sums
    for (jj = 0; jj < logical_y_dimension; jj++) {
        ybase = jj * (logical_x_dimension + padding_jump_value);

        // Form line sums at each X position
        for (i = 0; i < logical_x_dimension; i++) {
            sum = 0.;
            for (ii = x1start[i]; ii <= x1end[i]; ii++)
                sum += real_values[ii + ybase];
            for (ii = x2start[i]; ii <= x2end[i]; ii++)
                sum += real_values[ii + ybase];
            lineSums[i + jj * logical_x_dimension] = sum;
        }
    }

    // Loop over the output image
    for (j = 0; j < logical_y_dimension; j++) {
        j_sq = pow((j - physical_address_of_box_center_y), 2);

        for (i = 0; i < logical_x_dimension; i++) {
            i_sq = pow((i - physical_address_of_box_center_x), 2);

            radius_sq = float(i_sq + j_sq);

            if (radius_sq <= minimum_radius_sq) {
                output_image->real_values[address_within_output] = real_values[address_within_output];
            }
            else {
                output_image->real_values[address_within_output] = 0.0e0;
                num_voxels                                       = 0;

                // Loop over the lines to sum at this pixel to get the box sum
                for (m = -half_box_size; m <= half_box_size; m++) {
                    jj = j + m;
                    // wrap around
                    if (jj < 0) {
                        jj += logical_y_dimension;
                    }
                    if (jj >= logical_y_dimension) {
                        jj -= logical_y_dimension;
                    }

                    // In central cross?
                    // if ( abs(jj - physical_address_of_box_center_y) <= cross_half_width_to_ignore ) { continue; }
                    if (jj >= first_j_to_ignore && jj <= last_j_to_ignore) {
                        continue;
                    }

                    output_image->real_values[address_within_output] += lineSums[i + jj * logical_x_dimension];
                    num_voxels += numInLineSum[i];

                } // end of loop over the box

                if (num_voxels == 0) {
                    // DNM: if it happens, surely that should be from same address not whatever address_within_input was
                    output_image->real_values[address_within_output] = real_values[address_within_output];
                }
                else {
                    output_image->real_values[address_within_output] /= float(num_voxels);
                }
            }

            address_within_output++;
        }
        address_within_output += output_image->padding_jump_value;
    }

    delete[] x1start;
    delete[] x1end;
    delete[] x2start;
    delete[] x2end;
    delete[] lineSums;
    delete[] numInLineSum;
}

/*

void Image::SpectrumBoxConvolution(Image *output_image, int box_size, float minimum_radius)
{
    MyDebugAssertTrue(IsEven(box_size) == false,"Box size must be odd");
    MyDebugAssertTrue(logical_z_dimension == 1,"Volumes not supported");
    MyDebugAssertTrue(output_image->is_in_memory == true,"Output image not allocated");
    MyDebugAssertTrue(HasSameDimensionsAs(output_image),"Output image does not have same dimensions as image");

    // Variables
    int half_box_size = (box_size-1)/2;
    int cross_half_width_to_ignore = 1;
    int i;
    int i_friedel;
    int i_sq;
    int ii;
    int ii_friedel;
    int ii_sq;
    int iii;
    int j;
    int j_friedel;
    int j_sq;
    int jj;
    int jj_friedel;
    int jj_sq;
    int jjj;
    float radius;
    int num_voxels;
    int m;
    int l;

    // Addresses
    long address_within_output = 0;
    long address_within_input;

    // Loop over the output image. To save time, we only loop over one half of the image [BUG: actually this is looping
over the full image! for (j = 0; j < logical_y_dimension; j++)
    {
        j_friedel = 2 * physical_address_of_box_center_y - j;
        j_sq = powf((j - physical_address_of_box_center_y),2);

        for (i = 0; i < logical_x_dimension; i++)
        {
            i_friedel = 2 * physical_address_of_box_center_x - i;
            i_sq = powf((i - physical_address_of_box_center_x),2);

            //address_within_output = ReturnReal1DAddressFromPhysicalCoord(i,j,0);

            radius = sqrt(float(i_sq+j_sq));

            if ( radius <= minimum_radius )
            {
                output_image->real_values[address_within_output] = real_values[address_within_output];
            }
            else
            {
                output_image->real_values[address_within_output] = 0.0e0;
                num_voxels = 0;

                for ( m = - half_box_size; m <= half_box_size; m++)
                {
                    jj = j + m;
                    if (jj < 0) { jj += logical_y_dimension; }
                    if (jj >= logical_y_dimension) { jj -= logical_y_dimension; }
                    jj_friedel = 2 * physical_address_of_box_center_y - jj;
                    jj_sq = powf((jj - physical_address_of_box_center_y),2);

                    for ( l = - half_box_size; l <= half_box_size; l++)
                    {
                        ii = i + l;
                        if (ii < 0) { ii += logical_x_dimension; }
                        if (ii >= logical_x_dimension) { ii -= logical_x_dimension; }
                        ii_friedel = 2 * physical_address_of_box_center_x - ii;
                        ii_sq = powf((ii - physical_address_of_box_center_x),2);

                        // Friedel or not?
                        if ( ii > physical_address_of_box_center_x)
                        {
                            iii = ii_friedel;
                            jjj = jj_friedel;
                            if (jjj > logical_y_dimension - 1 || iii > logical_x_dimension - 1) { continue; }
                        }
                        else
                        {
                            iii = ii;
                            jjj = jj;
                        }

                        // In central cross?
                        if ( abs(iii - physical_address_of_box_center_x) <= cross_half_width_to_ignore || abs(jjj -
physical_address_of_box_center_y) <= cross_half_width_to_ignore ) { continue; }

                        address_within_input = ReturnReal1DAddressFromPhysicalCoord(iii,jjj,0);

                        if ( iii < logical_x_dimension && jjj < logical_y_dimension ) // it sometimes happens that we
end up on Nyquist Friedel mates that we don't have (perhaps this can be fixed)
                        {
                            output_image->real_values[address_within_output] += real_values[address_within_input];
                        }
                        num_voxels++; // not sure why this is not within the if branch, like the addition itself - is
this a bug?

                    }
                } // end of loop over the box

                if (num_voxels == 0)
                {
                    output_image->real_values[address_within_output] = real_values[address_within_input];
                }
                else
                {
                    output_image->real_values[address_within_output] /= float(num_voxels);
                }
            }

            if (j_friedel < logical_y_dimension && i_friedel < logical_x_dimension)
            {
                output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i_friedel,j_friedel,0)] =
output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i,j,0)];
            }
            address_within_output++;
        }
        address_within_output += output_image->padding_jump_value;
    }

    // There are a few pixels that are not set by the logical above
    for (i = physical_address_of_box_center_x + 1; i < logical_x_dimension; i++)
    {
        i_friedel = 2 * physical_address_of_box_center_x - i;
        output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i,0,0)]                     =
output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i_friedel,0,0)];
        output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i,logical_y_dimension-1,0)] =
output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i_friedel,logical_y_dimension - 1,0)];
    }
}
*/

//#pragma GCC push_options
//#pragma GCC optimize ("O0")
// Taper edges of image so that there are no sharp discontinuities in real space
// This is a re-implementation of the MRC program taperedgek.for (Richard Henderson, 1987)
void Image::TaperEdges() {
    MyDebugAssertTrue(is_in_memory, "Image not in memory");
    MyDebugAssertTrue(is_in_real_space, "Image not in real space");

    // Private variables
    const float fraction_averaging      = 30.0;
    const float fraction_tapering       = 30.0;
    const int   averaging_strip_width_x = int(logical_x_dimension / fraction_averaging); // 100
    const int   averaging_strip_width_y = int(logical_y_dimension / fraction_averaging);
    const int   averaging_strip_width_z = int(logical_z_dimension / fraction_averaging);
    const int   tapering_strip_width_x  = int(logical_x_dimension / fraction_tapering); // 500
    const int   tapering_strip_width_y  = int(logical_y_dimension / fraction_tapering);
    const int   tapering_strip_width_z  = int(logical_z_dimension / fraction_tapering);
    const int   smoothing_half_width_x  = 1; // 1
    const int   smoothing_half_width_y  = 1;
    const int   smoothing_half_width_z  = 1;
    int         current_dimension;
    int         number_of_dimensions;
    int         second_dimension;
    int         third_dimension;
    int         logical_current_dimension;
    int         logical_second_dimension;
    int         logical_third_dimension;
    int         current_tapering_strip_width;
    int         i, j, k;
    int         j_shift, k_shift;
    int         jj, kk;
    int         number_of_values_in_running_average;
    long        address;
    int         smoothing_half_width_third_dimension;
    int         smoothing_half_width_second_dimension;
    // 2D arrays
    float* average_for_current_edge_start         = NULL;
    float* average_for_current_edge_finish        = NULL;
    float* average_for_current_edge_average       = NULL;
    float* smooth_average_for_current_edge_start  = NULL;
    float* smooth_average_for_current_edge_finish = NULL;

    // Start work

    // Check dimensions of image are OK
    if (logical_x_dimension < 2 * tapering_strip_width_x || logical_y_dimension < 2 * tapering_strip_width_y) {
        MyPrintWithDetails("X,Y dimensions of input image are too small: %i %i\n", logical_x_dimension,
                           logical_y_dimension);
        DEBUG_ABORT;
    }
    if (logical_z_dimension > 1 && logical_z_dimension < 2 * tapering_strip_width_z) {
        MyPrintWithDetails("Z dimension is too small: %i\n", logical_z_dimension);
        DEBUG_ABORT;
    }

    if (logical_z_dimension > 1) {
        number_of_dimensions = 3;
    }
    else {
        number_of_dimensions = 2;
    }

    for (current_dimension = 1; current_dimension <= number_of_dimensions; current_dimension++) {
        switch (current_dimension) {
            case (1):
                second_dimension                      = 2;
                third_dimension                       = 3;
                logical_current_dimension             = logical_x_dimension;
                logical_second_dimension              = logical_y_dimension;
                logical_third_dimension               = logical_z_dimension;
                current_tapering_strip_width          = tapering_strip_width_x;
                smoothing_half_width_second_dimension = smoothing_half_width_y;
                smoothing_half_width_third_dimension  = smoothing_half_width_z;
                break;
            case (2):
                second_dimension                      = 1;
                third_dimension                       = 3;
                logical_current_dimension             = logical_y_dimension;
                logical_second_dimension              = logical_x_dimension;
                logical_third_dimension               = logical_z_dimension;
                current_tapering_strip_width          = tapering_strip_width_y;
                smoothing_half_width_second_dimension = smoothing_half_width_x;
                smoothing_half_width_third_dimension  = smoothing_half_width_z;
                break;
            case (3):
                second_dimension                      = 1;
                third_dimension                       = 2;
                logical_current_dimension             = logical_z_dimension;
                logical_second_dimension              = logical_x_dimension;
                logical_third_dimension               = logical_y_dimension;
                current_tapering_strip_width          = tapering_strip_width_z;
                smoothing_half_width_second_dimension = smoothing_half_width_x;
                smoothing_half_width_third_dimension  = smoothing_half_width_y;
                break;
        }

        // Allocate memory
        if (average_for_current_edge_start != NULL) {
            delete[] average_for_current_edge_start;
            delete[] average_for_current_edge_finish;
            delete[] average_for_current_edge_average;
            delete[] smooth_average_for_current_edge_start;
            delete[] smooth_average_for_current_edge_finish;
        }
        average_for_current_edge_start         = new float[logical_second_dimension * logical_third_dimension];
        average_for_current_edge_finish        = new float[logical_second_dimension * logical_third_dimension];
        average_for_current_edge_average       = new float[logical_second_dimension * logical_third_dimension];
        smooth_average_for_current_edge_start  = new float[logical_second_dimension * logical_third_dimension];
        smooth_average_for_current_edge_finish = new float[logical_second_dimension * logical_third_dimension];

        // Initialise memory
        for (i = 0; i < logical_second_dimension * logical_third_dimension; i++) {
            average_for_current_edge_start[i]         = 0.0;
            average_for_current_edge_finish[i]        = 0.0;
            average_for_current_edge_average[i]       = 0.0;
            smooth_average_for_current_edge_start[i]  = 0.0;
            smooth_average_for_current_edge_finish[i] = 0.0;
        }

        /*
         * Deal with X=0 and X=logical_x_dimension edges
         */
        i = 1;
        for (k = 1; k <= logical_third_dimension; k++) {
            for (j = 1; j <= logical_second_dimension; j++) {
                switch (current_dimension) {
                    case (1):
                        for (i = 1; i <= averaging_strip_width_x; i++) {
                            average_for_current_edge_start[(j - 1) + (k - 1) * logical_second_dimension] +=
                                real_values[ReturnReal1DAddressFromPhysicalCoord(i - 1, j - 1, k - 1)];
                            average_for_current_edge_finish[(j - 1) + (k - 1) * logical_second_dimension] +=
                                real_values[ReturnReal1DAddressFromPhysicalCoord(logical_x_dimension - i, j - 1,
                                                                                 k - 1)];
                        }
                        average_for_current_edge_start[(j - 1) + (k - 1) * logical_second_dimension] /=
                            float(averaging_strip_width_x);
                        average_for_current_edge_finish[(j - 1) + (k - 1) * logical_second_dimension] /=
                            float(averaging_strip_width_x);
                        break;
                    case (2):
                        for (i = 1; i <= averaging_strip_width_y; i++) {
                            average_for_current_edge_start[(j - 1) + (k - 1) * logical_second_dimension] +=
                                real_values[ReturnReal1DAddressFromPhysicalCoord(j - 1, i - 1, k - 1)];
                            average_for_current_edge_finish[(j - 1) + (k - 1) * logical_second_dimension] +=
                                real_values[ReturnReal1DAddressFromPhysicalCoord(j - 1, logical_y_dimension - i,
                                                                                 k - 1)];
                        }
                        average_for_current_edge_start[(j - 1) + (k - 1) * logical_second_dimension] /=
                            float(averaging_strip_width_y);
                        average_for_current_edge_finish[(j - 1) + (k - 1) * logical_second_dimension] /=
                            float(averaging_strip_width_y);
                        break;
                    case (3):
                        for (i = 1; i <= averaging_strip_width_z; i++) {
                            average_for_current_edge_start[(j - 1) + (k - 1) * logical_second_dimension] +=
                                real_values[ReturnReal1DAddressFromPhysicalCoord(j - 1, k - 1, i - 1)];
                            average_for_current_edge_finish[(j - 1) + (k - 1) * logical_second_dimension] +=
                                real_values[ReturnReal1DAddressFromPhysicalCoord(j - 1, k - 1,
                                                                                 logical_z_dimension - i)];
                        }
                        average_for_current_edge_start[(j - 1) + (k - 1) * logical_second_dimension] /=
                            float(averaging_strip_width_z);
                        average_for_current_edge_finish[(j - 1) + (k - 1) * logical_second_dimension] /=
                            float(averaging_strip_width_z);
                        break;
                }
            }
        }

        for (address = 0; address < logical_second_dimension * logical_third_dimension; address++) {
            average_for_current_edge_average[address] =
                0.5 * (average_for_current_edge_finish[address] + average_for_current_edge_start[address]);
            average_for_current_edge_start[address] -= average_for_current_edge_average[address];
            average_for_current_edge_finish[address] -= average_for_current_edge_average[address];
        }

        // Apply smoothing parallel to edge in the form of a running average
        for (k = 1; k <= logical_third_dimension; k++) {
            for (j = 1; j <= logical_second_dimension; j++) {
                number_of_values_in_running_average = 0;
                // Loop over neighbourhood of non-smooth arrays
                for (k_shift = -smoothing_half_width_third_dimension; k_shift <= smoothing_half_width_third_dimension;
                     k_shift++) {
                    kk = k + k_shift;
                    if (kk < 1 || kk > logical_third_dimension)
                        continue;
                    for (j_shift = -smoothing_half_width_second_dimension;
                         j_shift <= smoothing_half_width_second_dimension; j_shift++) {
                        jj = j + j_shift;
                        if (jj < 1 || jj > logical_second_dimension)
                            continue;
                        number_of_values_in_running_average++;

                        smooth_average_for_current_edge_start[(j - 1) + (k - 1) * logical_second_dimension] +=
                            average_for_current_edge_start[(jj - 1) + (kk - 1) * logical_second_dimension];
                        smooth_average_for_current_edge_finish[(j - 1) + (k - 1) * logical_second_dimension] +=
                            average_for_current_edge_finish[(jj - 1) + (kk - 1) * logical_second_dimension];
                    }
                }
                // Now we can compute the average
                smooth_average_for_current_edge_start[(j - 1) + (k - 1) * logical_second_dimension] /=
                    float(number_of_values_in_running_average);
                smooth_average_for_current_edge_finish[(j - 1) + (k - 1) * logical_second_dimension] /=
                    float(number_of_values_in_running_average);
            }
        }

        // Taper the image
        for (i = 1; i <= logical_current_dimension; i++) {
            if (i <= current_tapering_strip_width) {
                switch (current_dimension) {
                    case (1):
                        for (k = 1; k <= logical_third_dimension; k++) {
                            for (j = 1; j <= logical_second_dimension; j++) {
                                real_values[ReturnReal1DAddressFromPhysicalCoord(i - 1, j - 1, k - 1)] -=
                                    smooth_average_for_current_edge_start[(j - 1) +
                                                                          (k - 1) * logical_second_dimension] *
                                    float(current_tapering_strip_width - i + 1) / float(current_tapering_strip_width);
                            }
                        }
                        break;
                    case (2):
                        for (k = 1; k <= logical_third_dimension; k++) {
                            for (j = 1; j <= logical_second_dimension; j++) {
                                real_values[ReturnReal1DAddressFromPhysicalCoord(j - 1, i - 1, k - 1)] -=
                                    smooth_average_for_current_edge_start[(j - 1) +
                                                                          (k - 1) * logical_second_dimension] *
                                    float(current_tapering_strip_width - i + 1) / float(current_tapering_strip_width);
                            }
                        }
                        break;
                    case (3):
                        for (k = 1; k <= logical_third_dimension; k++) {
                            for (j = 1; j <= logical_second_dimension; j++) {
                                real_values[ReturnReal1DAddressFromPhysicalCoord(j - 1, k - 1, i - 1)] -=
                                    smooth_average_for_current_edge_start[(j - 1) +
                                                                          (k - 1) * logical_second_dimension] *
                                    float(current_tapering_strip_width - i + 1) / float(current_tapering_strip_width);
                            }
                        }
                        break;
                }
            }
            else if (i >= logical_current_dimension - current_tapering_strip_width + 1) {
                switch (current_dimension) {
                    case (1):
                        for (k = 1; k <= logical_third_dimension; k++) {
                            for (j = 1; j <= logical_second_dimension; j++) {
                                real_values[ReturnReal1DAddressFromPhysicalCoord(i - 1, j - 1, k - 1)] -=
                                    smooth_average_for_current_edge_finish[(j - 1) +
                                                                           (k - 1) * logical_second_dimension] *
                                    float(current_tapering_strip_width + i - logical_current_dimension) /
                                    float(current_tapering_strip_width);
                            }
                        }
                        break;
                    case (2):
                        for (k = 1; k <= logical_third_dimension; k++) {
                            for (j = 1; j <= logical_second_dimension; j++) {
                                real_values[ReturnReal1DAddressFromPhysicalCoord(j - 1, i - 1, k - 1)] -=
                                    smooth_average_for_current_edge_finish[(j - 1) +
                                                                           (k - 1) * logical_second_dimension] *
                                    float(current_tapering_strip_width + i - logical_current_dimension) /
                                    float(current_tapering_strip_width);
                            }
                        }
                        break;
                    case (3):
                        for (k = 1; k <= logical_third_dimension; k++) {
                            for (j = 1; j <= logical_second_dimension; j++) {
                                real_values[ReturnReal1DAddressFromPhysicalCoord(j - 1, k - 1, i - 1)] -=
                                    smooth_average_for_current_edge_finish[(j - 1) +
                                                                           (k - 1) * logical_second_dimension] *
                                    float(current_tapering_strip_width + i - logical_current_dimension) /
                                    float(current_tapering_strip_width);
                            }
                        }
                        break;
                }
            }
        }

    } // end of loop over dimensions

    // Cleanup
    delete[] average_for_current_edge_start;
    delete[] average_for_current_edge_finish;
    delete[] average_for_current_edge_average;
    delete[] smooth_average_for_current_edge_start;
    delete[] smooth_average_for_current_edge_finish;
}
//#pragma GCC pop_options

// END_FOR_STAND_ALONE_CTFFIND

void Image::Sine1D(int number_of_periods) {
    int   i;
    int   j;
    float sine_value_i;
    float sine_value_j;

    for (j = 0; j < logical_y_dimension; j++) {
        sine_value_j = sin(2 * PI * (j - physical_address_of_box_center_y) * number_of_periods / logical_y_dimension);
        for (i = 0; i < logical_x_dimension; i++) {
            sine_value_i =
                sin(2 * PI * (i - physical_address_of_box_center_x) * number_of_periods / logical_x_dimension);
            real_values[ReturnReal1DAddressFromPhysicalCoord(i, j, 0)] = sine_value_i + sine_value_j;
        }
    }
}

// BEGIN_FOR_STAND_ALONE_CTFFIND

// An alternative to ClipInto which only works for 2D real space clipping into larger image. Should be faster.
void Image::ClipIntoLargerRealSpace2D(Image* other_image, float wanted_padding_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(other_image->is_in_memory, "Other image Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Image must be in real space");
    MyDebugAssertTrue(object_is_centred_in_box, "real space image, not centred in box");
    MyDebugAssertTrue(logical_z_dimension == 1, "Image must be 2D");
    MyDebugAssertTrue(logical_x_dimension <= other_image->logical_x_dimension &&
                          logical_y_dimension <= other_image->logical_y_dimension,
                      "Image must be smaller than other image");

    other_image->is_in_real_space         = is_in_real_space;
    other_image->object_is_centred_in_box = object_is_centred_in_box;

    // Looping variables
    long address_in_self  = 0;
    long address_in_other = 0;

    int i;
    int j;

    // The address boudaries in the other_image for the input image data
    // If we are clipping a (2,2) image into a (4,4) image, we should be
    // copying into addresses 1 to 2 in both directions
    // If we are clipping a logical dimension of 2 into a dimension of 5,
    // we are copying into addresses 1 to 2
    const int i_lower_bound = other_image->physical_address_of_box_center_x - physical_address_of_box_center_x;
    const int j_lower_bound = other_image->physical_address_of_box_center_y - physical_address_of_box_center_y;
    const int i_upper_bound = i_lower_bound + logical_x_dimension - 1;
    const int j_upper_bound = j_lower_bound + logical_y_dimension - 1;

    // Loop over the other (larger) image
    for (j = 0; j < other_image->logical_y_dimension; j++) {
        // Check whether this line is outside of the original image
        if (j < j_lower_bound || j > j_upper_bound) {
            // Fill this line with the padding value
            for (i = 0; i < other_image->logical_x_dimension; i++) {
                other_image->real_values[address_in_other] = wanted_padding_value;
                address_in_other++;
            }
        }
        else {
            // This line is within the central region
            for (i = 0; i < other_image->logical_x_dimension; i++) {
                if (i < i_lower_bound || i > i_upper_bound) {
                    // We are near the beginning or the end of the line
                    other_image->real_values[address_in_other] = wanted_padding_value;
                }
                else {
                    other_image->real_values[address_in_other] = real_values[address_in_self];
                    address_in_self++;
                }
                address_in_other++;
            }
        }
        // We've reached the end of the line
        address_in_other += other_image->padding_jump_value;
        if (j >= j_lower_bound)
            address_in_self += padding_jump_value;
    }
}

// NOTE: THIS METHOD ADDS, it does not replace.

void Image::InsertOtherImageAtSpecifiedPosition(Image* other_image, int wanted_x_coord, int wanted_y_coord,
                                                int wanted_z_coord, float threshold_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(other_image->is_in_memory, "Other image Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "Only real space make sense");

    int kk;
    int k;
    int kk_logi;

    int jj;
    int jj_logi;
    int j;

    int ii;
    int ii_logi;
    int i;

    long pixel_counter = 0;

    for (kk = 0; kk < other_image->logical_z_dimension; kk++) {
        // kk_logi = kk - other_image->physical_address_of_box_center_z;
        k = physical_address_of_box_center_z + wanted_z_coord - other_image->physical_address_of_box_center_z + kk;
        //		k=0;

        for (jj = 0; jj < other_image->logical_y_dimension; jj++) {
            // jj_logi = jj - other_image->physical_address_of_box_center_y - 1;
            j = physical_address_of_box_center_y + wanted_y_coord - other_image->physical_address_of_box_center_y + jj;

            for (ii = 0; ii < other_image->logical_x_dimension; ii++) {
                // ii_logi = ii - other_image->physical_address_of_box_center_x - 1;
                i = physical_address_of_box_center_x + wanted_x_coord - other_image->physical_address_of_box_center_x +
                    ii;

                if (k < 0 || k >= logical_z_dimension || j < 0 || j >= logical_y_dimension || i < 0 ||
                    i >= logical_x_dimension) {
                }
                else {
                    if (other_image->real_values[pixel_counter] > threshold_value)
                        real_values[ReturnReal1DAddressFromPhysicalCoord(i, j, k)] +=
                            other_image->real_values[pixel_counter];
                }

                pixel_counter++;
            }

            pixel_counter += other_image->padding_jump_value;
        }
    }
}

// If you don't want to clip from the center, you can give wanted_coordinate_of_box_center_{x,y,z}. This will define the
// pixel in the image at which other_image will be centered. (0,0,0) means center of image.
void Image::ClipInto(Image* other_image, float wanted_padding_value, bool fill_with_noise, float wanted_noise_sigma,
                     int wanted_coordinate_of_box_center_x, int wanted_coordinate_of_box_center_y,
                     int wanted_coordinate_of_box_center_z) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(other_image->is_in_memory, "Other image Memory not allocated");
    MyDebugAssertFalse(is_in_real_space == true && fill_with_noise == true, "Fill with noise, only for fourier space");
    MyDebugAssertFalse((!is_in_real_space) &&
                           (wanted_coordinate_of_box_center_x != 0 || wanted_coordinate_of_box_center_y != 0 ||
                            wanted_coordinate_of_box_center_z != 0),
                       "Cannot clip off-center in Fourier space");

    long pixel_counter = 0;
    int  array_address = 0;

    int temp_logical_x;
    int temp_logical_y;
    int temp_logical_z;

    int kk;
    int k;
    int kk_logi;

    int jj;
    int jj_logi;
    int j;

    int ii;
    int ii_logi;
    int i;

    double junk;

    // take other following attributes

    other_image->is_in_real_space         = is_in_real_space;
    other_image->object_is_centred_in_box = object_is_centred_in_box;

    if (is_in_real_space == true) {
        MyDebugAssertTrue(object_is_centred_in_box, "real space image, not centred in box");

        for (kk = 0; kk < other_image->logical_z_dimension; kk++) {
            kk_logi = kk - other_image->physical_address_of_box_center_z;
            k       = physical_address_of_box_center_z + wanted_coordinate_of_box_center_z + kk_logi;

            for (jj = 0; jj < other_image->logical_y_dimension; jj++) {
                jj_logi = jj - other_image->physical_address_of_box_center_y;
                j       = physical_address_of_box_center_y + wanted_coordinate_of_box_center_y + jj_logi;

                for (ii = 0; ii < other_image->logical_x_dimension; ii++) {
                    ii_logi = ii - other_image->physical_address_of_box_center_x;
                    i       = physical_address_of_box_center_x + wanted_coordinate_of_box_center_x + ii_logi;

                    if (k < 0 || k >= logical_z_dimension || j < 0 || j >= logical_y_dimension || i < 0 ||
                        i >= logical_x_dimension) {
                        other_image->real_values[pixel_counter] = wanted_padding_value;
                    }
                    else {
                        other_image->real_values[pixel_counter] = ReturnRealPixelFromPhysicalCoord(i, j, k);
                    }

                    pixel_counter++;
                }

                pixel_counter += other_image->padding_jump_value;
            }
        }
    }
    else {
        for (kk = 0; kk <= other_image->physical_upper_bound_complex_z; kk++) {
            temp_logical_z = other_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Z(kk);

            // if (temp_logical_z > logical_upper_bound_complex_z || temp_logical_z < logical_lower_bound_complex_z)
            // continue;

            for (jj = 0; jj <= other_image->physical_upper_bound_complex_y; jj++) {
                temp_logical_y = other_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Y(jj);

                // if (temp_logical_y > logical_upper_bound_complex_y || temp_logical_y < logical_lower_bound_complex_y)
                // continue;

                for (ii = 0; ii <= other_image->physical_upper_bound_complex_x; ii++) {
                    temp_logical_x = ii;

                    // if (temp_logical_x > logical_upper_bound_complex_x || temp_logical_x <
                    // logical_lower_bound_complex_x) continue;

                    if (fill_with_noise == false)
                        other_image->complex_values[pixel_counter] = ReturnComplexPixelFromLogicalCoord(
                            temp_logical_x, temp_logical_y, temp_logical_z, wanted_padding_value + I * 0.0f);
                    else {

                        if (temp_logical_x < logical_lower_bound_complex_x ||
                            temp_logical_x > logical_upper_bound_complex_x ||
                            temp_logical_y < logical_lower_bound_complex_y ||
                            temp_logical_y > logical_upper_bound_complex_y ||
                            temp_logical_z < logical_lower_bound_complex_z ||
                            temp_logical_z > logical_upper_bound_complex_z) {
                            other_image->complex_values[pixel_counter] =
                                (global_random_number_generator.GetNormalRandom() * wanted_noise_sigma) +
                                (I * global_random_number_generator.GetNormalRandom() * wanted_noise_sigma);
                        }
                        else {
                            other_image->complex_values[pixel_counter] =
                                complex_values[ReturnFourier1DAddressFromLogicalCoord(temp_logical_x, temp_logical_y,
                                                                                      temp_logical_z)];
                        }
                    }
                    pixel_counter++;
                }
            }
        }

        // When we are clipping into a larger volume in Fourier space, there is a half-plane (vol) or half-line (2D
        // image) at Nyquist for which FFTW does not explicitly tell us the values. We need to fill them in.
        if (logical_y_dimension < other_image->logical_y_dimension ||
            logical_z_dimension < other_image->logical_z_dimension) {
            // For a 2D image
            if (logical_z_dimension == 1) {
                jj = physical_index_of_first_negative_frequency_y;
                for (ii = 0; ii <= physical_upper_bound_complex_x; ii++) {
                    other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii, jj, 0)] =
                        complex_values[ReturnFourier1DAddressFromPhysicalCoord(ii, jj, 0)];
                }
            }
            // For a 3D volume
            else {

                // Deal with the positive Nyquist of the 2nd dimension
                for (kk_logi = logical_lower_bound_complex_z; kk_logi <= logical_upper_bound_complex_z; kk_logi++) {
                    jj      = physical_index_of_first_negative_frequency_y;
                    jj_logi = logical_lower_bound_complex_y;
                    for (ii = 0; ii <= physical_upper_bound_complex_x; ii++) {
                        other_image
                            ->complex_values[other_image->ReturnFourier1DAddressFromLogicalCoord(ii, jj, kk_logi)] =
                            complex_values[ReturnFourier1DAddressFromLogicalCoord(ii, jj_logi, kk_logi)];
                    }
                }

                // Deal with the positive Nyquist in the 3rd dimension
                kk            = physical_index_of_first_negative_frequency_z;
                int kk_mirror = other_image->logical_z_dimension - physical_index_of_first_negative_frequency_z;
                // wxPrintf("\nkk = %i; kk_mirror = %i\n",kk,kk_mirror);
                int jj_mirror;
                // wxPrintf("Will loop jj from %i to %i\n",1,physical_index_of_first_negative_frequency_y);
                for (jj = 1; jj <= physical_index_of_first_negative_frequency_y; jj++) {
                    // jj_mirror = other_image->logical_y_dimension - jj;
                    jj_mirror = jj;
                    for (ii = 0; ii <= physical_upper_bound_complex_x; ii++) {
                        // wxPrintf("(1) ii = %i; jj = %i; kk = %i; jj_mirror = %i; kk_mirror =
                        // %i\n",ii,jj,kk,jj_mirror,kk_mirror);
                        other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii, jj, kk)] =
                            other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(
                                ii, jj_mirror, kk_mirror)];
                    }
                }
                // wxPrintf("Will loop jj from %i to %i\n", other_image->logical_y_dimension -
                // physical_index_of_first_negative_frequency_y, other_image->logical_y_dimension - 1);
                for (jj = other_image->logical_y_dimension - physical_index_of_first_negative_frequency_y;
                     jj <= other_image->logical_y_dimension - 1; jj++) {
                    // jj_mirror = other_image->logical_y_dimension - jj;
                    jj_mirror = jj;
                    for (ii = 0; ii <= physical_upper_bound_complex_x; ii++) {
                        // wxPrintf("(2) ii = %i; jj = %i; kk = %i; jj_mirror = %i; kk_mirror =
                        // %i\n",ii,jj,kk,jj_mirror,kk_mirror);
                        other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii, jj, kk)] =
                            other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(
                                ii, jj_mirror, kk_mirror)];
                    }
                }
                jj = 0;
                for (ii = 0; ii <= physical_upper_bound_complex_x; ii++) {
                    other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii, jj, kk)] =
                        other_image
                            ->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii, jj, kk_mirror)];
                }
            }
        }
    }
}

void Image::ChangePixelSize(Image* other_image, float wanted_factor, float wanted_tolerance, bool return_fft) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(other_image->is_in_memory, "Other image Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Image must be in real space");

    if (fabsf(wanted_factor - 1.0f) < wanted_tolerance && return_fft) {
        if (other_image == this)
            ForwardFFT();
        else {
            this->ClipInto(other_image);
            other_image->ForwardFFT();
        }
        return;
    }
    if (fabsf(wanted_factor - 1.0f) < wanted_tolerance) {
        if (other_image != this)
            this->ClipInto(other_image);
        return;
    }

    int   old_x_dimension, old_y_dimension, old_z_dimension = 1;
    int   new_x_dimension, new_y_dimension, new_z_dimension = 1;
    int   pad_x_dimension, pad_y_dimension, pad_z_dimension = 1;
    int   min_dimension;
    float intermediate_factor;
    float padding_value = 0.0f;
    Image temp_image;

    old_x_dimension = logical_x_dimension;
    old_y_dimension = logical_y_dimension;
    old_z_dimension = logical_z_dimension;

    min_dimension = std::min(logical_x_dimension, logical_y_dimension);
    if (logical_z_dimension > 1)
        min_dimension = std::min(min_dimension, logical_z_dimension);

    if (wanted_tolerance < 1.0f / (min_dimension * wanted_factor)) {
        intermediate_factor = 1.0f / (wanted_tolerance * min_dimension);
        pad_x_dimension     = old_x_dimension * intermediate_factor;
        if (IsEven(old_x_dimension) != IsEven(pad_x_dimension))
            pad_x_dimension++;
        new_x_dimension = int(pad_x_dimension / wanted_factor);
        if (IsEven(old_x_dimension) != IsEven(new_x_dimension))
            new_x_dimension++;
        pad_y_dimension = old_y_dimension * intermediate_factor;
        if (IsEven(old_y_dimension) != IsEven(pad_y_dimension))
            pad_y_dimension++;
        new_y_dimension = int(pad_y_dimension / wanted_factor);
        if (IsEven(old_y_dimension) != IsEven(new_y_dimension))
            new_y_dimension++;
        if (logical_z_dimension > 1) {
            pad_z_dimension = old_z_dimension * intermediate_factor;
            if (IsEven(old_z_dimension) != IsEven(pad_z_dimension))
                pad_z_dimension++;
            new_z_dimension = int(pad_z_dimension / wanted_factor);
            if (IsEven(old_z_dimension) != IsEven(new_z_dimension))
                new_z_dimension++;
        }
    }
    else {
        intermediate_factor = wanted_factor;
        pad_x_dimension     = old_x_dimension * intermediate_factor;
        if (IsEven(old_x_dimension) != IsEven(pad_x_dimension))
            pad_x_dimension++;
        new_x_dimension = old_x_dimension;
        pad_y_dimension = old_y_dimension * intermediate_factor;
        if (IsEven(old_y_dimension) != IsEven(pad_y_dimension))
            pad_y_dimension++;
        new_y_dimension = old_y_dimension;
        if (logical_z_dimension > 1) {
            pad_z_dimension = old_z_dimension * intermediate_factor;
            if (IsEven(old_z_dimension) != IsEven(pad_z_dimension))
                pad_z_dimension++;
            new_z_dimension = old_z_dimension;
        }
    }

    temp_image.CopyFrom(this);
    if (pad_x_dimension > old_x_dimension || pad_y_dimension > old_y_dimension || pad_z_dimension > old_z_dimension)
        padding_value = temp_image.ReturnAverageOfRealValues(0.45f * min_dimension);
    temp_image.Resize(pad_x_dimension, pad_y_dimension, pad_z_dimension, padding_value);
    temp_image.ForwardFFT();
    temp_image.Resize(new_x_dimension, new_y_dimension, new_z_dimension);

    if (new_x_dimension == other_image->logical_x_dimension && new_y_dimension == other_image->logical_y_dimension &&
        new_z_dimension == other_image->logical_z_dimension) {
        other_image->CopyFrom(&temp_image);
        if (!return_fft)
            other_image->BackwardFFT();
        return;
    }

    temp_image.BackwardFFT();
    temp_image.ClipInto(other_image, padding_value);

    if (return_fft)
        other_image->ForwardFFT();
}

// Bilinear interpolation in real space, at point (x,y) where x and y are physical coordinates (i.e. first pixel has x,y
// = 0,0)
void Image::GetRealValueByLinearInterpolationNoBoundsCheckImage(float& x, float& y, float& interpolated_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(logical_z_dimension == 1, "Not for volumes");
    MyDebugAssertTrue(is_in_real_space, "Need to be in real space");

    const int   i_start  = int(x);
    const int   j_start  = int(y);
    const float x_dist   = x - float(i_start);
    const float y_dist   = y - float(j_start);
    const float x_dist_m = 1.0 - x_dist;
    const float y_dist_m = 1.0 - y_dist;

    const int address_1 = j_start * (logical_x_dimension + padding_jump_value) + i_start;
    const int address_2 = address_1 + logical_x_dimension + padding_jump_value;

    MyDebugAssertTrue(address_1 + 1 <= real_memory_allocated && address_1 >= 0, "Out of bounds, address 1\n");
    MyDebugAssertTrue(address_2 + 1 <= real_memory_allocated && address_2 >= 0, "Out of bounds, address 2\n");

    interpolated_value = x_dist_m * y_dist_m * real_values[address_1] + x_dist * y_dist_m * real_values[address_1 + 1] +
                         x_dist_m * y_dist * real_values[address_2] + x_dist * y_dist * real_values[address_2 + 1];
}

void Image::Resize(int wanted_x_dimension, int wanted_y_dimension, int wanted_z_dimension, float wanted_padding_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(wanted_x_dimension != 0 && wanted_y_dimension != 0 && wanted_z_dimension != 0,
                      "Resize dimension is zero");

    if (logical_x_dimension == wanted_x_dimension && logical_y_dimension == wanted_y_dimension &&
        logical_z_dimension == wanted_z_dimension)
        return;

    Image temp_image;

    temp_image.Allocate(wanted_x_dimension, wanted_y_dimension, wanted_z_dimension, is_in_real_space);
    ClipInto(&temp_image, wanted_padding_value);

    // CopyFrom(&temp_image);
    Consume(&temp_image);
}

void Image::RealSpaceBinning(int bin_x, int bin_y, int bin_z, bool symmetrical, bool exclude_incomplete_bins) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(bin_x >= 1 && bin_y >= 1 && bin_z >= 1, "Invalid bin factor");
    MyDebugAssertTrue(!(symmetrical && exclude_incomplete_bins),
                      "symmetrical and exclude_incomplete_bins flags cannot be active at the same time");

    if (bin_x == 1 && bin_y == 1 && bin_z == 1)
        return;

    int    i, j, k;
    int    l, m, n;
    int    ix, iy, iz;
    int    new_x_dimension = ceilf(float(logical_x_dimension) / bin_x);
    int    new_y_dimension = ceilf(float(logical_y_dimension) / bin_y);
    int    new_z_dimension = ceilf(float(logical_z_dimension) / bin_z);
    int    bin_x2          = 0;
    int    bin_y2          = 0;
    int    bin_z2          = 0;
    long   pointer         = 0;
    long   counter;
    double average;

    if (exclude_incomplete_bins) {
        new_x_dimension = floorf(float(logical_x_dimension) / bin_x);
        new_y_dimension = floorf(float(logical_y_dimension) / bin_y);
        new_z_dimension = floorf(float(logical_z_dimension) / bin_z);
    }

    Image temp_image;
    temp_image.Allocate(new_x_dimension, new_y_dimension, new_z_dimension);

    if (symmetrical) {
        bin_x2 = bin_x / 2;
        bin_y2 = bin_y / 2;
        bin_z2 = bin_z / 2;
    }

    for (n = 0; n < new_z_dimension; n++) {
        for (m = 0; m < new_y_dimension; m++) {
            for (l = 0; l < new_x_dimension; l++) {
                average = 0.0f;
                counter = 0.0f;
                for (k = -bin_z2; k < bin_z - bin_z2; k++) {
                    iz = bin_z * n + k;
                    for (j = -bin_y2; j < bin_y - bin_y2; j++) {
                        iy = bin_y * m + j;
                        for (i = -bin_x2; i < bin_x - bin_x2; i++) {
                            ix = bin_x * l + i;

                            if (ix >= 0 && ix < logical_x_dimension && iy >= 0 && iy < logical_y_dimension && iz >= 0 &&
                                iz < logical_z_dimension) {
                                counter++;
                                average += ReturnRealPixelFromPhysicalCoord(ix, iy, iz);
                            }
                        }
                    }
                }
                average /= counter;
                temp_image.real_values[pointer] = average;
                pointer++;
            }
            pointer += temp_image.padding_jump_value;
        }
    }
    Consume(&temp_image);
}

float Image::ReturnVarianceOfRealValuesTiled(int bin_x, int bin_y, int bin_z, bool exclude_incomplete_bins) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(bin_x >= 1 && bin_y >= 1 && bin_z >= 1, "Invalid bin factor");

    if (bin_x == 1 && bin_y == 1 && bin_z == 1)
        return 0.0f;

    int    i, j, k;
    int    l, m, n;
    int    ix, iy, iz;
    int    new_x_dimension = ceilf(float(logical_x_dimension) / bin_x);
    int    new_y_dimension = ceilf(float(logical_y_dimension) / bin_y);
    int    new_z_dimension = ceilf(float(logical_z_dimension) / bin_z);
    long   counter1;
    long   counter2;
    double average;
    double variance;
    double temp_float;
    double average_variance;

    if (exclude_incomplete_bins) {
        new_x_dimension = floorf(float(logical_x_dimension) / bin_x);
        new_y_dimension = floorf(float(logical_y_dimension) / bin_y);
        new_z_dimension = floorf(float(logical_z_dimension) / bin_z);
    }

    average_variance = 0.0f;
    counter2         = 0;
    for (n = 0; n < new_z_dimension; n++) {
        for (m = 0; m < new_y_dimension; m++) {
            for (l = 0; l < new_x_dimension; l++) {
                average  = 0.0f;
                variance = 0.0f;
                counter1 = 0.0f;
                for (k = 0; k < bin_z; k++) {
                    iz = bin_z * n + k;
                    for (j = 0; j < bin_y; j++) {
                        iy = bin_y * m + j;
                        for (i = 0; i < bin_x; i++) {
                            ix = bin_x * l + i;

                            if (ix >= 0 && ix < logical_x_dimension && iy >= 0 && iy < logical_y_dimension && iz >= 0 &&
                                iz < logical_z_dimension) {
                                counter1++;
                                temp_float = ReturnRealPixelFromPhysicalCoord(ix, iy, iz);
                                average += temp_float;
                                variance += pow(temp_float, 2);
                            }
                        }
                    }
                }
                counter2++;
                average /= counter1;
                average_variance += variance / counter1 - pow(average, 2);
            }
        }
    }

    return float(average_variance / counter2);
}

void Image::CopyFrom(Image* other_image) { *this = other_image; }

void Image::CopyLoopingAndAddressingFrom(Image* other_image) {
    object_is_centred_in_box = other_image->object_is_centred_in_box;
    logical_x_dimension      = other_image->logical_x_dimension;
    logical_y_dimension      = other_image->logical_y_dimension;
    logical_z_dimension      = other_image->logical_z_dimension;

    physical_upper_bound_complex_x = other_image->physical_upper_bound_complex_x;
    physical_upper_bound_complex_y = other_image->physical_upper_bound_complex_y;
    physical_upper_bound_complex_z = other_image->physical_upper_bound_complex_z;

    physical_address_of_box_center_x = other_image->physical_address_of_box_center_x;
    physical_address_of_box_center_y = other_image->physical_address_of_box_center_y;
    physical_address_of_box_center_z = other_image->physical_address_of_box_center_z;

    // physical_index_of_first_negative_frequency_x = other_image->physical_index_of_first_negative_frequency_x;
    physical_index_of_first_negative_frequency_y = other_image->physical_index_of_first_negative_frequency_y;
    physical_index_of_first_negative_frequency_z = other_image->physical_index_of_first_negative_frequency_z;

    fourier_voxel_size_x = other_image->fourier_voxel_size_x;
    fourier_voxel_size_y = other_image->fourier_voxel_size_y;
    fourier_voxel_size_z = other_image->fourier_voxel_size_z;

    logical_upper_bound_complex_x = other_image->logical_upper_bound_complex_x;
    logical_upper_bound_complex_y = other_image->logical_upper_bound_complex_y;
    logical_upper_bound_complex_z = other_image->logical_upper_bound_complex_z;

    logical_lower_bound_complex_x = other_image->logical_lower_bound_complex_x;
    logical_lower_bound_complex_y = other_image->logical_lower_bound_complex_y;
    logical_lower_bound_complex_z = other_image->logical_lower_bound_complex_z;

    logical_upper_bound_real_x = other_image->logical_upper_bound_complex_x;
    logical_upper_bound_real_y = other_image->logical_upper_bound_complex_y;
    logical_upper_bound_real_z = other_image->logical_upper_bound_complex_z;

    logical_lower_bound_real_x = other_image->logical_lower_bound_complex_x;
    logical_lower_bound_real_y = other_image->logical_lower_bound_complex_y;
    logical_lower_bound_real_z = other_image->logical_lower_bound_complex_z;

    padding_jump_value = other_image->padding_jump_value;
}

void Image::Consume(Image* other_image) // copy the parameters then directly steal the memory of another image, leaving
                                        // it an empty shell
{
    MyDebugAssertTrue(other_image->is_in_memory, "Other image Memory not allocated");

    if (is_in_memory == true) {
        Deallocate();
    }

    is_in_real_space      = other_image->is_in_real_space;
    real_memory_allocated = other_image->real_memory_allocated;
    CopyLoopingAndAddressingFrom(other_image);

    real_values    = other_image->real_values;
    complex_values = other_image->complex_values;
    is_in_memory   = other_image->is_in_memory;

    plan_fwd = other_image->plan_fwd;
    plan_bwd = other_image->plan_bwd;
    planned  = other_image->planned;

    other_image->real_values    = NULL;
    other_image->complex_values = NULL;
    other_image->is_in_memory   = false;

    other_image->plan_fwd = NULL;
    other_image->plan_bwd = NULL;
    other_image->planned  = false;

    number_of_real_space_pixels = other_image->number_of_real_space_pixels;
    ft_normalization_factor     = other_image->ft_normalization_factor;
}

void Image::RealSpaceIntegerShift(int wanted_x_shift, int wanted_y_shift, int wanted_z_shift) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    int    i, j, k;
    long   pixel_counter = 0;
    long   shifted_counter;
    float* buffer = new float[number_of_real_space_pixels];

    shifted_counter = -wanted_x_shift - logical_x_dimension * (wanted_y_shift + logical_y_dimension * wanted_z_shift);
    shifted_counter = remainderf(float(shifted_counter), float(number_of_real_space_pixels));
    if (shifted_counter < 0)
        shifted_counter += number_of_real_space_pixels;

    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                buffer[shifted_counter] = real_values[pixel_counter];
                pixel_counter++;
                shifted_counter++;
                if (shifted_counter >= number_of_real_space_pixels)
                    shifted_counter -= number_of_real_space_pixels;
            }
            pixel_counter += padding_jump_value;
        }
    }
    shifted_counter = 0;
    pixel_counter   = 0;
    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                real_values[pixel_counter] = buffer[shifted_counter];
                pixel_counter++;
                shifted_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }
    delete[] buffer;
}

void Image::DilateBinarizedMask(float dilation_radius) {
    // To get a ring of new pixels 1 pixel wide, a value of 2.0 must be used for dilation_radius. I do not know why.
    // FIXME

    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    int    i, j, k;
    int    l, m, n, m2, n2;
    int    lim, nlim;
    long   pixel_counter           = 0;
    long   shifted_counter         = 0;
    float  dilation_radius_squared = powf(dilation_radius, 2);
    float* buffer                  = new float[number_of_real_space_pixels];

    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                buffer[shifted_counter] = real_values[pixel_counter];
                pixel_counter++;
                shifted_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }

    lim = myroundint(dilation_radius);
    if (IsEven(lim))
        lim++;
    nlim = lim;
    if (logical_z_dimension == 1)
        nlim = 0;
    for (n = -nlim; n <= nlim; n += 2) {
        n2 = n * n;
        for (m = -lim; m <= lim; m += 2) {
            m2 = m * m;
            for (l = -lim; l <= lim; l += 2) {
                if (l * l + m2 + n2 <= dilation_radius_squared) {
                    shifted_counter = -l - logical_x_dimension * (m + logical_y_dimension * n);
                    shifted_counter = remainderf(float(shifted_counter), float(number_of_real_space_pixels));
                    if (shifted_counter < 0)
                        shifted_counter += number_of_real_space_pixels;

                    pixel_counter = 0;
                    for (k = 0; k < logical_z_dimension; k++) {
                        for (j = 0; j < logical_y_dimension; j++) {
                            for (i = 0; i < logical_x_dimension; i++) {
                                real_values[pixel_counter] += buffer[shifted_counter];
                                pixel_counter++;
                                shifted_counter++;
                                if (shifted_counter >= number_of_real_space_pixels)
                                    shifted_counter -= number_of_real_space_pixels;
                            }
                            pixel_counter += padding_jump_value;
                        }
                    }
                }
            }
        }
    }

    pixel_counter = 0;
    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                if (real_values[pixel_counter] != 0.0)
                    real_values[pixel_counter] = 1.0;
                pixel_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }
    delete[] buffer;
}

void Image::ErodeBinarizedMask(float erosion_radius) {
    // Not sure this is the best way, but it is straightforward.
    Image buffer_1;
    buffer_1.CopyFrom(this);

    // To get a ring of new pixels 1 pixel wide, a value of 2.0 must be used for dilation_radius. I do not know why.
    // FIXME
    buffer_1.DilateBinarizedMask(2.0f);

    buffer_1.SubtractImage(this);
    for (int iDilation = 0; iDilation < myroundint(erosion_radius); iDilation++) {
        buffer_1.DilateBinarizedMask(2.0f);
    }
    buffer_1.BinariseInverse(0.5f);

    MultiplyPixelWise(buffer_1);
}

void Image::MakeAbsolute() {
    for (long counter = 0; counter < real_memory_allocated; counter++) {
        real_values[counter] = fabsf(real_values[counter]);
    }
}

void Image::PhaseShift(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    bool need_to_fft = false;

    long pixel_counter = 0;

    int k;
    int k_logical;

    int j;
    int j_logical;

    int i;

    float phase_z;
    float phase_y;
    float phase_x;

    std::complex<float> total_phase_shift;

    if (is_in_real_space == true) {
        ForwardFFT();
        need_to_fft = true;
    }

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        k_logical = ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k);
        phase_z   = ReturnPhaseFromShift(wanted_z_shift, k_logical, logical_z_dimension);

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            j_logical = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j);
            phase_y   = ReturnPhaseFromShift(wanted_y_shift, j_logical, logical_y_dimension);

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {

                phase_x = ReturnPhaseFromShift(wanted_x_shift, i, logical_x_dimension);

                total_phase_shift = Return3DPhaseFromIndividualDimensions(phase_x, phase_y, phase_z);
                complex_values[pixel_counter] *= total_phase_shift;

                pixel_counter++;
            }
        }
    }

    if (need_to_fft == true)
        BackwardFFT();
}

// END_FOR_STAND_ALONE_CTFFIND

void Image::ApplyCTFPhaseFlip(CTF ctf_to_apply) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == false, "image not in Fourier space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Volumes not supported");

    int j;
    int i;

    long pixel_counter = 0;

    float y_coord_sq;
    float x_coord_sq;

    float y_coord;
    float x_coord;

    float frequency_squared;
    float azimuth;
    float ctf_value;

    for (j = 0; j <= physical_upper_bound_complex_y; j++) {
        y_coord    = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y;
        y_coord_sq = powf(y_coord, 2.0);

        for (i = 0; i <= physical_upper_bound_complex_x; i++) {
            x_coord    = i * fourier_voxel_size_x;
            x_coord_sq = powf(x_coord, 2);

            // Compute the azimuth
            if (i == 0 && j == 0) {
                azimuth = 0.0;
            }
            else {
                azimuth = atan2f(y_coord, x_coord);
            }

            // Compute the square of the frequency
            frequency_squared = x_coord_sq + y_coord_sq;

            ctf_value = ctf_to_apply.Evaluate(frequency_squared, azimuth);

            if (ctf_value < 0.0)
                complex_values[pixel_counter] = -1.0f * complex_values[pixel_counter];
            pixel_counter++;
        }
    }
}

void Image::ApplyCTF(CTF ctf_to_apply, bool absolute, bool apply_beam_tilt, bool apply_envelope) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == false, "image not in Fourier space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Volumes not supported");

    int j;
    int i;

    long pixel_counter = 0;

    float y_coord_sq;
    float x_coord_sq;

    float y_coord;
    float x_coord;

    float frequency_squared;
    float azimuth;
    float ctf_value;

    for (j = 0; j <= physical_upper_bound_complex_y; j++) {
        y_coord    = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y;
        y_coord_sq = powf(y_coord, 2.0);

        for (i = 0; i <= physical_upper_bound_complex_x; i++) {
            x_coord    = i * fourier_voxel_size_x;
            x_coord_sq = powf(x_coord, 2);

            // Compute the azimuth
            if (i == 0 && j == 0) {
                azimuth = 0.0;
            }
            else {
                azimuth = atan2f(y_coord, x_coord);
            }

            // Compute the square of the frequency
            frequency_squared = x_coord_sq + y_coord_sq;

            if (apply_envelope) {
                ctf_value = ctf_to_apply.EvaluateWithEnvelope(frequency_squared, azimuth);
            }
            else {
                ctf_value = ctf_to_apply.Evaluate(frequency_squared, azimuth);
            }

            if (absolute)
                ctf_value = fabsf(ctf_value);

            complex_values[pixel_counter] *= ctf_value;

            if (apply_beam_tilt && (ctf_to_apply.GetBeamTiltX() != 0.0f || ctf_to_apply.GetBeamTiltY() != 0.0f))
                complex_values[pixel_counter] *= ctf_to_apply.EvaluateBeamTiltPhaseShift(frequency_squared, azimuth);

            pixel_counter++;
        }
    }
    //	Image temp_image;
    //	temp_image.Allocate(logical_x_dimension, logical_y_dimension, false);
    //	ComputeAmplitudeSpectrumFull2D(&temp_image, true);
    //	temp_image.QuickAndDirtyWriteSlice("junk.mrc", 1);
    //	exit(0);
}

void Image::SharpenMap(float pixel_size, float resolution_limit, bool invert_hand, float inner_mask_radius,
                       float outer_mask_radius, float start_res_for_whitening, float additional_bfactor_low,
                       float additional_bfactor_high, float filter_edge, bool should_auto_mask, Image* input_mask,
                       ResolutionStatistics* input_resolution_statistics, float statistics_scale_factor,
                       Curve* original_log_plot, Curve* sharpened_log_plot) {

    float cosine_edge = 10.0;

    Curve power_spectrum;
    Curve number_of_terms;
    Curve copy_of_rec_SSNR;

    if (input_resolution_statistics != NULL)
        copy_of_rec_SSNR = input_resolution_statistics->rec_SSNR;

    power_spectrum.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
    number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));

    if (original_log_plot != NULL)
        original_log_plot->SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
    if (sharpened_log_plot != NULL)
        sharpened_log_plot->SetupXAxis(0.0, 0.5 * sqrtf(3.0),
                                       int((logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));

    Image buffer_image;
    buffer_image.Allocate(logical_x_dimension, logical_y_dimension, logical_z_dimension, true);
    buffer_image.CopyFrom(this);

    if (outer_mask_radius == 0.0)
        outer_mask_radius = logical_x_dimension / 2.0;

    else if (should_auto_mask == true) {
        MyDebugAssertTrue(input_mask == NULL, "Error: Automask requested, and mask supplied...") input_mask = new Image;
        input_mask->CopyFrom(this);
        input_mask->ConvertToAutoMask(pixel_size, outer_mask_radius, 10.0f, 0.1);
    }

    if (input_mask != NULL) {
        buffer_image.ApplyMask(*input_mask, cosine_edge / pixel_size, 0.0, 0.0, 0.0);
    }
    else {
        buffer_image.CosineRingMask(inner_mask_radius / pixel_size, outer_mask_radius / pixel_size,
                                    cosine_edge / pixel_size);
    }

    ForwardFFT();
    buffer_image.ForwardFFT();

    buffer_image.Compute1DPowerSpectrumCurve(&power_spectrum, &number_of_terms);
    power_spectrum.SquareRoot();

    if (original_log_plot != NULL) {
        for (int counter = 0; counter < original_log_plot->number_of_points; counter++) {
            if (original_log_plot->data_x[counter] == 0.0) {
                original_log_plot->data_x[counter] = (pixel_size * original_log_plot->data_x[counter + 1]);
                original_log_plot->data_y[counter] = logf(power_spectrum.data_y[counter + 1]);
            }
            else {
                original_log_plot->data_x[counter] = (pixel_size * original_log_plot->data_x[counter]);
                if (power_spectrum.data_y[counter] <= 0.0f)
                    original_log_plot->data_y[counter] = 0.0;
                original_log_plot->data_y[counter] = logf(power_spectrum.data_y[counter]);
            }
        }
    }

    ApplyBFactorAndWhiten(power_spectrum, additional_bfactor_low / pixel_size / pixel_size,
                          additional_bfactor_high / pixel_size / pixel_size, pixel_size / start_res_for_whitening);

    if (input_resolution_statistics != NULL) {
        if (statistics_scale_factor != 1.0)
            copy_of_rec_SSNR.MultiplyByConstant(statistics_scale_factor);
        OptimalFilterSSNR(copy_of_rec_SSNR);
    }

    CosineMask(pixel_size / resolution_limit - pixel_size / 2.0 / filter_edge, pixel_size / filter_edge);

    BackwardFFT();

    if (sharpened_log_plot != NULL) {
        buffer_image.CopyFrom(this);
        if (input_mask == NULL)
            buffer_image.CosineRingMask(inner_mask_radius / pixel_size, outer_mask_radius / pixel_size,
                                        cosine_edge / pixel_size);
        else {
            buffer_image.ApplyMask(*input_mask, cosine_edge / pixel_size, 0.0, 0.0, 0.0);
        }

        buffer_image.ForwardFFT();

        buffer_image.Compute1DPowerSpectrumCurve(&power_spectrum, &number_of_terms);
        power_spectrum.SquareRoot();

        for (int counter = 0; counter < sharpened_log_plot->number_of_points; counter++) {
            if (sharpened_log_plot->data_x[counter] == 0.0) {
                sharpened_log_plot->data_x[counter] = (pixel_size * sharpened_log_plot->data_x[counter + 1]);
                sharpened_log_plot->data_y[counter] = logf(power_spectrum.data_y[counter + 1]);
            }
            else {
                sharpened_log_plot->data_x[counter] = (pixel_size * sharpened_log_plot->data_x[counter]);
                if (power_spectrum.data_y[counter] <= 0.0f)
                    sharpened_log_plot->data_y[counter] = 0.0;
                sharpened_log_plot->data_y[counter] = logf(power_spectrum.data_y[counter]);
            }
        }
    }

    if (invert_hand == true)
        InvertHandedness();

    // normalise to 0.03 (this is abritrary), but seems to work better for model building.

    Normalize(0.03);
}

void Image::InvertHandedness() {
    int  i, j;
    long offset, pixel_counter;
    pixel_counter = 0;

    Image buffer_image;
    buffer_image.CopyFrom(this);

    for (j = logical_z_dimension - 1; j >= 0; j--) {
        offset = j * (logical_x_dimension + padding_jump_value) * logical_y_dimension;

        for (i = 0; i < (logical_x_dimension + padding_jump_value) * logical_y_dimension; i++) {
            real_values[pixel_counter] = buffer_image.real_values[i + offset];
            pixel_counter++;
        }
    }
}

void Image::ApplyBFactor(float bfactor) // add real space and windows later, probably to an overloaded function
{
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == false, "image not in Fourier space");

    int k;
    int j;
    int i;

    long pixel_counter = 0;

    float z_coord;
    float y_coord;
    float x_coord;

    float frequency_squared;
    float filter_value;
    // float frequency;

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        z_coord = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            y_coord = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                x_coord = powf(i * fourier_voxel_size_x, 2);

                // compute squared radius, in units of reciprocal pixels

                frequency_squared = x_coord + y_coord + z_coord;
                // frequency = sqrt(frequency_squared);
                filter_value = exp(-bfactor * frequency_squared * 0.25);
                complex_values[pixel_counter] *= filter_value;
                pixel_counter++;
            }
        }
    }
}

void Image::ApplyBFactorAndWhiten(Curve& power_spectrum, float bfactor_low, float bfactor_high,
                                  float bfactor_res_limit) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == false, "image not in Fourier space");

    int i, j, k;
    int bin;
    int number_of_bins2 = ReturnLargestLogicalDimension();

    long pixel_counter = 0;

    float z_coord;
    float y_coord;
    float x_coord;

    float frequency_squared;
    float filter_value;
    float bfactor_res_limit2 = powf(bfactor_res_limit, 2);
    float filter_value_blimit;

    bin                 = int(sqrtf(bfactor_res_limit2) * number_of_bins2);
    filter_value_blimit = power_spectrum.data_y[bin];
    //	filter_value_blimit = exp(-bfactor * bfactor_res_limit2 * 0.25) * power_spectrum.data_y[bin];

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        z_coord = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            y_coord = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                x_coord = powf(i * fourier_voxel_size_x, 2);

                // compute squared radius, in units of reciprocal pixels

                frequency_squared = x_coord + y_coord + z_coord;
                // frequency = sqrt(frequency_squared);

                // compute radius, in units of physical Fourier pixels
                bin = int(sqrtf(frequency_squared) * number_of_bins2);

                if (frequency_squared <= bfactor_res_limit2)
                    filter_value = exp(-bfactor_low * frequency_squared * 0.25);
                else if ((frequency_squared > bfactor_res_limit2) && (frequency_squared <= 0.25) &&
                         (bin < power_spectrum.number_of_points))
                    filter_value = filter_value_blimit *
                                   exp(-(bfactor_low * bfactor_res_limit2 +
                                         bfactor_high * (frequency_squared - bfactor_res_limit2)) *
                                       0.25) /
                                   power_spectrum.data_y[bin];
                //				else if ((frequency_squared > bfactor_res_limit2) && (frequency_squared <= 0.25) && (bin <
                //power_spectrum.number_of_points)) filter_value = filter_value_blimit / power_spectrum.data_y[bin];
                else
                    filter_value = 0.0;

                complex_values[pixel_counter] *= filter_value;
                pixel_counter++;
            }
        }
    }
}

void Image::CalculateDerivative(float direction_in_x, float direction_in_y, float direction_in_z) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    int k;
    int j;
    int i;

    long pixel_counter = 0;

    float x_coord;
    float y_coord;
    float z_coord;

    float length = sqrtf(powf(direction_in_x, 2) + powf(direction_in_y, 2) + powf(direction_in_z, 2));
    float unit_x = direction_in_x / length;
    float unit_y = direction_in_y / length;
    float unit_z = direction_in_z / length;

    bool apply_fft         = is_in_real_space;
    bool radial_derivative = (direction_in_x == 0.0f && direction_in_y == 0.0f && direction_in_z == 0.0f);

    if (apply_fft)
        ForwardFFT();

    for (k = 0; k <= physical_upper_bound_complex_z; k++) {
        z_coord = ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z;

        for (j = 0; j <= physical_upper_bound_complex_y; j++) {
            y_coord = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y;

            for (i = 0; i <= physical_upper_bound_complex_x; i++) {
                x_coord = i * fourier_voxel_size_x;

                if (radial_derivative)
                    complex_values[pixel_counter] *=
                        2.0f * PI * (x_coord * x_coord + y_coord * y_coord + z_coord * z_coord);
                else
                    complex_values[pixel_counter] *=
                        2.0f * (float)PI * I * (x_coord * unit_x + y_coord * unit_y + z_coord * unit_z);
                pixel_counter++;
            }
        }
    }

    if (apply_fft)
        BackwardFFT();
}

// If you set half_width to 1, only the central row or column of pixels will be masked. Half_width of 2 means 3 pixels
// will be masked.
void Image::MaskCentralCross(int vertical_half_width, int horizontal_half_width) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(logical_z_dimension == 1, "Only 2D currently supported");
    MyDebugAssertTrue(vertical_half_width > 0 && horizontal_half_width > 0, "Half width must be greater than 0");

    int pixel_counter;
    int width_counter;

    if (!is_in_real_space) {
        for (pixel_counter = logical_lower_bound_complex_y; pixel_counter <= logical_upper_bound_complex_y;
             pixel_counter++) {
            for (width_counter = -(horizontal_half_width - 1); width_counter <= (horizontal_half_width - 1);
                 width_counter++) {
                complex_values[ReturnFourier1DAddressFromLogicalCoord(width_counter, pixel_counter, 0)] =
                    0.0f + 0.0f * I;
            }
        }

        for (pixel_counter = 0; pixel_counter <= logical_upper_bound_complex_x; pixel_counter++) {
            for (width_counter = -(vertical_half_width - 1); width_counter <= (vertical_half_width - 1);
                 width_counter++) {
                complex_values[ReturnFourier1DAddressFromLogicalCoord(pixel_counter, width_counter, 0)] =
                    0.0f + 0.0f * I;
            }
        }
    }
    else {
        long address_x_start  = physical_address_of_box_center_x - (vertical_half_width - 1);
        long address_x_finish = physical_address_of_box_center_x + (vertical_half_width - 1);

        long address_y_start  = physical_address_of_box_center_y - (horizontal_half_width - 1);
        long address_y_finish = physical_address_of_box_center_y + (horizontal_half_width - 1);

        int y_counter;
        int x_counter;

        // Loop over lines
        for (y_counter = 0; y_counter < logical_y_dimension; y_counter++) {
            for (x_counter = address_x_start; x_counter <= address_x_finish; x_counter++) {
                real_values[x_counter] = 0.0;
            }
            address_x_start += logical_x_dimension + padding_jump_value;
            address_x_finish += logical_x_dimension + padding_jump_value;
        }

        // Loop over columns
        address_y_start *= (logical_x_dimension + padding_jump_value);
        address_y_finish *= (logical_y_dimension + padding_jump_value);
        for (x_counter = 0; x_counter < logical_x_dimension; x_counter++) {
            for (y_counter = address_y_start; y_counter <= address_y_finish;
                 y_counter += logical_x_dimension + padding_jump_value) {
                real_values[y_counter] = 0.0;
            }
            address_y_start++;
            address_y_finish++;
        }
    }
}

// BEGIN_FOR_STAND_ALONE_CTFFIND

bool Image::HasSameDimensionsAs(Image* other_image) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(other_image->is_in_memory, "Other image Memory not allocated");

    if (logical_x_dimension == other_image->logical_x_dimension &&
        logical_y_dimension == other_image->logical_y_dimension &&
        logical_z_dimension == other_image->logical_z_dimension)
        return true;
    else
        return false;
}

// END_FOR_STAND_ALONE_CTFFIND

void Image::SwapRealSpaceQuadrants() {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");

    bool must_fft = false;

    float x_shift_to_apply;
    float y_shift_to_apply;
    float z_shift_to_apply;

    if (is_in_real_space == true) {
        must_fft = true;
        ForwardFFT();
    }

    if (object_is_centred_in_box == true) {
        x_shift_to_apply = float(physical_address_of_box_center_x);
        y_shift_to_apply = float(physical_address_of_box_center_y);
        z_shift_to_apply = float(physical_address_of_box_center_z);
    }
    else {
        if (IsEven(logical_x_dimension) == true) {
            x_shift_to_apply = float(physical_address_of_box_center_x);
        }
        else {
            x_shift_to_apply = float(physical_address_of_box_center_x) - 1.0;
        }

        if (IsEven(logical_y_dimension) == true) {
            y_shift_to_apply = float(physical_address_of_box_center_y);
        }
        else {
            y_shift_to_apply = float(physical_address_of_box_center_y) - 1.0;
        }

        if (IsEven(logical_z_dimension) == true) {
            z_shift_to_apply = float(physical_address_of_box_center_z);
        }
        else {
            z_shift_to_apply = float(physical_address_of_box_center_z) - 1.0;
        }
    }

    if (logical_z_dimension == 1) {
        z_shift_to_apply = 0.0;
    }

    PhaseShift(x_shift_to_apply, y_shift_to_apply, z_shift_to_apply);

    if (must_fft == true)
        BackwardFFT();

    // keep track of center;
    if (object_is_centred_in_box == true)
        object_is_centred_in_box = false;
    else
        object_is_centred_in_box = true;
}

void Image::CalculateCrossCorrelationImageWith(Image* other_image) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == other_image->is_in_real_space, "Images are in different spaces");
    MyDebugAssertTrue(HasSameDimensionsAs(other_image) == true, "Images are different sizes");

    long pixel_counter;
    bool must_fft = false;

    // do we have to fft..

    if (is_in_real_space == true) {
        must_fft = true;
        ForwardFFT();
        other_image->ForwardFFT();
    }

    // multiply by the complex conjugate

#ifdef MKL
    // Use the MKL
    vmcMulByConj(real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(complex_values),
                 reinterpret_cast<MKL_Complex8*>(other_image->complex_values),
                 reinterpret_cast<MKL_Complex8*>(complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    for (pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++) {
        complex_values[pixel_counter] *= conj(other_image->complex_values[pixel_counter]);
    }
#endif

    if (object_is_centred_in_box == true) {
        object_is_centred_in_box = false;
        SwapRealSpaceQuadrants();
    }

    BackwardFFT();

    if (must_fft == true)
        other_image->BackwardFFT();
}

Peak Image::FindPeakAtOriginFast2D(int wanted_max_pix_x, int wanted_max_pix_y) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");
    MyDebugAssertTrue(!object_is_centred_in_box, "Peak centered in image");

    int j;
    int i;
    int jj;
    int pixel_counter;
    int y_dim     = logical_y_dimension + padding_jump_value;
    int max_pix_x = wanted_max_pix_x;
    int max_pix_y = wanted_max_pix_y;

    if (max_pix_x > physical_address_of_box_center_x)
        max_pix_x = physical_address_of_box_center_x;
    if (max_pix_y > physical_address_of_box_center_y)
        max_pix_y = physical_address_of_box_center_y;

    Peak found_peak;
    found_peak.value = -FLT_MAX;
    found_peak.x     = 0.0;
    found_peak.y     = 0.0;
    found_peak.z     = 0.0;

    for (j = 0; j <= max_pix_y; j++) {
        jj = j * y_dim;
        for (i = 0; i <= max_pix_x; i++) {
            pixel_counter = jj + i;
            if (real_values[pixel_counter] > found_peak.value) {
                found_peak.value                         = real_values[pixel_counter];
                found_peak.x                             = i;
                found_peak.y                             = j;
                found_peak.physical_address_within_image = pixel_counter;
            }
        }
    }

    for (j = logical_y_dimension - max_pix_y - 1; j <= logical_y_dimension - 1; j++) {
        jj = j * y_dim;
        for (i = 0; i <= max_pix_x; i++) {
            pixel_counter = jj + i;
            if (real_values[pixel_counter] > found_peak.value) {
                found_peak.value                         = real_values[pixel_counter];
                found_peak.x                             = i;
                found_peak.y                             = j;
                found_peak.physical_address_within_image = pixel_counter;
            }
        }
    }

    for (j = 0; j <= max_pix_y; j++) {
        jj = j * y_dim;
        for (i = logical_x_dimension - max_pix_x - 1; i <= logical_x_dimension - 1; i++) {
            pixel_counter = jj + i;
            if (real_values[pixel_counter] > found_peak.value) {
                found_peak.value                         = real_values[pixel_counter];
                found_peak.x                             = i;
                found_peak.y                             = j;
                found_peak.physical_address_within_image = pixel_counter;
            }
        }
    }

    for (j = logical_y_dimension - max_pix_y - 1; j <= logical_y_dimension - 1; j++) {
        jj = j * y_dim;
        for (i = logical_x_dimension - max_pix_x - 1; i <= logical_x_dimension - 1; i++) {
            pixel_counter = jj + i;
            if (real_values[pixel_counter] > found_peak.value) {
                found_peak.value                         = real_values[pixel_counter];
                found_peak.x                             = i;
                found_peak.y                             = j;
                found_peak.physical_address_within_image = pixel_counter;
            }
        }
    }

    if (found_peak.x > physical_address_of_box_center_x)
        found_peak.x -= logical_x_dimension;
    if (found_peak.y > physical_address_of_box_center_y)
        found_peak.y -= logical_y_dimension;
    return found_peak;
}

Peak Image::FindPeakWithIntegerCoordinates(float wanted_min_radius, float wanted_max_radius,
                                           int wanted_min_distance_from_edges) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");
    MyDebugAssertFalse((!object_is_centred_in_box) && wanted_min_distance_from_edges > 0,
                       "Minimum distance from edges only implemented when object is centered in the box. Sorry");

    int k;
    int j;
    int i;
    int kk;
    int jj;
    int ii;

    float z;
    float y;
    float x;

    long pixel_counter = 0;

    float inv_max_radius_sq_x;
    float inv_max_radius_sq_y;
    float inv_max_radius_sq_z;

    float distance_from_origin;

    bool radii_are_fractional;

    Peak found_peak;
    found_peak.value = -FLT_MAX;
    found_peak.x     = 0.0;
    found_peak.y     = 0.0;
    found_peak.z     = 0.0;

    // if they are > 0.0 and < 1.0 then the radius will be interpreted as fraction, otherwise
    // it is interpreted as absolute.

    if (wanted_min_radius >= 0.0 && wanted_min_radius < 1.0 && wanted_max_radius >= 0.0 && wanted_max_radius < 1.0)
        radii_are_fractional = true;
    else
        radii_are_fractional = false;

    wanted_min_radius = powf(wanted_min_radius, 2);
    wanted_max_radius = powf(wanted_max_radius, 2);

    //
    int k_min, k_max;
    if (logical_z_dimension > 1) {
        k_min = wanted_min_distance_from_edges;
        k_max = logical_z_dimension - wanted_min_distance_from_edges;
    }
    else {
        k_min = 0;
        k_max = logical_z_dimension;
    }
    const int j_min = wanted_min_distance_from_edges;
    const int j_max = logical_y_dimension - wanted_min_distance_from_edges;
    const int i_min = wanted_min_distance_from_edges;
    const int i_max = logical_x_dimension - wanted_min_distance_from_edges;

    //

    if (object_is_centred_in_box) {
        if (radii_are_fractional == true) {
            inv_max_radius_sq_x = 1.0 / powf(physical_address_of_box_center_x, 2);
            inv_max_radius_sq_y = 1.0 / powf(physical_address_of_box_center_y, 2);

            if (logical_z_dimension == 1)
                inv_max_radius_sq_z = 0;
            else
                inv_max_radius_sq_z = 1.0 / powf(physical_address_of_box_center_z, 2);

            for (k = 0; k < logical_z_dimension; k++) {
                z = powf(k - physical_address_of_box_center_z, 2) * inv_max_radius_sq_z;

                for (j = 0; j < logical_y_dimension; j++) {
                    y = powf(j - physical_address_of_box_center_y, 2) * inv_max_radius_sq_y;

                    for (i = 0; i < logical_x_dimension; i++) {
                        x = powf(i - physical_address_of_box_center_x, 2) * inv_max_radius_sq_x;

                        distance_from_origin = x + y + z;

                        if (distance_from_origin >= wanted_min_radius && distance_from_origin <= wanted_max_radius &&
                            i >= i_min && i <= i_max && j >= j_min && j <= j_max && k >= k_min && k <= k_max) {
                            if (real_values[pixel_counter] > found_peak.value) {
                                found_peak.value                         = real_values[pixel_counter];
                                found_peak.x                             = i - physical_address_of_box_center_x;
                                found_peak.y                             = j - physical_address_of_box_center_y;
                                found_peak.z                             = k - physical_address_of_box_center_z;
                                found_peak.physical_address_within_image = pixel_counter;
                            }
                        }

                        // wxPrintf("new peak %f, %f, %f (%f)\n", found_peak.x, found_peak.y, found_peak.z,
                        // found_peak.value); wxPrintf("value %f, %f, %f (%f)\n", x, y, z, real_values[pixel_counter]);

                        pixel_counter++;
                    }

                    pixel_counter += padding_jump_value;
                }
            }
        }
        else {
            for (k = 0; k < logical_z_dimension; k++) {
                z = powf(k - physical_address_of_box_center_z, 2);

                for (j = 0; j < logical_y_dimension; j++) {
                    y = powf(j - physical_address_of_box_center_y, 2);

                    for (i = 0; i < logical_x_dimension; i++) {
                        x = powf(i - physical_address_of_box_center_x, 2);

                        distance_from_origin = x + y + z;

                        if (distance_from_origin >= wanted_min_radius && distance_from_origin <= wanted_max_radius &&
                            i >= i_min && i <= i_max && j >= j_min && j <= j_max && k >= k_min && k <= k_max) {
                            if (real_values[pixel_counter] > found_peak.value) {
                                found_peak.value                         = real_values[pixel_counter];
                                found_peak.x                             = i - physical_address_of_box_center_x;
                                found_peak.y                             = j - physical_address_of_box_center_y;
                                found_peak.z                             = k - physical_address_of_box_center_z;
                                found_peak.physical_address_within_image = pixel_counter;
                                // wxPrintf("new peak %f, %f, %f (%f)\n", found_peak.x, found_peak.y, found_peak.z,
                                // found_peak.value); wxPrintf("new peak %i, %i, %i (%f)\n", i, j, k, found_peak.value);
                            }
                        }

                        pixel_counter++;
                    }

                    pixel_counter += padding_jump_value;
                }
            }
        }
    }
    else {
        if (radii_are_fractional == true) {
            inv_max_radius_sq_x = 1.0 / powf(physical_address_of_box_center_x, 2);
            inv_max_radius_sq_y = 1.0 / powf(physical_address_of_box_center_y, 2);

            if (logical_z_dimension == 1)
                inv_max_radius_sq_z = 0;
            else
                inv_max_radius_sq_z = 1.0 / powf(physical_address_of_box_center_z, 2);

            for (k = 0; k < logical_z_dimension; k++) {
                kk = k;
                if (kk > physical_address_of_box_center_z)
                    kk -= logical_z_dimension;
                z = powf(float(kk), 2) * inv_max_radius_sq_z;

                for (j = 0; j < logical_y_dimension; j++) {
                    jj = j;
                    if (jj > physical_address_of_box_center_y)
                        jj -= logical_y_dimension;
                    y = powf(float(jj), 2) * inv_max_radius_sq_y;

                    for (i = 0; i < logical_x_dimension; i++) {
                        ii = i;
                        if (ii > physical_address_of_box_center_x)
                            ii -= logical_x_dimension;
                        x = powf(float(ii), 2) * inv_max_radius_sq_x;

                        distance_from_origin = x + y + z;

                        if (distance_from_origin >= wanted_min_radius && distance_from_origin <= wanted_max_radius) {
                            if (real_values[pixel_counter] > found_peak.value) {
                                found_peak.value                         = real_values[pixel_counter];
                                found_peak.x                             = ii;
                                found_peak.y                             = jj;
                                found_peak.z                             = kk;
                                found_peak.physical_address_within_image = pixel_counter;
                            }
                        }

                        // wxPrintf("new peak %f, %f, %f (%f)\n", found_peak.x, found_peak.y, found_peak.z,
                        // found_peak.value); wxPrintf("value %f, %f, %f (%f)\n", x, y, z, real_values[pixel_counter]);

                        pixel_counter++;
                    }

                    pixel_counter += padding_jump_value;
                }
            }
        }
        else {
            for (k = 0; k < logical_z_dimension; k++) {
                kk = k;
                if (kk > physical_address_of_box_center_z)
                    kk -= logical_z_dimension;
                z = powf(float(kk), 2);

                for (j = 0; j < logical_y_dimension; j++) {
                    jj = j;
                    if (jj > physical_address_of_box_center_y)
                        jj -= logical_y_dimension;
                    y = powf(float(jj), 2);

                    for (i = 0; i < logical_x_dimension; i++) {
                        ii = i;
                        if (ii > physical_address_of_box_center_x)
                            ii -= logical_x_dimension;
                        x = powf(float(ii), 2);

                        distance_from_origin = x + y + z;

                        if (distance_from_origin >= wanted_min_radius && distance_from_origin <= wanted_max_radius) {
                            if (real_values[pixel_counter] > found_peak.value) {
                                found_peak.value                         = real_values[pixel_counter];
                                found_peak.x                             = ii;
                                found_peak.y                             = jj;
                                found_peak.z                             = kk;
                                found_peak.physical_address_within_image = pixel_counter;
                            }
                        }

                        pixel_counter++;
                    }

                    pixel_counter += padding_jump_value;
                }
            }
        }
    }

    return found_peak;
}

float Image::FindBeamTilt(CTF& input_ctf, float pixel_size, Image& phase_error_output, Image& beamtilt_output,
                          Image& difference_image, float& beamtilt_x, float& beamtilt_y, float& particle_shift_x,
                          float& particle_shift_y, float phase_multiplier, bool progress_bar,
                          int first_position_to_search, int last_position_to_search, MyApp* app_for_result) {
    int cycle_counter;
    int counter;

    float current_beamtilt_azimuth;
    float best_beamtilt_azimuth     = 0.0f;
    float best_max_beamtilt_azimuth = 0.0f;

    float current_beamtilt;
    float best_beamtilt     = 0.0f;
    float best_max_beamtilt = 0.0f;

    float current_particle_shift;
    float best_particle_shift     = 0.0f;
    float best_max_particle_shift = 0.0f;

    float current_particle_shift_azimuth;
    float best_particle_shift_azimuth     = 0.0f;
    float best_max_particle_shift_azimuth = 0.0f;

    float score;
    float best_max_score;
    float best_alternate_score;

    float best_score;
    float mask_radius_local;

    float beam_tilt_search_start_value = 0.0f;
    float beam_tilt_search_end_value   = 0.005f; // 5 radians
    float beam_tilt_search_step_size   = 0.0001; // 0.1 radians

    float beam_tilt_azimuth_search_start_value = -PI;
    float beam_tilt_azimuth_search_end_value   = PI;
    float beam_tilt_azimuth_search_step_size   = PI / 180.0f; // 1 degree steps

    float particle_shift_search_start_value = 0.0f;
    float particle_shift_search_end_value   = 1.0f;
    float particle_shift_search_step_size   = 0.1f;

    float particle_shift_azimuth_search_start_value; // these will be set off the initial angle search
    float particle_shift_azimuth_search_end_value;
    float particle_shift_azimuth_search_step_size;

    AnglesAndShifts rotation_matrix;
    ProgressBar*    my_progress;
    Image*          phase_difference_spectrum = new Image;
    phase_difference_spectrum->Allocate(logical_x_dimension, logical_y_dimension, false);
    Image* beamtilt_spectrum = new Image;
    beamtilt_spectrum->Allocate(logical_x_dimension, logical_y_dimension, false);
    Image* temp_image = new Image;
    temp_image->Allocate(logical_x_dimension, logical_y_dimension, false);

    // Determine filter
    ComputeAmplitudeSpectrumFull2D(temp_image);
    temp_image->MultiplyByConstant(float(temp_image->logical_x_dimension));
    temp_image->Binarise(0.00002f);

    // estimate a mask radius
    mask_radius_local = sqrtf(
        (temp_image->ReturnAverageOfRealValues() * temp_image->logical_x_dimension * temp_image->logical_y_dimension) /
        PI);
    // wxPrintf("mask radius = %f\n", mask_radius_local);

    ComputeAmplitudeSpectrumFull2D(phase_difference_spectrum, true);
    // phase_difference_spectrum->Binarise(0.0f);
    temp_image->CopyFrom(phase_difference_spectrum);
    temp_image->ApplyMirrorAlongY();

    best_score = std::numeric_limits<float>::max();

    for (cycle_counter = 0; cycle_counter < 1; cycle_counter++) {
        counter = 0;
        for (current_beamtilt_azimuth = beam_tilt_azimuth_search_start_value;
             current_beamtilt_azimuth <= beam_tilt_azimuth_search_end_value;
             current_beamtilt_azimuth += beam_tilt_azimuth_search_step_size) {
            //  This method rotates the image at "input angle" back by that amount, so give the negative value.
            rotation_matrix.GenerateRotationMatrix2D(rad_2_deg(-1.0f * current_beamtilt_azimuth));
            temp_image->Rotate2D(*beamtilt_spectrum, rotation_matrix, 0.45 * beamtilt_spectrum->logical_x_dimension);
            beamtilt_spectrum->SubtractImage(phase_difference_spectrum);
            score = beamtilt_spectrum->ReturnSumOfSquares(mask_radius_local);

            if (score < best_score) {
                best_score            = score;
                best_beamtilt_azimuth = current_beamtilt_azimuth;
                //		wxPrintf("Writing for %f, %f", rad_2_deg(current_beamtilt_azimuth), score);
            }

            counter++;
            // wxPrintf("cycle %i, search %i\n", cycle_counter + 1, counter);
        }

        // set the next values for the next cycle..

        beam_tilt_azimuth_search_start_value = best_beamtilt_azimuth - beam_tilt_azimuth_search_step_size;
        beam_tilt_azimuth_search_end_value   = best_beamtilt_azimuth + beam_tilt_azimuth_search_step_size;
        beam_tilt_azimuth_search_step_size /= 10.0f; // 10 times finer search;
    }

    best_beamtilt_azimuth /= 2.0f;
    MyDebugPrint("Best Azimuth = %f\n", best_beamtilt_azimuth);

    ComputeAmplitudeSpectrumFull2D(phase_difference_spectrum, true, phase_multiplier);
    phase_error_output.CopyFrom(phase_difference_spectrum);
    // phase_difference_spectrum->Binarise(0.0f);

    best_score     = std::numeric_limits<float>::max();
    best_max_score = std::numeric_limits<float>::lowest();

    beam_tilt_azimuth_search_start_value = (best_beamtilt_azimuth)-0.1f; // search 10 degrees either side
    beam_tilt_azimuth_search_end_value   = (best_beamtilt_azimuth) + 0.1f;
    beam_tilt_azimuth_search_step_size   = 0.025f;

    particle_shift_azimuth_search_start_value = -0.1f;
    particle_shift_azimuth_search_end_value   = 0.1f;
    particle_shift_azimuth_search_step_size   = 0.025f;

    beam_tilt_search_start_value = -0.005f;
    beam_tilt_search_end_value   = 0.005f;  // 5 radians
    beam_tilt_search_step_size   = 0.0001f; // 0.1 radians

    particle_shift_search_start_value = -1.0f;
    particle_shift_search_end_value   = 1.0f;
    particle_shift_search_step_size   = 0.05f;

    // count number of points in the refinement for the progress bar (this must be kept the same as the loops below -
    // Messy, sorry)

    counter = 0;

    for (current_beamtilt_azimuth = beam_tilt_azimuth_search_start_value;
         current_beamtilt_azimuth <= beam_tilt_azimuth_search_end_value;
         current_beamtilt_azimuth += beam_tilt_azimuth_search_step_size) {
        for (current_particle_shift_azimuth = particle_shift_azimuth_search_start_value;
             current_particle_shift_azimuth <= particle_shift_azimuth_search_end_value;
             current_particle_shift_azimuth += particle_shift_azimuth_search_step_size) {
            for (current_beamtilt = beam_tilt_search_start_value; current_beamtilt <= beam_tilt_search_end_value;
                 current_beamtilt += beam_tilt_search_step_size) {
                for (current_particle_shift = particle_shift_search_start_value;
                     current_particle_shift <= particle_shift_search_end_value;
                     current_particle_shift += particle_shift_search_step_size) {
                    if (counter >= first_position_to_search && counter <= last_position_to_search)
                        counter++;
                    //	wxPrintf("%i = %f, %f, %f, %f\n",cycle_counter, current_beamtilt_azimuth,
                    //current_particle_shift_azimuth, current_beamtilt, current_particle_shift);
                }
            }
        }
    }

    //	wxPrintf("There are %i positions\n", counter);
    // actual search

    beam_tilt_azimuth_search_start_value = (best_beamtilt_azimuth)-0.1f; // search 10 degrees either side
    beam_tilt_azimuth_search_end_value   = (best_beamtilt_azimuth) + 0.1f;
    beam_tilt_azimuth_search_step_size   = 0.025f;

    particle_shift_azimuth_search_start_value = -0.1f;
    particle_shift_azimuth_search_end_value   = 0.1f;
    particle_shift_azimuth_search_step_size   = 0.025f;

    beam_tilt_search_start_value = -0.005f;
    beam_tilt_search_end_value   = 0.005f;  // 5 milli-radians
    beam_tilt_search_step_size   = 0.0001f; // 0.1 milli-radians

    particle_shift_search_start_value = -1.0f;
    particle_shift_search_end_value   = 1.0f;
    particle_shift_search_step_size   = 0.05f;

    if (progress_bar && ReturnThreadNumberOfCurrentThread() == 0) {
        wxPrintf("\nEstimating beamtilt...\n\n");
        my_progress = new ProgressBar(counter);
    }

    // mask out the centre..

    phase_difference_spectrum->CosineRingMask(5.0f, phase_difference_spectrum->ReturnLargestLogicalDimension(), 2.0f);

    // setup a scorer - this is important for running the subsequent downhill simplex..

    BeamTiltScorer tilt_scorer(&input_ctf, phase_difference_spectrum, pixel_size, mask_radius_local, phase_multiplier);
    double         input_values[5];

    counter                  = 0;
    int progress_bar_counter = 0;

    for (current_beamtilt_azimuth = beam_tilt_azimuth_search_start_value;
         current_beamtilt_azimuth <= beam_tilt_azimuth_search_end_value;
         current_beamtilt_azimuth += beam_tilt_azimuth_search_step_size) {
        for (current_particle_shift_azimuth = particle_shift_azimuth_search_start_value;
             current_particle_shift_azimuth <= particle_shift_azimuth_search_end_value;
             current_particle_shift_azimuth += particle_shift_azimuth_search_step_size) {
            for (current_beamtilt = beam_tilt_search_start_value; current_beamtilt <= beam_tilt_search_end_value;
                 current_beamtilt += beam_tilt_search_step_size) {
                for (current_particle_shift = particle_shift_search_start_value;
                     current_particle_shift <= particle_shift_search_end_value;
                     current_particle_shift += particle_shift_search_step_size) {

                    if (counter >= first_position_to_search && counter <= last_position_to_search) {
                        /*					UserInput *test_input = new UserInput("test", 0);
                        float beam_tilt_x = test_input->GetFloatFromUser("Beam Tilt X (mrad)", "", "0") / 1000;
                        float beam_tilt_y = test_input->GetFloatFromUser("Beam Tilt Y (mrad)", "", "0") / 1000;
                        float particle_shift_x = test_input->GetFloatFromUser("Particle Shift X (A)", "", "0") /
                        pixel_size; float particle_shift_y = test_input->GetFloatFromUser("Particle Shift Y (A)", "",
                        "0") / pixel_size; delete test_input;

                        input_ctf.SetBeamTilt(beam_tilt_x, beam_tilt_y, particle_shift_x, particle_shift_y);
                        temp_image->CalculateBeamTiltImage(input_ctf);
                        temp_image->ComputeAmplitudeSpectrumFull2D(beamtilt_spectrum, true, phase_multiplier);
                        //	beamtilt_spectrum->Binarise(0.0f);
                        beamtilt_spectrum->QuickAndDirtyWriteSlice("/tmp/current.mrc", 1, true);
                        */

                        input_values[1] = current_beamtilt_azimuth;
                        input_values[2] = current_beamtilt_azimuth + current_particle_shift_azimuth;
                        input_values[3] = current_beamtilt;
                        input_values[4] = current_particle_shift;
                        score           = tilt_scorer.ScoreValues(input_values);

                        if (score < best_score) {
                            best_score                  = score;
                            best_beamtilt_azimuth       = current_beamtilt_azimuth;
                            best_particle_shift_azimuth = current_particle_shift_azimuth;
                            best_beamtilt               = current_beamtilt;
                            best_particle_shift         = current_particle_shift;

                            //						difference_image.CopyFrom(beamtilt_spectrum);
                            //						beamtilt_output.CopyFrom(temp_image);
                        }

                        if (score > best_max_score) {
                            // Saving the worst score assuming it is near the best magnitude but opposite phase
                            best_max_score                  = score;
                            best_max_beamtilt_azimuth       = current_beamtilt_azimuth;
                            best_max_particle_shift_azimuth = current_particle_shift_azimuth;
                            best_max_beamtilt               = -current_beamtilt;
                            best_max_particle_shift         = -current_particle_shift;

                            //						difference_image.CopyFrom(beamtilt_spectrum);
                            //						beamtilt_output.CopyFrom(temp_image);
                        }

                        progress_bar_counter++;
                        if (progress_bar && ReturnThreadNumberOfCurrentThread() == 0)
                            my_progress->Update(progress_bar_counter);
                        if (app_for_result != NULL) {
                            JobResult* intermediate_result  = new JobResult;
                            intermediate_result->job_number = app_for_result->my_current_job.job_number;
                            float result_line[2];

                            result_line[0] = progress_bar_counter;
                            result_line[1] = score;

                            intermediate_result->SetResult(2, result_line);
                            app_for_result->AddJobToResultQueue(intermediate_result);
                        }
                    }

                    counter++;
                }
            }
        }
    }

    // Before the simplex, check to make sure nothing has gone terribly wrong. If the phase difference image is garbage
    // the simplex can get stuck and not terminate.
    double sanity_check[5];
    sanity_check[0] = 0;
    sanity_check[1] = best_beamtilt_azimuth;
    sanity_check[2] = best_beamtilt_azimuth + best_particle_shift_azimuth;
    sanity_check[3] = best_beamtilt;
    sanity_check[4] = best_particle_shift;
    best_score      = tilt_scorer.ScoreValues(sanity_check);

    temp_image->CopyFrom(&tilt_scorer.temp_image); // should be the best..
    temp_image->ComputeAmplitudeSpectrumFull2D(&beamtilt_output, true, phase_multiplier);

    float test_significance = phase_difference_spectrum->ReturnBeamTiltSignificanceScore(beamtilt_output);

    MyDebugPrint("Before minimization, score = %f / %f\n", best_score, best_max_score);
    MyDebugPrint("Values found = %f, %f, %f, %f\nMax Values   = %f, %f, %f, %f\n", best_beamtilt_azimuth,
                 best_particle_shift_azimuth, best_beamtilt, best_particle_shift, best_max_beamtilt_azimuth,
                 best_max_particle_shift_azimuth, best_max_beamtilt, best_max_particle_shift);
    MyDebugPrint("Significance score on thread %i is %f\n", ReturnThreadNumberOfCurrentThread(), test_significance);

    if (test_significance >
        MINIMUM_BEAM_TILT_SIGNIFICANCE_SCORE / 10.0f) // 10x less stringent than our current final condition. FIXME
                                                      // these should be specified somehwere else
    {
        // do a downhill simplex minimization..

        //	MyDebugPrint("Before minimization, score = %f / %f\n", best_score, best_max_score);
        //	MyDebugPrint("Values found = %f, %f, %f, %f\nMax Values   = %f, %f, %f, %f\n", best_beamtilt_azimuth,
        //best_particle_shift_azimuth, best_beamtilt, best_particle_shift, best_max_beamtilt_azimuth,
        //best_max_particle_shift_azimuth, best_max_beamtilt, best_max_particle_shift);

        DownhillSimplex simplex_minimzer(4);

        double start_values[5];
        double ranges[5];
        double min_values[5];

        start_values[0] = 0;
        start_values[1] = best_beamtilt_azimuth;
        start_values[2] = best_beamtilt_azimuth + best_particle_shift_azimuth;
        start_values[3] = best_beamtilt;
        start_values[4] = best_particle_shift;

        ranges[0] = 0;
        ranges[1] = 0.1;
        ranges[2] = 0.1;
        ranges[3] = 0.0005;
        ranges[4] = 0.2;

        simplex_minimzer.SetIinitalValues(start_values, ranges);

        simplex_minimzer.initial_values[1][1] = best_beamtilt_azimuth * simplex_minimzer.value_scalers[1];
        simplex_minimzer.initial_values[1][2] =
            (best_beamtilt_azimuth + best_particle_shift_azimuth) * simplex_minimzer.value_scalers[2];
        simplex_minimzer.initial_values[1][3] = best_beamtilt * simplex_minimzer.value_scalers[3];
        simplex_minimzer.initial_values[1][4] = best_particle_shift * simplex_minimzer.value_scalers[4];

        simplex_minimzer.initial_values[2][1] =
            simplex_minimzer.initial_values[1][1] + ranges[1] * simplex_minimzer.value_scalers[1];
        simplex_minimzer.initial_values[2][2] =
            simplex_minimzer.initial_values[1][2] + ranges[2] * simplex_minimzer.value_scalers[2];
        simplex_minimzer.initial_values[2][3] =
            simplex_minimzer.initial_values[1][3] + ranges[3] * simplex_minimzer.value_scalers[3];
        simplex_minimzer.initial_values[2][4] =
            simplex_minimzer.initial_values[1][4] + ranges[4] * simplex_minimzer.value_scalers[4];

        simplex_minimzer.initial_values[3][1] =
            simplex_minimzer.initial_values[1][1] - ranges[1] * simplex_minimzer.value_scalers[1];
        simplex_minimzer.initial_values[3][2] =
            simplex_minimzer.initial_values[1][2] - ranges[2] * simplex_minimzer.value_scalers[2];
        simplex_minimzer.initial_values[3][3] =
            simplex_minimzer.initial_values[1][3] - ranges[3] * simplex_minimzer.value_scalers[3];
        simplex_minimzer.initial_values[3][4] =
            simplex_minimzer.initial_values[1][4] - ranges[4] * simplex_minimzer.value_scalers[4];

        simplex_minimzer.initial_values[4][1] =
            simplex_minimzer.initial_values[1][1] - ranges[1] * simplex_minimzer.value_scalers[1];
        simplex_minimzer.initial_values[4][2] =
            simplex_minimzer.initial_values[1][2] + ranges[2] * simplex_minimzer.value_scalers[2];
        simplex_minimzer.initial_values[4][3] =
            simplex_minimzer.initial_values[1][3] - ranges[3] * simplex_minimzer.value_scalers[3];
        simplex_minimzer.initial_values[4][4] =
            simplex_minimzer.initial_values[1][4] + ranges[4] * simplex_minimzer.value_scalers[4];

        simplex_minimzer.initial_values[5][1] =
            simplex_minimzer.initial_values[1][1] + ranges[1] * simplex_minimzer.value_scalers[1];
        simplex_minimzer.initial_values[5][2] =
            simplex_minimzer.initial_values[1][2] - ranges[2] * simplex_minimzer.value_scalers[2];
        simplex_minimzer.initial_values[5][3] =
            simplex_minimzer.initial_values[1][3] + ranges[3] * simplex_minimzer.value_scalers[3];
        simplex_minimzer.initial_values[5][4] =
            simplex_minimzer.initial_values[1][4] - ranges[4] * simplex_minimzer.value_scalers[4];

        // simplex_minimzer.SetIinitalValues(start_values, ranges);
        simplex_minimzer.MinimizeFunction(&tilt_scorer, BeamTiltScoreFunctionForSimplex);
        simplex_minimzer.GetMinimizedValues(min_values);

        //	MyDebugPrint("After Minimization = %f, %f, %f, %f\n", best_beamtilt_azimuth, best_particle_shift_azimuth,
        //best_beamtilt, best_particle_shift);

        best_score = tilt_scorer.ScoreValues(min_values);
        temp_image->CopyFrom(&tilt_scorer.temp_image); // should be the best..
        best_beamtilt_azimuth       = min_values[1];
        best_particle_shift_azimuth = min_values[2];
        best_beamtilt               = min_values[3];
        best_particle_shift         = min_values[4];

        // rerun with the "max" values (aka the worst initial values, with the phase inverted)

        start_values[0] = 0;
        start_values[1] = best_max_beamtilt_azimuth;
        start_values[2] = best_max_beamtilt_azimuth + best_max_particle_shift_azimuth;
        start_values[3] = best_max_beamtilt;
        start_values[4] = best_max_particle_shift;

        simplex_minimzer.SetIinitalValues(start_values, ranges);

        simplex_minimzer.initial_values[1][1] = best_max_beamtilt_azimuth * simplex_minimzer.value_scalers[1];
        simplex_minimzer.initial_values[1][2] =
            (best_max_beamtilt_azimuth + best_max_particle_shift_azimuth) * simplex_minimzer.value_scalers[2];
        simplex_minimzer.initial_values[1][3] = best_max_beamtilt * simplex_minimzer.value_scalers[3];
        simplex_minimzer.initial_values[1][4] = best_max_particle_shift * simplex_minimzer.value_scalers[4];

        simplex_minimzer.initial_values[2][1] =
            simplex_minimzer.initial_values[1][1] + ranges[1] * simplex_minimzer.value_scalers[1];
        simplex_minimzer.initial_values[2][2] =
            simplex_minimzer.initial_values[1][2] + ranges[2] * simplex_minimzer.value_scalers[2];
        simplex_minimzer.initial_values[2][3] =
            simplex_minimzer.initial_values[1][3] + ranges[3] * simplex_minimzer.value_scalers[3];
        simplex_minimzer.initial_values[2][4] =
            simplex_minimzer.initial_values[1][4] + ranges[4] * simplex_minimzer.value_scalers[4];

        simplex_minimzer.initial_values[3][1] =
            simplex_minimzer.initial_values[1][1] - ranges[1] * simplex_minimzer.value_scalers[1];
        simplex_minimzer.initial_values[3][2] =
            simplex_minimzer.initial_values[1][2] - ranges[2] * simplex_minimzer.value_scalers[2];
        simplex_minimzer.initial_values[3][3] =
            simplex_minimzer.initial_values[1][3] - ranges[3] * simplex_minimzer.value_scalers[3];
        simplex_minimzer.initial_values[3][4] =
            simplex_minimzer.initial_values[1][4] - ranges[4] * simplex_minimzer.value_scalers[4];

        simplex_minimzer.initial_values[4][1] =
            simplex_minimzer.initial_values[1][1] - ranges[1] * simplex_minimzer.value_scalers[1];
        simplex_minimzer.initial_values[4][2] =
            simplex_minimzer.initial_values[1][2] + ranges[2] * simplex_minimzer.value_scalers[2];
        simplex_minimzer.initial_values[4][3] =
            simplex_minimzer.initial_values[1][3] - ranges[3] * simplex_minimzer.value_scalers[3];
        simplex_minimzer.initial_values[4][4] =
            simplex_minimzer.initial_values[1][4] + ranges[4] * simplex_minimzer.value_scalers[4];

        simplex_minimzer.initial_values[5][1] =
            simplex_minimzer.initial_values[1][1] + ranges[1] * simplex_minimzer.value_scalers[1];
        simplex_minimzer.initial_values[5][2] =
            simplex_minimzer.initial_values[1][2] - ranges[2] * simplex_minimzer.value_scalers[2];
        simplex_minimzer.initial_values[5][3] =
            simplex_minimzer.initial_values[1][3] + ranges[3] * simplex_minimzer.value_scalers[3];
        simplex_minimzer.initial_values[5][4] =
            simplex_minimzer.initial_values[1][4] - ranges[4] * simplex_minimzer.value_scalers[4];

        // simplex_minimzer.SetIinitalValues(start_values, ranges);
        simplex_minimzer.MinimizeFunction(&tilt_scorer, BeamTiltScoreFunctionForSimplex);
        simplex_minimzer.GetMinimizedValues(min_values);

        best_alternate_score = tilt_scorer.ScoreValues(min_values);

        if (best_alternate_score < best_score) {
            temp_image->CopyFrom(&tilt_scorer.temp_image); // should be the best..
            best_score                  = best_alternate_score;
            best_beamtilt_azimuth       = min_values[1];
            best_particle_shift_azimuth = min_values[2];
            best_beamtilt               = min_values[3];
            best_particle_shift         = min_values[4];
        }

        MyDebugPrint("After, score = %f\n", best_score);
        MyDebugPrint("Best Results = %f, %f, %f, %f\n", best_beamtilt_azimuth, best_particle_shift_azimuth,
                     best_beamtilt, best_particle_shift);
        MyDebugPrint("Minimization took %s\n", simplex_minimzer.ReturnTimeSpanOfMinimization().Format());

        wxPrintf("Values found = %f, %f, %f, %f\n", best_beamtilt_azimuth, best_particle_shift_azimuth, best_beamtilt,
                 best_particle_shift);

        temp_image->ComputeAmplitudeSpectrumFull2D(&beamtilt_output, true, phase_multiplier);

    } // if sanity check on score to skip simplex

    beamtilt_x = best_beamtilt * cosf(best_beamtilt_azimuth);
    beamtilt_y = best_beamtilt * sinf(best_beamtilt_azimuth);

    particle_shift_x = best_particle_shift * cosf(best_particle_shift_azimuth);
    particle_shift_y = best_particle_shift * sinf(best_particle_shift_azimuth);

    difference_image.CopyFrom(temp_image);
    difference_image.SubtractImage(phase_difference_spectrum);

    difference_image.InvertRealValues();

    delete phase_difference_spectrum;
    delete beamtilt_spectrum;
    delete temp_image;

    if (progress_bar && ReturnThreadNumberOfCurrentThread() == 0)
        delete my_progress;

    return best_score;
}

Peak Image::FindPeakWithParabolaFit(float wanted_min_radius, float wanted_max_radius) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Only 2D images supported for now");

    Peak integer_peak;
    Peak found_peak;

    int best_x;
    int best_y;
    int x_counter;
    int y_counter;
    int current_x;
    int current_y;

    float average_of_square = 0.0;
    float scale_factor;
    float scaled_square[3][3];

    float coefficient_one;
    float coefficient_two;
    float coefficient_three;
    float coefficient_four;
    float coefficient_five;
    float coefficient_six;
    float denominator;

    float x_max;
    float y_max;

    integer_peak = FindPeakWithIntegerCoordinates(wanted_min_radius, wanted_max_radius);

    // wxPrintf("Integer Peak = %f, %f\n", integer_peak.x, integer_peak.y);

    best_x = integer_peak.x + physical_address_of_box_center_x;
    best_y = integer_peak.y + physical_address_of_box_center_y;

    for (y_counter = -1; y_counter <= 1; y_counter++) {
        for (x_counter = -1; x_counter <= 1; x_counter++) {
            current_x = best_x + x_counter;
            current_y = best_y + y_counter;

            if (current_x < 0 || current_x >= logical_x_dimension || current_y < 0 || current_y >= logical_y_dimension)
                scaled_square[x_counter + 1][y_counter + 1] = 0.0;
            else
                scaled_square[x_counter + 1][y_counter + 1] = ReturnRealPixelFromPhysicalCoord(current_x, current_y, 0);

            average_of_square += scaled_square[x_counter + 1][y_counter + 1];
        }
    }

    average_of_square /= 9.0;

    if (average_of_square != 0.0)
        scale_factor = 1.0 / average_of_square;
    else
        scale_factor = 1.0;

    for (y_counter = 0; y_counter < 3; y_counter++) {
        for (x_counter = 0; x_counter < 3; x_counter++) {
            scaled_square[x_counter][y_counter] *= scale_factor;
        }
    }

    coefficient_one = (26.0e0 * scaled_square[0][0] - scaled_square[0][1] + 2.0e0 * scaled_square[0][2] -
                       scaled_square[1][0] - 19.0e0 * scaled_square[1][1] - 7.0e0 * scaled_square[1][2] +
                       2.0e0 * scaled_square[2][0] - 7.0e0 * scaled_square[2][1] + 14.0e0 * scaled_square[2][2]) /
                      9.0e0;
    coefficient_two = (8.0e0 * scaled_square[0][0] - 8.0e0 * scaled_square[0][1] + 5.0e0 * scaled_square[1][0] -
                       8.0e0 * scaled_square[1][1] + 3.0e0 * scaled_square[1][2] + 2.0e0 * scaled_square[2][0] -
                       8.0e0 * scaled_square[2][1] + 6.0e0 * scaled_square[2][2]) /
                      (-1.0e0 * 6.0e0);
    coefficient_three = (scaled_square[0][0] - 2.0e0 * scaled_square[0][1] + scaled_square[0][2] + scaled_square[1][0] -
                         2.0e0 * scaled_square[1][1] + scaled_square[1][2] + scaled_square[2][0] -
                         2.0e0 * scaled_square[2][1] + scaled_square[2][2]) /
                        6.0e0;
    coefficient_four = (8.0e0 * scaled_square[0][0] + 5.0e0 * scaled_square[0][1] + 2.0e0 * scaled_square[0][2] -
                        8.0e0 * scaled_square[1][0] - 8.0e0 * scaled_square[1][1] - 8.0e0 * scaled_square[1][2] +
                        3.0e0 * scaled_square[2][1] + 6.0e0 * scaled_square[2][2]) /
                       (-1.0e0 * 6.0e0);
    coefficient_five = (scaled_square[0][0] - scaled_square[0][2] - scaled_square[2][0] + scaled_square[2][2]) / 4.0e0;
    coefficient_six  = (scaled_square[0][0] + scaled_square[0][1] + scaled_square[0][2] - 2.0e0 * scaled_square[1][0] -
                       2.0e0 * scaled_square[1][1] - 2.0e0 * scaled_square[1][2] + scaled_square[2][0] +
                       scaled_square[2][1] + scaled_square[2][2]) /
                      6.0e0;
    denominator = 4.0e0 * coefficient_three * coefficient_six - powf(coefficient_five, 2);

    if (denominator == 0.0)
        found_peak = integer_peak;
    else {
        y_max = (coefficient_four * coefficient_five - 2.0e0 * coefficient_two * coefficient_six) / denominator;
        x_max = (coefficient_two * coefficient_five - 2.0e0 * coefficient_four * coefficient_three) / denominator;

        y_max = y_max - 2.0e0;
        x_max = x_max - 2.0e0;

        if (y_max > 1.05e0 || y_max < -1.05e0)
            y_max = 0.0e0;

        if (x_max > 1.05e0 || x_max < -1.05e0)
            x_max = 0.0e0;

        found_peak = integer_peak;

        found_peak.x += x_max;
        found_peak.y += y_max;

        found_peak.value = 4.0e0 * coefficient_one * coefficient_three * coefficient_six -
                           coefficient_one * powf(coefficient_five, 2) - powf(coefficient_two, 2) * coefficient_six +
                           coefficient_two * coefficient_four * coefficient_five -
                           powf(coefficient_four, 2) * coefficient_three;
        found_peak.value = found_peak.value * average_of_square / denominator;

        if (fabsf((found_peak.value - integer_peak.value) / (found_peak.value + integer_peak.value)) > 0.15)
            found_peak.value = integer_peak.value;
    }

    // wxPrintf("%f %f %f %f\n", integer_peak.x, integer_peak.y, found_peak.x, found_peak.y);
    return found_peak;
}

void Image::SubSampleWithNoisyResampling(Image* first_sampled_image, Image* second_sampled_image) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == false, "Image must be in Fourier space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Only 2D images supported for now");
    MyDebugAssertTrue(HasSameDimensionsAs(first_sampled_image) == true &&
                          HasSameDimensionsAs(second_sampled_image) == true,
                      "Images are different dimensions");

    Image scaled_image;
    Image first_temp_image;
    Image second_temp_image;

    scaled_image.Allocate(logical_x_dimension * 2, logical_y_dimension * 2, false);
    first_temp_image.Allocate(logical_x_dimension * 2, logical_y_dimension * 2, true);
    second_temp_image.Allocate(logical_x_dimension * 2, logical_y_dimension * 2, true);

    float current_sigma = ReturnSigmaOfFourierValuesOnEdgesAndCorners();

    ClipInto(&scaled_image, 0, true, current_sigma);
    scaled_image.BackwardFFT();
    scaled_image.SubSampleMask(&first_temp_image, &second_temp_image);

    first_temp_image.ForwardFFT();
    first_temp_image.ClipInto(first_sampled_image);
    second_temp_image.ForwardFFT();
    second_temp_image.ClipInto(second_sampled_image);
}

void Image::SubSampleMask(Image* first_sampled_image, Image* second_sampled_image) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Only 2D images supported for now");
    //	MyDebugAssertTrue(HasSameDimensionsAs(first_sampled_image) == true && HasSameDimensionsAs(second_sampled_image)
    //== true, "Images are different dimensions");

    long pixel_counter = 0;

    int i;
    int j;

    float first_mask  = 0.;
    float second_mask = 1.;

    bool odd_x;

    if (IsOdd(logical_x_dimension) == true)
        odd_x = true;
    else
        odd_x = false;

    first_sampled_image->CopyFrom(this);
    second_sampled_image->CopyFrom(this);

    for (j = 0; j < logical_y_dimension; j++) {
        for (i = 0; i < logical_x_dimension; i++) {
            if (first_mask == 0)
                first_mask = 1.;
            else
                first_mask = 0;

            if (second_mask == 0)
                second_mask = 1.;
            else
                second_mask = 0;

            first_sampled_image->real_values[pixel_counter] *= first_mask;
            second_sampled_image->real_values[pixel_counter] *= second_mask;

            if (i == logical_x_dimension - 1 && odd_x == false) {
                if (first_mask == 0)
                    first_mask = 1;
                else
                    first_mask = 0;

                if (second_mask == 0)
                    second_mask = 1;
                else
                    second_mask = 0;
            }

            pixel_counter++;
        }

        pixel_counter += padding_jump_value;
    }
}

bool Image::ContainsRepeatedLineEdges() // relion line artifacts
{
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(logical_z_dimension == 1, "3D not implemented");

    bool        weird_edge            = false;
    const float correlation_threshold = 0.85; // this will exclude some valid images as well probably

    // top..

    /*	if (ReturnCorrelationBetweenTwoHorizontalLines(1, 2) > correlation_threshold &&
            ReturnCorrelationBetweenTwoHorizontalLines(1, 3) > correlation_threshold &&
            ReturnCorrelationBetweenTwoHorizontalLines(2, 3) > correlation_threshold) weird_edge = true;
        else
        if (ReturnCorrelationBetweenTwoHorizontalLines(logical_y_dimension - 2, logical_y_dimension - 3) >
       correlation_threshold && ReturnCorrelationBetweenTwoHorizontalLines(logical_y_dimension - 2, logical_y_dimension
       - 4) > correlation_threshold && ReturnCorrelationBetweenTwoHorizontalLines(logical_y_dimension - 3,
       logical_y_dimension - 4) > correlation_threshold) weird_edge = true; else if
       (ReturnCorrelationBetweenTwoVerticalLines(1, 2) > correlation_threshold &&
            ReturnCorrelationBetweenTwoVerticalLines(1, 3) > correlation_threshold &&
            ReturnCorrelationBetweenTwoVerticalLines(2, 3) > correlation_threshold) weird_edge = true;
        else
        if (ReturnCorrelationBetweenTwoVerticalLines(logical_x_dimension - 2, logical_x_dimension - 3) >
       correlation_threshold && ReturnCorrelationBetweenTwoVerticalLines(logical_x_dimension - 2, logical_x_dimension -
       4) > correlation_threshold && ReturnCorrelationBetweenTwoVerticalLines(logical_x_dimension - 3,
       logical_x_dimension - 4) > correlation_threshold) weird_edge = true;

    */
    if (ReturnCorrelationBetweenTwoHorizontalLines(1, 2) > correlation_threshold)
        weird_edge = true;
    else if (ReturnCorrelationBetweenTwoHorizontalLines(logical_y_dimension - 2, logical_y_dimension - 3) >
             correlation_threshold)
        weird_edge = true;
    else if (ReturnCorrelationBetweenTwoVerticalLines(1, 2) > correlation_threshold)
        weird_edge = true;
    else if (ReturnCorrelationBetweenTwoVerticalLines(logical_x_dimension - 2, logical_x_dimension - 3) >
             correlation_threshold)
        weird_edge = true;

    return weird_edge;
}

bool Image::ContainsBlankEdges(float mask_radius) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(logical_z_dimension == 1, "3D not implemented");

    int   i;
    long  pixel_counter;
    float variance;
    float line_average, line_variance;
    float tiny       = 0.0000000001;
    float threshold  = 0.5;
    bool  blank_edge = false;

    // Test all four sides of a 2D image

    variance = 0.0;
    if (mask_radius > 0.0)
        variance = ReturnVarianceOfRealValues(mask_radius);

    line_average  = 0.0;
    line_variance = 0.0;
    for (i = 0; i < logical_x_dimension; i++) {
        line_average += real_values[i];
        line_variance += powf(real_values[i], 2);
    }
    line_variance = line_variance / logical_x_dimension - powf(line_average / logical_x_dimension, 2);
    //	wxPrintf("var = %g line = %g\n", variance, line_variance);
    if (variance > 0.0 && line_variance / variance < threshold)
        blank_edge = true;
    else if (line_variance < logical_x_dimension * tiny)
        blank_edge = true;

    if (!blank_edge) {
        pixel_counter = 0;
        line_average  = 0.0;
        line_variance = 0.0;
        for (i = 0; i < logical_y_dimension; i++) {
            line_average += real_values[pixel_counter];
            line_variance += powf(real_values[pixel_counter], 2);
            pixel_counter += logical_x_dimension + padding_jump_value;
        }
        line_variance = line_variance / logical_y_dimension - powf(line_average / logical_y_dimension, 2);
        //		wxPrintf("var = %g line = %g\n", variance, line_variance);
        if (variance > 0.0 && line_variance / variance < threshold)
            blank_edge = true;
        else if (line_variance < logical_x_dimension * tiny)
            blank_edge = true;
    }

    if (!blank_edge) {
        pixel_counter = logical_x_dimension - 1;
        line_average  = 0.0;
        line_variance = 0.0;
        for (i = 0; i < logical_y_dimension; i++) {
            line_average += real_values[pixel_counter];
            line_variance += powf(real_values[pixel_counter], 2);
            pixel_counter += logical_x_dimension + padding_jump_value;
        }
        line_variance = line_variance / logical_y_dimension - powf(line_average / logical_y_dimension, 2);
        //		wxPrintf("var = %g line = %g\n", variance, line_variance);
        if (variance > 0.0 && line_variance / variance < threshold)
            blank_edge = true;
        else if (line_variance < logical_x_dimension * tiny)
            blank_edge = true;
    }

    if (!blank_edge) {
        pixel_counter = (logical_x_dimension + padding_jump_value) * (logical_y_dimension - 1);
        line_average  = 0.0;
        line_variance = 0.0;
        for (i = 0; i < logical_x_dimension; i++) {
            line_average += real_values[pixel_counter];
            line_variance += powf(real_values[pixel_counter], 2);
            pixel_counter++;
        }
        line_variance = line_variance / logical_x_dimension - powf(line_average / logical_x_dimension, 2);
        //		wxPrintf("var = %g line = %g\n", variance, line_variance);
        if (variance > 0.0 && line_variance / variance < threshold)
            blank_edge = true;
        else if (line_variance < logical_x_dimension * tiny)
            blank_edge = true;
    }

    return blank_edge;
}

void Image::Rotate3DByRotationMatrixAndOrApplySymmetry(RotationMatrix& wanted_matrix, float wanted_max_radius_in_pixels,
                                                       wxString wanted_symmetry) {
    Rotate3DByRotationMatrixAndOrApplySymmetryThenShift(wanted_matrix, 0.0f, 0.0f, 0.0f, wanted_max_radius_in_pixels,
                                                        wanted_symmetry);
}

void Image::Rotate3DByRotationMatrixAndOrApplySymmetryThenShift(RotationMatrix& wanted_matrix, float wanted_x_shift,
                                                                float wanted_y_shift, float wanted_z_shift,
                                                                float    wanted_max_radius_in_pixels,
                                                                wxString wanted_symmetry) // like above but with shift
{
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    long  pixel_counter = 0;
    int   i, j, k;
    float x, y, z;
    int   symmetry_counter;
    float x_squared, y_squared, z_squared;
    float rotated_x, rotated_y, rotated_z;

    Image          buffer_image;
    SymmetryMatrix symmetry_matrices;
    RotationMatrix temp_matrix;
    symmetry_matrices.Init(wanted_symmetry);

    buffer_image.Allocate(logical_x_dimension, logical_y_dimension, logical_z_dimension);
    buffer_image.SetToConstant(0.0f);

    float max_radius_squared;
    if (wanted_max_radius_in_pixels == 0.0f)
        wanted_max_radius_in_pixels = ReturnSmallestLogicalDimension() / 2.0f - 1.0f;
    max_radius_squared = powf(wanted_max_radius_in_pixels, 2);

    for (k = 0; k < logical_z_dimension; k++) {
        z         = k - physical_address_of_box_center_z;
        z_squared = powf(z, 2);

        for (j = 0; j < logical_y_dimension; j++) {
            y         = j - physical_address_of_box_center_y;
            y_squared = powf(y, 2);

            for (i = 0; i < logical_x_dimension; i++) {
                x = i - physical_address_of_box_center_x;

                if (z_squared + y_squared + powf(x, 2) < max_radius_squared) {
                    for (symmetry_counter = 0; symmetry_counter < symmetry_matrices.number_of_matrices;
                         symmetry_counter++) {
                        temp_matrix = symmetry_matrices.rot_mat[symmetry_counter] * wanted_matrix;
                        temp_matrix.RotateCoords(x, y, z, rotated_x, rotated_y, rotated_z);

                        rotated_x += float(physical_address_of_box_center_x) + wanted_x_shift;
                        rotated_y += float(physical_address_of_box_center_y) + wanted_y_shift;
                        rotated_z += float(physical_address_of_box_center_z) + wanted_z_shift;

                        if (rotated_x >= 0 && rotated_x < logical_x_dimension - 1 && rotated_y >= 0 &&
                            rotated_y < logical_y_dimension - 1 && rotated_z >= 0 &&
                            rotated_z < logical_z_dimension - 1) {
                            buffer_image.AddByLinearInterpolationReal(rotated_x, rotated_y, rotated_z,
                                                                      real_values[pixel_counter]);
                        }
                    }
                }

                pixel_counter++;
            }

            pixel_counter += padding_jump_value;
        }
    }

    Consume(&buffer_image);
}

void Image::Rotate3DThenShiftThenApplySymmetry(RotationMatrix& wanted_matrix, float wanted_x_shift,
                                               float wanted_y_shift, float wanted_z_shift,
                                               float wanted_max_radius_in_pixels, wxString wanted_symmetry) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    long  pixel_counter = 0;
    int   i, j, k;
    float x, y, z;
    int   symmetry_counter;
    float x_squared, y_squared, z_squared;
    float rotated_x, rotated_y, rotated_z;
    float symmetry_x, symmetry_y, symmetry_z;

    Image          buffer_image;
    SymmetryMatrix symmetry_matrices;
    RotationMatrix temp_matrix;
    symmetry_matrices.Init(wanted_symmetry);

    buffer_image.Allocate(logical_x_dimension, logical_y_dimension, logical_z_dimension);
    buffer_image.SetToConstant(0.0f);

    float max_radius_squared;
    if (wanted_max_radius_in_pixels == 0.0f)
        wanted_max_radius_in_pixels = ReturnSmallestLogicalDimension() / 2.0f - 1.0f;
    max_radius_squared = powf(wanted_max_radius_in_pixels, 2);

    for (k = 0; k < logical_z_dimension; k++) {
        z         = k - physical_address_of_box_center_z;
        z_squared = powf(z, 2);

        for (j = 0; j < logical_y_dimension; j++) {
            y         = j - physical_address_of_box_center_y;
            y_squared = powf(y, 2);

            for (i = 0; i < logical_x_dimension; i++) {
                x = i - physical_address_of_box_center_x;

                if (z_squared + y_squared + powf(x, 2) < max_radius_squared) {
                    wanted_matrix.RotateCoords(x, y, z, rotated_x, rotated_y, rotated_z);
                    rotated_x += wanted_x_shift;
                    rotated_y += wanted_y_shift;
                    rotated_z += wanted_z_shift;

                    for (symmetry_counter = 0; symmetry_counter < symmetry_matrices.number_of_matrices;
                         symmetry_counter++) {

                        symmetry_matrices.rot_mat[symmetry_counter].RotateCoords(rotated_x, rotated_y, rotated_z,
                                                                                 symmetry_x, symmetry_y, symmetry_z);

                        symmetry_x += float(physical_address_of_box_center_x);
                        symmetry_y += float(physical_address_of_box_center_y);
                        symmetry_z += float(physical_address_of_box_center_z);

                        if (symmetry_x >= 0 && symmetry_x < logical_x_dimension - 1 && symmetry_y >= 0 &&
                            symmetry_y < logical_y_dimension - 1 && symmetry_z >= 0 &&
                            symmetry_z < logical_z_dimension - 1) {
                            buffer_image.AddByLinearInterpolationReal(symmetry_x, symmetry_y, symmetry_z,
                                                                      real_values[pixel_counter]);
                        }
                    }
                }

                pixel_counter++;
            }

            pixel_counter += padding_jump_value;
        }
    }

    Consume(&buffer_image);
}

// Note: This function allocates new memory, so it is not really "in-place" compared to FFT, but only
// in that the same Image object is retained.
void Image::Rotate2DInPlace(float rotation_in_degrees, float mask_radius_in_pixels) {
    Image buffer_image;
    buffer_image.Allocate(logical_x_dimension, logical_y_dimension, true);
    AnglesAndShifts rotation;
    rotation.GenerateRotationMatrix2D(rotation_in_degrees);
    Rotate2D(buffer_image, rotation, mask_radius_in_pixels);
    Consume(&buffer_image);
}

// Note: This function allocates new memory, so it is not really "in-place" compared to FFT, but only
// in that the same Image object is retained.
void Image::RotateInPlaceAboutZBy90Degrees(bool rotate_by_positive_90_degrees, bool preserve_origin) {
    // Rotate without interpolation by swapping indices.
    // For an even sized image the image feature at the box center will be shifted -1 in x (positive rotation) or -1 in
    // y (negative rotation) Include and additional shift to compensate if(preserve_origin) Note: you lose a row of
    // pixels if true.
    Image buffer_image;
    buffer_image.Allocate(logical_y_dimension, logical_x_dimension, logical_z_dimension, true);
    long current_address_old_image = 0;
    int  k;
    int  y_new = -1;
    int  x_new = -1;
    int  y_old = 0;
    int  x_old = 0;

    int shift_value = 0;
    if (preserve_origin) {
        if (rotate_by_positive_90_degrees && IsEven(logical_y_dimension))
            shift_value = 1;
        else if (!rotate_by_positive_90_degrees && IsEven(logical_x_dimension))
            shift_value = 1;
        else
            shift_value = 0;
    }
    // Negative rotation is clockwise looking down the image normal (Y --> X)
    if (rotate_by_positive_90_degrees) {
        for (k = 0; k < logical_z_dimension; k++) {
            for (y_old = 0; y_old < logical_y_dimension; y_old++) {
                x_new = logical_y_dimension - y_old - 1 + shift_value;
                for (x_old = 0; x_old < logical_x_dimension; x_old++) {
                    y_new = x_old;
                    if (x_new == logical_y_dimension) {
                        // We are shifting the new indices +1 in X, so will be out of bounds here. B/c x(0) is undefined
                        // with this simple index interpolation, set it to zero.
                        buffer_image.real_values[buffer_image.ReturnReal1DAddressFromPhysicalCoord(0, y_new, k)] = 0.f;
                    }
                    else {
                        buffer_image.real_values[buffer_image.ReturnReal1DAddressFromPhysicalCoord(x_new, y_new, k)] =
                            ReturnRealPixelFromPhysicalCoord(x_old, y_old, k);
                    }
                }
            }
        }
    }
    else {
        for (k = 0; k < logical_z_dimension; k++) {
            for (y_old = 0; y_old < logical_y_dimension; y_old++) {
                x_new = y_old;
                for (x_old = shift_value; x_old < logical_x_dimension; x_old++) {
                    y_new = logical_x_dimension - x_old - 1 + shift_value;
                    if (y_new == logical_x_dimension) {
                        buffer_image.real_values[buffer_image.ReturnReal1DAddressFromPhysicalCoord(x_new, 0, k)] = 0.f;
                    }
                    else {
                        buffer_image.real_values[buffer_image.ReturnReal1DAddressFromPhysicalCoord(x_new, y_new, k)] =
                            ReturnRealPixelFromPhysicalCoord(x_old, y_old, k);
                    }
                }
            }
        }
    }
    Consume(&buffer_image);
}

void Image::Rotate2D(Image& rotated_image, AnglesAndShifts& rotation_angle, float mask_radius_in_pixels) {
    MyDebugAssertTrue(rotated_image.logical_z_dimension == 1, "Error: attempting to rotate into 3D image");
    MyDebugAssertTrue(logical_z_dimension == 1, "Error: attempting to rotate from 3D image");
    MyDebugAssertTrue(rotated_image.is_in_memory, "Memory not allocated for receiving image");
    MyDebugAssertTrue(is_in_real_space, "Image is in Fourier space");
    //	MyDebugAssertTrue(IsSquare(), "Image to rotate is not square");
    MyDebugAssertTrue(rotated_image.logical_x_dimension == logical_x_dimension &&
                          rotated_image.logical_y_dimension == logical_y_dimension,
                      "Error: Images different sizes");
    MyDebugAssertTrue(object_is_centred_in_box, "Image not centered in box");

    int i;
    int j;

    long pixel_counter = 0;

    float x_coordinate_2d;
    float y_coordinate_2d;

    float x_rotated;
    float y_rotated;

    float mask_radius_sq;
    float y_rad_sq;

    //	float edge_value = ReturnAverageOfRealValuesOnEdges();
    float edge_value = ReturnAverageOfRealValues(
        std::min(physical_address_of_box_center_x - 2, physical_address_of_box_center_y - 2), true);

    if (mask_radius_in_pixels == 0.0) {
        mask_radius_sq = powf(physical_address_of_box_center_x - 1, 2);
    }
    else {
        mask_radius_sq = powf(mask_radius_in_pixels, 2);
    }

    for (j = 0; j < rotated_image.logical_y_dimension; j++) {
        y_coordinate_2d = j - physical_address_of_box_center_y;
        y_rad_sq        = powf(y_coordinate_2d, 2);
        for (i = 0; i < rotated_image.logical_x_dimension; i++) {
            x_coordinate_2d = i - physical_address_of_box_center_x;
            if (y_rad_sq + powf(x_coordinate_2d, 2) < mask_radius_sq) {
                rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_rotated, y_rotated);
                x_rotated += physical_address_of_box_center_x;
                y_rotated += physical_address_of_box_center_y;
                rotated_image.real_values[pixel_counter] = ReturnLinearInterpolated2D(x_rotated, y_rotated);
                //				wxPrintf("x_coordinate_2d, y_coordinate_2d, x_rotated, y_totated = %g, %g, %g, %g, value =
                //%g\n", x_coordinate_2d, y_coordinate_2d, x_rotated, y_rotated,
                //rotated_image.real_values[pixel_counter]);
            }
            else {
                rotated_image.real_values[pixel_counter] = edge_value;
            }
            pixel_counter++;
        }
        pixel_counter += rotated_image.padding_jump_value;
    }

    rotated_image.is_in_real_space         = true;
    rotated_image.object_is_centred_in_box = true;
}

void Image::Rotate2DSample(Image& rotated_image, AnglesAndShifts& rotation_angle, float mask_radius_in_pixels) {
    MyDebugAssertTrue(rotated_image.logical_z_dimension == 1, "Error: attempting to rotate into 3D image");
    MyDebugAssertTrue(logical_z_dimension == 1, "Error: attempting to rotate from 3D image");
    MyDebugAssertTrue(rotated_image.is_in_memory, "Memory not allocated for receiving image");
    MyDebugAssertTrue(is_in_real_space, "Image is in Fourier space");
    MyDebugAssertTrue(IsSquare(), "Image to rotate is not square");
    //	MyDebugAssertTrue(padding_factor * rotated_image.logical_x_dimension == logical_x_dimension && padding_factor *
    //rotated_image.logical_y_dimension == logical_y_dimension, "Error: Images have different sizes");
    MyDebugAssertTrue(object_is_centred_in_box, "Image not centered in box");

    int i;
    int j;

    long pixel_counter = 0;

    float x_coordinate_2d;
    float y_coordinate_2d;

    float x_rotated;
    float y_rotated;

    float mask_radius_sq;
    float y_rad_sq;

    float padding_factor = float(logical_x_dimension) / float(rotated_image.logical_x_dimension);

    //	float edge_value = ReturnAverageOfRealValuesOnEdges();
    float edge_value = ReturnAverageOfRealValues(physical_address_of_box_center_x - 2.0, true);

    if (mask_radius_in_pixels == 0.0) {
        mask_radius_sq = powf(physical_address_of_box_center_x - 1, 2);
    }
    else {
        mask_radius_sq = powf(mask_radius_in_pixels, 2);
    }

    for (j = 0; j < rotated_image.logical_y_dimension; j++) {
        y_coordinate_2d = (j - rotated_image.physical_address_of_box_center_y) * padding_factor;
        y_rad_sq        = powf(y_coordinate_2d, 2);
        for (i = 0; i < rotated_image.logical_x_dimension; i++) {
            x_coordinate_2d = (i - rotated_image.physical_address_of_box_center_x) * padding_factor;
            if (y_rad_sq + powf(x_coordinate_2d, 2) < mask_radius_sq) {
                rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_rotated, y_rotated);
                x_rotated += physical_address_of_box_center_x;
                y_rotated += physical_address_of_box_center_y;
                rotated_image.real_values[pixel_counter] = ReturnLinearInterpolated2D(x_rotated, y_rotated);
                //				rotated_image.real_values[pixel_counter] = ReturnNearest2D(x_rotated, y_rotated);
                //				wxPrintf("x_coordinate_2d, y_coordinate_2d, x_rotated, y_totated = %g, %g, %g, %g, value =
                //%g\n", x_coordinate_2d, y_coordinate_2d, x_rotated, y_rotated,
                //rotated_image.real_values[pixel_counter]);
            }
            else {
                rotated_image.real_values[pixel_counter] = edge_value;
            }
            pixel_counter++;
        }
        pixel_counter += rotated_image.padding_jump_value;
    }

    rotated_image.is_in_real_space         = true;
    rotated_image.object_is_centred_in_box = true;
}

float Image::Skew2D(Image& skewed_image, float height_offset, float minimum_height, float tilt_axis, float tilt_angle,
                    bool adjust_signal) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(skewed_image.is_in_memory, "skewed_image memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");
    MyDebugAssertTrue(height_offset > 0.0f, "height_offset is <= 0");

    int i, j;
    //	int height_max_i;
    //	int height_max_j;
    int  offset_i;
    int  offset_j;
    long pixel_counter;
    long nonzero_pixels = 0;
    //	long counter_offset;
    //	float pad_value = ReturnAverageOfRealValues();
    float pad_value = ReturnAverageOfRealValuesOnEdges();
    float height;
    float height_max;
    //	float shrink_factor;
    float shrink_factor_squared;
    float tan_angle;
    float x_coordinate_2d;
    float y_coordinate_2d;
    float x_rotated;
    float y_rotated;
    float x_skewed;
    float y_skewed;
    float y0;
    float y0_skewed;
    float min_x, max_x, min_y, max_y;
    float padding_factor = float(logical_x_dimension) / float(skewed_image.logical_x_dimension);

    skewed_image.is_in_real_space = true;
    skewed_image.SetToConstant(pad_value);

    if (fabsf(tilt_angle) < 0.1f) {
        pixel_counter = 0;
        for (j = 0; j < skewed_image.logical_y_dimension; j++) {
            for (i = 0; i < skewed_image.logical_x_dimension; i++) {
                x_rotated = i * padding_factor;
                y_rotated = j * padding_factor;
                //				skewed_image.real_values[pixel_counter] = ReturnNearest2D(x_rotated, y_rotated);
                skewed_image.real_values[pixel_counter] = ReturnLinearInterpolated2D(x_rotated, y_rotated);
                //				nonzero_pixels++;

                pixel_counter++;
            }
            pixel_counter += skewed_image.padding_jump_value;
        }
        return 1.0f;
    }

    AnglesAndShifts rotation_angle;
    AnglesAndShifts inverse_rotation_angle;

    if (tilt_angle < 0.0f) {
        tilt_angle = -tilt_angle;
        tilt_axis += 180.0f;
    }
    tan_angle = tanf(deg_2_rad(tilt_angle));

    // Add 90deg to tilt_axis since we need direction normal to axis (see original ctftilt code)
    //	rotation_angle.GenerateRotationMatrix2D(tilt_axis + 90.0f);
    //	inverse_rotation_angle.GenerateRotationMatrix2D(-tilt_axis - 90.0f);
    rotation_angle.GenerateRotationMatrix2D(tilt_axis);
    inverse_rotation_angle.GenerateRotationMatrix2D(-tilt_axis);

    //	if (tan_angle > 0.01f)
    //	{
    //		y0 = 2.0f * sqrtf(height_offset) / tan_angle * (sqrtf(minimum_height) - sqrtf(minimum_height +
    //height_offset));
    y0        = -2.0f * height_offset / tan_angle;
    y0_skewed = -height_offset / tan_angle;
    //	}
    //	else
    //	{
    //		y0 = FLT_MAX;
    //		y0_skewed = - FLT_MAX;
    //	}
    //	wxPrintf("y0, y0_skewed = %g %g\n", y0, y0_skewed);

    //	height_max = -FLT_MAX;
    min_x         = float(skewed_image.logical_x_dimension) - 1.0f;
    max_x         = 0.0f;
    min_y         = float(skewed_image.logical_y_dimension) - 1.0f;
    max_y         = 0.0f;
    pixel_counter = 0;
    for (j = 0; j < skewed_image.logical_y_dimension; j++) {
        y_coordinate_2d = j - skewed_image.physical_address_of_box_center_y;
        for (i = 0; i < skewed_image.logical_x_dimension; i++) {
            x_coordinate_2d = i - skewed_image.physical_address_of_box_center_x;
            rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_rotated, y_rotated);

            //			y_skewed = powf(y_rotated, 2) / 4.0f * tan_angle / height_offset + y_rotated * sqrtf(minimum_height
            /// height_offset + 1.0f);
            y_skewed = powf(y_rotated, 2) / 4.0f * tan_angle / height_offset + y_rotated;
            if (y_rotated < y0)
                y_skewed = y0 - y_skewed;

            height = fabsf(height_offset + y_skewed * tan_angle);
            if (height > minimum_height) {
                //			if (y_rotated * tan_angle >= - height_offset) y_skewed = powf(y_rotated, 2) / 4.0f * tan_angle /
                //height_offset + y_rotated * sqrtf(minimum_height / height_offset + 1.0f); 			else y_skewed =
                //powf(y_rotated, 2) / 4.0f * tan_angle / height_offset - y_rotated * sqrtf(minimum_height /
                //height_offset - 1.0f); 			x_skewed = x_rotated * sqrtf((fabsf(height_offset + y_skewed * tan_angle) +
                //minimum_height) / height_offset);
                x_skewed = x_rotated * sqrtf(height / height_offset);

                inverse_rotation_angle.euler_matrix.RotateCoords2D(x_skewed, y_skewed, x_rotated, y_rotated);
                x_rotated += skewed_image.physical_address_of_box_center_x;
                y_rotated += skewed_image.physical_address_of_box_center_y;

                min_x = std::min(x_rotated, min_x);
                max_x = std::max(x_rotated, max_x);
                min_y = std::min(y_rotated, min_y);
                max_y = std::max(y_rotated, max_y);
                //			height = fabsf(height_offset + y_skewed * tan_angle) + minimum_height;
                //			if (height > height_max)
                //			{
                //				height_max_i = i;
                //				height_max_j = j;
                //				height_max = height;
                //			}
            }
            pixel_counter++;
        }
        pixel_counter += skewed_image.padding_jump_value;
    }
    offset_i = skewed_image.physical_address_of_box_center_x - myroundint((min_x + max_x) / 2.0f);
    offset_j = skewed_image.physical_address_of_box_center_y - myroundint((min_y + max_y) / 2.0f);

    //	counter_offset = long(logical_x_dimension + padding_jump_value) * long(offset_j) + long(offset_i);
    //	offset_i = myroundint((physical_address_of_box_center_x - (min_x + max_x)) / 2.0f);
    //	offset_j = myroundint((physical_address_of_box_center_y - (min_y + max_y)) / 2.0f);
    //	if (min_x < 0.0f) offset_i = int(ceil(-min_x));
    //	if (max_x > float(logical_x_dimension) - 1.0f) offset_i = - int(ceil(max_x - float(logical_x_dimension)
    //+ 1.0f)); 	if (min_y < 0.0f) offset_j = int(ceil(-min_y)); 	if (max_y > float(logical_y_dimension) - 1.0f) offset_j
    //= - int(ceil(max_y - float(logical_y_dimension) + 1.0f)); 	wxPrintf("minx, maxx, miny, maxy, offx, offy = %g %g %g
    //%g %i %i\n", min_x, max_x, min_y, max_y, offset_i, offset_j);

    //	counter_offset = long(skewed_image.logical_x_dimension + skewed_image.padding_jump_value) * long(1000) +
    //long(500);
    pixel_counter = 0;
    for (j = 0; j < skewed_image.logical_y_dimension; j++) {
        y_coordinate_2d = j - skewed_image.physical_address_of_box_center_y + offset_j;
        for (i = 0; i < skewed_image.logical_x_dimension; i++) {
            x_coordinate_2d = i - skewed_image.physical_address_of_box_center_x + offset_i;
            rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_rotated, y_rotated);

            //			if (pixel_counter == counter_offset) wxPrintf("x_rotated, y_rotated = %g %g\n", x_rotated,
            //y_rotated); 			if (pixel_counter == counter_offset) wxPrintf("tan_angle, height_offset, minimum_height = %g
            //%g %g\n", tan_angle, height_offset, minimum_height);

            //			x_rotated, y_rotated = -296.31 -1764.21
            //			tan_angle, height_offset, minimum_height = 1.19175 1360 60

            //			y_skewed = powf(y_rotated, 2) / 4.0f * tan_angle / height_offset + y_rotated * sqrtf(minimum_height
            /// height_offset + 1.0f);
            y_skewed = powf(y_rotated, 2) / 4.0f * tan_angle / height_offset + y_rotated;
            //			if (x_coordinate_2d == y_coordinate_2d) wxPrintf("tan, off, x, y, rad', rad = %g %g %g %g %g %g\n",
            //tan_angle, height_offset, x_coordinate_2d, y_coordinate_2d, y_rotated, y_skewed); 			if (y_rotated <
            //y0_skewed) y_skewed = y0 - y_skewed;
            if (y_rotated < y0)
                y_skewed = y0 - y_skewed;

            height = fabsf(height_offset + y_skewed * tan_angle);
            if (height > minimum_height) {
                //			if (y_rotated * tan_angle >= - height_offset) y_skewed = powf(y_rotated, 2) / 4.0f * tan_angle /
                //height_offset + y_rotated * sqrtf(minimum_height / height_offset + 1.0f); 			else y_skewed =
                //powf(y_rotated, 2) / 4.0f * tan_angle / height_offset - y_rotated * sqrtf(minimum_height /
                //height_offset - 1.0f); 			x_skewed = x_rotated * sqrtf((fabsf(height_offset + y_skewed * tan_angle) +
                //minimum_height) / height_offset);
                shrink_factor_squared = height / height_offset;
                x_skewed              = x_rotated * sqrtf(shrink_factor_squared);
                //			if (pixel_counter == counter_offset) wxPrintf("x_skewed, y_skewed = %g %g\n", x_skewed,
                //y_skewed);

                inverse_rotation_angle.euler_matrix.RotateCoords2D(x_skewed, y_skewed, x_rotated, y_rotated);
                x_rotated = x_rotated * padding_factor + physical_address_of_box_center_x;
                y_rotated = y_rotated * padding_factor + physical_address_of_box_center_y;
                //			wxPrintf("x in, out = %g %g    y in, out = %g %g\n", x_coordinate_2d, x_rotated,
                //y_coordinate_2d, y_rotated);
                // Keep margin of one pixel to make sure interpolation is reasonable.
                if (x_rotated >= 1.0f && x_rotated < logical_x_dimension - 1 && y_rotated >= 1.0f &&
                    y_rotated < logical_y_dimension - 1) {
                    //				skewed_image.real_values[pixel_counter] = ReturnNearest2D(x_rotated, y_rotated);
                    skewed_image.real_values[pixel_counter] = ReturnLinearInterpolated2D(x_rotated, y_rotated);
                    if (adjust_signal)
                        skewed_image.real_values[pixel_counter] *= shrink_factor_squared;
                    nonzero_pixels++;
                }
                //			else
                //			{
                //				skewed_image.real_values[pixel_counter] = pad_value;
                //				nonzero_pixels++;
                //			}
            }

            pixel_counter++;
        }
        pixel_counter += skewed_image.padding_jump_value;
    }

    return float(nonzero_pixels) / float(skewed_image.number_of_real_space_pixels);
}

// BEGIN_FOR_STAND_ALONE_CTFFIND
float Image::ReturnLinearInterpolated2D(float& wanted_physical_x_coordinate, float& wanted_physical_y_coordinate) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Is in Fourier space");

    if (wanted_physical_x_coordinate < 0 || wanted_physical_x_coordinate > logical_x_dimension - 1)
        return 0.0;
    if (wanted_physical_y_coordinate < 0 || wanted_physical_y_coordinate > logical_y_dimension - 1)
        return 0.0;

    int i;
    int j;
    int int_x_coordinate;
    int int_y_coordinate;
    int int_x_coordinate1;
    int int_y_coordinate1;
    int int_y;

    float weight_x;
    float weight_y;

    float sum = 0.0;

    int_x_coordinate  = int(floorf(wanted_physical_x_coordinate));
    int_y_coordinate  = int(floorf(wanted_physical_y_coordinate));
    int_x_coordinate1 = int_x_coordinate + 1;
    int_y_coordinate1 = int_y_coordinate + 1;
    int_x_coordinate1 = std::min(int_x_coordinate1, logical_x_dimension - 1);
    int_y_coordinate1 = std::min(int_y_coordinate1, logical_y_dimension - 1);

    for (j = int_y_coordinate; j <= int_y_coordinate1; j++) {
        weight_y = (1.0 - fabsf(wanted_physical_y_coordinate - j));
        int_y    = (logical_x_dimension + padding_jump_value) * j;
        for (i = int_x_coordinate; i <= int_x_coordinate1; i++) {
            weight_x = (1.0 - fabsf(wanted_physical_x_coordinate - i));
            sum += real_values[int_y + i] * weight_x * weight_y;
        }
    }

    return sum;
}
// END_FOR_STAND_ALONE_CTFFIND

float Image::ReturnNearest2D(float& wanted_physical_x_coordinate, float& wanted_physical_y_coordinate) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Is in Fourier space");

    if (wanted_physical_x_coordinate < 0 || wanted_physical_x_coordinate > logical_x_dimension - 1)
        return 0.0;
    if (wanted_physical_y_coordinate < 0 || wanted_physical_y_coordinate > logical_y_dimension - 1)
        return 0.0;

    int i_nearest;
    int j_nearest;

    //	float weight_x;
    //	float weight_y;

    i_nearest = int(wanted_physical_x_coordinate + 0.5);
    j_nearest = int(wanted_physical_y_coordinate + 0.5);

    //	weight_x = (1.0 - fabsf(wanted_physical_x_coordinate - i_nearest));
    //	weight_y = (1.0 - fabsf(wanted_physical_y_coordinate - j_nearest));

    //	return real_values[(logical_x_dimension + padding_jump_value) * j_nearest + i_nearest] * weight_x * weight_y;
    return real_values[(logical_x_dimension + padding_jump_value) * j_nearest + i_nearest];
}

// BEGIN_FOR_STAND_ALONE_CTFFIND
void Image::CorrectMagnificationDistortion(float distortion_angle, float distortion_major_axis,
                                           float distortion_minor_axis) {
    MyDebugAssertTrue(logical_z_dimension == 1, "Only 2D Images supported");

    long  pixel_counter    = 0;
    float angle_in_radians = deg_2_rad(distortion_angle);

    float x_scale_factor = 1.0 / distortion_major_axis;
    float y_scale_factor = 1.0 / distortion_minor_axis;

    float average_edge_value = ReturnAverageOfRealValuesOnEdges();

    float new_x;
    float new_y;

    float final_x;
    float final_y;

    int x, y;

    Image buffer_image;
    buffer_image.Allocate(logical_x_dimension, logical_y_dimension, logical_z_dimension, is_in_real_space);
    // buffer_image.CopyFrom(this);

    for (y = 0; y < logical_y_dimension; y++) {
        for (x = 0; x < logical_x_dimension; x++) {
            // first rotation

            new_x = float(y - physical_address_of_box_center_y) * sinf(-angle_in_radians) +
                    float(x - physical_address_of_box_center_x) * cosf(-angle_in_radians);
            new_y = float(y - physical_address_of_box_center_y) * cosf(-angle_in_radians) -
                    float(x - physical_address_of_box_center_x) * sinf(-angle_in_radians);

            // scale factor

            new_x *= x_scale_factor;
            new_y *= y_scale_factor;

            new_x += physical_address_of_box_center_x;
            new_y += physical_address_of_box_center_y;

            // rotate back

            final_x = float(new_y - physical_address_of_box_center_y) * sinf(angle_in_radians) +
                      float(new_x - physical_address_of_box_center_x) * cosf(angle_in_radians);
            final_y = float(new_y - physical_address_of_box_center_y) * cosf(angle_in_radians) -
                      float(new_x - physical_address_of_box_center_x) * sinf(angle_in_radians);

            final_x += physical_address_of_box_center_x;
            final_y += physical_address_of_box_center_y;

            if (final_x < 0 || final_x > logical_x_dimension - 1 || final_y < 0 || final_y > logical_y_dimension - 1)
                real_values[pixel_counter] = average_edge_value;
            else {
                buffer_image.real_values[pixel_counter] = ReturnLinearInterpolated2D(final_x, final_y);
            }

            pixel_counter++;
        }

        pixel_counter += padding_jump_value;
    }

    Consume(&buffer_image);
}
// END_FOR_STAND_ALONE_CTFFIND

float Image::ApplyMask(Image& mask_volume, float cosine_edge_width, float weight_outside_mask,
                       float low_pass_filter_radius, float filter_cosine_edge_width, float outside_mask_value,
                       bool use_outside_mask_value) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Is in Fourier space");
    MyDebugAssertTrue(mask_volume.is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(mask_volume.is_in_real_space, "Is in Fourier space");
    MyDebugAssertTrue(HasSameDimensionsAs(&mask_volume), "mask_volume has different dimensions");
    MyDebugAssertTrue(cosine_edge_width < physical_address_of_box_center_x, "Edge too wide");

    int    i, j, k;
    int    int_edge = ceil(cosine_edge_width);
    int    int_edge_z;
    long   pixel_counter;
    long   edge_sum = 0;
    float  dx, dy, dz;
    float  radius_squared;
    float  edge_squared = powf(cosine_edge_width, 2);
    float  edge;
    float  tiny = 1.0 / 1000.0;
    double edge_value;
    double cos_volume;
    double sum = 0.0;

    Image* cosine_edge = new Image;
    cosine_edge->Allocate(logical_x_dimension, logical_y_dimension, logical_z_dimension, true);
    Image* mask_cosine_edge = new Image;
    mask_cosine_edge->Allocate(logical_x_dimension, logical_y_dimension, logical_z_dimension, true);
    Image* mask_cosine_double_edge = new Image;
    mask_cosine_double_edge->Allocate(logical_x_dimension, logical_y_dimension, logical_z_dimension, true);

    if (logical_z_dimension == 1)
        int_edge_z = 1;
    else
        int_edge_z = int_edge;

    // Binarize input
    for (pixel_counter = 0; pixel_counter < mask_volume.real_memory_allocated; pixel_counter++) {
        if (mask_volume.real_values[pixel_counter] > 0.0)
            mask_cosine_edge->real_values[pixel_counter] = 1.0;
        else
            mask_cosine_edge->real_values[pixel_counter] = 0.0;
    }

    if (cosine_edge_width > 0.0) {
        // Create cosine kernel
        cosine_edge->SetToConstant(0.0);
        cosine_edge->real_values[0] = 1.0;
        cos_volume                  = 1.0;
        for (k = 0; k < int_edge_z; k++) {
            dz = powf(k, 2);
            for (j = 0; j < int_edge; j++) {
                dy = powf(j, 2);
                for (i = 0; i < int_edge; i++) {
                    if (i + j + k > 0) {
                        dx             = powf(i, 2);
                        radius_squared = dx + dy + dz;
                        if (radius_squared <= edge_squared) {
                            edge          = (1.0 + cosf(PI * sqrtf(radius_squared) / cosine_edge_width)) / 2.0;
                            pixel_counter = ReturnReal1DAddressFromPhysicalCoord(i, j, k);
                            cosine_edge->real_values[pixel_counter] = edge;
                            cos_volume += edge;
                            if (i > 0) {
                                pixel_counter = ReturnReal1DAddressFromPhysicalCoord(logical_x_dimension - i, j, k);
                                cosine_edge->real_values[pixel_counter] = edge;
                                cos_volume += edge;
                            }
                            if (j > 0) {
                                pixel_counter = ReturnReal1DAddressFromPhysicalCoord(i, logical_y_dimension - j, k);
                                cosine_edge->real_values[pixel_counter] = edge;
                                cos_volume += edge;
                            }
                            if (k > 0) {
                                pixel_counter = ReturnReal1DAddressFromPhysicalCoord(i, j, logical_z_dimension - k);
                                cosine_edge->real_values[pixel_counter] = edge;
                                cos_volume += edge;
                            }
                            if (i > 0 && j > 0) {
                                pixel_counter = ReturnReal1DAddressFromPhysicalCoord(logical_x_dimension - i,
                                                                                     logical_y_dimension - j, k);
                                cosine_edge->real_values[pixel_counter] = edge;
                                cos_volume += edge;
                            }
                            if (i > 0 && k > 0) {
                                pixel_counter = ReturnReal1DAddressFromPhysicalCoord(logical_x_dimension - i, j,
                                                                                     logical_z_dimension - k);
                                cosine_edge->real_values[pixel_counter] = edge;
                                cos_volume += edge;
                            }
                            if (i > 0 && j > 0 && k > 0) {
                                pixel_counter = ReturnReal1DAddressFromPhysicalCoord(
                                    logical_x_dimension - i, logical_y_dimension - j, logical_z_dimension - k);
                                cosine_edge->real_values[pixel_counter] = edge;
                                cos_volume += edge;
                            }
                            if (j > 0 && k > 0) {
                                pixel_counter = ReturnReal1DAddressFromPhysicalCoord(i, logical_y_dimension - j,
                                                                                     logical_z_dimension - k);
                                cosine_edge->real_values[pixel_counter] = edge;
                                cos_volume += edge;
                            }
                        }
                    }
                }
            }
        }

        cosine_edge->ForwardFFT(false);
        mask_cosine_edge->ForwardFFT();
        mask_cosine_edge->MultiplyPixelWise(*cosine_edge);
        mask_cosine_edge->BackwardFFT();
        mask_cosine_edge->MultiplyByConstant(1.0 / cos_volume);
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
            if (fabsf(mask_cosine_edge->real_values[pixel_counter]) < tiny)
                mask_cosine_edge->real_values[pixel_counter] = 0.0;
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
            if (mask_cosine_edge->real_values[pixel_counter] > 0.1f)
                mask_cosine_double_edge->real_values[pixel_counter] = 1.0f;
            else
                mask_cosine_double_edge->real_values[pixel_counter] = 0.0f;
        }
        mask_cosine_double_edge->ForwardFFT();
        mask_cosine_double_edge->MultiplyPixelWise(*cosine_edge);
        mask_cosine_double_edge->BackwardFFT();
        mask_cosine_double_edge->MultiplyByConstant(1.0 / cos_volume);
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
            if (fabsf(mask_cosine_double_edge->real_values[pixel_counter]) < tiny)
                mask_cosine_double_edge->real_values[pixel_counter] = 0.0;
    }

    if (use_outside_mask_value) {
        edge_value = outside_mask_value;
    }
    else {
        //		edge_value = 0.0;
        //		pixel_counter = 0;
        //		for (k = 0; k < logical_z_dimension; k++) {
        //			for (j = 0; j < logical_y_dimension; j++) {
        //				for (i = 0; i < logical_x_dimension; i++) {
        //					if (mask_cosine_edge->real_values[pixel_counter] > tiny &&
        //mask_cosine_double_edge->real_values[pixel_counter] < 0.2f) {edge_value += real_values[pixel_counter];
        //edge_sum++;} 					pixel_counter++;
        //				}
        //				pixel_counter += padding_jump_value;
        //			}
        //		}
        //		if (edge_sum != 0) edge_value /= edge_sum;
        edge_value = ReturnAverageOfRealValues(0.4f * logical_x_dimension, true);
    }

    //	mask_cosine_edge->QuickAndDirtyWriteSlices("mask_cosine_edge.mrc", 1, logical_z_dimension);
    //	mask_cosine_double_edge->QuickAndDirtyWriteSlices("mask_cosine_double_edge.mrc", 1, logical_z_dimension);
    if (low_pass_filter_radius > 0.0 && weight_outside_mask > 0.0 && cosine_edge_width > 0.0) {
        cosine_edge->CopyFrom(this);
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
            cosine_edge->real_values[pixel_counter] =
                mask_cosine_double_edge->real_values[pixel_counter] * edge_value +
                (1.0 - mask_cosine_double_edge->real_values[pixel_counter]) * cosine_edge->real_values[pixel_counter];
        }
        //		cosine_edge->QuickAndDirtyWriteSlices("inside_mask_removed.mrc", 1, cosine_edge->logical_z_dimension);
        cosine_edge->ForwardFFT();
        cosine_edge->CosineMask(low_pass_filter_radius, filter_cosine_edge_width);
        cosine_edge->BackwardFFT();
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
            real_values[pixel_counter] = mask_cosine_edge->real_values[pixel_counter] * real_values[pixel_counter] +
                                         weight_outside_mask * (1.0 - mask_cosine_edge->real_values[pixel_counter]) *
                                             cosine_edge->real_values[pixel_counter];
            sum += mask_cosine_edge->real_values[pixel_counter];
        }
    }
    else {
        for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++) {
            real_values[pixel_counter] = (1.0 - mask_cosine_edge->real_values[pixel_counter]) * edge_value +
                                         mask_cosine_edge->real_values[pixel_counter] * real_values[pixel_counter];
            sum += mask_cosine_edge->real_values[pixel_counter];
        }
    }

    delete cosine_edge;
    delete mask_cosine_edge;
    delete mask_cosine_double_edge;

    return float(sum);
}

Peak Image::CenterOfMass(float threshold, bool apply_threshold) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");

    int    i, j, k;
    long   pixel_counter = 0;
    float  temp_float;
    double sum_xd = 0.0;
    double sum_yd = 0.0;
    double sum_zd = 0.0;
    double sum_d  = 0.0;
    Peak   center_of_mass;

    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                temp_float = real_values[pixel_counter];
                if (apply_threshold) {
                    temp_float = std::max(real_values[pixel_counter] - threshold, 0.0f);
                }
                //				temp_float = (i - physical_address_of_box_center_x);
                //				if (fabsf(temp_float) > 0.0) sum_xd += temp_float / powf(fabsf(temp_float), 0.66) *
                //real_values[pixel_counter]; 				temp_float = (j - physical_address_of_box_center_y); 				if (fabsf(temp_float)
                //> 0.0) sum_yd += temp_float / powf(fabsf(temp_float), 0.66) * real_values[pixel_counter]; 				temp_float =
                //(k - physical_address_of_box_center_z); 				if (fabsf(temp_float) > 0.0) sum_zd += temp_float /
                //powf(fabsf(temp_float), 0.66) * real_values[pixel_counter];
                sum_xd += (i - physical_address_of_box_center_x) * temp_float;
                sum_yd += (j - physical_address_of_box_center_y) * temp_float;
                sum_zd += (k - physical_address_of_box_center_z) * temp_float;
                sum_d += temp_float;
                //				wxPrintf("%g %g %g %g\n", temp_float, powf(fabsf(temp_float), 0.66),
                //real_values[pixel_counter], sum_xd);
                pixel_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }

    if (sum_d != 0.0) {
        center_of_mass.x     = sum_xd / sum_d;
        center_of_mass.y     = sum_yd / sum_d;
        center_of_mass.z     = sum_zd / sum_d;
        center_of_mass.value = 1.0;
    }
    else {
        center_of_mass.x     = 0.0;
        center_of_mass.y     = 0.0;
        center_of_mass.z     = 0.0;
        center_of_mass.value = 0.0;
    }

    return center_of_mass;
}

Peak Image::StandardDeviationOfMass(float threshold, bool apply_threshold, bool invert_densities) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");

    int    i, j, k;
    long   pixel_counter = 0;
    float  temp_float;
    double sum_xd = 0.0;
    double sum_yd = 0.0;
    double sum_zd = 0.0;
    double sum_d  = 0.0;
    Peak   standard_deviations;

    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                if (invert_densities) {
                    if (apply_threshold) {
                        temp_float = std::max(threshold - real_values[pixel_counter], 0.0f);
                    }
                    else
                        temp_float = -real_values[pixel_counter];
                }
                else {
                    if (apply_threshold) {
                        temp_float = std::max(real_values[pixel_counter] - threshold, 0.0f);
                    }
                    else
                        temp_float = real_values[pixel_counter];
                }

                sum_xd += powf(i - physical_address_of_box_center_x, 2) * temp_float;
                sum_yd += powf(j - physical_address_of_box_center_y, 2) * temp_float;
                sum_zd += powf(k - physical_address_of_box_center_z, 2) * temp_float;
                sum_d += temp_float;
                pixel_counter++;
            }
            pixel_counter += padding_jump_value;
        }
    }

    if (sum_d != 0.0) {
        standard_deviations.x     = sqrtf(sum_xd / sum_d);
        standard_deviations.y     = sqrtf(sum_yd / sum_d);
        standard_deviations.z     = sqrtf(sum_zd / sum_d);
        standard_deviations.value = 1.0;
    }
    else {
        standard_deviations.x     = 0.0;
        standard_deviations.y     = 0.0;
        standard_deviations.z     = 0.0;
        standard_deviations.value = 0.0;
    }

    return standard_deviations;
}

float Image::ReturnAverageOfMinN(int number_of_pixels_to_average, float wanted_mask_radius) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");

    int   i, j, k;
    long  pixel_counter1 = 0;
    long  pixel_counter2 = 0;
    int   mask_radius;
    float average_density_min = 0.0;

    if (wanted_mask_radius == 0.0)
        mask_radius = int(ReturnSmallestLogicalDimension() / 2.0);
    else
        mask_radius = std::min(int(ReturnSmallestLogicalDimension() / 2.0), int(wanted_mask_radius));

    float* temp_3d = new float[logical_x_dimension * logical_y_dimension + logical_x_dimension * logical_z_dimension +
                               logical_y_dimension * logical_z_dimension];
    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                if (k >= physical_address_of_box_center_z - mask_radius &&
                    k < physical_address_of_box_center_z + mask_radius &&
                    j >= physical_address_of_box_center_y - mask_radius &&
                    j < physical_address_of_box_center_y + mask_radius &&
                    i >= physical_address_of_box_center_x - mask_radius &&
                    i < physical_address_of_box_center_x + mask_radius) {
                    if (i == physical_address_of_box_center_x || j == physical_address_of_box_center_y ||
                        k == physical_address_of_box_center_z) {
                        temp_3d[pixel_counter2] = real_values[pixel_counter1];
                        pixel_counter2++;
                    }
                }
                pixel_counter1++;
            }
            pixel_counter1 += padding_jump_value;
        }
    }
    std::sort(temp_3d, temp_3d + pixel_counter2);
    if (number_of_pixels_to_average > pixel_counter2)
        number_of_pixels_to_average = pixel_counter2;

    for (i = 0; i < number_of_pixels_to_average; i++)
        average_density_min += temp_3d[i];
    average_density_min /= float(number_of_pixels_to_average);
    delete[] temp_3d;

    return average_density_min;
}

float Image::ReturnAverageOfMaxN(int number_of_pixels_to_average, float wanted_mask_radius) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");

    int   i, j, k;
    long  pixel_counter1 = 0;
    long  pixel_counter2 = 0;
    int   mask_radius;
    float average_density_max = 0.0;

    if (wanted_mask_radius == 0.0)
        mask_radius = int(ReturnSmallestLogicalDimension() / 2.0);
    else
        mask_radius = std::min(int(ReturnSmallestLogicalDimension() / 2.0), int(wanted_mask_radius));

    float* temp_3d = new float[logical_x_dimension * logical_y_dimension + logical_x_dimension * logical_z_dimension +
                               logical_y_dimension * logical_z_dimension];
    for (k = 0; k < logical_z_dimension; k++) {
        for (j = 0; j < logical_y_dimension; j++) {
            for (i = 0; i < logical_x_dimension; i++) {
                if (k >= physical_address_of_box_center_z - mask_radius &&
                    k < physical_address_of_box_center_z + mask_radius &&
                    j >= physical_address_of_box_center_y - mask_radius &&
                    j < physical_address_of_box_center_y + mask_radius &&
                    i >= physical_address_of_box_center_x - mask_radius &&
                    i < physical_address_of_box_center_x + mask_radius) {
                    if (i == physical_address_of_box_center_x || j == physical_address_of_box_center_y ||
                        k == physical_address_of_box_center_z) {
                        temp_3d[pixel_counter2] = real_values[pixel_counter1];
                        pixel_counter2++;
                    }
                }
                pixel_counter1++;
            }
            pixel_counter1 += padding_jump_value;
        }
    }
    std::sort(temp_3d, temp_3d + pixel_counter2);
    pixel_counter1 = pixel_counter2 - number_of_pixels_to_average;
    if (pixel_counter1 < 0)
        pixel_counter1 = 0;
    for (i = pixel_counter1; i < pixel_counter2; i++)
        average_density_max += temp_3d[i];
    average_density_max /= (pixel_counter2 - pixel_counter1);
    delete[] temp_3d;

    return average_density_max;
}

void Image::AddSlices(Image& input_image, int first_slice, int last_slice, bool calculate_average) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    //	MyDebugAssertTrue(is_in_real_space, "Is in Fourier space");
    MyDebugAssertTrue(input_image.is_in_memory, "input_image memory not allocated");
    MyDebugAssertTrue(input_image.is_in_real_space, "input_image is in Fourier space");
    MyDebugAssertTrue(logical_x_dimension == input_image.logical_x_dimension &&
                          logical_y_dimension == input_image.logical_y_dimension,
                      "input_image has different x,y dimensions");
    MyDebugAssertTrue(first_slice >= 0, "Invalid first_slice");
    MyDebugAssertTrue(last_slice <= input_image.logical_z_dimension, "Invalid last_slice");
    MyDebugAssertTrue(last_slice >= first_slice, "last_slice not larger or equal first_slice");

    int   slice;
    int   pixel_counter_2d;
    long  pixel_counter_3d;
    float number_of_slices;

    is_in_real_space = true;
    if (first_slice == 0 && last_slice == 0) {
        first_slice = 1;
        last_slice  = input_image.logical_z_dimension;
    }
    pixel_counter_3d = real_memory_allocated * (first_slice - 1);

    SetToConstant(0.0);

    for (slice = first_slice - 1; slice < last_slice; slice++) {
        for (pixel_counter_2d = 0; pixel_counter_2d < real_memory_allocated; pixel_counter_2d++) {
            real_values[pixel_counter_2d] += input_image.real_values[pixel_counter_3d];
            pixel_counter_3d++;
        }
    }
    if (calculate_average && last_slice != first_slice) {
        number_of_slices = last_slice - first_slice + 1.0;
        for (pixel_counter_2d = 0; pixel_counter_2d < real_memory_allocated; pixel_counter_2d++) {
            real_values[pixel_counter_2d] /= number_of_slices;
        }
    }
}

void Image::CreateOrthogonalProjectionsImage(Image* image_to_create, bool include_projections, float scale_factor,
                                             float mask_radius_in_pixels) {
    MyDebugAssertTrue(this->IsCubic() == true, "Only Cubic Volumes Supported");
    // don't allocate so i can use Allocateaspointing to slice in 3d.

#ifdef DEBUG
    if (include_projections == true) {
        MyDebugAssertTrue(
            image_to_create->logical_x_dimension == myroundint(float(logical_x_dimension) * scale_factor) * 3.0 &&
                image_to_create->logical_y_dimension == myroundint(float(logical_y_dimension) * scale_factor) * 2 &&
                image_to_create->is_in_real_space == true,
            "Output image not setup correctly");
    }
    else
        MyDebugAssertTrue(
            image_to_create->logical_x_dimension == myroundint(float(logical_x_dimension) * scale_factor) * 3.0 &&
                image_to_create->logical_y_dimension == myroundint(float(logical_y_dimension) * scale_factor) &&
                image_to_create->is_in_real_space == true,
            "Output image not setup correctly");
#endif

    int i, j, k;

    Image slice_one;
    Image slice_two;
    Image slice_three;

    Image proj_one;
    Image proj_two;
    Image proj_three;

    slice_one.Allocate(this->logical_x_dimension, this->logical_y_dimension, true);
    slice_two.Allocate(this->logical_x_dimension, this->logical_y_dimension, true);
    slice_three.Allocate(this->logical_x_dimension, this->logical_y_dimension, true);

    slice_one.SetToConstant(0.0);
    slice_two.SetToConstant(0.0);
    slice_three.SetToConstant(0.0);

    if (include_projections == true) {
        proj_one.Allocate(this->logical_x_dimension, this->logical_y_dimension, true);
        proj_two.Allocate(this->logical_x_dimension, this->logical_y_dimension, true);
        proj_three.Allocate(this->logical_x_dimension, this->logical_y_dimension, true);

        proj_one.SetToConstant(0.0);
        proj_two.SetToConstant(0.0);
        proj_three.SetToConstant(0.0);
    }

    long input_counter = 0;
    long output_counter;

    if (include_projections == true) {

        for (k = 0; k < this->logical_z_dimension; k++) {
            for (j = 0; j < this->logical_y_dimension; j++) {
                for (i = 0; i < this->logical_x_dimension; i++) {
                    proj_one.real_values[proj_one.ReturnReal1DAddressFromPhysicalCoord(i, j, 0)] +=
                        this->real_values[input_counter];
                    proj_two.real_values[proj_two.ReturnReal1DAddressFromPhysicalCoord(j, k, 0)] +=
                        this->real_values[input_counter];
                    proj_three.real_values[proj_three.ReturnReal1DAddressFromPhysicalCoord(i, k, 0)] +=
                        this->real_values[input_counter];

                    input_counter++;
                }

                input_counter += this->padding_jump_value;
            }
        }
    }

    output_counter = 0;

    for (j = 0; j < slice_one.logical_y_dimension; j++) {
        for (i = 0; i < slice_one.logical_x_dimension; i++) {

            slice_one.real_values[output_counter] =
                this->ReturnRealPixelFromPhysicalCoord(i, j, this->physical_address_of_box_center_z);
            slice_two.real_values[output_counter] =
                this->ReturnRealPixelFromPhysicalCoord(this->physical_address_of_box_center_x, i, j);
            slice_three.real_values[output_counter] =
                this->ReturnRealPixelFromPhysicalCoord(i, this->physical_address_of_box_center_x, j);

            output_counter++;
        }

        output_counter += slice_one.padding_jump_value;
    }

    float min_value = FLT_MAX;
    float max_value = -FLT_MAX;

    float current_min_value;
    float current_max_value;

    if (scale_factor != 1.0) {
        slice_one.ForwardFFT();
        slice_one.Resize(myroundint(this->logical_x_dimension * scale_factor),
                         myroundint(this->logical_y_dimension * scale_factor), 1);
        slice_one.BackwardFFT();
        slice_one.Normalize(1.0);

        slice_two.ForwardFFT();
        slice_two.Resize(myroundint(this->logical_x_dimension * scale_factor),
                         myroundint(this->logical_y_dimension * scale_factor), 1);
        slice_two.BackwardFFT();
        slice_two.Normalize(1.0);

        slice_three.ForwardFFT();
        slice_three.Resize(myroundint(this->logical_x_dimension * scale_factor),
                           myroundint(this->logical_y_dimension * scale_factor), 1);
        slice_three.BackwardFFT();
        slice_three.Normalize(1.0);

        if (include_projections == true) {
            proj_one.ForwardFFT();
            proj_one.Resize(myroundint(this->logical_x_dimension * scale_factor),
                            myroundint(this->logical_y_dimension * scale_factor), 1);
            proj_one.BackwardFFT();
            proj_one.Normalize(1.0);

            proj_two.ForwardFFT();
            proj_two.Resize(myroundint(this->logical_x_dimension * scale_factor),
                            myroundint(this->logical_y_dimension * scale_factor), 1);
            proj_two.BackwardFFT();
            proj_two.Normalize(1.0);

            proj_three.ForwardFFT();
            proj_three.Resize(myroundint(this->logical_x_dimension * scale_factor),
                              myroundint(this->logical_y_dimension * scale_factor), 1);
            proj_three.BackwardFFT();
            proj_three.Normalize(1.0);
        }
    }
    slice_one.GetMinMax(min_value, max_value);

    slice_two.GetMinMax(current_min_value, current_max_value);
    min_value = std::min(min_value, current_min_value);
    max_value = std::max(max_value, current_max_value);

    slice_three.GetMinMax(current_min_value, current_max_value);
    min_value = std::min(min_value, current_min_value);
    max_value = std::max(max_value, current_max_value);

    slice_one.AddConstant(-min_value);
    slice_one.DivideByConstant(max_value - min_value);

    slice_two.AddConstant(-min_value);
    slice_two.DivideByConstant(max_value - min_value);

    slice_three.AddConstant(-min_value);
    slice_three.DivideByConstant(max_value - min_value);

    if (include_projections == true) {

        proj_one.GetMinMax(min_value, max_value);

        proj_two.GetMinMax(current_min_value, current_max_value);
        min_value = std::min(min_value, current_min_value);
        max_value = std::max(max_value, current_max_value);

        proj_three.GetMinMax(current_min_value, current_max_value);
        min_value = std::min(min_value, current_min_value);
        max_value = std::max(max_value, current_max_value);

        proj_one.AddConstant(-min_value);
        proj_one.DivideByConstant(max_value - min_value);

        proj_two.AddConstant(-min_value);
        proj_two.DivideByConstant(max_value - min_value);

        proj_three.AddConstant(-min_value);
        proj_three.DivideByConstant(max_value - min_value);
    }

    output_counter = 0;

    if (mask_radius_in_pixels != 0.0f) {
        slice_one.CircleMaskWithValue(mask_radius_in_pixels * scale_factor, slice_one.ReturnAverageOfRealValuesAtRadius(
                                                                                mask_radius_in_pixels * scale_factor));
        slice_two.CircleMaskWithValue(mask_radius_in_pixels * scale_factor, slice_two.ReturnAverageOfRealValuesAtRadius(
                                                                                mask_radius_in_pixels * scale_factor));
        slice_three.CircleMaskWithValue(
            mask_radius_in_pixels * scale_factor,
            slice_three.ReturnAverageOfRealValuesAtRadius(mask_radius_in_pixels * scale_factor));

        if (include_projections == true) {
            proj_one.CircleMaskWithValue(
                mask_radius_in_pixels * scale_factor,
                proj_one.ReturnAverageOfRealValuesAtRadius(mask_radius_in_pixels * scale_factor));
            proj_two.CircleMaskWithValue(
                mask_radius_in_pixels * scale_factor,
                proj_two.ReturnAverageOfRealValuesAtRadius(mask_radius_in_pixels * scale_factor));
            proj_three.CircleMaskWithValue(
                mask_radius_in_pixels * scale_factor,
                proj_three.ReturnAverageOfRealValuesAtRadius(mask_radius_in_pixels * scale_factor));
        }
    }

    for (j = 0; j < image_to_create->logical_y_dimension; j++) {
        for (i = 0; i < image_to_create->logical_x_dimension; i++) {
            if (j < slice_one.logical_y_dimension && include_projections == true) {
                if (i < proj_one.logical_x_dimension) {
                    image_to_create->real_values[output_counter] = proj_one.ReturnRealPixelFromPhysicalCoord(i, j, 0);
                }
                else if (i < proj_one.logical_x_dimension * 2) {
                    image_to_create->real_values[output_counter] =
                        proj_two.ReturnRealPixelFromPhysicalCoord(i - proj_two.logical_x_dimension, j, 0);
                }
                else {
                    image_to_create->real_values[output_counter] =
                        proj_three.ReturnRealPixelFromPhysicalCoord(i - proj_three.logical_x_dimension * 2, j, 0);
                }
            }
            else {
                if (i < slice_one.logical_x_dimension) {
                    if (include_projections == true)
                        image_to_create->real_values[output_counter] =
                            slice_one.ReturnRealPixelFromPhysicalCoord(i, j - slice_one.logical_y_dimension, 0);
                    else
                        image_to_create->real_values[output_counter] =
                            slice_one.ReturnRealPixelFromPhysicalCoord(i, j, 0);
                }
                else if (i < slice_one.logical_x_dimension * 2) {
                    if (include_projections == true)
                        image_to_create->real_values[output_counter] = slice_two.ReturnRealPixelFromPhysicalCoord(
                            i - slice_two.logical_x_dimension, j - slice_two.logical_y_dimension, 0);
                    else
                        image_to_create->real_values[output_counter] =
                            slice_two.ReturnRealPixelFromPhysicalCoord(i - slice_two.logical_x_dimension, j, 0);
                }
                else {
                    if (include_projections == true)
                        image_to_create->real_values[output_counter] = slice_three.ReturnRealPixelFromPhysicalCoord(
                            i - slice_three.logical_x_dimension * 2, j - slice_three.logical_y_dimension, 0);
                    else
                        image_to_create->real_values[output_counter] =
                            slice_three.ReturnRealPixelFromPhysicalCoord(i - slice_three.logical_x_dimension * 2, j, 0);
                }
            }
            output_counter++;
        }

        output_counter += image_to_create->padding_jump_value;
    }
}

float Image::ReturnCorrelationBetweenTwoHorizontalLines(int first_line, int second_line) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "input_image is in Fourier space");
    MyDebugAssertTrue(first_line >= 0, "Invalid first_slice");
    MyDebugAssertTrue(second_line < logical_x_dimension, "Invalid last_slice");

    float correlation = 0.0f;
    float ab_buffer   = 0.0f;
    float aa_buffer   = 0.0f;
    float bb_buffer   = 0.0f;

    float first_line_average  = 0.0f;
    float second_line_average = 0.0f;

    int counter;

    for (counter = 0; counter < logical_x_dimension; counter++) {
        first_line_average += real_values[ReturnReal1DAddressFromPhysicalCoord(counter, first_line, 0)];
        second_line_average += real_values[ReturnReal1DAddressFromPhysicalCoord(counter, second_line, 0)];
    }

    first_line_average /= logical_x_dimension;
    second_line_average /= logical_x_dimension;

    for (counter = 0; counter < logical_x_dimension; counter++) {
        ab_buffer += (real_values[ReturnReal1DAddressFromPhysicalCoord(counter, first_line, 0)] - first_line_average) *
                     (real_values[ReturnReal1DAddressFromPhysicalCoord(counter, second_line, 0)] - second_line_average);
        aa_buffer +=
            powf(real_values[ReturnReal1DAddressFromPhysicalCoord(counter, first_line, 0)] - first_line_average, 2);
        bb_buffer +=
            powf(real_values[ReturnReal1DAddressFromPhysicalCoord(counter, second_line, 0)] - second_line_average, 2);
    }

    aa_buffer /= logical_x_dimension - 1;
    bb_buffer /= logical_x_dimension - 1;

    correlation = ab_buffer / sqrtf(aa_buffer * bb_buffer);
    return correlation / logical_x_dimension;
}

float Image::ReturnCorrelationBetweenTwoVerticalLines(int first_line, int second_line) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "input_image is in Fourier space");
    MyDebugAssertTrue(first_line >= 0, "Invalid first_slice");
    MyDebugAssertTrue(second_line < logical_y_dimension, "Invalid last_slice");

    float correlation = 0.0f;
    float ab_buffer   = 0.0f;
    float aa_buffer   = 0.0f;
    float bb_buffer   = 0.0f;

    float first_line_average  = 0.0f;
    float second_line_average = 0.0f;

    int counter;

    for (counter = 0; counter < logical_y_dimension; counter++) {
        first_line_average += real_values[ReturnReal1DAddressFromPhysicalCoord(first_line, counter, 0)];
        second_line_average += real_values[ReturnReal1DAddressFromPhysicalCoord(second_line, counter, 0)];
    }

    first_line_average /= logical_y_dimension;
    second_line_average /= logical_y_dimension;

    for (counter = 0; counter < logical_y_dimension; counter++) {
        ab_buffer += (real_values[ReturnReal1DAddressFromPhysicalCoord(first_line, counter, 0)] - first_line_average) *
                     (real_values[ReturnReal1DAddressFromPhysicalCoord(second_line, counter, 0)] - second_line_average);
        aa_buffer +=
            powf(real_values[ReturnReal1DAddressFromPhysicalCoord(first_line, counter, 0)] - first_line_average, 2);
        bb_buffer +=
            powf(real_values[ReturnReal1DAddressFromPhysicalCoord(second_line, counter, 0)] - second_line_average, 2);
    }

    aa_buffer /= logical_y_dimension - 1;
    bb_buffer /= logical_y_dimension - 1;

    correlation = ab_buffer / sqrtf(aa_buffer * bb_buffer);
    return correlation / logical_y_dimension;
}

BeamTiltScorer::BeamTiltScorer(CTF* pointer_to_wanted_ctf, Image* pointer_to_wanted_binarized_phase_diff_spectrum,
                               float wanted_pixel_size, float wanted_mask_radius, float wanted_phase_multiplier) {
    pointer_to_ctf_to_use_for_calculation       = pointer_to_wanted_ctf;
    pointer_binarised_phase_difference_spectrum = pointer_to_wanted_binarized_phase_diff_spectrum;
    pixel_size                                  = wanted_pixel_size;
    mask_radius                                 = wanted_mask_radius;
    phase_multiplier                            = wanted_phase_multiplier;

    temp_image.Allocate(pointer_to_wanted_binarized_phase_diff_spectrum->logical_x_dimension,
                        pointer_to_wanted_binarized_phase_diff_spectrum->logical_y_dimension, true);
    beamtilt_spectrum.Allocate(pointer_to_wanted_binarized_phase_diff_spectrum->logical_x_dimension,
                               pointer_to_wanted_binarized_phase_diff_spectrum->logical_y_dimension, true);
}

double BeamTiltScorer::ScoreValues(double input_values[]) {
    // 0 = ignore, stupid code conversion
    // 1 = beamtilt_azimuth
    // 2 = shift_azimuth
    // 3 = beamtilt
    // 4 = shift

    //	// Test a limit on the particle shift
    //	float testParticleRadius = 190.0f; // Ang
    //	if (fabsf(testParticleRadius*input_values[3]) < fabsf(input_values[4]))
    //	{
    //		return std::numeric_limits<float>::max();
    //	}
    //	else
    //	{
    pointer_to_ctf_to_use_for_calculation->SetBeamTilt(
        input_values[3] * cosf(input_values[1]), input_values[3] * sinf(input_values[1]),
        input_values[4] * cosf(input_values[2]) / pixel_size, input_values[4] * sinf(input_values[2]) / pixel_size);
    temp_image.CalculateBeamTiltImage(*pointer_to_ctf_to_use_for_calculation);
    temp_image.ComputeAmplitudeSpectrumFull2D(&beamtilt_spectrum, true, phase_multiplier);
    // beamtilt_spectrum.Binarise(0.0f);

    beamtilt_spectrum.CosineRingMask(5.0f, beamtilt_spectrum.ReturnLargestLogicalDimension(), 2.0f);
    beamtilt_spectrum.SubtractImage(pointer_binarised_phase_difference_spectrum);

    return beamtilt_spectrum.ReturnSumOfSquares(mask_radius);
    //	}
}

double BeamTiltScoreFunctionForSimplex(void* pt2Object, double values[]) {
    BeamTiltScorer* scorer_to_use = reinterpret_cast<BeamTiltScorer*>(pt2Object);
    //	float score = scorer_to_use->ScoreValues(values);
    //	wxPrintf("%f, %f, %f, %f = %f\n", values[1], values[2], values[3], values[4], score);
    return scorer_to_use->ScoreValues(values);
}

float Image::ReturnBeamTiltSignificanceScore(Image calculated_beam_tilt) {
    Image buffer;
    buffer.CopyFrom(this);
    // If this is from the sanity check, we want to keep the input image unmodified, so pass by value.
    // work significance
    buffer.MultiplyByConstant(float(buffer.logical_x_dimension));
    buffer.Binarise(0.00002f);
    calculated_beam_tilt.Binarise(0.0f);

    float mask_radius_local =
        sqrtf((buffer.ReturnAverageOfRealValues() * buffer.logical_x_dimension * buffer.logical_y_dimension) / PIf);
    buffer.SubtractImage(&calculated_beam_tilt);
    float binarized_score = buffer.ReturnSumOfSquares(mask_radius_local);
    return 0.5f * PIf * powf((0.5f - binarized_score) * mask_radius_local, 2);
}

__global__ void ConvertToHalfPrecisionKernelComplex(cufftComplex* complex_32f_values, __half2* complex_16f_values, int4 dims, int3 physical_upper_bound_complex);
__global__ void ConvertToHalfPrecisionKernelReal(cufftReal* real_32f_values, __half* real_16f_values, int4 dims);


__global__ void MultiplyPixelWiseComplexConjugateKernel(cufftComplex* ref_complex_values, cufftComplex* img_complex_values, int4 dims, int3 physical_upper_bound_complex);
__global__ void MipPixelWiseKernel(cufftReal *mip, const cufftReal *correlation_output, const int4 dims);
__global__ void MipPixelWiseKernel(cufftReal* mip, cufftReal *other_image, cufftReal *psi, cufftReal *phi, cufftReal *theta,
                                   int4 dims,float c_psi, float c_phi, float c_theta);
__global__ void MipPixelWiseKernel(cufftReal* mip, cufftReal *other_image, cufftReal *psi, cufftReal *phi, cufftReal *theta, cufftReal *defocus, cufftReal *pixel, const int4 dims,
                                   float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel);


__global__ void ApplyBFactorKernel(cufftComplex* d_input, 
                                   const int4 dims, 
                                   const int3 physical_index_of_first_negative_frequency,
                                   const int3 physical_upper_bound_complex,
                                   float bfactor);

__global__ void ApplyBFactorKernel(cufftComplex* d_input, 
                                   const int4 dims, 
                                   const int3 physical_index_of_first_negative_frequency,
                                   const int3 physical_upper_bound_complex,
                                   float bfactor,
                                   const float vertical_mask_size,
                                   const float horizontal_mask_size); // Specialization for unblur refinement, merges MaskCentralCross()                
                                   
__global__ void PhaseShiftKernel(cufftComplex* d_input, 
                                 int4 dims, float3 shifts, 
                                 int3 physical_address_of_box_center, 
                                 int3 physical_index_of_first_negative_frequency,
                                 int3 physical_upper_bound_complex);

__global__ void ClipIntoRealKernel(cufftReal* real_values_gpu,
                                   cufftReal* other_image_real_values_gpu,
                                   int4 dims, 
                                   int4 other_dims,
                                   int3 physical_address_of_box_center, 
                                   int3 other_physical_address_of_box_center, 
                                   int3 wanted_coordinate_of_box_center, 
                                   float wanted_padding_value);

// cuFFT callbacks
__device__ cufftReal CB_ConvertInputf16Tof32(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

__device__ cufftReal CB_ConvertInputf16Tof32(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
{

	const __half element = ((__half*)dataIn)[offset];
	return (cufftReal)(__half2float(element));
}
__device__ cufftCallbackLoadR d_ConvertInputf16Tof32Ptr = CB_ConvertInputf16Tof32;

__device__ void CB_scaleFFTAndStore(void* dataOut, size_t offset, cufftComplex element, void* callerInfo, void* sharedPtr);

__device__ void CB_scaleFFTAndStore(void* dataOut, size_t offset, cufftComplex element, void* callerInfo, void* sharedPtr)
{
	float scale_factor = *(float *)callerInfo;

	((cufftComplex*)dataOut)[offset] = (cufftComplex)ComplexScale(element, scale_factor);
}
//__device__ cufftCallbackLoadR d_ConvertInputf16Tof32Ptr = CB_ConvertInputf16Tof32;
__device__ cufftCallbackStoreC d_scaleFFTAndStorePtr = CB_scaleFFTAndStore;


__device__ void CB_mipCCGAndStore(void* dataOut, size_t offset, cufftReal element, void* callerInfo, void* sharedPtr);


__device__ void CB_mipCCGAndStore(void* dataOut, size_t offset, cufftReal element, void* callerInfo, void* sharedPtr)
{

	__half* data_out_half = (__half*)callerInfo;
//	data_out_half[offset] = __float2half(element);
//	((cufftReal *)dataOut)[offset] = element;

#ifdef DISABLECACHEHINTS
	((__half *)data_out_half)[offset] = __float2half(element);
#else
	__stcs( &data_out_half[offset], __float2half(element) );
#endif

//	)_(dataOut)[offset] = __float2half(element);
//
}
__device__ cufftCallbackStoreR d_mipCCGAndStorePtr = CB_mipCCGAndStore;

template<typename T> struct CB_complexConjMulLoad_params
{
	T* 		target;
	float 	scale;

} ;

static __device__ cufftComplex CB_complexConjMulLoad_32f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);
static __device__ cufftComplex CB_complexConjMulLoad_16f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

static __device__ cufftComplex CB_complexConjMulLoad_32f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
 {
	CB_complexConjMulLoad_params<cufftComplex>* my_params = (CB_complexConjMulLoad_params<cufftComplex> *)callerInfo;
	return (cufftComplex)ComplexConjMulAndScale(my_params->target[offset],((Complex *)dataIn)[offset],my_params->scale);
//    return (cufftComplex)make_float2(my_params->scale * __fmaf_rn(my_params->target[offset].x ,  ((Complex *)dataIn)[offset].x, my_params->target[offset].y * ((Complex *)dataIn)[offset].y),
//    		my_params->scale * __fmaf_rn(my_params->target[offset].y , -((Complex *)dataIn)[offset].x, my_params->target[offset].x * ((Complex *)dataIn)[offset].y));


 }
static __device__ cufftComplex CB_complexConjMulLoad_16f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
 {
	CB_complexConjMulLoad_params<__half2>* my_params = (CB_complexConjMulLoad_params<__half2> *)callerInfo;
	return (cufftComplex)ComplexConjMulAndScale((cufftComplex)__half22float2(my_params->target[offset]),((Complex *)dataIn)[offset],my_params->scale);
 }

 __device__ cufftCallbackLoadC d_complexConjMulLoad_32f = CB_complexConjMulLoad_32f;
 __device__ cufftCallbackLoadC d_complexConjMulLoad_16f = CB_complexConjMulLoad_16f;

typedef struct _CB_realLoadAndClipInto_params
{
	int* 		mask;
	cufftReal*	target;

} CB_realLoadAndClipInto_params;


static __device__ cufftReal CB_realLoadAndClipInto(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

static __device__ cufftReal CB_realLoadAndClipInto(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
{

	 CB_realLoadAndClipInto_params* my_params = (CB_realLoadAndClipInto_params *)callerInfo;
	 int idx = my_params->mask[offset];
	 if (idx == 0)
	 {
		 return 0.0f;
	 }
	 else
	 {
		 return my_params->target[idx];
	 }



}

  __device__ cufftCallbackLoadR d_realLoadAndClipInto = CB_realLoadAndClipInto;



 // Inline declarations
  __device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_X(int physical_index, 
                                                int logical_x_dimension,
                                                int physical_address_of_box_center_x);

 __device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(int physical_index,
                                                int logical_y_dimension,
                                                int physical_index_of_first_negative_frequency_y);

 __device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(int physical_index,
                                                int logical_z_dimension,
                                                int physical_index_of_first_negative_frequency_x);

 __device__ __forceinline__ float
d_ReturnPhaseFromShift(float real_space_shift, float distance_from_origin, float dimension_size);

 __device__ __forceinline__ void
d_Return3DPhaseFromIndividualDimensions( float phase_x, float phase_y, float phase_z, float2 &angles);

 __device__ __forceinline__ int
d_ReturnReal1DAddressFromPhysicalCoord(int3 coords, int4 img_dims);

 __device__ __forceinline__ int
d_ReturnFourier1DAddressFromPhysicalCoord(int3 wanted_dims, int3 physical_upper_bound_complex);

__device__ __forceinline__ int
d_ReturnFourier1DAddressFromLogicalCoord(int wanted_x_coord, int wanted_y_coord, int wanted_z_coord, const int3& dims, const int3& physical_upper_bound_complex);


 __inline__ int
ReturnFourierLogicalCoordGivenPhysicalCoord_Y(int physical_index,
                                              int logical_y_dimension,
                                              int physical_index_of_first_negative_frequency_y)
{
    if (physical_index >= physical_index_of_first_negative_frequency_y)
    {
    	 return physical_index - logical_y_dimension;
    }
    else return physical_index;
};
__inline__ int
ReturnFourierLogicalCoordGivenPhysicalCoord_Z(int physical_index,
                                              int logical_z_dimension,
                                              int physical_index_of_first_negative_frequency_z)
{
    if (physical_index >= physical_index_of_first_negative_frequency_z)
    {
    	 return physical_index - logical_z_dimension;
    }
    else return physical_index;
};

////////////////// For thrust
typedef struct 
{
  __host__ __device__
    float operator()(const float& x) const {
      return x * x;
    }
} square;
////////////////////////

GpuImage::GpuImage()
{ 
  SetupInitialValues();
}


GpuImage::GpuImage(Image &cpu_image) 
{

  SetupInitialValues();
  Init(cpu_image);
	
}

GpuImage::GpuImage( const GpuImage &other_gpu_image) // copy constructor
{

	SetupInitialValues();
	*this = other_gpu_image;
}

GpuImage & GpuImage::operator = (const GpuImage &other_gpu_image)
{
	*this = &other_gpu_image;
	return *this;
}


GpuImage & GpuImage::operator = (const GpuImage *other_gpu_image)
{
	// Check for self assignment
	if(this != other_gpu_image)
	{

		MyAssertTrue(other_gpu_image->is_in_memory_gpu, "Other image Memory not allocated");

		if (is_in_memory_gpu == true)
		{

			if (dims.x != other_gpu_image->dims.x || dims.y != other_gpu_image->dims.y || dims.z != other_gpu_image->dims.z)
			{
				Deallocate();
				Allocate(other_gpu_image->dims.x, other_gpu_image->dims.y, other_gpu_image->dims.z, other_gpu_image->is_in_real_space);
			}
		}
		else
		{
			Allocate(other_gpu_image->dims.x, other_gpu_image->dims.y, other_gpu_image->dims.z, other_gpu_image->is_in_real_space);
		}

		// by here the memory allocation should be ok..

		is_in_real_space = other_gpu_image->is_in_real_space;
		object_is_centred_in_box = other_gpu_image->object_is_centred_in_box;

		precheck
		cudaErr(cudaMemcpyAsync(real_values_gpu,other_gpu_image->real_values_gpu,sizeof(cufftReal)*real_memory_allocated,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
		postcheck
	}

   return *this;
}

GpuImage::~GpuImage() 
{
	Deallocate();
	// cudaErr(cudaFree(tmpVal));
	// cudaErr(cudaFree(tmpValComplex));
}


void GpuImage::Init(Image &cpu_image)
{
	CopyFromCpuImage(cpu_image);
}

void GpuImage::SetupInitialValues()
{

	dims = make_int4(0, 0, 0, 0); pitch = 0;
  // FIXME: Tempororay for compatibility with the IMage class
  logical_x_dimension = 0; logical_y_dimension = 0; logical_z_dimension = 0;
	physical_upper_bound_complex = make_int3(0, 0, 0);
	physical_address_of_box_center = make_int3(0, 0, 0);
	physical_index_of_first_negative_frequency = make_int3(0, 0, 0);
	logical_upper_bound_complex = make_int3(0, 0, 0);
	logical_lower_bound_complex = make_int3(0, 0, 0);
	logical_upper_bound_real = make_int3(0, 0, 0);
	logical_lower_bound_real = make_int3(0, 0, 0);

	fourier_voxel_size = make_float3(0.0f, 0.0f, 0.0f);


	number_of_real_space_pixels = 0;


	real_values = NULL;
	complex_values = NULL;

	real_memory_allocated = 0;


	padding_jump_value = 0;

	ft_normalization_factor = 0;

	real_values_gpu = NULL;									// !<  Real array to hold values for REAL images.
	complex_values_gpu = NULL;								// !<  Complex array to hold values for COMP images.


	gpu_plan_id = -1;

	insert_into_which_reconstruction = 0;
	hostImage = NULL;

	cudaErr(cudaEventCreateWithFlags(&nppCalcEvent, cudaEventDisableTiming);)

	cudaErr(cudaGetDevice(&device_idx));
	cudaErr(cudaDeviceGetAttribute(&number_of_streaming_multiprocessors, cudaDevAttrMultiProcessorCount, device_idx));
	limit_SMs_by_threads = 1;

	UpdateBoolsToDefault();

}

void GpuImage::CopyFrom(GpuImage *other_image)
{
	*this = other_image;
}

void GpuImage::CopyFromCpuImage(Image &cpu_image) 
{

	UpdateBoolsToDefault();

	dims = make_int4(cpu_image.logical_x_dimension,
				   cpu_image.logical_y_dimension,
				   cpu_image.logical_z_dimension,
				   cpu_image.logical_x_dimension + cpu_image.padding_jump_value);

  logical_x_dimension = cpu_image.logical_x_dimension;
  logical_y_dimension = cpu_image.logical_y_dimension;
  logical_z_dimension = cpu_image.logical_z_dimension;

	pitch = dims.w * sizeof(float);

	physical_upper_bound_complex = make_int3(cpu_image.physical_upper_bound_complex_x,
										   cpu_image.physical_upper_bound_complex_y,
										   cpu_image.physical_upper_bound_complex_z);

	physical_address_of_box_center   = make_int3(cpu_image.physical_address_of_box_center_x,
											   cpu_image.physical_address_of_box_center_y,
											   cpu_image.physical_address_of_box_center_z);

	physical_index_of_first_negative_frequency = make_int3(0,
														 cpu_image.physical_index_of_first_negative_frequency_y,
														 cpu_image.physical_index_of_first_negative_frequency_z);


	logical_upper_bound_complex = make_int3(cpu_image.logical_upper_bound_complex_x,
										  cpu_image.logical_upper_bound_complex_y,
										  cpu_image.logical_upper_bound_complex_z);


	logical_lower_bound_complex = make_int3(cpu_image.logical_lower_bound_complex_x,
										  cpu_image.logical_lower_bound_complex_y,
										  cpu_image.logical_lower_bound_complex_z);


	logical_upper_bound_real = make_int3(cpu_image.logical_upper_bound_real_x,
									   cpu_image.logical_upper_bound_real_y,
									   cpu_image.logical_upper_bound_real_z);

	logical_lower_bound_real = make_int3(cpu_image.logical_lower_bound_real_x,
									   cpu_image.logical_lower_bound_real_y,
									   cpu_image.logical_lower_bound_real_z);


	is_in_real_space = cpu_image.is_in_real_space;
	number_of_real_space_pixels = cpu_image.number_of_real_space_pixels;
	object_is_centred_in_box = cpu_image.object_is_centred_in_box;

	fourier_voxel_size = make_float3(cpu_image.fourier_voxel_size_x,
								   cpu_image.fourier_voxel_size_y,
								   cpu_image.fourier_voxel_size_z);


	insert_into_which_reconstruction = cpu_image.insert_into_which_reconstruction;
	real_values = cpu_image.real_values;
	complex_values = cpu_image.complex_values;

	is_in_memory = cpu_image.is_in_memory;

	padding_jump_value = cpu_image.padding_jump_value;
	image_memory_should_not_be_deallocated = cpu_image.image_memory_should_not_be_deallocated; // TODO what is this for?


	real_values_gpu = NULL;									// !<  Real array to hold values for REAL images.
	complex_values_gpu = NULL;								// !<  Complex array to hold values for COMP images.
	is_in_memory_gpu = false;
	real_memory_allocated =  cpu_image.real_memory_allocated;


	ft_normalization_factor = cpu_image.ft_normalization_factor;

	// FIXME for now always pin the memory - this might be a bad choice for single copy or small images, but is required for asynch xfer and is ~2x as fast after pinning
	cudaHostRegister(real_values, sizeof(float)*real_memory_allocated, cudaHostRegisterDefault);
	is_host_memory_pinned = true;
	is_meta_data_initialized = true;
	cudaHostGetDevicePointer( &pinnedPtr, real_values, 0);

	cudaMallocManaged(&tmpVal, sizeof(float));
	cudaMallocManaged(&tmpValComplex, sizeof(double));


	hostImage = &cpu_image;
 
}

void GpuImage::UpdateCpuFlags() 
{

  // Call after re-copying. The main image properites are all assumed to be static.
  is_in_real_space = hostImage->is_in_real_space;
  object_is_centred_in_box = hostImage->object_is_centred_in_box;

}

void GpuImage::printVal(std::string msg, int idx)
{

  float h_printVal = -9999.0f;

  cudaErr(cudaMemcpy(&h_printVal, &real_values_gpu[idx], sizeof(float), cudaMemcpyDeviceToHost));
  cudaStreamSynchronize(cudaStreamPerThread);
  wxPrintf("%s %6.6e\n", msg, h_printVal);

};

bool GpuImage::HasSameDimensionsAs(GpuImage *other_image)
{
	// Functions that call this method also assume these asserts are being called here, so do not remove.
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(other_image->is_in_memory_gpu, "Other image Memory not allocated");
	// end of dependent asserts.

	if (dims.x == other_image->dims.x && dims.y == other_image->dims.y && dims.z == other_image->dims.z) return true;
	else return false;
}

void GpuImage::MultiplyPixelWiseComplexConjugate(GpuImage &other_image)
{
	// FIXME when adding real space complex images
	MyAssertFalse( is_in_real_space, "Image is in real space");
	MyAssertFalse( other_image.is_in_real_space, "Other image is in real space");
	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");


	//  NppInit();
	//  Conj();
	//  npp_stat = nppiMul_32sc_C1IRSfs((const Npp32sc *)complex_values_gpu, 1, (Npp32sc*)other_image.complex_values_gpu, 1, npp_ROI_complex, 0);

	precheck
	ReturnLaunchParamters(dims, false);
	MultiplyPixelWiseComplexConjugateKernel<< <gridDims, threadsPerBlock,0, cudaStreamPerThread>> > (complex_values_gpu, other_image.complex_values_gpu,this->dims, this->physical_upper_bound_complex);
	postcheck

}


__global__ void ReturnSumOfRealValuesOnEdgesKernel(cufftReal *real_values_gpu, int4 dims, int padding_jump_value, float* returnValue);

float GpuImage::ReturnAverageOfRealValuesOnEdges()
{
	// FIXME to use a masked routing, this is slow af
	MyAssertTrue(is_in_memory, "Memory not allocated");
	MyAssertTrue(dims.z == 1, "ReturnAverageOfRealValuesOnEdges only implemented in 2d");

	precheck
	*tmpVal = 5.0f;
	ReturnSumOfRealValuesOnEdgesKernel<< <1, 1, 0, cudaStreamPerThread>> >(real_values_gpu, dims, padding_jump_value, tmpVal);
	postcheck

	// Need to wait on the return value
	cudaErr(cudaStreamSynchronize(cudaStreamPerThread));


  return *tmpVal;
}

__global__ void ReturnSumOfRealValuesOnEdgesKernel(cufftReal *real_values_gpu, int4 dims, int padding_jump_value, float* returnValue)
{

	int pixel_counter;
	int line_counter;
//	int plane_counter;

	double sum = 0.0;
	int number_of_pixels = 0;
	int address = 0;


		// Two-dimensional image
		// First line
		for (pixel_counter=0; pixel_counter < dims.x; pixel_counter++)
		{
			sum += real_values_gpu[address];
			address++;
		}
		number_of_pixels += dims.x;
		address += padding_jump_value;

		// Other lines
		for (line_counter=1; line_counter < dims.y-1; line_counter++)
		{
			sum += real_values_gpu[address];
			address += dims.x-1;
			sum += real_values_gpu[address];
			address += padding_jump_value + 1;
			number_of_pixels += 2;
		}

		// Last line
		for (pixel_counter=0; pixel_counter < dims.x; pixel_counter++)
		{
			sum += real_values_gpu[address];
			address++;
		}
		number_of_pixels += dims.x;

   *returnValue = (float)sum / (float)number_of_pixels;
}

//void GpuImage::CublasInit()
//{
//  if ( ! is_cublas_loaded )
//  {
//    cublasCreate(&cublasHandle);
//    is_cublas_loaded = true;
//    cublasSetStream(cublasHandle, cudaStreamPerThread);
//  }
//}

void GpuImage::NppInit()
{
  if ( ! is_npp_loaded )
  {

	int sharedMem;
	nppStream.hStream = cudaStreamPerThread; // FIXME to use member stream
	cudaGetDevice(&nppStream.nCudaDeviceId);
	cudaDeviceGetAttribute(&nppStream.nMultiProcessorCount,cudaDevAttrMultiProcessorCount,nppStream.nCudaDeviceId);
	cudaDeviceGetAttribute(&nppStream.nMaxThreadsPerMultiProcessor,cudaDevAttrMaxThreadsPerMultiProcessor,nppStream.nCudaDeviceId);
	cudaDeviceGetAttribute(&nppStream.nMaxThreadsPerBlock,cudaDevAttrMaxThreadsPerBlock,nppStream.nCudaDeviceId);
	cudaDeviceGetAttribute(&nppStream.nMaxThreadsPerMultiProcessor,cudaDevAttrMaxThreadsPerMultiProcessor,nppStream.nCudaDeviceId);
	cudaDeviceGetAttribute(&sharedMem,cudaDevAttrMaxSharedMemoryPerBlock,nppStream.nCudaDeviceId);
	nppStream.nSharedMemPerBlock = (size_t)sharedMem;
	cudaDeviceGetAttribute(&nppStream.nCudaDevAttrComputeCapabilityMajor,cudaDevAttrComputeCapabilityMajor,nppStream.nCudaDeviceId);
	cudaDeviceGetAttribute(&nppStream.nCudaDevAttrComputeCapabilityMinor,cudaDevAttrComputeCapabilityMinor,nppStream.nCudaDeviceId);

//    nppSetStream(cudaStreamPerThread);

	npp_ROI_real_space.width = dims.x;
	npp_ROI_real_space.height = dims.y * dims.z;

	npp_ROI_fourier_space.width = dims.w / 2;
	npp_ROI_fourier_space.height = dims.y * dims.z;

	// This is used only in special cases, and is explicit.
    npp_ROI_fourier_with_real_functor.width = dims.w;
    npp_ROI_fourier_with_real_functor.height = dims.y * dims.z;

    // This is switched appropriately in FFT/iFFT calls
    if (is_in_real_space)
    {
    	npp_ROI = npp_ROI_real_space;
    }
    else
    {
    	npp_ROI = npp_ROI_fourier_space;
    }

    is_npp_loaded = true;

  }
}

void GpuImage::BufferInit(BufferType bt)
{

  switch (bt)
  {
    case b_image :
        if ( ! is_allocated_image_buffer )
        {
          image_buffer = new GpuImage;
          *image_buffer = *this;
          is_allocated_image_buffer = true;
        }     
        break;

    case b_16f :
    	if ( ! is_allocated_16f_buffer )
    	{
    		cudaErr(cudaMalloc(&real_values_16f, sizeof(__half)*real_memory_allocated));
    		complex_values_16f = (__half2 *)real_values_16f;
    		is_allocated_16f_buffer = true;
    	}
    	break;

    case b_sum :
        if ( ! is_allocated_sum_buffer ) 
        {
          int n_elem;
          nppiSumGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
          cudaErr(cudaMalloc(&this->sum_buffer, n_elem));
          is_allocated_sum_buffer = true;
        }     
        break;   

    case b_min :
        if ( ! is_allocated_min_buffer )
        {
          int n_elem;
          nppiMinGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
          cudaErr(cudaMalloc(&this->min_buffer, n_elem));
          is_allocated_min_buffer = true;
        }
        break;

  case b_minIDX :
      if ( ! is_allocated_minIDX_buffer )
      {
        int n_elem;
        nppiMinIndxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        cudaErr(cudaMalloc(&this->minIDX_buffer, n_elem));
        is_allocated_minIDX_buffer = true;
      }
      break;

  case b_max :
      if ( ! is_allocated_max_buffer )
      {
        int n_elem;
        nppiMaxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        cudaErr(cudaMalloc(&this->max_buffer, n_elem));
        is_allocated_max_buffer = true;
      }
      break;

  case b_maxIDX :
      if ( ! is_allocated_maxIDX_buffer )
      {
        int n_elem;
        nppiMaxIndxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        cudaErr(cudaMalloc(&this->maxIDX_buffer, n_elem));
        is_allocated_maxIDX_buffer = true;
      }
      break;

  case b_minmax :
      if ( ! is_allocated_minmax_buffer )
      {
        int n_elem;
        nppiMinMaxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        cudaErr(cudaMalloc(&this->minmax_buffer, n_elem));
        is_allocated_minmax_buffer = true;
      }
      break;

  case b_minmaxIDX :
      if ( ! is_allocated_minmaxIDX_buffer )
      {
        int n_elem;
        nppiMinMaxIndxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        cudaErr(cudaMalloc(&this->minmaxIDX_buffer, n_elem));
        is_allocated_minmaxIDX_buffer = true;
      }
      break;

  case b_mean :
      if ( ! is_allocated_mean_buffer )
      {
        int n_elem;
        nppiMeanGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        cudaErr(cudaMalloc(&this->mean_buffer, n_elem));
        is_allocated_mean_buffer = true;
      }
      break;

  case b_meanstddev :
      if ( ! is_allocated_meanstddev_buffer )
      {
        int n_elem;
        nppiMeanGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        cudaErr(cudaMalloc(&this->meanstddev_buffer, n_elem));
        is_allocated_meanstddev_buffer = true;
      }
      break;

  case b_countinrange :
      if ( ! is_allocated_countinrange_buffer )
      {
        int n_elem;
        nppiCountInRangeGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        cudaErr(cudaMalloc(&this->countinrange_buffer, n_elem));
        is_allocated_countinrange_buffer = true;
      }
      break;

  case b_l2norm :
	  if (! is_allocated_l2norm_buffer)
	  {
		  int n_elem;
		  nppiNormL2GetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
		  cudaErr(cudaMalloc(&this->l2norm_buffer,n_elem));

		  is_allocated_l2norm_buffer = true;
	  }
	  break;

  case b_dotproduct :
	  if ( ! is_allocated_dotproduct_buffer )
	  {
		  int n_elem;
		  nppiDotProdGetBufferHostSize_32f64f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
		  cudaErr(cudaMalloc(&this->dotproduct_buffer, n_elem));
		  is_allocated_dotproduct_buffer = true;
	  }
	  break;


	}
}

void GpuImage::BufferDestroy()
{

	if ( is_allocated_image_buffer )
	{
		image_buffer->Deallocate();
		cudaErr(cudaFree(image_buffer));
		is_allocated_image_buffer = false;
	}     

	if ( is_allocated_16f_buffer )
	{
		cudaErr(cudaFree(real_values_16f));
		is_allocated_16f_buffer = false;
	}

	if ( is_allocated_sum_buffer ) 
	{
		cudaErr(cudaFree(sum_buffer));
		is_allocated_sum_buffer = false;
	}     

	if ( is_allocated_min_buffer )
	{
		cudaErr(cudaFree(min_buffer));
		is_allocated_min_buffer = false;
	}

	if ( is_allocated_minIDX_buffer )
	{
		cudaErr(cudaFree(minIDX_buffer));
		is_allocated_minIDX_buffer = false;
	}

	if ( is_allocated_max_buffer )
	{
		cudaErr(cudaFree(max_buffer));
		is_allocated_max_buffer = false;
	}

	if ( is_allocated_maxIDX_buffer )
	{
		cudaErr(cudaFree(maxIDX_buffer));
		is_allocated_maxIDX_buffer = false;
	}

	if ( is_allocated_minmax_buffer )
	{
		cudaErr(cudaFree(minmax_buffer));
		is_allocated_minmax_buffer = false;
	}

	if ( is_allocated_minmaxIDX_buffer )
	{
		cudaErr(cudaFree(minmaxIDX_buffer));
		is_allocated_minmaxIDX_buffer = false;
	}

	if ( is_allocated_mean_buffer )
	{
		cudaErr(cudaFree(mean_buffer));
		is_allocated_mean_buffer = false;
	}

	if ( is_allocated_meanstddev_buffer )
	{
		cudaErr(cudaFree(meanstddev_buffer));
		is_allocated_meanstddev_buffer = false;
	}

	if ( is_allocated_countinrange_buffer )
	{
		cudaErr(cudaFree(countinrange_buffer));
		is_allocated_countinrange_buffer = false;
	}

	if (is_allocated_l2norm_buffer)
	{
		cudaErr(cudaFree(l2norm_buffer));
		is_allocated_l2norm_buffer = false;
	}

	if ( is_allocated_dotproduct_buffer )
	{
		cudaErr(cudaFree(dotproduct_buffer));
		is_allocated_dotproduct_buffer = false;
	}


}



float GpuImage::ReturnSumOfSquares()
{

	// FIXME this assumes padded values are zero which is not strictly true
	MyAssertTrue(is_in_memory_gpu, "Image not allocated");
	MyAssertTrue(is_in_real_space, "This method is for real space, use ReturnSumSquareModulusComplexValues for Fourier space")


	BufferInit(b_l2norm);
	NppInit();

	nppErr(nppiNorm_L2_32f_C1R_Ctx((Npp32f *)real_values_gpu, pitch, npp_ROI,
									 	   (Npp64f *)tmpValComplex, (Npp8u *)this->l2norm_buffer, nppStream));

	cudaErr(cudaStreamSynchronize(nppStream.hStream));

	return (float)(*tmpValComplex * *tmpValComplex);

//	CublasInit();
//	// With real and complex interleaved, treating as real is equivalent to taking the conj dot prod
//	cublas_stat = cublasSdot( cublasHandle, real_memory_allocated,
//							real_values_gpu, 1,
//							real_values_gpu, 1,
//							&returnValue);
//
//	if (cublas_stat) {
//	wxPrintf("Cublas return val %s\n", cublas_stat); }
//
//
//	return returnValue;

}



float GpuImage::ReturnSumSquareModulusComplexValues()
{

	//
	MyAssertTrue(is_in_memory_gpu, "Image not allocated");
	MyAssertFalse(is_in_real_space, "This method is NOT for real space, use ReturnSumofSquares for realspace")
	int address = 0;
	bool x_is_even = IsEven(dims.x);
	int i,j,k,jj,kk;
	const std::complex<float> c1(sqrtf(0.25f),sqrtf(0.25));
	const std::complex<float> c2(sqrtf(0.5f),sqrtf(0.5f)); // original code is pow(abs(Val),2)*0.5
	const std::complex<float> c3(1.0,1.0);
	const std::complex<float> c4(0.0,0.0);
//	float returnValue;

	if ( ! is_allocated_mask_CSOS )
	{

		wxPrintf("is mask allocated %d\n", is_allocated_mask_CSOS);
		mask_CSOS = new GpuImage;
		is_allocated_mask_CSOS = true;
		wxPrintf("is mask allocated %d\n", is_allocated_mask_CSOS);
		// create a mask that can be reproduce the correct weighting from Image::ReturnSumOfSquares on complex images

		wxPrintf("\n\tMaking mask_CSOS\n");
		mask_CSOS->Allocate(dims.x, dims.y, dims.z, true);
		// The mask should always be in real_space, and starts out not centered
		mask_CSOS->is_in_real_space = false;
		mask_CSOS->object_is_centred_in_box = true;
		// Allocate pinned host memb
		cudaErr(cudaHostAlloc(&mask_CSOS->real_values, sizeof(float)*real_memory_allocated, cudaHostAllocDefault));
		mask_CSOS->complex_values = (std::complex<float>*) mask_CSOS->real_values;

		for (k = 0; k <= physical_upper_bound_complex.z; k++)
		{

			kk = ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k, dims.z, physical_index_of_first_negative_frequency.z);
			for (j = 0; j <= physical_upper_bound_complex.y; j++)
			{
				jj = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j,dims.y, physical_index_of_first_negative_frequency.y);
				for (i = 0; i <= physical_upper_bound_complex.x; i++)
				{
					if ((i  == 0 || (i  == logical_upper_bound_complex.x && x_is_even)) &&
						(jj == 0 || (jj == logical_lower_bound_complex.y && x_is_even)) &&
						(kk == 0 || (kk == logical_lower_bound_complex.z && x_is_even)))
					{
						mask_CSOS->complex_values[address] = c2;
					}
					else if ((i == 0 || (i == logical_upper_bound_complex.x && x_is_even)) && dims.z != 1) 
					{
						mask_CSOS->complex_values[address] = c1;
					}
					else if ((i != 0 && (i != logical_upper_bound_complex.x || ! x_is_even)) || (jj >= 0 && kk >= 0)) 
					{
						mask_CSOS->complex_values[address] = c3;
					}
					else
					{
						mask_CSOS->complex_values[address] = c4;
					}

					address++;
				}
			}
		}   



		precheck
		cudaErr(cudaMemcpyAsync(mask_CSOS->real_values_gpu, mask_CSOS->real_values,sizeof(float)*real_memory_allocated,cudaMemcpyHostToDevice,cudaStreamPerThread));
		precheck
		// TODO change this to an event that can then be later checked prior to deleteing
		cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
		cudaErr(cudaFreeHost(mask_CSOS->real_values));

	} // end of mask creation


	BufferInit(b_image);
	precheck
	cudaErr(cudaMemcpyAsync(image_buffer->real_values_gpu, mask_CSOS->real_values_gpu, sizeof(float)*real_memory_allocated,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
	postcheck

	image_buffer->is_in_real_space = false;
	image_buffer->npp_ROI = image_buffer->npp_ROI_fourier_space;
	image_buffer->MultiplyPixelWise(*this);

//	CublasInit();
	// With real and complex interleaved, treating as real is equivalent to taking the conj dot prod
	precheck


	BufferInit(b_l2norm);
	NppInit();
	nppErr(nppiNorm_L2_32f_C1R_Ctx((Npp32f *)image_buffer->real_values_gpu, pitch, npp_ROI_fourier_with_real_functor,
									 	   (Npp64f *)tmpValComplex, (Npp8u *)this->l2norm_buffer, nppStream));

	cudaErr(cudaStreamSynchronize(nppStream.hStream));

	return (float)(*tmpValComplex * *tmpValComplex );

//	cublasSdot( cublasHandle, real_memory_allocated,
//			  image_buffer->real_values_gpu, 1,
//			  image_buffer->real_values_gpu, 1,
//			  &returnValue);

	postcheck

//	return (float)(*dotProd * 2.0f);
//	return (float)(returnValue * 2.0f);

  
}


__global__ void MultiplyPixelWiseComplexConjugateKernel(cufftComplex* ref_complex_values, cufftComplex* img_complex_values, int4 dims, int3 physical_upper_bound_complex)
{
  int3 coords = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z);

    if (coords.x < dims.w / 2 && coords.y < dims.y && coords.z < dims.z)
    {

	    int address = d_ReturnFourier1DAddressFromPhysicalCoord(coords, physical_upper_bound_complex);

	    ref_complex_values[address] = (cufftComplex)ComplexConjMul((Complex)img_complex_values[address],(Complex)ref_complex_values[address]);
    }
    

}

void GpuImage::MipPixelWise(GpuImage &other_image)
{

	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
	precheck
	ReturnLaunchParamters(dims, true);
	MipPixelWiseKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> > (real_values_gpu, other_image.real_values_gpu, this->dims);
	postcheck

}
__global__ void MipPixelWiseKernel(cufftReal *mip, const cufftReal *correlation_output, const int4 dims)
{

    int3 coords = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z);

    if (coords.x < dims.x && coords.y < dims.y && coords.z < dims.z)
    {
	    int address = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);
	    mip[address] = MAX(mip[address], correlation_output[address]);
    }
}

void GpuImage::MipPixelWise(GpuImage &other_image, GpuImage &psi, GpuImage &phi, GpuImage &theta, GpuImage &defocus, GpuImage &pixel,
                            float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel)
{

	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
	precheck
	ReturnLaunchParamters(dims, true);
	MipPixelWiseKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >(real_values_gpu, other_image.real_values_gpu,
																	   psi.real_values_gpu,phi.real_values_gpu,theta.real_values_gpu,defocus.real_values_gpu,pixel.real_values_gpu,
																		this->dims, c_psi, c_phi, c_theta, c_defocus, c_pixel);
	postcheck


}



__global__ void MipPixelWiseKernel(cufftReal* mip, cufftReal *correlation_output, cufftReal *psi, cufftReal *phi, cufftReal *theta, cufftReal *defocus, cufftReal *pixel, const int4 dims,
                                   float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel)
{

    int3 coords = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z);

    if (coords.x < dims.x && coords.y < dims.y && coords.z < dims.z)
    {
	    int address = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);
      if (correlation_output[address] > mip[address])
      {
        mip[address] = correlation_output[address];
        psi[address] = c_psi;
        phi[address] = c_phi;
        theta[address] = c_theta;
        defocus[address] = c_defocus;
        pixel[address] = c_pixel;
      }

    }
}

void GpuImage::MipPixelWise(GpuImage &other_image, GpuImage &psi, GpuImage &phi, GpuImage &theta,
                            float c_psi, float c_phi, float c_theta)
{

	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
	precheck
	ReturnLaunchParamters(dims, true);
	MipPixelWiseKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >(real_values_gpu, other_image.real_values_gpu,
																	   psi.real_values_gpu,phi.real_values_gpu,theta.real_values_gpu,
																	   this->dims, c_psi, c_phi, c_theta);
	postcheck

}



__global__ void MipPixelWiseKernel(cufftReal* mip, cufftReal *correlation_output, cufftReal *psi, cufftReal *phi, cufftReal *theta, const int4 dims,
                                   float c_psi, float c_phi, float c_theta)
{

    int3 coords = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z);

    if (coords.x < dims.x && coords.y < dims.y && coords.z < dims.z)
    {
	  int address = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);
      if (correlation_output[address] > mip[address])
      {
        mip[address] = correlation_output[address];
        psi[address] = c_psi;
        phi[address] = c_phi;
        theta[address] = c_theta;
      }

    }
}

void GpuImage::ApplyBFactor(float bfactor)
{

  MyAssertFalse(is_in_real_space, "This function is only for Fourier space images.");

  precheck
  ReturnLaunchParamters(dims, false);
  ApplyBFactorKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> > (complex_values_gpu, 
                                                                             dims, 
                                                                             physical_index_of_first_negative_frequency,
                                                                             physical_upper_bound_complex,
                                                                             bfactor * 0.25f);
  postcheck
}

__global__ void ApplyBFactorKernel(cufftComplex* d_input, 
                                   const int4 dims, 
                                   const int3 physical_index_of_first_negative_frequency,
                                   const int3 physical_upper_bound_complex,
                                   float bfactor)
{

  int3 physical_dims = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                                 blockIdx.y*blockDim.y + threadIdx.y,
                                 blockIdx.z);

  if (physical_dims.x <= physical_upper_bound_complex.x && 
      physical_dims.y <= physical_upper_bound_complex.y && 
      physical_dims.z <= physical_upper_bound_complex.z)
  {
    const int address = d_ReturnFourier1DAddressFromPhysicalCoord(physical_dims, physical_upper_bound_complex);
    int ret_val;
    int frequency_squared;

    frequency_squared = d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(physical_dims.y, dims.y, physical_index_of_first_negative_frequency.y);
    frequency_squared *= frequency_squared;

    ret_val = d_ReturnFourierLogicalCoordGivenPhysicalCoord_X(physical_dims.x, dims.x, physical_index_of_first_negative_frequency.x);
    frequency_squared += ret_val * ret_val;

    ret_val = d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(physical_dims.z, dims.z, physical_index_of_first_negative_frequency.z);
    frequency_squared += ret_val * ret_val;

    ComplexScale((Complex *)&d_input[address], expf(-bfactor * frequency_squared));
  }

}

void GpuImage::ApplyBFactor(float bfactor, const float vertical_mask_size, const float horizontal_mask_size)
{

  MyDebugAssertFalse(is_in_real_space, "This function is only for Fourier space images.");
  MyDebugAssertTrue(dims.z == 1, "This function is only for 2D images.");
	MyDebugAssertTrue(vertical_half_width > 0 && horizontal_half_width > 0, "Half width must be greater than 0");

  precheck
  ReturnLaunchParamters(dims, false);
  ApplyBFactorKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> > (complex_values_gpu, 
                                                                             dims, 
                                                                             physical_index_of_first_negative_frequency,
                                                                             physical_upper_bound_complex,
                                                                             bfactor * 0.25f, 
                                                                             vertical_mask_size,
                                                                             horizontal_mask_size);
  postcheck
}

__global__ void ApplyBFactorKernel(cufftComplex* d_input, 
                                  const int4 dims, 
                                  const int3 physical_index_of_first_negative_frequency,
                                  const int3 physical_upper_bound_complex,
                                  float bfactor, 
                                  const float vertical_mask_size,
                                  const float horizontal_mask_size)
{

  int3 physical_dims = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                                 blockIdx.y*blockDim.y + threadIdx.y,
                                 blockIdx.z);

  if (physical_dims.x <= physical_upper_bound_complex.x && 
      physical_dims.y <= physical_upper_bound_complex.y)
  {
    int address = d_ReturnFourier1DAddressFromPhysicalCoord(physical_dims, physical_upper_bound_complex);

    if (physical_dims.x <= horizontal_mask_size - 1 ||
        physical_dims.y <= vertical_mask_size - 1 ||
        physical_dims.y >= physical_upper_bound_complex.y - vertical_mask_size)
    {
      // TODO: confirm this is correct
      // Mask the central cross
      d_input[address].x = 0.f;
      d_input[address].y = 0.f;
    }
    else
    {
      int x2, y2;

      y2 = d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(physical_dims.y, dims.y, physical_index_of_first_negative_frequency.y);
      y2 *= y2;
  
      x2 = d_ReturnFourierLogicalCoordGivenPhysicalCoord_X(physical_dims.x, dims.x, physical_index_of_first_negative_frequency.x);
      x2 *= x2;
  
      ComplexScale((Complex *)&d_input[address], expf(-bfactor * (x2 + y2)) );
    }
  }

}

void GpuImage::CalculateCrossCorrelationImageWith(GpuImage *other_image)
{

  MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space == other_image->is_in_real_space, "Images are in different spaces");
	MyDebugAssertTrue(HasSameDimensionsAs(other_image) == true, "Images are different sizes");

	bool must_fft = false;

	// do we have to fft..

	if (is_in_real_space == true)
	{
		must_fft = true;
		ForwardFFT();
		other_image->ForwardFFT();
	}

  if (object_is_centred_in_box == true)
	{
		object_is_centred_in_box = false;
		SwapRealSpaceQuadrants();
	}

  if (other_image->object_is_centred_in_box == true)
	{
		other_image->object_is_centred_in_box = false;
		other_image->SwapRealSpaceQuadrants();
	}

  BackwardFFTAfterComplexConjMul(other_image->complex_values_gpu, false);


	if (must_fft == true) other_image->BackwardFFT();

}

Peak GpuImage::FindPeakWithParabolaFit(float inner_radius_for_peak_search, float outer_radius_for_peak_search)
{

  MyDebugAssertTrue(is_in_real_space, "This function is only for real space images.");
  MyDebugAssertTrue(dims.z == 1, "This function is only for 2D images.");

  Peak my_peak;
  GpuImage buffer;
  int size = myroundint(outer_radius_for_peak_search * 2.0f)+1;
  if (size % 2 != 0) size++;
  buffer.Allocate(size, size, true);
  ClipIntoRealSpace(&buffer, 0.f, false, 0.f, 0, 0, 0);
  // Since we free the GPU memory after the copy, we have to make sure the kernel is complete.
  cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

  bool should_block_until_complete = true;
  bool free_gpu_memory = true;
  Image cpu_buffer = buffer.CopyDeviceToNewHost(should_block_until_complete, free_gpu_memory);
  my_peak = cpu_buffer.FindPeakWithParabolaFit(inner_radius_for_peak_search, outer_radius_for_peak_search);
  wxPrintf("Peak found at %f, %f\n", my_peak.x, my_peak.y);

  return my_peak; 

}


void GpuImage::Abs()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");

  NppInit();
  nppErr(nppiAbs_32f_C1IR_Ctx((Npp32f *)real_values_gpu, pitch, npp_ROI,nppStream));
}

void GpuImage::AbsDiff(GpuImage &other_image)
{
	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");


	BufferInit(b_image);
	NppInit();

	nppErr(nppiAbsDiff_32f_C1R_Ctx((const Npp32f *)real_values_gpu, pitch,
									 (const Npp32f *)other_image.real_values_gpu, pitch,
									 (      Npp32f *)this->image_buffer->real_values_gpu, pitch, npp_ROI,nppStream));

	precheck
	cudaErr(cudaMemcpyAsync(real_values_gpu,this->image_buffer->real_values_gpu,sizeof(cufftReal)*real_memory_allocated,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
	postcheck

}

void GpuImage::AbsDiff(GpuImage &other_image, GpuImage &output_image)
{
  // In place abs diff (see overload for out of place)
	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
	MyAssertTrue(HasSameDimensionsAs(&output_image), "Images have different dimension.");



  NppInit();
  
  nppErr(nppiAbsDiff_32f_C1R_Ctx((const Npp32f *)real_values_gpu, pitch,
                                     (const Npp32f *)other_image.real_values_gpu, pitch,
                                     (      Npp32f *)output_image.real_values_gpu, pitch, npp_ROI,nppStream));

}

void GpuImage::Min()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");

	NppInit();
	BufferInit(b_min);
	nppErr(nppiMin_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, min_buffer, (Npp32f *)&min_value,nppStream));
	cudaErr(cudaStreamSynchronize(nppStream.hStream));
}
void GpuImage::MinAndCoords()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");

	NppInit();
	BufferInit(b_minIDX);
	nppErr(nppiMinIndx_32f_C1R_Ctx((const Npp32f *)real_values_gpu, pitch, npp_ROI, minIDX_buffer, (Npp32f *)&min_value, &min_idx.x, &min_idx.y,nppStream));
	cudaErr(cudaStreamSynchronize(nppStream.hStream));

}
void GpuImage::Max()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	BufferInit(b_max);
	nppErr(nppiMax_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, max_buffer, (Npp32f *)&max_value,nppStream));
	cudaErr(cudaStreamSynchronize(nppStream.hStream));

}
void GpuImage::MaxAndCoords()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	BufferInit(b_maxIDX);
	nppErr(nppiMaxIndx_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, maxIDX_buffer, (Npp32f *)&max_value, &max_idx.x, &max_idx.y, nppStream));
	cudaErr(cudaStreamSynchronize(nppStream.hStream));

}

void GpuImage::MinMax()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	BufferInit(b_minmax);
	nppErr(nppiMinMax_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, (Npp32f *)&min_value, (Npp32f *)&max_value, minmax_buffer,nppStream));
	cudaErr(cudaStreamSynchronize(nppStream.hStream));

}
void GpuImage::MinMaxAndCoords()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	BufferInit(b_minmaxIDX);
	nppErr(nppiMinMaxIndx_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, (Npp32f *)&min_value, (Npp32f *)&max_value,  &min_idx, &max_idx,minmax_buffer,nppStream));
	cudaErr(cudaStreamSynchronize(nppStream.hStream));

}

void GpuImage::Mean()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");

	NppInit();
	BufferInit(b_mean);
	nppErr(nppiMean_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, mean_buffer, npp_mean, nppStream));
	cudaErr(cudaStreamSynchronize(nppStream.hStream));

	this->img_mean   = (float)*npp_mean;

}

void GpuImage::MeanStdDev()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	BufferInit(b_meanstddev);
	nppErr(nppiMean_StdDev_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, meanstddev_buffer, npp_mean, npp_stdDev,nppStream));
	cudaErr(cudaStreamSynchronize(nppStream.hStream));

	this->img_mean   = (float)*npp_mean;
	this->img_stdDev = (float)*npp_stdDev;
}

void GpuImage::MultiplyPixelWise(GpuImage &other_image)
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");

	NppInit();
	if (is_in_real_space)
	{
		nppErr(nppiMul_32f_C1IR_Ctx((Npp32f *)other_image.real_values_gpu, pitch, (Npp32f*)real_values_gpu, pitch, npp_ROI,nppStream));
	}
	else
	{
		nppErr(nppiMul_32fc_C1IR_Ctx((Npp32fc *)other_image.complex_values_gpu, pitch, (Npp32fc *)complex_values_gpu, pitch, npp_ROI, nppStream));
	}
}


void GpuImage::AddConstant(const float add_val)
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	nppErr(nppiAddC_32f_C1IR_Ctx((Npp32f)add_val, (Npp32f*)real_values_gpu, pitch, npp_ROI,nppStream));
}

void GpuImage::AddConstant(const Npp32fc add_val)
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Image in real space.")

	NppInit();
	nppErr(nppiAddC_32fc_C1IR_Ctx((Npp32fc)add_val, (Npp32fc*)complex_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::SquareRealValues()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");

	NppInit();
	nppErr(nppiSqr_32f_C1IR_Ctx((Npp32f *)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::SquareRootRealValues()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	nppErr(nppiSqrt_32f_C1IR_Ctx((Npp32f *)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::LogarithmRealValues()
{
  MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
  
  NppInit();
  nppErr(nppiLn_32f_C1IR_Ctx((Npp32f *)real_values_gpu, pitch, npp_ROI,nppStream));

}

void GpuImage::ExponentiateRealValues()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	nppErr(nppiExp_32f_C1IR_Ctx((Npp32f *)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::CountInRange(float lower_bound, float upper_bound)
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	nppErr(nppiCountInRange_32f_C1R_Ctx((const Npp32f *)real_values_gpu, pitch, npp_ROI, &number_of_pixels_in_range,
											(Npp32f)lower_bound,(Npp32f)upper_bound,countinrange_buffer, nppStream));
	cudaErr(cudaStreamSynchronize(nppStream.hStream));


}

float GpuImage::ReturnSumOfRealValues()
{
	// FIXME assuming padded values are zero
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");

	NppInit();
	BufferInit(b_sum);
	nppErr(nppiSum_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI,sum_buffer, (Npp64f *)tmpValComplex, nppStream));
	cudaErr(cudaStreamSynchronize(nppStream.hStream));

	return (float)*tmpValComplex;
}
void GpuImage::AddImage(GpuImage &other_image)
{
  // Add the real_values_gpu into a double array
	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");


	NppInit();
	nppErr(nppiAdd_32f_C1IR_Ctx((const Npp32f*)other_image.real_values_gpu, pitch, (Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));

} 
void GpuImage::SubtractImage(GpuImage &other_image)
{
  // Add the real_values_gpu into a double array
	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");

	NppInit();
	nppErr(nppiSub_32f_C1IR_Ctx((const Npp32f*)other_image.real_values_gpu, pitch, (Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));

} 
void GpuImage::AddSquaredImage(GpuImage &other_image)
{
	// Add the real_values_gpu into a double array
	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");
	MyAssertTrue(is_in_real_space, "Image is not in real space");

	NppInit();
	nppErr(nppiAddSquare_32f_C1IR_Ctx((const Npp32f*)other_image.real_values_gpu,  pitch, (Npp32f*)real_values_gpu,  pitch, npp_ROI, nppStream));
} 

void GpuImage::MultiplyByConstant(float scale_factor)
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");

	NppInit();
	if (is_in_real_space)
	{
		nppErr(nppiMulC_32f_C1IR_Ctx((Npp32f) scale_factor, (Npp32f*)real_values_gpu,  pitch, npp_ROI, nppStream));
	}
	else
	{
		nppErr(nppiMulC_32f_C1IR_Ctx((Npp32f) scale_factor, (Npp32f*)real_values_gpu,  pitch, npp_ROI_fourier_with_real_functor, nppStream));
	}
}


void GpuImage::SetToConstant(float scale_factor)
{
	MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");

	NppInit();
  if (is_in_real_space)
  {
    nppErr(nppiSet_32f_C1R_Ctx((Npp32f) scale_factor, (Npp32f*)real_values_gpu,  pitch, npp_ROI, nppStream));
  }
  else
  {
    Npp32fc scale_factor_complex = {scale_factor, scale_factor};
    SetToConstant( scale_factor_complex );
  }
}

void GpuImage::SetToConstant(Npp32fc scale_factor_complex)
{
	MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");

	NppInit();
  nppErr(nppiSet_32fc_C1R_Ctx((Npp32fc) scale_factor_complex, (Npp32fc*)complex_values_gpu,  pitch, npp_ROI, nppStream));
}



void GpuImage::Conj()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertFalse(is_in_real_space, "Conj only supports complex images");

	Npp32fc scale_factor;
	scale_factor.re =  1.0f;
	scale_factor.im = -1.0f;
	NppInit();
	nppErr(nppiMulC_32fc_C1IR_Ctx((Npp32fc)scale_factor, (Npp32fc*)complex_values_gpu, pitch, npp_ROI,nppStream));

}

void GpuImage::Zeros()
{
 
  MyAssertTrue(is_meta_data_initialized, "Host meta data has not been copied");

  if ( ! is_in_memory_gpu )
  {
    cudaErr(cudaMalloc(&real_values_gpu, real_memory_allocated*sizeof(float)));
    complex_values_gpu = (cufftComplex *)real_values_gpu;
    is_in_memory_gpu = true;
  }

  precheck
  cudaErr(cudaMemsetAsync(real_values_gpu, 0, real_memory_allocated*sizeof(float), cudaStreamPerThread));
  postcheck

}


void GpuImage::CopyHostToDevice()
{
 
	MyAssertTrue(is_in_memory, "Host memory not allocated");

	if ( ! is_in_memory_gpu )
	{
		cudaErr(cudaMalloc(&real_values_gpu, real_memory_allocated*sizeof(float)));
		complex_values_gpu = (cufftComplex *)real_values_gpu;
		is_in_memory_gpu = true;
	}

	precheck
	cudaErr(cudaMemcpyAsync( real_values_gpu, pinnedPtr, real_memory_allocated*sizeof(float),cudaMemcpyHostToDevice,cudaStreamPerThread));
	postcheck

	UpdateCpuFlags();

}

void GpuImage::CopyDeviceToHost(bool free_gpu_memory, bool unpin_host_memory)
{
 
	MyAssertTrue(is_in_memory_gpu, "GPU memory not allocated");
	// TODO other asserts on size etc.
	precheck
	cudaErr(cudaMemcpyAsync(pinnedPtr, real_values_gpu, real_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
	postcheck
	//  cudaErr(cudaMemcpyAsync(real_values, real_values_gpu, real_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
	// TODO add asserts etc.
	if (free_gpu_memory) 
	{ 
		Deallocate();
	}
	if (unpin_host_memory && is_host_memory_pinned)
	{
		cudaHostUnregister(real_values);
		is_host_memory_pinned = false;
	}

}

void GpuImage::CopyDeviceToHost(Image &cpu_image, bool should_block_until_complete, bool free_gpu_memory)
{

	MyAssertTrue(is_in_memory_gpu, "GPU memory not allocated");
	// TODO other asserts on size etc.


	float* tmpPinnedPtr;
	// FIXME for now always pin the memory - this might be a bad choice for single copy or small images, but is required for asynch xfer and is ~2x as fast after pinning
	cudaHostRegister(cpu_image.real_values, sizeof(float)*real_memory_allocated, cudaHostRegisterDefault);
	cudaHostGetDevicePointer( &tmpPinnedPtr, cpu_image.real_values, 0);

	precheck
	cudaErr(cudaMemcpyAsync(tmpPinnedPtr, real_values_gpu, real_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
	postcheck

	if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
	// TODO add asserts etc.
	if (free_gpu_memory) { cudaFree(real_values_gpu) ; } // FIXME what about the other structures

	cudaHostUnregister(tmpPinnedPtr);

}

void GpuImage::CopyDeviceToNewHost(Image &cpu_image, bool should_block_until_complete, bool free_gpu_memory)
{
  cpu_image.logical_x_dimension = dims.x;
  cpu_image.logical_y_dimension = dims.y;
  cpu_image.logical_z_dimension = dims.z;
  cpu_image.padding_jump_value = dims.w - dims.x;

  cpu_image.physical_upper_bound_complex_x = physical_upper_bound_complex.x;
  cpu_image.physical_upper_bound_complex_y = physical_upper_bound_complex.y;
  cpu_image.physical_upper_bound_complex_z = physical_upper_bound_complex.z;

  cpu_image.physical_address_of_box_center_x = physical_address_of_box_center.x;
  cpu_image.physical_address_of_box_center_y = physical_address_of_box_center.y;
  cpu_image.physical_address_of_box_center_z = physical_address_of_box_center.z;

  // when copied to GPU image, this is set to 0. not sure if required to be copied.
  //cpu_image.physical_index_of_first_negative_frequency_x = physical_index_of_first_negative_frequency.x;

  cpu_image.physical_index_of_first_negative_frequency_y = physical_index_of_first_negative_frequency.y;
  cpu_image.physical_index_of_first_negative_frequency_z = physical_index_of_first_negative_frequency.z;

  cpu_image.logical_upper_bound_complex_x = logical_upper_bound_complex.x;
  cpu_image.logical_upper_bound_complex_y = logical_upper_bound_complex.y;
  cpu_image.logical_upper_bound_complex_z = logical_upper_bound_complex.z;

  cpu_image.logical_lower_bound_complex_x = logical_lower_bound_complex.x;
  cpu_image.logical_lower_bound_complex_y = logical_lower_bound_complex.y;
  cpu_image.logical_lower_bound_complex_z = logical_lower_bound_complex.z;

  cpu_image.logical_upper_bound_real_x = logical_upper_bound_real.x;
  cpu_image.logical_upper_bound_real_y = logical_upper_bound_real.y;
  cpu_image.logical_upper_bound_real_z = logical_upper_bound_real.z;

  cpu_image.logical_lower_bound_real_x = logical_lower_bound_real.x;
  cpu_image.logical_lower_bound_real_y = logical_lower_bound_real.y;
  cpu_image.logical_lower_bound_real_z = logical_lower_bound_real.z;

  cpu_image.is_in_real_space = is_in_real_space;

  cpu_image.number_of_real_space_pixels = number_of_real_space_pixels;
  cpu_image.object_is_centred_in_box = object_is_centred_in_box;

  cpu_image.fourier_voxel_size_x = fourier_voxel_size.x;
  cpu_image.fourier_voxel_size_y = fourier_voxel_size.y;
  cpu_image.fourier_voxel_size_z = fourier_voxel_size.z;

  cpu_image.insert_into_which_reconstruction = insert_into_which_reconstruction;

  // cpu_image.real_values = real_values_gpu;

  // cpu_image.complex_values = complex_values_gpu;

  cpu_image.is_in_memory = is_in_memory;
  //cpu_image.padding_jump_value = padding_jump_value;
  cpu_image.image_memory_should_not_be_deallocated = image_memory_should_not_be_deallocated;

  //cpu_image.real_memory_allocated = real_memory_allocated;
  cpu_image.ft_normalization_factor = ft_normalization_factor;

  CopyDeviceToHost(cpu_image, should_block_until_complete, free_gpu_memory);

  cpu_image.is_in_memory = true;
  cpu_image.is_in_real_space = is_in_real_space;
}

Image GpuImage::CopyDeviceToNewHost(bool should_block_until_complete, bool free_gpu_memory)
{
  Image new_cpu_image;
  new_cpu_image.Allocate(dims.x, dims.y, dims.z, true, false);

  //new_cpu_image.Allocate(dims.x,dims.y);
  CopyDeviceToNewHost(new_cpu_image, should_block_until_complete, free_gpu_memory);
  return new_cpu_image;
}

void GpuImage::CopyVolumeHostToDevice()
{


  // FIXME not working
    bool is_working = false;
    MyAssertTrue(is_working, "CopyVolumeHostToDevice is not properly worked out");

		d_pitchedPtr = { 0 };
		d_extent = make_cudaExtent(dims.x * sizeof(float), dims.y, dims.z);
		cudaErr(cudaMalloc3D(&d_pitchedPtr, d_extent)); // Complex values need to be pointed
    this->real_values_gpu = (cufftReal *)d_pitchedPtr.ptr; // Set the values here

		d_3dparams        = { 0 };
		d_3dparams.srcPtr = make_cudaPitchedPtr((void*)real_values, dims.x * sizeof(float), dims.x, dims.y);
		d_3dparams.dstPtr = d_pitchedPtr;
		d_3dparams.extent = d_extent;
		d_3dparams.kind   = cudaMemcpyHostToDevice;
		cudaErr(cudaMemcpy3D(&d_3dparams));

}

void GpuImage::CopyVolumeDeviceToHost(bool free_gpu_memory, bool unpin_host_memory)
{

  // FIXME not working
    bool is_working = false;
    MyAssertTrue(is_working, "CopyVolumeDeviceToHost is not properly worked out");

    if ( ! is_in_memory )
    {
		  cudaErr(cudaMallocHost(&real_values, real_memory_allocated*sizeof(float)));
    }
    h_pitchedPtr = make_cudaPitchedPtr((void*)real_values, dims.x * sizeof(float), dims.x, dims.y);
		h_extent = make_cudaExtent(dims.x * sizeof(float), dims.y, dims.z);
		h_3dparams        = { 0 };
		h_3dparams.srcPtr = d_pitchedPtr;
		h_3dparams.dstPtr = h_pitchedPtr;
		h_3dparams.extent = h_extent;
		h_3dparams.kind   = cudaMemcpyDeviceToHost;
		cudaErr(cudaMemcpy3D(&h_3dparams));

    is_in_memory = true;

    // TODO add asserts etc.
    if (free_gpu_memory) 
    { cudaFree(d_pitchedPtr.ptr) ; } // FIXME what about the other structures
    if (unpin_host_memory && is_host_memory_pinned) 
    {
      cudaHostUnregister(real_values);
      is_host_memory_pinned = false;
    }
}

void GpuImage::ForwardFFT(bool should_scale)
{

	bool is_half_precision = false;

	MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
	MyAssertTrue(is_in_real_space, "Image alread in Fourier space");

	if ( ! is_fft_planned )
	{
		SetCufftPlan();
	}

	// For reference to clear cufftXtClearCallback(cufftHandle lan, cufftXtCallbackType type);
	if ( is_half_precision && ! is_set_convertInputf16Tof32 )
	{
		cufftCallbackLoadR h_ConvertInputf16Tof32Ptr;
		cudaErr(cudaMemcpyFromSymbol(&h_ConvertInputf16Tof32Ptr,d_ConvertInputf16Tof32Ptr, sizeof(h_ConvertInputf16Tof32Ptr)));
		cudaErr(cufftXtSetCallback(cuda_plan_forward, (void **)&h_ConvertInputf16Tof32Ptr, CUFFT_CB_LD_REAL, 0));
		is_set_convertInputf16Tof32 = true;
		//	  cudaErr(cudaFree(norm_factor));
		//	  this->MultiplyByConstant(ft_normalization_factor*ft_normalization_factor);
	}
	if (should_scale)
	{
		this->MultiplyByConstant(ft_normalization_factor*ft_normalization_factor);
	}

//	if (should_scale && ! is_set_scaleFFTAndStore)
//	{
//
//		float ft_norm_sq = ft_normalization_factor*ft_normalization_factor;
//		cudaErr(cudaMalloc((void **)&d_scale_factor, sizeof(float)));
//		cudaErr(cudaMemcpyAsync(d_scale_factor, &ft_norm_sq, sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
//		cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
//
//		cufftCallbackStoreC h_scaleFFTAndStorePtr;
//		cudaErr(cudaMemcpyFromSymbol(&h_scaleFFTAndStorePtr,d_scaleFFTAndStorePtr, sizeof(h_scaleFFTAndStorePtr)));
//		cudaErr(cufftXtSetCallback(cuda_plan_forward, (void **)&h_scaleFFTAndStorePtr, CUFFT_CB_ST_COMPLEX, (void **)&d_scale_factor));
//		is_set_scaleFFTAndStore = true;
//	}


//	BufferInit(b_image);
//    cudaErr(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)image_buffer->complex_values));

    cudaErr(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)complex_values_gpu));

    is_in_real_space = false;
	npp_ROI = npp_ROI_fourier_space;



}

void GpuImage::ForwardFFTAndClipInto(GpuImage &image_to_insert, bool should_scale)
{

	MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
	MyAssertTrue(image_to_insert.is_in_memory_gpu, "Gpu memory in image to insert not allocated");
	MyAssertTrue(is_in_real_space, "Image alread in Fourier space");
	MyAssertTrue(image_to_insert.is_in_real_space, "I in image to insert alread in Fourier space");


	if ( ! is_fft_planned )
	{
		SetCufftPlan();
	}

	// For reference to clear cufftXtClearCallback(cufftHandle lan, cufftXtCallbackType type);
	if ( ! is_set_realLoadAndClipInto )
	{

		// We need to make the mask
		image_to_insert.ClipIntoReturnMask(this);

		cufftCallbackLoadR h_realLoadAndClipInto;
		CB_realLoadAndClipInto_params* d_params;
		CB_realLoadAndClipInto_params h_params;

		h_params.target = (cufftReal *)image_to_insert.real_values_gpu;
		h_params.mask   = (int *)clip_into_mask;

		cudaErr(cudaMalloc((void **)&d_params,sizeof(CB_realLoadAndClipInto_params)));
		cudaErr(cudaMemcpyAsync(d_params, &h_params, sizeof(CB_realLoadAndClipInto_params), cudaMemcpyHostToDevice, cudaStreamPerThread));
		cudaErr(cudaMemcpyFromSymbol(&h_realLoadAndClipInto,d_realLoadAndClipInto, sizeof(h_realLoadAndClipInto)));
		cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

		cudaErr(cufftXtSetCallback(cuda_plan_forward, (void **)&h_realLoadAndClipInto, CUFFT_CB_LD_REAL, (void **)&d_params));
		is_set_realLoadAndClipInto = true;


		//	  cudaErr(cudaFree(norm_factor));
		//	  this->MultiplyByConstant(ft_normalization_factor*ft_normalization_factor);
	}
	if (should_scale)
	{
		this->MultiplyByConstant(ft_normalization_factor*ft_normalization_factor);
	}




//	BufferInit(b_image);
//    cudaErr(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)image_buffer->complex_values));

    cudaErr(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)complex_values_gpu));

    is_in_real_space = false;
    npp_ROI = npp_ROI_fourier_space;


}

void GpuImage::BackwardFFT()
{

	MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
	MyAssertFalse(is_in_real_space, "Image is already in real space");

	if ( ! is_fft_planned )
	{
		SetCufftPlan();
	}

	//  BufferInit(b_image);
	//  cudaErr(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)image_buffer->complex_values, (cufftReal*)real_values_gpu));

	cudaErr(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)complex_values_gpu, (cufftReal*)real_values_gpu));

	is_in_real_space = true;
	npp_ROI = npp_ROI_real_space;


}

template < typename T > void GpuImage::BackwardFFTAfterComplexConjMul(T* image_to_multiply, bool load_half_precision)
{

	MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
	MyAssertFalse(is_in_real_space, "Image is already in real space");
	MyAssertTrue(load_half_precision ? is_allocated_16f_buffer : true,  "FP16 memory is not allocated, but is requested.");

	if ( ! is_fft_planned )
	{
		SetCufftPlan();
	}
	if ( ! is_set_complexConjMulLoad )
	{
		cufftCallbackStoreC h_complexConjMulLoad;
		cufftCallbackStoreR h_mipCCGStore;
		CB_complexConjMulLoad_params<T>* d_params;
		CB_complexConjMulLoad_params<T> h_params;
		h_params.scale = ft_normalization_factor*ft_normalization_factor;
		h_params.target = (T *)image_to_multiply;
		cudaErr(cudaMalloc((void **)&d_params,sizeof(CB_complexConjMulLoad_params<T>)));
		cudaErr(cudaMemcpyAsync(d_params, &h_params, sizeof(CB_complexConjMulLoad_params<T>), cudaMemcpyHostToDevice, cudaStreamPerThread));
		if (load_half_precision)
		{
			cudaErr(cudaMemcpyFromSymbol(&h_complexConjMulLoad,d_complexConjMulLoad_16f, sizeof(h_complexConjMulLoad)));
		}
		else
		{
			cudaErr(cudaMemcpyFromSymbol(&h_complexConjMulLoad,d_complexConjMulLoad_32f, sizeof(h_complexConjMulLoad)));
		}


		cudaErr(cudaMemcpyFromSymbol(&h_mipCCGStore,d_mipCCGAndStorePtr, sizeof(h_mipCCGStore)));
		cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
		cudaErr(cufftXtSetCallback(cuda_plan_inverse, (void **)&h_complexConjMulLoad, CUFFT_CB_LD_COMPLEX, (void **)&d_params));
//		void** fake_params;real_values_16f
		cudaErr(cufftXtSetCallback(cuda_plan_inverse, (void **)&h_mipCCGStore, CUFFT_CB_ST_REAL, (void **)&real_values_16f));

//		d_complexConjMulLoad;
		is_set_complexConjMulLoad = true;
	}

	//  BufferInit(b_image);
	//  cudaErr(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)image_buffer->complex_values, (cufftReal*)real_values_gpu));

	cudaErr(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)complex_values_gpu, (cufftReal*)real_values_gpu));

	is_in_real_space = true;
	npp_ROI = npp_ROI_real_space;

}

template void GpuImage::BackwardFFTAfterComplexConjMul(__half2* image_to_multiply, bool load_half_precision);
template void GpuImage::BackwardFFTAfterComplexConjMul(cufftComplex* image_to_multiply, bool load_half_precision);



void GpuImage::Record()
{
	cudaErr(cudaEventRecord(nppCalcEvent,cudaStreamPerThread));
}
void GpuImage::Wait()
{
	cudaErr(cudaStreamWaitEvent(cudaStreamPerThread, nppCalcEvent, 0));
//  cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
}
void GpuImage::RecordAndWait()
{
	Record();
	Wait();
}

void GpuImage::SwapRealSpaceQuadrants()
{

	MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");

	bool must_fft = false;

	float x_shift_to_apply;
	float y_shift_to_apply;
	float z_shift_to_apply;

	if (is_in_real_space == true)
	{
		must_fft = true;
		ForwardFFT(true);
	}

	if (object_is_centred_in_box == true)
	{
		x_shift_to_apply = float(physical_address_of_box_center.x);
		y_shift_to_apply = float(physical_address_of_box_center.y);
		z_shift_to_apply = float(physical_address_of_box_center.z);
	}
	else
	{
		if (IsEven(dims.x) == true)
		{
			x_shift_to_apply = float(physical_address_of_box_center.x);
		}
		else
		{
			x_shift_to_apply = float(physical_address_of_box_center.x) - 1.0;
		}

		if (IsEven(dims.y) == true)
		{
			y_shift_to_apply = float(physical_address_of_box_center.y);
		}
		else
		{
			y_shift_to_apply = float(physical_address_of_box_center.y) - 1.0;
		}

		if (IsEven(dims.z) == true)
		{
			z_shift_to_apply = float(physical_address_of_box_center.z);
		}
		else
		{
			z_shift_to_apply = float(physical_address_of_box_center.z) - 1.0;
		}
	}


	if (dims.z == 1)
	{
		z_shift_to_apply = 0.0;
	}

	PhaseShift(x_shift_to_apply, y_shift_to_apply, z_shift_to_apply);

	if (must_fft == true) BackwardFFT();


	// keep track of center;
	if (object_is_centred_in_box == true) object_is_centred_in_box = false;
	else object_is_centred_in_box = true;
}




void GpuImage::PhaseShift(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift)
{

	MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");

	bool need_to_fft = false;
	if (is_in_real_space == true)
	{
    wxPrintf("Doing forward fft in phase shift function\n\n");
		ForwardFFT(true);
		need_to_fft = true;
	}

  float3 shifts = make_float3(wanted_x_shift, wanted_y_shift, wanted_z_shift);
  // TODO set the TPB and inline function for grid

  
  dim3 threadsPerBlock(32, 32, 1);
  dim3 gridDims((dims.w/2 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                (dims.y + threadsPerBlock.y - 1) / threadsPerBlock.y, dims.z); 

	precheck
	PhaseShiftKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >(complex_values_gpu, 
                                                        dims, shifts,
                                                        physical_address_of_box_center,
                                                        physical_index_of_first_negative_frequency,
                                                        physical_upper_bound_complex);
  
    postcheck

	if (need_to_fft == true) BackwardFFT();

}

__device__ __forceinline__ float
d_ReturnPhaseFromShift(float real_space_shift, float distance_from_origin, float dimension_size)
{
	return real_space_shift * distance_from_origin * 2.0 * PI / dimension_size;
}

__device__ __forceinline__ void
d_Return3DPhaseFromIndividualDimensions( float phase_x, float phase_y, float phase_z, float2 &angles)
{
	float temp_phase = -phase_x-phase_y-phase_z;
	__sincosf(temp_phase, &angles.y, &angles.x); // To use as cos.x + i*sin.y
}


 __device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_X(int physical_index, 
                                                int logical_x_dimension,
                                                int physical_address_of_box_center_x)
{
//	MyAssertTrue(is_in_memory, "Memory not allocated");
//	MyAssertTrue(physical_index <= physical_upper_bound_complex_x, "index out of bounds");

    //if (physical_index >= physical_index_of_first_negative_frequency_x)
    if (physical_index > physical_address_of_box_center_x)
    {
    	 return physical_index - logical_x_dimension;
    }
    else return physical_index;
}


 __device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(int physical_index,
                                                int logical_y_dimension,
                                                int physical_index_of_first_negative_frequency_y )
{
//	MyAssertTrue(is_in_memory, "Memory not allocated");
//	MyAssertTrue(physical_index <= physical_upper_bound_complex_y, "index out of bounds");

    if (physical_index >= physical_index_of_first_negative_frequency_y)
    {
    	 return physical_index - logical_y_dimension;
    }
    else return physical_index;
}


 __device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(int physical_index,

                                                int logical_z_dimension,
                                                int physical_index_of_first_negative_frequency_z )
{
//	MyAssertTrue(is_in_memory, "Memory not allocated");
//	MyAssertTrue(physical_index <= physical_upper_bound_complex_z, "index out of bounds");

    if (physical_index >= physical_index_of_first_negative_frequency_z)
    {
    	 return physical_index - logical_z_dimension;
    }
    else return physical_index;
}

 __device__ __forceinline__ int
d_ReturnReal1DAddressFromPhysicalCoord(int3 coords, int4 img_dims)
{
	return ( (((int)coords.z*(int)img_dims.y + coords.y) * (int)img_dims.w)  + (int)coords.x) ;
}

 __device__ __forceinline__ int
d_ReturnFourier1DAddressFromPhysicalCoord(int3 wanted_dims, int3 physical_upper_bound_complex)
{
	return ( (int)((physical_upper_bound_complex.y + 1) * wanted_dims.z + wanted_dims.y) *
            (int)(physical_upper_bound_complex.x + 1) + (int)wanted_dims.x );
}

__device__ __forceinline__ int
d_ReturnFourier1DAddressFromLogicalCoord(int wanted_x_coord, int wanted_y_coord, int wanted_z_coord, const int3& dims, const int3& physical_upper_bound_complex)
{


  if (wanted_x_coord >= 0)
  {
    if (wanted_y_coord < 0)
    {
      wanted_y_coord += dims.y;
    }
    if (wanted_z_coord < 0)
    {
      wanted_z_coord += dims.z;
    }
  }
  else
  {
    wanted_x_coord = -wanted_x_coord;

    if (wanted_y_coord > 0)
    {
      wanted_y_coord = dims.y - wanted_y_coord;
    }
    else
    {
      wanted_y_coord = -wanted_y_coord;
    }

    if (wanted_z_coord > 0)
    {
      wanted_z_coord = dims.z - wanted_z_coord;
    }
    else
    {
      wanted_z_coord = -wanted_z_coord;
    }
  }
  return d_ReturnFourier1DAddressFromPhysicalCoord(make_int3(wanted_x_coord, wanted_y_coord, wanted_z_coord), physical_upper_bound_complex);
}


__global__ void ClipIntoRealKernel(cufftReal* real_values_gpu,
                                   cufftReal* other_image_real_values_gpu,
                                   int4 dims, 
                                   int4 other_dims,
                                   int3 physical_address_of_box_center, 
                                   int3 other_physical_address_of_box_center,
                                   int3 wanted_coordinate_of_box_center, 
                                   float wanted_padding_value)
{
  int3 other_coord = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                               blockIdx.y*blockDim.y + threadIdx.y,
                               blockIdx.z);

  int3 coord = make_int3(0, 0, 0); 


  if (other_coord.x < other_dims.x &&
      other_coord.y < other_dims.y &&
      other_coord.z < other_dims.z)
  {

    coord.z = physical_address_of_box_center.z + wanted_coordinate_of_box_center.z + 
              other_coord.z - other_physical_address_of_box_center.z;

    coord.y = physical_address_of_box_center.y + wanted_coordinate_of_box_center.y + 
              other_coord.y - other_physical_address_of_box_center.y;

    coord.x = physical_address_of_box_center.x + wanted_coordinate_of_box_center.x + 
              other_coord.x - other_physical_address_of_box_center.x;

    if (coord.z < 0 || coord.z >= dims.z || 
        coord.y < 0 || coord.y >= dims.y ||
        coord.x < 0 || coord.x >= dims.x)
    {
      other_image_real_values_gpu[ d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims) ] = wanted_padding_value;
    }
    else
    {
      other_image_real_values_gpu[ d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims) ] = 
                  real_values_gpu[ d_ReturnReal1DAddressFromPhysicalCoord(coord, dims) ];
    }





  }


}

__global__ void ClipIntoMaskKernel(
                                   int* mask_values,
                                   int4 dims,
                                   int4 other_dims,
                                   int3 physical_address_of_box_center,
                                   int3 other_physical_address_of_box_center,
                                   int3 wanted_coordinate_of_box_center,
                                   float wanted_padding_value)
{

  int3 other_coord = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                               blockIdx.y*blockDim.y + threadIdx.y,
                               blockIdx.z);

  int3 coord = make_int3(0, 0, 0);

  if (other_coord.x < other_dims.x &&
      other_coord.y < other_dims.y &&
      other_coord.z < other_dims.z)
  {

    coord.z = physical_address_of_box_center.z + wanted_coordinate_of_box_center.z +
              other_coord.z - other_physical_address_of_box_center.z;

    coord.y = physical_address_of_box_center.y + wanted_coordinate_of_box_center.y +
              other_coord.y - other_physical_address_of_box_center.y;

    coord.x = physical_address_of_box_center.x + wanted_coordinate_of_box_center.x +
              other_coord.x - other_physical_address_of_box_center.x;

    if (coord.z < 0 || coord.z >= dims.z ||
        coord.y < 0 || coord.y >= dims.y ||
        coord.x < 0 || coord.x >= dims.x)
    {
    	// Assumes that the pixel value at pixel 0 should be zero too.
    	mask_values[ d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims) ] = (int)0;
    }
    else
    {
    	mask_values[ d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims) ] = d_ReturnReal1DAddressFromPhysicalCoord(coord, dims) ;
    }





  }


}
__global__ void PhaseShiftKernel(cufftComplex* d_input, 
                                 int4 dims, float3 shifts, 
                                 int3 physical_address_of_box_center, 
                                 int3 physical_index_of_first_negative_frequency,
                                 int3 physical_upper_bound_complex)
{
	
// FIXME it probably makes sense so just just a linear grid launch and save the extra indexing
  int3 wanted_dims = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                               blockIdx.y*blockDim.y + threadIdx.y,
                               blockIdx.z);

  float2 init_vals;
  float2 angles;

// FIXME This should probably use cuBlas
  if (wanted_dims.x <= physical_upper_bound_complex.x && 
      wanted_dims.y <= physical_upper_bound_complex.y && 
      wanted_dims.z <= physical_upper_bound_complex.z)
  {
  
    
    d_Return3DPhaseFromIndividualDimensions( d_ReturnPhaseFromShift(
                                              shifts.x, 
                                              wanted_dims.x,
                                              dims.x), 
                                             d_ReturnPhaseFromShift(
                                              shifts.y, 
                                                d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(
                                                  wanted_dims.y,
                                                  dims.y,
                                                  physical_index_of_first_negative_frequency.y), 
                                              dims.y),
                                             d_ReturnPhaseFromShift(
                                              shifts.z, 
                                              d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(
                                                wanted_dims.z,
                                                dims.z,
                                                physical_index_of_first_negative_frequency.z), 
                                              dims.z),
                                             angles);
    
    int address = d_ReturnFourier1DAddressFromPhysicalCoord(wanted_dims, physical_upper_bound_complex);
    init_vals.x = d_input[ address ].x;
    init_vals.y = d_input[ address ].y;
    d_input[ address ].x = init_vals.x*angles.x - init_vals.y*angles.y;
    d_input[ address ].y = init_vals.x*angles.y + init_vals.y*angles.x;
  }
  

}


// If you don't want to clip from the center, you can give wanted_coordinate_of_box_center_{x,y,z}. This will define the pixel in the image at which other_image will be centered. (0,0,0) means center of image. This is a dumbed down version that does not fill with noise.
void GpuImage::ClipInto(GpuImage *other_image, float wanted_padding_value,                 
                        bool fill_with_noise, float wanted_noise_sigma,
                        int wanted_coordinate_of_box_center_x, 
                        int wanted_coordinate_of_box_center_y, 
                        int wanted_coordinate_of_box_center_z)
{

	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(other_image->is_in_memory_gpu, "Other image Memory not allocated");
	MyAssertTrue(is_in_real_space, "Clip into is only set up for real space on the gpu currently");


  int3 wanted_coordinate_of_box_center = make_int3(wanted_coordinate_of_box_center_x, 
                                                   wanted_coordinate_of_box_center_y, 
                                                   wanted_coordinate_of_box_center_z);

	other_image->is_in_real_space = is_in_real_space;
	other_image->object_is_centred_in_box = object_is_centred_in_box;

	if (is_in_real_space == true)
	{

		MyAssertTrue(object_is_centred_in_box, "real space image, not centred in box");

    ReturnLaunchParamters(other_image->dims, true);

	precheck
    ClipIntoRealKernel<< <gridDims, threadsPerBlock, 0, cudaStreamPerThread>> >(real_values_gpu,
                                                              other_image->real_values_gpu,
                                                              dims, 
                                                              other_image->dims,
                                                              physical_address_of_box_center,
                                                              other_image->physical_address_of_box_center, 
                                                              wanted_coordinate_of_box_center, 
                                                              wanted_padding_value);
    postcheck
  }

}


void GpuImage::ClipIntoReturnMask(GpuImage *other_image)
{

	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Clip into is only set up for real space on the gpu currently");


  int3 wanted_coordinate_of_box_center = make_int3(0,0,0);


	other_image->is_in_real_space = is_in_real_space;
	other_image->object_is_centred_in_box = object_is_centred_in_box;

	cudaErr(cudaMalloc(&other_image->clip_into_mask, sizeof(int)*other_image->real_memory_allocated));
	other_image->is_allocated_clip_into_mask = true;

	if (is_in_real_space == true)
	{

		MyAssertTrue(object_is_centred_in_box, "real space image, not centred in box");

    ReturnLaunchParamters(other_image->dims, true);

	precheck
    ClipIntoMaskKernel<< <gridDims, threadsPerBlock, 0, cudaStreamPerThread>> >(other_image->clip_into_mask,
                                                              dims,
                                                              other_image->dims,
                                                              physical_address_of_box_center,
                                                              other_image->physical_address_of_box_center,
                                                              wanted_coordinate_of_box_center,
                                                              0.0f);
    postcheck
  }

}

void GpuImage::QuickAndDirtyWriteSlices(std::string filename, int first_slice, int last_slice)
{

	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
  Image buffer_img;
  buffer_img.Allocate(dims.x, dims.y, dims.z, true);

  buffer_img.is_in_real_space = is_in_real_space;
  buffer_img.object_is_centred_in_box = object_is_centred_in_box;
  // Implicitly waiting on work to finish since copy is queued in the stream
  cudaErr(cudaMemcpy((void*)buffer_img.real_values,(const void*)real_values_gpu, real_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost));
  bool OverWriteSlices = true;
  float pixelSize = 0.0f;

  buffer_img.QuickAndDirtyWriteSlices(filename, first_slice, last_slice, OverWriteSlices, pixelSize);
  buffer_img.Deallocate();
}

void GpuImage::SetCufftPlan(bool use_half_precision)
{

    int rank;
    long long int* fftDims;
    long long int* inembed;
    long long int* onembed;


    cudaErr(cufftCreate(&cuda_plan_forward));
    cudaErr(cufftCreate(&cuda_plan_inverse));

    cudaErr(cufftSetStream(cuda_plan_forward, cudaStreamPerThread));
    cudaErr(cufftSetStream(cuda_plan_inverse, cudaStreamPerThread));

    if (dims.z > 1) 
    { 
      rank = 3;
      fftDims = new long long int[rank];
      inembed = new long long int[rank];
      onembed = new long long int[rank];
  
      fftDims[0] = dims.z;
      fftDims[1] = dims.y;
      fftDims[2] = dims.x;

      inembed[0] = dims.z;
      inembed[1] = dims.y;
      inembed[2] = dims.w; // Storage dimension (padded)

      onembed[0] = dims.z;
      onembed[1] = dims.y;
      onembed[2] = dims.w/2; // Storage dimension (padded)

      
    }
    else if (dims.y > 1) 
    { 
      rank = 2;
      fftDims = new long long int[rank];
      inembed = new long long int[rank];
      onembed = new long long int[rank];

      fftDims[0] = dims.y;
      fftDims[1] = dims.x;

      inembed[0] = dims.y;
      inembed[1] = dims.w;

      onembed[0] = dims.y;
      onembed[1] = dims.w/2;

    }    
    else 
    { 
      rank = 1; 
      fftDims = new long long int[rank];
      inembed = new long long int[rank];
      onembed = new long long int[rank];
  
      fftDims[0] = dims.x;

      inembed[0] = dims.w;
      onembed[0] = dims.w/2;
    }



    int iBatch = 1;

// As far as I can tell, the padded layout must be assumed and onembed/inembed
// are not needed. TODO ask John about this.

    if (use_half_precision)
    {
    	cudaErr(cufftXtMakePlanMany(cuda_plan_forward, rank, fftDims,
    				  NULL, NULL, NULL, CUDA_R_16F,
    				  NULL, NULL, NULL, CUDA_C_16F, iBatch, &cuda_plan_worksize_forward, CUDA_C_16F));
    	cudaErr(cufftXtMakePlanMany(cuda_plan_inverse, rank, fftDims,
    				  NULL, NULL, NULL, CUDA_C_16F,
    				  NULL, NULL, NULL, CUDA_R_16F, iBatch, &cuda_plan_worksize_inverse, CUDA_R_16F));
    }
    else
    {
    	cudaErr(cufftXtMakePlanMany(cuda_plan_forward, rank, fftDims,
    				  NULL, NULL, NULL, CUDA_R_32F,
    				  NULL, NULL, NULL, CUDA_C_32F, iBatch, &cuda_plan_worksize_forward, CUDA_C_32F));
    	cudaErr(cufftXtMakePlanMany(cuda_plan_inverse, rank, fftDims,
    				  NULL, NULL, NULL, CUDA_C_32F,
    				  NULL, NULL, NULL, CUDA_R_32F, iBatch, &cuda_plan_worksize_inverse, CUDA_R_32F));
    }

//    cufftPlanMany(&dims.cuda_plan_forward, rank, fftDims, 
//                  inembed, iStride, iDist, 
//                  onembed, oStride, oDist, CUFFT_R2C, iBatch);
//    cufftPlanMany(&dims.cuda_plan_inverse, rank, fftDims, 
//                  onembed, oStride, oDist, 
//                  inembed, iStride, iDist, CUFFT_C2R, iBatch);

 

    delete [] fftDims;
    delete [] inembed;
    delete [] onembed;

	is_fft_planned = true;

  }  



void GpuImage::Deallocate()
{

  if (is_host_memory_pinned)
	{
		cudaErr(cudaHostUnregister(real_values));
		is_host_memory_pinned = false;
	} 
	if (is_in_memory_gpu) 
	{
		cudaErr(cudaFree(real_values_gpu));
		cudaErr(cudaFree(tmpVal));
		cudaErr(cudaFree(tmpValComplex));
		is_in_memory_gpu = false;
	}	

	BufferDestroy();


  if (is_fft_planned)
  {
    cudaErr(cufftDestroy(cuda_plan_inverse));
    cudaErr(cufftDestroy(cuda_plan_forward));
    is_fft_planned = false;
		is_set_complexConjMulLoad = false;
  }

//  if (is_cublas_loaded)
//  {
//    cudaErr(cublasDestroy(cublasHandle));
//    is_cublas_loaded = false;
//  }

  if (is_allocated_mask_CSOS)
  {
    mask_CSOS->Deallocate();
		delete mask_CSOS;
  }

  if (is_allocated_image_buffer)
  {
    image_buffer->Deallocate();
		delete image_buffer;
  }

  if (is_allocated_clip_into_mask) 
	{
		cudaErr(cudaFree(clip_into_mask));
		delete clip_into_mask;
	}

}

void GpuImage::ConvertToHalfPrecision(bool deallocate_single_precision)
{

	// FIXME when adding real space complex images.
	// FIXME should probably be called COPYorConvert
	MyAssertTrue( is_in_memory_gpu, "Image is in not on the GPU!");

	BufferInit(b_16f);

	precheck
	if (is_in_real_space)
	{
		ReturnLaunchParamters(dims, true);
		ConvertToHalfPrecisionKernelReal<< <gridDims, threadsPerBlock,0, cudaStreamPerThread>> > (real_values_gpu, real_values_16f, this->dims);
	}
	else
	{
		ReturnLaunchParamters(dims, false);
		ConvertToHalfPrecisionKernelComplex<< <gridDims, threadsPerBlock,0, cudaStreamPerThread>> > (complex_values_gpu, complex_values_16f, this->dims, this->physical_upper_bound_complex);
	}
	postcheck

	if (deallocate_single_precision)
	{
		cudaErr(cudaFree(real_values_gpu));
		is_in_memory_gpu = false;
	}
}


__global__ void ConvertToHalfPrecisionKernelReal(cufftReal* real_32f_values, __half* real_16f_values, int4 dims)
{

	int3 coords = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
						  blockIdx.y*blockDim.y + threadIdx.y,
						  blockIdx.z);

	if (coords.x < dims.w  && coords.y < dims.y && coords.z < dims.z)
	{

		int address = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);

		real_16f_values[address] = __float2half_rn(real_32f_values[address]);
	}


}

__global__ void ConvertToHalfPrecisionKernelComplex(cufftComplex* complex_32f_values, __half2* complex_16f_values, int4 dims, int3 physical_upper_bound_complex)
{

	int3 coords = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
						  blockIdx.y*blockDim.y + threadIdx.y,
						  blockIdx.z);

	if (coords.x < dims.w / 2 && coords.y < dims.y && coords.z < dims.z)
	{

		int address = d_ReturnFourier1DAddressFromPhysicalCoord(coords, physical_upper_bound_complex);

		complex_16f_values[address] = __float22half2_rn(complex_32f_values[address]);
	}


}
void GpuImage::Allocate(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool should_be_in_real_space)
{

	MyAssertTrue(wanted_x_size > 0 && wanted_y_size > 0 && wanted_z_size > 0,"Bad dimensions: %i %i %i\n",wanted_x_size,wanted_y_size,wanted_z_size);

	// check to see if we need to do anything?

	if (is_in_memory_gpu == true)
	{
		is_in_real_space = should_be_in_real_space;
		if (wanted_x_size == dims.x && wanted_y_size == dims.y && wanted_z_size == dims.z)
		{
			// everything is already done..
			is_in_real_space = should_be_in_real_space;
	//			wxPrintf("returning\n");
			return;
		}
		else
		{
		  Deallocate();
		}
	}

	SetupInitialValues();
	this->is_in_real_space = should_be_in_real_space;
	dims.x = wanted_x_size; dims.y = wanted_y_size; dims.z = wanted_z_size;

	// if we got here we need to do allocation..

	// first_x_dimension
	if (IsEven(wanted_x_size) == true) real_memory_allocated =  wanted_x_size / 2 + 1;
	else real_memory_allocated = (wanted_x_size - 1) / 2 + 1;

	real_memory_allocated *= wanted_y_size * wanted_z_size; // other dimensions
	real_memory_allocated *= 2; // room for complex

	// TODO consider option to add host mem here. For now, just do gpu mem.
	//////	real_values = (float *) fftwf_malloc(sizeof(float) * real_memory_allocated);
	//////	complex_values = (std::complex<float>*) real_values;  // Set the complex_values to point at the newly allocated real values;
//	wxPrintf("\n\n\tAllocating mem\t\n\n");
	cudaErr(cudaMalloc(&real_values_gpu, real_memory_allocated*sizeof(cufftReal)));
	complex_values_gpu = (cufftComplex *)real_values_gpu;
	is_in_memory_gpu = true;

	// Update addresses etc..
	UpdateLoopingAndAddressing(wanted_x_size, wanted_y_size, wanted_z_size);

	if (IsEven(wanted_x_size) == true) padding_jump_value = 2;
	else padding_jump_value = 1;

	// record the full length ( pitch / 4 )
	dims.w = dims.x + padding_jump_value;
	pitch = dims.w * sizeof(float);

	number_of_real_space_pixels = int(dims.x) * int(dims.y) * int(dims.z);
	ft_normalization_factor = 1.0 / sqrtf(float(number_of_real_space_pixels));


	// Set other gpu vals

	is_host_memory_pinned = false;
	is_meta_data_initialized = true;

}

void GpuImage::UpdateBoolsToDefault()
{

	is_meta_data_initialized = false;

	is_in_memory = false;
	is_in_real_space = true;
	object_is_centred_in_box = true;
	image_memory_should_not_be_deallocated = false;

	is_in_memory_gpu = false;
	is_host_memory_pinned = false;

	// libraries
	is_fft_planned = false;
//	is_cublas_loaded = false;
	is_npp_loaded = false;

	// Buffers
	is_allocated_image_buffer = false;
	is_allocated_mask_CSOS = false;

	is_allocated_sum_buffer = false;
	is_allocated_min_buffer = false;
	is_allocated_minIDX_buffer = false;
	is_allocated_max_buffer = false;
	is_allocated_maxIDX_buffer = false;
	is_allocated_minmax_buffer = false;
	is_allocated_minmaxIDX_buffer = false;
	is_allocated_mean_buffer = false;
	is_allocated_meanstddev_buffer = false;
	is_allocated_countinrange_buffer = false;
	is_allocated_l2norm_buffer = false;
	is_allocated_dotproduct_buffer = false;
	is_allocated_16f_buffer = false;

	// Callbacks
	is_set_convertInputf16Tof32 = false;
	is_set_scaleFFTAndStore = false;
	is_set_complexConjMulLoad = false;
	is_allocated_clip_into_mask = false;
	is_set_realLoadAndClipInto = false;

}

//!>  \brief  Update all properties related to looping & addressing in real & Fourier space, given the current logical dimensions.

void GpuImage::UpdateLoopingAndAddressing(int wanted_x_size, int wanted_y_size, int wanted_z_size)
{


	dims.x = wanted_x_size;
	dims.y = wanted_y_size;
	dims.z = wanted_z_size;

	physical_address_of_box_center.x = wanted_x_size / 2;
	physical_address_of_box_center.y= wanted_y_size / 2;
	physical_address_of_box_center.z= wanted_z_size / 2;

	physical_upper_bound_complex.x= wanted_x_size / 2;
	physical_upper_bound_complex.y= wanted_y_size - 1;
	physical_upper_bound_complex.z= wanted_z_size - 1;


	//physical_index_of_first_negative_frequency.x= wanted_x_size / 2 + 1;
	if (IsEven(wanted_y_size) == true)
	{
		physical_index_of_first_negative_frequency.y= wanted_y_size / 2;
	}
	else
	{
		physical_index_of_first_negative_frequency.y= wanted_y_size / 2 + 1;
	}

	if (IsEven(wanted_z_size) == true)
	{
		physical_index_of_first_negative_frequency.z= wanted_z_size / 2;
	}
	else
	{
		physical_index_of_first_negative_frequency.z= wanted_z_size / 2 + 1;
	}


    // Update the Fourier voxel size

	fourier_voxel_size.x= 1.0 / double(wanted_x_size);
	fourier_voxel_size.y= 1.0 / double(wanted_y_size);
	fourier_voxel_size.z= 1.0 / double(wanted_z_size);

	// Logical bounds
	if (IsEven(wanted_x_size) == true)
	{
		logical_lower_bound_complex.x= -wanted_x_size / 2;
		logical_upper_bound_complex.x=  wanted_x_size / 2;
	    logical_lower_bound_real.x   = -wanted_x_size / 2;
	    logical_upper_bound_real.x   =  wanted_x_size / 2 - 1;
	}
	else
	{
		logical_lower_bound_complex.x= -(wanted_x_size-1) / 2;
		logical_upper_bound_complex.x=  (wanted_x_size-1) / 2;
		logical_lower_bound_real.x   = -(wanted_x_size-1) / 2;
		logical_upper_bound_real.x    =  (wanted_x_size-1) / 2;
	}


	if (IsEven(wanted_y_size) == true)
	{
	    logical_lower_bound_complex.y= -wanted_y_size / 2;
	    logical_upper_bound_complex.y=  wanted_y_size / 2 - 1;
	    logical_lower_bound_real.y   = -wanted_y_size / 2;
	    logical_upper_bound_real.y   =  wanted_y_size / 2 - 1;
	}
	else
	{
	    logical_lower_bound_complex.y= -(wanted_y_size-1) / 2;
	    logical_upper_bound_complex.y=  (wanted_y_size-1) / 2;
	    logical_lower_bound_real.y   = -(wanted_y_size-1) / 2;
	    logical_upper_bound_real.y    =  (wanted_y_size-1) / 2;
	}

	if (IsEven(wanted_z_size) == true)
	{
		logical_lower_bound_complex.z= -wanted_z_size / 2;
		logical_upper_bound_complex.z=  wanted_z_size / 2 - 1;
		logical_lower_bound_real.z   = -wanted_z_size / 2;
		logical_upper_bound_real.z   =  wanted_z_size / 2 - 1;

	}
	else
	{
		logical_lower_bound_complex.z= -(wanted_z_size - 1) / 2;
		logical_upper_bound_complex.z=  (wanted_z_size - 1) / 2;
		logical_lower_bound_real.z   = -(wanted_z_size - 1) / 2;
		logical_upper_bound_real.z    =  (wanted_z_size - 1) / 2;
	}
}


/* ///////////////////////////////////////
GPU resize from here:
/////////////////////////////////////// */

__global__ void ClipIntoRealSpaceKernel(cufftReal *real_values_gpu,
                                        cufftReal *other_image_real_values_gpu,
                                        int4 dims,
                                        int4 other_dims,
                                        int3 physical_address_of_box_center,
                                        int3 other_physical_address_of_box_center,
                                        int3 wanted_coordinate_of_box_center,
                                        float wanted_padding_value);

  __global__ void ClipIntoRealSpaceKernel(cufftReal *real_values_gpu,
                                        cufftReal *other_image_real_values_gpu,
                                        int4 dims,
                                        int4 other_dims,
                                        int3 physical_address_of_box_center,
                                        int3 other_physical_address_of_box_center,
                                        int3 wanted_coordinate_of_box_center,
                                        float wanted_padding_value)
  {
    int3 other_coord = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z);

    int3 coord = make_int3(0, 0, 0);

    if (other_coord.x < other_dims.x &&
        other_coord.y < other_dims.y &&
        other_coord.z < other_dims.z)
    {

	  coord.z = physical_address_of_box_center.z + wanted_coordinate_of_box_center.z +
	            other_coord.z - other_physical_address_of_box_center.z;

	  coord.y = physical_address_of_box_center.y + wanted_coordinate_of_box_center.y +
	            other_coord.y - other_physical_address_of_box_center.y;

	  coord.x = physical_address_of_box_center.x + wanted_coordinate_of_box_center.x +
	            other_coord.x - other_physical_address_of_box_center.x;

	  if (coord.z < 0 || coord.z >= dims.z ||
	      coord.y < 0 || coord.y >= dims.y ||
	      coord.x < 0 || coord.x >= dims.x)
	  {
	    other_image_real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] = wanted_padding_value;
	  }
	  else
	  {
	    other_image_real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] =
	        real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(coord, dims)];
	  }
    }
  }





__global__ void SetUniformComplexValueKernel(cufftComplex* complex_values_gpu, cufftComplex value, int4 dims);




// __global__ void SetUniformComplexValueKernel(cufftComplex* complex_values_gpu, cufftComplex value, int4 dims)
//   {
//       int3 kernel_coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
//       blockIdx.y * blockDim.y + threadIdx.y,
//       blockIdx.z);
//       long position1d = get_1d_position_from_3d_coords(kernel_coords, dims);
//       complex_values_gpu[position1d] = value;
//   }


__global__ void ClipIntoFourierSpaceKernel(cufftComplex* source_complex_values_gpu,
                                           cufftComplex* destination_complex_values_gpu,
                                           int4 source_dims,
                                           int4 destination_dims,
                                           int3 destination_image_physical_index_of_first_negative_frequency,
                                           int3 source_logical_lower_bound_complex,
                                           int3 source_logical_upper_bound_complex,
                                           int3 source_physical_upper_bound_complex,
                                           int3 destination_physical_upper_bound_complex,
                                           cufftComplex out_of_bounds_value
                                      );
  __global__ void ClipIntoFourierSpaceKernel(cufftComplex* source_complex_values_gpu,
                                             cufftComplex* destination_complex_values_gpu,
                                             int4 source_dims,
                                             int4 destination_dims,
                                             int3 destination_image_physical_index_of_first_negative_frequency,
                                             int3 source_logical_lower_bound_complex,
                                             int3 source_logical_upper_bound_complex,
                                             int3 source_physical_upper_bound_complex,
                                             int3 destination_physical_upper_bound_complex,
                                             cufftComplex out_of_bounds_value
                                            )
  {
    int3 index_coord = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z);

  if (index_coord.y > destination_dims.y ||
      index_coord.z > destination_dims.z ||
      index_coord.x >= destination_dims.w / 2
      ) 
      {
        return;
      }


  int destination_index = d_ReturnFourier1DAddressFromPhysicalCoord(index_coord, destination_physical_upper_bound_complex);

  int3 source_coord = index_coord;

  source_coord.y = d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(source_coord.y, destination_dims.y, destination_image_physical_index_of_first_negative_frequency.y);
  source_coord.z = d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(source_coord.z, destination_dims.z, destination_image_physical_index_of_first_negative_frequency.z);

  if (source_coord.y >= destination_image_physical_index_of_first_negative_frequency.y)  source_coord.y -= destination_dims.y;
  if (source_coord.z >= destination_image_physical_index_of_first_negative_frequency.z)  source_coord.z -= destination_dims.z;

  cufftComplex new_value;
    if (source_coord.x < source_logical_lower_bound_complex.x || 
        source_coord.x > source_logical_upper_bound_complex.x || 
        source_coord.y < source_logical_lower_bound_complex.y || 
        source_coord.y > source_logical_upper_bound_complex.y || 
        source_coord.z < source_logical_lower_bound_complex.z || 
        source_coord.z > source_logical_upper_bound_complex.z) // these can only be true if the destination image has a dimension bigger than the source
                                                              // consider creating a second kenel for just smaller images clipinto...
       {
  
        new_value = out_of_bounds_value;
    
       }
    else 
       {
         int3 physical_address;
          if (source_coord.x >= 0)
              {
                physical_address.x = source_coord.x;

                if (source_coord.y >= 0)
                {
                  physical_address.y = source_coord.y;
                }
                else
                {
                  physical_address.y = source_dims.y + source_coord.y;
                }

                if (source_coord.z >= 0)
                {
                  physical_address.z = source_coord.z;
                }
                else
                {
                  physical_address.z = source_dims.z + source_coord.z;
                }
              }
              else
              {
                physical_address.x = -source_coord.x;

                if (source_coord.y > 0)
                {
                  physical_address.y = source_dims.y - source_coord.y;
                }
                else
                {
                  physical_address.y = -source_coord.y;
                }

                if (source_coord.z > 0)
                {
                  physical_address.z = source_dims.z - source_coord.z;
                }
                else
                {
                  physical_address.z = -source_coord.z;
                }
              }


        int source_index = d_ReturnFourier1DAddressFromPhysicalCoord(physical_address, source_physical_upper_bound_complex);
        new_value = source_complex_values_gpu[source_index];


       }

      destination_complex_values_gpu[destination_index] = new_value;
  }




void GpuImage::Resize(int wanted_x_dimension, int wanted_y_dimension, int wanted_z_dimension, float wanted_padding_value)
{
  MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
  MyDebugAssertTrue(wanted_x_dimension != 0 && wanted_y_dimension != 0 && wanted_z_dimension != 0, "Resize dimension is zero");

  // this is never called for some reason?
  MyDebugAssertTrue(is_in_real_space, "Resize is only set up for real space on the gpu currently");

  if (dims.x == wanted_x_dimension && dims.y == wanted_y_dimension && dims.z == wanted_z_dimension)
  {
	wxPrintf("Wanted dimensions are the same as current dimensions.\n");
	return;
  }

  // wxPrintf("Init temp GPUImage.\n");
  GpuImage temp_image;

  // wxPrintf("Allocating memory.\n");
  temp_image.Allocate(wanted_x_dimension, wanted_y_dimension, wanted_z_dimension, is_in_real_space);


  // wxPrintf("Clipping temp image.\n");
  precheck
  if (is_in_real_space) {
    // wxPrintf("Clipping real space.\n");
    ClipIntoRealSpace(&temp_image, wanted_padding_value, false, 1.0, 0, 0, 0);
  }
  else {
    // wxPrintf("Clipping Fourier space.\n");
    ClipIntoFourierSpace(&temp_image, wanted_padding_value);
  }
  //ClipInto(&temp_image, wanted_padding_value, false, 1.0, 0, 0, 0);
  postcheck



  // wxPrintf("Consuming temp image\n");
  Consume(temp_image);

}


/*
 Overwrite current GpuImage with a new image, then deallocate new image.
*/
void GpuImage::Consume(GpuImage &temp_image) // copy the parameters then directly steal the memory of another image, leaving it an empty shell
{
  MyDebugAssertTrue(temp_image.is_in_memory_gpu, "Other image Memory not allocated");

  if (this == &temp_image)
  { // no need to consume, its the same image.
	  return;
  }
  UpdateBoolsToDefault();

  is_in_real_space = temp_image.is_in_real_space;

  number_of_real_space_pixels = temp_image.number_of_real_space_pixels;

  insert_into_which_reconstruction = temp_image.insert_into_which_reconstruction;

  object_is_centred_in_box = temp_image.object_is_centred_in_box;

  dims = temp_image.dims;

  // FIXME: temp for comp
  logical_x_dimension = temp_image.dims.x;
  logical_y_dimension = temp_image.dims.y;
  logical_z_dimension = temp_image.dims.z;

  pitch = temp_image.pitch;

  physical_upper_bound_complex = temp_image.physical_upper_bound_complex;

  physical_address_of_box_center = temp_image.physical_address_of_box_center;

  physical_index_of_first_negative_frequency = temp_image.physical_index_of_first_negative_frequency;

  fourier_voxel_size = temp_image.fourier_voxel_size;

  logical_upper_bound_complex = temp_image.logical_upper_bound_complex;

  logical_lower_bound_complex = temp_image.logical_lower_bound_complex;

  logical_upper_bound_real = temp_image.logical_upper_bound_real;

  logical_lower_bound_real = temp_image.logical_lower_bound_real;

  padding_jump_value = temp_image.padding_jump_value;

  ft_normalization_factor = temp_image.ft_normalization_factor;

  real_values = NULL;
  complex_values = NULL;
//   cudaErr(cudaFree(real_values_gpu));
//   cudaErr(cudaFree(complex_values_gpu));
  real_memory_allocated = temp_image.real_memory_allocated;


  is_in_memory = false; //temp_image.is_in_memory;
  is_meta_data_initialized = true;
  precheck
  Deallocate();
  postcheck
  //Allocate(dims.x, dims.y, dims.z, is_in_real_space);
  
  // temp_image.printVal("Value 1:", 1);
  
  if (is_in_real_space) {
    precheck
    Zeros();
    postcheck

    precheck
    cudaErr(cudaMemcpyAsync(real_values_gpu, temp_image.real_values_gpu, sizeof(cufftReal) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    postcheck
  }
  else {
    precheck
    // why doesn't this work?? ---> //cudaErr(cudaMemcpyAsync(complex_values_gpu, temp_image.complex_values_gpu, sizeof(cufftComplex) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    cudaErr(cudaMemcpyAsync(complex_values_gpu, temp_image.complex_values_gpu, sizeof(cufftReal) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    postcheck
  }
  


  precheck
  cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
  postcheck

  // this results in core dump...
  //wxPrintf(" After copy [0] = %s %s\n", std::to_string(real_values_gpu[0]), std::to_string(temp_image.real_values_gpu[0]));
  
  precheck
  is_in_memory_gpu = true;
  postcheck

  // complex values are not dealt with

  

  precheck
  cudaMallocManaged(&tmpVal, sizeof(float));
  cudaMallocManaged(&tmpValComplex, sizeof(double));
  postcheck

  precheck
  temp_image.Deallocate();
  postcheck
}


void GpuImage::ClipIntoRealSpace(GpuImage *other_image, float wanted_padding_value,
                                bool fill_with_noise, float wanted_noise_sigma,
                                int wanted_coordinate_of_box_center_x,
                                int wanted_coordinate_of_box_center_y,
                                int wanted_coordinate_of_box_center_z)
{

  MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
  MyDebugAssertTrue(other_image->is_in_memory_gpu, "Other image Memory not allocated");
  MyDebugAssertTrue(is_in_real_space, "Clip into is only set up for real space on the gpu currently");

  int3 wanted_coordinate_of_box_center = make_int3(wanted_coordinate_of_box_center_x,
                                                   wanted_coordinate_of_box_center_y,
                                                   wanted_coordinate_of_box_center_z);

  other_image->is_in_real_space = is_in_real_space;
  other_image->object_is_centred_in_box = object_is_centred_in_box;

  MyDebugAssertTrue(object_is_centred_in_box, "real space image, not centred in box");

  ReturnLaunchParamters(other_image->dims, true);

  precheck
  ClipIntoRealKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values_gpu,
                                                        other_image->real_values_gpu,
                                                        dims,
                                                        other_image->dims,
                                                        physical_address_of_box_center,
                                                        other_image->physical_address_of_box_center,
                                                        wanted_coordinate_of_box_center,
                                                        wanted_padding_value);
  postcheck

}

void GpuImage::ClipIntoFourierSpace(GpuImage *destination_image, float wanted_padding_value)
{
  MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
  MyAssertTrue(destination_image->is_in_memory_gpu, "Destination image memory not allocated");
  MyAssertTrue(destination_image->object_is_centred_in_box && object_is_centred_in_box, "ClipInto assumes both images are centered at the moment.");
  MyAssertFalse(is_in_real_space && destination_image->is_in_real_space, "ClipIntoFourierSpace assumes both images are in fourier space");
  
  destination_image->object_is_centred_in_box = object_is_centred_in_box;


  ReturnLaunchParamters(destination_image->dims, false);

  precheck
  ClipIntoFourierSpaceKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values_gpu,
                                                                                    destination_image->complex_values_gpu,
                                                                                    dims,
                                                                                    destination_image->dims,
                                                                                    destination_image->physical_index_of_first_negative_frequency,
                                                                                    logical_lower_bound_complex,
                                                                                    logical_upper_bound_complex,
                                                                                    physical_upper_bound_complex,
                                                                                    destination_image->physical_upper_bound_complex,
                                                                                    make_cuComplex(wanted_padding_value, 0.0)
                                                                                    );

  postcheck
  cudaStreamSynchronize(cudaStreamPerThread);

}
