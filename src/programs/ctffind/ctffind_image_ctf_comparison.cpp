#include "../../core/core_headers.h"
#include "./ctffind.h"

ImageCTFComparison::ImageCTFComparison(int wanted_number_of_images, CTF wanted_ctf, float wanted_pixel_size, bool should_find_phase_shift, bool wanted_astigmatism_is_known, float wanted_known_astigmatism, float wanted_known_astigmatism_angle, bool should_fit_defocus_sweep) {
    MyDebugAssertTrue(wanted_number_of_images > 0, "Bad wanted number of images: %i\n", wanted_number_of_images);
    number_of_images = wanted_number_of_images;
    img              = new Image[wanted_number_of_images];

    ctf                       = wanted_ctf;
    pixel_size                = wanted_pixel_size;
    find_phase_shift          = should_find_phase_shift;
    astigmatism_is_known      = wanted_astigmatism_is_known;
    known_astigmatism         = wanted_known_astigmatism;
    known_astigmatism_angle   = wanted_known_astigmatism_angle;
    fit_defocus_sweep         = should_fit_defocus_sweep;
    azimuths                  = NULL;
    spatial_frequency_squared = NULL;
    addresses                 = NULL;
    number_to_correlate       = 0;
    image_mean                = 0.0;
    norm_image                = 0.0;
}

ImageCTFComparison::~ImageCTFComparison( ) {
    for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        img[image_counter].Deallocate( );
    }
    delete[] img;
    delete[] azimuths;
    delete[] spatial_frequency_squared;
    delete[] addresses;
    number_to_correlate = 0;
}

void ImageCTFComparison::SetImage(int wanted_image_number, Image* new_image) {
    MyDebugAssertTrue(wanted_image_number >= 0 && wanted_image_number < number_of_images, "Wanted image number (%i) is out of bounds", wanted_image_number);
    img[wanted_image_number].CopyFrom(new_image);
}

void ImageCTFComparison::SetCTF(CTF new_ctf) {
    ctf = new_ctf;
}

void ImageCTFComparison::SetFitWithThicknessNodes(bool wanted_fit_with_thickness_nodes) {
    fit_with_thickness_nodes = wanted_fit_with_thickness_nodes;
}

void ImageCTFComparison::SetupQuickCorrelation( ) {
    img[0].SetupQuickCorrelationWithCTF(ctf, number_to_correlate, norm_image, image_mean, NULL, NULL, NULL);
    azimuths                  = new float[number_to_correlate];
    spatial_frequency_squared = new float[number_to_correlate];
    addresses                 = new int[number_to_correlate];
    img[0].SetupQuickCorrelationWithCTF(ctf, number_to_correlate, norm_image, image_mean, addresses, spatial_frequency_squared, azimuths);
}

CTF ImageCTFComparison::ReturnCTF( ) { return ctf; }

bool ImageCTFComparison::AstigmatismIsKnown( ) { return astigmatism_is_known; }

float ImageCTFComparison::ReturnKnownAstigmatism( ) { return known_astigmatism; }

float ImageCTFComparison::ReturnKnownAstigmatismAngle( ) { return known_astigmatism_angle; }

bool ImageCTFComparison::FindPhaseShift( ) { return find_phase_shift; }
