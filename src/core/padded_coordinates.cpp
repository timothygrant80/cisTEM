#include "core_headers.h"

#include "../constants/constants.h"

Coords::Coords( ) {
    is_set_specimen        = false;
    is_set_fft_padding     = false;
    is_set_solvent_padding = false;
    largest_specimen       = make_int3(0, 0, 0);
}

Coords::~Coords( ) {
    // DO NOTHING
}

#ifndef ENABLEGPU

int3 Coords::make_int3(int x, int y, int z) {
    int3 retVal;
    retVal.x = x;
    retVal.y = y;
    retVal.z = z;
    return retVal;
}

float3 Coords::make_float3(float x, float y, float z) {
    float3 retVal;
    retVal.x = x;
    retVal.y = y;
    retVal.z = z;
    return retVal;
}

#endif

void Coords::SetSpecimenVolume(int nx, int ny, int nz) {
    // Update with the current specimen dimensions
    specimen = make_int3(nx, ny, nz);
    if ( ! is_set_specimen ) {
        is_set_specimen = true;
    }
    // Keep a running record of the largest specimen dimensions encountered yet
    SetLargestSpecimenVolume(nx, ny, nz);
}

int3 Coords::GetSpecimenVolume( ) {
    CheckVectorIsSet(is_set_specimen);
    return specimen;
}

// The minimum specimen dimensions can vary depending on the current orientation. We need to track the largest dimension for output
void Coords::SetLargestSpecimenVolume(int nx, int ny, int nz) {
    CheckVectorIsSet(is_set_specimen);
    largest_specimen = make_int3(std::max(specimen.x, nx),
                                 std::max(specimen.y, ny),
                                 std::max(specimen.z, nz));
}

int3 Coords::GetLargestSpecimenVolume( ) {
    CheckVectorIsSet(is_set_specimen);
    return largest_specimen;
}

void Coords::SetSolventPadding(int nx, int ny, int nz) {
    solvent_padding        = make_int3(nx, ny, nz);
    is_set_solvent_padding = true;
}

void Coords::SetSolventPadding_Z(int wanted_nz) {
    MyAssertTrue(is_set_solvent_padding, "Solvent padding must be set in all dimensions before updating the Z dimension");
    solvent_padding.z = wanted_nz;
}

int3 Coords::GetSolventPadding( ) {
    CheckVectorIsSet(is_set_solvent_padding);
    return solvent_padding;
}

void Coords::SetFFTPadding(int max_factor, int pad_by) {

    CheckVectorIsSet(is_set_solvent_padding);

    fft_padding        = make_int3(ReturnClosestFactorizedUpper(solvent_padding.x * pad_by, max_factor, true),
                                   ReturnClosestFactorizedUpper(solvent_padding.y * pad_by, max_factor, true),
                                   (int)0);
    is_set_fft_padding = true;
}

int3 Coords::GetFFTPadding( ) {
    CheckVectorIsSet(is_set_fft_padding);
    return fft_padding;
}

float3 Coords::ReturnOrigin(PaddingStatus status) {
    float3 output;
    switch ( status ) {
            // TODO think about why the 0.5 is needed. It doesn't seem like it should be, but when checking the crosshair pdb it seems to be the best solution to keep everything centered.
        case none:
            output = make_float3(specimen.x / 2 + 0.0f, specimen.y / 2 + 0.0f, specimen.z / 2 + 0.0f);
            break;

        case fft:
            output = make_float3(fft_padding.x / 2 + 0.0f, fft_padding.y / 2 + 0.0f, fft_padding.z / 2 + 0.0f);
            break;

        case solvent:
            output = make_float3(solvent_padding.x / 2 + 0.0f, solvent_padding.y / 2 + 0.0f, solvent_padding.z / 2 + 0.0f);
            break;
    }
    return output;
}

int Coords::ReturnLargestDimension(int dimension) {
    int output;
    switch ( dimension ) {
        case 0:
            output = largest_specimen.x - N_TAPERS * TAPERWIDTH;
            break;

        case 1:
            output = largest_specimen.y - N_TAPERS * TAPERWIDTH;
            break;

        case 2:
            output = largest_specimen.z;
            break;
    }
    if ( IsOdd(output) )
        output += 1;
    return output;
}

void Coords::Allocate(Image* image_to_allocate, PaddingStatus status, bool should_be_in_real_space, bool only_2d) {
    int3 size;
    switch ( status ) {
        case none:
            size = GetSpecimenVolume( );
            if ( only_2d ) {
                size.z = 1;
            }
            // Sets to zero and returns fals if no allocation needed.
            if ( IsAllocationNecessary(image_to_allocate, size) ) {
                image_to_allocate->Allocate(size.x, size.y, size.z, should_be_in_real_space);
            }
            break;

        case fft:
            size = GetFFTPadding( );
            if ( only_2d ) {
                size.z = 1;
            }
            // Sets to zero and returns fals if no allocation needed.
            if ( IsAllocationNecessary(image_to_allocate, size) ) {
                image_to_allocate->Allocate(size.x, size.y, size.z, should_be_in_real_space);
            }
            break;

        case solvent:
            size = GetSolventPadding( );
            if ( only_2d ) {
                size.z = 1;
            }

            // Sets to zero and returns fals if no allocation needed.
            if ( IsAllocationNecessary(image_to_allocate, size) ) {
                image_to_allocate->Allocate(size.x, size.y, size.z, should_be_in_real_space);
            }
            break;
    }
}

void Coords::PadSolventToFFT(Image* image_to_resize) {
    // I expect the edges to already be tapered to 0;
    // The assumption is that we are working in 2d at this point.

    CheckVectorIsSet(is_set_fft_padding);
    image_to_resize->Resize(fft_padding.x, fft_padding.y, 1, 0.0f);
}

void Coords::PadFFTToSolvent(Image* image_to_resize) {
    // The assumption is that we are working in 2d at this point.
    CheckVectorIsSet(is_set_solvent_padding);
    image_to_resize->Resize(solvent_padding.x, solvent_padding.y, 1, 0.0f);
}

void Coords::PadFFTToSpecimen(Image* image_to_resize) {
    CheckVectorIsSet(is_set_specimen);
    image_to_resize->Resize(specimen.x, specimen.y, 1, 0.0f);
}

void Coords::PadToLargestSpecimen(Image* image_to_resize, bool should_be_square) {

    CheckVectorIsSet(is_set_specimen);

    if ( should_be_square ) {
        int sq_dim = std::max(ReturnLargestDimension(0), ReturnLargestDimension(1));
        image_to_resize->Resize(sq_dim, sq_dim, 1);
    }

    else {

        image_to_resize->Resize(ReturnLargestDimension(0), ReturnLargestDimension(1), 1);
    }
}

void Coords::PadToWantedSize(Image* image_to_resize, int wanted_size) {

    if ( wanted_size < 0 ) {
        PadToLargestSpecimen(image_to_resize, true);
    }
    else {
        image_to_resize->Resize(wanted_size, wanted_size, 1);
    }
}

bool Coords::IsAllocationNecessary(Image* image_to_allocate, int3 size) {

    bool allocation_required = true;

    if ( image_to_allocate->is_in_memory == true &&
         size.x == image_to_allocate->logical_x_dimension &&
         size.y == image_to_allocate->logical_y_dimension &&
         size.z == image_to_allocate->logical_z_dimension ) {
        allocation_required = false;
    }

    return allocation_required;
}
