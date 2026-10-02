#include "../../core/core_headers.h"
#include "../../../include/catch2/catch.hpp"

// The directional (conical) statistics behind ReconstructedVolume::Calculate3DOptimalDirectional.

static void fill_random(Image& image, RandomNumberGenerator& rng, float scale) {
    for ( long i = 0; i < image.real_memory_allocated; i++ )
        image.real_values[i] = rng.GetNormalRandom( ) * scale;
}

TEST_CASE("Cone directions partition the sphere, Friedel mates alike", "[ResolutionStatistics]") {
    ResolutionStatistics statistics(1.0f, 32);
    statistics.SetupCones(36);
    REQUIRE(statistics.number_of_cones == 36);
    REQUIRE(statistics.cone_directions.size( ) == 3 * 36);
    for ( int c = 0; c < 36; c++ ) {
        float x = statistics.cone_directions[3 * c], y = statistics.cone_directions[3 * c + 1], z = statistics.cone_directions[3 * c + 2];
        REQUIRE(fabsf(x * x + y * y + z * z - 1.0f) < 1e-4f); // unit vectors
        REQUIRE(z >= 0.0f); // on the hemisphere
        REQUIRE(statistics.ReturnConeIndex(x, y, z) == c); // each direction finds itself
        REQUIRE(statistics.ReturnConeIndex(-x, -y, -z) == c); // and so does its Friedel mate
    }
    REQUIRE(statistics.ReturnConeIndex(0.0f, 0.0f, 0.0f) == 0);
    // Every cone receives some directions: count assignments over a grid of directions.
    std::vector<int> counts(36, 0);
    for ( int a = 0; a < 40; a++ )
        for ( int b = 0; b < 40; b++ ) {
            float theta = PI * (a + 0.5f) / 40.0f, phi = 2.0f * PI * (b + 0.5f) / 40.0f;
            counts[statistics.ReturnConeIndex(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta))]++;
        }
    for ( int c = 0; c < 36; c++ )
        REQUIRE(counts[c] > 0);
}

TEST_CASE("Conical FSC of identical volumes is one, of independent noise is small, and cones bracket the shell", "[ResolutionStatistics]") {
    const int             box = 32;
    RandomNumberGenerator rng(17, true);
    Image                 a, b;
    a.Allocate(box, box, box, true);
    b.Allocate(box, box, box, true);
    fill_random(a, rng, 1.0f);
    b.CopyFrom(&a);
    a.ForwardFFT( );
    b.ForwardFFT( );

    ResolutionStatistics statistics(1.0f, box);
    statistics.CalculateFSC(a, b, false);
    statistics.CalculateConicalFSC(a, b, 12, false);
    REQUIRE(statistics.number_of_cones == 12);
    for ( int c = 0; c < 12; c++ ) {
        REQUIRE(statistics.cone_FSC[c].NumberOfPoints( ) == statistics.FSC.NumberOfPoints( ));
        for ( int i = 1; i < statistics.number_of_bins; i++ )
            REQUIRE(statistics.cone_FSC[c].data_y[i] == Approx(1.0f).margin(1e-4));
        REQUIRE(statistics.ReturnConeEstimatedResolution(c) == Approx(2.0f)); // Nyquist at 1 A/px
    }

    // Independent noise: the shell FSC is near zero, and so is every cone's, within the noise of fewer voxels.
    Image c2;
    c2.Allocate(box, box, box, true);
    fill_random(c2, rng, 1.0f);
    c2.ForwardFFT( );
    ResolutionStatistics noise(1.0f, box);
    noise.CalculateFSC(a, c2, false);
    noise.CalculateConicalFSC(a, c2, 12, false);
    for ( int i = 4; i < noise.number_of_bins; i++ ) {
        REQUIRE(fabsf(noise.FSC.data_y[i]) < 0.2f);
        float lo = 1.0f, hi = -1.0f;
        for ( int c = 0; c < 12; c++ ) {
            lo = std::min(lo, noise.cone_FSC[c].data_y[i]);
            hi = std::max(hi, noise.cone_FSC[c].data_y[i]);
        }
        REQUIRE(lo <= noise.FSC.data_y[i] + 1e-4f); // the shell value lies within the cones' spread
        REQUIRE(hi >= noise.FSC.data_y[i] - 1e-4f);
    }
}

TEST_CASE("Conical particle SSNR is capped at the shell and floored at a fraction of it", "[ResolutionStatistics]") {
    const int             box = 32;
    RandomNumberGenerator rng(5, true);
    Image                 a, b;
    a.Allocate(box, box, box, true);
    b.Allocate(box, box, box, true);
    fill_random(a, rng, 1.0f);
    fill_random(b, rng, 1.0f);
    b.AddImage(&a); // correlated halves: FSC about 0.7
    a.ForwardFFT( );
    b.ForwardFFT( );
    ResolutionStatistics statistics(1.0f, box);
    statistics.CalculateFSC(a, b, false);
    statistics.CalculateConicalFSC(a, b, 8, false);
    // A reconstruction with uniform sampling: every voxel weight one.
    Image sampling;
    sampling.Allocate(box, box, box, false);
    std::vector<float> ctf_sum(sampling.real_memory_allocated / 2, 1.0f);
    statistics.CalculateParticleSSNR(sampling, ctf_sum.data( ), 1.0f);
    statistics.CalculateConicalParticleSSNR(sampling, ctf_sum.data( ), 1.0f, 0.05f);
    for ( int c = 0; c < 8; c++ ) {
        REQUIRE(statistics.cone_part_SSNR[c].NumberOfPoints( ) == statistics.part_SSNR.NumberOfPoints( ));
        for ( int i = 1; i < statistics.number_of_bins; i++ ) {
            float shell = statistics.part_SSNR.data_y[i], cone = statistics.cone_part_SSNR[c].data_y[i];
            REQUIRE(cone <= shell + 1e-6f);
            REQUIRE(cone >= 0.05f * shell - 1e-6f);
        }
    }
}
