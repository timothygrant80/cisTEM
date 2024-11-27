#include "../../core/core_headers.h"
#include "../../core/scattering_potential.h"

class
        QuickTestApp : public MyApp {

  public:
    bool     DoCalculation( );
    void     DoInteractiveUserInput( );
    wxString symmetry_symbol;
    bool     my_test_1 = false;
    bool     my_test_2 = true;

    std::array<wxString, 2> input_starfile_filename;

  private:
};

IMPLEMENT_APP(QuickTestApp)

// override the DoInteractiveUserInput

void QuickTestApp::DoInteractiveUserInput( ) {

    std::cerr << "here" << std::endl;
    UserInput* my_input = new UserInput("Unblur", 2.0);

    input_starfile_filename.at(0) = my_input->GetFilenameFromUser("Input starfile filename 1", "", "", false);
    input_starfile_filename.at(1) = my_input->GetFilenameFromUser("Input starfile filename 2", "", "", false);
    symmetry_symbol               = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");

    delete my_input;
}

// override the do calculation method which will be what is actually run..

template <typename T>
inline void welford_update(T& mean, T& var_x_n1, T& n, T& value) {
    n += 1.0;
    T delta = value - mean;
    mean += delta / n;
    T delta2 = value - mean;
    var_x_n1 += delta * delta2;
}

// The regular update is just a special version with nb = 1 mean_update = value and var_x_n1_update = 0
template <typename T>
inline void welford_parallel_update(T& mean_a, const T mean_update, T& var_x_n1, const T var_x_n1_update, T& n, const T n_update) {

    T mean_ab = (n * mean_a + n_update * mean_update) / (n + n_update);
    var_x_n1  = var_x_n1 + var_x_n1_update + (n * (n + n_update) / n_update) * powf((mean_ab - mean_a), 2);
}

template <typename T>
inline void check_outlier(const T m2, const T n, const T variance_threshold) {
    if ( m2 / (n - 1) > variance_threshold ) {
        std::cerr << "Outlier detected: " << value << std::endl;
    }
}

bool QuickTestApp::DoCalculation( ) {

    // Large array of double
    std::vector<double> values(10000000, 0.0);
    std::vector<float>  values_float(10000000, 0.0);

    // Get random normal for the values
    RandomNumberGenerator my_rand(pi_v<float>);

    const int n_loops = 100;
    double    mean_d  = 0.0;
    double    var_d   = 0.0;
    double    mean_f  = 0.0;
    double    var_f   = 0.0;
    double    mean_w  = 0.0;
    double    var_w   = 0.0;
    double    mean_b{ };
    double    var_b{ };
    double    mean_k{ };
    double    var_k{ };

    for ( int iloop = 0; iloop < n_loops; iloop++ ) {
        std::cerr << "Loop: " << iloop << std::endl;
        for ( int i = 0; i < values.size( ); i++ ) {
            values.at(i)       = my_rand.GetNormalRandomSTD(0.0, 1.0);
            values_float.at(i) = float(values.at(i));
        }

        // Use a standard empirical distribution
        EmpiricalDistribution<double> my_dist;
        EmpiricalDistribution<float>  my_dist_float;
        EmpiricalDistribution<float>  my_dist_float_welford;
        EmpiricalDistribution<float>  my_dist_float_welford_batched;
        EmpiricalDistribution<float>  my_dist_float_kahan;

        for ( auto& value : values ) {
            my_dist.AddSampleValue(value);
        }

        mean_d += my_dist.GetSampleMean( );
        var_d += my_dist.GetSampleVariance( );

        my_dist_float.Reset( );
        for ( auto& value : values_float ) {
            my_dist_float.AddSampleValue(value);
        }

        mean_f += my_dist_float.GetSampleMean( );
        var_f += my_dist_float.GetSampleVariance( );

        // Use a Welford accumulator
        my_dist_float.Reset( );
        for ( auto& value : values_float ) {
            my_dist_float_welford.AddSampleValueForWelford(value);
        }
        mean_w += my_dist_float_welford.GetSampleMean( );
        var_w += my_dist_float_welford.GetUnbiasedEstimateOfPopulationVariance( );

        // Use a batched Welford accumulator
        float mean_batch{ };
        float var_batch{ };
        float n_batch = 20.0f;
        float counter = 0.0f;
        for ( int i = 0; i < values_float.size( ); i += n_batch ) {
            mean_batch += values_float.at(i);
            var_batch += values_float.at(i) * values_float.at(i);
            counter += 1.0f;
            if ( counter == n_batch ) {
                my_dist_float_welford_batched.AddSampleValueForWelfordBatched(mean_batch, var_batch - mean_batch * mean_batch, n_batch);
                mean_batch = 0.0f;
                var_batch  = 0.0f;
                counter    = 0.0f;
            }
        }
        mean_b += my_dist_float_welford.GetSampleMean( );
        var_b += my_dist_float_welford.GetUnbiasedEstimateOfPopulationVariance( );

        for ( int i = 0; i < values_float.size( ); i++ ) {
            my_dist_float_kahan.AddSampleValueWithKahanCorrection(values_float.at(i));
        }
        mean_k += my_dist_float_kahan.GetSampleMean( );
        var_k += my_dist_float_kahan.GetUnbiasedEstimateOfPopulationVariance( );
    }

    // Print out the results and relative erross
    std::cerr << "Mean and variance for double: " << mean_d / n_loops << " " << var_d / n_loops << std::endl;
    std::cerr << "Mean and variance for float: " << mean_f / n_loops << " " << var_f / n_loops << " error : " << 100. * (var_f / n_loops - var_d / n_loops) / var_d / n_loops << std::endl;
    std::cerr << "Mean and variance for Welford: " << mean_w / n_loops << " " << var_w / n_loops << " error : " << 100. * (var_w / n_loops - var_d / n_loops) / var_d / n_loops << std::endl;
    std::cerr << "Mean and variance for Welford batched: " << mean_b / n_loops << " " << var_b / n_loops << " error : " << 100. * (var_b / n_loops - var_d / n_loops) / var_d / n_loops << std::endl;
    std::cerr << "Mean and variance for Kahan: " << mean_k / n_loops << " " << var_k / n_loops << " error : " << 100. * (var_k / n_loops - var_d / n_loops) / var_d / n_loops << std::endl;

    return true;
}
