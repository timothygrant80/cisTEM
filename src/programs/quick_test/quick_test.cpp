

#include "../../core/core_headers.h"
#include "../../constants/constants.h"

#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/DeviceManager.h"
#include "../../gpu/TemplateMatchingCore.h"
#include "quick_test_gpu.h"
#else
#include "../../core/core_headers.h"
#endif

#include "../../constants/constants.h"

#include "../../core/scattering_potential.h"

// Toggle which experiment to build — define exactly one:
//#define cisTEM_test_shifts
#define cisTEM_test_astigmatism

class
        QuickTestApp : public MyApp {

  public:
    bool     DoCalculation( );
    void     DoInteractiveUserInput( );
    void     AddCommandLineOptions( );
    wxString symmetry_symbol;
    bool     my_test_1 = false;
    bool     my_test_2 = true;
    int      idx;

    std::array<wxString, 2> input_starfile_filename;

  private:
};

IMPLEMENT_APP(QuickTestApp)

// Optional command-line stuff
void QuickTestApp::AddCommandLineOptions( ) {
    command_line_parser.AddLongSwitch("disable-user-input", "Disable interactive user input prompts. Default false");
}

// override the DoInteractiveUserInput

void QuickTestApp::DoInteractiveUserInput( ) {
    // This flag allows skipping interactive prompts, useful for automated testing with Copilot.
    if ( command_line_parser.FoundSwitch("disable-user-input") ) {
        std::cout << "Skipping interactive user input as per command line flag." << std::endl;
        return;
    }
    UserInput* my_input = new UserInput("QuickTest", 2.0);

    idx                           = my_input->GetIntFromUser("Index", "", "", 0, 1000);
    input_starfile_filename.at(0) = my_input->GetFilenameFromUser("Input starfile filename 1", "", "", false);
    input_starfile_filename.at(1) = my_input->GetFilenameFromUser("Input starfile filename 2", "", "", false);
    symmetry_symbol               = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");

    delete my_input;
}

bool QuickTestApp::DoCalculation( ) {

#ifdef ENABLEGPU
    // DeviceManager gpuDev;
    // gpuDev.ListDevices( );

    // QuickTestGPU quick_test_gpu;3
    // quick_test_gpu.callHelloFromGPU(idx);
#endif

#ifdef cisTEM_test_shifts
    const int vol_size     = 512;
    const int prj_size     = 1024;
    const int n_replicates = 10;
    // vols at  0.5 1.0 2.0 4.0 /scratch/etna/sub_pixel_and_astig_exp/7A4M-assembly1-pixel_size_0.5.mrc

    // Let's actually make an vector of std pairs of pixel_size and peak_file_name
    std::vector<std::pair<float, std::string>> pixel_size_and_peak_file_names;
    pixel_size_and_peak_file_names.push_back(std::make_pair(0.5f, "/scratch/etna/sub_pixel_and_astig_exp/7A4M-assembly1-pixel_size_0.5.mrc"));
    pixel_size_and_peak_file_names.push_back(std::make_pair(1.0f, "/scratch/etna/sub_pixel_and_astig_exp/7A4M-assembly1-pixel_size_1.0.mrc"));
    pixel_size_and_peak_file_names.push_back(std::make_pair(2.0f, "/scratch/etna/sub_pixel_and_astig_exp/7A4M-assembly1-pixel_size_2.0.mrc"));
    pixel_size_and_peak_file_names.push_back(std::make_pair(4.0f, "/scratch/etna/sub_pixel_and_astig_exp/7A4M-assembly1-pixel_size_4.0.mrc"));

    Image prj        = Image(vol_size, vol_size, 1, true);
    Image padded_prj = Image(prj_size, prj_size, 1, true);
    // Over each vol we'll read in, prep for projection using an AnglesAndShifts object, and then project using the Image::ExtractSlice method.
    // Identity projection (no rotation) and we'll loop over shifts from 0.0 to sqrt(2)*0.5 in steps of 0.05 equal in x and y.
    std::vector<float> shifts = {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7};

    for ( int i = 0; i < n_replicates; i++ ) {
        // For each replicate we'll make two noise images, one before and one after the CTF is applied. These will be  size prj and padded_prj.
        Image noise_before_ctf = Image(prj_size, prj_size, 1, true);
        Image noise_after_ctf  = Image(prj_size, prj_size, 1, true);
        noise_before_ctf.FillWithNoise(NoiseType::GAUSSIAN, 0.0f, 1.0f);
        noise_after_ctf.FillWithNoise(NoiseType::GAUSSIAN, 0.0f, 8.0f);

        for ( const auto& pixel_size_and_peak_file_name : pixel_size_and_peak_file_names ) {
            auto peak_file_name = pixel_size_and_peak_file_name.second;
            wxPrintf("Processing peak file: %s\n", peak_file_name.c_str( ));
            float pixel_size = pixel_size_and_peak_file_name.first;
            // For this experiment we'll use a single CTF for all replicates.
            CTF   ctf(300, 2.7, 0.07, 5200, 5000, 45, pixel_size, 0);
            Image vol;
            vol.QuickAndDirtyReadSlices(peak_file_name, 1, 512);
            vol.ForwardFFT( );

            vol.SwapRealSpaceQuadrants( );

            for ( const auto& shift : shifts ) {
                // Print shifts
                wxPrintf("Shift: %f\n", shift);
                AnglesAndShifts angles_and_shifts(0.0, 0.0, 0.0, shift, shift);
                padded_prj.SetToConstant(0.0f);

                vol.ExtractSlice(prj, angles_and_shifts, 0.0, false);
                prj.SwapRealSpaceQuadrants( );
                // Angles and shifts apparently does not shift on projection
                // If the shift is 0.0 add a tiny shift
                float shift_to_use = shift == 0.0f ? 0.005f : shift;

                prj.PhaseShift(shift_to_use, shift_to_use, 0.0f);
                prj.BackwardFFT( );
                prj.AddConstant(-1.0f * prj.ReturnAverageOfRealValuesOnEdges( ));
                prj.ClipInto(&padded_prj);

                padded_prj.ZeroFloatAndNormalize( );
                padded_prj.AddImage(&noise_before_ctf);
                padded_prj.ForwardFFT( );
                padded_prj.ApplyCTF(ctf);
                padded_prj.BackwardFFT( );
                padded_prj.AddImage(&noise_after_ctf);

                // output dir /scratch/etna/sub_pixel_and_astig_exp/pixel_size_4.0
                std::string pixel_size_str = wxString::Format("%.1f", pixel_size).ToStdString( );
                std::string shift_str      = wxString::Format("%.2f", shift).ToStdString( );
                std::string output_dir     = "/scratch/etna/sub_pixel_and_astig_exp/pixel_size_Z_" + pixel_size_str;
                padded_prj.QuickAndDirtyWriteSlice(output_dir + "/test_proj_shift_" + shift_str + "_replicate_" + std::to_string(i) + ".mrc", 1);
            }

        } // end of peak_file_names

    } // end of n_replicates
#endif // cisTEM_test_shifts

#ifdef cisTEM_test_astigmatism
    const int   vol_size     = 512;
    const int   img_size     = 4096;
    const int   n_replicates = 10;
    const float pixel_size   = 0.5f;

    // 8x8 grid of 512x512 cells tiling the 4096x4096 image = 64 particles
    const int   grid_n          = 8;
    const int   cell_size       = img_size / grid_n; // 512, same as vol_size
    const int   particle_radius = 110; // ~110 Ang / 0.5 Ang/pix = 220 px diameter, 110 px radius
    const int   edge_margin     = 10;
    const float max_nudge       = static_cast<float>(cell_size / 2 - particle_radius - edge_margin); // 136 px

    // Grid cell centers relative to image center
    // Cells at indices 0..7, centers at (i+0.5)*512 in image coords
    // Relative to image center (2048): (i+0.5)*512 - 2048 = (i-3.5)*512
    std::vector<std::pair<int, int>> grid_positions;
    for ( int gy = 0; gy < grid_n; gy++ ) {
        for ( int gx = 0; gx < grid_n; gx++ ) {
            int cx = static_cast<int>((gx - 3.5f) * cell_size);
            int cy = static_cast<int>((gy - 3.5f) * cell_size);
            grid_positions.push_back({cx, cy});
        }
    }

    // Defocus conditions: {def1, def2}, all with astigmatism angle = 45
    std::vector<std::pair<float, float>> defocus_conditions = {
            {2000.0f, 2000.0f},
            {4000.0f, 4000.0f},
            {6000.0f, 6000.0f},
            {9000.0f, 9000.0f},
            {12000.0f, 12000.0f},
            {6000.0f, 2000.0f},
            {12000.0f, 6000.0f}};
    const float defocus_angle = 45.0f;

    std::string vol_filename = "/scratch/etna/sub_pixel_and_astig_exp/7A4M-assembly1-pixel_size_0.5.mrc";
    Image       vol;
    vol.QuickAndDirtyReadSlices(vol_filename, 1, vol_size);
    vol.ForwardFFT( );
    vol.SwapRealSpaceQuadrants( );

    Image prj      = Image(vol_size, vol_size, 1, true);
    Image full_img = Image(img_size, img_size, 1, true);

    for ( int i = 0; i < n_replicates; i++ ) {
        Image noise_before_ctf = Image(img_size, img_size, 1, true);
        Image noise_after_ctf  = Image(img_size, img_size, 1, true);
        noise_before_ctf.FillWithNoise(NoiseType::GAUSSIAN, 0.0f, 1.0f);
        noise_after_ctf.FillWithNoise(NoiseType::GAUSSIAN, 0.0f, 4.0f);

        for ( const auto& defocus : defocus_conditions ) {
            wxPrintf("Replicate %d, defocus %.0f/%.0f\n", i, defocus.first, defocus.second);
            CTF ctf(300, 2.7, 0.07, defocus.first, defocus.second, defocus_angle, pixel_size, 0);

            full_img.SetToConstant(0.0f);

            for ( const auto& grid_pos : grid_positions ) {
                // Uniform random rotation on SO(3)
                float phi   = (global_random_number_generator.GetUniformRandom( ) + 1.0f) * 180.0f;
                float theta = rad_2_deg(acosf(global_random_number_generator.GetUniformRandom( )));
                float psi   = (global_random_number_generator.GetUniformRandom( ) + 1.0f) * 180.0f;

                // Random nudge within cell: ±136 px max, applied as phase shift before BackwardFFT
                float nudge_x = global_random_number_generator.GetUniformRandom( ) * max_nudge;
                float nudge_y = global_random_number_generator.GetUniformRandom( ) * max_nudge;

                AnglesAndShifts angles(phi, theta, psi, 0.0f, 0.0f);
                vol.ExtractSlice(prj, angles, 0.0, false);
                prj.SwapRealSpaceQuadrants( );
                prj.PhaseShift(nudge_x, nudge_y, 0.0f);
                prj.BackwardFFT( );
                prj.AddConstant(-1.0f * prj.ReturnAverageOfRealValuesOnEdges( ));

                full_img.InsertOtherImageAtSpecifiedPosition(&prj, grid_pos.first, grid_pos.second, 0);
            }

            full_img.ZeroFloatAndNormalize( );
            full_img.AddImage(&noise_before_ctf);
            full_img.ForwardFFT( );
            full_img.ApplyCTF(ctf);
            full_img.BackwardFFT( );
            full_img.AddImage(&noise_after_ctf);

            int         def1_int   = static_cast<int>(defocus.first);
            int         def2_int   = static_cast<int>(defocus.second);
            std::string def_str    = std::to_string(def1_int) + "_" + std::to_string(def2_int);
            std::string output_dir = "/scratch/etna/sub_pixel_and_astig_exp/astigmatism_def_" + def_str;
            full_img.QuickAndDirtyWriteSlice(output_dir + "/astigmatism_def_" + def_str + "_replicate_" + std::to_string(i) + ".mrc", 1);
        }

    } // end of n_replicates
#endif // cisTEM_test_astigmatism

    return true;
}
