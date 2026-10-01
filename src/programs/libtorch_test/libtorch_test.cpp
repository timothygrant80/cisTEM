#include "../../core/core_headers.h"
#undef N_
#undef UNCHECKED
#include <torch/torch.h>
#include <chrono>

// #include <mkl.h>

class LibTorchTest : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(LibTorchTest)

void LibTorchTest::DoInteractiveUserInput( ) {
    UserInput* my_input = new UserInput("LibTorch Test", 1.00f);
    int        threads  = my_input->GetIntFromUser("Num threads", "Number of threads to use in the operation", "1", 1, 28);

    delete my_input;
    my_current_job.Reset(1);
    my_current_job.ManualSetArguments("i", threads);
}

bool LibTorchTest::DoCalculation( ) {
    // Check the number of threads MKL is using:
    // wxPrintf("%s\n", torch::show_config( ));

    const int threads = my_current_job.arguments[0].ReturnIntegerArgument( );

    torch::set_num_threads(threads);
    torch::set_num_interop_threads(1);
    // wxPrintf("MKL max threads: %i\n", mkl_get_max_threads( ));
    // wxPrintf("ATen max threads: %i\n", torch::get_num_threads( ));
    // wxPrintf("Parallel backend: %s\n", at::get_parallel_info( ).c_str( ));

    wxPrintf("Number of threads: %i\n", threads);

    const int64_t N          = 2048;
    const int     iterations = 40;

    torch::TensorOptions opts =
            torch::TensorOptions( ).dtype(torch::kFloat32);

    torch::Tensor a = torch::rand({N, N}, opts);
    torch::Tensor b = torch::rand({N, N}, opts);

    auto start = std::chrono::high_resolution_clock::now( );

    for ( int i = 0; i < iterations; ++i ) {
        // Scale to avoid overflow
        torch::Tensor c = torch::matmul(a, b) / static_cast<float>(N);

        // Non-trivial extra work
        c = torch::relu(c);
        c = c / c.norm( ); // normalize each iteration

        a = c;
    }

    auto                          end     = std::chrono::high_resolution_clock::now( );
    std::chrono::duration<double> elapsed = end - start;

    wxPrintf("Final tensor norm: %f\n", a.norm( ).item<float>( ));
    wxPrintf("Elapsed time: %lf seconds\n", elapsed.count( ));

    return true;
}
