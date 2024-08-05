/*  \brief  RandomNumberGenerator class */
constexpr bool debug_with_constant_seed = false;

class RandomNumberGenerator {

  public:
    int            random_seed; // Seed for all instances of this class even though it is not declared static!
    bool           use_internal; // Use internal implementation of rand and srand to enable multiple PRNGs within a single program
    unsigned int   next_seed; // State variable for internal PRNG
    const uint64_t random_seed_std_routines = std::chrono::high_resolution_clock::now( ).time_since_epoch( ).count( );
    const uint64_t const_seed_std_routines  = 1147;

    // Constructor
    RandomNumberGenerator(bool internal = false);
    RandomNumberGenerator(int random_seed, bool internal = false);

    RandomNumberGenerator(float thread_id) : rng(rd( )) {
        if constexpr ( debug_with_constant_seed ) {
            rng.seed(const_seed_std_routines);
        }
        else {
            rng.seed(random_seed_std_routines);
        }
    }

    // Seed generator
    void SetSeed(int random_seed);

    // Generate random numbers from various distributions
    float GetUniformRandom( ); // Uniform random number [-1,1]
    float GetNormalRandom( ); // Normal random number, sigma = 1
    void  Internal_srand(unsigned int random_seed);
    int   Internal_rand( );

    // Distributions are lightweight, create on the fly
    inline int GetPoissonRandomSTD(float mean_value) { return std::poisson_distribution<int>{mean_value}(rng); }

    inline float GetUniformRandomSTD(float min_value, float max_value) { return std::uniform_real_distribution<float>{min_value, max_value}(rng); }

    inline float GetNormalRandomSTD(float mean_value, float std_deviation) { return std::normal_distribution<float>{mean_value, std_deviation}(rng); }

    inline float GetExponentialRandomSTD(float lambda) { return std::exponential_distribution<float>{lambda}(rng); }

    inline float GetGammaRandomSTD(float alpha, float beta) { return std::gamma_distribution<float>{alpha, beta}(rng); }

  private:
    std::random_device   rd;
    typedef std::mt19937 MyRng;
    MyRng                rng;
};
