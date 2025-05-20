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
    // TODO: it might be nice to add some contraints on the template parameters.
    template <typename T_in, typename T_out = int>
    inline T_out GetPoissonRandomSTD(T_in mean_value) { return std::poisson_distribution<T_out>{mean_value}(rng); }

    template <typename T>
    inline T GetUniformRandomSTD(T min_value, T max_value) {
        if constexpr ( std::is_integral<T>::value ) {
            return std::uniform_int_distribution<T>{min_value, max_value}(rng);
        }
        else {
            return std::uniform_real_distribution<T>{min_value, max_value}(rng);
        }
        return 0;
    }

    template <typename T>
    inline T GetNormalRandomSTD(T mean_value, T std_deviation) { return std::normal_distribution<T>{mean_value, std_deviation}(rng); }

    template <typename T>
    inline T GetExponentialRandomSTD(T lambda) { return std::exponential_distribution<T>{lambda}(rng); }

    template <typename T>
    inline T GetGammaRandomSTD(T alpha, T beta) { return std::gamma_distribution<T>{alpha, beta}(rng); }

  private:
    std::random_device   rd;
    typedef std::mt19937 MyRng;
    MyRng                rng;
};
