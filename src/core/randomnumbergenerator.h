/*  \brief  RandomNumberGenerator class */

class RandomNumberGenerator {
	
	public:

 	int random_seed; 		// Seed for all instances of this class even though it is not declared static!
 	bool use_internal;		// Use internal implementation of rand and srand to enable multiple PRNGs within a single program
 	unsigned int next_seed;	// State variable for internal PRNG
	const uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();



	// Constructor	
	RandomNumberGenerator(bool internal = false);
	RandomNumberGenerator(int random_seed, bool internal = false);
	RandomNumberGenerator(float thread_id) : rng(rd()) {rng.seed(seed);}

	

	// Seed generator
	void SetSeed(int random_seed);
	
	// Generate random numbers from various distributions
	float GetUniformRandom(); 	// Uniform random number [-1,1]
	float GetNormalRandom();	// Normal random number, sigma = 1
	void Internal_srand(unsigned int random_seed);
	int Internal_rand();

	// Distributions are lightweight, create on the fly
	inline int   GetPoissonRandomSTD(float mean_value) { return std::poisson_distribution<int>{mean_value}(rng); }
	inline float GetUniformRandomSTD(float min_value, float max_value) { return std::uniform_real_distribution<float>{min_value, max_value}(rng); }
	inline float GetNormalRandomSTD(float mean_value, float std_deviation) { return std::normal_distribution<float>{mean_value, std_deviation}(rng); }
	inline float GetExponentialRandomSTD(float lambda) { return std::exponential_distribution<float>{lambda}(rng) ;}
	inline float GetGammaRandomSTD(float alpha, float beta) { return std::gamma_distribution<float>{alpha, beta}(rng); }

	private:

		std::random_device rd;
		typedef std::mt19937 MyRng;
		MyRng rng;

};
