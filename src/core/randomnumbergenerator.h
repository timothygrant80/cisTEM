/*  \brief  RandomNumberGenerator class */

class RandomNumberGenerator {
	
	public:

 	int random_seed; 		// Seed for all instances of this class even though it is not declared static!
 	bool use_internal;		// Use internal implementation of rand and srand to enable multiple PRNGs within a single program
 	unsigned int next_seed;	// State variable for internal PRNG

	// Constructor	
	RandomNumberGenerator(bool internal = false);
	RandomNumberGenerator(int random_seed, bool internal = false);
	
	// Seed generator
	void SetSeed(int random_seed);
	
	// Generate random numbers from various distributions
	float GetUniformRandom(); 	// Uniform random number [-1,1]
	float GetNormalRandom();	// Normal random number, sigma = 1
	void Internal_srand(unsigned int random_seed);
	int Internal_rand();
};
