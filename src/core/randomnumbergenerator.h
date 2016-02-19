/*  \brief  RandomNumberGenerator class */

class RandomNumberGenerator {
	
	public:
	
 	int random_seed; 	// Seed for all instances of this class even though it is not declared static!
	
	// Constructor	
	RandomNumberGenerator();
	RandomNumberGenerator(int random_seed);
	
	// Seed generator
	void SetSeed(int random_seed);
	
	// Generate random numbers from various distributions
	float GetUniformRandom(); 	// Uniform random number [-1,1]
	float GetNormalRandom();	// Normal random number, sigma = 1
	
};
