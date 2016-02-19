#include "core_headers.h"

RandomNumberGenerator::RandomNumberGenerator() {
	SetSeed(4711);
}


RandomNumberGenerator::RandomNumberGenerator(int random_seed) {
	SetSeed(random_seed);
}

// Set seed for random number generator
void RandomNumberGenerator::SetSeed(int random_seed) {

	if (random_seed == -1)
	{
		this->random_seed = time(NULL);
	}
	else this->random_seed = random_seed;

	srand((unsigned int) this->random_seed);

}

// Return a random number in the interval [-1,1]
// from a uniform distribution
float RandomNumberGenerator::GetUniformRandom() {
	float rnd1 = (float)rand();
	float hmax = ((float) RAND_MAX) / 2.0;
	float rnd2 = (rnd1 - hmax) / hmax;
	return rnd2;
}

// Return a random number according to the normal distribution using the
// the Box-Muller transformation (polar variant)
// standard normal distribution:
//              p(X) = 1/(2*Pi)^0.5 * exp(-x^2/2)
float RandomNumberGenerator::GetNormalRandom() {
	float x1;
	float x2;
	float R;
	float y1;
	do {
		x1 = GetUniformRandom();
		x2 = GetUniformRandom();
		R = x1 * x1 + x2 * x2;
	} while (R == 0.0 || R > 1.0);
	y1 = x1 * sqrtf(-2.0 * log(R) / R);
	// y2 = x2 * sqrtf(-2.0 * log(R) / R);
	return y1;
}
