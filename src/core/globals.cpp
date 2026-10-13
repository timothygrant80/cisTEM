#include "core_headers.h"

// The seed of the global generator comes from the clock unless cisTEM_RANDOM_SEED names one (a non-negative
// integer), which makes a run reproducible: the random starting angles, the random in-plane offsets and the
// random subsets drawn from this generator then repeat, so two builds can be compared on the same data.
static int InitialGlobalSeed( ) {
    const char* seed = getenv("cisTEM_RANDOM_SEED");
    if ( seed != NULL && seed[0] != '\0' ) {
        int value = atoi(seed);
        if ( value >= 0 )
            return value;
    }
    return -1;
}

RandomNumberGenerator global_random_number_generator(InitialGlobalSeed( ));
