#include <cistem_config.h>

#ifdef _CISTEM_MODULES

import hello;
import hello2;

int main(void) {
    hello::greeter( );

    hello2::greeter( );
    return 0;
}

#endif