#include <cistem_config.h>

#ifdef _CISTEM_MODULES

module;

// #include <iostream>
// #include <string_view>

export module hello2;

namespace hello2 {

// export void greeter(std::string_view const& name) {
//     std::cout << "Hello from 2 " << name << "!\n";
// }

export void greeter( ){ };

/**
 * @brief Used in testing module function from cistem program console_test.
 * 
 * @return int 42 
 */
export int test_return_value( ) {
    return 42;
}

} // namespace hello2

#endif