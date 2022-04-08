#include <cistem_config.h>

#ifdef _CISTEM_MODULES

module;

// #include <iostream>
// #include <string_view>

export module hello;

namespace hello {

// export void greeter(std::string_view const& name) {
//     std::cout << "Hello " << name << "!\n";
// }
export void greeter( ){ };

/**
 * @brief Used in testing module function from cistem program console_test.
 * 
 * @return int 40 
 */
export int test_return_value( ) {
    return 40;
}

} // namespace hello

#endif
