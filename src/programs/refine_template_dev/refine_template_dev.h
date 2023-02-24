#ifndef _src_programs_refine_template_dev_refine_template_dev_h
#define _src_programs_refine_template_dev_refine_template_dev_h
#include <boost/hana.hpp>
namespace hana = boost::hana;
using namespace hana::literals;

struct cisTEM_job_argument {
    std::string name;
    std::string description;
    std::string default_value;
};

auto refine_template_arguments = hana::make_tuple(cisTEM_job_argument{"input_starfile", "Starfile containing match information", "matches.star"});

#endif // _src_programs_refine_template_dev_refine_template_dev_h