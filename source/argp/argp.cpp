
#define ARGP_ONLY
#include "argp.hpp"
#undef ARGP_ONLY

using  namespace std;
using  namespace hcp::apps::argp;

int main(int argc, char *argv[])
{
    //
    // argument parser
    //
    auto hicops_args = hcp::apps::argp::get_instance(argc, argv);

    if (hicops_args.verbose)
        hicops_args.print();

    // printParser();

    gParams params;
    hcp::apps::argp::getParams(params);

    // print all params
    std::cout << std::endl << std::endl << "ARGP: Here are the params that I got" << std::endl <<std::endl;

    params.print();
}
