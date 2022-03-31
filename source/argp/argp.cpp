
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
    gParams params;

    hcp::apps::argp::parseAndgetParams(argc, argv, params);

    // printParser();

    // print all params
    params.print();
}
