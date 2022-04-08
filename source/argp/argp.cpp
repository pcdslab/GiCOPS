/*
 * Copyright (C) 2022 Muhammad Haseeb, Fahad Saeed
 * Florida International University, Miami, FL
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */
 
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
