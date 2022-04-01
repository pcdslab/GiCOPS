/*
 *  Copyright (C) 2019  Muhammad Haseeb, Fahad Saeed
 *  Florida International University, Miami, FL
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include "counter.hpp"

#define ARGP_ONLY
#include "argp.hpp"
#undef ARGP_ONLY


using namespace std;

/* Global Variables */
string_t dbfile;

extern ull_t cumusize;
extern ull_t ions;

gParams params;

/* FUNCTION: SLM_Main (main)
 *
 * DESCRIPTION: Driver Application
 *
 * INPUT: none
 *
 * OUTPUT
 * @status: Status of execution
 */
status_t main(int_t argc, char_t* argv[])
{
    status_t status = SLM_SUCCESS;

    char_t extension[] = ".peps";

    /* Parse the parameters */
    hcp::apps::argp::parseAndgetParams(argc, argv, params);

    /* Create local variables to avoid trouble */
    uint_t minlen = params.min_len;
    uint_t maxlen = params.max_len;

    for (uint_t peplen = minlen; peplen <= maxlen && status == SLM_SUCCESS; peplen++)
    {
        dbfile = params.dbpath + "/" + std::to_string(peplen) + extension;

        /* Count the number of ">" entries in FASTA */
        status = DBCounter(dbfile);
    }

    /* The only output should be the cumulative size of the index */
    std::cout << "spectra:" << cumusize << std::endl;
    std::cout << "ions:" << ions << std::endl;

    return status;
}
