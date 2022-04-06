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
#define ARGP_NOMPI
#include "argp.hpp"
#undef ARGP_NOMPI

using namespace std;

gParams params;

/* FUNCTION: main
 *
 * DESCRIPTION: Counter application
 *
 * INPUT: none
 *
 * OUTPUT
 * @status: Status of execution
 */
status_t main(int_t argc, char_t* argv[])
{
    status_t status = SLM_SUCCESS;

    ull_t cumusize = 0;
    ull_t ions = 0;

    char_t extension[] = ".peps";

    /* Parse the parameters */
    hcp::apps::argp::parseAndgetParams(argc, argv, params);

    // initialize mod information here only once
    InitializeModInfo(&params.vModInfo);

    /* Create local variables to avoid trouble */
    int minlen = params.min_len;
    int maxlen = params.max_len;

    // compute max parallel threads
    int maxthreads = std::max(static_cast<int>(std::thread::hardware_concurrency()), (maxlen-minlen+1));
    int threads = std::max(2, static_cast<int>(std::ceil(static_cast<double>(maxthreads)/params.threads)));

    // cannot use multiple threads as modcounter.cpp has global vars.
#if defined(USE_OMP)
//#pragma omp parallel for num_threads(threads) schedule(dynamic) reduction(+:cumusize,ions)
#endif // defined(USE_OMP)
    for (int peplen = minlen; peplen <= maxlen; peplen++)
    {
        string_t dbfile = params.dbpath + "/" + std::to_string(peplen) + extension;

        /* Count the number of ">" entries in FASTA */
        auto&& ret = DBCounter(dbfile);

        cumusize += std::get<0>(ret);
        ions += std::get<1>(ret);
    }

    /* The only output should be the cumulative size of the index */
    std::cout << "spectra:" << cumusize << std::endl;
	std::cout << "ions:" << ions << std::endl;

    return status;
}
