/*
 * Copyright (C) 2021  Muhammad Haseeb, and Fahad Saeed
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
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include "ms2prep.hpp"

//
// TODO: insert instrumentation for the work performed in this file
// Specially measure the compute to overhead (comm and I/O) ratio.
//

// include cereal to read/archive MS2 index
// #include <cereal/archives/binary.hpp>

// external query filenames
extern vector<string_t> queryfiles;

// extern params
extern gParams params;

namespace hcp
{

namespace mpi
{

//
// FUNCTION: getPartitionSize (templated over numbers)
//
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
T getPartitionSize(T isize)
{
    T loc_size = isize / params.nodes;

    if (params.myid < (isize % params.nodes))
        loc_size++;

    return loc_size;
}

//
// FUNCTION: barrier (wrapper for MPI_Barrier)
//
status_t barrier()
{
    return MPI_Barrier(MPI_COMM_WORLD);
}

//
// TODO: still to implement
//
status_t allgather()
{
    return SLM_SUCCESS;
}

} // namespace mpi

namespace ms2
{

//
// FUNCTION: synchronize
//
status_t synchronize()
{
    status_t status = SLM_SUCCESS;

    // get ptrs
    MSQuery **ptrs = get_instance();

    // MPI synchronize
    status = hcp::mpi::barrier();

    // TODO: synchronous allgather of ptrs
    // How do we gather??

    return status;
}

//
// FUNCTION: get_instance
// 
MSQuery **& get_instance()
{
    static MSQuery** ptrs = new MSQuery*[queryfiles.size()];
    return ptrs;
}

//
// FUNCTION: initialize
// 
status_t initialize(lwqueue<MSQuery *>** qfPtrs, int_t& nBatches, int_t& dssize)
{
    status_t status = SLM_SUCCESS;
    bool_t qfindex = false;

    int_t nfiles = queryfiles.size();

    // get the ptrs instance
    MSQuery **ptrs = get_instance();

    if (ptrs == nullptr)
        status = ERR_INVLD_PTR;

    if (status == SLM_SUCCESS)
    {
        /* Initialize the queue with already created nfiles */
        *qfPtrs = new lwqueue<MSQuery*>(nfiles, false);

        // initialize the ptrs instances
        for (auto lptr = ptrs; lptr < ptrs + nfiles; lptr++)
            *lptr = new MSQuery;

        // get local partition size
        auto pfiles = hcp::mpi::getPartitionSize(nfiles);

        // fill the vector with locally processed MS2 file indices
        // fixme: indices needed not size (single node run produces '1' here)
        std::vector<int_t> ms2local(pfiles);
        std::generate(std::begin(ms2local), std::end(ms2local), 
                      [n=params.myid] () mutable { return n += params.nodes; });

#ifdef USE_OMP
#pragma omp parallel for schedule (dynamic, 1)
#endif/* _OPENMP */
        for (auto fid = 0; fid < pfiles; fid++)
        {
            // fix the ms2local numbers (indices not sizes)
            auto loc_fid = ms2local[fid];
            ptrs[loc_fid]->InitQueryFile(&queryfiles[loc_fid], fid);
        }

        //
        // Synchronize superstep 2
        //
        status = hcp::ms2::synchronize();

        // -------------------------------------------------------------------------------------- //

        //
        // create/read the MS2 index
        //

        // Push zeroth as is
        (*qfPtrs)->push(ptrs[0]);
        dssize += ptrs[0]->getQAcount();

        // Update batch numbers
        for (auto fid = 1; fid < nfiles; fid++)
        {
            ptrs[fid]->Curr_chunk() = ptrs[fid - 1]->Curr_chunk() + ptrs[fid - 1]->Nqchunks();
            (*qfPtrs)->push(ptrs[fid]);
            dssize += ptrs[fid]->getQAcount();
        }

        // Compute the total number of batches in the dataset
        nBatches = ptrs[nfiles-1]->Curr_chunk() + ptrs[nfiles-1]->Nqchunks();

        if (params.myid == 0)
            std::cout << "\nDataset Size = " << dssize << std::endl << std::endl;

        // clear the vector
        ms2local.clear();
    }
    
    return status;
}

//
// FUNCTION: deinitialize
//
void deinitialize()
{
    MSQuery **ptrs = get_instance();

    /* Delete ptrs */
    if (ptrs != nullptr)
    {
        delete[] ptrs;
        ptrs = nullptr;
    }
}

} // namespace ms2
} // namespace hcp
