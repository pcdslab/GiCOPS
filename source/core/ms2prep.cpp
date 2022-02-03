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

// external query filenames
extern std::vector<string_t> queryfiles;

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
inline status_t barrier()
{
    status_t status = SLM_SUCCESS;

#ifdef USE_MPI
    status = MPI_Barrier(MPI_COMM_WORLD);
#endif // USE_MPI

    return status;
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

    // synchronize
    status = hcp::mpi::barrier();

    // write index to file
    MSQuery::write_index();

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
    int_t nfiles = queryfiles.size();

    int_t cputhreads = 1; // params.threads; FIXME

#if 1 // defined (GPU) && defined (CUDA)

    int_t gputhreads = 0; // 1 FIXME
    cputhreads -= gputhreads;

#endif // GPU && CUDA

    // get the ptrs instance
    MSQuery **ptrs = get_instance();

    if (ptrs == nullptr)
        status = ERR_INVLD_PTR;
    else
    {
        /* Initialize the queue with already created nfiles */
        *qfPtrs = new lwqueue<MSQuery*>(nfiles, false);

        // initialize the ptrs instances
        for (auto lptr = ptrs; lptr < ptrs + nfiles; lptr++)
            *lptr = new MSQuery;

        // get local partition size
        auto pfiles = hcp::mpi::getPartitionSize(nfiles);

        // initialize the MSQuery index
        bool summaryExists = MSQuery::init_index();

        // if pfiles > 0 and !summaryExists
        if (pfiles && !summaryExists)
        {
            // fill the vector with locally processed MS2 file indices
            std::vector<int_t> ms2local(pfiles);

            // first element is params.myid
            ms2local[0] = params.myid;

            // rest of the files in cyclic order
            std::generate(std::begin(ms2local) + 1, std::end(ms2local), 
                          [n=params.myid] () mutable { return n += params.nodes; });

#if 1 // defined (GPU) && defined (CUDA)

            // initialize a thread-safe MSQuery queue for CPU + GPU processing
            std::queue<int_t> ms2QueueIdx;
            std::mutex ms2QueueMutex;

            // fill with correct file indices
            for (auto loc_fid : ms2local)
                ms2QueueIdx.push(loc_fid);

            auto workerthread = [&](bool gpu)
            {
                auto loc_fid = 0;

                while (true)
                {
                    bool breakloop = false;

                    // code block for unique_lock
                    {
                        std::unique_lock<std::mutex> lock(ms2QueueMutex);

                        if (ms2QueueIdx.empty())
                            breakloop = true;
                        else
                        {
                            loc_fid = ms2QueueIdx.front();
                            ms2QueueIdx.pop();
                        }
                    }

                    // break the loop here
                    if (breakloop)
                        break;

                    // check if working with GPU
                    if (gpu)
                    {
                        // TODO: GPU kernel is needed here.
                        // hcp::gpu::cuda::s2::initialize(ptrs[loc_fid], &queryfiles[loc_fid], loc_fid);
                        ptrs[loc_fid]->initialize(&queryfiles[loc_fid], loc_fid);
                    }
                    else
                        ptrs[loc_fid]->initialize(&queryfiles[loc_fid], loc_fid);
                
                // archive the index if using MPI
#ifdef USE_MPI
                    ptrs[loc_fid]->archive(loc_fid);
#endif // USE_MPI
                }
            };

            workerthread(false);
#if 0
            // launch threads on GPU and CPU
            std::vector<std::thread> wThreads(cputhreads + gputhreads);

            for (int gth = 0 ; gth < gputhreads; gth++)
                wThreads.emplace_back(std::thread(workerthread, true));

            for (int cth = 0 ; cth < cputhreads; cth++)
                wThreads.emplace_back(std::thread(workerthread, false));

            // make sure all GPU threads are done
            for (auto& wth : wThreads)
                wth.join();

            wThreads.clear();
#endif 


#else

#ifdef USE_OMP
#pragma omp parallel for schedule (dynamic, 1)
#endif/* _OPENMP */
            for (auto fid = 0; fid < pfiles; fid++)
            {
                auto loc_fid = ms2local[fid];
                ptrs[loc_fid]->initialize(&queryfiles[loc_fid], loc_fid);
#ifdef USE_MPI
                ptrs[loc_fid]->archive(loc_fid);
#endif // USE_MPI
            }

#endif // GPU && CUDA
    
            // in case of one node, we need to write in order
            if (params.nodes == 1)
            {
                for (auto fid = 0; fid < pfiles; fid++)
                {
                    auto loc_fid = ms2local[fid];
                    ptrs[loc_fid]->archive(loc_fid);
                }
            }

            // clear ms2local vector
            ms2local.clear();
        }

        // synchronize
        if (params.nodes > 1)
            status = hcp::ms2::synchronize();

        //
        // Synchronize global data summary
        //

        // if more than one nodes or summary file exists
        if (params.nodes > 1 || summaryExists)
        {
            info_t *findex = new info_t[nfiles];

            MSQuery::read_index(findex, nfiles);

            // copy global data from the index
            for (auto fid = 0; fid < nfiles; fid ++)
            {
                // use the index information to
                // initialize the remaining index
                if (!ptrs[fid]->isinit())
                {
                    ptrs[fid]->Info() = findex[fid];
                    ptrs[fid]->vinitialize(&queryfiles[fid], fid);
                }
            }

            delete[] findex;
        }

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
            std::cout << "Dataset Size = " << dssize << std::endl << std::endl;
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
