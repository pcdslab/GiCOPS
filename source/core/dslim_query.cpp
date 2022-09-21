/*
 * Copyright (C) 2019  Muhammad Haseeb, and Fahad Saeed
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

#include <thread>
#include <semaphore.h>
#include <unistd.h>
#include "dslim_fileout.h"
#include "msquery.hpp"
#include "dslim.h"
#include "lwqueue.h"
#include "lwbuff.h"
#include "scheduler.h"
#include "ms2prep.hpp"
#include "hicops_instr.hpp"

#include "cuda/superstep3/kernel.hpp"

using namespace std;

extern gParams   params;
extern BYICount  *Score;

/* Global variables */
float_t *hyperscores         = nullptr;
uchar_t *sCArr               = nullptr;

#ifdef USE_MPI
DSLIM_Comm *CommHandle    = nullptr;
hCell      *CandidatePSMS  = nullptr;
#endif // USE_MPI

Scheduler  *SchedHandle    = nullptr;
expeRT     *ePtrs          = nullptr;

/* Lock for query file vector and thread manager */
lock_t qfilelock;
lwqueue<MSQuery *> *qfPtrs = nullptr;

int_t spectrumID             = 0;
int_t currSpecID             = 0;
int_t nBatches               = 0;
int_t gBatchID               = 0;
int_t dssize                 = 0;
double gtime                 = 0;

// lock for the global batch id
std::mutex gBatchlock;

/* Expt spectra data buffer */
lwbuff<Queries<spectype_t>> *qPtrs     = nullptr;

#if defined (USE_GPU)

std::mutex gtimelock;

#endif // USE_GPU

#ifdef USE_MPI

lock_t qfoutlock;
lwqueue<ebuffer*> *qfout   = nullptr;
std::vector<std::thread> fouts;
VOID DSLIM_FOut_Thread_Entry();
std::atomic<bool> exitSignal(false);

#endif // USE_MPI

/* A queue containing I/O thread state when preempted */
lwqueue<MSQuery *> *ioQ = nullptr;
lock_t ioQlock;
std::atomic<bool> scheduler_init(false);

//
// -------------------------- Static functions ----------------------------------
//

static BOOL   DSLIM_BinarySearch(Index *, float_t, int_t&, int_t&);
static int_t  DSLIM_BinFindMin(pepEntry *entries, float_t pmass1, int_t min, int_t max);
static int_t  DSLIM_BinFindMax(pepEntry *entries, float_t pmass2, int_t min, int_t max);
static inline status_t DSLIM_Deinit_IO();

//
// ------------------------------------------------------------------------------
//

/* FUNCTION: DSLIM_WaitFor_IO
 *
 * DESCRIPTION:
 *
 * INPUT:
 * @none
 *
 * OUTPUT:
 * @status: status of execution
 *
 */
static inline status_t DSLIM_WaitFor_IO(Queries<spectype_t>* &, int_t &);

static inline status_t DSLIM_WaitFor_IO(Queries<spectype_t> *&workPtr, int_t &batchsize)
{
    status_t status;

    batchsize = 0;

    /* Wait for a I/O request */
    status = qPtrs->lockr_();

    /* Check for either a buffer or stopSignal */
    while (qPtrs->isEmptyReadyQ())
    {
        /* If both conditions fail,
         * the I/O threads still working */
        status = qPtrs->unlockr_();

        // safety from a rare race condition
        if (!SchedHandle->getNumActivThds())
            SchedHandle->dispatchThread();

        sleep(0.1);

        status = qPtrs->lockr_();
    }

    if (status == SLM_SUCCESS)
    {
        /* Get the I/O ptr from the wait queue */
        workPtr = qPtrs->getWorkPtr();

        if (workPtr == nullptr)
            status = ERR_INVLD_PTR;

        status = qPtrs->unlockr_();

        batchsize = workPtr->numSpecs;
    }

    return status;
}

status_t DSLIM_MS2Initialize()
{
    status_t status = SLM_SUCCESS;

    //
    // MS/MS Data Initialization
    //
#if defined (USE_TIMEMORY)
    wall_tuple_t init("MS2_init");
#endif // USE_TIMEMORY

    // print current progress
    printProgress(MS2 Index Initialization);

    MARK_START(ms2init);

    /* The mutex for queryfile vector */
    if (status == SLM_SUCCESS)
    {
        status = sem_init(&qfilelock, 0, 1);

        status = hcp::ms2::initialize(&qfPtrs, nBatches, dssize);
    }

    /* Initialize the lw double buffer queues with
     * capacity, min and max thresholds */
    if (status == SLM_SUCCESS)
        qPtrs = new lwbuff<Queries<spectype_t>>(20, 5, 15); // cap, th1, th2

    /* Initialize the ePtrs */
    if (status == SLM_SUCCESS)
        ePtrs = new expeRT[params.threads];

    /* Create queries buffers and push them to the lwbuff */
    if (status == SLM_SUCCESS)
    {
        /* Create new Queries */
        for (int_t wq = 0; wq < qPtrs->len(); wq++)
        {
            Queries<spectype_t> *nPtr = new Queries<spectype_t>;

            /* Initialize the query buffer */
            nPtr->init();

            /* Add them to the buffer */
            qPtrs->Add(nPtr);
        }
    }

    if (status == SLM_SUCCESS)
    {
        /* Let's do a queue of 10 MSQuery elements -
         * should be more than enough */

        ioQ = new lwqueue<MSQuery*>(10);

        /* Check for correct allocation */
        if (ioQ == nullptr)
            status = ERR_BAD_MEM_ALLOC;

        /* Initialize the ioQlock */
        status = sem_init(&ioQlock, 0, 1);
    }

    MARK_END(ms2init);

    if (params.myid == 0)
    {
        std::cout << "DONE: MS2 Index Init: \tstatus: " << status << std::endl;
        PRINT_ELAPSED(ELAPSED_SECONDS(ms2init));
    }

#if defined (USE_TIMEMORY)
    init.stop();
#endif // USE_TIMEMORY

    return status;
}

// --------------------------------------------------------------------------------------------- //

status_t DSLIM_Setup_Handles()
{
    status_t status = SLM_SUCCESS;

    if (params.nodes == 1)
        status = DFile_InitFiles();

#ifdef USE_MPI
    else if (params.nodes > 1)
    {
        status = sem_init(&qfoutlock, 0, 1);
        qfout = new lwqueue<ebuffer*> (nBatches);

        // create two threads for fout
        for (int i = 0; i < 2; i++)
            fouts.push_back(std::move(std::thread(DSLIM_FOut_Thread_Entry)));
    }
#endif /* USE_MPI */

    /* Initialize the Comm module */
#ifdef USE_MPI

    /* Only required if nodes > 1 */
    if (params.nodes > 1)
    {
        /* Allocate a new DSLIM Comm handle */
        if (status == SLM_SUCCESS)
        {
            CommHandle = new DSLIM_Comm(nBatches);

            if (CommHandle == nullptr)
                status = ERR_BAD_MEM_ALLOC;
        }

        if (status == SLM_SUCCESS)
        {
            CandidatePSMS = new hCell[dssize];

            if (CandidatePSMS == nullptr)
                status = ERR_BAD_MEM_ALLOC;

        }
    }

#endif /* USE_MPI */

    /* Create a new Scheduler handle */
    if (status == SLM_SUCCESS)
    {
        SchedHandle = new Scheduler;

        /* Check for correct allocation */
        if (SchedHandle == nullptr)
            status = ERR_BAD_MEM_ALLOC;
        else
            scheduler_init = true;
    }

    return status;
}

// --------------------------------------------------------------------------------------------- //

status_t DSLIM_SearchManager(Index *index)
{
    status_t status = SLM_SUCCESS;

    //
    // MS/MS initialization and queue setup.
    //
    status = DSLIM_MS2Initialize();

    //
    // setup the comm and scheduling handles
    //
    if (status == SLM_SUCCESS)
        status = DSLIM_Setup_Handles(); 

    //
    // parallel database search
    //
    if (status == SLM_SUCCESS)
        status = DistributedSearch(index);

    //
    // destroy handles and stop threads
    //
    if (status == SLM_SUCCESS)
        status = DSLIM_Destroy_Handles(index);

    return status;
}

// --------------------------------------------------------------------------------------------- //

#if defined (USE_GPU)

void GPU_DistributedSearch(Index *index)
{
    status_t status = SLM_SUCCESS;
    int_t maxlen = params.max_len;
    int_t minlen = params.min_len;
    double ptime = 0;
    int_t batchsize = 0;

    Queries<spectype_t> *gWorkPtr = nullptr;
    int myspecId = 0;

    for (int bid = 0;;)
    {
        // lock global batch lock
        gBatchlock.lock();

        bid = gBatchID;

        // if all batches completed, then break
        if (bid >= nBatches)
        {
            // make sure to unlock before breaking
            gBatchlock.unlock();
            break;
        }
        
        // increase the gBatchID
        gBatchID++;

#if defined(USE_TIMEMORY)
        static thread_local wall_tuple_t gpu_sched_penalty("GPU_DAG_Penalty", false);
        sched_penalty.start();
#endif

        /* Start computing penalty */
        MARK_START(penal);

        status = DSLIM_WaitFor_IO(gWorkPtr, batchsize);

        // update the local spec id and update the global one
        myspecId = spectrumID;

        spectrumID += gWorkPtr->numSpecs;

        // unlock as soon as batch extracted
        gBatchlock.unlock();

#if defined(USE_TIMEMORY)
        gpu_sched_penalty.stop();
#endif
        /* Compute the penalty */
        MARK_END(penal);

        auto penalty = ELAPSED_SECONDS(penal);
        
        ptime += penalty;

#ifndef DIAGNOSE

        if (params.myid == 0)
        {
            std::cout << "GPU PENALTY:   \t" << penalty << "s" << std::endl;
            std::cout << "\nGPU Cumulative Penalty:     " << ptime << "s" << std::endl;
        }

#endif /* DIAGNOSE */

        // if last batch then no need for the scheduler
        if (bid != (nBatches - 1))
        {
            /* Check the status of buffer queues */
            qPtrs->lockr_();
            int_t dec = qPtrs->readyQStatus();
            qPtrs->unlockr_();

            /* Run the Scheduler to manage thread between compute and I/O */
            SchedHandle->runManager(penalty, dec);
        }

        if (params.myid == 0)
        {
            std::cout << "\ngBatch:\t\t" << gWorkPtr->batchNum << std::endl;
            std::cout << "gSpectra:\t" << gWorkPtr->numSpecs << std::endl;
        }

        MARK_START(gpu_search_time);

#ifdef USE_MPI
        // Query the chunk
        status = hcp::gpu::cuda::s3::search(gWorkPtr, index, (maxlen - minlen + 1), myspecId, &CandidatePSMS[myspecId]);
#else
        // Query the chunk
        status = hcp::gpu::cuda::s3::search(gWorkPtr, index, (maxlen - minlen + 1), myspecId);
#endif // USE_MPI

        status = qPtrs->lockw_();

        /* Request next I/O chunk */
        qPtrs->Replenish(gWorkPtr);

        status = qPtrs->unlockw_();

        MARK_END(gpu_search_time);

        // locked section for gtime
        {
            std::unique_lock<std::mutex> gtime_lock(gtimelock);
            // Compute Duration
            gtime +=  ELAPSED_SECONDS(gpu_search_time);
        }

#ifndef DIAGNOSE
        if (params.myid == 0)
            std::cout << "\nGPU Search Time:\t" << ELAPSED_SECONDS(gpu_search_time) << "s" << std::endl;
#endif /* DIAGNOSE */
    }

    // free all GPU resources
    status = hcp::gpu::cuda::s3::deinitialize();

    return;
}

#endif // USE_GPU

// --------------------------------------------------------------------------------------------- //

//
// Parallel Search
//
status_t DistributedSearch(Index *index)
{
    status_t status = SLM_SUCCESS;

    // variable in which the current batchsize will be populated
    // by the DSLIM_WaitFor_IO function
    int_t batchsize = 0;

    double qtime = 0;
    double ptime = 0;

    int_t maxlen = params.max_len;
    int_t minlen = params.min_len;

    //
    // the parallel search
    //

#if defined (USE_GPU)

    // vector for GPU threads
    vector<std::thread> gpuSearchThds;

    // initialze GPU structures and constant data (mods, lgfact etc.)
    hcp::gpu::cuda::s3::initialize();

    // create a GPU thread
    if (params.useGPU)
       gpuSearchThds.push_back(std::move(std::thread(GPU_DistributedSearch, index)));

#endif // USE_GPU

    // print current progress
    printProgress(Database Search);

#if defined (USE_TIMEMORY)
    static search_tuple_t search_inst("SearchAlg");
    search_inst.start();

#   if defined (_UNIX)
        static hw_counters_t search_cntr ("SearchAlg");
        search_cntr.start();
#   endif // _UNIX

#endif // USE_TIMEMORY

    // initialize to nullptr
    Queries<spectype_t> *workPtr = nullptr;
    int myspecId = 0;

    /* The main search loop starts here */
    for (int_t bid = 0;;)
    {
        // lock global batch lock
        gBatchlock.lock();

        bid = gBatchID;

        // if all batches completed, then break
        if (bid >= nBatches)
        {
            // make sure to unlock before breaking
            gBatchlock.unlock();
            break;
        }
        
        // increase the gBatchID
        gBatchID++;

#if defined(USE_TIMEMORY)
        static wall_tuple_t sched_penalty("DAG_Penalty", false);
        sched_penalty.start();
#endif

        /* Start computing penalty */
        MARK_START(penal);

        status = DSLIM_WaitFor_IO(workPtr, batchsize);

        // update the local spec id and update the global one
        myspecId = spectrumID;
        spectrumID += workPtr->numSpecs;

        // unlock as soon as batch extracted
        gBatchlock.unlock();

#if defined(USE_TIMEMORY)
        sched_penalty.stop();
#endif
        /* Compute the penalty */
        MARK_END(penal);

        auto penalty = ELAPSED_SECONDS(penal);
        ptime += penalty;

#ifndef DIAGNOSE
        if (params.myid == 0)
            std::cout << "PENALTY:   \t" << penalty << "s" << std::endl;
#endif /* DIAGNOSE */

        // if last batch then no need for the scheduler
        if (bid != (nBatches - 1))
        {
            /* Check the status of buffer queues */
            qPtrs->lockr_();
            int_t dec = qPtrs->readyQStatus();
            qPtrs->unlockr_();

            /* Run the Scheduler to manage thread between compute and I/O */
            SchedHandle->runManager(penalty, dec);
        }

#ifndef DIAGNOSE
        if (params.myid == 0)
        {
            std::cout << "\nBatch:\t\t" << workPtr->batchNum << std::endl;
            std::cout << "Spectra:\t" << workPtr->numSpecs << std::endl;
        }
#endif /* DIAGNOSE */

        MARK_START(search_time);

        if (status == SLM_SUCCESS)
            /* Query the chunk */
            status = DSLIM_QuerySpectrum(workPtr, index, (maxlen - minlen + 1), myspecId);

        status = qPtrs->lockw_();

        /* Request next I/O chunk */
        qPtrs->Replenish(workPtr);

        status = qPtrs->unlockw_();

        MARK_END(search_time);

        /* Compute Duration */
        qtime +=  ELAPSED_SECONDS(search_time);

#ifndef DIAGNOSE
        if (params.myid == 0)
            std::cout << "\nSearch Time:\t" << ELAPSED_SECONDS(search_time) << "s" << std::endl;
#endif /* DIAGNOSE */
    }

#if defined (USE_TIMEMORY)
        search_inst.stop();
#   if defined (_UNIX)
        search_cntr.stop();
#   endif // _UNIX
#endif // USE_TIMEMORY

    //
    // Overheads
    //

#if defined (USE_GPU)
    // FIXME: remove this - wait for GPU thread to stop
    for (auto &thd : gpuSearchThds)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        thd.join();
    }
#endif // USE_GPU

    // print cumulative search time and penalty
    if (params.myid == 0)
    {
        std::cout << "\nCumulative Penalty:     " << ptime << "s" << std::endl;
        std::cout << "\nCumulative Search Time: " << qtime + gtime << "s" << std::endl << std::endl;
    }

    /* Return the status of execution */
    return status;

}

// --------------------------------------------------------------------------------------------- //

status_t DSLIM_Destroy_Handles(Index *index)
{
    status_t status = SLM_SUCCESS;

#ifdef USE_MPI
    /* Deinitialize the Communication module */
    if (params.nodes > 1)
    {
        // signal fout threads to exit
        exitSignal = true;

#if defined (USE_TIMEMORY)
        wall_tuple_t comm_penalty("comm_ovhd");
        comm_penalty.start();
#endif // USE_TIMEMORY

        MARK_START(comm_ovd);

        for (auto& itr : fouts)
            itr.join();

        fouts.clear();

        MARK_END(comm_ovd);

#if defined (USE_TIMEMORY)
        comm_penalty.stop();
#endif // USE_TIMEMORY

        if (params.myid == 0)
            std::cout << "Total Comm Overhead: " << ELAPSED_SECONDS(comm_ovd) << 's'<< std::endl;

        //
        // Synchronization
        //

        MARK_START(sync);

        if (params.myid == 0)
            std::cout << std::endl << "Waiting to Sync" << std::endl;
#if defined (USE_TIMEMORY)
        wall_tuple_t sync_penalty("sync_penalty");
        sync_penalty.start();

        // wait for synchronization
        tim::mpi::barrier(MPI_COMM_WORLD);

        sync_penalty.stop();
#else

        status = MPI_Barrier(MPI_COMM_WORLD);
#endif // USE_TIMEMORY

        MARK_END(sync);

        if (params.myid == 0)
            std::cout << "Superstep Sync Penalty: " << ELAPSED_SECONDS(sync) << "s" << std::endl<< std::endl;

        //
        // Carry forward dsts for next superstep
        //

        // Carry forward the data to the distributed scoring module
        status = DSLIM_CarryForward(index, CommHandle, ePtrs, CandidatePSMS, spectrumID);

        /* Delete the instance of CommHandle */
        if (CommHandle != nullptr)
        {
            delete CommHandle;
            CommHandle = nullptr;
        }
    }
#endif /* USE_MPI */

    //
    // Deinitialize
    //

    /* Delete the scheduler object */
    if (SchedHandle != nullptr)
    {
        /* Deallocate the scheduler module */
        delete SchedHandle;
        SchedHandle = nullptr;
    }

    /* Deinitialize the IO module */
    status = DSLIM_Deinit_IO();

    if (status == SLM_SUCCESS && params.nodes == 1)
    {
        status = DFile_DeinitFiles();

        delete[] ePtrs;
        ePtrs = nullptr;
    }

    // deinitialize MS2 prep pointers
    hcp::ms2::deinitialize();

    return status;
}

// --------------------------------------------------------------------------------------------- //

status_t DSLIM_QuerySpectrum(Queries<spectype_t> *ss, Index *index, uint_t idxchunk, int currSpecID)
{
    status_t status = SLM_SUCCESS;
    int_t threads = (int_t) params.threads - (int_t) SchedHandle->getNumActivThds();
    uint_t maxz = params.maxz;
    uint_t dF = params.dF;
    uint_t scale = params.scale;
    double_t maxmass = params.max_mass;
    ebuffer *liBuff = nullptr;
    partRes *txArray = nullptr;

    // static instance of the log(factorial(x)) array
    static auto lgfact = hcp::utils::lgfact<hcp::utils::maxshp>();

    if (params.nodes > 1)
    {
        liBuff = new ebuffer;

        txArray = liBuff->packs;
        liBuff->isDone = false;
        liBuff->batchNum = ss->batchNum;
    }
    else
    {
        UNUSED_PARAM(liBuff);
    }

#if !defined(USE_OMP)
    UNUSED_PARAM(threads);
#endif /* USE_OMP */

    /* Sanity checks */
    if (Score == nullptr || (txArray == nullptr && params.nodes > 1))
        status = ERR_INVLD_MEMORY;

    if (status == SLM_SUCCESS)
    {
        /* Should at least be 1 and min 75% */
        int_t minthreads = MAX(1, (params.threads * 3)/4);

        threads = MAX(threads, minthreads);

#ifndef DIAGNOSE
        /* Print how many threads are we using here */
        if (params.myid == 0)
        {
            /* Print the number of query threads */
            std::cout << "Threads:\t" << threads * params.nodes << std::endl;
        }
#endif /* DIAGNOSE */

        /* Process all the queries in the chunk.
         * Setting chunk size to 4 to avoid false sharing
         */
#ifdef USE_OMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
#endif /* USE_OMP */
        for (int_t queries = 0; queries < ss->numSpecs; queries++)
        {
            /* Pointer to each query spectrum */
            auto *QAPtr = ss->moz + ss->idx[queries];
            float_t pmass = ss->precurse[queries];
            auto    pchg  = ss->charges[queries];
            auto    rtime = ss->rtimes[queries];
            auto    *iPtr = ss->intensity + ss->idx[queries];
            auto qspeclen = ss->idx[queries + 1] - ss->idx[queries];
            auto thno = omp_get_thread_num();

            BYC *bycPtr     = Score[thno].byc;
            Results *resPtr = &Score[thno].res;
            expeRT  *expPtr = ePtrs + thno;
            ebuffer *inBuff = inBuff + thno;

#if defined (PROGRESS)
            if (thno == 0 && params.myid == 0)
                std::cout << "\rDONE:\t\t" << (queries * 100) /ss->numSpecs << "%";
#endif // PROGRESS

            for (uint_t ixx = 0; ixx < idxchunk; ixx++)
            {
                uint_t speclen = (index[ixx].pepIndex.peplen - 1) * maxz * iSERIES;
                uint_t halfspeclen = speclen / 2;
#ifdef MATCH_CHARGE
                uint_t peplen_1 = index[ixx].pepIndex.peplen - 1;
#endif // MATCH_CHARGE

                for (uint_t chno = 0; chno < index[ixx].nChunks; chno++)
                {
                    /* Query each chunk in parallel */
                    uint_t *bAPtr = index[ixx].ionIndex[chno].bA;
                    uint_t *iAPtr = index[ixx].ionIndex[chno].iA;

                    int_t minlimit = 0;
                    int_t maxlimit = 0;

                    BOOL val = DSLIM_BinarySearch(index + ixx, ss->precurse[queries], minlimit, maxlimit);

                    // FIXME: remove me // std::cout << " qno = " << queries << " ixx = " << ixx << " chno = " << chno << " minlimit = " << minlimit << " maxlimit = " << maxlimit << std::endl;

                    /* Spectrum violates limits */
                    if (val == false || (maxlimit < minlimit))
                        continue;

                    /* Query all fragments in each spectrum */
                    for (uint_t k = 0; k < qspeclen; k++)
                    {
                        /* Do this to save mem boundedness */
                        auto qion = QAPtr[k];
                        uint_t intn = iPtr[k];

                        /* Check for any zeros
                         * Zero = Trivial query */
                        if (qion > dF && qion < ((maxmass * scale) - 1 - dF))
                        {
                            for (auto bin = qion - dF; bin < qion + 1 + dF; bin++)
                            {
                                /* Locate iAPtr start and end */
                                uint_t start = bAPtr[bin];
                                uint_t end = bAPtr[bin + 1];

                                /* If no ions in the bin */
                                if (end - start < 1)
                                    continue;

                                auto ptr = std::lower_bound(iAPtr + start, iAPtr + end, minlimit * speclen);
                                int_t stt = start + std::distance(iAPtr + start, ptr);

                                ptr = std::upper_bound(iAPtr + stt, iAPtr + end, (((maxlimit + 1) * speclen) - 1));
                                int_t ends = stt + std::distance(iAPtr + stt, ptr) - 1;

                                /* Loop through located iAions */
                                for (auto ion = stt; ion <= ends; ion++)
                                {
                                    uint_t raw = iAPtr[ion];

                                    /* Calculate parent peptide ID */
                                    int_t ppid = (raw / speclen);

                                    /* Calculate the residue */
                                    int_t residue = (raw % speclen);

                                    /* Either 0 or 1 */
                                    int_t isY = residue / halfspeclen;
                                    int_t isB = 1 - isY;

#ifdef MATCH_CHARGE

                                    // FIXME: Is this ichg computation and usage correct?
                                    int_t ichg = (residue / peplen_1) % maxz;
                                    ichg += 1;

                                    // Check if the matched ion's charge is less than or equal to the precursor charge
                                    isY *= (ichg <= pchg);
                                    isB *= (ichg <= pchg);

#endif // MATCH_CHARGE
                                    /* Get the map element */
                                    BYC *elmnt = bycPtr + ppid;

                                    /* Update */
                                    elmnt->bc += isB;
                                    elmnt->ibc += intn * isB;

                                    elmnt->yc += isY;
                                    elmnt->iyc += intn * isY;
                                }
                            }
                        }
                    }

                    /* Compute the chunksize to look further into */
                    int_t csize = maxlimit - minlimit + 1;

                    /* Look for candidate PSMs */
                    for (int_t it = minlimit; it <= maxlimit; it++)
                    {
                        ushort_t bcc = bycPtr[it].bc;
                        ushort_t ycc = bycPtr[it].yc;
                        ushort_t shpk = bcc + ycc;

                        /* Filter by the min shared peaks */
                        if (shpk >= params.min_shp)
                        {
                            /* Create a heap cell */
                            hCell cell;

                            // get the precomputed log(factorial(x))
                            double_t h1 = lgfact[bcc] + lgfact[ycc];

                            /* Fill in the information */
                            cell.hyperscore = h1 + log10(1 + bycPtr[it].ibc) + log10(1 + bycPtr[it].iyc) - 6;

                            /* hyperscore < 0 means either b- or y- ions were not matched */
                            if (cell.hyperscore > 0)
                            {
                                cell.idxoffset = ixx;
                                cell.psid = it;
                                cell.sharedions = shpk;
                                cell.totalions = speclen;
                                cell.pmass = pmass;
                                cell.pchg = pchg;
                                cell.rtime = rtime;
                                cell.fileIndex = ss->fileNum;

                                /* Insert the cell in the heap dst */
                                resPtr->topK.insert(cell);

                                /* Increase the N */
                                resPtr->cpsms += 1;

                                /* Update the histogram */
                                resPtr->survival[(int_t) (cell.hyperscore * 10 + 0.5)] += 1;
                            }
                        }
                    }

                    /* Clear the scorecard */
                    std::memset(bycPtr + minlimit, 0x0, sizeof(BYC) * csize);
                }
            }

#ifdef USE_MPI
            /* Distributed memory mode - Model partial Gumbel
             * and transmit parameters to rx machine */
            if (params.nodes > 1)
            {
                /* Set the params.min_cpsm in dist mem mode to 1 */
                if (resPtr->cpsms >= 1)
                {
                    /* Extract the top PSM */
                    hCell&& psm = resPtr->topK.getMax();

                    /* Put it in the list */
                    CandidatePSMS[currSpecID + queries] = psm;

                    resPtr->maxhypscore = (psm.hyperscore * 10 + 0.5);

                    status = expPtr->StoreIResults(resPtr, queries, liBuff);

                    /* Fill in the Tx array cells */
                    txArray[queries].min  = resPtr->minhypscore;
                    txArray[queries].max2 = resPtr->nexthypscore;
                    txArray[queries].max  = psm.hyperscore;
                    txArray[queries].N    = resPtr->cpsms;
                    txArray[queries].qID  = currSpecID + queries;
                }
                else
                {
                    /* Extract the top result
                     * and put it in the list */
                    CandidatePSMS[currSpecID + queries] = 0;

                    /* Get the handle to the txArr
                     * Fill it up and move on */
                    txArray[queries] = 0;
                    txArray[queries].qID  = currSpecID + queries;
                }
            }

            /* Shared memory mode - Do complete
             * modeling and print results */
            else
#endif /* USE_MPI */
            {
                /* Check for minimum number of PSMs */
                if (resPtr->cpsms >= params.min_cpsm)
                {
                    /* Extract the top PSM */
                    hCell&& psm = resPtr->topK.getMax();

                    resPtr->maxhypscore = (psm.hyperscore * 10 + 0.5);

                    /* Compute expect score if there
                     * are any candidate PSMs */
#ifdef TAILFIT
                    status = expPtr->ModelTailFit(resPtr);

                    /* Linear Regression Parameters */
                    double_t w = resPtr->mu;
                    double_t b = resPtr->beta;

                    w /= 1e6;
                    b /= 1e6;

                    /* Estimate the log (s(x)); x = log(hyperscore) */
                    double_t lgs_x = (w * resPtr->maxhypscore) + b;

                    /* Compute the s(x) */
                    double_t e_x = pow(10, lgs_x);

                    /* e(x) = n * s(x) */
                    e_x *= resPtr->cpsms;

#else
                    status = expPtr->ModelSurvivalFunction(resPtr);

                    /* Extract e(x) = n * s(x) = mu * 1e6 */
                    double_t e_x = resPtr->mu;

                    e_x /= 1e6;

#endif /* TAILFIT */

                    /* Do not print any scores just yet */
                    if (e_x < params.expect_max)
                    {
                        /* Printing the scores in OpenMP mode */
                        status = DFile_PrintScore(index, currSpecID + queries, pmass, &psm, e_x, resPtr->cpsms);
                    }
                }
            }

            /* Reset the results */
            resPtr->reset();
        }
#ifdef USE_MPI
        if (params.nodes > 1)
            liBuff->currptr = ss->numSpecs * Xsamples * sizeof(ushort_t);
#endif // USE_MPI
    }

    // Add a thread
#ifdef USE_MPI
    if (params.nodes > 1)
        AddliBuff(liBuff);

#endif // USE_MPI

    return status;
}

#ifdef USE_MPI
void AddliBuff(ebuffer *liBuff)
{
    sem_wait(&qfoutlock);
    qfout->push(liBuff);
    sem_post(&qfoutlock);
}
#endif // USE_MPI

/*
 * FUNCTION: DSLIM_BinarySearch
 *
 * DESCRIPTION: The Binary Search Algorithm
 *
 * INPUT:
 *
 * OUTPUT
 * none
 */
static BOOL DSLIM_BinarySearch(Index *index, float_t precmass, int_t &minlimit, int_t &maxlimit)
{
    /* Get the float_t precursor mass */
    float_t pmass1 = precmass - params.dM;
    float_t pmass2 = precmass + params.dM;
    pepEntry *entries = index->pepEntries;

    BOOL rv = false;

    uint_t min = 0;
    uint_t max = index->lcltotCnt - 1;

    if (params.dM < 0.0)
    {
        minlimit = min;
        maxlimit = max;

        return rv;
    }

    /* Check for base case */
    if (pmass1 < entries[min].Mass)
    {
        minlimit = min;
    }
    else if (pmass1 > entries[max].Mass)
    {
        minlimit = max;
        maxlimit = max;
        return rv;
    }
    else
    {
        /* Find the minlimit here */
        minlimit = DSLIM_BinFindMin(entries, pmass1, min, max);
    }

    min = 0;
    max = index->lcltotCnt - 1;


    /* Check for base case */
    if (pmass2 > entries[max].Mass)
    {
        maxlimit = max;
    }
    else if (pmass2 < entries[min].Mass)
    {
        minlimit = min;
        maxlimit = min;
        return rv;
    }
    else
    {
        /* Find the maxlimit here */
        maxlimit = DSLIM_BinFindMax(entries, pmass2, min, max);
    }

    if (entries[maxlimit].Mass <= pmass2 && entries[minlimit].Mass >= pmass1)
        rv = true;

    return rv;
}


static int_t DSLIM_BinFindMin(pepEntry *entries, float_t pmass1, int_t min, int_t max)
{
    int_t half = (min + max)/2;

    if (max - min < 20)
    {
        int_t current = min;

        while (entries[current].Mass < pmass1)
            current++;

        return current;
    }

    if (pmass1 > entries[half].Mass)
    {
        min = half;
        return DSLIM_BinFindMin(entries, pmass1, min, max);
    }
    else if (pmass1 < entries[half].Mass)
    {
        max = half;
        return DSLIM_BinFindMin(entries, pmass1, min, max);
    }

    if (pmass1 == entries[half].Mass)
    {
        while (pmass1 == entries[half].Mass)
            half--;

        half++;
    }

    return half;


}

static int_t DSLIM_BinFindMax(pepEntry *entries, float_t pmass2, int_t min, int_t max)
{
    int_t half = (min + max)/2;

    if (max - min < 20)
    {
        int_t current = max;

        while (entries[current].Mass > pmass2)
            current--;

        return current;
    }

    if (pmass2 > entries[half].Mass)
    {
        min = half;
        return DSLIM_BinFindMax(entries, pmass2, min, max);
    }
    else if (pmass2 < entries[half].Mass)
    {
        max = half;
        return DSLIM_BinFindMax(entries, pmass2, min, max);
    }

    if (pmass2 == entries[half].Mass)
    {
        half++;

        while (pmass2 == entries[half].Mass)
            half++;

        half--;
    }

    return half;

}
/*
 * FUNCTION: DSLIM_IO_Threads_Entry
 *
 * DESCRIPTION: Entry function for all
 *              I/O threads
 *
 * INPUT:
 * @argv: Pointer to void arguments
 *
 * OUTPUT:
 * @nullptr: Nothing
 */
VOID DSLIM_IO_Threads_Entry()
{
    status_t status = SLM_SUCCESS;
    Queries<spectype_t> *ioPtr = nullptr;
    bool eSignal = false;
    bool preempt = false;

    // TODO: verify thread local performance
#if defined (USE_TIMEMORY)
    thread_local prep_tuple_t prep_inst("preprocess");
    prep_inst.start();
#endif // USE_TIMEMORY

    /* local object is fine since it will be copied
     * to the queue object at the time of preemption */
    MSQuery *Query = nullptr;

    int_t rem_spec = 0;

    while (!scheduler_init) { usleep(1); }

    /* Initialize and process Query Spectra */
    for (;status == SLM_SUCCESS;)
    {
        /* Check if the Query object is not initialized */
        if (Query == nullptr || Query->isDeInit())
        {
            /* Try getting the Query object from queue if present */
            sem_wait(&ioQlock);

            if (!ioQ->isEmpty())
            {
                Query = ioQ->front();
                status = ioQ->pop();
            }

            sem_post(&ioQlock);

            /* If the queue is empty */
            if (Query == nullptr || Query->isDeInit())
            {
                /* Otherwise, initialize the object from a file */
                /* lock the query file */
                sem_wait(&qfilelock);

                /* Check if anymore query file pointers */
                if (!qfPtrs->isEmpty())
                {
                    Query = qfPtrs->front();
                    qfPtrs->pop();
                    // Init to 1 for first loop to run
                    rem_spec = Query->getQAcount();
                }
                else
                {
                    // Raise the exit signal
                    eSignal = true;
                }

                /* Unlock the query file */
                sem_post(&qfilelock);
            }
        }

        /* If no more files, then break the inf loop */
        if (eSignal == true)
            break;

        /*********************************************
         * At this point, we have the data ready     *
         *********************************************/

        /* Wait for a I/O request */
        status = qPtrs->lockw_();
        sem_wait(&ioQlock);

        /* Empty wait queue or Scheduler preemption signal raised  */
        preempt = SchedHandle->checkPreempt();

        if (qPtrs->isEmptyWaitQ() || preempt)
        {
            status = qPtrs->unlockw_();

            status = ioQ->push(Query);

            sem_post(&ioQlock);

            /* Break from loop */
            break;
        }

        sem_post(&ioQlock);

        /* Otherwise, get the I/O ptr from the wait queue */
        ioPtr = qPtrs->getIOPtr();

        status = qPtrs->unlockw_();

        /* Reset the ioPtr */
        ioPtr->reset();

        /* Extract a chunk and return the chunksize */
        status = Query->extractbatch<int>(QCHUNK, ioPtr, rem_spec);
        
        // update remaining Query entries
        ioPtr->batchNum = Query->Curr_chunk();
        ioPtr->fileNum  = Query->getQfileIndex();
        Query->Curr_chunk()++;

        /* Lock the ready queue */
        qPtrs->lockr_();

#ifdef USE_MPI
        if (params.nodes > 1)
        {
            /* Add an entry of the added buffer to the CommHandle */
            status = CommHandle->AddBatch(ioPtr->batchNum,ioPtr->numSpecs, Query->getQfileIndex());
        }
#endif /* USE_MPI */

        /*************************************
         * Add available data to ready queue *
         *************************************/
        qPtrs->IODone(ioPtr);

        /* Unlock the ready queue */
        qPtrs->unlockr_();

        /* If no more remaining spectra, then deinit */
        if (rem_spec < 1)
        {
            status = Query->DeinitQueryFile();

            if (Query != nullptr)
            {
                delete Query;
                Query = nullptr;
            }
        }

    }

#if defined (USE_TIMEMORY)
    prep_inst.stop();
#endif // USE_TIMEMORY

    if (!preempt)
        /* Request pre-emption */
        SchedHandle->takeControl();
}

#ifdef USE_MPI
VOID DSLIM_FOut_Thread_Entry()
{
    int_t batchSize = 0;
    ebuffer *lbuff = nullptr;

    for (;;usleep(1))
    {
        sem_wait(&qfoutlock);

        if (qfout->isEmpty())
        {
            sem_post(&qfoutlock);

            if (exitSignal)
                return;

            continue;
        }

        lbuff = qfout->front();
        qfout->pop();

        sem_post(&qfoutlock);

        ofstream *fh = new ofstream;
        string_t fn = params.workspace + "/" +
                    std::to_string(lbuff->batchNum) +
                    "_" + std::to_string(params.myid) + ".dat";
        batchSize = lbuff->currptr / (Xsamples * sizeof(ushort_t));
        fh->open(fn, ios::out | ios::binary);
        fh->write((char_t *)lbuff->packs, batchSize * sizeof(partRes));
        fh->write(lbuff->ibuff, lbuff->currptr * sizeof (char_t));
        fh->close();

        delete lbuff;
    }
}
#endif /* USE_MPI */

static inline status_t DSLIM_Deinit_IO()
{
    status_t status = SLM_SUCCESS;

    Queries<spectype_t> *ptr = nullptr;

    while (!qPtrs->isEmptyReadyQ())
    {
        ptr = qPtrs->getWorkPtr();

        if (ptr != nullptr)
        {
            delete ptr;
            ptr = nullptr;
        }
    }

    while (!qPtrs->isEmptyWaitQ())
    {
        ptr = qPtrs->getIOPtr();

        if (ptr != nullptr)
        {
            delete ptr;
            ptr = nullptr;
        }
    }

    /* Delete the qPtrs buffer handle */
    delete qPtrs;

    qPtrs = nullptr;

    /* Deallocate the I/O queues */
    delete ioQ;
    ioQ = nullptr;

    /* Destroy the ioQ lock semaphore */
    status = sem_destroy(&qfilelock);
    status = sem_destroy(&ioQlock);

    return status;
}
