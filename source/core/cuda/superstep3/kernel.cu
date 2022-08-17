/*
 * Copyright (C) 2022 Muhammad Haseeb, and Fahad Saeed
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

#include <cuda.h>
#include <string>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/distance.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <thread>

#include "cuda/driver.hpp"

#include "cuda/superstep1/kernel.hpp"
#include "cuda/superstep3/kernel.hpp"

using namespace std;

#define SEARCH_STREAM               0
#define DATA_STREAM                 1

#define HISTOGRAM_SIZE             (1 + (MAX_HYPERSCORE * 10) + 1)

// -------------------------------------------------------------------------------------------- //

// include the CUDA constant memory objects for A generation
// array to store PTM masses
extern __constant__ float_t modMass[ALPHABETS];

// amino acid masses
extern __constant__ float_t aaMass[ALPHABETS];

// static mod masses
extern __constant__ float_t statMass[ALPHABETS];

// log(factorial(n))
__constant__ double_t d_lgFact[hcp::utils::maxshp];

// -------------------------------------------------------------------------------------------- //

// host side global parameters
extern gParams params;

// -------------------------------------------------------------------------------------------- //

namespace hcp 
{

namespace gpu
{

namespace cuda
{

namespace s3
{

// -------------------------------------------------------------------------------------------- //

//
// CUDA kernel declarations
//
__global__ void SpSpGEMM(dQueries<spectype_t> *d_WorkPtr, uint_t* d_bA, uint_t *d_iA, float_t *d_hscores, float_t *d_evalues, int iter, dScores *d_Scores, int dF, double dM, int speclen, int maxmass, int scale, int ixx);

// database search kernel host wrapper
__host__ status_t SearchKernel(Queries<spectype_t> *, int, int);

// compute min and max limits for the spectra
__host__ status_t MinMaxLimits(Queries<spectype_t> *, Index *, double dM);

template <typename T>
__global__ void vector_plus_constant(T *vect, T val, int size);

extern __device__ void compute_minmaxions(int *minions, int *maxions, int *QAPtr, uint *d_bA, uint *d_iA, int dF, int qspeclen, int speclen, int minlimit, int maxlimit);

extern __device__ void getMaxdhCell(dhCell *topscores, dhCell *out);
// -------------------------------------------------------------------------------------------- //

void dScores::init(int size)
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    // allocate memory for the scores
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(scores, size, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(survival, QCHUNK * HISTOGRAM_SIZE, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(topscore, QCHUNK, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(cpsms, QCHUNK, driver->stream[DATA_STREAM]));
}

// -------------------------------------------------------------------------------------------- //

dScores::~dScores()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(scores, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(survival, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(topscore, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(cpsms, driver->stream[DATA_STREAM]));

    scores = nullptr;
    survival = nullptr;
    topscore = nullptr;
    cpsms = nullptr;
}

// -------------------------------------------------------------------------------------------- //

template <typename T>
dQueries<T>::dQueries()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(this->idx, QCHUNK, driver->stream[DATA_STREAM]));
    //hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(this->precurse, QCHUNK, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(this->minlimits, QCHUNK, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(this->maxlimits, QCHUNK, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(this->moz, QCHUNK * QALEN, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(this->intensity, QCHUNK * QALEN, driver->stream[DATA_STREAM]));

    numPeaks        = 0;
    numSpecs        = 0;
}

// -------------------------------------------------------------------------------------------- //

template <typename T>
void dQueries<T>::H2D(Queries<T> &rhs)
{
    auto driver = hcp::gpu::cuda::driver::get_instance();
    int chunksize = rhs.numSpecs;
    
    this->numSpecs = rhs.numSpecs;
    this->numPeaks = rhs.numPeaks;

    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(this->idx, rhs.idx, chunksize, driver->stream[DATA_STREAM]));
    //hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(this->precurse, rhs.precurse, chunksize, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(this->moz, rhs.moz, this->numPeaks, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(this->intensity, rhs.intensity, this->numPeaks, driver->stream[DATA_STREAM]));

    // driver->stream_sync(DATA_STREAM);
}

// -------------------------------------------------------------------------------------------- //

template <typename T>
dQueries<T>::~dQueries()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    numPeaks = 0;
    numSpecs = 0;

    /* Deallocate the memory */
    if (this->moz != nullptr)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(this->moz, driver->stream[DATA_STREAM]));
        this->moz = nullptr;
    }
    if (this->intensity != nullptr)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(this->intensity, driver->stream[DATA_STREAM]));
        this->intensity = nullptr;
    }
    if (this->minlimits != nullptr)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(this->minlimits, driver->stream[DATA_STREAM]));
        this->minlimits = nullptr;
    }
    if (this->maxlimits != nullptr)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(this->maxlimits, driver->stream[DATA_STREAM]));
        this->maxlimits = nullptr;
    }

    if (this->idx != nullptr)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(this->idx, driver->stream[DATA_STREAM]));
        this->idx = nullptr;
    }

    driver->stream_sync(DATA_STREAM);
}

// -------------------------------------------------------------------------------------------- //

//
// REMOVE ME: test if all constant data is intact
//
__global__ void test_kernel()
{
    if (threadIdx.x == 0)
    {
        for (int i = 0 ; i< ALPHABETS ; i++)
        {
            printf("ModMass[%d]: %f, ", i, modMass[i]);
        }
    }
    if (threadIdx.x == 1)
    {
        for (int i = 0 ; i< ALPHABETS ; i++)
        {
            printf("aaMass[%d]: %f, ", i, aaMass[i]);
        }
    }
    if (threadIdx.x == 2)
    {
        for (int i = 0 ; i< ALPHABETS ; i++)
        {
            printf("statMass[%d]: %f, ", i, statMass[i]);
        }
    }
    if (threadIdx.x == 3)
    {
        for (int i = 0 ; i < hcp::utils::maxshp ; i++)
        {
            printf("d_lgFact[%d]: %f, ", i, d_lgFact[i]);
        }
    }

    __syncthreads();

}

// -------------------------------------------------------------------------------------------- //

__host__ status_t initialize()
{
    // static instance of the log(factorial(x)) array
    static auto h_lgfact = hcp::utils::lgfact<hcp::utils::maxshp>();

    // copy to CUDA constant arrays
    hcp::gpu::cuda::error_check(cudaMemcpyToSymbol(d_lgFact, &h_lgfact.val, sizeof(double_t) * hcp::utils::maxshp)); 

    // TODO: Remove this test kernel 
    test_kernel<<<1,4>>>();

    return SLM_SUCCESS;

}

// -------------------------------------------------------------------------------------------- //

__host__ void getHostMem(float *&h_hscores, float *&h_evalues)
{
    static thread_local float *h_hscores_ptr = h_hscores;
    static thread_local float *h_evalues_ptr = h_evalues;

    if (!h_hscores_ptr)
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::host_pinned_allocate<float>(h_hscores_ptr, QCHUNK));

    if (!h_evalues_ptr)
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::host_pinned_allocate<float>(h_evalues_ptr, QCHUNK));

    h_hscores = h_hscores_ptr;
    h_evalues = h_evalues_ptr;
}

// -------------------------------------------------------------------------------------------- //

dScores *& getScorecard(int chunksize)
{
    auto driver = hcp::gpu::cuda::driver::get_instance();
    static thread_local dScores *d_Scores = nullptr;

    if (!d_Scores)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async<dScores>(d_Scores, SEARCHINSTANCES, driver->stream[DATA_STREAM]));

        for (int i = 0 ; i < SEARCHINSTANCES ; i++)
            d_Scores[i].init(chunksize);
    }

    return d_Scores;
}

// -------------------------------------------------------------------------------------------- //

void freeScorecard()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();
    
    auto d_Scores = getScorecard(0);

    // free the d_Scores
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_Scores, driver->stream[DATA_STREAM]));

    driver->stream_sync(DATA_STREAM);

    // set to nullptr
    d_Scores = nullptr;
}

// -------------------------------------------------------------------------------------------- //

__host__ void freeHostMem()
{
    float *h_hscores = nullptr;
    float *h_evalues = nullptr;

    getHostMem(h_hscores, h_evalues);
    
    if (h_hscores)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::host_pinned_free(h_hscores));
        h_hscores = nullptr;
    }

    if (h_evalues)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::host_pinned_free(h_evalues));
        h_evalues = nullptr;
    }
}

// -------------------------------------------------------------------------------------------- //

__host__ dQueries<spectype_t> *& getdQueries()
{
    static thread_local dQueries<spectype_t> *dqueries = nullptr;
    
    if (!dqueries)
        dqueries = new dQueries<spectype_t>();

    return dqueries;
}

// -------------------------------------------------------------------------------------------- //

__host__ void freedQueries()
{
    auto dqueries = getdQueries();

    if (dqueries)
    {
        delete dqueries;
        dqueries = nullptr;
    }
}

// -------------------------------------------------------------------------------------------- //

__host__ void getDeviceMem(float *&d_hscores, float *&d_evalues)
{
    static auto driver = hcp::gpu::cuda::driver::get_instance();

    static thread_local float *d_hscores_ptr = nullptr;
    static thread_local float *d_evalues_ptr = nullptr;

    if (!d_hscores_ptr)
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async<float>(d_hscores_ptr, QCHUNK, driver->stream[DATA_STREAM]));

    if (!d_evalues_ptr)
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async<float>(d_evalues_ptr, QCHUNK, driver->stream[DATA_STREAM]));

    d_hscores = d_hscores_ptr;
    d_evalues = d_evalues_ptr;
}

// -------------------------------------------------------------------------------------------- //

__host__ void freeDeviceMem()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    float *d_hscores = nullptr;
    float *d_evalues = nullptr;

    getDeviceMem(d_hscores, d_evalues);
    
    if (d_hscores)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_hscores, driver->stream[DATA_STREAM]));
        d_hscores = nullptr;
    }

    if (d_evalues)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_evalues, driver->stream[DATA_STREAM]));
        d_evalues = nullptr;
    }
}

// -------------------------------------------------------------------------------------------- //

// the database search kernel
__host__ status_t search(Queries<spectype_t> *gWorkPtr, Index *index, uint_t idxchunks)
{
    status_t status = SLM_SUCCESS;

    auto dqueries = getdQueries();

    // transfer experimental data to device
    dqueries->H2D(*gWorkPtr);

    // number of spectra in the current batch
    int nspectra = gWorkPtr->numSpecs;

    // get instance of the driver
    static thread_local auto driver = hcp::gpu::cuda::driver::get_instance();

    // host memory to store the top hyperscores and evalues
    float *h_hscores = nullptr;
    float *h_evalues = nullptr;

    // allocate host memory for the top hyperscores and evalues
    // FIXME: Enable later
    //getHostMem(h_hscores, h_evalues);

    // device memory for the top hyperscores and evalues
    float *d_hscores = nullptr;
    float *d_evalues = nullptr;

    // allocate device memory
    // FIXME: Enable later
    //getDeviceMem(d_hscores, d_evalues);

    // sync all data streams
    driver->stream_sync(DATA_STREAM);

    // search for each database chunk (by length)
    for (int i = 0; i < idxchunks ; i++)
    {
        // get the current index portion (by length)
        Index *curr_index = &index[i];

        // construct for each intra-chunk (within each length)
        for (int chno = 0; chno < index->nChunks && status == SLM_SUCCESS; chno++)
        {
            // FIXME: make min-max limits for the spectra in the current chunk
            status = hcp::gpu::cuda::s3::MinMaxLimits(gWorkPtr, curr_index, params.dM);
            uint_t speclen = (curr_index->pepIndex.peplen - 1) * params.maxz * iSERIES;
#if 1
            // build the AT columns. i.e. the iAPtr
            status = hcp::gpu::cuda::s1::ConstructIndexChunk(curr_index, chno, true);
            auto d_iA = hcp::gpu::cuda::s1::getATcols();
#else
            // copy the At columns to the device instead
            auto d_iA = hcp::gpu::cuda::s1::getATcols(iAsize);

            /* Check if this chunk is the last chunk */
            uint_t nsize = ((chno == curr_index->nChunks - 1) && (curr_index->nChunks > 1))?
                   curr_index->lastchunksize : curr_index->chunksize;

            uint_t *iAPtr = curr_index->ionIndex[chno].iA;
            uint_t iAsize = nsize * speclen;

            hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_iA, iAPtr, iAsize, driver->stream[SEARCH_STREAM]));
#endif // 1

            // copy the At rows to device 
            auto d_bA = hcp::gpu::cuda::s1::getbA();
            uint_t bAsize = ((uint_t)(params.max_mass * params.scale)) + 1;

            hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_bA, curr_index->ionIndex[chno].bA, bAsize, driver->stream[SEARCH_STREAM]));

            // synch both streams
            driver->stream_sync(SEARCH_STREAM);

            // search against the database
            status = hcp::gpu::cuda::s3::SearchKernel(gWorkPtr, speclen, i);
            
            // free the AT columns
            hcp::gpu::cuda::s1::freeATcols();
        }
    }

    // all chunks are done. combine the results
    // FIXME: Enable later
    //hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_evalues, d_evalues, nspectra, driver->stream[SEARCH_STREAM]));
    //hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_hscores, d_hscores, nspectra, driver->stream[SEARCH_STREAM]));

    return SLM_SUCCESS;
}

// -------------------------------------------------------------------------------------------- //

__host__ status_t MinMaxLimits(Queries<spectype_t> *h_WorkPtr, Index *index, double dM)
{
    status_t status = SLM_SUCCESS;
    auto driver = hcp::gpu::cuda::driver::get_instance();

    auto d_WorkPtr = getdQueries();

    // extract all peptide masses in an array to simplify computations
    float_t *h_mzs = new float_t[index->lcltotCnt];
    // hcp::gpu::cuda::host_pinned_allocate<float_t>(h_mzs, index->lcltotCnt);

    // simplify the peptide masses
    for (int i = 0; i < index->lcltotCnt; i++)
        h_mzs[i] = index->pepEntries[i].Mass - dM;

    auto size = index->lcltotCnt;

    // initialize device vector with mzs
    float *d_mzs = nullptr;
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_mzs, size, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_mzs, h_mzs, size, driver->stream[0]));

    // binary search the start of each ion and store in minlimits
    thrust::lower_bound(thrust::device.on(driver->get_stream(0)), d_mzs, d_mzs + size, d_mzs, d_mzs + h_WorkPtr->numSpecs, d_WorkPtr->minlimits);

    const int nthreads = 1024;
    int nblocks = size / 1024;
    nblocks += (size % 1024 == 0)? 0 : 1;

    // add + 2*dM to set for maxlimit
    hcp::gpu::cuda::s3::vector_plus_constant<<<nblocks, nthreads, KBYTES(1), driver->get_stream()>>>(d_mzs, (float)(2*dM), size);

    // binary search the end of each spectrum and store in maxlimits
    thrust::upper_bound(thrust::device.on(driver->get_stream(0)), d_mzs, d_mzs + size, d_mzs, d_mzs + h_WorkPtr->numSpecs, d_WorkPtr->maxlimits);

    // d_mzs is no longer needed - free
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_mzs, driver->stream[0]));

    // free the mzs array
    delete[] h_mzs;

    return status;
}

// -------------------------------------------------------------------------------------------- //

__host__ status_t SearchKernel(Queries<spectype_t> *gWorkPtr, int speclen, int ixx)
{
    status_t status = SLM_SUCCESS;

    auto d_WorkPtr = getdQueries();

    // number of spectra in the current batch
    int nspectra = gWorkPtr->numSpecs;
    //int npeaks = gWorkPtr->numPeaks;

    auto d_iA = hcp::gpu::cuda::s1::getATcols();
    auto d_bA = hcp::gpu::cuda::s1::getbA();

    // device memory for the top hyperscores and evalues
    float *d_hscores = nullptr;
    float *d_evalues = nullptr;

    // get driver object
    auto driver = hcp::gpu::cuda::driver::get_instance();

    // get device memory
    getDeviceMem(d_hscores, d_evalues);

    const int itersize = SEARCHINSTANCES;

    int niters = nspectra / itersize;
    niters += (nspectra % itersize == 0)? 0 : 1;

    // get scorecard instance
    auto d_Scores = hcp::gpu::cuda::s3::getScorecard(0);

    for (int iter = 0 ; iter < niters ; iter++)
    {
        int nblocks = itersize;
        int blocksize = 1024;

        // if last iteration, adjust the number of blocks
        if (iter == niters - 1)
            nblocks = nspectra - iter * itersize;

        // Sparse matrix Sparse matrix multiplication
        hcp::gpu::cuda::s3::SpSpGEMM<<<nblocks, blocksize, KBYTES(48), driver->get_stream(0)>>>(d_WorkPtr, d_bA, d_iA, d_hscores, d_evalues,iter, d_Scores, params.dF, params.dM, speclen, params.max_mass, params.scale, ixx);
    }

    // synchronize the stream
    driver->stream_sync(0);

    return status;
}

// -------------------------------------------------------------------------------------------- //

__global__ void SpSpGEMM(dQueries<spectype_t> *d_WorkPtr, uint_t* d_bA, uint_t *d_iA, float_t *d_hscores, float_t *d_evalues, int iter, dScores *d_Scores, int dF, double dM, int speclen, int maxmass, int scale, int ixx)
{
    // get the scorecard pointer
    auto *resPtr = d_Scores + blockIdx.x;

    auto *bycPtr = resPtr->scores;

    // get spectrum data
    int qnum = iter + blockIdx.x;

    auto *QAPtr = d_WorkPtr->moz + d_WorkPtr->idx[qnum];
    auto *iPtr = d_WorkPtr->intensity + d_WorkPtr->idx[qnum];
    int qspeclen = d_WorkPtr->idx[qnum + 1] - d_WorkPtr->idx[qnum];

    int minlimit = d_WorkPtr->minlimits[qnum];
    int maxlimit = d_WorkPtr->maxlimits[qnum];

    //
    // for all ions +- dF, get the start/stt and end/ends ranges in the shared memory
    // then only iterate over those ranges and update the scorecard
    //
    // how to write the hyperscores to the null distribution simultaneously? - reduce but how?
    //
#define MAXDF                          2

    extern __shared__ int sharedmem[];

    int *minions = sharedmem;
    int *maxions = sharedmem + (2*MAXDF + 1) * QALEN;

    // fixme Start
    int halfspeclen = speclen / 2;

    hcp::gpu::cuda::s3::compute_minmaxions(minions, maxions, QAPtr, d_bA, d_iA, dF, qspeclen, speclen, minlimit, maxlimit);

    for (int k = 0; k < qspeclen; k++)
    {
        uint_t intn = iPtr[k];
        auto qion = QAPtr[k];

        for (auto bin = qion - dF; bin < qion + 1 + dF; bin++)
        {
            int stt = minions[bin];
            int ends = maxions[bin];

            for (auto ion = stt + threadIdx.x; ion <= ends; ion+= blockDim.x)
            {
                uint_t raw = d_iA[ion];

                /* Calculate parent peptide ID */
                int_t ppid = (raw / speclen);

                /* Calculate the residue */
                int_t residue = (raw % speclen);

                /* Either 0 or 1 */
                int_t isY = residue / halfspeclen;
                int_t isB = 1 - isY;

                /* Get the map element */
                auto *elmnt = bycPtr + ppid;

                /* Update */
                elmnt->bc += isB;
                elmnt->ibc += intn * isB;

                elmnt->yc += isY;
                elmnt->iyc += intn * isY;
            }
        }
    }

    __shared__ dhCell topscores[1024];
    __shared__ int histogram[HISTOGRAM_SIZE];

    for (int i = 0; i < HISTOGRAM_SIZE; i++)
        histogram[i] = 0;

    __syncthreads();

    /* Look for candidate PSMs */
    for (int_t it = minlimit + threadIdx.x; it <= maxlimit; it+= blockDim.x)
    {
        ushort_t bcc = bycPtr[it].bc;
        ushort_t ycc = bycPtr[it].yc;
        ushort_t shpk = bcc + ycc;

        /* Filter by the min shared peaks */
        if (shpk >= 4) // FIXME:
        {
            /* Create a heap cell */
            dhCell cell;

            // get the precomputed log(factorial(x))
            double_t h1 = d_lgFact[bcc] + d_lgFact[ycc];

            /* Fill in the information */
            cell.hyperscore = h1 + log10f(1 + bycPtr[it].ibc) + log10f(1 + bycPtr[it].iyc) - 6;

            /* hyperscore < 0 means either b- or y- ions were not matched */
            if (cell.hyperscore > 0)
            {
                // Update the histogram
                atomicAdd(&histogram[threadIdx.x * HISTOGRAM_SIZE + (int)(cell.hyperscore * 10 + 0.5)], 1);

                cell.idxoffset = ixx;
                cell.psid = it;
                cell.sharedions = shpk;
                // cell.fileIndex = dQueries->fileNum;

                if (cell.hyperscore > topscores[threadIdx.x].hyperscore)
                {
                    topscores[threadIdx.x].hyperscore = cell.hyperscore;
                    topscores[threadIdx.x].psid       = cell.psid;
                    topscores[threadIdx.x].idxoffset  = cell.idxoffset;
                    topscores[threadIdx.x].sharedions = cell.sharedions;
                }

                __syncthreads();
            }
        }
    }

    __syncthreads();

    hcp::gpu::cuda::s3::getMaxdhCell(topscores, resPtr->topscore);

    // copy histogram to the global memory
    for (int ii = threadIdx.x; ii < HISTOGRAM_SIZE; ii+=blockDim.x)
            resPtr->survival[ii] += histogram[HISTOGRAM_SIZE];

    // reset the bycPtr
    for (int f = minlimit + threadIdx.x; f <= maxlimit; f += blockDim.x)
    {
        bycPtr[f].bc = 0;
        bycPtr[f].yc = 0;
        bycPtr[f].ibc = 0;
        bycPtr[f].iyc = 0;
    }

    return;
}

// -------------------------------------------------------------------------------------------- //

template <typename T>
__global__ void vector_plus_constant(T *vect, T val, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size)
        vect[i] += val;
}

// -------------------------------------------------------------------------------------------- //

status_t deinitialize()
{
    hcp::gpu::cuda::s1::freeFragIon();
    hcp::gpu::cuda::s1::freebA();
    hcp::gpu::cuda::s1::freeATcols();

    freeHostMem();
    freeDeviceMem();
    freedQueries();

    // sync all streams
    hcp::gpu::cuda::driver::get_instance()->all_streams_sync();

    return SLM_SUCCESS;
}

// -------------------------------------------------------------------------------------------- //

} // namespace s3

} // namespace cuda

} // namespace gpu

} // namespace hcp