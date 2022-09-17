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
#include "cuda/superstep4/kernel.hpp"

using namespace std;

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

namespace s4
{

template <typename T>
extern __device__ void ArraySum(T *arr, int size, T *sum);

}
namespace s3
{

// -------------------------------------------------------------------------------------------- //

//
// CUDA kernel declarations
//
__global__ void SpSpGEMM(spectype_t *dQ_moz, spectype_t *dQ_intensity, uint_t *dQ_idx, int *dQ_minlimits, int *dQ_maxlimits, 
                        uint_t* d_bA, uint_t *d_iA, int iter, BYC *bycP, int maxchunk, dScores *d_Scores, int dF, int speclen, 
                        int maxmass, int scale, short min_shp, int ixx);

// database search kernel host wrapper
__host__ status_t SearchKernel(Queries<spectype_t> *, int, int);

// compute min and max limits for the spectra
__host__ status_t MinMaxLimits(Queries<spectype_t> *, Index *, double dM);

template <typename T>
__global__ void vector_plus_constant(T *vect, T val, int size);

__global__ void assign_dScores(dScores *a, double_t *survival, dhCell *topscores, int *cpsms);

__global__ void reset_dScores(dScores *d_Scores);

extern __device__ void compute_minmaxions(int *minions, int *maxions, int *QAPtr, uint *d_bA, uint *d_iA, int dF, int qspeclen, int speclen, int minlimit, int maxlimit, int maxmass, int scale);

extern __device__ void getMaxdhCell(dhCell *topscores, dhCell *out);

extern __device__ void getMinsurvival(double_t *survival, double_t *out);


// -------------------------------------------------------------------------------------------- //

dScores::dScores()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    double_t *l_survival;
    dhCell    *l_topscore;
    int      *l_cpsms;

    // allocate memory for the scores
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(l_survival, HISTOGRAM_SIZE * QCHUNK, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(l_topscore, QCHUNK, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(l_cpsms, QCHUNK, driver->stream[DATA_STREAM]));

    assign_dScores<<<1,1, 32, driver->stream[DATA_STREAM]>>>(this, l_survival, l_topscore, l_cpsms);

    driver->stream_sync(DATA_STREAM);
}

// -------------------------------------------------------------------------------------------- //

dScores::~dScores()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    dScores *h_dScores = new dScores();
    dScores *d_dScores = this;

    // copy pointers from the device
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_dScores, d_dScores, 1, driver->stream[DATA_STREAM]));

    // free all memory
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(h_dScores->survival, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(h_dScores->topscore, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(h_dScores->cpsms, driver->stream[DATA_STREAM]));

    driver->stream_sync(DATA_STREAM);

    // free the host memory
    delete h_dScores;
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
void dQueries<T>::H2D(Queries<T> *rhs)
{
    auto driver = hcp::gpu::cuda::driver::get_instance();
    int chunksize = rhs->numSpecs;
    
    this->numSpecs = rhs->numSpecs;
    this->numPeaks = rhs->numPeaks;

    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(this->moz, rhs->moz, this->numPeaks, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(this->intensity, rhs->intensity, this->numPeaks, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(this->idx, rhs->idx, chunksize, driver->stream[DATA_STREAM]));
    //hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(this->precurse, rhs.precurse, chunksize, driver->stream[DATA_STREAM]));

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

__host__ status_t initialize()
{
    // static instance of the log(factorial(x)) array
    static auto h_lgfact = hcp::utils::lgfact<hcp::utils::maxshp>();

    // copy to CUDA constant arrays
    hcp::gpu::cuda::error_check(cudaMemcpyToSymbol(d_lgFact, &h_lgfact.val, sizeof(double_t) * hcp::utils::maxshp)); 

    return SLM_SUCCESS;

}

// -------------------------------------------------------------------------------------------- //

std::pair<BYC *, int>& getBYC(int chunksize)
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    static std::pair<BYC *, int> bycPair;
    static BYC *d_BYC = nullptr;
    static int maxchunk = 0;

    if (!d_BYC)
    {
        maxchunk = chunksize;

        if (!maxchunk)
            std::cout << "Error: getBYC: chunksize is zero" << std::endl;
        else
            hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async<BYC>(d_BYC, maxchunk * SEARCHINSTANCES, driver->stream[DATA_STREAM]));
    }

    driver->stream_sync(DATA_STREAM);

    // update the pair
    bycPair = make_pair(d_BYC, maxchunk);

    // return the static pair
    return bycPair;
}

// -------------------------------------------------------------------------------------------- //

void freeBYC()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();
    
    auto pBYC = getBYC();

    auto d_BYC = std::get<0>(pBYC);
    auto maxchunk = std::get<1>(pBYC);

    if (d_BYC)
        // free the d_Scores
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_BYC, driver->stream[DATA_STREAM]));

    if (maxchunk)
        maxchunk = 0;

    driver->stream_sync(DATA_STREAM);

    // set to nullptr
    d_BYC = nullptr;
}

// -------------------------------------------------------------------------------------------- //

dScores *& getScorecard()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();
    static dScores *d_Scores = nullptr;

    if (!d_Scores)
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async<dScores>(d_Scores, 1, driver->stream[DATA_STREAM]));

    driver->stream_sync(DATA_STREAM);

    return d_Scores;
}

// -------------------------------------------------------------------------------------------- //

void freeScorecard()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();
    
    auto d_Scores = getScorecard();

    if (d_Scores)
        // free the d_Scores
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_Scores, driver->stream[DATA_STREAM]));

    driver->stream_sync(DATA_STREAM);

    // set to nullptr
    d_Scores = nullptr;
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
    auto &&dqueries = getdQueries();

    if (dqueries)
    {
        delete dqueries;
        dqueries = nullptr;
    }
}

// -------------------------------------------------------------------------------------------- //

// the database search kernel
__host__ status_t search(Queries<spectype_t> *gWorkPtr, Index *index, uint_t idxchunks, int gpucurrSpecID, hCell *CandidatePSMS)
{
    status_t status = SLM_SUCCESS;

    auto dqueries = getdQueries();

    // transfer experimental data to device
    dqueries->H2D(gWorkPtr);

    // number of spectra in the current batch
    //int nspectra = gWorkPtr->numSpecs;

    // get instance of the driver
    static thread_local auto driver = hcp::gpu::cuda::driver::get_instance();

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

#if 1       // TODO: Leave this or remove this

            // build the AT columns. i.e. the iAPtr
            status = hcp::gpu::cuda::s1::ConstructIndexChunk(curr_index, chno, true);
            auto d_iA = hcp::gpu::cuda::s1::getATcols();

#else

            /* Check if this chunk is the last chunk */
            uint_t nsize = ((chno == curr_index->nChunks - 1) && (curr_index->nChunks > 1))?
                   curr_index->lastchunksize : curr_index->chunksize;

            uint_t *iAPtr = curr_index->ionIndex[chno].iA;
            uint_t iAsize = nsize * speclen;

            // copy the At columns to the device instead
            auto d_iA = hcp::gpu::cuda::s1::getATcols(iAsize);

            hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_iA, iAPtr, iAsize, driver->stream[DATA_STREAM]));
#endif // 1

            // copy the At rows to device 
            auto d_bA = hcp::gpu::cuda::s1::getbA();
            uint_t bAsize = ((uint_t)(params.max_mass * params.scale)) + 1;

            hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_bA, curr_index->ionIndex[chno].bA, bAsize, driver->stream[DATA_STREAM]));

            // synch both streams
            driver->stream_sync(DATA_STREAM);

            // search against the database
            status = hcp::gpu::cuda::s3::SearchKernel(gWorkPtr, speclen, i);

            // free the AT columns
            hcp::gpu::cuda::s1::freeATcols();
        }
    }

    hcp::gpu::cuda::s3::reset_dScores();

#ifdef USE_MPI

    if (params.nodes > 1)
        status = hcp::gpu::cuda::s4::getIResults(index, gWorkPtr, gpucurrSpecID, CandidatePSMS);
    else
#else
        // combine the results
        status = hcp::gpu::cuda::s4::processResults(index, gWorkPtr, gpucurrSpecID);
#endif // USE_MPI

    return status;
}

// -------------------------------------------------------------------------------------------- //

__host__ status_t MinMaxLimits(Queries<spectype_t> *h_WorkPtr, Index *index, double dM)
{
    status_t status = SLM_SUCCESS;
    auto driver = hcp::gpu::cuda::driver::get_instance();

    auto d_WorkPtr = getdQueries();

    // extract all peptide masses in an array to simplify computations
    float_t *h_mzs; // = new float_t[index->lcltotCnt];
    hcp::gpu::cuda::host_pinned_allocate<float_t>(h_mzs, index->lcltotCnt);

    // simplify the peptide masses
    for (int i = 0; i < index->lcltotCnt; i++)
        h_mzs[i] = index->pepEntries[i].Mass;

    auto size = index->lcltotCnt;

    // initialize device vector with mzs
    float *d_mzs = nullptr;
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_mzs, size, driver->stream[SEARCH_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_mzs, h_mzs, size, driver->stream[SEARCH_STREAM]));

    float *d_precurse = nullptr;
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_precurse, h_WorkPtr->numSpecs, driver->stream[SEARCH_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_precurse, h_WorkPtr->precurse, h_WorkPtr->numSpecs, driver->stream[SEARCH_STREAM]));

    const int nthreads = 1024;
    int nblocks = h_WorkPtr->numSpecs / 1024;
    nblocks += (h_WorkPtr->numSpecs % 1024 == 0)? 0 : 1;

    // add -dM to set for minlimit
    hcp::gpu::cuda::s3::vector_plus_constant<<<nblocks, nthreads, KBYTES(1), driver->get_stream(SEARCH_STREAM)>>>(d_precurse, (float)(-dM), h_WorkPtr->numSpecs);

    // binary search the start of each ion and store in minlimits
    thrust::lower_bound(thrust::device.on(driver->get_stream(SEARCH_STREAM)), d_mzs, d_mzs + size, d_precurse, d_precurse + h_WorkPtr->numSpecs, d_WorkPtr->minlimits);

    // add -dM to set for minlimit
    hcp::gpu::cuda::s3::vector_plus_constant<<<nblocks, nthreads, KBYTES(1), driver->get_stream(SEARCH_STREAM)>>>(d_precurse, (float)(2*dM), h_WorkPtr->numSpecs);

    // binary search the end of each spectrum and store in maxlimits
    thrust::upper_bound(thrust::device.on(driver->get_stream(SEARCH_STREAM)), d_mzs, d_mzs + size, d_precurse, d_precurse + h_WorkPtr->numSpecs, d_WorkPtr->maxlimits);

    // d_mzs is no longer needed - free
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_mzs, driver->stream[SEARCH_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_precurse, driver->stream[SEARCH_STREAM]));

    // free the mzs array
    hcp::gpu::cuda::host_pinned_free(h_mzs);

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

    // get driver object
    auto driver = hcp::gpu::cuda::driver::get_instance();


    const int itersize = SEARCHINSTANCES;

    int niters = nspectra / itersize;
    niters += (nspectra % itersize == 0)? 0 : 1;

    // get resPtr instance
    auto d_Scores = hcp::gpu::cuda::s3::getScorecard();

    // get BYC scorecard
    auto pBYC = hcp::gpu::cuda::s3::getBYC();

    auto d_BYC = std::get<0>(pBYC);
    auto maxchunk = std::get<1>(pBYC);

    // fixme: remove me
    //int *d_debug = nullptr;
    //hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_debug, MBYTES(20), driver->stream[SEARCH_STREAM]));

    for (int iter = 0 ; iter < niters ; iter++)
    {
        int nblocks = itersize;
        int blocksize = 1024;

        // if last iteration, adjust the number of blocks
        if (iter == niters - 1)
            nblocks = nspectra - iter * itersize;

        // set the shared memory to 48KB
        cudaFuncSetAttribute(SpSpGEMM, cudaFuncAttributeMaxDynamicSharedMemorySize, KBYTES(48));

        // FIXME:: revert nblocks from 1 to nblocks    
        hcp::gpu::cuda::s3::SpSpGEMM<<<nblocks, blocksize, KBYTES(48), driver->stream[SEARCH_STREAM]>>>(d_WorkPtr->moz, d_WorkPtr->intensity, d_WorkPtr->idx, d_WorkPtr->minlimits, d_WorkPtr->maxlimits, d_bA, d_iA, iter, d_BYC, maxchunk, d_Scores, params.dF, speclen, params.max_mass, params.scale, params.min_shp, ixx);
    }

    // synchronize the stream
    driver->stream_sync(SEARCH_STREAM);

    // fixme: remove me
    //int szzzzz = (38911-31657+1);
    //int *h_debug = new int[MBYTES(20)];
    //hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_debug, d_debug, szzzzz * 4, driver->stream[SEARCH_STREAM]));

    //driver->stream_sync(SEARCH_STREAM);

    //for (int i = 0; i <= szzzzz; i++)
    //{
    //    if (h_debug[i*3 + 0])
     //       std::cout << "i = " << i + 31657 << ", " << h_debug[i*3 + 0] << ", " << h_debug[i*3 + 1] << ", " << h_debug[i*3 + 2] << ", " << std::endl;
    //}

    // fixme: remove me
/*     for (int i = 0; i < HISTOGRAM_SIZE; i++)
    {
        std::cout << "i = " << i << " : " << h_debug[i] << std::endl;
    }

    std::cout << std::endl;

    std::cout << "topscore.hyperscore = " << h_debug[HISTOGRAM_SIZE] << std::endl;
    std::cout << "topscore.psid = " << h_debug[HISTOGRAM_SIZE+1] << std::endl;
    std::cout << "topscore.idxoffset = " << h_debug[HISTOGRAM_SIZE+2] << std::endl;
    std::cout << "topscore.sharedions = " << h_debug[HISTOGRAM_SIZE+3] << std::endl; */

    //hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_debug, driver->stream[SEARCH_STREAM]));

    return status;
}

__host__ void reset_dScores()
{
    auto d_Scores = hcp::gpu::cuda::s3::getScorecard();

    // get driver object
    auto driver = hcp::gpu::cuda::driver::get_instance();

    // reset the scorecard
    hcp::gpu::cuda::s3::reset_dScores<<<256, 256, 0, driver->stream[SEARCH_STREAM]>>>(d_Scores);
}

// -------------------------------------------------------------------------------------------- //

__global__ void reset_dScores(dScores *d_Scores)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int ik = tid; ik < QCHUNK * HISTOGRAM_SIZE; ik+=stride)
        d_Scores->survival[ik] = 0;

    for (int ik = tid; ik < QCHUNK; ik+=stride)
    {
        d_Scores->cpsms[ik] = 0;
        d_Scores->topscore[ik].hyperscore = 0;
        d_Scores->topscore[ik].psid = 0;
        d_Scores->topscore[ik].idxoffset = 0;
        d_Scores->topscore[ik].sharedions = 0;
    }
}

// -------------------------------------------------------------------------------------------- //

__global__ void SpSpGEMM(spectype_t *dQ_moz, spectype_t *dQ_intensity, uint_t *dQ_idx, int *dQ_minlimits, int *dQ_maxlimits, 
                        uint_t* d_bA, uint_t *d_iA, int iter, BYC *bycP, int maxchunk, dScores *d_Scores, int dF, int speclen, 
                        int maxmass, int scale, short min_shp, int ixx)
{
    dScores *resPtr = d_Scores;

    BYC *bycPtr = &bycP[blockIdx.x * maxchunk];

    // get spectrum data
    int qnum = iter + blockIdx.x;

    auto *survival = &resPtr->survival[qnum * HISTOGRAM_SIZE];
    auto *cpsms = resPtr->cpsms + qnum;
    auto *QAPtr = dQ_moz + dQ_idx[qnum];
    auto *iPtr = dQ_intensity + dQ_idx[qnum];
    int qspeclen = dQ_idx[qnum + 1] - dQ_idx[qnum];
    int halfspeclen = speclen / 2;

    int minlimit = dQ_minlimits[qnum];
    int maxlimit = dQ_maxlimits[qnum] - 1; // maxlimit = upper_bound - 1

    // fixme: remove me
    //printf("qnum=%d, minlimit=%d, maxlimit=%d, qspeclen=%d\n", qnum, minlimit, maxlimit, qspeclen);

#define MAXDF                          2

    __shared__ int minions[(2*MAXDF + 1) * QALEN];
    __shared__  int maxions[(2*MAXDF + 1) * QALEN];

    // setup shared memory here
    __syncthreads();

    // compute min and maxlimits for ions in minions and maxions
    hcp::gpu::cuda::s3::compute_minmaxions(minions, maxions, QAPtr, d_bA, d_iA, dF, qspeclen, speclen, minlimit, maxlimit, maxmass, scale);

    // fixme: remove me
    //printf("qnum=%d, minlimit=%d, maxlimit=%d, qspeclen=%d\n", qnum, minions[threadIdx.x], maxions[threadIdx.x], qspeclen);

    // iterate over all ions
    for (int k = 0; k < qspeclen; k++)
    {
        uint_t intn = iPtr[k];
        auto qion = QAPtr[k];

        // ion +-dF
        for (auto bin = qion - dF; bin < qion + 1 + dF; bin++)
        {
            short binidx = k *(2*dF + 1) + dF - (qion - bin);

            int off1 = minions[binidx];
            int off2 = maxions[binidx];

            int stt = d_bA[bin] +off1;
            int ends = d_bA[bin] + off2;

            //if (threadIdx.x == 0 && stt <= ends)
                //printf("k=%d, bin=%d, stt=%d, ends=%d, ss1=%d\n", binidx, bin, off1, off2, d_bA[bin]);

            for (auto ion = stt + threadIdx.x; ion <= ends; ion+= blockDim.x)
            {
                uint_t raw = d_iA[ion];

                /* Calculate parent peptide ID */
                int_t ppid = (raw / speclen);

                /* Calculate the residue */
                int_t residue = (raw % speclen);

                /* Either 0 or 1 */
                bool isY = residue / halfspeclen;
                bool isB = !isY;

                auto iYC = intn * isY;
                auto iBC = intn * isB;

                // fixme: remove me 
                //if (isB)
                    //printf("bin=%d, raw=%d, slen=%d, ppid=%d, intn=%d, isY=%d, isB=%d, iyc=%d, ibc=%d, binidx=%d, off1=%d, off2=%d, stt=%d, ends=%d\n", bin, raw, speclen, ppid, intn, isY, isB, iYC, iBC, binidx, off1, off2, stt, ends);

                /* Get the map element */
                BYC *elmnt = bycPtr + ppid;

                elmnt->yc += isY;
                elmnt->bc += isB;
                elmnt->iyc += iYC;
                elmnt->ibc += iBC;
            }
            __syncthreads();
        }
    }

    // fixme: remove me 
/*     for (int jj = minlimit + threadIdx.x; jj <= maxlimit; jj += blockDim.x)
    {
        BYC *elmnt = bycPtr + jj;
        int idx = jj - minlimit;
        debug[4*idx + 0] = elmnt->yc;
        debug[4*idx + 1] = elmnt->bc;
        debug[4*idx + 2] = elmnt->iyc;
        debug[4*idx + 3] = elmnt->ibc;
    } */

    //if (threadIdx.x ==0)
        //printf("qnum=%d, minlimit=%d, maxlimit=%d, qspeclen=%d\n", qnum, minlimit, maxlimit, qspeclen);

    // shared memory for topscores and the histogram
    __shared__ int histogram[HISTOGRAM_SIZE];
    __shared__ dhCell topscores[1024];
    __shared__ int l_cpsms[1024];

    for (int ij = threadIdx.x; ij < HISTOGRAM_SIZE; ij+=blockDim.x)
        histogram[ij] = 0;

    for (int ij = threadIdx.x; ij < 1024; ij+=blockDim.x)
    {
        l_cpsms[ij] = 0;
        topscores[ij].hyperscore = 0;
        topscores[ij].psid = 0;
        topscores[ij].idxoffset = 0;
        topscores[ij].sharedions = 0;

    }

    __syncthreads();

    /* Look for candidate PSMs */
    for (int_t it = minlimit + threadIdx.x; it <= maxlimit; it+= blockDim.x)
    {
        ushort_t bcc = bycPtr[it].bc;
        ushort_t ycc = bycPtr[it].yc;
        ushort_t shpk = bcc + ycc;

        // fixme: remove me 
        //printf("tid:%d, it=%d, shpk=%d, ibc=%u, iyc=%u\n", threadIdx.x, it, shpk, bycPtr[it].ibc, bycPtr[it].iyc);

        /* Filter by the min shared peaks */
        if (shpk >= min_shp) 
        {
            /* Create a heap cell */
            dhCell cell;

            // get the precomputed log(factorial(x))
            double_t h1 = d_lgFact[bcc] + d_lgFact[ycc];

            /* Fill in the information */
            cell.hyperscore = h1 + log10f(1 + bycPtr[it].ibc) + log10f(1 + bycPtr[it].iyc) - 6;

            // fixme: remove me 
            //printf("tid:%d, it=%d, shpk=%d, hs=%f\n", threadIdx.x, it, bcc, ycc, shpk, cell.hyperscore);

            //int ii = it-minlimit;
            //debug[ii*3 + 0] = it;
            //debug[ii*3 + 1] = shpk;
            //debug[ii*3 + 2] = (int)(cell.hyperscore * 10 + 0.5);

            /* hyperscore < 0 means either b- or y- ions were not matched */
            if (cell.hyperscore > 0.01)
            {
                // Update the histogram
                atomicAdd(&histogram[(int)(cell.hyperscore * 10 + 0.5)], 1);

                cell.idxoffset = ixx;
                cell.psid = it;
                cell.sharedions = shpk;

                l_cpsms[threadIdx.x] +=1;

                if (cell.hyperscore > topscores[threadIdx.x].hyperscore)
                {
                    topscores[threadIdx.x].hyperscore = cell.hyperscore;
                    topscores[threadIdx.x].psid       = cell.psid;
                    topscores[threadIdx.x].idxoffset  = cell.idxoffset;
                    topscores[threadIdx.x].sharedions = cell.sharedions;
                }
            }
        }
    }

    __syncthreads();

    // get max dhcell
    hcp::gpu::cuda::s3::getMaxdhCell(topscores, &resPtr->topscore[qnum]);

    // get the cpsms
    hcp::gpu::cuda::s4::ArraySum(l_cpsms, 1024, cpsms);

    // copy histogram to the global memory
    for (int ii = threadIdx.x; ii < HISTOGRAM_SIZE; ii+=blockDim.x)
            survival[ii] += histogram[ii];

/*     // fixme: remove me
    for (int o = threadIdx.x; o < HISTOGRAM_SIZE; o += blockDim.x)
        debug[o] = survival[o];

    debug[HISTOGRAM_SIZE] = (int)(resPtr->topscore->hyperscore* 10 + 0.5);
    debug[HISTOGRAM_SIZE + 1] = (int)resPtr->topscore->psid;
    debug[HISTOGRAM_SIZE+2] = (int)resPtr->topscore->idxoffset;
    debug[HISTOGRAM_SIZE + 3] = (int)resPtr->topscore->sharedions; */

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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        vect[i] += val;
}

// -------------------------------------------------------------------------------------------- //

__global__ void assign_dScores(dScores *a, double_t *survival, dhCell *topscores, int *cpsms)
{
    a->survival = survival;
    a->topscore = topscores;
    a->cpsms = cpsms;
}

// -------------------------------------------------------------------------------------------- //

status_t deinitialize()
{
    hcp::gpu::cuda::s1::freeFragIon();
    hcp::gpu::cuda::s1::freebA();
    // FIXME: why cuda error here
    // even if reallocate, then only one unit mem
    hcp::gpu::cuda::s1::freeATcols();

    freedQueries();

    hcp::gpu::cuda::s4::freed_eValues();

    // sync all streams
    hcp::gpu::cuda::driver::get_instance()->all_streams_sync();

    return SLM_SUCCESS;
}

// -------------------------------------------------------------------------------------------- //

} // namespace s3

} // namespace cuda

} // namespace gpu

} // namespace hcp