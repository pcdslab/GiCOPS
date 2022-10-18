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
#include <iostream>
#include <thread>

#include "dslim_fileout.h"

#include "cuda/driver.hpp"
#include "cuda/superstep4/kernel.hpp"

#ifdef USE_MPI

#include "dslim.h"
#include "dslim_score.h"

#endif // USE_MPI

using namespace std;

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

struct compare_dhCell
{
    __host__ __device__ 
    bool operator()(dhCell lhs, dhCell rhs)
    {
        return lhs.hyperscore < rhs.hyperscore; 
    }
};

// gumbel curve fitting
__global__ void logWeibullFit(dScores_t *d_Scores, double *evalues, short min_cpsm);

// tail fit method
__global__ void TailFit(double *survival, int *cpsms, dhCell *topscore, double *evalues, short min_cpsms);

// alternate tail fit method
__global__ void TailFit(double_t *data, float_t *hyp, int *cpsms, double *evalues, short min_cpsms);

template <typename T>
__device__ void LinearFit(T* x, T* y, int_t n, double_t *a, double_t *b);

template <typename T>
__device__ void argmax(T *data, short i1, short i2, T val, short &out);

template <typename T>
__device__ void largmax(T *data, short i1, short i2, T val, short &out);

template <typename T>
__device__ void rargmax(T *data, short i1, short i2, T val, short &out);

template <typename T>
extern __device__ void Assign(T *p_x, T *beg, T *end);

template <typename T>
extern __device__ void prefixSum(T *beg, T *end, T *out);

template <typename T>
extern __device__ void XYbar(T *x, T *y, int n, double &xbar, double &ybar);

template <typename T>
__device__ void TopBot(T *x, T *y, int n, const double xbar, const double ybar, double &top, double &bot);

// -------------------------------------------------------------------------------------------- //

__host__ double *& getd_eValues()
{
    static auto driver = hcp::gpu::cuda::driver::get_instance();

    static thread_local double *d_evalues = nullptr;

    if (!d_evalues)
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async<double>(d_evalues, QCHUNK, driver->stream[DATA_STREAM]));

    return d_evalues;
}

// -------------------------------------------------------------------------------------------- //

__host__ void freed_eValues()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    auto &&d_evalues = getd_eValues();
    
    if (d_evalues)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_evalues, driver->stream[DATA_STREAM]));
        d_evalues = nullptr;
    }
}

// -------------------------------------------------------------------------------------------- //

// tail fit method
__global__ void TailFit(double *survival, int *cpsms, dhCell *topscore, double *evalues, short min_cpsms)
{
    // each block will process one result
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    /* Assign to internal variables */
    double *yy = &survival[bid * HISTOGRAM_SIZE];

    // make sure the hyp is scaled to int here
    int hyp = (topscore[bid].hyperscore * 10 + 0.5);

    extern __shared__ double shmem[];

    double *p_x = &shmem[0];
    double *sx = &shmem[HISTOGRAM_SIZE];
    double *X = &shmem[2*HISTOGRAM_SIZE];

    double mu = 0.0;
    double beta = 4.0;

    short stt1 = 0;
    short stt =  0;
    short end1 = HISTOGRAM_SIZE - 1;
    //short ends = HISTOGRAM_SIZE - 1;

    /* Find the curve region */
    hcp::gpu::cuda::s4::rargmax<double_t>(yy, 0, hyp-1, 1.0, end1);
    hcp::gpu::cuda::s4::argmax<double_t>(yy, 0, end1, 1.0, stt1);

    /* To handle special cases */
    if (stt1 == end1)
    {
        stt1 = end1;
        end1 += 1;
    }

    /* Slice off yyt between stt1 and end1 */
    hcp::gpu::cuda::s4::Assign<double>(p_x, yy + stt1, yy + end1 + 1);

    /* Database size
     * vaa = accumulate(yy, yy + hyp + 1, 0); */
    int vaa = cpsms[bid];

    /* Check if no distribution data except for hyp */
    if (vaa < min_cpsms)
    {
        //mu = 0;
        //beta = 100;
        evalues[bid] = MAX_HYPERSCORE;
        stt = stt1;
        //ends = end1;
    }
    else
    {
        /* Filter p_x again */
        //ends = end1;
        stt = stt1;

        int p_x_size = end1 - stt1 + 1;

        /* Compute survival function s(x) */
        hcp::gpu::cuda::s4::Assign(sx, p_x, p_x + p_x_size);

        /* cumulative_sum(sx) */
        hcp::gpu::cuda::s4::prefixSum(p_x, p_x + p_x_size, sx);

        short sx_size = end1 - stt1 + 1;
        short sx_size_1 = sx_size - 1;

        /* Survival function s(x) */
        for (int j = tid; j < sx_size; j+=blockDim.x)
        {
            // divide by vaa
            sx[j] = sx[j]/vaa;
            // s(x) = -(s(x) - 1) = 1 - s(x)
            sx[j] = 1 - sx[j];

            // take care of the case where s(x) > 1
            if (sx[j] > 1)
                sx[j] = 0.999;
        }

        __syncthreads();

        /* Adjust for negatives */
        short replacement = 0;
        hcp::gpu::cuda::s4::rargmax<double_t>(sx, (short)0, sx_size_1, (double)(1e-4), replacement);

        // take care of the case where s(x) < 0
        for (int j = tid; j < sx_size; j += blockDim.x)
        {
            if (sx[j] <= 0)
                sx[j] = sx[replacement];
        }

        __syncthreads();

        // compute log10(s(x))
        for (int j = tid; j < sx_size; j += blockDim.x)
            sx[j] = log10f(sx[j]);

        __syncthreads();

        /* Offset markers */
        short mark = 0;
        short mark2 = 0;
        double hgt = sx[sx_size_1] - sx[0];
        auto hgt_22 = sx[0] + hgt * 0.22;
        auto hgt_87 = sx[0] + hgt * 0.87;

        /* If length > 4, then find thresholds */
        if (sx_size > 3)
        {
            hcp::gpu::cuda::s4::largmax<double_t>(sx, 0, sx_size_1, hgt_22, mark);
            mark -= 1;
            hcp::gpu::cuda::s4::rargmax<double_t>(sx, 0, sx_size_1, hgt_87, mark2);

            if (mark2 == sx_size)
            {
                mark2 -= 1;
            }

            /* To handle special cases */
            if (mark >= mark2)
            {
                mark = mark2 - 1;
            }
        }
        /* If length < 4 business as usual */
        else if (sx_size == 3)
        {
            /* Mark the start of the regression point */
            hcp::gpu::cuda::s4::largmax<double_t>(sx, 0, sx_size_1, hgt_22, mark);
            mark -= 1;
            mark2 = sx_size - 1;

            /* To handle special cases */
            if (mark >= mark2)
            {
                mark = mark2 - 1;
            }
        }
        else
        {
            mark = 0;
            mark2 = sx_size - 1;
        }

        __syncthreads();

        for (int jj = stt + mark + tid; jj <= stt + mark2; jj+=blockDim.x)
            // X->AddRange(mark, mark2);
            X[jj - stt - mark] = jj;

        __syncthreads();

        // update sx_size to mark2 - mark + 1
        sx_size = mark2 - mark + 1;
        sx_size_1 = sx_size - 1;

        for (int jj = tid; jj <= (mark2-mark); jj+=blockDim.x)
            // sx->clip(mark, mark2);
            sx[jj] = sx[jj+mark];

        __syncthreads();

        hcp::gpu::cuda::s4::LinearFit<double_t>(X, sx, sx_size, &mu, &beta);

        /* Estimate the log(s(x)) */
        double_t lgs_x = (mu * hyp) + beta;

        /* Compute the e(x) = n * s(x) = n * 10^(lg(s(x))) */
        evalues[bid] = vaa * pow(10.0, lgs_x);
    }
}

// -------------------------------------------------------------------------------------------- //

// tail fit method
__global__ void TailFit(double_t *data, float *hyps, int *cpsms, double *evalues, short min_cpsms)
{
    // each block will process one result
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    /* Assign to internal variables */
    auto yy = &data[HISTOGRAM_SIZE * bid];

    // make sure the hyp is scaled to int here
    int hyp = hyps[bid] * 10 + 0.5;

    extern __shared__ double shmem[];

    double *p_x = &shmem[0];
    double *sx = &shmem[HISTOGRAM_SIZE];
    double *X = &shmem[2*HISTOGRAM_SIZE];

    double mu = 0.0;
    double beta = 4.0;

    short stt1 = 0;
    short stt =  0;
    short end1 = HISTOGRAM_SIZE - 1;
    //short ends = HISTOGRAM_SIZE - 1;

    /* Find the curve region */
    hcp::gpu::cuda::s4::rargmax<double_t>(yy, 0, hyp-1, 1.0, end1);
    hcp::gpu::cuda::s4::argmax<double_t>(yy, 0, end1, 1.0, stt1);

    /* To handle special cases */
    if (stt1 == end1)
    {
        stt1 = end1;
        end1 += 1;
    }

    /* Slice off yyt between stt1 and end1 */
    hcp::gpu::cuda::s4::Assign<double>(p_x, yy + stt1, yy + end1 + 1);

    /* Database size
     * vaa = accumulate(yy, yy + hyp + 1, 0); */
    int vaa = cpsms[bid];

    /* Check if no distribution data except for hyp */
    if (vaa < min_cpsms)
    {
        mu = 0;
        beta = 100;
        stt = stt1;
        evalues[bid] = MAX_HYPERSCORE;
        //ends = end1;
    }
    else
    {
        /* Filter p_x again */
        //ends = end1;
        stt = stt1;

        int p_x_size = end1 - stt1 + 1;

        /* Compute survival function s(x) */
        hcp::gpu::cuda::s4::Assign(sx, p_x, p_x + p_x_size);

        /* cumulative_sum(sx) */
        hcp::gpu::cuda::s4::prefixSum(p_x, p_x + p_x_size, sx);

        short sx_size = end1 - stt1 + 1;
        short sx_size_1 = sx_size - 1;

        /* Survival function s(x) */
        for (int j = tid; j < sx_size; j+=blockDim.x)
        {
            // divide by vaa
            sx[j] = sx[j]/vaa;
            // s(x) = -(s(x) - 1) = 1 - s(x)
            sx[j] = 1 - sx[j];

            // take care of the case where s(x) > 1
            if (sx[j] > 1)
                sx[j] = 0.999;
        }

        __syncthreads();

        /* Adjust for negatives */
        short replacement = 0;
        hcp::gpu::cuda::s4::rargmax<double_t>(sx, (short)0, sx_size_1, (double)(1e-4), replacement);

        // take care of the case where s(x) < 0
        for (int j = tid; j < sx_size; j += blockDim.x)
        {
            if (sx[j] <= 0)
                sx[j] = sx[replacement];
        }

        __syncthreads();

        // compute log10(s(x))
        for (int j = tid; j < sx_size; j += blockDim.x)
            sx[j] = log10f(sx[j]);

        __syncthreads();

        /* Offset markers */
        short mark = 0;
        short mark2 = 0;
        auto hgt = sx[sx_size - 1] - sx[0];
        auto hgt_22 = sx[0] + hgt * 0.22;
        auto hgt_87 = sx[0] + hgt * 0.87;

        /* If length > 4, then find thresholds */
        if (sx_size > 3)
        {
            hcp::gpu::cuda::s4::largmax<double_t>(sx, 0, sx_size_1, hgt_22, mark);
            mark -= 1;
            hcp::gpu::cuda::s4::rargmax<double_t>(sx, 0, sx_size_1, hgt_87, mark2);

            if (mark2 == sx_size)
            {
                mark2 -= 1;
            }

            /* To handle special cases */
            if (mark >= mark2)
            {
                mark = mark2 - 1;
            }
        }
        /* If length < 4 business as usual */
        else if (sx_size == 3)
        {
            /* Mark the start of the regression point */
            hcp::gpu::cuda::s4::largmax<double_t>(sx, 0, sx_size_1, hgt_22, mark);
            mark -= 1;
            mark2 = sx_size - 1;

            /* To handle special cases */
            if (mark >= mark2)
            {
                mark = mark2 - 1;
            }
        }
        else
        {
            mark = 0;
            mark2 = sx_size - 1;
        }

        __syncthreads();

        for (int jj = stt + mark + tid; jj <= stt + mark2; jj+=blockDim.x)
            // X->AddRange(mark, mark2);
            X[jj - stt - mark] = jj;

        __syncthreads();

        // update sx_size to mark2 - mark + 1
        sx_size = mark2 - mark + 1;
        sx_size_1 = sx_size - 1;

        for (int jj = tid; jj <= (mark2-mark); jj+=blockDim.x)
            // sx->clip(mark, mark2);
            sx[jj] = sx[jj+mark];

        __syncthreads();

        hcp::gpu::cuda::s4::LinearFit<double_t>(X, sx, sx_size, &mu, &beta);

        /* Estimate the log(s(x)) */
        double_t lgs_x = (mu * hyp) + beta;

        /* Compute the e(x) = n * s(x) = n * 10^(lg(s(x))) */
        evalues[bid] = vaa * pow(10.0, lgs_x);
    }

}

// -------------------------------------------------------------------------------------------- //

__host__ status_t processResults(Index *index, Queries<spectype_t> *gWorkPtr, int currspecID)
{
    status_t status = SLM_SUCCESS;

    // get driver object
    auto driver = hcp::gpu::cuda::driver::get_instance();

    // get scorecard instance
    auto d_Scores = hcp::gpu::cuda::s3::getScorecard();

    auto d_evalues = getd_eValues();

    int numSpecs = gWorkPtr->numSpecs;

    // get top scores back to CPU memory
    dhCell *h_topscore = new dhCell[numSpecs];

    // transfer data to host
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_topscore, d_Scores->topscore, numSpecs, driver->stream[DATA_STREAM]));

    // get the cell with the largest topscore (hyperscore)
    dhCell *largest = std::max_element(h_topscore, h_topscore + numSpecs, compare_dhCell());

    // make sure the data stream is in sync
    driver->stream_sync(DATA_STREAM);

    // max nthreads per block
    const int nthreads = 1024;

    // use the top hscore to set the number of threads per block
    int maxhscore = (largest->hyperscore * 10 + 0.5) + 1;

    int blockSize = std::min(nthreads, HISTOGRAM_SIZE);
    blockSize = std::min(blockSize, maxhscore);

    // contingency in case blockSize comes out to zero
    blockSize = std::max(blockSize, 256);


#if defined (TAILFIT) || true

    // use function pointers to point to the correct overload
    auto TailFit_ptr = static_cast<void (*)(double *, int *, dhCell *, double *, short)>(&TailFit);

    // IMPORTANT: make sure at least 32KB+ shared memory is available to the TailFit kernel
    cudaFuncSetAttribute(*TailFit_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, KBYTES(32));

    // the tailfit kernel
    hcp::gpu::cuda::s4::TailFit<<<numSpecs, blockSize, KBYTES(32), driver->get_stream(SEARCH_STREAM)>>>(d_Scores->survival, d_Scores->cpsms, d_Scores->topscore, d_evalues, (short)params.min_cpsm);

#else

    // IMPORTANT: make sure at least 32KB+ shared memory is available to the logWeibullfit kernel
    //cudaFuncSetAttribute(logWeibullFit, cudaFuncAttributeMaxDynamicSharedMemorySize, KBYTES(32));

    // the logWeibullfit kernel
    //hcp::gpu::cuda::s4::logWeibullFit<<<numSpecs, blockSize, KBYTES(32), driver->get_stream(SEARCH_STREAM)>>>(d_Scores, d_evalues, min_cpsm);

#endif // TAILFIT

    // synchronize the search stream
    driver->stream_sync(SEARCH_STREAM);

    // asynchronously copy the dhCell and cpsms to hostmem for writing to file
    int *h_cpsms = new int[numSpecs];
    double *h_evalues = new double[numSpecs];

    // transfer data to host
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_cpsms, d_Scores->cpsms, numSpecs, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_evalues, d_evalues, numSpecs, driver->stream[DATA_STREAM]));

    // synchronize the stream
    driver->stream_sync(DATA_STREAM);

    // write all results to file
    for (int s = 0; s < numSpecs; s++)
    {
        if (h_evalues[s] < params.expect_max)
        {
            hCell psm;

            psm.idxoffset = h_topscore[s].idxoffset;
            psm.hyperscore = h_topscore[s].hyperscore;
            psm.sharedions = h_topscore[s].sharedions;
            psm.psid = h_topscore[s].psid;
            psm.totalions = (index[psm.idxoffset].pepIndex.peplen - 1) * params.maxz * iSERIES;
            psm.rtime = gWorkPtr->rtimes[s];
            psm.pchg = gWorkPtr->charges[s];
            psm.fileIndex = gWorkPtr->fileNum;

            /* Printing the scores in OpenMP mode */
            status = DFile_PrintScore(index, currspecID + s, gWorkPtr->precurse[s], &psm, h_evalues[s], h_cpsms[s]);
        }
    }

    // delete the temp memory
    delete[] h_topscore;
    delete[] h_cpsms;
    delete[] h_evalues;

    h_topscore = nullptr;
    h_cpsms = nullptr;
    h_evalues = nullptr;

    return status;
}

// -------------------------------------------------------------------------------------------- //

#ifdef USE_MPI

__host__ status_t getIResults(Index *index, Queries<spectype_t> *gWorkPtr, int currSpecID, hCell *CandidatePSMS)
{
    status_t status = SLM_SUCCESS;

    ebuffer *liBuff = nullptr;
    partRes *txArray = nullptr;

    liBuff = new ebuffer;

    txArray = liBuff->packs;
    liBuff->isDone = false;
    liBuff->batchNum = gWorkPtr->batchNum;

    int numSpecs = gWorkPtr->numSpecs;

   // get driver object
    auto driver = hcp::gpu::cuda::driver::get_instance();

    // get device scorecard instance
    auto d_Scores = hcp::gpu::cuda::s3::getScorecard();

    // host dScores
    dScores_t *h_dScores = new hcp::gpu::cuda::s3::dScores();

    // copy pointers from the device
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_dScores, d_Scores, 1, driver->stream[DATA_STREAM]));

    driver->stream_sync(DATA_STREAM);

    // asynchronously copy the dhCell and cpsms to hostmem for writing to file
    dhCell *h_topscore = new dhCell[numSpecs];
    int *h_cpsms = new int[numSpecs];
    double *h_survival = new double [HISTOGRAM_SIZE * numSpecs];

    // transfer data to host
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_topscore, h_dScores->topscore, numSpecs, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_cpsms, h_dScores->cpsms, numSpecs, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_survival, h_dScores->survival, HISTOGRAM_SIZE * numSpecs, driver->stream[DATA_STREAM]));

    // synchronize the stream
    driver->stream_sync(DATA_STREAM);

    // process all results
    for (int s = 0; s < numSpecs; s++)
    {
        if (h_cpsms[s] >= 1)
        {
            hCell psm;

            psm.idxoffset = h_topscore[s].idxoffset;
            psm.hyperscore = h_topscore[s].hyperscore;
            psm.sharedions = h_topscore[s].sharedions;
            psm.psid = h_topscore[s].psid;
            psm.totalions = (index[psm.idxoffset].pepIndex.peplen - 1) * params.maxz * iSERIES;
            psm.rtime = gWorkPtr->rtimes[s];
            psm.pchg = gWorkPtr->charges[s];
            psm.fileIndex = gWorkPtr->fileNum;

            /* Put it in the list */
            CandidatePSMS[s] = psm;

            // FIXME: Is this needed?
            //resPtr->maxhypscore = (psm.hyperscore * 10 + 0.5);

            auto &&minnext = expeRT::StoreIResults(&h_survival[s * HISTOGRAM_SIZE], s, h_cpsms[s], liBuff);

            /* Fill in the Tx array cells */
            txArray[s].min  = minnext[0]; // minhypscore
            txArray[s].max2 = minnext[1]; // nexthypscore
            txArray[s].max  = psm.hyperscore;
            txArray[s].N    = h_cpsms[s];
            txArray[s].qID  = currSpecID + s;
        }
    }

    // add liBuff to sub-task K
    if (params.nodes > 1)
    {
        liBuff->currptr = numSpecs * Xsamples * sizeof(ushort_t);
        AddliBuff(liBuff);
    }

    // delete the temp memory
    delete[] h_topscore;
    delete[] h_cpsms;
    delete[] h_survival;

    delete h_dScores;

    h_topscore = nullptr;
    h_cpsms = nullptr;
    h_survival = nullptr;
    h_dScores = nullptr;

    return status;
}

#endif // USE_MPI

// -------------------------------------------------------------------------------------------- //

template <class T>
__device__ void LinearFit(T* x, T* y, int n, double *a, double *b)
{
    double bot;
    double top;
    double xbar;
    double ybar;

    //
    //  Special case.
    //
    if (n == 1)
    {
        *a = 0.0;
        *b = y[0];
    }
    else
    {
        //
        //  Average X and Y.
        //
        xbar = 0.0;
        ybar = 0.0;

        hcp::gpu::cuda::s4::XYbar<double>(x, y, n, xbar, ybar);

        xbar = xbar / (double) n;
        ybar = ybar / (double) n;

        //
        //  Compute Beta.
        //

        top = 0.0;
        bot = 0.0;

        hcp::gpu::cuda::s4::TopBot<double>(x, y, n, xbar, ybar, top, bot);

        *a = top / bot;
        *b = ybar - (*a) * xbar;
    }

    return;
}

// -------------------------------------------------------------------------------------------- //

__host__ void processResults(double *h_data, float *h_hyp, int *h_cpsms, double *h_evalues, int bsize)
{
    double_t *d_data;
    int *d_cpsms;
    float *d_hyp;
    double *d_evalues;

    // driver
    auto driver = hcp::gpu::cuda::driver::get_instance();

    // allocate device memory
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_data, HISTOGRAM_SIZE * bsize, driver->get_stream()));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_cpsms, bsize, driver->get_stream()));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_hyp, bsize, driver->get_stream()));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_evalues, bsize, driver->get_stream()));

    // H2D
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_data, h_data, HISTOGRAM_SIZE * bsize, driver->get_stream()));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_cpsms, h_cpsms, bsize, driver->get_stream()));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_hyp, h_hyp, bsize, driver->get_stream()));

    const int nthreads = 1024;
    int blockSize = std::min(nthreads, HISTOGRAM_SIZE);

#if defined (TAILFIT) || true

    // use function pointers to point to the correct overload
    auto TailFit_ptr = static_cast<void (*)(double *, float *, int *, double *, short)>(&TailFit);

    // IMPORTANT: make sure at least 32KB+ shared memory is available to the TailFit kernel
    cudaFuncSetAttribute(*TailFit_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, KBYTES(32));

    // the tailfit kernel
    hcp::gpu::cuda::s4::TailFit<<<bsize, blockSize, KBYTES(32), driver->get_stream()>>>(d_data, d_hyp, d_cpsms, d_evalues, (short)params.min_cpsm);
#else
    // IMPORTANT: make sure at least 32KB+ shared memory is available to the logWeibullfit kernel
    //cudaFuncSetAttribute(logWeibullFit, cudaFuncAttributeMaxDynamicSharedMemorySize, KBYTES(32));

    // the logWeibullfit kernel
    //hcp::gpu::cuda::s4::logWeibullFit<<<numSpecs, blockSize, KBYTES(32), driver->get_stream(SEARCH_STREAM)>>>(d_Scores, d_evalues, min_cpsm);
#endif // TAILFIT

    // D2H
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_evalues, d_evalues, bsize, driver->get_stream()));

    // free device memory
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_evalues, driver->get_stream()));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_data, driver->get_stream()));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_cpsms, driver->get_stream()));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_hyp, driver->get_stream()));

    // synchronize
    driver->stream_sync();
}

// -------------------------------------------------------------------------------------------- //

} // namespace s4
} // namespace cuda
} // namespace gpu
} // namespace hcp

// -------------------------------------------------------------------------------------------- //

#if defined (USE_GPU) && defined (USE_MPI)

status_t DSLIM_Score::GPUCombineResults()
{
    status_t status = SLM_SUCCESS;
    string_t fn;

    /* Each node sent its sample */
    const int_t nSamples = params.nodes;
    auto startSpec= 0;
    ifstream *fhs = NULL;
    ebuffer  *iBuffs = NULL;

    if (this->myRXsize > 0)
    {
        fhs = new ifstream[nSamples];
        iBuffs = new ebuffer[nSamples];
    }

    auto vbatch = params.myid;

    for (auto batchNum = 0; batchNum < this->nBatches; batchNum++, vbatch += nSamples)
    {
#if defined (PROGRESS)
        if (params.myid == 0)
            std::cout << "\rDONE:\t\t" << (batchNum * 100) /this->nBatches << "%";
#endif // PROGRESS

        auto bSize = sizeArray[batchNum];

        double_t *h_data;
        int *h_cpsms;
        float_t *h_hyp;
        double *h_evalues;
        int *h_keys = new int[bSize];

        hcp::gpu::cuda::error_check(hcp::gpu::cuda::host_pinned_allocate<double_t>(h_data, expeRT::SIZE * bSize));
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::host_pinned_allocate(h_cpsms, bSize));
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::host_pinned_allocate(h_hyp, bSize));
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::host_pinned_allocate(h_evalues, bSize));

        // read data from .dat files
        for (int_t saa = 0; saa < nSamples; saa++)
        {
            fn = params.workspace + "/" + std::to_string(vbatch) + "_"
                    + std::to_string(saa) + ".dat";

            fhs[saa].open(fn, ios::in | ios::binary);

            if (fhs[saa].is_open())
            {
                fhs[saa].read((char_t *)iBuffs[saa].packs, bSize * sizeof(partRes));
                fhs[saa].read(iBuffs[saa].ibuff, bSize * Xsamples * sizeof(ushort_t));

                if (fhs[saa].fail())
                {
                    status = ERR_FILE_NOT_FOUND;
                    std::cout << "FATAL: File Read Failed" << std::endl;
                    exit(status);
                }
            }

            fn.clear();
        }

#ifdef USE_OMP
#pragma omp parallel for schedule (dynamic, 4) num_threads(params.threads)
#endif /* USE_OMP */
        for (int_t spec = 0; spec < bSize; spec++)
        {
            int_t thno = omp_get_thread_num();

            /* Results pointer to use */
            expeRT *expPtr = this->ePtr + thno;

            // reset the host vectors
            h_cpsms[spec] = 0;
            h_evalues[spec] = params.expect_max;

            for (int ij = 0; ij < expeRT::SIZE; ij++)
                h_data[spec * expeRT::SIZE + ij] = 0;

            /* Record locators */
            h_keys[spec] = params.nodes;
            h_hyp[spec] = -1;

            /* For all samples, update the histogram */
            for (int_t sno = 0; sno < nSamples; sno++)
            {
                /* Pointer to Result sample */
                partRes *sResult = iBuffs[sno].packs + spec;

                if (*sResult == 0)
                    continue;

                /* Update the number of samples */
                h_cpsms[spec] += sResult->N;

                /* Only take out data if present */
                if (sResult->N >= 1)
                {
                    /* Reconstruct the partial histogram */
                    // expPtr->Reconstruct(&iBuffs[sno], spec, sResult);
                    expPtr->Reconstruct(&iBuffs[sno], spec, sResult, &h_data[spec * expeRT::SIZE]);

                    /* Record the maxhypscore and its key */
                    if (sResult->max > 0 && sResult->max > h_cpsms[spec])
                    {
                        h_hyp[spec] = sResult->max;
                        h_keys[spec] = sno;
                    }
                }
            }
        }

        // compute evalues
        hcp::gpu::cuda::s4::processResults(h_data, h_hyp, h_cpsms, h_evalues, bSize);

#ifdef USE_OMP
#pragma omp parallel for schedule (dynamic, 4) num_threads(params.threads)
#endif /* USE_OMP */
        for (int_t spec = 0; spec < bSize; spec++)
        {
            int_t thno = omp_get_thread_num();

            /* Results pointer to use */
            expeRT *expPtr = this->ePtr + thno;

            /* Combine the fResult */
            fResult *psm = &TxValues[startSpec + spec];

            /* Need further processing only if enough results */
            if (h_keys[spec] < (int_t) params.nodes && h_cpsms[spec] >= (int_t) params.min_cpsm)
            {
                double_t e_x = h_evalues[spec];

                /* If the scores are good enough */
                if (e_x < params.expect_max)
                {
                    partRes *ssResult = iBuffs[h_keys[spec]].packs + spec;

                    psm->eValue = e_x * 1e6;
                    psm->specID = ssResult-> qID;
                    psm->npsms = h_cpsms[spec];

                    /* Must be an atomic update */
#ifdef USE_OMP
#pragma omp atomic update
                    txSizes[h_keys[spec]] += 1;
#else
                    txSizes[h_keys[spec]] += 1;
#endif /* USE_OMP */

                    /* Update the key */
                    keys[startSpec + spec] = h_keys[spec];
                }
                else
                {
                    psm->eValue = params.expect_max * 1e6;
                    psm->specID = -1;
                    psm->npsms = 0;
                    keys[startSpec + spec] = params.nodes;
                }
            }
            else
            {
                expPtr->ResetPartialVectors();
                psm->eValue = params.expect_max * 1e6;
                psm->specID = -1;
                psm->npsms = 0;
                keys[startSpec + spec] = params.nodes;
            }
        }

        /* Update the counters */
        startSpec += sizeArray[batchNum];

        /* Remove the files when no longer needed */
        for (int_t saa = 0; saa < nSamples; saa++)
        {
            fn = params.workspace + "/" + std::to_string(vbatch) + "_" + std::to_string(saa)
                    + ".dat";

            if (fhs[saa].is_open())
            {
                fhs[saa].close();
                std::remove((const char_t *) fn.c_str());
            }

            fn.clear();
        }
    }

    if (this->myRXsize > 0)
    {
        delete[] fhs;
        fhs = NULL;

        delete[] iBuffs;
        iBuffs = NULL;
    }

    /* Check if we have RX data */
    if (myRXsize > 0)
    {
        /* Sort the TxValues by keys (mchID) */
        KeyVal_Parallel<int_t, fResult>(keys, TxValues, myRXsize, params.threads);
    }
    else
    {
        /* Set all sizes to zero */
        for (uint_t ky = 0; ky < params.nodes - 1; ky++)
            txSizes[ky] = 0;
    }

    /* return the status of execution */
    return status;
}

#endif /* USE_GPU */