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

// FIXME

#ifdef USE_MPI

#include "dslim.h"

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

// gumbel curve fitting
__global__ void logWeibullFit(dScores_t *d_Scores, double *evalues, short min_cpsm);

// tail fit method
__global__ void TailFit(dScores_t *d_Scores, double *evalues);

// alternate tail fit method
__global__ void TailFit(double_t *data, float_t *hyp, int *cpsms, double *evalues);

template <class T>
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
extern __device__ void partialSum(T *beg, T *end, T *out);

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
__global__ void TailFit(dScores_t *d_Scores, double *evalues)
{
    // each block will process one result
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    /* Assign to internal variables */
    auto yy = d_Scores[bid].survival;

    // make sure the hyp is scaled to int here
    int hyp = (d_Scores[bid].topscore->hyperscore * 10 + 0.5);

    double p_x_mem[HISTOGRAM_SIZE];
    double s_x_mem[HISTOGRAM_SIZE];
    double x_axis_mem[HISTOGRAM_SIZE];

    double *p_x = &p_x_mem[0];
    double *sx = &s_x_mem[0];
    double *X = &x_axis_mem[0];

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
    int vaa = d_Scores[bid].cpsms[tid];

    /* Check if no distribution data except for hyp */
    if (vaa < 1)
    {
        mu = 0;
        beta = 100;
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
        hcp::gpu::cuda::s4::partialSum(p_x, p_x + p_x_size, sx);

        /* Adjust for negatives */
        short replacement = 0;

        short sx_size = end1 - stt1 + 1;
        short sx_size_1 = sx_size - 1;
        hcp::gpu::cuda::s4::rargmax<double_t>(sx, (short)0, sx_size_1, (double)(1e-4), replacement);

        /* Survival function s(x) */
        for (int j = tid; j < end1 - stt1 + 1; j+=blockDim.x)
        {
            // divide by vaa
            sx[j] /= vaa;
            // s(x) = -(s(x) - 1) = 1 - s(x)
            sx[j] = 1 - sx[j];

            // take care of the case where s(x) > 1
            if (sx[j] > 1)
                sx[j] = 0.999;
            // take care of the case where s(x) < 0
            else if (sx[j] < 0)
                sx[j] = sx[replacement];

            // log10(s(x))
            sx[j] = log(sx[j]);
        }

        __syncthreads();

        /* Offset markers */
        short mark = 0;
        short mark2 = 0;
        auto hgt = sx[sx_size - 1] - sx[0];

        /* If length > 4, then find thresholds */
        if (sx_size > 3)
        {
            hcp::gpu::cuda::s4::largmax<double_t>(sx, 0, sx_size-1, sx[0] + hgt * 0.22, mark);
            mark -= 1;
            hcp::gpu::cuda::s4::rargmax<double_t>(sx, 0, sx_size-1, sx[0] + hgt*0.87, mark2);

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
            hcp::gpu::cuda::s4::largmax<double_t>(sx, 0, sx_size-1, (sx[0] + hgt * 0.22), mark);
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


        for (int jj = stt + tid; jj <= stt + mark2; jj+=blockDim.x)
            // X->AddRange(mark, mark2);
            X[jj - stt] = jj;

        __syncthreads();

        for (int jj = tid; jj <= (mark2-mark); jj+=blockDim.x)
            // sx->clip(mark, mark2);
            sx[jj] = sx[jj+mark];

        __syncthreads();

        hcp::gpu::cuda::s4::LinearFit<double_t>(X, sx, sx_size, &mu, &beta);

        //std::cout << "y = " << mu_t << "x + " << beta_t << std::endl;
        //std::cout << "eValue: " << pow(10, hyp * mu_t + beta_t) * vaa << std::endl;
    }

    /* Estimate the log(s(x)) */
    double_t lgs_x = (mu * hyp) + beta;

    /* Compute the e(x) = n * s(x) = n * 10^(lg(s(x))) */
    evalues[bid] = d_Scores[bid].cpsms[tid] * pow(10.0, lgs_x);

}

// -------------------------------------------------------------------------------------------- //

// tail fit method
__global__ void TailFit(double_t *data, float *hyps, int *cpsms, double *evalues)
{
    // each block will process one result
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    /* Assign to internal variables */
    auto yy = &data[HISTOGRAM_SIZE * bid];

    // make sure the hyp is scaled to int here
    int hyp = hyps[bid] * 10 + 0.5;

    double p_x_mem[HISTOGRAM_SIZE];
    double s_x_mem[HISTOGRAM_SIZE];
    double x_axis_mem[HISTOGRAM_SIZE];

    double *p_x = &p_x_mem[0];
    double *sx = &s_x_mem[0];
    double *X = &x_axis_mem[0];

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
    if (vaa < 1)
    {
        mu = 0;
        beta = 100;
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
        hcp::gpu::cuda::s4::partialSum(p_x, p_x + p_x_size, sx);

        /* Adjust for negatives */
        short replacement = 0;

        short sx_size = end1 - stt1 + 1;
        short sx_size_1 = sx_size - 1;
        hcp::gpu::cuda::s4::rargmax<double_t>(sx, (short)0, sx_size_1, (double)(1e-4), replacement);

        /* Survival function s(x) */
        for (int j = tid; j < end1 - stt1 + 1; j+=blockDim.x)
        {
            // divide by vaa
            sx[j] /= vaa;
            // s(x) = -(s(x) - 1) = 1 - s(x)
            sx[j] = 1 - sx[j];

            // take care of the case where s(x) > 1
            if (sx[j] > 1)
                sx[j] = 0.999;
            // take care of the case where s(x) < 0
            else if (sx[j] < 0)
                sx[j] = sx[replacement];

            // log10(s(x))
            sx[j] = log(sx[j]);
        }

        __syncthreads();

        /* Offset markers */
        short mark = 0;
        short mark2 = 0;
        auto hgt = sx[sx_size - 1] - sx[0];

        /* If length > 4, then find thresholds */
        if (sx_size > 3)
        {
            hcp::gpu::cuda::s4::largmax<double_t>(sx, 0, sx_size-1, sx[0] + hgt * 0.22, mark);
            mark -= 1;
            hcp::gpu::cuda::s4::rargmax<double_t>(sx, 0, sx_size-1, sx[0] + hgt*0.87, mark2);

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
            hcp::gpu::cuda::s4::largmax<double_t>(sx, 0, sx_size-1, (sx[0] + hgt * 0.22), mark);
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


        for (int jj = stt + tid; jj <= stt + mark2; jj+=blockDim.x)
            // X->AddRange(mark, mark2);
            X[jj - stt] = jj;

        __syncthreads();

        for (int jj = tid; jj <= (mark2-mark); jj+=blockDim.x)
            // sx->clip(mark, mark2);
            sx[jj] = sx[jj+mark];

        __syncthreads();

        hcp::gpu::cuda::s4::LinearFit<double_t>(X, sx, sx_size, &mu, &beta);

        //std::cout << "y = " << mu_t << "x + " << beta_t << std::endl;
        //std::cout << "eValue: " << pow(10, hyp * mu_t + beta_t) * vaa << std::endl;
    }

    /* Estimate the log(s(x)) */
    double_t lgs_x = (mu * hyp) + beta;

    /* Compute the e(x) = n * s(x) = n * 10^(lg(s(x))) */
    evalues[bid] = cpsms[bid] * pow(10.0, lgs_x);

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

    int blockSize = std::min(1024, HISTOGRAM_SIZE);
    // short min_cpsm = params.min_cpsm;

    // make sure the data stream is in sync
    driver->stream_sync(DATA_STREAM);

#if defined (TAILFIT) || true

    // use function pointers to point to the correct overload
    auto TailFit_ptr = static_cast<void (*)(dScores_t *, double *)>(&TailFit);

    // IMPORTANT: make sure at least 32KB+ shared memory is available to the TailFit kernel
    cudaFuncSetAttribute(*TailFit_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, KBYTES(48));

    // the tailfit kernel
    hcp::gpu::cuda::s4::TailFit<<<numSpecs, blockSize, KBYTES(48), driver->get_stream(SEARCH_STREAM)>>>(d_Scores, d_evalues);
#else
    // IMPORTANT: make sure at least 32KB+ shared memory is available to the logWeibullfit kernel
    //cudaFuncSetAttribute(logWeibullFit, cudaFuncAttributeMaxDynamicSharedMemorySize, KBYTES(48));

    // the logWeibullfit kernel
    //hcp::gpu::cuda::s4::logWeibullFit<<<numSpecs, blockSize, KBYTES(48), driver->get_stream(SEARCH_STREAM)>>>(d_Scores, d_evalues, min_cpsm);
#endif // TAILFIT

    // synchronize the search stream
    driver->stream_sync(SEARCH_STREAM);

    // host dScores
    hcp::gpu::cuda::s3::dScores *h_dScores = new hcp::gpu::cuda::s3::dScores();

    // copy pointers from the device
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_dScores, d_Scores, 1, driver->stream[DATA_STREAM]));

    // asynchronously copy the dhCell and cpsms to hostmem for writing to file
    dhCell *h_topscore = new dhCell[numSpecs];
    int *h_cpsms = new int[numSpecs];
    double *h_evalues = new double[numSpecs];

    // transfer data to host
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_topscore, h_dScores->topscore, numSpecs, driver->stream[DATA_STREAM]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_cpsms, h_dScores->cpsms, numSpecs, driver->stream[DATA_STREAM]));
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

    delete h_dScores;

    h_topscore = nullptr;
    h_cpsms = nullptr;
    h_evalues = nullptr;
    h_dScores = nullptr;

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

    int blockSize = std::min(1024, HISTOGRAM_SIZE);

#if defined (TAILFIT) || true

    // use function pointers to point to the correct overload
    auto TailFit_ptr = static_cast<void (*)(double *, float *, int *, double *)>(&TailFit);

    // IMPORTANT: make sure at least 32KB+ shared memory is available to the TailFit kernel
    cudaFuncSetAttribute(*TailFit_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, KBYTES(48));

    // the tailfit kernel
    hcp::gpu::cuda::s4::TailFit<<<bsize, blockSize, KBYTES(48), driver->get_stream()>>>(d_data, d_hyp, d_cpsms, d_evalues);
#else
    // IMPORTANT: make sure at least 32KB+ shared memory is available to the logWeibullfit kernel
    //cudaFuncSetAttribute(logWeibullFit, cudaFuncAttributeMaxDynamicSharedMemorySize, KBYTES(48));

    // the logWeibullfit kernel
    //hcp::gpu::cuda::s4::logWeibullFit<<<numSpecs, blockSize, KBYTES(48), driver->get_stream(SEARCH_STREAM)>>>(d_Scores, d_evalues, min_cpsm);
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
