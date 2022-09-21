/*
 * Copyright (C) 2019  Muhammad Haseeb, Fahad Saeed
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

#pragma once

#include "algos.cuh"

const int HISTOGRAM_SIZE = (1 + (MAX_HYPERSCORE * 10) + 1);

struct dScores
{
    BYC        *scores;
    double_t *survival;
    dhCell    *topscore;
    int      *cpsms;

    dScores() = default;
    ~dScores();
    void init(int size);
};

__global__ void assign_dScores(dScores *a, double_t *survival, BYC *scores, dhCell *topscores, int *cpsms)
{
    a->survival = survival;
    a->scores = scores;
    a->topscore = topscores;
    a->cpsms = cpsms;
}

void dScores::init(int size)
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    double_t *l_survival;
    BYC        *l_scores;
    dhCell    *l_topscore;
    int      *l_cpsms;

    // allocate memory for the scores
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(l_survival, 10000 * 1024, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(l_scores, size, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(l_topscore, 10000, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(l_cpsms, 10000, driver->stream[0]));

    assign_dScores<<<1,1, 1024, driver->stream[0]>>>(this, l_survival, l_scores, l_topscore, l_cpsms);

    driver->stream_sync(0);
}


dScores::~dScores()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    dScores *h_dScores = new dScores();
    dScores *d_dScores = this;

    // copy pointers from the device
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_dScores, d_dScores, 1, driver->stream[0]));

    // free all memory
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(h_dScores->survival, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(h_dScores->scores, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(h_dScores->topscore, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(h_dScores->cpsms, driver->stream[0]));

    driver->stream_sync(0);

    // free the host memory
    delete h_dScores;
}

// test kernel
template <typename T>
__global__ void vector_add(T *d_a, T *d_b, T *d_c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        d_c[i] = d_a[i] + d_b[i];
    }
}

__global__ void testing(dScores *dscores, int *testarray)
{
    dScores *resPtr = &dscores[blockIdx.x];
    BYC *bycPtr = resPtr->scores;
    //auto *survival = resPtr->survival;

    for (int i = threadIdx.x; i < 500000; i += blockDim.x)
    {
        BYC *pp = bycPtr + i;
        
        bool bc = i;
        bool yc = !bc;

        pp->bc = bc;
        pp->yc = yc;
        pp->iyc= i + 1;
        pp->ibc= i + 2;
    }

    for (int i = threadIdx.x ; i < 500000; i+= blockDim.x)
    {
        BYC *pp = bycPtr + i;
        testarray[i*4 + 0] = pp->bc;
        testarray[i*4 + 1] = pp->yc;
        testarray[i*4 + 2] = pp->iyc;
        testarray[i*4 + 3] = pp->ibc;
    }
}

dScores *& getScorecard(int chunksize)
{
    auto driver = hcp::gpu::cuda::driver::get_instance();
    static thread_local dScores *d_Scores = nullptr;

    if (!d_Scores)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async<dScores>(d_Scores, 10, driver->stream[0]));

        for (int i = 0 ; i < 10 ; i++)
            d_Scores[i].init(chunksize);
    }

    return d_Scores;
}

// tail fit method
__global__ void TailFit(double *survival, double *evalues, int *hyperscore, int *cpsms, double *debug)
{
    // each block will process one result
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    /* Assign to internal variables */
    double *yy = &survival[bid * HISTOGRAM_SIZE];
    int hyp = hyperscore[bid];

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

/*     for (int t = tid; t < HISTOGRAM_SIZE; t += blockDim.x)
    {
        debug[t] = yy[t];
    }
 */
 
    /* Find the curve region */
    rargmax<double_t>(yy, 0, hyp-1, 1.0, end1);
    argmax<double_t>(yy, 0, end1, 1.0, stt1);

    debug[bid * 32] = end1;
    debug[bid * 32 +1] = stt1;

    /* To handle special cases */
    if (stt1 == end1)
    {
        stt1 = end1;
        end1 += 1;
    }


    /* Slice off yyt between stt1 and end1 */
    Assign<double>(p_x, yy + stt1, yy + end1 + 1);

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
        Assign(sx, p_x, p_x + p_x_size);

        /* cumulative_sum(sx) */
        prefixSum(p_x, p_x + p_x_size, sx);

        short sx_size = end1 - stt1 + 1;
        short sx_size_1 = sx_size - 1;

        /* Survival function s(x) */
        for (int j = tid; j < sx_size; j+=blockDim.x)
        {
            // divide by vaa
            sx[j] /= vaa;
            // s(x) = -(s(x) - 1) = 1 - s(x)
            sx[j] = 1 - sx[j];
            // adjust for > 1
            if (sx[j] > 1)
                sx[j] = 0.999;
        }

        __syncthreads();


        /* Adjust for negatives */
        short replacement = 0;
        rargmax<double_t>(sx, (short)0, sx_size_1, (double)(1e-4), replacement);

        debug[bid * 32+2] = replacement;

        for (int j = tid; j < end1 - stt1 + 1; j+=blockDim.x)
        {
            // take care of the case where s(x) < 0
            if (sx[j] < 0)
                sx[j] = sx[replacement];
            sx[j] = log10(sx[j]);
        }

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
            largmax<double_t>(sx, 0, sx_size_1, hgt_22, mark);
            mark -= 1;
            rargmax<double_t>(sx, 0, sx_size_1, hgt_87, mark2);

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
            largmax<double_t>(sx, 0, sx_size-1, (sx[0] + hgt * 0.22), mark);
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

        debug[bid * 32 +3] = mark;
        debug[bid * 32 +4] = mark2;
        debug[bid * 32 +5] = hgt_22;
        debug[bid * 32 +6] = hgt_87;

        // Add Range
        for (int jj = stt + mark + tid; jj <= stt + mark2; jj+=blockDim.x)
            // X->AddRange(mark, mark2);
            X[jj - stt - mark] = jj;

        __syncthreads();

        // update sx_size to mark2 - mark + 1
        sx_size = mark2 - mark + 1;
        sx_size_1 = sx_size - 1;

        // clip s(x) to [mark, mark2]
        for (int jj = tid; jj < sx_size; jj+=blockDim.x)
            // sx->clip(mark, mark2);
            sx[jj] = sx[jj+mark];

        __syncthreads();

        // linear fit between mark and mark2
        LinearFit<double_t>(X, sx, sx_size, &mu, &beta, debug);

        //std::cout << "y = " << mu_t << "x + " << beta_t << std::endl;
        //std::cout << "eValue: " << pow(10, hyp * mu_t + beta_t) * vaa << std::endl;
    }

    debug[bid * 32 + 7] = mu;
    debug[bid * 32 +8] = beta;

    /* Estimate the log(s(x)) */
    double_t lgs_x = (mu * hyp) + beta;

    /* Compute the e(x) = n * s(x) = n * 10^(lg(s(x))) */
    evalues[bid] = vaa * pow(10.0, lgs_x);

    return;
}


// -------------------------------------------------------------------------------------------- //
