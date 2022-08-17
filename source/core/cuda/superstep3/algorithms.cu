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
#include <cuda_runtime.h>
#include "cuda/driver.hpp"
#include "cuda/superstep3/kernel.hpp"

namespace hcp
{

namespace gpu
{

namespace cuda
{

namespace s3
{

__device__ void lower_bound(uint_t *data, int size, int *lbound, int t)
{
    int n = size;
    *lbound = -1;

    // if there are no elements in nums
    if(n==0 || data[n-1] < t)
        return;
    else if (data[0] >= t)
    {
        *lbound = 0;
        return;
    }

    // initialized low(l), and high(r)
    int l=0;
    int r = n-1;
    int m = l + (r-l)/2;

    while(l <= r)
    {
        m = l + (r-l)/2;
        if(data[m]>=t)
            r = m-1;
        else
            l = m+1;
    }

    *lbound = l;

    return;
}

__device__ void upper_bound(uint_t *data, int size, int *ubound, int t)
{
    int n = size;
    *ubound = -1;

    // if there are no elements in nums
    if(n==0 || data[0] > t)
        return;
    else if (data[n-1] <= t)
    {
        *ubound = n-1;
        return;
    }

    // initialized low(l), and high(r)
    int l=0;
    int r = n-1;
    int m = l + (r-l)/2;

    while(l <= r)
    {
        m = l + (r-l)/2;
        if(data[m]<=t)
            l = m+1;
        else
            r = m-1;
    }

    *ubound = r;

    return;
}

__device__ void compute_minmaxions(int *minions, int *maxions, int *QAPtr, uint *d_bA, uint *d_iA, int dF, int qspeclen, int speclen, int minlimit, int maxlimit)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int irange = qspeclen * (2*dF+1);

    // fixme
    int maxmass = 5000;
    int scale = 100;

    for (int ion = 0; ion < irange ; ion += blockDim.x)
    {
        auto myion = QAPtr[tid/(2*dF+1)];
        int myion_offset = (ion % (2*dF+1)) - dF;

        auto qion = myion + myion_offset;

        if (qion > dF && qion < ((maxmass * scale) - 1 - dF))
        {
            // locate iAPtr start and end
            uint_t *data_ptr = d_iA + d_bA[qion];
            int data_size = d_bA[qion+1] - d_bA[qion];

            // if no ions in the bin
            if (data_size < 1)
                continue;

            int *lbound = minions + tid;
            int *ubound = maxions + tid;

            int target = minlimit * speclen;

            lower_bound(data_ptr, data_size, lbound, target);

            target = (((maxlimit + 1) * speclen) - 1);
            upper_bound(data_ptr, data_size, ubound, target);

        }
        else
        {
            minions[tid] = -1;
            maxions[tid] = -1;
        }
    }

    __syncthreads();
}

__device__ void getMaxdhCell(dhCell *topscores, dhCell *out)
{
    int tid = threadIdx.x;
    int warpsize = 32;
    int warpId = tid / warpsize;
    int laneId = tid % warpsize;
    int nwarps = blockDim.x / warpsize;
    int nthreads = blockDim.x;

    // get the max element
    int myIdx = threadIdx.x;
    float myhScore = topscores[myIdx].hyperscore;

    unsigned mask  = __ballot_sync(0xffffffff, tid < nthreads);

    for(int offset = warpsize / 2; offset > 0; offset /= 2)
    {
        int tempScore = __shfl_down_sync(mask, myhScore, offset);
        int tempIdx = __shfl_down_sync(mask, myIdx, offset);

        if (tempScore > myhScore)
        {
            myhScore = tempScore;
            myIdx = tempIdx;
        }
    }

    __syncthreads();

    __shared__ int lochScore[32];
    __shared__ int locIdx[32];

    if (laneId == 0)
    {
        lochScore[warpId] = myhScore;
        locIdx[warpId] = myIdx;
    }

    __syncthreads();

    if (tid < nwarps)
    {
        myhScore = lochScore[tid];
        myIdx = locIdx[tid];
    }
    else
    {
        myhScore = 0;
        myIdx = -1;
    }

    __syncthreads();

    if (warpId == 0)
    {
        unsigned int mask  = __ballot_sync(0xffffffff, tid < nwarps);

        for(int offset = warpsize / 2; offset > 0; offset /= 2)
        {
            int tempScore = __shfl_down_sync(mask, myhScore, offset);
            int tempIdx = __shfl_down_sync(mask, myIdx, offset);

            if (tempScore > myhScore)
            {
                myhScore = tempScore;
                myIdx = tempIdx;
            }
        }
    }

    __syncthreads();

    // the final value should be at location zero
    if (tid == 0)
    {
        out->hyperscore = topscores[myIdx].hyperscore;
        out->psid       = topscores[myIdx].psid;
        out->idxoffset  = topscores[myIdx].idxoffset;
        out->sharedions = topscores[myIdx].sharedions;
    }

    return;
}

} // namespace s3

} // namespace cuda

} // namespace gpu

} // namespace hcp