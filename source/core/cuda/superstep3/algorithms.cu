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

// -------------------------------------------------------------------------------------------- //

__device__ void lower_bound(uint_t *data, int size, int *lbound, int t)
{
    int n = size;
 
    // if there are no elements in nums
    if(n==0 || data[n-1] < t)
        *lbound = n;
    else if (data[0] >= t)
        *lbound = 0;
    else
    {
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
    }

    return;
}

// -------------------------------------------------------------------------------------------- //

__device__ void upper_bound(uint_t *data, int size, int *ubound, int t)
{
    int n = size;

    // if there are no elements in nums
    if(n==0 || data[0] > t)
        *ubound = -1;
    else if (data[n-1] <= t)
        *ubound = n-1;
    else
    {
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
    }

    return;
}

// -------------------------------------------------------------------------------------------- //

__device__ void compute_minmaxions(int *minions, int *maxions, int *QAPtr, uint *d_bA, uint *d_iA, int dF, int qspeclen, int speclen, int minlimit, int maxlimit, int maxmass, int scale)
{
    int tid = threadIdx.x;
    short bucket = (2*dF+1);

    // total ions with mass +-dF
    int irange = qspeclen * bucket;

    for (int a = tid; a < irange; a+=blockDim.x)
    {
        minions[a] = 0;
        maxions[a] = -1;
    }

    __syncthreads();

    // for all ions
    for (int ion = tid; ion < irange ; ion += blockDim.x)
    {
        // main ion
        auto myion = QAPtr[ion/bucket];
        // dF offset
        int myion_offset = (ion % bucket) - dF;

        // new ion mass
        auto qion = myion + myion_offset;

        int maxionmass = (maxmass * scale) - 1 - dF;

        //printf("tid: %d, qion: %d\n", tid, qion);

        // check for legal ion mass
        if (myion > dF && myion <= maxionmass)
        {
            // locate iAPtr start and end
            uint_t *data_ptr = d_iA + d_bA[qion];
            int data_size = d_bA[qion+1] - d_bA[qion];

            // if no ions in the bin
            if (data_size < 1)
                continue;

            // lowerbound limit
            int target = minlimit * speclen;

            // compute lower bound
            lower_bound(data_ptr, data_size, &minions[ion], target);

            __threadfence_block();

            // upperbound limit
            target = (((maxlimit + 1) * speclen) - 1);

            upper_bound(data_ptr, data_size, &maxions[ion], target);

            __threadfence_block();
        }
    }

    __syncthreads();

    return;

}

// -------------------------------------------------------------------------------------------- //

__device__ void getMaxdhCell(dhCell &topscores, dhCell &out)
{
    int tid = threadIdx.x;
    int warpsize = 32;
    int warpId = tid / warpsize;
    int laneId = tid % warpsize;
    int nwarps = blockDim.x / warpsize;
    int nthreads = blockDim.x;

    // get the max element
    int myIdx = tid;
    float myhScore = topscores.hyperscore;

    unsigned mask  = __ballot_sync(0xffffffff, tid < nthreads);

    for(int offset = warpsize / 2; offset > 0; offset /= 2)
    {
        float tempScore = __shfl_down_sync(mask, myhScore, offset);
        int tempIdx = __shfl_down_sync(mask, myIdx, offset);

        if (tempScore > myhScore)
        {
            myhScore = tempScore;
            myIdx = tempIdx;
        }
    }

    __syncthreads();

    __shared__ float lochScore[32];
    __shared__ int locIdx[32];
    __shared__ dhCell thetopscore;

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
            float tempScore = __shfl_down_sync(mask, myhScore, offset);
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
        locIdx[0] = myIdx;

    __syncthreads();

    // write the topscore at the shared memory
    if (tid == locIdx[0])
    {
        thetopscore.hyperscore = topscores.hyperscore;
        thetopscore.psid = topscores.psid;
        thetopscore.idxoffset = topscores.idxoffset;
        thetopscore.sharedions = topscores.sharedions;
    }

    __syncthreads();

    // pick the topscore from the shared memory
    out.hyperscore = thetopscore.hyperscore;
    out.psid = thetopscore.psid;
    out.idxoffset = thetopscore.idxoffset;
    out.sharedions = thetopscore.sharedions;

    return;
}

// -------------------------------------------------------------------------------------------- //

template <typename T>
__device__ void blockSum(T val, T &sum)
{
    short tid = threadIdx.x;
    short warpsize = 32;
    short warpId = tid / warpsize;
    short laneId = tid % warpsize;
    short bsize = blockDim.x;
    short nwarps = bsize / warpsize;
    nwarps += (bsize % warpsize) ? 1 : 0;

    // sum a warp
    unsigned mask  = __ballot_sync(0xffffffff, tid < bsize);

    for (int offset = warpsize / 2; offset > 0; offset /= 2)
    {
        T tempVal = __shfl_down_sync(mask, val, offset);
        val += tempVal;
    }

    __syncthreads();

    // sum a block
    __shared__ T bSum[32];

    if (laneId == 0)
        bSum[warpId] = val;

    __syncthreads();

    if (tid < nwarps)
        val = bSum[tid];
    else
        val = 0;

    if (warpId == 0)
    {
        mask  = __ballot_sync(0xffffffff, tid < nwarps);

        for (int offset = warpsize / 2; offset > 0; offset /= 2)
        {
            T tempVal = __shfl_down_sync(mask, val, offset);
            val += tempVal;
        }
    }

    if (tid == 0)
        bSum[0] = val;

    __syncthreads();

    // fetch the final sum from the shared memory
    sum = bSum[0];

    return;
}

// -------------------------------------------------------------------------------------------- //

// instantiate the templates
template __device__ void blockSum<int>(int val, int &sum);

// -------------------------------------------------------------------------------------------- //

} // namespace s3

} // namespace cuda

} // namespace gpu

} // namespace hcp