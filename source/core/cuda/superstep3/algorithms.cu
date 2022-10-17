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

// will return the index of the element where lower_bound satisfies
template <typename T>
__device__ int lower_bound_inner(T *start, T *end, T target)
{
    int sz = end - start;

    // start element is larger than the target
    if (sz == 0 || start[0] >= target)
        return 0;

    // last element is smaller than the target
    if (end[-1] < target)
        return sz;

    __shared__ int idx;

    if (threadIdx.x > 0 && threadIdx.x < sz && start[threadIdx.x] >= target && start[threadIdx.x - 1] < target)
        idx = threadIdx.x;

    __syncthreads();

    int l_idx = idx;

    return l_idx;
}

// -------------------------------------------------------------------------------------------- //

template <typename T>
__device__ T * lower_bound(T *start, const int n, T target)
{
    short tid = threadIdx.x;
    short bsize = blockDim.x;

    T *l_start = start;
    T *l_end = start + n;
    int l_size = n;

    // make splitters and keep finding the range
    while (l_size > bsize)
    {
        // start element is larger than the target
        if (l_size == 0 || l_start[0] >= target)
            return l_start;

        // last element is smaller than the target
        if (l_end[-1] < target)
            return l_end;

        int partsize = l_size / (bsize - 1);
        T myVal = l_start[partsize * tid];

        if (tid == bsize - 1)
            myVal = l_end[-1];

        T lastVal = 0;

        if (tid > 0)
            lastVal = l_start[partsize * (tid - 1)];

        __shared__ int idx;


        if (threadIdx.x > 0 && myVal >= target && lastVal < target)
            idx = threadIdx.x;

        __syncthreads();

        // update limits
        if (idx < bsize - 1)
            l_end = l_start + (partsize * idx) + 1;

        // warning: do not update l_start before updating l_end
        l_start = l_start + partsize * (idx - 1);
        l_size = l_end - l_start;

        __syncthreads();

    }

    if (l_size < bsize)
        return l_start + lower_bound_inner(l_start, l_end, target);
}

// -------------------------------------------------------------------------------------------- //

// will return the index of the element where lower_bound satisfies
template <typename T>
__device__ int upper_bound_inner(T *start, T *end, T target)
{
    int size = end - start;

    // start element is larger than the target
    if (size == 0 || start[0] > target)
        return -1;

    // last element is smaller than the target
    if (end[-1] <= target)
        return size-1;

    __shared__ int idx;

    if (threadIdx.x < size && threadIdx.x < size && start[threadIdx.x + 1] > target && start[threadIdx.x] <= target)
        idx = threadIdx.x;

    __syncthreads();

    int l_idx = idx;

    return l_idx;
}

// -------------------------------------------------------------------------------------------- //

template <typename T>
__device__ T * upper_bound(T *start, const int n, T target)
{
    short tid = threadIdx.x;
    short bsize = blockDim.x;

    T *l_start = start;
    T *l_end = start + n;
    int l_size = n;

    // make splitters and keep finding the range
    while (l_size > bsize)
    {
        // start element is larger than the target
        if (l_size == 0 || l_start[0] > target)
            return l_start-1;

        // last element is smaller than the target
        if (l_end[-1] <= target)
            return l_end-1;

        int partsize = l_size / (bsize - 1);
        T myVal = l_start[partsize * tid];

        if (tid == bsize - 1)
            myVal = l_end[-1];

        T nextVal = 0;

        if (tid < bsize - 2)
            nextVal = l_start[partsize * (tid +1)];
        else
            nextVal = l_end[-1];

        __shared__ int idx;

        if (threadIdx.x < bsize && myVal <= target && nextVal > target)
            idx = threadIdx.x;

        __syncthreads();

        // update limits
        if (idx < bsize - 2)
            l_end = l_start + partsize * (idx + 1) + 1;

        // warning: do not update l_start before updating l_end
        l_start = l_start + partsize * idx;

        l_size = l_end - l_start;

        __syncthreads();

    }

    if (l_size < bsize)
        return l_start + upper_bound_inner(l_start, l_end, target);
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
    for (int ion = 0; ion < irange; ion++)
    {
        // main ion
        auto myion = QAPtr[ion/bucket];
        // dF offset
        int myion_offset = (ion % bucket) - dF;

        // new ion mass
        auto qion = myion + myion_offset;

        int maxionmass = (maxmass * scale) - 1 - dF;

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
            uint_t target = minlimit * speclen;

            __syncthreads();

            // compute lower bound
            auto lower = lower_bound(data_ptr, data_size, target) - data_ptr;

            // upperbound limit
            target = (((maxlimit + 1) * speclen) - 1);

            __syncthreads();

            auto upper = upper_bound(data_ptr, data_size, target) - data_ptr;

            __syncthreads();

            minions[ion] = lower;
            maxions[ion] = upper;

            __syncthreads();

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
template __device__ uint_t * lower_bound<uint_t>(uint_t *start, const int n, uint_t target);
template __device__ uint_t * upper_bound<uint_t>(uint_t *start, const int n, uint_t target);

// -------------------------------------------------------------------------------------------- //

} // namespace s3

} // namespace cuda

} // namespace gpu

} // namespace hcp