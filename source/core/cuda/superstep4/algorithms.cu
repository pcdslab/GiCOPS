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
#include "cuda/superstep4/kernel.hpp"
#include "cuda/superstep1/kernel.hpp"

namespace hcp
{

namespace gpu
{

namespace cuda
{

// -------------------------------------------------------------------------------------------- //

namespace s1
{

extern __device__ int log2ceil(unsigned long long x);

}

// -------------------------------------------------------------------------------------------- //

namespace s4
{

// -------------------------------------------------------------------------------------------- //

template <typename T>
__device__ void largmax(T *data, short i1, short i2, T val, short &out)
{
    short tid = threadIdx.x;
    short warpsize = 32;
    short warpId = tid / warpsize;
    short laneId = tid % warpsize;
    short size = (i2-i1+1);
    short nwarps = size / warpsize;
    nwarps += (size % warpsize) ? 1 : 0;

    // get the max element
    short myIdx = i1 + threadIdx.x;
    T myVal = 0;

    if (myIdx <= i2)
        myVal = data[myIdx];

    unsigned mask  = __ballot_sync(0xffffffff, myIdx <= i2);

    for(int offset = warpsize / 2; offset > 0; offset /= 2)
    {
        T tempVal = __shfl_down_sync(mask, myVal, offset);
        short tempIdx = __shfl_down_sync(mask, myIdx, offset);

        if (tempVal <= val)
        {
            if (myVal > val || tempIdx < myIdx)
            {
                myVal = tempVal;
                myIdx = tempIdx;
            }
        }
    }


    __shared__ T locVal[32];
    __shared__ short locIdx[32];

    if (laneId == 0)
    {
        locVal[warpId] = myVal;
        locIdx[warpId] = myIdx;
    }

    __syncthreads();

    if (tid < nwarps)
    {
        myVal = locVal[tid];
        myIdx = locIdx[tid];
    }

   __syncthreads();

    // check if the val at the zeroth idx is good to go
    if (warpId == 0)
    {
        unsigned int mask  = __ballot_sync(0xffffffff, tid < nwarps);

        for(int offset = warpsize / 2; offset > 0; offset /= 2)
        {
            T tempVal = __shfl_down_sync(mask, myVal, offset);
            short tempIdx = __shfl_down_sync(mask, myIdx, offset);

            if (tempVal <= val)
            {
                if (myVal > val || tempIdx < myIdx)
                {
                    myVal = tempVal;
                    myIdx = tempIdx;
                }
            }
        }
    }

    __syncthreads();

    // the final value should be at location zero
    if (tid == 0)
    {
        if (myVal <= val)
            locIdx[0] = myIdx;
        else
            locIdx[0] = i1;
    }

    __syncthreads();

    out = locIdx[0];
}

// -------------------------------------------------------------------------------------------- //

template <typename T>
__device__ void argmax(T *data, short i1, short i2, T val, short &out)
{
    short tid = threadIdx.x;
    short warpsize = 32;
    short warpId = tid / warpsize;
    short laneId = tid % warpsize;
    short size = (i2-i1+1);
    short nwarps = size / warpsize;
    nwarps += (size % warpsize) ? 1 : 0;

    // get the max element
    short myIdx = i1 + threadIdx.x;
    T myVal = 0;

    if (myIdx <= i2)
        myVal = data[myIdx];

    unsigned mask  = __ballot_sync(0xffffffff, myIdx <= i2);

    for(int offset = warpsize / 2; offset > 0; offset /= 2)
    {
        T tempVal = __shfl_down_sync(mask, myVal, offset);
        short tempIdx = __shfl_down_sync(mask, myIdx, offset);

        if (tempVal >= val)
        {
            if (myVal < val || tempIdx < myIdx)
            {
                myVal = tempVal;
                myIdx = tempIdx;
            }
        }
    }

    __shared__ T locVal[32];
    __shared__ short locIdx[32];

    if (laneId == 0)
    {
        locVal[warpId] = myVal;
        locIdx[warpId] = myIdx;
    }

    __syncthreads();

    if (tid < nwarps)
    {
        myVal = locVal[tid];
        myIdx = locIdx[tid];
    }

    // check if the val at the zeroth idx is good to go
    if (warpId == 0)
    {
        unsigned int mask  = __ballot_sync(0xffffffff, tid < nwarps);

        for(int offset = warpsize / 2; offset > 0; offset /= 2)
        {
            T tempVal = __shfl_down_sync(mask, myVal, offset);
            short tempIdx = __shfl_down_sync(mask, myIdx, offset);

            if (tempVal >= val)
            {
                if (myVal < val || tempIdx < myIdx)
                {
                    myVal = tempVal;
                    myIdx = tempIdx;
                }
            }
        }
    }

    __syncthreads();

    // the final value should be at location zero
    if (tid == 0)
    {
        if (myVal >= val)
            locIdx[0] = myIdx;
        else
            locIdx[0] = i1;
    }

    __syncthreads();

    out = locIdx[0];

}

// -------------------------------------------------------------------------------------------- //

template <typename T>
__device__ void rargmax(T *data, short i1, short i2, T val, short &out)
{
    short tid = threadIdx.x;
    short warpsize = 32;
    short warpId = tid / warpsize;
    short laneId = tid % warpsize;
    short size = (i2-i1+1);
    short nwarps = size / warpsize;
    nwarps += (size % warpsize) ? 1 : 0;

    // get the max element
    short myIdx = i1 + threadIdx.x;
    T myVal = 0;

    if (myIdx <= i2)
        myVal = data[myIdx];

    unsigned mask  = __ballot_sync(0xffffffff, myIdx <= i2);

    for(int offset = warpsize / 2; offset > 0; offset /= 2)
    {
        T tempVal = __shfl_down_sync(mask, myVal, offset);
        short tempIdx = __shfl_down_sync(mask, myIdx, offset);

        if (tempIdx <= i2 && tempVal >= val && tempIdx > myIdx)
        {
            myVal = tempVal;
            myIdx = tempIdx;
        }
    }

    __shared__ T locVal[32];
    __shared__ short locIdx[32];

    if (laneId == 0)
    {
        locVal[warpId] = myVal;
        locIdx[warpId] = myIdx;
    }

    __syncthreads();

    if (tid < nwarps)
    {
        myVal = locVal[tid];
        myIdx = locIdx[tid];
    }

    // check if the val at the zeroth idx is good to go
    if (warpId == 0)
    {
        unsigned int mask  = __ballot_sync(0xffffffff, tid < nwarps);

        for(int offset = warpsize / 2; offset > 0; offset /= 2)
        {
            T tempVal = __shfl_down_sync(mask, myVal, offset);
            short tempIdx = __shfl_down_sync(mask, myIdx, offset);

            if (tempIdx <= i2 && tempVal >= val && tempIdx > myIdx)
            {
                myVal = tempVal;
                myIdx = tempIdx;
            }
        }
    }

    __syncthreads();

    // the final value should be at location zero
    if (tid == 0)
    {
        if (myVal >= val)
            locIdx[0] = myIdx;
        else
            locIdx[0] = i2;
    }

    __syncthreads();

    out = locIdx[0];

}

// -------------------------------------------------------------------------------------------- //

/* Slice off yyt between stt1 and end1 */
template <typename T>
__device__ void Assign(T *out, T *beg, T *end)
{
    short tid = threadIdx.x;
    int size = end - beg;

    for (int ij = tid; ij < size; ij+=blockDim.x)
        out[ij] = beg[ij];

    __syncthreads();
}

// -------------------------------------------------------------------------------------------- //

template <typename T>
__device__ void prefixSum(T *beg, T *end, T *out)
{
    short tid = threadIdx.x;
    int size = end - beg;

    // compute number of iterations
    int iterations = hcp::gpu::cuda::s1::log2ceil(size);

    // compute prefix sum
    for (int ij = 0; ij < iterations; ij ++)
    {
        int offset = 1 << ij;
        T tempVal = 0;
    
        if (tid >= offset && tid < size)
            tempVal = beg[tid - offset];

        __syncthreads();

        if (tid >= offset && tid < size)
            beg[tid] += tempVal;

        __syncthreads();
    }

    for (int im = tid; im < size; im+=blockDim.x)
        out[im] = beg[im];

    __syncthreads();

    return;
}

template <typename T>
__device__ void ArraySum(T *arr, int size, T *sum)
{
    short tid = threadIdx.x;

    // compute number of iterations
    int iterations = hcp::gpu::cuda::s1::log2ceil(size);

    // compute prefix sum
    for (int ij = 0; ij < iterations; ij++)
    {
        int offset = 1 << ij;
        T tempVal = 0;

        if (tid >= offset && tid < size)
            tempVal = arr[tid - offset];

        __syncthreads();

        if (tid >= offset && tid < size)
            arr[tid] += tempVal;

        __syncthreads();
    }

    __syncthreads();

    // sum will be at the last element position
    *sum = arr[size-1];

    return;
}

// -------------------------------------------------------------------------------------------- //

template <typename T>
__device__ void XYbar(T *x, T *y, int n, double &xbar, double &ybar)
{
    short tid = threadIdx.x;
    short warpsize = 32;
    short warpId = tid / warpsize;
    short laneId = tid % warpsize;
    short nwarps = n / warpsize;
    nwarps += (n % warpsize) ? 1 : 0;

    T myX = 0;
    T myY = 0;

    if (tid < n)
    {
        myX = x[tid];
        myY = y[tid];
    }

    // compute number of iterations
    short iterations = hcp::gpu::cuda::s1::log2ceil(warpsize);

    unsigned mask  = __ballot_sync(0xffffffff, tid < n);

    // compute prefix sum
    for (int ij = 0; ij < iterations; ij ++)
    {
        int offset = 1 << ij;

        T tempX = __shfl_down_sync(mask, myX, offset);
        T tempY = __shfl_down_sync(mask, myY, offset);

        myX += tempX;
        myY += tempY;
    }


    // shared memory to spill partial sums
    __shared__ T pX[32];
    __shared__ T pY[32];

    if (laneId == 0)
    {
        pX[warpId] = myX;
        pY[warpId] = myY;
    }

    __syncthreads();

    // write to spill memory
    if (tid < nwarps)
    {
        myX = pX[tid];
        myY = pY[tid];
    }

    // compute number of iterations
    iterations = hcp::gpu::cuda::s1::log2ceil(nwarps);

    mask  = __ballot_sync(0xffffffff, tid < nwarps);

    // compute prefix sum
    for (int ij = 0; ij < iterations; ij ++)
    {
        int offset = 1 << ij;

        T tempX = __shfl_down_sync(mask, myX, offset);
        T tempY = __shfl_down_sync(mask, myY, offset);

        myX += tempX;
        myY += tempY;
    }

    __syncthreads();

    // tid == 0 has the complete sum now
    if (tid == 0)
    {
        pX[0] = myX;
        pY[0] = myY;
    }

    __syncthreads();

    xbar = pX[0];
    ybar = pY[0];
}

// -------------------------------------------------------------------------------------------- //

template <typename T>
__device__ void TopBot(T *x, T *y, int n, const double xbar, const double ybar, double &top, double &bot)
{
    short tid = threadIdx.x;
    short warpsize = 32;
    short warpId = tid / warpsize;
    short laneId = tid % warpsize;
    short nwarps = n / warpsize;
    nwarps += (n % warpsize) ? 1 : 0;

    T myX = 0;
    T myY = 0; 

    if (tid < n)
    {
        myX = (x[tid] - xbar) * (y[tid] - ybar);
        myY = (x[tid] - xbar) * (x[tid] - xbar);
    }

    // compute number of iterations
    short iterations = hcp::gpu::cuda::s1::log2ceil(warpsize);

    unsigned mask  = __ballot_sync(0xffffffff, tid < n);

    // compute prefix sum
    for (int ij = 0; ij < iterations; ij ++)
    {
        int offset = 1 << ij;

        T tempX = __shfl_down_sync(mask, myX, offset);
        T tempY = __shfl_down_sync(mask, myY, offset);

        myX += tempX;
        myY += tempY;
    }

    // shared memory to spill partial sums
    __shared__ T pX[32];
    __shared__ T pY[32];

    if (laneId == 0)
    {
        pX[warpId] = myX;
        pY[warpId] = myY;
    }

    __syncthreads();

    // write to spill memory
    if (tid < nwarps)
    {
        myX = pX[tid];
        myY = pY[tid];
    }

    // compute number of iterations
    iterations = hcp::gpu::cuda::s1::log2ceil(nwarps);

    mask  = __ballot_sync(0xffffffff, tid < nwarps);

    // compute prefix sum
    for (int ij = 0; ij < iterations; ij ++)
    {
        int offset = 1 << ij;

        T tempX = __shfl_down_sync(mask, myX, offset);
        T tempY = __shfl_down_sync(mask, myY, offset);

        myX += tempX;
        myY += tempY;
    }

    __syncthreads();

    // tid == 0 has the complete sum now
    if (tid == 0)
    {
        pX[0] = myX;
        pY[0] = myY;
    }

    __syncthreads();

    top = pX[0];
    bot = pY[0];

    return;
}

// -------------------------------------------------------------------------------------------- //

template __device__ void argmax<double_t>(double_t *data, short i1, short i2, double_t val, short &out);
template __device__ void largmax<double_t>(double_t *data, short i1, short i2, double_t val, short &out);
template __device__ void rargmax<double_t>(double_t *data, short i1, short i2, double_t val, short &out);
template __device__ void Assign<double_t>(double_t *p_x, double_t *beg, double_t *end);
template __device__ void prefixSum<double_t>(double_t *beg, double_t *end, double_t *out);
template __device__ void XYbar<double_t>(double_t *x, double_t *y, int n, double_t &xbar, double_t &ybar);
template __device__ void TopBot<double_t>(double_t *x, double_t *y, int n, double_t xbar, double_t ybar, double_t &top, double_t &bot);
template __device__ void ArraySum<int>(int *arr,int size, int *sum);


// -------------------------------------------------------------------------------------------- //

} // namespace s4
} // namespace cuda
} // namespace gpu
} // namespace hcp