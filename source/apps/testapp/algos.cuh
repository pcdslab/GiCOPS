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

#include "common.hpp"
#include "slm_dsts.h"
#include "cuda/driver.hpp"


__device__ int log2ceil(unsigned long long x)
{
    static const unsigned long long t[6] = {
        0xFFFFFFFF00000000ull,
        0x00000000FFFF0000ull,
        0x000000000000FF00ull,
        0x00000000000000F0ull,
        0x000000000000000Cull,
        0x0000000000000002ull
    };

    int y = (((x & (x - 1)) == 0) ? 0 : 1);
    int j = 32;
    int i;

    for (i = 0; i < 6; i++) 
    {
        int k = (((x & t[i]) == 0) ? 0 : j);
        y += k;
        x >>= k;
        j >>= 1;
    }

  return y;
}

// will return the index of the element where lower_bound satisfies
template <typename T>
__device__ int lower_bound_inner(T *start, T *end, T target)
{
    int size = end - start;

    // start element is larger than the target
    if (size == 0 || start[0] >= target)
        return 0;

    // last element is smaller than the target
    if (end[-1] < target)
        return size;

    __shared__ int idx;

    if (threadIdx.x > 0 && threadIdx.x < size && start[threadIdx.x] >= target && start[threadIdx.x - 1] < target)
        idx = threadIdx.x;

    __syncthreads();

    if (!threadIdx.x)
    {
        printf("Inner: size: %d, idx:%d\n", size, idx);
    }

    int l_idx = idx;

    return l_idx;
}

template <typename T>
__device__ T * lower_bound(T *start, T *end, T target)
{
    short tid = threadIdx.x;
    short bsize = blockDim.x;

    T *l_start = start;
    T *l_end = end;
    int l_size = l_end - l_start;

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

        if (!tid)
        {
            printf("start: %d, end: %d, size: %d, partsize: %d, myVal: %d, lastVal: %d\n", (int)(l_start-start), (int)(l_end - start), l_size, partsize, myVal, lastVal);
        }
    
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

        if (!tid)
        {
            printf("start: %d, end: %d, size: %d, partsize: %d, myVal: %d, lastVal: %d, idx:%d\n", (int)(l_start-start), (int)(l_end - start), l_size, partsize, myVal, lastVal, idx);
        }

    }

    if (l_size < bsize)
        return l_start + lower_bound_inner(l_start, l_end, target);
}


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

    if (!threadIdx.x)
    {
        printf("Inner: size: %d, idx:%d\n", size, idx);
    }

    int l_idx = idx;

    return l_idx;
}

template <typename T>
__device__ T * upper_bound(T *start, T *end, T target)
{
    short tid = threadIdx.x;
    short bsize = blockDim.x;

    T *l_start = start;
    T *l_end = end;
    int l_size = l_end - l_start;

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

        if (!tid)
        {
            printf("start: %d, end: %d, size: %d, partsize: %d, myVal: %d, nextVal: %d\n", (int)(l_start-start), (int)(l_end - start), l_size, partsize, myVal, nextVal);
        }
    
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

        if (!tid)
        {
            printf("start: %d, end: %d, size: %d, partsize: %d, myVal: %d, nextVal: %d, idx:%d\n", (int)(l_start-start), (int)(l_end - start), l_size, partsize, myVal, nextVal, idx);
        }

    }

    if (l_size < bsize)
        return l_start + upper_bound_inner(l_start, l_end, target);
}
