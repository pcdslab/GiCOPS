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

template <typename T>
__global__ void vector_plus_constant(T *vect, T val, int size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        vect[i] += val;
}

template <typename T>
__global__ void lbound(T *start, T *end, T target, int *val)
{
    T *pos = lower_bound(start, end, target);

    *val = pos - start;
}

template <typename T>
__global__ void ubound(T *start, T *end, T target, int *val)
{
    T *pos = upper_bound(start, end, target);

    *val = pos - start;
}


// -------------------------------------------------------------------------------------------- //
