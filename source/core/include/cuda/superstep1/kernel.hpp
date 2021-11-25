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

#pragma once

// include cuda.h here
#include <cuda.h>
#include "cuda/driver.hpp"

namespace hcp
{

namespace gpu
{

namespace cuda
{

namespace s1
{

// test kernel
template <typename T>
__global__ void vector_add(T *d_a, T *d_b, T *d_c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        d_c[i] = d_a[i] + d_b[i];
    }
}

} // namespace s1

} // namespace cuda

} // namespace gpu

} // namespace hcp
