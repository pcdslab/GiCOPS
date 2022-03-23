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

#include "config.hpp"
#include "slm_dsts.h"
#include "slmerr.h"
#include "msquery.hpp"

namespace hcp
{

namespace gpu
{

namespace cuda
{

namespace s2
{

std::array<int, 2> readAndPreprocess(string_t& filename);

// function to preprocess MS2 data and convert to binary
void preprocess(MSQuery *, string_t &, int);

// gpu arraysort kernel
status_t ArraySort(spectype_t *intns, spectype_t *mzs, int *lens, int & m_idx, int count, int maxslen, spectype_t *m_intn, spectype_t *m_mzs);

} // namespace s1

} // namespace cuda

} // namespace gpu

} // namespace hcp
