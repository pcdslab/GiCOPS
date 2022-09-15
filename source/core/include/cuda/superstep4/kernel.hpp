/*
 * Copyright (C) 2023 Muhammad Haseeb, and Fahad Saeed
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
#include "dslim.h"

#include "cuda/superstep3/kernel.hpp"

using dScores_t = hcp::gpu::cuda::s3::dScores;

namespace hcp
{

namespace gpu
{

namespace cuda
{

namespace s4
{

double *& getd_eValues();

void freed_eValues();

status_t processResults(Index *, Queries<spectype_t> *, int);

void processResults(double *h_data, float *h_hyp, int *h_cpsms, double *h_evalues, int bsize);

status_t getIResults(Index *, Queries<spectype_t> *, int, hCell *CandidatePSMS = nullptr);

} // namespace s4
} // namespace cuda
} // namespace gpu
} // namespace hcp