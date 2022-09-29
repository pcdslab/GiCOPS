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

namespace hcp
{

namespace gpu
{

namespace cuda
{

namespace s1
{

// function to initialize mods
void initMods(SLM_vMods *vMods);

void initParams(gParams *params);

// function to compute partial sums
template <typename T>
void exclusiveScan(T *array, int size, T init = 0);

// function to sort the peptide index
void SortPeptideIndex(Index *index);

// function to comnstruct a complete chunk of the DSLIM index (both iA and bA)
status_t ConstructIndexChunk(Index *index, int_t chunk_number, bool isSearch = false);

// kernel to stable sort the fragment-ion data
void StableKeyValueSort(uint_t *keys, uint_t* data, int size);

// free the device memory allocated for the fragment ion data
void freeFragIon();

void freebA();

void freeATcols();

uint_t*& getbA();

uint_t*& getATcols(int size = 0);

uint_t*& getFragIon();

dIndex*& getdIndex(Index *Index = nullptr);

void freedIndex();

} // namespace s1

} // namespace cuda

} // namespace gpu

} // namespace hcp
