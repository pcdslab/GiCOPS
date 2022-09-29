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
#include "dslim.h"

const int SEARCHINSTANCES = 128;

const int SEARCH_STREAM = 0;
const int DATA_STREAM = 1;

const int HISTOGRAM_SIZE = (1 + (MAX_HYPERSCORE * 10) + 1);

namespace hcp
{

namespace gpu
{

namespace cuda
{

namespace s3
{

/* Experimental MS/MS spectra data */
template <typename T>
struct dQueries
{
    T        *moz; /* Stores the m/z values of the spectra */
    T  *intensity; /* Stores the intensity values of the experimental spectra */
    uint_t        *idx; /* Row ptr. Starting index of each row */
    //float_t  *precurse; /* Stores the precursor mass of each spectrum. */
    int      *minlimits; // min index limits for each spectrum
    int      *maxlimits; // max index limits for each spectrum
    // int_t     *charges; // not needed yet
    // float_t    *rtimes; // not needed yet
    int_t     numPeaks;
    int_t     numSpecs; /* Number of theoretical spectra */

    dQueries();
    ~dQueries();
    void H2D(Queries<T> *rhs);

    void reset()
    {
        numPeaks = 0;
        numSpecs = 0;
    }

};

#ifdef USE_GPU
struct dScores
{
    double_t *survival;
    dhCell    *topscore;
    int      *cpsms;

    dScores();
    ~dScores();
};

dScores *&getScorecard();

#endif // USE_GPU

std::pair<BYC *, int>& getBYC(int chunksize=0);

void reset_dScores();

void freeScorecard();

void freeBYC();

status_t initialize();

status_t search(Queries<spectype_t> *, Index *, uint_t, int, hCell *CandidatePSMS = nullptr);

status_t deinitialize();

} // namespace s3

} // namespace cuda

} // namespace gpu

} // namespace hcp