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

#include <omp.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "cuda/superstep1/kernel.hpp"

extern gParams params;

namespace hcp 
{

namespace gpu
{

namespace cuda
{

namespace s1
{

// sort peptide entries by mass on GPU
__host__ void SortpepEntries(Index *index, float_t *h_mz)
{
    auto pepIndex = index->pepEntries;
    auto size = index->lcltotCnt;

    // initialize device vector with mzs
    thrust::device_vector<float_t> d_pepMZs(h_mz, h_mz + size);

    // initilize values to sequence
    thrust::device_vector<int> d_indices(size);
    thrust::sequence(d_indices.begin(), d_indices.end());

    // sort the values using MZs as keys
    thrust::sort_by_key(d_pepMZs.begin(), d_pepMZs.end(), d_indices.begin());

    // vector to store sorted indices
    std::vector<int> h_sorted_indices(size);

    // copy sorted MZs to host vector
    thrust::copy(d_indices.begin(), d_indices.end(), h_sorted_indices.begin());

    // allocate a new pepEntry array
    pepEntry *pepEntries = new pepEntry[size];

    // sort pepEntries using sorted indices
#if defined (_USE_OMP)
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
#endif // _USE_OMP
    for (int i = 0; i < size; i++)
    {
        pepEntries[i] = pepIndex[h_sorted_indices[i]];
    }

    // delete the old pepIndex
    delete[] index->pepEntries;

    // set local variable to nullptr
    pepIndex = nullptr;

    // update the index->pepIndex with the new one
    index->pepEntries = pepEntries;

    return;
}

// stable sort fragment-ion data on GPU
__host__ void StableKeyValueSort(uint_t *keys, uint_t* data, int size)
{
    // initialize device vectors
    thrust::device_vector<uint_t> d_keys(keys, keys + size);
    thrust::device_vector<uint_t> d_indices(size);

    // enumerate indices
    thrust::sequence(d_indices.begin(), d_indices.end());

    // sort the data using keys
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());

    // copy sorted data to host array
    thrust::copy(d_indices.begin(), d_indices.end(), data);

    return;
}

// construct the index on the GPU
__host__ status_t ConstructIndexChunk(Index *index, int_t chunk_number)
{
    status_t status = SLM_SUCCESS;


    // initialize local variables
    const uint_t peplen_1 = index->pepIndex.peplen - 1;
    const uint_t peplen   = index->pepIndex.peplen;
    const uint_t speclen  = params.maxz * iSERIES * peplen_1;

    const double_t minmass = params.min_mass;
    const double_t maxmass = params.max_mass;
    const uint_t scale = params.scale;

    // check if last chunk
    bool lastChunk = (chunk_number == (index->nChunks - 1))? true: false;

    int start_idx = chunk_number * index->chunksize;

    int interval = (lastChunk == true && index->nChunks > 1)? 
                    interval = index->lastchunksize : 
                    index->chunksize;

    // fragment-ion data size
    size_t iAsize = interval * speclen;

    // allocate device vector for fragment-ion data
    thrust::device_vector<uint_t> fragIon(iAsize);

    // FIXME
#if 0
#ifdef USE_OMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
#endif /* USE_OMP */
        for (uint_t k = start_idx; k < (start_idx + interval); k++)
        {
            /* Filling point */
            uint_t nfilled = (k - start_idx) * speclen;

            /* Temporary Array needed for Theoretical Spectra */
            uint_t* Spectrum = new uint_t[speclen];

            /* Extract peptide Information */
            float_t pepMass = 0.0;
            char_t *seq = NULL;
            uint_t pepID = k;

            /* Extract from pepEntries */
            pepEntry *entry = index->pepEntries + pepID;

            seq = &index->pepIndex.seqs[entry->seqID * peplen];

            /* Check if pepID belongs to peps or mods */
            if (entry->sites.modNum == 0)
            {
                /* Generate the Theoretical Spectrum */
                pepMass = UTILS_GenerateSpectrum(seq, peplen, Spectrum);
            }
            else
            {
                /* Generate the Mod. Theoretical Spectrum */
                pepMass = UTILS_GenerateModSpectrum(seq, (uint_t) peplen, Spectrum, entry->sites);
            }

            /* If a legal peptide */
            if (pepMass >= minmass && pepMass <= maxmass)
            {
                /* Fill the ions */
                for (uint_t ion = 0; ion < speclen; ion++)
                {
                    /* Check if legal ion */
                    if (Spectrum[ion] >= (maxmass * scale))
                    {
                        Spectrum[ion] = (maxmass * scale) - 1;
                        //assert (Spectrum[ion] < (maxmass * scale));
                    }

                    fragIon[nfilled + ion] = Spectrum[ion]; // Fill in the ion
                }
            }

            /* Illegal peptide, fill in the container with zeros */
            else
            {
                /* Fill zeros for illegal peptides
                 * FIXME: Should not be filled into the chunk
                 *  and be removed from peptide index as well
                 */
                std::memset(&fragIon[nfilled], 0x0, sizeof(uint_t) * speclen);
            }
        }

#endif // 0

    return status;
}


} // namespace s1

} // namespace cuda

} // namespace gpu

} // namespace hcp
