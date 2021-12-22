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
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "cuda/driver.hpp"
#include "cuda/superstep1/kernel.hpp"

#define ALPHABETS      26
#define MAXZ           6

// Macro to obtain amino acid masses
#define PROTONS(z)                 ((PROTON) * (z))
#define AAMASS(x)                  ((aaMass[AAidx(x)]) + (statMass[AAidx(x)]))
#define MODMASS(x)                 (modMass[AAidx(x)])

extern gParams params;

// array to store PTM masses
__constant__ float_t modMass[ALPHABETS];

// amino acid masses
__constant__ float_t aaMass[ALPHABETS];

// static mod masses
__constant__ float_t statMass[ALPHABETS];

namespace hcp 
{

namespace gpu
{

namespace cuda
{

namespace s1
{

// -------------------------------------------------------------------------------------------- //

// y = ceil(log2(x))
__device__ int log2ceil(unsigned long long x);

// kernel to generate the fragment ion data
__global__ void GenerateFragIonData(uint_t *, pepEntry *, char *, short, int, short, short, float_t, float_t);

// sort pepEntries sub-kernel
__host__ void SortpepEntries(Index *, float_t *);

// compute bA from the sorted fragment-ion data
__host__ void ConstructbA(Index *index, size_t iAsize, uint chunk_number);

// -------------------------------------------------------------------------------------------- //

// initialize mod information
__host__ void initMods(SLM_vMods *vMods)
{
    // amino acid masses
    constexpr float_t AAMass[26] = {71.03712, NAA, 103.00919, 115.030, 129.0426, 147.068, 57.02146, 137.060, 113.084, NAA, 128.094, 113.084, 131.0405, 114.043, NAA, 97.0527, 128.05858, 156.1012, 87.032, 101.0476, NAA, 99.06841, 186.0793, NAA, 163.0633, NAA};

    // static mod masses
    constexpr float_t statMods[ALPHABETS] = {0, 0, 57.021464, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // variable mod masses
    float_t vMass[26] = {0};

    int nMods = vMods->num_vars;

    for (int i = 0; i < nMods; i++)
    {
        std::string residues = std::string(&vMods->vmods[i].residues[0]);

        for (auto aa : residues)
            vMass[AAidx(aa)] = vMods->vmods[i].modMass/params.scale;
    }

    // copy to CUDA constant arrays
    hcp::gpu::cuda::error_check(cudaMemcpyToSymbol(modMass, vMass, sizeof(int) * 26));
    hcp::gpu::cuda::error_check(cudaMemcpyToSymbol(aaMass, AAMass, sizeof(int) * 26));
    hcp::gpu::cuda::error_check(cudaMemcpyToSymbol(statMass, statMods, sizeof(int) * 26));
}

// -------------------------------------------------------------------------------------------- //

__host__ void exclusiveScan(uint_t *array, int size, int init)
{
    // initialize a device vector
    thrust::device_vector<uint_t> d_array(array, array + size);

    // compute exclusive scan
    thrust::exclusive_scan(thrust::device, d_array.begin(), d_array.end(), d_array.begin(), init);

    // copy back to host
    thrust::copy(d_array.begin(), d_array.end(), array);
}

// -------------------------------------------------------------------------------------------- //
__host__ void SortPeptideIndex(Index *index)
{
    // extract all peptide masses in an array to simplify computations
    float_t *h_mzs = nullptr;
    hcp::gpu::cuda::host_pinned_allocate<float_t>(h_mzs, index->lcltotCnt);

#if defined (USE_OMP)
#pragma omp parallel for num_threads(threads) schedule (static)
#endif // USE_OMP
        // simplify the peptide masses
        for (int i = 0; i < index->lcltotCnt; i++)
            h_mzs[i] = index->pepEntries[i].Mass;

        //sort by masses on GPU
        SortpepEntries(index, h_mzs);

        // free the masses array
        hcp::gpu::cuda::host_pinned_free(h_mzs);
}

// -------------------------------------------------------------------------------------------- //

// sort peptide entries by mass on GPU
void SortpepEntries(Index *index, float_t *h_mz)
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

// -------------------------------------------------------------------------------------------- //

// stable sort fragment-ion data on GPU
__host__ void StableKeyValueSort(uint_t *d_keys, uint_t* h_data, int size)
{
    // initialize device vector
    thrust::device_vector<uint_t> d_data(size);

    // enumerate indices
    thrust::sequence(d_data.begin(), d_data.end());

    // sort the data using keys
    thrust::stable_sort_by_key(d_keys, d_keys + size, d_data.begin());

    // copy sorted data to host array
    thrust::copy(d_data.begin(), d_data.end(), h_data);

    return;
}

// -------------------------------------------------------------------------------------------- //

__host__  uint_t*& getFragIon()
{
    static uint_t *d_fragIon = nullptr;

    // allocate device vector only once
    if (d_fragIon == nullptr)
        hcp::gpu::cuda::device_allocate(d_fragIon, MAX_IONS);

    return d_fragIon;
}

// -------------------------------------------------------------------------------------------- //

__host__ void freeFragIon()
{
    auto d_fragIon = getFragIon();

    // free the device vector only once
    if (d_fragIon != nullptr)
    {
        hcp::gpu::cuda::device_free(d_fragIon);
        d_fragIon = nullptr;
    }

    return;
}

// -------------------------------------------------------------------------------------------- //

// construct the index on the GPU
__host__ status_t ConstructIndexChunk(Index *index, int_t chunk_number)
{
    status_t status = SLM_SUCCESS;

    // initialize local variables
    const uint_t peplen_1 = index->pepIndex.peplen - 1;
    const uint_t peplen   = index->pepIndex.peplen;
    const uint_t speclen  = params.maxz * iSERIES * peplen_1;

    const float_t minmass = params.min_mass;
    const float_t maxmass = params.max_mass;
    const uint_t scale = params.scale;
    short maxz = params.maxz;

    // get driver object
    auto driver = hcp::gpu::cuda::driver::get_instance();


    // check if last chunk
    bool lastChunk = (chunk_number == (index->nChunks - 1))? true: false;

    int start_idx = chunk_number * index->chunksize;

    int interval = (lastChunk == true && index->nChunks > 1)? 
                    interval = index->lastchunksize : 
                    index->chunksize;

    // fragment-ion data size
    size_t iAsize = interval * speclen;

    // pepEntries for device
    static pepEntry *d_pepEntries = nullptr;

    if (d_pepEntries == nullptr)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate(d_pepEntries, index->lcltotCnt));
        // copy peptide entries to device
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_pepEntries, index->pepEntries, index->lcltotCnt, driver));
    }

    static char *d_seqs = nullptr;

    if (d_seqs == nullptr)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate(d_seqs, index->pepIndex.AAs));
        // copy peptide sequences to device
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_seqs, index->pepIndex.seqs, index->pepIndex.AAs, driver));
    }

    // device vector for fragment-ion data
    auto d_fragIon = getFragIon();

    // synchronize data transfers before calling the kernel
    driver->stream_sync();

    // speclen will be the blockSize
    int blockSize = peplen_1 * iSERIES;

    int shmemBytes = blockSize * sizeof(float_t);

    // generate fragment ion data
    GenerateFragIonData<<<interval, blockSize, shmemBytes, driver->get_stream()>>>(d_fragIon, d_pepEntries, d_seqs, peplen, start_idx, scale, maxz, minmass, maxmass);

    driver->stream_sync();

    uint_t *iAPtr = index->ionIndex[chunk_number].iA;

    // Stable keyValue sort the fragment-ion data and copy to iAPtr
    StableKeyValueSort(d_fragIon, iAPtr, iAsize);

    // construct corresponding DSLIM.bA
    ConstructbA(index, iAsize, chunk_number);

    // free memory
    if (lastChunk == true)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free(d_pepEntries));
        d_pepEntries = nullptr;

        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free(d_seqs));
        d_seqs = nullptr;
    }

    return status;
}

// -------------------------------------------------------------------------------------------- //

__host__ void ConstructbA(Index *index, size_t iAsize, uint chunk_number)
{
    // size of the bA
    uint_t bAsize = ((uint_t)(params.max_mass * params.scale)) + 1;

    // device vector for fragment-ion data
    auto d_sortedFragIon = getFragIon();

    // device vector for bA data
    thrust::device_vector<uint_t> d_bA(bAsize);

    // enumerate indices
    thrust::sequence(d_bA.begin(), d_bA.end());

    // binary search the start of each ion and store in d_bA
    thrust::lower_bound(thrust::device, d_sortedFragIon, d_sortedFragIon + iAsize, d_bA.begin(), d_bA.end(), d_bA.begin());

    // copy bA data back to CPU
    thrust::copy(d_bA.begin(), d_bA.end(), index->ionIndex[chunk_number].bA);
}

// -------------------------------------------------------------------------------------------- //

//
// compute y = log2(ceil(x))
//
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

// -------------------------------------------------------------------------------------------- //

//
// CUDA kernel to generate fragment ion data
//
__global__ void GenerateFragIonData(uint_t *d_fragIon, pepEntry *d_pepEntry, char *d_seqs, short peplen, 
                                    int start_idx, short scale, short maxz, float_t minmass, float_t maxmass)
{
    // blockId MUST be int type
    int   blockId  = blockIdx.x;

    // thread and block sizes could be short
    short threadId = threadIdx.x;
    short blockSize = blockDim.x;
    short peplen_1 = peplen - 1;
    short isYion  = threadId / peplen_1;
    short isBion  = 1 - isYion;

    // pre-compute
    short speclen = peplen_1 * iSERIES * maxz;
    short myAA = (isBion)? (threadId % peplen_1) : peplen_1 - (threadId % peplen_1);

    // local peptide Entry
    pepEntry *_entry = d_pepEntry + start_idx + blockId;

    // local peptide sequence
    char_t * _seq = &d_seqs[_entry->seqID * peplen];

    // peptide m/z
    float_t pepMass = _entry->Mass;

    // pointer to spectrum data
    uint_t *Spectrum = d_fragIon + (speclen * blockId);

    // write zeros to the entire spectrum
    for (int i = threadId; i < speclen; i += blockSize)
        Spectrum[threadId] = 0;

    // if valid peptide or variant
    if (pepMass >= minmass && pepMass <= maxmass)
    {
        // myVal will contain the mass of the ion
        float_t myVal = 0;

        // shared memory to spill partial sums
        __shared__ float_t pSums[1];

        // shared memory to store spectra
        extern __shared__ float_t f_Spectrum[];

        // ensure everything is initialized
        __syncthreads();

        // set my value to the amino acid mass
        myVal += AAMASS(_seq[myAA]);

        if (_entry->sites.modNum != 0)
            if (_entry->sites.sites >> myAA)
                myVal += MODMASS(_seq[myAA]);

        // copy to shared memory
        f_Spectrum[threadId] = myVal;

        // make sure the write to shared memory is completed
        __syncthreads();

        // compute number of iterations
        int iterations = log2ceil(blockSize);

        // compute prefix sum
        for (int i = 0; i < iterations; i ++)
        {
            int offset = 1 << i;

            if (threadId >= offset)
                f_Spectrum[threadId] += f_Spectrum[threadId - offset];

            // make sure all writes are done
            __syncthreads();
        }

        // copy from shared memory
        myVal = f_Spectrum[threadId] + (isYion * H2O);

        // spill bIONs sum to the shared memory
        if (threadId == peplen_1 -1)
            pSums[0] = myVal;

        // make sure all shm writes are done
        __syncthreads();

        // only if charge idx is > 0
        if (isYion)
            myVal -= pSums[0];

        // compute multiple charges and protons
        for (int i = 0; i < maxz; i++)
        {
            int myCharge = i + 1;
            int idx = (speclen/2 - peplen_1) * isYion +  i * peplen_1 + threadId;
            Spectrum[idx] = (myVal + PROTONS(myCharge)) * scale / myCharge;
        }
    }

    // do we need this?
    // __syncthreads();
}

// -------------------------------------------------------------------------------------------- //

} // namespace s1

} // namespace cuda

} // namespace gpu

} // namespace hcp
