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
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "cuda/driver.hpp"
#include "cuda/superstep1/kernel.hpp"

// Macro to obtain amino acid masses
#define PROTONS(z)                 ((PROTON) * (z))
#define AAMASS(x)                  ((aaMass[AAidx(x)]) + (statMass[AAidx(x)]))
#define MODMASS(x)                 (modMass[AAidx(x)])

extern gParams params;

// include CUDA constant variables
#include "cuda/constants.cuh"

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
template <typename T>
__host__ void SortpepEntries(Index *, T *);

// compute bA from the sorted fragment-ion data
__host__ void ConstructbA(Index *index, size_t iAsize, uint chunk_number);

// -------------------------------------------------------------------------------------------- //

__host__ void initParams(gParams *hparams)
{
    //hcp::gpu::cuda::error_check(cudaMemcpyToSymbol(dParams, hparams, sizeof(gParams)));

    gParams dParams;

    dParams.threads = hparams->threads;
    
    dParams.gputhreads = hparams->gputhreads;
    dParams.min_len = hparams->min_len;
    dParams.max_len = hparams->max_len;
    dParams.maxz = hparams->maxz;
    dParams.res = hparams->res;
    dParams.scale = hparams->scale;
    dParams.dM = hparams->dM;
    dParams.dF = hparams->dF;
    dParams.min_mass = hparams->min_mass;
    dParams.max_mass = hparams->max_mass;
    dParams.topmatches = hparams->topmatches;
    dParams.expect_max = hparams->expect_max;
    dParams.min_shp = hparams->min_shp;
    dParams.min_cpsm = hparams->min_cpsm;
    dParams.base_int = hparams->base_int;
    dParams.min_int = hparams->min_int;
    dParams.myid = hparams->myid;
    dParams.nodes = hparams->nodes;
}

// initialize mod information
__host__ void initMods(SLM_vMods *vMods)
{
    // amino acid masses
    constexpr float_t AAMass[ALPHABETS] = {71.03712, NAA, 103.00919, 115.030, 129.0426, 147.068, 57.02146, 137.060, 113.084, NAA, 128.094, 113.084, 131.0405, 114.043, NAA, 97.0527, 128.05858, 156.1012, 87.032, 101.0476, NAA, 99.06841, 186.0793, NAA, 163.0633, NAA};

    // static mod masses
    constexpr float_t statMods[ALPHABETS] = {0, 0, 57.021464, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // variable mod masses
    float_t vMass[ALPHABETS] = {0};

    int nMods = vMods->num_vars;

    for (int i = 0; i < nMods; i++)
    {
        std::string residues = std::string(&vMods->vmods[i].residues[0]);

        for (auto aa : residues)
            vMass[AAidx(aa)] = static_cast<double>(vMods->vmods[i].modMass)/params.scale;
    }

    // copy to CUDA constant arrays
    hcp::gpu::cuda::error_check(cudaMemcpyToSymbol(modMass, vMass, sizeof(float_t) * ALPHABETS));
    hcp::gpu::cuda::error_check(cudaMemcpyToSymbol(aaMass, AAMass, sizeof(float_t) * ALPHABETS));
    hcp::gpu::cuda::error_check(cudaMemcpyToSymbol(statMass, statMods, sizeof(float_t) * ALPHABETS));
}

// -------------------------------------------------------------------------------------------- //

template <typename T>
__host__ void exclusiveScan(T *array, int size, T init)
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    // initialize a device vector
    T *d_array = nullptr;
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_array, size, driver->stream[0]));

    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_array, array, size, driver->stream[0]));

    //thrust::device_vector<T> d_array(array, array + size);

    // compute exclusive scan
    thrust::exclusive_scan(thrust::device.on(driver->get_stream()), d_array, d_array + size, d_array, init);

    // copy back to host
    hcp::gpu::cuda::error_check(D2H(array, d_array, size, driver->stream[0]));
    // thrust::copy(d_array.begin(), d_array.end(), array);

    // free device memory
    hcp::gpu::cuda::device_free_async(d_array, driver->stream[0]);
}

// instantiate the template for int and uint_t
template __host__ void exclusiveScan<int>(int *array, int size, int init);
template __host__ void exclusiveScan<uint_t>(uint_t *array, int size, uint_t init);

// -------------------------------------------------------------------------------------------- //
__host__ void SortPeptideIndex(Index *index)
{
    // extract all peptide masses in an array to simplify computations
    float_t *h_mzs = new float_t[index->lcltotCnt];
    // hcp::gpu::cuda::host_pinned_allocate<float_t>(h_mzs, index->lcltotCnt);

#if defined (USE_OMP)
#pragma omp parallel for num_threads(threads) schedule (static)
#endif // USE_OMP
        // simplify the peptide masses
        for (int i = 0; i < index->lcltotCnt; i++)
            h_mzs[i] = index->pepEntries[i].Mass;

        //sort by masses on GPU
        SortpepEntries(index, h_mzs);

        // free the masses array
        delete[] h_mzs;
        h_mzs = nullptr;

        // no need for pinned memory as only one use
        //hcp::gpu::cuda::host_pinned_free(h_mzs);

}

// -------------------------------------------------------------------------------------------- //

// sort peptide entries by mass on GPU
template <typename T>
void SortpepEntries(Index *index, T *h_mz)
{
    auto pepIndex = index->pepEntries;
    auto size = index->lcltotCnt;
    auto driver = hcp::gpu::cuda::driver::get_instance();

    // initialize device vector with mzs
    T *d_pepMZs = nullptr;
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_pepMZs, size, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_pepMZs, h_mz, size, driver->stream[0]));
    
    //thrust::device_vector<T> d_pepMZs(h_mz, h_mz + size);

    // initilize values to sequence
    int *d_indices = nullptr;
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_indices, size, driver->stream[0]));
    //thrust::device_vector<int> d_indices(size);
    thrust::sequence(thrust::device.on(driver->get_stream()), d_indices, d_indices+size);

    // sort the values using MZs as keys
    thrust::sort_by_key(thrust::device.on(driver->get_stream()), d_pepMZs, d_pepMZs+size, d_indices);

    // vector to store sorted indices
    int *h_sorted_indices = new int[size];
    // std::vector<int> h_sorted_indices(size);

    // copy sorted MZs to host vector
    hcp::gpu::cuda::error_check(D2H(h_sorted_indices, d_indices, size, driver->stream[0]));
    //thrust::copy(d_indices.begin(), d_indices.end(), h_sorted_indices.begin());

    // deallocate device memory as well
    hcp::gpu::cuda::device_free_async(d_indices, driver->stream[0]);
    hcp::gpu::cuda::device_free_async(d_pepMZs, driver->stream[0]);

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

    // remove h_sorted_indices
    delete[] h_sorted_indices;
    return;
}

// -------------------------------------------------------------------------------------------- //

// stable sort fragment-ion data on GPU
__host__ void StableKeyValueSort(uint_t *d_keys, uint_t* h_data, int size, bool isSearch)
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    uint_t *d_data = getATcols(size);

    //thrust::device_vector<uint_t> d_data(size);
    // enumerate indices
    thrust::sequence(thrust::device.on(driver->get_stream()), d_data, d_data + size);

    // sort the data using keys
    thrust::stable_sort_by_key(thrust::device.on(driver->get_stream()), d_keys, d_keys + size, d_data);

    if (!isSearch)
        // copy sorted data to host array
        hcp::gpu::cuda::error_check(D2H(h_data, d_data, size, driver->stream[0]));
        //thrust::copy(d_data.begin(), d_data.end(), h_data);

    if (!isSearch)
        // free the device columns in indexing phase only
        freeATcols();

    return;
}

// -------------------------------------------------------------------------------------------- //

__host__ uint_t*& getFragIon()
{
    static thread_local uint_t *d_fragIon = nullptr;
    auto driver = hcp::gpu::cuda::driver::get_instance();

    // allocate device vector only once
    if (d_fragIon == nullptr)
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_fragIon, MAX_IONS, driver->stream[0]));

    return d_fragIon;
}

// -------------------------------------------------------------------------------------------- //

__host__ uint_t*& getATcols(int size)
{
    static thread_local uint_t *d_ATcols = nullptr;

    auto driver = hcp::gpu::cuda::driver::get_instance();

    // allocate device vector only once
    if (d_ATcols == nullptr)
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_ATcols, size, driver->stream[0]));

    return d_ATcols;
}

// -------------------------------------------------------------------------------------------- //

__host__ dIndex*& getdIndex(Index *index)
{
    static thread_local dIndex *d_index = nullptr;
    auto driver = hcp::gpu::cuda::driver::get_instance();

    int idxchunks = params.max_len - params.min_len + 1;

    // allocate device vector only once
    if (d_index == nullptr && index != nullptr)
    {
        const uint_t bAsize = ((uint_t)(params.max_mass * params.scale)) + 1;

        d_index = new dIndex[idxchunks];

        for (int ij = 0; ij < idxchunks; ij++)
        {
            Index *curr_index = &index[ij];
            dIndex *curr_dindex = &d_index[ij];

            curr_dindex->nChunks = curr_index->nChunks;

            // allocate nChunks of SLMchunk
            curr_dindex->ionIndex = new spmat_t[curr_index->nChunks];

            uint_t speclen = (curr_index->pepIndex.peplen - 1) * params.maxz * iSERIES;

            // construct for each intra-chunk (within each length)
            for (int chno = 0; chno < curr_index->nChunks; chno++)
            {
                /* Check if this chunk is the last chunk */
                uint_t nsize = ((chno == curr_index->nChunks - 1) && (curr_index->nChunks > 1))?
                    curr_index->lastchunksize : curr_index->chunksize;

                uint_t iAsize = nsize * speclen;
                hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(curr_dindex->ionIndex[chno].bA, bAsize, driver->stream[1]));

                hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(curr_dindex->ionIndex[chno].bA, curr_index->ionIndex[chno].bA, bAsize, driver->stream[1]));

                hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(curr_dindex->ionIndex[chno].iA, iAsize, driver->stream[1]));

                hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(curr_dindex->ionIndex[chno].iA, curr_index->ionIndex[chno].iA, iAsize, driver->stream[1]));
            }
        }
    }

    driver->stream_sync(1);

    return d_index;
}

// -------------------------------------------------------------------------------------------- //

__host__ void freedIndex()
{
    auto &&d_index = getdIndex();

    auto driver = hcp::gpu::cuda::driver::get_instance();

    int idxchunks = params.max_len - params.min_len + 1;

    if (d_index != nullptr)
    {
        for (int ij = 0; ij < idxchunks; ij++)
        {
            dIndex *curr_dindex = &d_index[ij];

            int nchunks = curr_dindex->nChunks;

            if (nchunks > 0)
            {
                for (int chno = 0; chno < nchunks; chno++)
                {
                    if (curr_dindex->ionIndex[chno].bA)
                        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(curr_dindex->ionIndex[chno].bA, driver->stream[0]));

                    if (curr_dindex->ionIndex[chno].iA)
                        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(curr_dindex->ionIndex[chno].iA, driver->stream[0]));
                }
            }

            curr_dindex->nChunks = 0;
        }

        driver->stream_sync(0);

        delete[] d_index;
        d_index = nullptr;
    }
}

// -------------------------------------------------------------------------------------------- //

__host__ uint_t*& getbA()
{
    static thread_local uint_t *d_bA = nullptr;
    // size of the bA
    uint_t bAsize = ((uint_t)(params.max_mass * params.scale)) + 1;

    auto driver = hcp::gpu::cuda::driver::get_instance();

    // allocate device vector only once
    if (d_bA == nullptr)
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_bA, bAsize, driver->stream[0]));

    return d_bA;
}

// -------------------------------------------------------------------------------------------- //

__host__ void freeATcols()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();
    auto &&d_ATcols = getATcols(1);

    // free the device vector only once
    if (d_ATcols != nullptr)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_ATcols, driver->stream[0]));
        d_ATcols = nullptr;
    }

    // must sync here to release all allocated memory 
    // before next superstep
    driver->stream_sync();

    return;
}

// -------------------------------------------------------------------------------------------- //

__host__ void freeFragIon()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();

    auto &&d_fragIon = getFragIon();

    // free the device vector only once
    if (d_fragIon != nullptr)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_fragIon, driver->stream[0]));
        d_fragIon = nullptr;
    }

    // must sync here to release all allocated memory 
    // before next superstep
    driver->stream_sync();

    return;
}

// -------------------------------------------------------------------------------------------- //

__host__ void freebA()
{
    auto driver = hcp::gpu::cuda::driver::get_instance();
    auto &&d_bA = getbA();

    // free the device vector only once
    if (d_bA != nullptr)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_bA, driver->stream[0]));
        d_bA = nullptr;
    }

    // must sync here to release all allocated memory 
    // before next superstep
    driver->stream_sync();

    return;
}

// -------------------------------------------------------------------------------------------- //

// construct the index on the GPU
__host__ status_t ConstructIndexChunk(Index *index, int_t chunk_number, bool isSearch)
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
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_pepEntries, index->lcltotCnt, driver->stream[0]));
        // copy peptide entries to device
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_pepEntries, index->pepEntries, index->lcltotCnt, driver->stream[0]));
    }

    static char *d_seqs = nullptr;

    if (d_seqs == nullptr)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_seqs, index->pepIndex.AAs, driver->stream[0]));
        // copy peptide sequences to device
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_seqs, index->pepIndex.seqs, index->pepIndex.AAs, driver->stream[0]));
    }

    // device vector for fragment-ion data
    auto d_fragIon = getFragIon();

    // speclen will be the blockSize
    int blockSize = peplen_1 * iSERIES;

    int shmemBytes = blockSize * sizeof(float_t);

    // make sure the required shared memory is available to the kernel
    cudaFuncSetAttribute(GenerateFragIonData, cudaFuncAttributeMaxDynamicSharedMemorySize, shmemBytes);

    // generate fragment ion data
    GenerateFragIonData<<<interval, blockSize, shmemBytes, driver->get_stream()>>>(d_fragIon, d_pepEntries, d_seqs, peplen, start_idx, scale, maxz, minmass, maxmass);

    driver->stream_sync();

    uint_t *iAPtr = index->ionIndex[chunk_number].iA;

    // Stable keyValue sort the fragment-ion data and copy to iAPtr
    StableKeyValueSort(d_fragIon, iAPtr, iAsize, isSearch);

    if (!isSearch)
        // construct corresponding DSLIM.bA
        ConstructbA(index, iAsize, chunk_number);

    // free memory
    if (lastChunk == true)
    {
        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_pepEntries, driver->stream[0]));
        d_pepEntries = nullptr;

        hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_seqs, driver->stream[0]));
        d_seqs = nullptr;
    }

    // synchronize stream
    driver->stream_sync();

    return status;
}

// -------------------------------------------------------------------------------------------- //

__host__ void ConstructbA(Index *index, size_t iAsize, uint chunk_number)
{
    // size of the bA
    uint_t bAsize = ((uint_t)(params.max_mass * params.scale)) + 1;

    auto driver = hcp::gpu::cuda::driver::get_instance();

    // device vector for fragment-ion data
    auto d_sortedFragIon = getFragIon();

    // device vector for bA data
    uint_t *d_bA = getbA();

    // enumerate indices
    thrust::sequence(thrust::device.on(driver->get_stream()), d_bA, d_bA + bAsize);

    // binary search the start of each ion and store in d_bA
    thrust::lower_bound(thrust::device.on(driver->get_stream()), d_sortedFragIon, d_sortedFragIon + iAsize, d_bA, d_bA + bAsize, d_bA);

    // copy bA data back to CPU
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(index->ionIndex[chunk_number].bA, d_bA, bAsize, driver->stream[0]));
    //thrust::copy(d_bA, d_bA + bAsize, index->ionIndex[chunk_number].bA);
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
        __shared__ float_t pSums[2];

        // shared memory to store spectra
        extern __shared__ float_t f_Spectrum[];

        // ensure everything is initialized
        __syncthreads();

        // set my value to the amino acid mass
        myVal += AAMASS(_seq[myAA]);

        if (_entry->sites.modNum != 0)
            if ((_entry->sites.sites >> myAA) & 0x01)
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
}

// -------------------------------------------------------------------------------------------- //

} // namespace s1

} // namespace cuda

} // namespace gpu

} // namespace hcp
