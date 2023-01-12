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

#include <cuda.h>
#include <string>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <thread>

#include "cuda/driver.hpp"
#include "cuda/superstep2/kernel.hpp"

using namespace std;

extern gParams params;

// GPU thread block size
const int MAXBLOCK = 1024;
const int TEMPVECTOR_SIZE = KBYTES(16);
const int BATCHSIZE = 20000;


// -------------------------------------------------------------------------------------------- //

namespace hcp 
{

namespace gpu
{

namespace cuda
{

namespace s2
{

// -------------------------------------------------------------------------------------------- //

//
// preprocess
//
void preprocess(MSQuery *query, string_t &filename, int fileindex)
{
    // Read and preprocess the input MS2 data.
    if (params.filetype == gParams::FileType_t::PBIN)
    {
        // local variables
        int maxlen = 0;
        int nqueries = 0;

        // read MS2 file, preprocess and write data to disk
        auto vals = readAndPreprocess(filename);

        nqueries = vals[0];
        maxlen = vals[1];

        // compute number of chunks
        int nchunks = (nqueries / QCHUNK) + (nqueries % QCHUNK > 0);

        // new filename with .bin extension
        string_t filename_bin(std::move(filename + ".pbin"));

        // set the filename
        query->setFilename(filename_bin);

        // initialize the MSQuery::info_t
        query->Info() = info_t(maxlen, nchunks, nqueries);
    }
    else
    {
        // no GPU processing needed here. simply
        // call the query->initialize function
        query->initialize(&filename, fileindex);
    }
}

// -------------------------------------------------------------------------------------------- //


//
// GPU kernel for picking peaks
//
__global__ void pickpeaks(spectype_t *d_intns, spectype_t * d_mzs, spectype_t *d_m_intns, spectype_t *d_m_mzs, int *d_lens, int *d_m_lens)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bsize = blockDim.x;

    int newspectrumsize = d_m_lens[bid+1] - d_m_lens[bid];

    auto src_intns = d_intns + d_lens[bid+1] - newspectrumsize;
    auto src_mzs = d_mzs + d_lens[bid+1] - newspectrumsize;

    auto dst_intns = d_m_intns + d_m_lens[bid];
    auto dst_mzs = d_m_mzs + d_m_lens[bid];

    for (; tid < newspectrumsize; tid+=bsize)
    {
        dst_intns[tid] = src_intns[tid];
        dst_mzs[tid] = src_mzs[tid];
    }
}

// -------------------------------------------------------------------------------------------- //

//
// kernel to preprocess the data, reduce and compute new spectrum lengths
//
__global__ void computenewlens(spectype_t *d_intns, int *d_lens, int *d_m_lens, int base_int, int min_int)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int  warpsize = 32;
    int  warpId = tid / warpsize;
    int laneId = tid % warpsize;

    // how many elements in the current array
    int startidx = d_lens[bid];
    int endidx = d_lens[bid+1];

    int SpectrumSize = endidx - startidx;
    int maxelements = MIN(QALEN, SpectrumSize);

    spectype_t *loc_intns = d_intns + endidx - maxelements;

    double factor = ((double_t) base_int) / loc_intns[maxelements - 1];

    // filter out intensities > params.min_int (or 1% of base peak)
    auto l_min_int = min_int;

    int myVal = 0;

    // normalize intensities
    if (tid < maxelements)
    {
        loc_intns[tid] *= factor;
        myVal = (loc_intns[tid] >= l_min_int) ? 1 : 0;
    }

    __syncthreads();

    unsigned mask  = __ballot_sync(0xffffffff, tid < maxelements);

    for(int offset = warpsize / 2; offset > 0; offset /= 2)
    {
        int tempVal = __shfl_down_sync(mask, myVal, offset);
        myVal += tempVal;
    }

    __syncthreads();

    __shared__ int locTots[32];

    if (laneId == 0)
    {
        locTots[warpId] = myVal;
    }

    __syncthreads();

    int nwarps = maxelements / warpsize + (maxelements % warpsize > 0);

    if (tid < nwarps)
        myVal = locTots[tid];
    else
        myVal = 0;

    __syncthreads();

    if (warpId == 0)
    {
        unsigned int mask  = __ballot_sync(0xffffffff, tid < nwarps);

        for(int offset = warpsize / 2; offset > 0; offset /= 2)
        {
            int tempVal = __shfl_down_sync(mask, myVal, offset);
            myVal += tempVal;
        }
    }

    __syncthreads();

    // the final value should be at location zero
    if (tid == 0)
        d_m_lens[bid] = myVal;
}

// -------------------------------------------------------------------------------------------- //

//
// kernel to generate array indices
//
template <typename T>
__global__ void generateArrayNums(int N, int * d_lens, T *d_arraynums, spectype_t *d_tmp)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bsize = blockDim.x;

    // how many elements in the current array
    int startidx = d_lens[bid];
    int endidx = d_lens[bid+1];

    // write the array numbers to d_arraynums
    for (int i = startidx + tid; i < endidx; i += bsize)
    {
        d_arraynums[i] = bid;
        d_tmp[i] = bid;
    }
}

// -------------------------------------------------------------------------------------------- //

//
// The GPU-ArraySort kernel
//
status_t ArraySort(spectype_t *intns, spectype_t *mzs, int *lens, int &idx, int count, int maxslen, spectype_t *m_intn, spectype_t *m_mzs)
{
    // get driver object
    auto driver = hcp::gpu::cuda::driver::get_instance();

    // device ptrs for raw and processed data and lengths
    spectype_t * d_intns = nullptr;
    spectype_t * d_mzs = nullptr;
    int *d_arraynums = nullptr;
    int *d_indices = nullptr;

    int *d_lens = nullptr;
    int *d_m_lens = nullptr;

    // output vectors
    spectype_t *d_m_intns = nullptr;
    spectype_t *d_m_mzs = nullptr;

    // the raw buffer size
    int rawsize = idx;

    // temporary vector for data gathering
    spectype_t *d_tmp = nullptr;
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_tmp, rawsize, driver->stream[3])); //BATCHSIZE * TEMPVECTOR_SIZE));

    // -------------------------------------------------------------------------------------------- //

    //
    // allocate device memory and transfer data
    //

    // memory for intensities
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_intns, rawsize, driver->stream[0]));

    // transfer intensities to the GPU
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_intns, intns, idx, driver->stream[0]));

    // memory for m/zs
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_mzs, rawsize, driver->stream[1]));

    // transfer m/zs to the GPU
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_mzs, mzs, idx, driver->stream[1]));

    // memory for spectrum lengths
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_lens, BATCHSIZE+1, driver->stream[2]));

    // transfer lengths to the GPU
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::H2D(d_lens, lens, count, driver->stream[2]));

    // memory for arraynums
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_arraynums, rawsize, driver->stream[2])); 

    // memory for indices
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_indices, rawsize, driver->stream[3])); 

    // memory for processed intensities
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_m_intns, QALEN * BATCHSIZE, driver->stream[3]));

    // memory for processed mzs
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_m_mzs, QALEN * BATCHSIZE, driver->stream[3]));

    // memory for new spectrum lengths
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_allocate_async(d_m_lens, BATCHSIZE+1, driver->stream[3]));

    // -------------------------------------------------------------------------------------------- //

    //
    // initialize sequences and arraynums
    //

    // initialize indices to sequences - independent of device data
    thrust::sequence(thrust::device.on(driver->get_stream(3)), d_indices, d_indices + idx);

    // compute an exclusive scan of the lengths
    thrust::exclusive_scan(thrust::device.on(driver->get_stream(2)), d_lens, d_lens + count + 1, d_lens, 0);

    // maxlen cannot be larger than the largest allowed blocksize on CUDA
    auto maxlen = MIN(MAXBLOCK, maxslen);

    // generate array numbers.
    // NOTE 2 SELF: DEPENDS on exclusive_scan(d_lens) - do not move it up
    hcp::gpu::cuda::s2::generateArrayNums<<<count, maxlen, 48, driver->get_stream(2)>>>(idx, d_lens, d_arraynums, d_tmp);

    // synchronize data transfers before calling the kernels
    driver->all_streams_sync();

    // -------------------------------------------------------------------------------------------- //

    //
    // the GPU-ArraySort algorithm
    //

    // stable sort by key
    thrust::stable_sort_by_key(thrust::device.on(driver->get_stream()), d_intns, d_intns + idx, d_indices);
    thrust::gather(thrust::device.on(driver->get_stream()), d_indices, d_indices + idx, d_mzs, d_tmp);
    thrust::gather(thrust::device.on(driver->get_stream()), d_indices, d_indices + idx, d_arraynums, d_mzs);

    // reinitialize indices
    thrust::sequence(thrust::device.on(driver->get_stream()), d_indices, d_indices + idx);

    // stable sort by key
    thrust::stable_sort_by_key(thrust::device.on(driver->get_stream()), d_mzs, d_mzs + idx, d_indices);
    thrust::gather(thrust::device.on(driver->get_stream()), d_indices, d_indices + idx, d_tmp, d_mzs);
    thrust::gather(thrust::device.on(driver->get_stream()), d_indices, d_indices + idx, d_intns, d_tmp);

    // synchronize before doing the swaps
    driver->stream_sync();

    // swap d_intns and d_tmp
    auto d_swap = d_tmp;
    d_tmp = d_intns;
    d_intns = d_swap;

    // not really needed but for completeness
    int shmem = QALEN * sizeof(short);

    // -------------------------------------------------------------------------------------------- //

    //
    // normalize the data, pick top 100 and compute new spectrum lengths
    //

    // pickpeaks in the output array
    hcp::gpu::cuda::s2::computenewlens<<<count, QALEN, shmem, driver->get_stream()>>>(d_intns, d_lens, d_m_lens, params.base_int, params.min_int);

    // -------------------------------------------------------------------------------------------- //

    // transfer processed lengths back to the CPU
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(lens, d_m_lens, count, driver->stream[0]));

    // -------------------------------------------------------------------------------------------- //

    //
    // compute an exclusive scan of the new lengths and copy to the output arrays
    //

    thrust::exclusive_scan(thrust::device.on(driver->get_stream()), d_m_lens, d_m_lens + count + 1, d_m_lens, 0);

    // pick the peaks
    hcp::gpu::cuda::s2::pickpeaks<<<count, QALEN, shmem, driver->get_stream()>>>(d_intns, d_mzs, d_m_intns, d_m_mzs, d_lens, d_m_lens);

    // -------------------------------------------------------------------------------------------- //

    // get new total length (idx)

    int *h_idxptr = &idx;
    int *d_idxptr = d_m_lens + count;
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(h_idxptr, d_idxptr, 1, driver->stream[0]));

    // this is critical for the next step
    driver->stream_sync();

    // -------------------------------------------------------------------------------------------- //

    //
    // transfer processed data back to CPU
    //

    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(m_intn, d_m_intns, idx, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::D2H(m_mzs, d_m_mzs, idx, driver->stream[0]));

    // synchronize all streams before deallocating device memory
    driver->all_streams_sync();

    // -------------------------------------------------------------------------------------------- //

    // free the GPU memory

    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_mzs, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_intns, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_arraynums, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_indices, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_tmp, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_m_mzs, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_m_intns, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_lens, driver->stream[0]));
    hcp::gpu::cuda::error_check(hcp::gpu::cuda::device_free_async(d_m_lens, driver->stream[0]));


    // return success status
    return SLM_SUCCESS;
}

// -------------------------------------------------------------------------------------------- //

//
// readAndPreprocess spectra
//
std::array<int, 2> readAndPreprocess(string_t &filename)
{
    int_t largestspec = 0;
    int_t largestspec_loc = 0;
    int_t count = 0;
    int_t globalcount = 0;
    int_t specsize = 0;
    int_t m_idx = 0;

    char_t *Zsave;
    char_t *Isave;

    // host data vectors
    spectype_t * mzs = nullptr;
    spectype_t *intns = nullptr;
    int *lens = nullptr;

    // reverse BATCHSIZE x (20 * 1024) host vectors
    hcp::gpu::cuda::host_pinned_allocate<spectype_t>(mzs, TEMPVECTOR_SIZE * BATCHSIZE);
    hcp::gpu::cuda::host_pinned_allocate<spectype_t>(intns, TEMPVECTOR_SIZE * BATCHSIZE);
    hcp::gpu::cuda::host_pinned_allocate<int>(lens, BATCHSIZE);

    // output data
    spectype_t *m_mzs = nullptr;
    spectype_t *m_intns = nullptr;

    hcp::gpu::cuda::host_pinned_allocate(m_mzs, BATCHSIZE * QALEN);
    hcp::gpu::cuda::host_pinned_allocate(m_intns, BATCHSIZE * QALEN);

    float *rtimes = new float[2 * BATCHSIZE];
    float *prec_mz = rtimes + BATCHSIZE;
    int *z = new int[BATCHSIZE];

    /* Get a new ifstream object and open file */
    ifstream *qqfile = new ifstream(filename);

    /* Check if file opened */
    if (qqfile->is_open())
    {
        string_t line;
        bool isFirst = true;

        /* While we still have lines in MS2 file */
        while (!qqfile->eof())
        {
            /* Read one line */
            getline(*qqfile, line);

            if (line.empty() || line[0] == 'H' || line[0] == 'D')
            {
                continue;
            }
            /* Scan: (S) */
            else if (line[0] == 'S')
            {
                if (!isFirst)
                {
                    largestspec = max(specsize, largestspec);
                    largestspec_loc = max(specsize, largestspec_loc);
                    // write the updated specsize
                    lens[count] = specsize;
                    //m_idx += specsize;

                    count++;
                    globalcount++;

                    // if the buffer is full, then dump to file
                    if (count == BATCHSIZE)
                    {
                        // lens will update after this
                        hcp::gpu::cuda::s2::ArraySort(intns, mzs, lens, m_idx, count, largestspec_loc, m_intns, m_mzs);

                        // flush to the binary file
                        MSQuery::flushBinaryFile(&filename, m_mzs, m_intns, rtimes, prec_mz, z, lens, count);
                        
                        count = 0;
                        m_idx = 0;
                        largestspec_loc = 0;
                    }
                }
                else
                    isFirst = false;

                specsize = 0;

            }
            else if (line[0] == 'Z')
            {
                char_t *mh = strtok_r((char_t *) line.c_str(), " \t", &Zsave);
                mh = strtok_r(NULL, " \t", &Zsave);
                string_t val = "1";

                if (mh != NULL)
                    val = string_t(mh);

                z[count] = MAX(1, std::atoi(val.c_str()));

                val = "0.01";
                mh = strtok_r(NULL, " \t", &Zsave);

                if (mh != NULL)
                    val = string_t(mh);

                prec_mz[count] = std::atof(val.c_str());
            }
            else if (line[0] == 'I')
            {
                char_t *mh = strtok_r((char_t *) line.c_str(), " \t", &Isave);
                mh = strtok_r(NULL, " \t", &Isave);
                string_t val = "";

                if (mh != NULL)
                {
                    val = string_t(mh);
                }

                if (val.compare("RTime") == 0)
                {
                    val = "0.00";
                    mh = strtok_r(NULL, " \t", &Isave);

                    if (mh != NULL)
                    {
                        val = string_t(mh);
                    }

                    rtimes[count] = MAX(0.0, std::atof(val.c_str()));
                }
            }
            /* MS/MS data: [m/z] [int] */
            else
            {
                /* Split line into two DOUBLEs
                 * using space as delimiter */

                char_t *mz1 = strtok_r((char_t *) line.c_str(), " ", &Zsave);
                char_t *intn1 = strtok_r(NULL, " ", &Zsave);
                string_t mz = "0.01";
                string_t intn = "0.01";

                if (mz1 != NULL)
                {
                    mz = string_t(mz1);
                }

                if (intn1 != NULL)
                {
                    intn = string_t(intn1);
                }

                // integrize the values if spectype_t is int
                if constexpr (std::is_same<int, spectype_t>::value)
                {
                    mzs[m_idx] = std::atof(mz.c_str()) * params.scale;
                    intns[m_idx] = std::atof(intn.c_str()) * YAXISMULTIPLIER;
                }
                else
                {
                    mzs[m_idx] = std::atof(mz.c_str());
                    intns[m_idx] = std::atof(intn.c_str());
                }

                // increment the spectrum size & m_idx (cumulative spectrum size)
                specsize++;
                m_idx++;
            }
        }

        largestspec = max(specsize, largestspec);
        largestspec_loc = max(specsize, largestspec_loc);


        lens[count] = specsize;
        // m_idx += specsize;

        count++;
        globalcount++;

        // lens will update after this
        hcp::gpu::cuda::s2::ArraySort(intns, mzs, lens, m_idx, count, largestspec_loc, m_intns, m_mzs);

        // flush the last batch to the binary file
        MSQuery::flushBinaryFile(&filename, m_mzs, m_intns, rtimes, prec_mz, z, lens, count, true);

        // no need to reset count and m_idx here

        /* Close the file */
        qqfile->close();

        delete qqfile;
    }
    else
        std::cout << "Error: Unable to open file: " << filename << std::endl;

    largestspec = max(specsize, largestspec);

    hcp::gpu::cuda::host_pinned_free(intns);
    hcp::gpu::cuda::host_pinned_free(mzs);
    hcp::gpu::cuda::host_pinned_free(lens);
    hcp::gpu::cuda::host_pinned_free(m_intns);
    hcp::gpu::cuda::host_pinned_free(m_mzs);

    // delete the temp arrays
    delete[] rtimes;
    delete[] z;

    // return global count and largest spectrum length
    return std::array<int, 2>{globalcount, largestspec};
}

// -------------------------------------------------------------------------------------------- //

} // namespace s2

} // namespace cuda

} // namespace gpu

} // namespace hcp