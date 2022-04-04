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
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include "lbe.h"
#include "cuda/superstep1/kernel.hpp"
using namespace std;

vector<string_t> Seqs;
vector<float_t> MZs;
uint_t cumusize = 0;
ifstream file;

extern gParams params;

/* Static function Prototypes */
static status_t LBE_AllocateMem(Index *index);
/*
 * FUNCTION: LBE_AllocateMem
 *
 * DESCRIPTION: Allocate memory for Data Structures
 *
 * INPUT:
 * @N: Number of peptides
 * @M: Number of mods
 *
 * OUTPUT:
 * @status: Status of execution
 */
static status_t LBE_AllocateMem(Index *index)
{
    status_t status = SLM_SUCCESS;
    uint_t M = index->lcltotCnt;

    index->pepEntries = NULL;

    /* Allocate Memory for seqPep */
    index->pepIndex.seqs = new AA[index->pepIndex.AAs];

    if (index->pepIndex.seqs == NULL)
    {
        status = ERR_BAD_MEM_ALLOC;
    }

    /* Allocate the seqMod */
    if (status == SLM_SUCCESS)
    {
        index->pepEntries = new pepEntry[M];

        if (index->pepEntries == NULL)
        {
            status = ERR_BAD_MEM_ALLOC;
        }
    }


    return status;
}

BOOL LBE_ApplyPolicy(Index *index,  BOOL pepmod, uint_t key)
{
    BOOL value = false;

    DistPolicy_t policy = params.policy;

    uint_t csize = index->lclmodCnt;

    if (pepmod == false)
    {
        csize = index->lclpepCnt;
    }

    if (policy == cyclic)
    {
        value = key % (params.nodes) == params.myid;
    }
    else if (policy == chunk)
    {
        value = (key / csize) == (params.myid);
    }
    else
    {
        std::cout << "This policy is not implemented yet\n";
        value = false;
    }

    return value;
}

/*
 * FUNCTION: LBE_Initialize
 *
 * DESCRIPTION: Initialize internal peptides
 *              database from FASTA file
 *
 * INPUT:
 * @threads      : Number of parallel threads
 * @filename     : Path to FASTA file
 * @modconditions: String with mod conditions
 *
 * OUTPUT:
 * @status: Status of execution
 */
status_t LBE_Initialize(Index *index)
{
    status_t status = SLM_SUCCESS;
    uint_t iCount = 1;
    string_t seq;
    string_t modconditions = params.modconditions;

#ifdef USE_OMP
    uint_t threads = params.threads;
#endif /* USE_OMP */

    /* Check if ">" entries are > 0 */
    if (index->lclpepCnt > 0)
        status = LBE_AllocateMem(index);
    else
        status = ERR_INVLD_PARAM;

    /* If Seqs was successfully filled */
    if (Seqs.size() != 0 && status == SLM_SUCCESS)
    {
        uint_t seqlen = Seqs.at(0).length();

#ifdef USE_OMP
#pragma omp parallel for num_threads(threads) schedule (static) reduction(+: iCount)
#endif
        for (uint_t i = 0; i < Seqs.size(); i++)
        {
            /* Extract Sequences */
            string_t seq = Seqs.at(i);

            /* Copy into the seqPep.seqs array */
            memcpy((void *) &index->pepIndex.seqs[i * (seqlen)], (const void *) seq.c_str(), seqlen);

            /* Increment the counters */
            iCount += 2;
        }

        /* Get the peptide count */
        iCount /= 2;
    }
    else
    {
        status = ERR_FILE_NOT_FOUND;
    }

    // populate peptide entries
    if (status == SLM_SUCCESS)
        status = LBE_GeneratePeps(index);

    // populate mod entries
    if (index->lclmodCnt > 0)
        status = MODS_GenerateMods(index);

    // clear Seqs
    Seqs.clear();
    // clear MZs
    MZs.clear();

    //Sort the peptide index based on peptide precursor mass
    if (status == SLM_SUCCESS)
    {
#if 1 //defined (GPU) && defined(CUDA)

        // sort pepEntries on the GPU
        hcp::gpu::cuda::s1::SortPeptideIndex(index);
#else
        // directly sort the pepEntries on the CPU
        std::sort(index->pepEntries, index->pepEntries + index->lcltotCnt, [](pepEntry &e1, pepEntry &e2) { return e1 < e2; });

#endif // GPU && CUDA
    }

    // 
    // FIXME: Soft remove invalid peptides
    // Will need to call LBE_Distribute and/or LBE_CreatePartitions again
    //
    auto removeInvalidPeps = [&]()
    {
        // local variables
        int total = index->lcltotCnt;
        int nmods = index->lclmodCnt;
        uint_t maxmass = params.max_mass;

        // check for any invalid mass peptides from the end.
        for (int i = index->lclmodCnt-1; i > index->lclpepCnt; i--)
        {
            // since the pepEntries are sorted in ascending order
            if (index->pepEntries[i].Mass > maxmass)
            {
                index->lcltotCnt--;
                index->lclmodCnt--;
                index->lastchunksize--;

                // if only one chunk
                if (index->nChunks == 1)
                    index->chunksize--;
            }
            // if small then break
            else
                break;
        }
    };

    // TODO: soft remvoe the invalid mods if present
    // removeInvalidPeps();

    if (status != SLM_SUCCESS)
        LBE_Deinitialize(index);

    return status;
}

status_t LBE_GeneratePeps(Index *index)
{
    status_t status = SLM_SUCCESS;
    uint_t interval = index->lclpepCnt;
    pepEntry *entries = index->pepEntries;
    uint_t seqlen = Seqs.at(0).length();

#ifdef USE_OMP
    uint_t threads = params.threads;
#endif /* USE_OMP */

#ifdef USE_OMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif /* USE_OMP */
    for (uint_t fill = 0; fill < interval; fill++)
    {
        uint_t idd = DSLIM_GenerateIndex(index, fill);

        entries[fill].Mass = MZs[idd];
        entries[fill].seqID = idd;
        entries[fill].sites.sites = 0x0;
        entries[fill].sites.modNum = 0x0;
    }

    return status;
}
/*
 * FUNCTION: LBE_Deinitialize
 *
 * DESCRIPTION: Deallocate all memory and
 *              reset variables
 *
 * INPUT: none
 *
 * OUTPUT:
 * @status: Status of execution
 */
status_t LBE_Deinitialize(Index *index) { return DSLIM_Deinitialize(index); }

/*
 * FUNCTION: LBE_Distribute
 *
 * DESCRIPTION: Apply Distribution Policy on peptides
 *
 * INPUT:
 * @threads   : Number of parallel threads
 * @policy    : Policy ID (enum)
 * @slm_chunks: Number of distribution chunks
 *
 * OUTPUT:
 * @status: Status of execution
 */
status_t LBE_Distribute(Index *index)
{
    status_t status = 0;
    uint_t N = index->lcltotCnt;
    uint_t speclen = (index->pepIndex.peplen-1) * params.maxz * iSERIES;
    uint_t maxchunksize = (MAX_IONS / speclen);
    uint_t maxchunksize2 = params.spadmem / (BYISIZE * params.threads);
    uint_t nchunks = 0;
    uint_t chunksize = 0;
    uint_t lastchunksize = 0;

    /* Calculate the chunksize */
    chunksize = std::min(N, maxchunksize);
    chunksize = std::min(chunksize, maxchunksize2);

    /* Set the number of chunks to p */
    nchunks = (N / chunksize);

    if ((N % chunksize) != 0)
    {
        nchunks += 1;
    }

    /* Calculate the size of last chunk */
    uint_t factor = N / chunksize;

    lastchunksize = ((N % chunksize) == 0)?
                     chunksize            :
                     N - (chunksize * factor);

    if (status == SLM_SUCCESS)
    {
        /* Return the number of chunks created */
        index->nChunks = nchunks;
        index->chunksize = chunksize;
        index->lastchunksize = lastchunksize;
    }

    return status;
}

/*
 * FUNCTION: LBE_CreatePartitions
 *
 * DESCRIPTION: Creates the partition size for the current node
 *
 * INPUT:
 * @index : (Distributed) Virtual Peptide ID.
 *
 * OUTPUT:
 * @status: Actual SPI peptide ID
 */
status_t LBE_CreatePartitions(Index *index)
{
    status_t status = SLM_SUCCESS;

    uint_t N = index->pepCount;
    uint_t p = params.nodes;
    uint_t myid = params.myid;

    uint_t chunksize = 0;

    /* More than one nodes in the system ? */
    if (p > 1)
    {
        /* Partition the pepCount */
        chunksize = N / p;

        if (((N % p) > 0) && ((N % p) > myid))
        {
            chunksize += 1;
        }

        index->lclpepCnt = chunksize;

        /* Partition the modCount */
        N = index->modCount;

        chunksize = N / p;

        if (((N % p) > 0) && ((N % p) > myid))
        {
            chunksize += 1;
        }

        index->lclmodCnt = chunksize;

        index->lcltotCnt = index->lclpepCnt + index->lclmodCnt;
    }
    else
    {
        index->lcltotCnt = index->totalCount;
        index->lclpepCnt = index->pepCount;
        index->lclmodCnt = index->modCount;
    }

    return status;
}

/*
 * FUNCTION: LBE_CountPeps
 *
 * DESCRIPTION: Count peptides in FASTA and the
 *              number of mods that will be generated
 *
 * INPUT:
 * @threads      : Number of parallel threads
 * @filename     : Path to FASTA file
 * @modconditions: Mod generation conditions
 *
 * OUTPUT:
 * @status: Status of execution
 */
status_t LBE_CountPeps(char_t *filename, Index *index, uint_t explen)
{
    status_t status = SLM_SUCCESS;
    string_t line;
    string_t modconditions = params.modconditions;
    uint_t maxmass= params.max_mass;
    uint_t minmass= params.min_mass;

    /* Initialize Index parameters */
    index->pepIndex.AAs = 0;
    index->pepCount = 0;
    index->modCount = 0;

    // print current progress
    printProgress(Database Indexing);

    /* Open file */
    file.open(filename);

    if (file.is_open())
    {
        while (getline(file, line))
        {
            if (line.at(0) != '>')
            {
                // remove any \r symbols at the eol
                if (line.at(line.length() - 1) == '\r')
                {
                    line = line.substr(0, line.size() - 1);
                }

                // transform to all upper case letters
                std::transform(line.begin(), line.end(), line.begin(), ::toupper);

                // check length
                if (line.length() != explen)
                {
                    status = ERR_INVLD_SIZE;
                    std::cerr << "Invalid peplen: " << line.length() << ", expected: " << explen << std::endl;
                }

                // validate precursor mass
                float_t pepmass = UTILS_CalculatePepMass((AA *)line.c_str(), line.length());

                // add to Seqs and MZ vectors
                if (pepmass >= minmass && pepmass <= maxmass)
                {
                    Seqs.emplace_back(std::move(line));
                    MZs.emplace_back(std::move(pepmass));
                }
            }
        }

        /* Close the file once done */
        file.close();

        // set the index properties
        index->pepCount = Seqs.size();
        index->pepIndex.AAs = Seqs[0].length() * Seqs.size();
    }
    else
    {
        std::cout << std::endl << "FATAL: Could not read FASTA file" << std::endl;
        status = ERR_INVLD_PARAM;
    }

    // Count the # of varmods given modification info
    if (status == SLM_SUCCESS)
        index->modCount = MODS_ModCounter();

    // check for errors in MODS_ModCounter
    if (index->modCount == (uint_t)(-1) || index->pepIndex.AAs != index->pepCount * explen)
        status = ERR_INVLD_SIZE;

    // Print if everything is okay 
    if (status == SLM_SUCCESS)
    {
        // Return the total count
        index->totalCount = index->pepCount + index->modCount;
        cumusize += index->totalCount;

        if (params.myid == 0)
        {
            std::cout << "Number of Peptides    =\t\t" << index->pepCount << std::endl;
            std::cout << "Number of Variants    =\t\t" << index->modCount << std::endl;
            std::cout << "Total Index Size      =\t\t" << index->totalCount << std::endl;
            std::cout << "Cumulative Index Size =\t\t" << cumusize << std::endl << std::endl;
        }
    }

    return status;
}
