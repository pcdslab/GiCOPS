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

#include <thread>
#include "utils.h"
#include "slm_dsts.h"
#include "cuda/superstep1/kernel.hpp"
#include "aamasses.hpp"

/* Global Mods Info  */
SLM_vMods      gModInfo;

extern gParams params;

// Macro to extract AA masses
#define GETAA(x,z)                 ((AAMass[AAidx(x)]) + (StatMods[AAidx(x)]) + ((PROTON) * (z)))

/*
 * FUNCTION: UTILS_GetNumProcs
 *
 * DESCRIPTION: Get the number of processors
 *              from the environment variable
 *
 * INPUT: none
 *
 * OUTPUT:
 * @procs: Number of processors
 */
uint_t UTILS_GetNumProcs()
{
    uint_t procs = 1;

#if (_WIN32 == 1)
    std::cout <<"HostOS: Windows\n";
    char var[] = "NUMBER_OF_PROCESSORS";
    char *val = std::getenv(var);
    procs = std::atoi(val);
#else
    std::cout <<"HostOS: Linux\n";
    procs = std::thread::hardware_concurrency();
#endif /* _WIN32 */

    return procs;
}

/*
 * FUNCTION: UTILS_Factorial
 *
 * DESCRIPTION: Calculate the factorial of a number
 *
 * INPUT:
 * @n : input value for which to calculate factorial
 *
 * OUTPUT:
 * @factorial : the factorial of the input number n
 */
ull_t factorial(ull_t n) { return (n < 2) ? 1 : factorial(n - 1) * n; }

/*
 * FUNCTION: UTILS_Shuffle
 *
 * DESCRIPTION: Shuffles an array

 *
 * INPUT:
 * @arr : Array to shuffle
 * @N   : Size of array
 *
 * OUTPUT:
 * @status: Status of execution
 */
status_t UTILS_Shuffle(uint_t *arr, uint_t N)
{
    status_t status = SLM_SUCCESS;

    /* Check if not already initialized */
    if (arr != NULL)
    {
        /* Create the default seed */
        ull_t defseed = 0xdefceed;

        /* Provide the default seed to UINT_Shuffle */
        return UTILS_ShuffleI(arr, N, defseed);
    }
    else
    {
        status = ERR_INVLD_PTR;
    }

    return status;
}

/*
 * FUNCTION: UTILS_ShuffleI
 *
 * DESCRIPTION: Shuffles an array using specific seed
 *
 * INPUT:
 * @arr : Array to shuffle
 * @N   : Size of array
 * @seed: The seed to use
 *
 * OUTPUT:
 * @status: Status of execution
 */
status_t UTILS_ShuffleI(uint_t *arr, uint_t N, ull_t seed)
{
    status_t status = SLM_SUCCESS;
    uint_t *indices = arr;

    /* Check if not already initialized */
    if (indices != NULL)
    {
        /* Check if default seed,
         * then create a random seed */
        if (seed == 0xdefceed)
        {
            seed = time(0);
        }

        /* Shuffle the sequence using the provided seed or use time(); */
        std::shuffle(indices, (indices + N), std::default_random_engine(seed));
    }
    else
    {
        status = ERR_INVLD_PTR;
    }

    return status;
}

/*
 * FUNCTION: UTILS_GenerateSpectrum
 *
 * DESCRIPTION: Generates theoretical spectrum of a peptide
 *
 * INPUT:
 * @seq     : Peptide sequence
 * @len     : Length of peptide
 * @Spectrum: Pointer to the theoretical spectrum
 *
 * OUTPUT:
 * @mass: Precursor mass of peptide
 */
float_t UTILS_GenerateSpectrum(char_t *seq, uint_t len, uint_t *Spectrum)
{
    /* Calculate Peptide sequences Mass */
    float_t mass = UTILS_CalculatePepMass(seq, len);

    uint_t maxz = params.maxz;
    uint_t scale = params.scale;

    /* If there is a non-AA char, the mass will be -ve */
    /* FIXME: No stupid characters should be allowed in
     *        peptide sequence */
    if (mass > 0)
    {
        // vector for floating point spectrum
        std::vector<float_t> f_Spectrum(iSERIES * maxz * (len-1));

        /* Generate Spectrum */
        for (uint_t z = 0; z < maxz; z++)
        {
            /* Indices for b and y series start */
            uint_t bstart = z * (len - 1);
            uint_t ystart = z * (len - 1) + maxz * (len - 1);

            /* Mass of fragment = [M + (z-1)H]/z */

            /* First b-ion */
            f_Spectrum[bstart] = GETAA(seq[0], z+1);
            /* First y-ion */
            f_Spectrum[ystart] = GETAA(seq[len-1], z+1) + H2O;

            /* Loop until length - 1 only */
            for (uint_t l = 1; l < len - 1; l++)
            {
                /* Extract b-ions */
                f_Spectrum[bstart + l] = f_Spectrum[bstart + (l-1)] + GETAA(seq[l], 0);

                /* Extract y-ions */
                f_Spectrum[ystart + l] = f_Spectrum[ystart + (l-1)] + GETAA(seq[len-1-l], 0);
            }
        }

        // integrize and copy to Spectrum
        for (uint_t z = 0; z < maxz; z++)
        {
            /* Indices for b and y series start */
            uint_t bstart = z * (len - 1);
            uint_t ystart = z * (len - 1) + maxz * (len - 1);

            /* Loop until length - 1 only */
            for (uint_t l = 0; l < len - 1; l++)
            {
                /* Extract b-ions */
                Spectrum[bstart + l] = (uint_t)((f_Spectrum[bstart + l] * scale)/(z+1));

                /* Extract y-ions */
                Spectrum[ystart + l] = (uint_t)((f_Spectrum[ystart + l] * scale)/(z+1));
            }
        }

        // clear the vector
        f_Spectrum.clear();

    }

    return mass;
}

/*
 * FUNCTION: UTILS_CalculatePepMass
 *
 * DESCRIPTION: Calculate precursor mass of a peptide
 *
 * INPUT:
 * @seq: Peptide sequence
 * @len: Length of peptide
 *
 * OUTPUT:
 * @mass: Precursor mass of peptide
 */
float_t UTILS_CalculatePepMass(AA *seq, uint_t len)
{
    /* Initialize mass to H2O */
    float_t mass = H2O;

    /* Calculate peptide mass */
    for (uint_t l = 0; l < len; l++)
    {
        mass += AAMass[AAidx(seq[l])] + StatMods[AAidx(seq[l])];
    }

    return mass;
}

/*
 * FUNCTION: UTILS_InitializeModInfo
 *
 * DESCRIPTION: Initialize the gModInfo structure
 *
 * INPUT:
 * @vMods: gModInfo information
 *
 * OUTPUT:
 * @status: Status of execution
 */

status_t UTILS_SetParams(gParams *params)
{
#if defined(USE_GPU)

    if (params->useGPU)
        // init mods info on GPU as well
        hcp::gpu::cuda::s1::initParams(params);

#endif // USE_GPU
    return SLM_SUCCESS;
}

status_t UTILS_InitializeModInfo(SLM_vMods *vMods)
{
    // initialize mod info on CPU
    gModInfo = *vMods;

#if defined(USE_GPU)

    if (params.useGPU)
        // init mods info on GPU as well
        hcp::gpu::cuda::s1::initMods(&params.vModInfo);

#endif // defined(USE_GPU)

    return SLM_SUCCESS;
}
/*
 * FUNCTION: UTILS_CalculateModMass
 *
 * DESCRIPTION: Calculate precursor mass of a mod
 *
 * INPUT:
 * @seq     : Modified Peptide sequence
 * @len     : Length of Modified Peptide
 * @vmodInfo: Information about the modifications
 *
 * OUTPUT:
 * @mass: Precursor mass of modified peptide
 */
float_t UTILS_CalculateModMass(AA *seq, uint_t len, uint_t vModInfo)
{
    /* Initialize mass to H2O */
    float_t mass = H2O;

    /* Calculate peptide mass */
    for (uint_t l = 0; l < len; l++)
    {
        mass += AAMass[AAidx(seq[l])];
        mass += StatMods[AAidx(seq[l])];
    }

    /* Add the mass of modifications present in the peptide */
    uint_t start = 0x0F;
    uint_t modNum = vModInfo & start;

    while (modNum != 0)
    {
        mass += ((float_t)(gModInfo.vmods[modNum - 1].modMass)/params.scale);
        start = (start << 4);
        modNum = ((vModInfo & start) >> start);
    }


    return mass;
}

/*
 * FUNCTION: UTILS_GenerateModSpectrum
 *
 * DESCRIPTION: Generates theoretical spectrum for a mod
 *
 * INPUT:
 * @seq     : Modified peptide sequence
 * @len     : Length of modified peptide
 * @Spectrum: Pointer to the theoretical spectrum
 * @modInfo : Modified peptide information
 *
 * OUTPUT:
 * @mass: Precursor mass of modified peptide
 */
float_t UTILS_GenerateModSpectrum(char_t *seq, uint_t len, uint_t *Spectrum, modAA modInfo)
{
    /* Calculate Mod Peptide Mass */
    status_t status = SLM_SUCCESS;
    float_t mass = 0;

    const double_t minmass = params.min_mass;
    const double_t maxmass = params.max_mass;
    const uint_t maxz = params.maxz;
    const uint_t scale = params.scale;

    char_t modPos[len] = {};
    int_t modNums[MAX_MOD_TYPES] = {};
    int_t modSeen = 0;

    /* Check if valid modInfo */
    if (modInfo.sites == 0 || modInfo.modNum == 0)
    {
        status = ERR_INVLD_MOD;
        mass = NAA;
    }

    /* Compute Mod Mass */
    if (status == SLM_SUCCESS)
    {
        mass = UTILS_CalculateModMass(seq, len, modInfo.modNum);
    }

    uint_t bitmask = modInfo.modNum;

    /* Check if a valid precursor mass */
    if (mass > minmass && mass < maxmass)
    {
        for (uint_t i = 0; i < MAX_MOD_TYPES; i++)
        {
            modNums[i] = bitmask & 0x0F;
            modNums[i] -= 1;
            bitmask = bitmask / 16;

            if (modNums[i] != -1)
            {
                modSeen++;
            }
        }

        for (uint_t i = 0; i < len; i++)
        {
            modPos[i] = ISBITSET(modInfo.sites,i) ? 1 : 0;
        }

        if (mass > 0)
        {
            // vector for floating point spectrum
            std::vector<float> f_Spectrum(iSERIES * maxz * (len-1));

            /* Generate Normal Spectrum */
            for (uint_t z = 0; z < maxz; z++)
            {
                /* Indices for b and y series start */
                uint_t bstart = z * (len - 1);
                uint_t ystart = z * (len -1)  + maxz * (len - 1);

                /* Mass of fragment = [M + (z-1)H]/z */

                /* First b-ion */
                f_Spectrum[bstart] = GETAA(seq[0], z+1);
                /* First y-ion */
                f_Spectrum[ystart] = GETAA(seq[len-1], z+1) + H2O;

                /* Loop until length - 1 only */
                for (uint_t l = 1; l < len - 1; l++)
                {
                    /* Extract b-ions */
                    f_Spectrum[bstart + l] = f_Spectrum[bstart + (l - 1)] + GETAA(seq[l], 0);

                    /* Extract y-ions */
                    f_Spectrum[ystart + l] = f_Spectrum[ystart + (l - 1)] + GETAA(seq[len-1-l], 0);
                }
            }

            /* Adjust b-ions with additional masses */
            for (uint_t z = 0; z < maxz; z++)
            {
                /* Indices for b series start */
                uint_t bstart = z * (len-1);
                uint_t counter = 0;

                /* Loop until length - 1 only */
                for (uint_t l = 0; l < len - 1; l++)
                {
                    counter += modPos[l];

                    for (uint_t k = 0; k < counter; k++)
                        f_Spectrum[bstart + l] += static_cast<double>(gModInfo.vmods[modNums[k]].modMass) / scale;


                    // integrize and write to Spectrum
                    Spectrum[bstart + l] = (uint_t)(f_Spectrum[bstart + l] * scale /(z + 1));
                }
            }

            /*  Adjust y-ions with additional masses */
            for (uint_t z = 0; z < maxz; z++)
            {
                /* Indices for y series start */
                uint_t ystart = z * (len -1) + maxz * (len -1);
                uint_t counter = 0;

                /* Loop until length - 1 only */
                for (int_t l = (len - 1); l > 0; l--)
                {
                    counter += modPos[l];

                    for (uint_t k = 0; k < counter; k++)
                        f_Spectrum[ystart + (len - 1) - l] += static_cast<double>(gModInfo.vmods[modNums[modSeen - 1 - k]].modMass)/scale;

                    // integrize and write to Spectrum
                    Spectrum[ystart + (len - 1) - l] = (uint_t)(f_Spectrum[ystart + (len - 1) - l] * scale / (z + 1));
                }
            }

            // clear the vector
            f_Spectrum.clear();
        }
    }

    return mass;
}

