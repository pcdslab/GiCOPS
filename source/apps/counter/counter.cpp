/*
 *  Copyright (C) 2019  Muhammad Haseeb, Fahad Saeed
 *  Florida International University, Miami, FL
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <algorithm>
#include "counter.hpp"
#include "aamasses.hpp"

using namespace std;

extern gParams params;

/* Global Mods Info  */
SLM_vMods      gModInfo;

extern gParams params;

std::array<ull_t, 2> DBCounter(string &filename)
{
    status_t status = SLM_SUCCESS;
    string_t line;
    float_t pepmass = 0.0;
    string_t modconditions = params.modconditions;
    uint_t maxmass= params.max_mass;
    uint_t minmass= params.min_mass;
    ull_t ions = 0;

    std::vector<string_t> Seqs;

    ull_t localpeps = 0;

    // open file
    std::ifstream file(filename);

    if (file.is_open())
    {
        while (getline(file, line))
        {
            if (line.at(0) != '>')
            {
                /* Linux has a weird \r at end of each line */
                if (line.at(line.length() - 1) == '\r')
                {
                    line = line.substr(0, line.size() - 1);
                }

                /* Transform to all upper case letters */
                std::transform(line.begin(), line.end(), line.begin(), ::toupper);

                /* Calculate mass of peptide */
                pepmass = CalculatePepMass((AA *)line.c_str(), line.length());

                /* Check if the peptide mass is legal */
                if (pepmass >= minmass && pepmass <= maxmass)
                {
                    Seqs.push_back(line);
                    localpeps++;
                }
            }
        }

        /* Close the file once done */
        file.close();
    }
    else
    {
        std::cout << std::endl << "FATAL: Could not read FASTA file" << std::endl;
        status = ERR_INVLD_PARAM;
    }

    /* Count the number of variable mods given
     * modification information */
    if (status == SLM_SUCCESS)
    {
        localpeps += ModCounter(Seqs);
    }

    ions += (localpeps * ((Seqs.at(0).length() - 1) * params.maxz * iSERIES));

    Seqs.clear();

    return std::array<ull_t, 2>{localpeps, ions};
}

/*
 * FUNCTION: UTILS_CalculatePepMass
 */
float_t CalculatePepMass(AA *seq, uint_t len)
{
    /* Initialize mass to H2O */
    float_t mass = H2O;

    /* Calculate peptide mass */
    for (uint_t l = 0; l < len; l++)
    {
        mass += AAMass[AAidx(seq[l])];
        mass += StatMods[AAidx(seq[l])];
    }

    return mass;
}

/*
 * FUNCTION: InitializeModInfo
 */
void InitializeModInfo(SLM_vMods *vMods) { gModInfo = *vMods; }

/*
 * FUNCTION: CalculateModMass
 */
float_t CalculateModMass(AA *seq, uint_t len, uint_t vModInfo)
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