/*
 * Copyright (C) 2019  Fatima Afzali, Muhammad Haseeb, Fahad Saeed
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
#include "counter.hpp"

using namespace std;

/* Global Variables */
static auto Comb = hcp::utils::Comb<hcp::utils::maxcombs>();
uint_t limit = 0;

/* External Variables */
extern gParams params;

/* Static Functions */
static longlong_t count(const string_t &s, const std::vector<string_t> &tokens);


/*
 * FUNCTION: partition2
 *
 * DESCRIPTION:
 *
 * INPUT:
 * @a: Count of AAs can be modified in seq
 * @b: Count of mods that can be made of condition in A.
 *
 * OUTPUT:
 * @sum: Number of ways to pick upto b elements
 *       from multiset A.
 */
longlong_t partition2(vector<int_t> &a, int_t b)
{
    int_t sum = 0;
    vector<int_t> a2;

    /* Set up basic conditions */
    for (int_t t : a)
        sum += t;

    if (b > sum)
        b = sum;

    if (b <= 0)
        return (longlong_t) (b == 0);

    if (a.size() == 0)
        return 1;

    sum = 0;

    for (uint_t i = 1; i < a.size(); i++)
        a2.push_back(a[i]);

    for (int_t i = 0; i < a[0] + 1; i++)
        sum += Comb[a[0]][i] * partition2(a2, b - i);

    return sum;
}

/*
 * FUNCTION: partition2
 *
 * DESCRIPTION:
 *
 * INPUT:
 * @A    : List of AA counts for each mod condition
 * @B    : Count of mods per modcondition
 * @limit: Max mods per peptide sequence
 *
 * OUTPUT:
 * @sum:
 */
longlong_t partition3(std::vector<std::vector<int_t>> &A, vector<int_t> &B, int_t limit)
{
    if (A.size() == 1)
    {
        if (B[0] <= limit)
        {
            return partition2(A[0], B[0]);
        }
        else
        {
            return partition2(A[0], limit);
        }
    }

    longlong_t sum = 0;
    vector<vector<int_t> > A2;
    vector<int_t> B2;

    for (uint_t i = 1; i < A.size(); i++)
    {
        A2.push_back(A[i]);
        B2.push_back(B[i]);
    }

    for (int_t i = 0; i < B[0] + 1; i++)
    {
        sum += (partition2(A[0], i) - partition2(A[0], i - 1)) * partition3(A2, B2, limit - i);
    }

    return sum;
}

/*
 * FUNCTION: count
 *
 * DESCRIPTION: Main partition method
 *
 * INPUT:
 * @s         : Peptide sequence
 * @conditions: Mod conditions
 *
 * OUTPUT:
 * @nmods: Number of mods generated for @s
 */
static longlong_t count(const string_t &s, const std::vector<string_t> &tokens)
{
    map<char_t, int_t> AAcounts;
    vector<vector<int_t>> A;
    vector<int_t> B;
    vector<int_t> temp;

    for (auto c : s)
    {
        AAcounts[c] += 1;
    }

    for (uint_t i = 0; i < (tokens.size() - 1) / 2; i++)
    {
        for (uint_t j = 0; j < tokens[2 * i + 1].length(); j++)
        {
            temp.push_back(AAcounts[tokens[2 * i + 1][j]]);
        }

        A.push_back(temp);
        temp.clear();
        B.push_back(stoi(tokens[2 * i + 2]));
    }

    return partition3(A, B, limit);
}

/*
 * FUNCTION: MODS_ModCounter
 *
 * DESCRIPTION: Counts the number of modifications for
 *              all peptides in the index
 *
 * INPUT:
 * @threads   : Number of parallel threads
 * @conditions: Conditions of mod generation
 *
 * OUTPUT:
 * @cumulative: Number of mods
 */
ull_t ModCounter(const vector<string_t> &Seqs)
{
    //static thread_local uint_t limit = 0;
    ull_t cumulative = 0;

    int threads = params.threads; 
    string_t conditions = params.modconditions;

    std::vector<string_t> tokens;

    string_t token;
    stringstream ss(conditions);

    while (ss >> token)
    {
        tokens.push_back(token);
    }

    limit = stoi(tokens[0]);

    /* Return if no mods to generate */
    if (limit > 0)
    {

        /* Parallel modcounter */
#ifdef USE_OMP
#pragma omp parallel for num_threads (threads) schedule(static) reduction(+: cumulative)
#endif // USE_OMP
        for (uint_t i = 0; i < Seqs.size(); i++)
        {
            cumulative += count(Seqs.at(i), tokens) - 1;
        }
    }

    tokens.clear();

    return cumulative;

}
