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

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include "common.hpp"
#include "utils.h"

/*
 * FUNCTION: MODS_ModCounter
 *
 * DESCRIPTION: Counts the number of modifications for
 *              all peptides in SLM Peptide Index
 *
 * INPUT:
 * @threads   : Number of parallel threads
 * @conditions: Conditions of mod generation
 *
 * OUTPUT:
 * @cumulative: Number of mods
 */
ull_t  MODS_ModCounter();

/*
 * FUNCTION: MODS_GenerateMods
 *
 * DESCRIPTION: Generate modEntries for all peptide
 *              sequences in the SLM Peptide Index
 *
 * INPUT:
 * @threads   : Number of parallel threads
 * @modCount  : Number of mods that should be generated
 * @conditions: Conditions of mod generation
 *
 * OUTPUT:
 * @status: Status of execution
 */
status_t MODS_GenerateMods(Index * index);

status_t MODS_Initialize();
