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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#pragma once 

#include <vector>
#include <array>
#include "common.hpp"
#include "slm_dsts.h"
#include "utils.h"

std::array<ull_t, 2> DBCounter(string_t &filename);

float_t CalculatePepMass(AA *seq, uint_t len);

void InitializeModInfo(SLM_vMods *vMods);

float_t CalculateModMass(AA *seq, uint_t len, uint_t vModInfo);

ull_t ModCounter(const std::vector<string_t> &Seqs);