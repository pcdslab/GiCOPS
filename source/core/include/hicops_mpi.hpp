/*
 * Copyright (C) 2022  Muhammad Haseeb, and Fahad Saeed
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


#include <type_traits>
#include "common.hpp"

namespace hcp
{

namespace mpi
{

//
// FUNCTION: barrier (wrapper for MPI_Barrier)
//
inline status_t barrier()
{
    status_t status = SLM_SUCCESS;

#ifdef USE_MPI
    status = MPI_Barrier(MPI_COMM_WORLD);
#endif // USE_MPI

    return status;
}


} // namespace mpi
} //namespace hcp