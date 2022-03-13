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
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include "testapp.hpp"

using namespace std;
using namespace hcp::gpu;

status_t main(int_t argc, char_t* argv[])
{
    status_t status = SLM_SUCCESS;

    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "This is a test app for CUDA functions, blocks and kernels." << std::endl;
    std::cout << "All code is disabled and is only for debugging/testing purposes." << std::endl;
    std::cout << "Feel free to modify and experiment with it. Enjoy!" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;

#if 0
    int n = 1e8;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    auto driver = cuda::driver::get_instance();

    // allocate host memory
    cuda::error_check(cuda::host_pinned_allocate(h_a, n));
    cuda::error_check(cuda::host_pinned_allocate(h_b, n));
    cuda::error_check(cuda::host_pinned_allocate(h_c, n));

    for (int i = 0; i < n; i++)
    {
        h_a[i] = i;
        h_b[i] = 2* i;
    }

    // allocate device memory
    cuda::error_check(cuda::device_allocate(d_a, n));
    cuda::error_check(cuda::device_allocate(d_b, n));
    cuda::error_check(cuda::device_allocate(d_c, n));

    // copy data to device
    cuda::error_check(cuda::H2D(d_a, h_a, n, driver));
    cuda::error_check(cuda::H2D(d_b, h_b, n, driver));

    driver->stream_sync();

    // run kernel
    cudaFuncSetAttribute(vector_add<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, 48*1024);
    vector_add<<<128, 256, 48*1024, driver->get_stream()>>>(d_a, d_b, d_c, n);

    // copy data back to host
    cuda::error_check(cuda::D2H(h_c, d_c, n, driver));

    // free device memory
    cuda::error_check(cuda::device_free(d_a));
    cuda::error_check(cuda::device_free(d_b));
    cuda::error_check(cuda::device_free(d_c));

    driver->stream_sync();

    // check result
    for (int i = 0; i < n; i++) 
    {
        if (h_c[i] != h_a[i] + h_b[i]) 
        {
            status = -1;
            std::cout << "FATAL: UNSUCCESSFUL at: " << i << " " << std::endl;
            break;
        }
    }

    // free host memory
    cuda::error_check(cuda::host_pinned_free(h_a));
    cuda::error_check(cuda::host_pinned_free(h_b));
    cuda::error_check(cuda::host_pinned_free(h_c));

#endif // 0

    return status;
}
