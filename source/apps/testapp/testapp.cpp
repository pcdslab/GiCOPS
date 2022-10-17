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
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

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

    auto driver = cuda::driver::get_instance();

    uint_t *d_data = nullptr;

    int_t size = 320;
    int search = 126;

    uint_t ddata[] = {89812, 132976, 139204, 190684, 202672, 276292, 304840, 311698, 314272, 399700, 420904, 435088, 440110, 440146, 443188, 444394, 551836, 556300, 588952, 608662, 615466, 772695, 783478, 810730, 984196, 1001367, 1029735, 1029879, 1076446, 1076482, 1076518, 1076626, 1086742, 1094751, 1098586, 1149615, 1331235, 1331974, 1332010, 1332046, 1332154, 1332262, 1354491, 1368675, 1368711, 1373139, 1396342, 1409878, 1611423, 1650412, 1662795, 1726767, 1727595, 1727631, 1736506, 1750510, 1766439, 1900503, 2044683, 2073771, 2090385, 2103687, 2106027, 2110743, 2166814, 2192589, 2205423, 2205783, 2205855, 2209203, 2251863, 2252115, 2385262, 2437515, 2446011, 2461743, 2463003, 2499831, 2501631, 2509263, 2509947, 2537451, 2539287, 2586933, 2589417, 2601927, 2696139, 2696211, 2857419, 2882259, 2887299, 2891835, 2918439, 2921787, 2922687, 2936979, 2965113, 2970027, 3022911, 3034071, 3056967, 3060495, 3084651, 3084687, 3094785, 3094821, 3096369, 3101553, 3138399, 3147381, 3221362, 3221650, 3269511, 3351483, 3351519, 3351627, 3355695, 3356775, 3390183, 3419163, 3434517, 3456855, 3464199, 3468591, 3492891, 3508065, 3508101, 3510225, 3526695, 3526731, 3554019, 3554703, 3554739, 3560877, 3562299, 3563343, 3568041, 3594879, 3666645, 3684303, 3685635, 3769047, 3812427, 3813435, 3819303, 3820059, 3841263, 3847995, 3850191, 3868011, 3871395, 3885939, 3889917, 3916089, 3920067, 3949461, 3949497, 3958245, 3959037, 3963753, 3963789, 3980331, 3988449, 4007889, 4007925, 4010193, 4015881, 4021893, 4021929, 4022685, 4041153, 4042809, 4056759, 4063491, 4065381, 4065561, 4084479, 4114953, 4191237, 4270635, 4277331, 4296087, 4297527, 4308831, 4311621, 4325499, 4338981, 4340799, 4349421, 4361931, 4372659, 4379877, 4383765, 4383801, 4407489, 4407525, 4408317, 4410441, 4418487, 4428441, 4431465, 4432401, 4433157, 4462713, 4476429, 4482369, 4522869, 4538187, 4559787, 4639959, 4647609, 4683879, 4684275, 4685229, 4713219, 4728987, 4730895, 4750029, 4750065, 4757031, 4760271, 4762899, 4772889, 4781709, 4786731, 4786767, 4795515, 4795821, 4795947, 4801779, 4802697, 4804587, 4809843, 4840353, 4847301, 4847337, 4857759, 4882293, 4884147, 4884183, 4916043, 4937283, 4966605, 5020911, 5038515, 5107311, 5107347, 5118993, 5122053, 5125941, 5137929, 5142033, 5142573, 5148315, 5148351, 5151519, 5154849, 5172003, 5197113, 5209551, 5210181, 5231097, 5281839, 5294169, 5296455, 5304627, 5309109, 5381631, 5390037, 5390073, 5397291, 5398875, 5413635, 5445783, 5454513, 5454549, 5468769, 5477805, 5477841, 5479101, 5510133, 5519943, 5521545, 5537637, 5560425, 5585193, 5591097, 5601249, 5603481, 5603661, 5623713, 5628195, 5648301, 5651613, 5651649, 5653665, 5653701, 5659137, 5706621, 5706765, 5706801, 5749299, 5754465, 5761629, 5771601, 5780745, 5791365, 5813757, 5815413, 5879205, 5883165, 5888565, 5894433, 5896197, 5896233, 5901813, 5906745, 5912343, 5986845, 6030909,};

    uint_t *h_data = &ddata[0];

    cuda::error_check(cuda::device_allocate_async(d_data, 320, driver->stream[0]));

    std::cout << "--------------------------------------------------------------------" << std::endl;

    cuda::error_check(cuda::H2D(d_data, h_data, size, driver->stream[0]));

    driver->stream_sync(0);

    int *d_val = nullptr;
    cuda::error_check(cuda::device_allocate_async(d_val, 1, driver->stream[0]));

    lbound<<<1, 256, 1024, driver->stream[0]>>>(d_data, d_data + size, (uint_t)5939316, d_val);

    driver->stream_sync(0);

    int v2 = thrust::lower_bound(thrust::device, d_data, d_data + size, (uint_t)5939316) - d_data;

    int h_val;
    int *h_val_ptr = &h_val;
    cuda::error_check(cuda::D2H(h_val_ptr, d_val, 1, driver->stream[0]));

    std::cout << std::endl << "--------------------------------------------------------------------" << std::endl;

    std::cout << search << " is at index: " << h_val << std::endl;
    std::cout << search << " is at index: " << v2 << std::endl;

    std::cout << std::endl << "--------------------------------------------------------------------" << std::endl;

    ubound<<<1, 256, 1024, driver->stream[0]>>>(d_data, d_data + size, (uint_t)5987951, d_val);

    driver->stream_sync(0);

    v2 = thrust::upper_bound(thrust::device, d_data, d_data + size, (uint_t)5987951) - d_data;

    cuda::error_check(cuda::D2H(h_val_ptr, d_val, 1, driver->stream[0]));

    std::cout << std::endl << "--------------------------------------------------------------------" << std::endl;

    std::cout << search << " is at index: " << h_val << std::endl;
    std::cout << search << " is at index: " << v2 << std::endl;

    std::cout << std::endl << "--------------------------------------------------------------------" << std::endl;

    for (int i = 0; i < size; i++)
    {
        std::cout << i << ":"<< h_data[i] << ", ";
    }

    std::cout << std::endl;

    return status;
}
