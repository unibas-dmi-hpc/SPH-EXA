/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Switch and stub classes for ParticlesData to abstract and manage GPU device acceleration behavior
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include "cstone/tree/accel_switch.hpp"

namespace sphnnet
{
/*! @brief nuclear data device facade */
template<typename T, typename I, typename M>
class DeviceNuclearDataFacade
{
public:
    void resizeAttached(size_t) {}

    void resize(size_t) {}

    template<class... Ts>
    void setConserved(Ts...)
    {
    }

    template<class... Ts>
    void setDependent(Ts...)
    {
    }

    template<class... Ts>
    void release(Ts...)
    {
    }

    template<class... Ts>
    void acquire(Ts...)
    {
    }

    inline static constexpr std::array fieldNames{0};
};

/*! @brief Forward declaration of DeviceNuclearDataType, defined to DeviceNuclearDataFacade for now ! */
template<typename T, typename I, typename Tmass>
class DeviceNuclearDataType;

template<class Accelerator, class T, class KeyType, class Tmass>
using DeviceNuclearData_t = typename cstone::AccelSwitchType<Accelerator, DeviceNuclearDataFacade,
                                                             DeviceNuclearDataType>::template type<T, KeyType, Tmass>;

} // namespace sphnnet