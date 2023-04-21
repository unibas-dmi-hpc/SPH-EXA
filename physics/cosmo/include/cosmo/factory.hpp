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
 * @brief Cosmological parameters and associated functions.
 *
 * @author Jonathan Coles <jonathan.coles@cscs.ch>
 */

#pragma once

#include "static.hpp"
#include "cdm.hpp"
#include "lcdm.hpp"

namespace cosmo
{

template<typename T>
std::unique_ptr<Cosmology<T>> cosmologyFactory(const char *fname [[maybe_unused]])
{
    throw std::runtime_error("filename argument to cosmology factory not yet supported.");
}

template<typename T>
std::unique_ptr<Cosmology<T>> cosmologyFactory(const std::string& fname)
{
    return cosmologyFactory<T>(fname.c_str());
}

template<typename T>
std::unique_ptr<Cosmology<T>> cosmologyFactory(T H0=0, T OmegaMatter=0, T OmegaRadiation=0, T OmegaLambda=0)
{
    if (H0 == 0 && OmegaMatter == 0 && OmegaRadiation == 0 && OmegaLambda == 0)
        return std::make_unique<StaticUniverse<T>>();

    if (H0 <= 0)
        throw std::domain_error("Hubble0 parameter must be strictly positive when other cosmological parameters are non-zero.");

    if (OmegaLambda == 0 && OmegaRadiation == 0)
        return std::make_unique<CDM<T>>(H0, OmegaMatter);

    return std::make_unique<LambdaCDM<T>>(H0, OmegaMatter, OmegaRadiation, OmegaLambda);
}

template<typename T>
std::unique_ptr<Cosmology<T>> cosmologyFactory(const struct CDM<T>::Parameters& p)
{
    return cosmologyFactory(p.H0, p.OmegaMatter);
}

template<typename T>
std::unique_ptr<Cosmology<T>> cosmologyFactory(const struct LambdaCDM<T>::Parameters& p)
{
    return cosmologyFactory(p.H0, p.OmegaMatter, p.OmegaRadiation, p.OmegaLambda);
}

} // namespace cosmo

