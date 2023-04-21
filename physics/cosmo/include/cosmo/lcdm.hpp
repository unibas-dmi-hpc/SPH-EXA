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
 * @brief Lambda Cold Dark Matter (CDM) Cosmology.
 *
 * @author Jonathan Coles <jonathan.coles@cscs.ch>
 */

#pragma once

#include <cmath>
#include "utils.hpp"

namespace cosmo
{

template<typename T, int rel_prec = -7, int abs_prec = -7>
class LambdaCDM : public Cosmology<T> // : public cstone::FieldStates<CosmologyData<T>>
{
public:

    T relativeError = pow(10, rel_prec);
    T absoluteError = pow(10, abs_prec);
    
    //! @brief Hubble Constant today [km/Mpc/s]
    T H0{0.0};
    //! @brief Matter Density
    T OmegaMatter{0.0};
    //! @brief Radiation Density
    T OmegaRadiation{0.0};
    //! @brief Cosmology Constant (vacuum density)
    T OmegaLambda{0.0};

    struct Parameters
    {
        T H0, OmegaMatter, OmegaRadiation, OmegaLambda;
    };

    //
    //! @brief Values from 2018 Planck final data release, TT,TE,EE+lowE+lensing+BAO
    //
    static constexpr Parameters Planck2018 =
    {
        .H0 = 67.66,
        .OmegaMatter = 0.3111,
        .OmegaRadiation = 0.0,
        .OmegaLambda = 0.6889,
    };


    LambdaCDM() = default;

    LambdaCDM(struct Parameters p)
        : LambdaCDM(p.H0, p.OmegaMatter, p.OmegaRadiation, p.OmegaLambda)
    {
    }

    LambdaCDM(T H0, T OmegaMatter, T OmegaRadiation, T OmegaLambda) 
        : H0(H0)
        , OmegaMatter(OmegaMatter)
        , OmegaRadiation(OmegaRadiation)
        , OmegaLambda(OmegaLambda)
    {
        if (H0 <= 0)
            throw std::domain_error("Hubble0 parameter must be strictly positive.");

        if (OmegaMatter < 0)
            throw std::domain_error("OmegaMatter parameter must be positive.");

        if (OmegaRadiation < 0)
            throw std::domain_error("OmegaRadiation parameter must be positive.");

        if (OmegaLambda < 0)
            throw std::domain_error("OmegaLambda parameter must be positive.");

        if (OmegaRadiation == 0 && OmegaLambda == 0)
            throw std::domain_error("In LambdaCDM at least one of OmegaRadiation or OmegaLambda must be positive. Otherwise use CDM.");

        if (OmegaMatter == 0)
        {
            if (OmegaRadiation == 0)
                throw std::domain_error("OmegaRadiation parameter must be strictly positive when OmegaMatter is zero.");
        }
    }

//  template<class Archive>
//  void loadOrStoreAttributes(Archive* ar)
//  {
//      //! @brief load or store an attribute, skips non-existing attributes on load.
//      auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
//      {
//          try
//          {
//              ar->stepAttribute(attribute, location, attrSize);
//          }
//          catch (std::out_of_range&)
//          {
//              std::cout << "Attribute " << attribute << " not set in file, setting to default value " << *location
//                        << std::endl;
//          }
//      };

//      optionalIO("Hubble0", &H0, 1);
//      optionalIO("OmegaMatter", &OmegaMatter, 1);
//      optionalIO("OmegaRadiation", &OmegaRadiation, 1);
//      optionalIO("OmegaLambda", &OmegaLambda, 1);
//  }

    T H(const T a)
    {
        T Omega0 = OmegaMatter + OmegaRadiation + OmegaLambda;
        T OmegaCurvature = T(1) - Omega0;

        T a2 = a * a;
        T a3 = a * a2;
        T a4 = a * a3;
        return (H0/a2) * sqrt(OmegaRadiation + OmegaMatter * a + OmegaCurvature
                * a2 + OmegaLambda * a4);
    }

    T time(const T a)
    {
        auto f = [this](const T x) { return 2. / (3. * x * H(pow(x, 2./3.))); };
        return romberg<T>(f, 0.0, pow(a, 1.5), 1e-2 * relativeError);
    }

    T a(T t)
    {
        auto f = [this, t](const T a) { return time(a) - t; };
        auto df = [this](const T a) { return 1.0/(a * H(a)); };

        return newton(f, df, t*H0, 0.0, 1.0e38, relativeError, absoluteError);
    }

    T driftTimeCorrection(T t, T dt) override
    {
        auto f = [this](const T x) { return -x / H(T(1.0) / x); };
        return romberg<T>(f, T(1.0) / a(t), T(1.0) / a(t + dt), relativeError);
    }

    T kickTimeCorrection(T t, T dt) override
    {
        auto f = [this](const T x) { return -T(1.0) / H(T(1.0) / x); };
        return romberg<T>(f, T(1.0) / a(t), T(1.0) / a(t + dt), relativeError);
    }
};

} // namespace cosmo

