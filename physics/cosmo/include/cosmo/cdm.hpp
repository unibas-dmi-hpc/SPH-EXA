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
 * @brief Cold Dark Matter (CDM) Cosmology.
 *
 * @author Jonathan Coles <jonathan.coles@cscs.ch>
 */

#pragma once

#include "utils.hpp"

namespace cosmo
{

template<typename T, int rel_prec = -7, int abs_prec = -7>
class CDM : public Cosmology<T> // : public cstone::FieldStates<CosmologyData<T>>
{
public:

    T relativeError = pow(10, rel_prec);
    T absoluteError = pow(10, abs_prec);
    
    //! @brief Hubble Constant today [km/Mpc/s]
    T H0{0.0};
    //! @brief Matter Density
    T OmegaMatter{0.0};

    struct Parameters
    {
        T H0, OmegaMatter;
    };
    
    CDM() = default;

    CDM(struct Parameters cd) : CDM(cd.H0, cd.OmegaMatter)
    {
    }

    CDM(T H0, T OmegaMatter) : H0(H0), OmegaMatter(OmegaMatter)
    {
        if (H0 <= 0)
            throw std::domain_error("Hubble0 parameter must be strictly positive.");

        if (OmegaMatter < 0)
            throw std::domain_error("OmegaMatter parameter must be positive.");

        // apparantly we can have a universe with nothing in it?!?
    }

    template<class Archive>
    void loadOrStoreAttributes(Archive* ar)
    {
        //! @brief load or store an attribute, skips non-existing attributes on load.
        auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            try
            {
                ar->stepAttribute(attribute, location, attrSize);
            }
            catch (std::out_of_range&)
            {
                std::cout << "Attribute " << attribute << " not set in file, setting to default value " << *location
                          << std::endl;
            }
        };

        optionalIO("Hubble0", &H0, 1);
        optionalIO("OmegaMatter", &OmegaMatter, 1);
    }

    T H(const T a)
    {
        T OmegaCurvature = T(1) - OmegaMatter;

        T a2 = a * a;
        return (H0/a2) * sqrt(OmegaMatter * a + OmegaCurvature * a2);
    }

    T time(const T a)
    {
        if (OmegaMatter == 1.0)
        {
            return a == 0.0
                   ? 0.0
                   : 2.0 / (3.0 * H0) * pow(a,1.5);
        }

        if (OmegaMatter > 1.0)
        {
            if (H0 == 0.0)
            {
                T B = 1.0 / sqrt(OmegaMatter);
                T eta = acos(1.0 - a);
                return B * (eta - sin(eta));
            }

            if (a == 0.0) return 0.0;

            T a0 = 1.0 / H0 / sqrt(OmegaMatter - 1.0);
            T A = 0.5 * OmegaMatter / (OmegaMatter - 1.0);
            T B = a0 * A;
            T eta = acos(1.0 - a/A);
            return B * (eta - sin(eta));

        }

        if (OmegaMatter > 0.0)
        {
            if (a == 0.0) return 0.0;
            T a0 = 1.0 / H0 / sqrt(1.0 - OmegaMatter);
            T A = 0.5 * OmegaMatter / (1.0 - OmegaMatter);
            T B = a0 * A;
            T eta = acosh(1.0 + a/A);
            return B * (sinh(eta) - eta);

        }

        if (OmegaMatter == 0.0)
        {
            if (a == 0.0) return 0.0;
            return a / H0;
        }

        assert(0); // This situation should be caught in the constructor.
    }

    T a(T t)
    {
        auto f = [this, t](const T a) { return time(a) - t; };
        auto df = [this](const T a) { return 1.0/(a * H(a)); };
        return newton(f, df, t*H0, 0.0, 1.0e38, relativeError, absoluteError);
    }


    T driftTimeCorrection(T t [[maybe_unused]], T dt [[maybe_unused]]) override
    {
        if (OmegaMatter == 1.0)
        {
        }

        if (OmegaMatter > 1.0)
        {
        }

        if (OmegaMatter > 0.0)
        {
        }

        if (OmegaMatter == 0.0)
        {
        }

        assert(0); // This situation should be caught in the constructor.
        return 0;
    }

    T kickTimeCorrection(T t [[maybe_unused]], T dt [[maybe_unused]]) override
    {
        if (OmegaMatter == 1.0)
        {
        }

        if (OmegaMatter > 1.0)
        {
        }

        if (OmegaMatter > 0.0)
        {
        }

        if (OmegaMatter == 0.0)
        {
        }

        assert(0); // This situation should be caught in the constructor.
        return 0;
    }

};

} // namespace cosmo

