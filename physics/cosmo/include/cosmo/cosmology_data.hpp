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

#include <cfloat>
#include <cassert>
#include <array>
#include <vector>
#include <variant>
#include <iostream>
#include <cmath>

#define DBG_COSMO if (0)

namespace cosmo
{

template<typename T, int rel_prec = -7, int abs_prec = -7>
class CosmologyData // : public cstone::FieldStates<CosmologyData<T>>
{
public:
    using RealType        = T;

    template<class ValueType>
    using FieldVector = std::vector<ValueType, std::allocator<ValueType>>;

    using FieldVariant =
        std::variant<FieldVector<float>*, FieldVector<double>*, FieldVector<unsigned>*, FieldVector<uint64_t>*>;

    T relativeError = pow(10, rel_prec);
    T absoluteError = pow(10, abs_prec);

    //
    // Non-zero default values from 2018 Planck final data release, TT,TE,EE+lowE+lensing+BAO
    //
    
    //! @brief Hubble Constant today [km/Mpc/s]
    T H0{0.0};
    //! @brief Matter Density
    T OmegaMatter{0.0};
    //! @brief Radiation Density
    T OmegaRadiation{0.0};
    //! @brief Cosmology Constant (vacuum density)
    T OmegaLambda{0.0};

    bool isComoving{false};


    struct Parameters
    {
        T H0, OmegaMatter, OmegaRadiation, OmegaLambda;
    };

    static constexpr struct Parameters Planck2018 =
    {
        .H0 = 67.66,
        .OmegaMatter = 0.3111,
        .OmegaRadiation = 0.0,
        .OmegaLambda = 0.6889,
    };

    static constexpr struct Parameters Static =
    {
        .H0 = 0.0,
        .OmegaMatter = 0.0,
        .OmegaRadiation = 0.0,
        .OmegaLambda = 0.0,
    };


    CosmologyData() = default;

    CosmologyData(struct Parameters cd)
        : CosmologyData(cd.H0, cd.OmegaMatter, cd.OmegaRadiation, cd.OmegaLambda)
    {
    }

    CosmologyData(T H0, T OmegaMatter, T OmegaRadiation, T OmegaLambda) 
        : H0(H0)
        , OmegaMatter(OmegaMatter)
        , OmegaRadiation(OmegaRadiation)
        , OmegaLambda(OmegaLambda)
    {
        isComoving = !(H0 == 0 && OmegaMatter == 0 && OmegaRadiation == 0 && OmegaLambda == 0);

        if (!isComoving) return;

        if (H0 <= 0)
            throw std::domain_error("Hubble0 parameter must be strictly positive.");

        if (OmegaMatter < 0)
            throw std::domain_error("OmegaMatter parameter must be positive.");

        if (OmegaRadiation < 0)
            throw std::domain_error("OmegaRadiation parameter must be positive.");

        if (OmegaLambda < 0)
            throw std::domain_error("OmegaLambda parameter must be positive.");

        if (OmegaMatter == 0)
        {
            if (OmegaRadiation == 0)
                throw std::domain_error("OmegaRadiation parameter must be strictly positive when OmegaMatter is zero.");
        }
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

        optionalIO("OmegaMatter", &OmegaMatter, 1);
        optionalIO("OmegaRadiation", &OmegaRadiation, 1);
        optionalIO("OmegaLambda", &OmegaLambda, 1);
    }

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
        if (!isComoving) 
            throw std::runtime_error("Calling t(a) for a static universe is undefined.");

        if (OmegaLambda == 0.0 && OmegaRadiation == 0.0)
        {
            if (OmegaMatter == 1.0)
            {
                assert(H0 > 0); // This situation should be catch in the constructor.
                return a == 0.0
                       ? 0.0
                       : 2.0 / (3.0 * H0) * pow(a,1.5);
            }

            if (OmegaMatter > 1.0)
            {
                assert(H0 >= 0); // This situation should be catch in the constructor.
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
                assert(H0 > 0); // This situation should be catch in the constructor.
                if (a == 0.0) return 0.0;
                T a0 = 1.0 / H0 / sqrt(1.0 - OmegaMatter);
                T A = 0.5 * OmegaMatter / (1.0 - OmegaMatter);
                T B = a0 * A;
                T eta = acosh(1.0 + a/A);
                return B * (sinh(eta) - eta);

            }

            if (OmegaMatter == 0.0)
            {
                assert(H0 > 0); // This situation should be catch in the constructor.
                if (a == 0.0) return 0.0;
                return a / H0;
            }

            assert(0); // This situation should be catch in the constructor.
        }

        return romberg<T>(
                [this](const T x)
                {
                    return 2. / (3. * x * H(pow(x, 2./3.)));
                }
                , 0.0
                , pow(a, 1.5)
                , 1e-2 * relativeError);
    }

    T a(T t)
    {
        if (!isComoving) return 1.0;

        return newton(
                [this, t](const T a)
                {
                    return time(a) - t;
                },
                [this](const T a)
                {
                    return 1.0/(a * H(a));
                },
                t*H0,
                0.0,
                1.0e38,
                relativeError,
                absoluteError);
    }

    T driftTimeCorrection(T t, T dt)
    {
        if (!isComoving) return dt;

        if (OmegaLambda == 0.0 && OmegaRadiation == 0.0)
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

            assert(0); // This situation should be catch in the constructor.
        }


        return romberg<T>(
                [this](const T x)
                {
                    return -x / H(T(1.0) / x);
                },
                T(1.0) / a(t),
                T(1.0) / a(t + dt),
                relativeError);
    }

    T kickTimeCorrection(T t, T dt)
    {
        if (!isComoving) return dt;

        if (OmegaLambda == 0.0 && OmegaRadiation == 0.0)
            return kickTimeCorrection_OnlyMatter(t, dt);

        return romberg<T>(
                [this](const T x)
                {
                    return -T(1.0) / H(T(1.0) / x);
                },
                T(1.0) / a(t),
                T(1.0) / a(t + dt),
                relativeError);
    }


private:

    T time_OnlyMatter(const T a)
    {
        assert(OmegaLambda == 0.0 && OmegaRadiation == 0.0);

        if (OmegaMatter == 1.0)
        {
            assert(H0 > 0); // This situation should be catch in the constructor.
            return a == 0.0
                   ? 0.0
                   : 2.0 / (3.0 * H0) * pow(a,1.5);
        }

        if (OmegaMatter > 1.0)
        {
            assert(H0 >= 0); // This situation should be catch in the constructor.
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
            assert(H0 > 0); // This situation should be catch in the constructor.
            if (a == 0.0) return 0.0;
            T a0 = 1.0 / H0 / sqrt(1.0 - OmegaMatter);
            T A = 0.5 * OmegaMatter / (1.0 - OmegaMatter);
            T B = a0 * A;
            T eta = acosh(1.0 + a/A);
            return B * (sinh(eta) - eta);

        }

        if (OmegaMatter == 0.0)
        {
            assert(H0 > 0); // This situation should be catch in the constructor.
            if (a == 0.0) return 0.0;
            return a / H0;
        }

        assert(0); // This situation should be catch in the constructor.
    }

    T driftTimeCorrection_OnlyMatter(T t, T dt)
    {
        assert(OmegaLambda == 0.0 && OmegaRadiation == 0.0);

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

        assert(0); // This situation should be catch in the constructor.
    }

    T kickTimeCorrection_OnlyMatter(T t, T dt)
    {
        assert(OmegaLambda == 0.0 && OmegaRadiation == 0.0);

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

        assert(0); // This situation should be catch in the constructor.
    }

};

template<typename T, int MAX_ITER = 40, class FnPtr, class FnPtr2>
T newton(FnPtr&& func, FnPtr2&& func_deriv, T x0, T l, T u, T rel_error = 1e-7, T abs_error = 1e-7)
{
    if (l > u)
        throw std::invalid_argument("Lower bound exceeds upper bound in Newton's Method.");

    T x0_save = x0;
    T x1 = x0;

    for (int i=0; i < MAX_ITER; i++)
    {
        T f = func(x0);
        if (f == 0) return x0;

        T fprime = func_deriv(x0);

        DBG_COSMO printf("%2i] %23.15e  ->  %23.15e  [l: %23.15e u: %23.15e] (diff: %23.15e  f: %23.15e  f': %23.15e)\n", i, x0_save, x0, l,u, fabs((x1 - x0_save)), f, fprime);

        if (f*fprime == 0) { x0 = std::min(u, x0+abs_error*(u - l)); continue; }
        //if (f*fprime == 0) { x0 = std::max(l, x0-abs_error*(u - l)); continue; }
 
        if (f*fprime < 0)
            l = x0;
        else 
            u = x0;

        x0_save = x0;

        x1 = x0 - f/fprime;

        if (fabs(x1 - x0) <= std::max(rel_error * std::max(fabs(x0), fabs(x1)), abs_error)) return x1;
        //if (fabs(x1 - x0) <= abs_error + rel_error * fabs(x0)) return x1;
        //if (fabs((x1 - x0)) <= abs_error) { return x1; }
        //if (fabs((x1 - x0) / x0) <= rel_error) { return x1; }

        if (l < x1 && x1 < u)
            x0 = x1;
        else
            x0 = 0.5*(l + u);
    }

    DBG_COSMO return -1234;
    throw std::runtime_error("Maximum number of iterations reached in Newton's Method.");
}

/*
 ** Romberg integrator for an open interval.
 */

template<typename T, int MAXLEVEL = 13, class FnPtr>
T romberg(FnPtr&& func, T a, T b, T eps)
{
    T tllnew;
    T tll;
    T tlk[MAXLEVEL+1];
    int n = 1;
    int nsamples = 1;

    tlk[0] = tllnew = (b-a)*func(0.5*(b+a));
    if (a == b) return tllnew;

    tll = FLT_MAX;

    while ((fabs((tllnew-tll)/tllnew) > eps) && (n < MAXLEVEL)) 
    {
        /*
         * midpoint rule.
         */

        nsamples *= 3;
        T dx = (b-a)/nsamples;

        T s = 0;
        for (int i=0; i<nsamples/3; i++) 
        {
            s += dx*func(a + (3*i + 0.5)*dx);
            s += dx*func(a + (3*i + 2.5)*dx);
        }

        T tmp = tlk[0];
        tlk[0] = tlk[0]/3.0 + s;

        /*
         * Romberg extrapolation.
         */

        for (int i=0; i < n; i++) 
        {
            T k = pow(9.0, i+1.0);
            T tlknew = (k*tlk[i] - tmp) / (k - 1.0);

            tmp = tlk[i+1];
            tlk[i+1] = tlknew;
        }

        tll = tllnew;
        tllnew = tlk[n];
        n++;
    }

    //printf("%23.15e\n", tllnew);
    //printf("%23.15e\n", tll);
    //printf("%23.15e\n", fabs((tllnew-tll)/(tllnew)));
    //printf("%23.15e\n", eps);
    //assert(fabs((tllnew-tll)/(tllnew)) <= eps);

    return tllnew;
}

} // namespace cosmo
