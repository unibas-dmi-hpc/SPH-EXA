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

#include <memory>
#include <cmath>
#include <cfloat>

#define DBG_COSMO_UTILS if (0)

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

        DBG_COSMO_UTILS printf("%2i] %23.15e  ->  %23.15e  [l: %23.15e u: %23.15e] (diff: %23.15e  f: %23.15e  f': %23.15e)\n", i, x0_save, x0, l,u, fabs((x1 - x0_save)), f, fprime);

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

    DBG_COSMO_UTILS return -1234;
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

