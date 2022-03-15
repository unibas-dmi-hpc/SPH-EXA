/*
 * MIT License
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
 *
 * @brief This class produces 1d solutions for a sedov blast wave propagating through a density gradient: rho =
 * rho**(-omega) , in planar(1D), cylindrical(2D) or spherical geometry(3D) for the 'standard', 'singular' and 'vaccum'
 * cases.
 *
 *        - standard case: a nonzero solution extends from the shock to the origin,       where the pressure is finite.
 *        - singular case: a nonzero solution extends from the shock to the origin,       where the pressure vanishes.
 *        - vacuum case  : a nonzero solution extends from the shock to a boundary point, where the density vanishes
 * making the pressure meaningless.
 *
 *        This routine is a C++ conversion of one Fortran code based in these two papers:
 *        - "Evaluation of the sedov-von neumann-taylor blast wave solution", Jim Kamm, la-ur-00-6055
 *        - "The sedov self-similiar point blast solutions in nonuniform media", David Book, shock waves, 4, 1, 1994
 *
 *        Although the ordinary differential equations are analytic, the sedov expressions appear to become singular for
 * various combinations of parameters and at the lower limits of the integration range. All these singularies are
 * removable and done so by this routine.
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 *
 */

#pragma once

#include <string>
#include <vector>
#include <functional>

using namespace std;

class SedovSolution
{
public:
    // Public global variables in the shock peak
    static double rho_shock; // Density
    static double p_shock;   // Pressure
    static double vel_shock; // Velocity 1D
    static double u_shock;   // Internal energy
    static double cs_shock;  // Sound speed

    static double sedovSol(const size_t          dim,     // geometry factor: 1=planar, 2=cylindircal, 3=spherical
                           const double          time,    // temporal point where solution is desired [seconds]
                           const double          eblast,  // energy of blast in the wave front [erg]
                           const double          omega_i, // density power law exponent in 'rho = rho0 * r**(-omega)'
                           const double          gamma_i, // gamma law equation of state
                           const double          rho0,    // ambient density g/cm**3 in 'rho = rho0 * r**(-omega)'
                           const double          u0,      // ambient internal energy [erg/g]
                           const double          p0,      // ambient pressure [erg/cm**3]
                           const double          vel0,    // ambient material speed [cm/s]
                           const double          cs0,     // ambient sound speed [cm/s]
                           const vector<double>& r,       // out: spatial points where solution is desired [cm]
                           vector<double>&       rho,     // out: density  [g/cm**3]
                           vector<double>&       p,       // out: presssure [erg/cm**3]
                           vector<double>&       u,       // out: specific internal energy [erg/g]
                           vector<double>&       vel,     // out: velocity [cm/s]
                           vector<double>&       cs);           // out: sound speed [cm/s]

private:
    // Constants
    static inline const double eps = 1.e-10;    // eps controls the integration accuracy, don't get too greedy or the
                                                // number of function evaluations required kills.
    static inline const double eps2   = 1.e-30; // eps2 controls the root find accuracy
    static inline const double osmall = 1.e-4;  // osmall controls the size of transition regions

    // Private global variables
    static double xgeom, omega, gamma;               //
    static double gamm1, gamp1, gpogm, xg2;          //
    static bool   lsingular, lstandard, lvacuum;     //
    static bool   lomega2, lomega3;                  //
    static double a0, a1, a2, a3, a4, a5;            //
    static double a_val, b_val, c_val, d_val, e_val; //
    static double rwant, vwant;                      //
    static double r2, v0, vv, rvv;                   //
    static double gam_int;                           //

    static void sedov_funcs(const double v,      // Similarity variable v
                            double&      l_fun,  // out: l_fun is book's zeta
                            double&      dlamdv, // out: l_fun derivative
                            double&      f_fun,  // out: f_fun is book's V
                            double&      g_fun,  // out: g_fun is book's D
                            double&      h_fun);      // out: h_fun is book's P

    static double efun01(const double v); //

    static double efun02(const double v); //

    static double sed_v_find(const double v); //

    static double sed_r_find(const double r); //

    static void midpnt(const size_t                   n,    //
                       function<double(const double)> func, //
                       const double                   a,    //
                       const double                   b,    //
                       double&                        s);                          //

    static double midpowl_func(function<double(const double)> funk, //
                               const double                   x,    //
                               const double                   aa);                    //

    static void midpowl(const size_t                   n,    //
                        function<double(const double)> funk, //
                        const double                   aa,   //
                        const double                   bb,   //
                        double&                        s);                          // out:

    static double midpowl2_func(function<double(const double)> funk, //
                                const double                   x,    //
                                const double                   aa);                    //

    static void midpowl2(const size_t                   n,    //
                         function<double(const double)> funk, //
                         const double                   aa,   //
                         const double                   bb,   //
                         double&                        s);                          // out:

    static void polint(double*      xa, //
                       double*      ya, //
                       const size_t n,  //
                       const double x,  //
                       double&      y,  // out:
                       double&      dy);     // out:

    static void
    qromo(function<double(const double)>                                                                    func,    //
          const double                                                                                      a,       //
          const double                                                                                      b,       //
          const double                                                                                      eps,     //
          double&                                                                                           ss,      //
          function<void(const size_t, function<double(const double)>, const double, const double, double&)> choose); //

    static double zeroin(
        const double                   ax, // Left endpoint of initial interval
        const double                   bx, // Right endpoint of initial interval
        function<double(const double)> f,  // Function subprogram which evaluates f(x) for any x in the interval [ax,bx]
        const double tol);                 // Desired length of the interval of uncertainty of the final result (>= 0.)
};
