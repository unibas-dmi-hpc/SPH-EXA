
#include "stir_init.hpp"
#include "st_ounoise.hpp"
#include "st_calcPhases.hpp"
#include "st_calcAccel.hpp"
#include <vector>
#include <iostream>
int main()
{
    using T            = double;
    T      dt          = 0.1;
    T      eps         = 1.e-16;
    size_t stMaxModes  = 100000;
    T      Lbox        = 1.0;
    T      velocity    = 0.3;
    int    stSeed      = 251299;
    size_t stSpectForm = 1;

    T twopi     = 8 * std::atan(1.0);
    T stEnergy  = 5.0e-3 * std::pow(velocity, 3) / Lbox;
    T stStirMin = (1.0 - eps) * twopi / Lbox;
    T stStirMax = (3.0 + eps) * twopi / Lbox;

    size_t         dim         = 3;
    T              stDecay     = Lbox / (2.0 * velocity);
    T              stSolWeight = 0.5;
    T              stSolWeightNorm;
    T              stOUVar;
    std::vector<T> stAmpl(stMaxModes);
    std::vector<T> stMode(stMaxModes * dim);

    size_t stNModes;

    stir_init(stAmpl, stMode, stNModes, stSolWeight, stSolWeightNorm, stOUVar, stDecay, Lbox, Lbox, Lbox, stMaxModes,
              stEnergy, stStirMax, stStirMin, dim, stSpectForm);

    std::vector<T> stOUPhases(6 * stNModes);
    st_ounoiseinit(stOUPhases, 6 * stNModes, stOUVar, stSeed);

    st_ounoiseupdate(stOUPhases, 6 * stNModes, stOUVar, dt, stDecay, stSeed);

    // Problem!!!, st_OUphases, st_nmodes, st_OUvar & st_seed must be output too, need to store values between
    // iterations
    std::vector<T> st_aka(dim * stNModes);
    std::vector<T> st_akb(dim * stNModes);

    st_calcPhases(stNModes, dim, stOUPhases, stSolWeight, stMode, st_aka, st_akb);

    std::vector<T> xCoord{-0.4, -0.2, 0.0, 0.2, 0.4, -0.5};
    std::vector<T> yCoord{0.4, -0.2, 0.0, -0.2, 0.4, -0.5};
    std::vector<T> zCoord{0.4, 0.2, 0.0, -0.2, -0.4, -0.5};
    std::vector<T> accx{0.0, 0.0, 0.0, 0.0, 0.0, -0.5};
    std::vector<T> accy{0.0, 0.0, 0.0, 0.0, 0.0, -0.5};
    std::vector<T> accz{0.0, 0.0, 0.0, 0.0, 0.0, -0.5};

    size_t npart = 6;

    st_calcAccel(0, npart, dim, xCoord, yCoord, zCoord, accx, accy, accz, stNModes, stMode, st_aka, st_akb, stAmpl,
                 stSolWeightNorm);

    for (int i = 0; i < npart; ++i)
    {

        std::cout << accx[i] << ' ' << accy[i] << ' ' << accz[i] << std::endl;
    }
    /*
    accx, accy, accz
    ----------------
    3.662459567254532E-01  1.259446841690182E-01 -1.125844123111067E-01
    1.384560977280313E-02 -3.799734689699691E-02 -1.793841332102599E-01
   -8.006708921589355E-02  5.960035530834547E-02 -4.877338914905545E-02
    2.394216045884232E-02 -7.479786887141186E-02 -1.467975759746282E-01
   -5.687601045457229E-02  1.256956770910929E-01  7.292293303985517E-02
    */
}
