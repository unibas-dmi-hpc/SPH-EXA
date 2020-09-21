//
// Created by Gabriel Zihlmann on 27.03.20.
//

#ifndef SPH_EXA_MINI_APP_DEBUGUTILS_HPP
#define SPH_EXA_MINI_APP_DEBUGUTILS_HPP

#endif //SPH_EXA_MINI_APP_DEBUGUTILS_HPP
#include <fenv.h>
#include <xmmintrin.h>
#include <iostream>


void show_fe_exceptions(void)
{
//    printf("current exceptions raised: ");
    if(fetestexcept(FE_DIVBYZERO))     printf(" FE_DIVBYZERO");
    if(fetestexcept(FE_INEXACT))       printf(" FE_INEXACT");
    if(fetestexcept(FE_INVALID))       printf(" FE_INVALID");
    if(fetestexcept(FE_OVERFLOW))      printf(" FE_OVERFLOW");
    if(fetestexcept(FE_UNDERFLOW))     printf(" FE_UNDERFLOW");
    if(fetestexcept(FE_ALL_EXCEPT)==0) printf(" none");
    printf("\n");
}

bool serious_fpe_raised(void)
{
    return fetestexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
}

void enable_fe_hwexceptions(void)
{
    // call this in your testcase if you want to raise a HW Exception when a NAN or INF occurs¡
    printf("Enabling HW Exceptions for floating point errors\n");
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_OVERFLOW);
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_UNDERFLOW);
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_DIV_ZERO);
//    feenableexcept(FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW | FE_DIVBYZERO); // will raise FPError (HW signal and kill if not handled) Unsuitable for debug because no good data access doesn't work on macos
//    feenableexcept(FE_ALL_EXCEPT); // all is bad because inexact happens very often and is not so bad
}

void crash_me(void)
{
    double zero = 0.0;
    double crash = 1.0/zero;
    printf("should raise exception because of %f\n", crash);
}

bool all_check_FPE(const std::string msg)
{
    bool fpe_raised = serious_fpe_raised();
    if (fpe_raised)
    {
        std::cout << "msg: " << msg << std::endl;
        show_fe_exceptions();
    }
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &fpe_raised, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
#endif
    return fpe_raised;
}
