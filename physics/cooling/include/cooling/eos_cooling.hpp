//
// Created by Noah Kubli on 09.07.23.
//

#pragma once

namespace cooling
{

//! @brief the maximum time-step based on local particles that Grackle can tolerate
template<class Dataset, typename Cooler, typename Chem>
auto coolingTimestep(size_t first, size_t last, Dataset& d, Cooler& cooler, Chem& chem)
{
    using T             = typename Dataset::RealType;
    using CoolingFields = typename Cooler::CoolingFields;

    std::vector<T> cooling_times(last - first);
    std::vector<double> rho_copy{d.rho.begin(), d.rho.end()};
    std::vector<double> u_copy{d.u.begin(), d.u.end()};

    cooler.cooling_time_arr(rho_copy.data() + first, u_copy.data() + first,
                            cstone::getPointers(get<CoolingFields>(chem), first), cooling_times.data(), last - first);
    T minTc = *std::min_element(cooling_times.begin(), cooling_times.end());
    return minTc;
//    T minTc(INFINITY);
//#pragma omp parallel for reduction(min : minTc)
//    for (size_t i = first; i < last; i++)
//    {
//        const T cooling_time = cooler.cooling_time(d.rho[i], d.u[i], cstone::getPointers(get<CoolingFields>(chem), i));
//        minTc                = std::min(std::abs(cooler.ct_crit * cooling_time), minTc);
//    }
//    return minTc;
}

template<typename HydroData, typename ChemData, typename Cooler>
void eos_cooling(size_t startIndex, size_t endIndex, HydroData& d, ChemData& chem, Cooler& cooler)
{
    using CoolingFields = typename Cooler::CoolingFields;
    using T             = typename HydroData::RealType;
    const auto* rho     = d.rho.data();

    auto* p = d.p.data();
    auto* c = d.c.data();

   // std::vector<double> rho_copy{d.rho.begin(), d.rho.end()};
   // std::vector<double> p_copy(endIndex - startIndex);
   // std::vector<double> u_copy{d.u.begin(), d.u.end()};
//
   // cooler.pressure_arr(rho_copy.data() + startIndex, u_copy.data() + startIndex,
   //                     cstone::getPointers(get<CoolingFields>(chem), startIndex), p_copy.data(), endIndex - startIndex);
   // std::copy(p_copy.begin(), p_copy.end(), d.p.begin() + startIndex);
//
   // std::vector<T> gammas(endIndex - startIndex);
   // cooler.adiabatic_index_arr(rho_copy.data() + startIndex, u_copy.data() + startIndex,
   //                     cstone::getPointers(get<CoolingFields>(chem), startIndex), gammas.data(), endIndex - startIndex);
//
#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        T pressure    = cooler.pressure(rho[i], d.u[i], cstone::getPointers(get<CoolingFields>(chem), i));
        T gamma       = cooler.adiabatic_index(rho[i], d.u[i], cstone::getPointers(get<CoolingFields>(chem), i));
        //T pressure = p[i];
        //T gamma = gammas[i - startIndex];
        T sound_speed = std::sqrt(gamma * pressure / rho[i]);
        p[i]          = pressure;
        c[i]          = sound_speed;
    }
}

} // namespace cooling