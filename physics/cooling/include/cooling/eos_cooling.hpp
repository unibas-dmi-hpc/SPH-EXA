//
// Created by Noah Kubli on 09.07.23.
//

#ifndef SPHEXA_EOS_COOLING_HPP
#define SPHEXA_EOS_COOLING_HPP

template<typename HydroData, typename ChemData, typename Cooler>
void eos_cooling(size_t startIndex, size_t endIndex, HydroData& d, ChemData& chem, Cooler& cooler)
{
    using CoolingFields = typename Cooler::CoolingFields;
    using T         = typename HydroData::RealType;
    const auto* rho = d.rho.data();

    auto* p = d.p.data();
    auto* c = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        T pressure    = cooler.pressure(rho[i], d.u[i], cstone::getPointers(get<CoolingFields>(chem), i));
        T gamma       = cooler.adiabatic_index(rho[i], d.u[i], cstone::getPointers(get<CoolingFields>(chem), i));
        T sound_speed = std::sqrt(gamma * pressure / rho[i]);
        p[i]          = pressure;
        c[i]          = sound_speed;
    }
}
#endif // SPHEXA_EOS_COOLING_HPP
