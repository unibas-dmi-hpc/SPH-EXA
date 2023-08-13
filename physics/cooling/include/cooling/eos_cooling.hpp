//
// Created by Noah Kubli on 09.07.23.
//

#ifndef SPHEXA_EOS_COOLING_HPP
#define SPHEXA_EOS_COOLING_HPP

template<typename HydroData, typename ChemData, typename Cooler>
void eos_cooling(size_t startIndex, size_t endIndex, HydroData& d, ChemData& chem, Cooler& cooler)
{
    using T         = typename HydroData::RealType;
    const auto* rho = d.rho.data();

    auto* p = d.p.data();
    auto* c = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {

        T pressure =
            cooler.pressure(d.rho[i], d.u[i], get<"HI_fraction">(chem)[i], get<"HII_fraction">(chem)[i],
                            get<"HM_fraction">(chem)[i], get<"HeI_fraction">(chem)[i], get<"HeII_fraction">(chem)[i],
                            get<"HeIII_fraction">(chem)[i], get<"H2I_fraction">(chem)[i], get<"H2II_fraction">(chem)[i],
                            get<"DI_fraction">(chem)[i], get<"DII_fraction">(chem)[i], get<"HDI_fraction">(chem)[i],
                            get<"e_fraction">(chem)[i], get<"metal_fraction">(chem)[i],
                            get<"volumetric_heating_rate">(chem)[i], get<"specific_heating_rate">(chem)[i],
                            get<"RT_heating_rate">(chem)[i], get<"RT_HI_ionization_rate">(chem)[i],
                            get<"RT_HeI_ionization_rate">(chem)[i], get<"RT_HeII_ionization_rate">(chem)[i],
                            get<"RT_H2_dissociation_rate">(chem)[i], get<"H2_self_shielding_length">(chem)[i]);
        T gamma = cooler.adiabatic_index(
            d.rho[i], d.u[i], get<"HI_fraction">(chem)[i], get<"HII_fraction">(chem)[i], get<"HM_fraction">(chem)[i],
            get<"HeI_fraction">(chem)[i], get<"HeII_fraction">(chem)[i], get<"HeIII_fraction">(chem)[i],
            get<"H2I_fraction">(chem)[i], get<"H2II_fraction">(chem)[i], get<"DI_fraction">(chem)[i],
            get<"DII_fraction">(chem)[i], get<"HDI_fraction">(chem)[i], get<"e_fraction">(chem)[i],
            get<"metal_fraction">(chem)[i], get<"volumetric_heating_rate">(chem)[i],
            get<"specific_heating_rate">(chem)[i], get<"RT_heating_rate">(chem)[i],
            get<"RT_HI_ionization_rate">(chem)[i], get<"RT_HeI_ionization_rate">(chem)[i],
            get<"RT_HeII_ionization_rate">(chem)[i], get<"RT_H2_dissociation_rate">(chem)[i],
            get<"H2_self_shielding_length">(chem)[i]);
        T sound_speed = std::sqrt(gamma * pressure / d.rho[i]);
        p[i]          = pressure;
        c[i]          = sound_speed;
    }
}
#endif // SPHEXA_EOS_COOLING_HPP
