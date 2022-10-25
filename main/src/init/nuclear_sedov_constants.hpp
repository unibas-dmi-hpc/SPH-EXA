#pragma once

#include <map>
#include <string>

namespace sphexa
{

std::map<std::string, double> nuclearSedovConstants()
{
    std::map<std::string, double> ret{
        {"dim", 3},           {"gamma", 5. / 3.}, {"omega", 0.},           {"r0", 0.},    {"r1", 0.5}, {"mTotal", 1e9},
        {"energyTotal", 1e9}, {"T0", 1e9},        {"width", 0.1},          {"rho0", 1e9}, {"u0", 1e1}, {"p0", 0.},
        {"vr0", 0.},          {"cs0", 0.},        {"firstTimeStep", 1e-6}, {"mui", 10}};

    ret["ener0"] = ret["energyTotal"] / std::pow(M_PI, 1.5) / 1. / std::pow(ret["width"], 3.0);
    return ret;
}

} // namespace sphexa
