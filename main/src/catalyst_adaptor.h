#pragma once

#include "sph/particles_data.hpp"

#include <catalyst.hpp>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace CatalystAdaptor
{

void Initialize(int argc, char* argv[])
{
    // TODO go over all the SPH-exa runtime flags, ignoring them, until we find our own flags
    // Else, specifiy our Catalyst flags *first*
    conduit_cpp::Node node;
    for (auto cc = 1; cc < argc; ++cc)
    {
        if (strcmp(argv[cc], "--catalyst") == 0 && (cc + 1) < argc)
        {
            const auto fname = std::string(argv[cc + 1]);
            const auto path  = "catalyst/scripts/script" + std::to_string(cc - 1);
            node[path + "/filename"].set_string(fname);
        }
    }

    // indicate that we want to load ParaView-Catalyst
    node["catalyst_load/implementation"].set_string("paraview");
    node["catalyst_load/search_paths/paraview"] = PARAVIEW_IMPL_DIR;

    catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
    if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to initialize Catalyst: " << err << std::endl; }
    std::cout << "CatalystAdaptor::Initialize" << std::endl;
}

template<class DataType>
void Execute(DataType& d, long startIndex, long endIndex)
{
    conduit_cpp::Node exec_params;
    // add time/cycle information
    auto state = exec_params["catalyst/state"];
    state["timestep"].set_external(&d.iteration);
    state["time"].set_external(&d.ttot);

    // We only have 1 channel here. Let's name it 'grid'.
    auto channel = exec_params["catalyst/channels/grid"];

    // Since this example is using Conduit Mesh Blueprint to define the mesh,
    // we set the channel's type to "mesh".
    channel["type"].set(
        "mesh"); // used to indicate that this channel is specified in accordance to the Conduit Mesh protocol.

    // now create the mesh.
    auto mesh = channel["data"];

    // start with coordsets (of course, the sequence is not important, just make
    // it easier to think in this order).
    mesh["coordsets/coords/type"].set("explicit");
    mesh["coordsets/coords/values/x"].set_external(&d.x[startIndex], endIndex - startIndex, 0, sizeof(double));
    mesh["coordsets/coords/values/y"].set_external(&d.y[startIndex], endIndex - startIndex, 0, sizeof(double));
    mesh["coordsets/coords/values/z"].set_external(&d.z[startIndex], endIndex - startIndex, 0, sizeof(double));

    // Next, add topology
    mesh["topologies/mesh/type"].set("unstructured");
    mesh["topologies/mesh/elements/shape"].set("point");
    mesh["topologies/mesh/coordset"].set("coords");

    std::vector<int> conn(endIndex - startIndex);
    std::iota(conn.begin(), conn.end(), 0);
    mesh["topologies/mesh/elements/connectivity"].set_external(conn);

    // Finally, add particle properties
    auto fields = mesh["fields"];
    // rho is vertex-data.
    fields["Density/association"].set("vertex");
    fields["Density/topology"].set("mesh");
    fields["Density/volume_dependent"].set("false");
    fields["Density/values"].set_external(&d.rho[startIndex], endIndex - startIndex, 0, sizeof(double)); // zero-copy
    // vx is vertex-data.
    fields["vx/association"].set("vertex");
    fields["vx/topology"].set("mesh");
    fields["vx/volume_dependent"].set("false");
    fields["vx/values"].set_external(&d.vx[startIndex], endIndex - startIndex); // zero-copy
    // vy is vertex-data.
    fields["vy/association"].set("vertex");
    fields["vy/topology"].set("mesh");
    fields["vy/volume_dependent"].set("false");
    fields["vy/values"].set_external(&d.vy[startIndex], endIndex - startIndex); // zero-copy
    // vz is vertex-data.
    fields["vz/association"].set("vertex");
    fields["vz/topology"].set("mesh");
    fields["vz/volume_dependent"].set("false");
    fields["vz/values"].set_external(&d.vz[startIndex], endIndex - startIndex); // zero-copy
    // rank is vertex-data.
    fields["rank/association"].set("vertex");
    fields["rank/topology"].set("mesh");
    fields["rank/volume_dependent"].set("false");
    fields["rank/values"].set_external(&d.vz[startIndex], endIndex - startIndex); // zero-copy
    // u is vertex-data.
    // u is not initialized properly and cannot be transferred to Catalyst yet.
    // TODO: look for the initial condition that can initialize u.
    // fields["Internal Energy/association"].set("vertex");
    // fields["Internal Energy/topology"].set("mesh");
    // fields["Internal Energy/volume_dependent"].set("false");
    // fields["Internal Energy/values"].set_external(&d.u[startIndex], endIndex - startIndex); // zero-copy
    // p is vertex-data.
    fields["Pressure/association"].set("vertex");
    fields["Pressure/topology"].set("mesh");
    fields["Pressure/volume_dependent"].set("false");
    fields["Pressure/values"].set_external(&d.p[startIndex], endIndex - startIndex); // zero-copy
    // h is vertex-data.
    fields["h/association"].set("vertex");
    fields["h/topology"].set("mesh");
    fields["h/volume_dependent"].set("false");
    fields["h/values"].set_external(&d.h[startIndex], endIndex - startIndex); // zero-copy
    // c is vertex-data.
    fields["Speed of sound/association"].set("vertex");
    fields["Speed of sound/topology"].set("mesh");
    fields["Speed of sound/volume_dependent"].set("false");
    fields["Speed of sound/values"].set_external(&d.c[startIndex], endIndex - startIndex); // zero-copy
    // m is vertex-data.
    fields["Mass/association"].set("vertex");
    fields["Mass/topology"].set("mesh");
    fields["Mass/volume_dependent"].set("false");
    fields["Mass/values"].set_external(&d.m[startIndex], endIndex - startIndex); // zero-copy

    catalyst_status err = catalyst_execute(conduit_cpp::c_node(&exec_params));
    if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to execute Catalyst: " << err << std::endl; }
}

void Finalize()
{
    conduit_cpp::Node node;
    catalyst_status   err = catalyst_finalize(conduit_cpp::c_node(&node));
    if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to finalize Catalyst: " << err << std::endl; }

    std::cout << "CatalystAdaptor::Finalize" << std::endl;
}
} // namespace CatalystAdaptor
