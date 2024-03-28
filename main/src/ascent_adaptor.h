#pragma once

#include "sph/particles_data.hpp"
#include "propagator/ipropagator.hpp"

#include <ascent/ascent.hpp>
#include "conduit_blueprint.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

namespace AscentAdaptor
{
ascent::Ascent a;
conduit::Node  actions;
double         dataTransferTime_, renderingTime_;

template<class DataType>
void Initialize([[maybe_unused]] DataType& d, [[maybe_unused]] long startIndex)
{
    conduit::Node ascent_options;
    ascent_options["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
    a.open(ascent_options);

    // Create an action that tells Ascent to:
    //  add a scene (s1) with one plot (p1)
    //  that will render a pseudocolor of
    //  the mesh field `rho`

    conduit::Node& add_pipe      = actions.append();
    add_pipe["action"]           = "add_pipelines";
    conduit::Node& pipes         = add_pipe["pipelines"];
    pipes["pl1/f1/type"]         = "threshold";
    conduit::Node& thresh_params = pipes["pl1/f1/params"];
    thresh_params["field"]       = "x";
    thresh_params["min_value"]   = -0.5;
    thresh_params["max_value"]   = 0.5;

    conduit::Node& add_scene = actions.append();
    add_scene["action"]      = "add_scenes";

    // declare a scene (s1) and pseudocolor plot (p1)
    conduit::Node& scenes                = add_scene["scenes"];
    scenes["s1/plots/p1/type"]           = "pseudocolor";
    scenes["s1/plots/p1/pipeline"]       = "pl1";
    scenes["s1/plots/p1/field"]          = "x";
    scenes["s1/renders/r1/image_prefix"] = "DensityThreshold1.4.%05d";

    double vec3[3];
    vec3[0] = vec3[1] = vec3[2] = 0;
    scenes["s1/renders/r1/camera/look_at"].set_float64_ptr(vec3, 3);

    vec3[0] = -2.1709899968205337;
    vec3[1] = 1.797907520678797;
    vec3[2] = 1.8029059671481107;
    scenes["s1/renders/r1/camera/position"].set_float64_ptr(vec3, 3);

    vec3[0] = 0.4479257557058854;
    vec3[1] = 0.8420981185224633;
    vec3[2] = -0.30038854198560727;
    scenes["s1/renders/r1/camera/up"].set_float64_ptr(vec3, 3);
    scenes["s1/renders/r1/camera/zoom"] = 2;

    /*
    scenes["s1/renders/r1/type"] = "cinema";
    scenes["s1/renders/r1/phi"] = 8;
    scenes["s1/renders/r1/theta"] = 8;
    scenes["s1/renders/r1/db_name"] = "example_db";
    */

    /* IO to disk */
    conduit::Node& add_extr = actions.append();
    add_extr["action"]      = "add_extracts";
    conduit::Node& savedata = add_extr["extracts"];

    // add a relay extract that will write mesh data to
    // blueprint hdf5 files
    savedata["e1/type"] = "relay";
    // savedata["e1/pipeline"] = "pl1";
    savedata["e1/params/path"]     = "out_export_particles";
    savedata["e1/params/protocol"] = "blueprint/mesh/hdf5";
}

/*! @brief Add a volume-independent vertex field to a mesh
 *
 * @tparam       FieldType  and elementary type like float, double, int, ...
 * @param[inout] mesh       the mesh to add the field to
 * @param[in]    name       the name of the field to use within the mesh
 * @param[in]    field      field base pointer to publish to the mesh as external (zero-copy)
 * @param[in]    start      first element of @p field to reveal to the mesh
 * @param[in]    end        last element of @p field to reveal to the meash
 */
template<class FieldType>
void addField(conduit::Node& mesh, const std::string& name, FieldType* field, size_t start, size_t end)
{
    mesh["fields/" + name + "/association"] = "vertex";
    mesh["fields/" + name + "/topology"]    = "mesh";
    mesh["fields/" + name + "/values"].set_external(field + start, end - start);
    mesh["fields/" + name + "/volume_dependent"].set("false");
}

/*! @brief Add a volume-independent vertex field to a mesh
 *
 * @tparam       FieldType  and elementary type like float, double, int, ...
 * @param[inout] mesh       the mesh to add the field to
 * @param[in]    name       the name of the field to use within the mesh
 * @param[in]    field      An rvalue to publish to the mesh. NOT zero-copy!
 */
template<class FieldType>
void addRankBenchmarkField(conduit::Node& mesh, const std::string& name, FieldType field)
{
    mesh["fields/" + name + "/association"] = "vertex";
    mesh["fields/" + name + "/topology"]    = "ranks";
    mesh["fields/" + name + "/values"]      = field;
    mesh["fields/" + name + "/volume_dependent"].set("false");
}

template<class DataType>
void Execute(DataType& d, long startIndex, long endIndex)
{
    size_t        rank = 0;
    conduit::Node mesh;
    mesh["state/cycle"].set_external(&d.iteration);
    mesh["state/time"].set_external(&d.ttot);
    mesh["state/domain_id"] = rank;

    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"].set_external(&d.x[startIndex], endIndex - startIndex);
    mesh["coordsets/coords/values/y"].set_external(&d.y[startIndex], endIndex - startIndex);
    mesh["coordsets/coords/values/z"].set_external(&d.z[startIndex], endIndex - startIndex);

    /* ===================================================== */
    // Create a grid mesh for contour calculation
    /* ===================================================== */

    mesh["topologies/mesh/type"]           = "unstructured";
    mesh["topologies/mesh/coordset"]       = "coords";
    mesh["topologies/mesh/elements/shape"] = "point";
    std::vector<conduit_int32> conn(endIndex - startIndex);
    std::iota(std::begin(conn), std::end(conn), 0);
    mesh["topologies/mesh/elements/connectivity"].set_external(conn);
    std::vector<conduit_int64> ranks(endIndex - startIndex, rank);
    addField(mesh, "ranks", ranks.data(), 0, endIndex - startIndex);
    // For all the fields to be rendered, output them with Conduit
    // It's the users' problem if they specify a field that cannot be visualized
    // And we have a dictionary for the namings. For now it's just hardcoded
    std::map<std::string, std::string> visDictionary = {{"x", "X"},
                                                        {"y", "Y"},
                                                        {"z", "Z"},
                                                        {"vx", "VX"},
                                                        {"vy", "VY"},
                                                        {"vz", "VZ"},
                                                        {"ax", "AX"},
                                                        {"ay", "AY"},
                                                        {"az", "AZ"},
                                                        {"v", "Velocity"},
                                                        {"p", "Pressure"},
                                                        {"rho", "Density"},
                                                        {"prho", "Prho"},
                                                        {"m", "Mass"},
                                                        {"u", "Internal Energy"},
                                                        {"c", "Speed of Sound"},
                                                        {"h", "Smoothing Length"}};

    auto indicesDone   = d.visFieldIndices;
    auto namesDone     = d.visFieldNames;
    auto fieldPointers = d.data();

    for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
    {
        int fidx = indicesDone[i];
        if (d.isAllocated(fidx))
        {
            int column =
                std::find(d.visFieldIndices.begin(), d.visFieldIndices.end(), fidx) - d.visFieldIndices.begin();
            auto temp = fieldPointers[fidx];

            if (auto* ptr = std::get_if<std::vector<float>*>(&fieldPointers[fidx]))
            {
                auto dataPtr = (*ptr)->data();
                addField(mesh, visDictionary[namesDone[i]], dataPtr, startIndex, endIndex);
            }
            else if (auto* ptr = std::get_if<std::vector<double>*>(&fieldPointers[fidx]))
            {
                auto dataPtr = (*ptr)->data();
                addField(mesh, visDictionary[namesDone[i]], dataPtr, startIndex, endIndex);
            }
            else if (auto* ptr = std::get_if<std::vector<unsigned>*>(&fieldPointers[fidx]))
            {
                auto dataPtr = (*ptr)->data();
                addField(mesh, visDictionary[namesDone[i]], dataPtr, startIndex, endIndex);
            }
            else
            {
                auto dataPtr = (*ptr)->data();
                addField(mesh, visDictionary[namesDone[i]], dataPtr, startIndex, endIndex);
            }
            indicesDone.erase(indicesDone.begin() + i);
            namesDone.erase(namesDone.begin() + i);
        }
    }

    if (!indicesDone.empty() && rank == 0)
    {
        std::cout << "WARNING: the following fields are not in use and therefore not visualized: ";
        for (int fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
        {
            std::cout << d.fieldNames[fidx] << ",";
        }
        std::cout << d.fieldNames[indicesDone.back()] << std::endl;
    }
    /* ===================================================== */
    // Set up another sub-mesh for rank-specific data export
    // Since usually the benchmark value is unique for each rank
    // there's only one coordinate possible
    // size_t numRanks                    = 1;
    // mesh["coordsets/rank_coords/type"] = "explicit";
    // mesh["coordsets/rank_coords/values/x"].set_external(&d.x[0], 1);
    // mesh["coordsets/rank_coords/values/y"].set_external(&d.y[0], 1);

    // // Set up topology
    // mesh["topologies/ranks/type"]           = "unstructured";
    // mesh["topologies/ranks/coordset"]       = "rank_coords";
    // mesh["topologies/ranks/elements/shape"] = "point";
    // mesh["topologies/ranks/elements/connectivity"].set_external(&rank, 1);
    /* ===================================================== */

    // Execution time for 1 iteration in current rank. Differs upon ranks
    // addRankBenchmarkField(mesh, "rank_time", 0.01);

    conduit::Node verify_info;
    if (!conduit::blueprint::mesh::verify(mesh, verify_info))
    {
        // verify failed, print error message
        CONDUIT_INFO("blueprint verify failed!" + verify_info.to_json());
    }

    // Start benchmarking from here
    a.publish(mesh);
    a.execute(actions);
}

void Finalize() { a.close(); }

} // namespace AscentAdaptor
