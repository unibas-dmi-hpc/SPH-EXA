#ifndef AscentAdaptor_h
#define AscentAdaptor_h

#include "ParticlesData.hpp"

#include <ascent/ascent.hpp>
#include "conduit_blueprint.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

using Real = double;
using CodeType = uint64_t;
using Dataset = ParticlesData<Real, CodeType>;

namespace AscentAdaptor
{
  ascent::Ascent a;
  conduit::Node actions;
        
void Initialize(Dataset d, long startIndex)
{
  conduit::Node ascent_options;
  //ascent_options["default_dir"] = "/scratch/snx3000/jfavre/DummySPH/datasets";
  ascent_options["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  a.open(ascent_options);

// Create an action that tells Ascent to:
//  add a scene (s1) with one plot (p1)
//  that will render a pseudocolor of 
//  the mesh field `rho`

  conduit::Node &add_pipe = actions.append(); 
  add_pipe["action"] = "add_pipelines";
  conduit::Node &pipes = add_pipe["pipelines"];
  pipes["pl1/f1/type"] = "threshold";
  conduit::Node &thresh_params = pipes["pl1/f1/params"];
  thresh_params["field"] = "Density";
  thresh_params["min_value"] = 1.4;
  thresh_params["max_value"] = 2000;

  conduit::Node &add_scene = actions.append(); 
  add_scene["action"] = "add_scenes";

// declare a scene (s1) and pseudocolor plot (p1)
  conduit::Node &scenes = add_scene["scenes"];
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/plots/p1/field"] = "Density";
  //scenes["s1/plots/p1/points/radius"] = .5;
  //scenes["s1/plots/p1/points/radius_delta"] = .01;
  scenes["s1/renders/r1/image_prefix"] = "DensityThreshold1.4.%05d";
  
  double vec3[3];
  vec3[0] = vec3[1] = vec3[2] = 0;
  scenes["s1/renders/r1/camera/look_at"].set_float64_ptr(vec3,3);

  vec3[0] = -2.1709899968205337; vec3[1] = 1.797907520678797; vec3[2] = 1.8029059671481107;
  scenes["s1/renders/r1/camera/position"].set_float64_ptr(vec3,3);

  vec3[0] = 0.4479257557058854; vec3[1] = 0.8420981185224633; vec3[2] = -0.30038854198560727;
  scenes["s1/renders/r1/camera/up"].set_float64_ptr(vec3,3);
  scenes["s1/renders/r1/camera/zoom"] = 2;

  /*
  scenes["s1/renders/r1/type"] = "cinema";
  scenes["s1/renders/r1/phi"] = 8;
  scenes["s1/renders/r1/theta"] = 8;
  scenes["s1/renders/r1/db_name"] = "example_db";
  */

  /* IO to disk */
  /*
  conduit::Node &add_extr = actions.append(); 
  add_extr["action"] = "add_extracts";
  conduit::Node &savedata = add_extr["extracts"];

  // add a relay extract that will write mesh data to
  // blueprint hdf5 files
  savedata["e1/type"] = "relay";
  //savedata["e1/pipeline"] = "pl1";
  savedata["e1/params/path"] = "out_export_particles";
  savedata["e1/params/protocol"] = "blueprint/mesh/hdf5";
  */
}

void Execute(Dataset d, long startIndex, long endIndex)
{
  conduit::Node mesh;
  mesh["state/cycle"].set_external(&d.iteration);
  mesh["state/time"].set_external(&d.ttot);

  mesh["coordsets/coords/type"] = "explicit";
  mesh["coordsets/coords/values/x"].set_external(&d.x[startIndex], endIndex - startIndex);
  mesh["coordsets/coords/values/y"].set_external(&d.y[startIndex], endIndex - startIndex);
  mesh["coordsets/coords/values/z"].set_external(&d.z[startIndex], endIndex - startIndex);

  mesh["topologies/mesh/type"] = "unstructured";
  mesh["topologies/mesh/coordset"] = "coords";

  mesh["fields/Density/association"] = "vertex";
  mesh["fields/Density/topology"] = "mesh";
  mesh["fields/Density/values"].set_external(&d.ro[startIndex], endIndex - startIndex);
  mesh["fields/Density/volume_dependent"].set("false");
  
  std::vector<conduit_int64> conn(endIndex - startIndex);
  std::iota(conn.begin(), conn.end(), 0);
  mesh["topologies/mesh/elements/connectivity"].set_external(conn);
  mesh["topologies/mesh/elements/shape"] = "point";

  conduit::Node verify_info;
  if(!conduit::blueprint::mesh::verify(mesh,verify_info))
  {
    // verify failed, print error message
    CONDUIT_INFO("blueprint verify failed!" + verify_info.to_json());
  }
  //else CONDUIT_INFO("blueprint verify success!" + verify_info.to_json());

  a.publish(mesh);
  a.execute(actions);
}

void Finalize()
{
  a.close();
}

}
#endif
