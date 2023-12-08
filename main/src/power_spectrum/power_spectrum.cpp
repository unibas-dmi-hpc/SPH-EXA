#include "mesh.hpp"
#include "util/utils.hpp"
#include "io/arg_parser.hpp"

using namespace sphexa;

void printSpectrumHelp(char* binName, int rank);
using MeshType = double;

int main(int argc, char** argv)
{
    auto [rank, numRanks] = initMpi();
    const ArgParser parser(argc, (const char**)argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printSpectrumHelp(argv[0], rank);
        return exitSuccess();
    }

    const std::string initFile  = parser.get("--init");
    size_t            numShells = parser.get("--numShells", 50);

    // read HDF5 checkpoint into cornerstone tree

    // get the dimensions from the checkpoint
    int gridDim = 100; // 100 is placeholder. Need to get this from the checkpoint
    numShells   = gridDim / 2;

    // init mesh
    Mesh<MeshType> mesh(rank, numRanks, gridDim, numShells);

    // convert cornerstone tree to mesh

    // calculate power spectrum
    mesh.calculate_power_spectrum();

    // write power spectrum to HDF5?

    return exitSuccess();
}

void printSpectrumHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n\n");

        printf("\t--init \t\t HDF5 file with simulation data\n\n");
        printf("\t--numShells \t\t Number of shells for averaging. Default is half of mesh dimension.\n\n");
    }
}