#pragma once

#include <iomanip>

namespace sphexa
{

class Printer
{
public:

    static void printConstants(size_t iteration, double totalTime, double minTimeStep, double totalEnergy,
                               double kineticEnergy, double internalEnergy, double gravitationalEnergy,
                               size_t totalNeighbors, std::ostream& out)
    {
        out << iteration << ' ' << totalTime << ' ' << minTimeStep << ' ' << totalEnergy << ' ' << kineticEnergy << ' '
            << internalEnergy << ' ';
        out << gravitationalEnergy << ' ' << totalNeighbors << ' ' << std::endl;
    }

    template<class Box>
    static void printCheck(double totalTime, double minTimeStep, double totalEnergy, double internalEnergy,
                           double kineticEnergy, double gravitationalEnergy, const Box& box, size_t totalParticleCount,
                           size_t particleCount, size_t nodeCount, size_t haloCount, size_t totalNeighbors,
                           std::ostream& out)
    {
        out << "### Check ### Global Tree Nodes: " << nodeCount << ", Particles: " << particleCount
            << ", Halos: " << haloCount << std::endl;
        out << "### Check ### Computational domain: " << box.xmin() << " " << box.xmax() << " " << box.ymin() << " "
            << box.ymax() << " " << box.zmin() << " " << box.zmax() << std::endl;
        out << "### Check ### Total Neighbors: " << totalNeighbors
            << ", Avg neighbor count per particle: " << totalNeighbors / totalParticleCount << std::endl;
        out << "### Check ### Total time: " << totalTime << ", current time-step: " << minTimeStep << std::endl;
        out << "### Check ### Total energy: " << totalEnergy << ", (internal: " << internalEnergy
            << ", cinetic: " << kineticEnergy;
        out << ", gravitational: " << gravitationalEnergy;
        out << ")" << std::endl;
    }

    static void printTotalIterationTime(size_t iteration, float duration, std::ostream& out)
    {
        out << "=== Total time for iteration(" << iteration << ") " << duration << "s" << std::endl << std::endl;
    }
};

} // namespace sphexa
