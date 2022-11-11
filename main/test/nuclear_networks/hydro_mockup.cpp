/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief More representative mockup of full SPH-EXA + nuclear-nets multi-particle simulation (net14 only).
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#include <vector>
#include <chrono>

#include "../../src/io/arg_parser.hpp"
#include "../../src/io/file_utils.hpp"

// base datatype
#include "../../src/sphexa/simulation_data.hpp"

// physical parameters
#include "nnet/parameterization/net14/net14.hpp"
#include "nnet/parameterization/net87/net87.hpp"
#include "nnet/parameterization/eos/helmholtz.hpp"
#include "nnet/parameterization/eos/ideal_gas.hpp"

// nuclear reaction wrappers
#include "sphnnet/nuclear_net.hpp"
#include "sphnnet/observables.hpp"
#include "sphnnet/initializers.hpp"

#if !defined(CUDA_CPU_TEST) && defined(USE_CUDA)
using AccType = cstone::GpuTag;
#else
using AccType = cstone::CpuTag;
#endif

/*
function stolen from SPH-EXA and retrofited for testing
*/
template<class Dataset>
void dump(Dataset& d, size_t firstIndex, size_t lastIndex,
          /*const cstone::Box<typename Dataset::RealType>& box,*/ std::string path)
{
    const char separator = ' ';
    // path += std::to_string(d.iteration) + ".txt";

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    for (int turn = 0; turn < numRanks; turn++)
    {
        if (turn == rank)
        {
            try
            {
                auto fieldPointers = cstone::getOutputArrays(d);

                bool append = rank != 0;
                sphexa::fileutils::writeAscii(firstIndex, lastIndex, path, append, fieldPointers, separator);
            }
            catch (std::runtime_error& ex)
            {
                throw std::runtime_error("ERROR: Terminating\n" /*, ex.what()*/);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

template<class Data>
double totalInternalEnergy(Data const& n)
{
    const size_t n_particles  = n.temp.size();
    double       total_energy = 0;
#pragma omp parallel for schedule(static) reduction(+ : total_energy)
    for (size_t i = 0; i < n_particles; ++i)
        total_energy += n.u[i] * n.m[i];

    MPI_Allreduce(MPI_IN_PLACE, &total_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return total_energy;
}

void printHelp(char* name, int rank);

// mockup of the step function
template<class Zvector, typename Float, typename KeyType, class AccType>
void step(int rank, size_t firstIndex, size_t lastIndex, sphexa::SimulationData<Float, KeyType, AccType>& d,
          const double dt, const nnet::ReactionList& reactions,
          const nnet::ComputeReactionRatesFunctor<Float>& construct_rates_BE, const nnet::EosFunctor<Float>& eos,
          const Float* BE, const Zvector& Z)
{
    size_t n_nuclear_particles = d.nuclearData.temp.size();

    // domain redecomposition

    sphnnet::computeNuclearPartition(firstIndex, lastIndex, d);

    // do hydro stuff

    std::swap(d.nuclearData.rho, d.nuclearData.rho_m1);
    sphnnet::syncHydroToNuclear(d, {"rho", "temp"});
    sphexa::transferToDevice(d.nuclearData, 0, n_nuclear_particles, {"rho_m1", "rho", "temp"});

    sphnnet::computeNuclearReactions(d.nuclearData, 0, n_nuclear_particles, dt, dt, reactions, construct_rates_BE, eos,
                                     /*considering expansion:*/ true);
    sphnnet::computeHelmEOS(d.nuclearData, 0, n_nuclear_particles, Z);

    sphexa::transferToHost(d.nuclearData, 0, n_nuclear_particles, {"temp", "c", "p", "cv", "u"});
    sphnnet::syncNuclearToHydro(d, {"temp"});

    // do hydro stuff

    /* !! needed for now !! */
    sphexa::transferToHost(d.nuclearData, 0, n_nuclear_particles, {"Y"});
    // print total nuclear energy
    Float total_nuclear_energy  = sphnnet::totalNuclearEnergy(d.nuclearData, BE, MPI_COMM_WORLD);
    Float total_internal_energy = totalInternalEnergy(d.nuclearData);
    if (rank == 0)
        std::cout << "etot=" << total_nuclear_energy + total_internal_energy << " (nuclear=" << total_nuclear_energy
                  << ", internal=" << total_internal_energy << ")\n";
}

int main(int argc, char* argv[])
{
    /* initial hydro data */
    const double rho_left = 1.1e9, rho_right = 0.8e9;
    const double T_left = 0.5e9, T_right = 1.5e9;

    int size = 1, rank = 0;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if COMPILE_DEVICE
    util::cuda::initCudaMpi(MPI_COMM_WORLD);
#endif

    nnet::eos::helmholtz::constants::copyTableToGPU();
    nnet::net87::electrons::constants::copyTableToGPU();

    const sphexa::ArgParser parser(argc, argv);
    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0], rank);
        MPI_Finalize();
        return 0;
    }

    const bool use_net86 = parser.exists("--use-net86");
    const bool use_net87 = parser.exists("--use-electrons") && use_net86;

    const double hydro_dt          = parser.get("--dt", 1e-1);
    const int    n_max             = parser.get("-n", 10);
    const int    n_print           = parser.get("--n-particle-print", 5);
    const size_t total_n_particles = parser.get("--n-particle", 1000);

    std::string test_case = parser.get("--test-case");
    const bool  isotherm  = parser.exists("--isotherm");
    const bool  idealGas  = parser.exists("--ideal-gas") || isotherm;

    util::array<double, 87> Y0_87, X_87;
    util::array<double, 14> Y0_14, X_14;
    if (use_net86)
    {
        for (int i = 0; i < 86; ++i)
            X_87[i] = 0;

        if (test_case == "C-O-burning")
        {
            X_87[nnet::net86::constants::net14SpeciesOrder[1]] = 0.5;
            X_87[nnet::net86::constants::net14SpeciesOrder[2]] = 0.5;
        }
        else if (test_case == "He-burning") { X_87[nnet::net86::constants::net14SpeciesOrder[0]] = 1; }
        else if (test_case == "Si-burning") { X_87[nnet::net86::constants::net14SpeciesOrder[5]] = 1; }
        else
        {
            printHelp(argv[0], rank);
            throw std::runtime_error("unknown nuclear test case!\n");
        }

        for (int i = 0; i < 86; ++i)
            Y0_87[i] = X_87[i] / nnet::net86::constants::A[i];

        Y0_87[nnet::net87::constants::electron] = 1;
    }
    else
    {
        for (int i = 0; i < 14; ++i)
            X_14[i] = 0;

        if (test_case == "C-O-burning")
        {
            X_14[1] = 0.5;
            X_14[2] = 0.5;
        }
        else if (test_case == "He-burning") { X_14[0] = 1; }
        else if (test_case == "Si-burning") { X_14[5] = 1; }
        else
        {
            printHelp(argv[0], rank);
            throw std::runtime_error("unknown nuclear test case!\n");
        }

        for (int i = 0; i < 14; ++i)
            Y0_14[i] = X_14[i] / nnet::net14::constants::A[i];
    }

    /* !!!!!!!!!!!!
    initialize the hydro state
    !!!!!!!!!!!! */

    sphexa::SimulationData<double, size_t, AccType> particle_data;
    particle_data.comm = MPI_COMM_WORLD;

    particle_data.hydro.setDependent("c", "p", "cv", "u", "m", "temp", "rho");
    particle_data.nuclearData.setDependent("nuclear_node_id", "nuclear_particle_id", "node_id", "particle_id", "dt",
                                           "c", "p", "cv", "u", "dpdT", "m", "temp", "rho", "rho_m1");
    particle_data.nuclearData.devData.setDependent("temp", "rho", "rho_m1", "dt", "c", "p", "cv", "u", "dpdT");

    const size_t n_particles = total_n_particles * (rank + 1) / size - total_n_particles * rank / size;
    const size_t offset      = 10 * rank;
    const size_t first = offset, last = n_particles + offset;

    particle_data.hydro.resize(last);
    for (size_t i = first; i < last; ++i)
    {
        particle_data.hydro.temp[i] = T_left + (T_right - T_left) *
                                                   ((float)(total_n_particles * rank / size + i - first)) /
                                                   ((float)(total_n_particles - 1));
        particle_data.hydro.rho[i] = rho_left + (rho_right - rho_left) *
                                                    ((float)(total_n_particles * rank / size + i - first)) /
                                                    ((float)(total_n_particles - 1));
    }

    //! @brief nuclear network reaction list
    nnet::ReactionList const* reactions;
    //! @brief nuclear network parameterization
    nnet::ComputeReactionRatesFunctor<double> const* construct_rates_BE;
    //! @brief eos
    nnet::EosFunctor<double>* eos;
    //! @brief BE
    double const* BE;
    //!@brief Z
    std::vector<double> Z;

    if (use_net87)
    {
        reactions          = &nnet::net87::reactionList;
        construct_rates_BE = &nnet::net87::computeReactionRates;
        BE                 = nnet::net87::BE.data();

        Z.resize(87);
        std::copy(nnet::net87::constants::Z.begin(), nnet::net87::constants::Z.begin() + 87, Z.begin());
    }
    else if (use_net86)
    {
        reactions          = &nnet::net86::reactionList;
        construct_rates_BE = &nnet::net86::computeReactionRates;
        BE                 = nnet::net86::BE.data();

        Z.resize(86);
        std::copy(nnet::net86::constants::Z.begin(), nnet::net86::constants::Z.begin() + 86, Z.begin());
    }
    else
    {
        reactions          = &nnet::net14::reactionList;
        construct_rates_BE = &nnet::net14::computeReactionRates;
        BE                 = nnet::net14::BE.data();

        Z.resize(14);
        std::copy(nnet::net14::constants::Z.begin(), nnet::net14::constants::Z.begin() + 14, Z.begin());
    }

    if (idealGas) { eos = new nnet::eos::IdealGasFunctor<double>(isotherm ? 1e-20 : 10.0); }
    else { eos = new nnet::eos::HelmholtzFunctor<double>(Z); }

    /* !!!!!!!!!!!!
    initialize nuclear data
    !!!!!!!!!!!! */

    sphnnet::initializeNuclearPointers(first, last, particle_data);

    if (use_net87)
    {
        particle_data.nuclearData.numSpecies = 87;

        for (int i = 0; i < 87; ++i)
        {
            particle_data.nuclearData.setDependent("Y" + std::to_string(i));
            particle_data.nuclearData.devData.setDependent("Y" + std::to_string(i));
        }

        sphnnet::initNuclearDataFromConst(first, last, particle_data, Y0_87);
    }
    else if (use_net86)
    {
        particle_data.nuclearData.numSpecies = 86;

        for (int i = 0; i < 86; ++i)
        {
            particle_data.nuclearData.setDependent("Y" + std::to_string(i));
            particle_data.nuclearData.devData.setDependent("Y" + std::to_string(i));
        }

        sphnnet::initNuclearDataFromConst(first, last, particle_data, Y0_87);
    }
    else
    {
        particle_data.nuclearData.numSpecies = 14;

        for (int i = 0; i < 14; ++i)
        {
            particle_data.nuclearData.setDependent("Y" + std::to_string(i));
            particle_data.nuclearData.devData.setDependent("Y" + std::to_string(i));
        }

        sphnnet::initNuclearDataFromConst(first, last, particle_data, Y0_14);
    }

    // initialize dt
    std::fill(particle_data.nuclearData.dt.begin(), particle_data.nuclearData.dt.end(), nnet::constants::initialDt);

    size_t n_nuclear_particles = particle_data.nuclearData.temp.size();
    for (int i = 0; i < particle_data.nuclearData.numSpecies; ++i)
        sphexa::transferToDevice(particle_data.nuclearData, 0, n_nuclear_particles, {"Y" + std::to_string(i)});
    sphexa::transferToDevice(particle_data.nuclearData, 0, n_nuclear_particles, {"dt"});

    std::fill(particle_data.nuclearData.m.begin(), particle_data.nuclearData.m.end(), 1.);

    std::vector<std::string> nuclearOutFields, hydroOutFields = {"temp", "rho"};
    particle_data.hydro.setOutputFields(hydroOutFields);
    if (use_net87 || use_net86)
    {
        nuclearOutFields = {
            "nuclear_particle_id", "nuclear_node_id", "temp", "rho", "cv", "u", "dpdT", "Y2", "Y4", "Y3"};

        particle_data.nuclearData.setOutputFields(nuclearOutFields);
    }
    else
    {
        nuclearOutFields = {
            "nuclear_particle_id", "nuclear_node_id", "temp", "rho", "cv", "u", "dpdT", "Y0", "Y2", "Y1"};

        particle_data.nuclearData.setOutputFields(nuclearOutFields);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto   start    = std::chrono::high_resolution_clock::now();
    double min_time = 3600, max_time = 0;

    /* !!!!!!!!!!!!
    do simulation
    !!!!!!!!!!!! */
    double t = 0;
    for (int i = 0; i < n_max; ++i)
    {
        if (rank == 0) std::cout << i << "th iteration...\n";
        MPI_Barrier(MPI_COMM_WORLD);

        auto start_it = std::chrono::high_resolution_clock::now();

        step(rank, first, last, particle_data, hydro_dt, *reactions, *construct_rates_BE, *eos, BE, Z);

        t += hydro_dt;

        MPI_Barrier(MPI_COMM_WORLD);

        auto end_it = std::chrono::high_resolution_clock::now();
        auto duration_it =
            ((float)std::chrono::duration_cast<std::chrono::milliseconds>(end_it - start_it).count()) / 1e3;
        min_time = std::min(min_time, duration_it);
        max_time = std::max(max_time, duration_it);

        if (rank == 0) std::cout << "\t...Ok\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        auto stop         = std::chrono::high_resolution_clock::now();
        auto duration     = ((float)std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()) / 1e3;
        auto avg_duration = duration / n_max;
        std::cout << "\nexec time: " << duration << "s (avg=" << avg_duration << "s/it, max=" << max_time
                  << "s/it, min=" << min_time << "s/it)\n\n";

        for (auto name : hydroOutFields)
            std::cout << name << " ";
        std::cout << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    dump(particle_data.hydro, first, first + n_print, "/dev/stdout");
    dump(particle_data.hydro, last - n_print, last, "/dev/stdout");

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "\n";
        for (auto name : nuclearOutFields)
            std::cout << name << " ";
        std::cout << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    sphexa::transferToHost(particle_data.nuclearData, 0, n_nuclear_particles, {"cv"});

    std::fill(particle_data.nuclearData.nuclear_node_id.begin(), particle_data.nuclearData.nuclear_node_id.end(), rank);
    std::iota(particle_data.nuclearData.nuclear_particle_id.begin(),
              particle_data.nuclearData.nuclear_particle_id.end(), 0);

    dump(particle_data.nuclearData, 0, n_print, "/dev/stdout");
    dump(particle_data.nuclearData, n_nuclear_particles - n_print, n_nuclear_particles, "/dev/stdout");

    MPI_Finalize();

    return 0;
}

void printHelp(char* name, int rank)
{
    if (rank == 0)
    {
        std::cout << "\nUsage:\n\n";
        std::cout << name << " [OPTION]\n";

        std::cout << "\nWhere possible options are:\n\n";

        std::cout << "\t'-n': number of iterations (default = 50)\n\n";
        std::cout << "\t'--dt': timestep (default = 1e-2s)\n\n";

        std::cout << "\t'--n-particle': total number of particles shared across nodes (default = 1000)\n\n";
        std::cout << "\t'--n-particle-print': number of particle to serialize at the end and begining of each node "
                     "(default = 5)\n\n";

        std::cout << "\t'--test-case': represent nuclear initial state, can be:\n\n";
        std::cout << "\t\t'C-O-burning: x(12C) = x(16O) = 0.5\n\n";
        std::cout << "\t\t'He-burning: x(4He) = 1\n\n";
        std::cout << "\t\t'Si-burning: x(28Si) = 1\n\n";

        std::cout << "\t'--isotherm': if exists cv=1e20, else use Helmholtz EOS\n\n";
        std::cout << "\t'--skip-coulomb-corr': if exists skip coulombian corrections\n\n";

        std::cout << "\t'--output-net14': if exists output results only for net14 species\n\n";
        std::cout << "\t'--debug-net86': if exists output debuging prints for net86 species\n\n";
    }
}