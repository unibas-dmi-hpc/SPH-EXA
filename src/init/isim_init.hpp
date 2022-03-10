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
 * @brief Test-case simulation data initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <map>

#include "cstone/sfc/box.hpp"

#include "io/mpi_file_utils.hpp"
#include "sedov_constants.hpp"
#include "grid.hpp"

namespace sphexa
{

template<class Dataset>
class ISimInitializer
{
public:
    virtual cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, Dataset& d) const = 0;
    virtual const std::map<std::string, double>&    constants() const                              = 0;

    virtual ~ISimInitializer() = default;
};

template<class Dataset>
class FileInit : public ISimInitializer<Dataset>
{
    std::map<std::string, double> constants_;
    std::string                   h5_fname;

public:
    FileInit(std::string fname)
        : h5_fname(fname)
    {
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, Dataset& d) const override
    {
#ifdef SPH_EXA_HAVE_H5PART
        using T = typename Dataset::RealType;

        H5PartFile* h5_file = nullptr;
#ifdef H5PART_PARALLEL_IO
        h5_file = H5PartOpenFileParallel(h5_fname, H5PART_READ, d.comm);
#else
        h5_file = H5PartOpenFile(h5_fname.c_str(), H5PART_READ);
#endif
        size_t numH5Steps = H5PartGetNumSteps(h5_file);
        H5PartSetStep(h5_file, numH5Steps - 1);

        size_t numParticles = H5PartGetNumParticles(h5_file);
        d.n                 = numParticles;
        if (numParticles < 1) { throw std::runtime_error("no particles in input file found\n"); }

        auto [first, last] = partitionRange(numParticles, rank, numRanks);
        d.count            = last - first;

        H5PartReadStepAttrib(h5_file, "time", &d.ttot);
        H5PartReadStepAttrib(h5_file, "minDt", &d.minDt);
        H5PartReadStepAttrib(h5_file, "step", &d.iteration);
        d.iteration++;
        H5PartReadStepAttrib(h5_file, "gravConstant", &d.g);

        double extents[6];
        H5PartReadStepAttrib(h5_file, "box", extents);
        int pbc[3];
        H5PartReadStepAttrib(h5_file, "pbc", pbc);
        cstone::Box<T> box(
            extents[0], extents[1], extents[2], extents[3], extents[4], extents[5], pbc[0], pbc[1], pbc[2]);

        resize(d, d.count);

        H5PartSetView(h5_file, first, last - 1);
        h5part_int64_t errors = H5PART_SUCCESS;
        errors |= fileutils::readH5PartField(h5_file, "x", d.x.data());
        errors |= fileutils::readH5PartField(h5_file, "y", d.y.data());
        errors |= fileutils::readH5PartField(h5_file, "z", d.z.data());
        errors |= fileutils::readH5PartField(h5_file, "h", d.h.data());
        errors |= fileutils::readH5PartField(h5_file, "m", d.m.data());
        errors |= fileutils::readH5PartField(h5_file, "u", d.u.data());

        if (errors != H5PART_SUCCESS) { throw std::runtime_error("Could not read essential fields x,y,z,h,m,u\n"); }

        initField(h5_file, rank, d.vx, "vx", 0.0);
        initField(h5_file, rank, d.vy, "vy", 0.0);
        initField(h5_file, rank, d.vz, "vz", 0.0);

        initField(h5_file, rank, d.dt_m1, "dt_m1", d.minDt);
        initField(h5_file, rank, d.du_m1, "du_m1", 0.0);
        initField(h5_file, rank, d.alpha, "alpha", d.alphamin);

        initXm1(h5_file, rank, d);

        std::fill(d.mue.begin(), d.mue.end(), 2.0);
        std::fill(d.mui.begin(), d.mui.end(), 10.0);

#else
        throw std::runtime_error("Cannot read from HDF5 file: H5Part not enabled\n");
#endif
        return box;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }

private:
    template<class Vector>
    static void initField(H5PartFile* h5_file, int rank, Vector& field, std::string name, double defaultValue)
    {
        if (field.size())
        {
            auto datasets = fileutils::datasetNames(h5_file);
            bool hasField = std::count(datasets.begin(), datasets.end(), name) == 1;
            if (hasField)
            {
                if (rank == 0) std::cout << "loading " + name + " from file\n";
                fileutils::readH5PartField(h5_file, name.c_str(), field.data());
            }
            else
            {
                if (rank == 0) std::cout << name << " not found, initializing to " << defaultValue << std::endl;
                std::fill(field.begin(), field.end(), defaultValue);
            }
        }
    }

    static void initXm1(H5PartFile* h5_file, int rank, Dataset& d)
    {
        auto   names  = fileutils::datasetNames(h5_file);
        size_t hasXm1 = std::count(names.begin(), names.end(), "x_m1") +
                        std::count(names.begin(), names.end(), "y_m1") + std::count(names.begin(), names.end(), "z_m1");
        if (hasXm1 == 3)
        {
            if (rank == 0) std::cout << "loading previous time-step coordinates from file\n";
            fileutils::readH5PartField(h5_file, "x_m1", d.x_m1.data());
            fileutils::readH5PartField(h5_file, "y_m1", d.y_m1.data());
            fileutils::readH5PartField(h5_file, "z_m1", d.z_m1.data());
        }
        else
        {
            if (rank == 0)
                std::cout << "no previous time-step coordinates found, initializing from current coordinates and "
                             "velocities\n";

#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < d.count; ++i)
            {
                d.x_m1[i] = d.x[i] - d.vx[i] * d.minDt;
                d.y_m1[i] = d.y[i] - d.vy[i] * d.minDt;
                d.z_m1[i] = d.z[i] - d.vz[i] * d.minDt;
            }
        }
    }
};

template<class Dataset>
class SedovGrid : public ISimInitializer<Dataset>
{
    std::map<std::string, double> constants_;

public:
    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, Dataset& d) const override
    {
        d.n = d.side * d.side * d.side;

        auto [first, last] = partitionRange(d.n, rank, numRanks);
        d.count            = last - first;

        resize(d, d.count);

        if (rank == 0)
        {
            std::cout << "Approx: " << d.count * (d.data().size() * 64.) / (8. * 1000. * 1000. * 1000.)
                      << "GB allocated on rank 0." << std::endl;
        }

        regularGrid(SedovConstants::r1, d.side, first, last, d.x, d.y, d.z);
        initFields(d);

        using T    = typename Dataset::RealType;
        T r        = SedovConstants::r1;
        T halfStep = SedovConstants::r1 / d.side;
        return cstone::Box<T>(-r - halfStep, r - halfStep, true);
    }

    const std::map<std::string, double>& constants() const override { return constants_; }

private:
    void initFields(Dataset& d) const
    {
        using T = typename Dataset::RealType;

        double step   = (2. * SedovConstants::r1) / d.side;
        double hIni   = 1.5 * step;
        double mPart  = SedovConstants::mTotal / d.n;
        double width  = SedovConstants::width;
        double width2 = width * width;

        double firstTimeStep = SedovConstants::firstTimeStep;

        std::fill(d.m.begin(), d.m.end(), mPart);
        std::fill(d.h.begin(), d.h.end(), hIni);
        std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
        std::fill(d.mui.begin(), d.mui.end(), 10.0);
        std::fill(d.dt.begin(), d.dt.end(), firstTimeStep);
        std::fill(d.dt_m1.begin(), d.dt_m1.end(), firstTimeStep);
        std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
        d.minDt = firstTimeStep;

        std::fill(d.vx.begin(), d.vx.end(), 0.0);
        std::fill(d.vy.begin(), d.vy.end(), 0.0);
        std::fill(d.vz.begin(), d.vz.end(), 0.0);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < d.count; i++)
        {
            T xi = d.x[i];
            T yi = d.y[i];
            T zi = d.z[i];
            T r2 = xi * xi + yi * yi + zi * zi;

            d.u[i] = SedovConstants::ener0 * exp(-(r2 / width2)) + SedovConstants::u0;

            d.x_m1[i] = xi - d.vx[i] * firstTimeStep;
            d.y_m1[i] = yi - d.vy[i] * firstTimeStep;
            d.z_m1[i] = zi - d.vz[i] * firstTimeStep;
        }
    }
};

template<class Dataset>
class NohGrid : public ISimInitializer<Dataset>
{
    std::map<std::string, double> constants_{{"r0", 0},
                                             {"r1", 0.5},
                                             {"mTotal", 1.},
                                             {"dim", 3},
                                             {"gamma", 5.0 / 3.0},
                                             {"rho0", 1.},
                                             {"u0", 1e-20},
                                             {"p0", 0.},
                                             {"vr0", -1.},
                                             {"cs0", 0.},
                                             {"firstTimeStep", 1e-4}};

public:
    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, Dataset& d) const override
    {
        using T = typename Dataset::RealType;
        d.n     = d.side * d.side * d.side;

        auto [first, last] = partitionRange(d.n, rank, numRanks);
        d.count            = last - first;

        resize(d, d.count);

        if (rank == 0)
        {
            std::cout << "Approx: " << d.count * (d.data().size() * 64.) / (8. * 1000. * 1000. * 1000.)
                      << "GB allocated on rank 0." << std::endl;
        }

        T r = constants_.at("r1");

        regularGrid(r, d.side, first, last, d.x, d.y, d.z);
        initFields(d);

        return cstone::Box<T>(-r, r, false);
    }

    const std::map<std::string, double>& constants() const override { return constants_; }

private:
    void initFields(Dataset& d) const
    {
        using T = typename Dataset::RealType;

        double r1            = constants_.at("r1");
        double step          = (2. * r1) / d.side;
        double hIni          = 1.5 * step;
        double mPart         = constants_.at("mTotal") / d.n;
        double firstTimeStep = constants_.at("firstTimeStep");

        std::fill(d.m.begin(), d.m.end(), mPart);
        std::fill(d.h.begin(), d.h.end(), hIni);
        std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
        std::fill(d.mui.begin(), d.mui.end(), 10.0);
        std::fill(d.dt.begin(), d.dt.end(), firstTimeStep);
        std::fill(d.dt_m1.begin(), d.dt_m1.end(), firstTimeStep);
        std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
        d.minDt = firstTimeStep;

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < d.count; i++)
        {
            T radius = std::sqrt((d.x[i] * d.x[i]) + (d.y[i] * d.y[i]) + (d.z[i] * d.z[i]));
            radius   = std::max(radius, 1e-10);

            d.u[i] = constants_.at("u0");

            d.vx[i] = constants_.at("vr0") * (d.x[i] / radius);
            d.vy[i] = constants_.at("vr0") * (d.y[i] / radius);
            d.vz[i] = constants_.at("vr0") * (d.z[i] / radius);

            d.x_m1[i] = d.x[i] - d.vx[i] * firstTimeStep;
            d.y_m1[i] = d.y[i] - d.vy[i] * firstTimeStep;
            d.z_m1[i] = d.z[i] - d.vz[i] * firstTimeStep;
        }
    }
};

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>> initializerFactory(std::string testCase)
{
    if (testCase == "sedov") { return std::make_unique<SedovGrid<Dataset>>(); }
    if (testCase == "noh") { return std::make_unique<NohGrid<Dataset>>(); }
    else
    {
        return std::make_unique<FileInit<Dataset>>(testCase);
    }
}

} // namespace sphexa
