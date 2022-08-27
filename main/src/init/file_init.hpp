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
 * @brief Simulation data initialization from an HDF5 file
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <map>

#include "cstone/sfc/box.hpp"

#include "io/mpi_file_utils.hpp"
#include "isim_init.hpp"

namespace sphexa
{

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

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t /*n*/, Dataset& d) const override
    {
        using T        = typename Dataset::RealType;
        using Boundary = cstone::BoundaryType;

        H5PartFile* h5_file    = fileutils::openH5Part(h5_fname, H5PART_READ, d.comm);
        size_t      numH5Steps = H5PartGetNumSteps(h5_file);
        H5PartSetStep(h5_file, numH5Steps - 1);

        size_t numParticles  = H5PartGetNumParticles(h5_file);
        d.numParticlesGlobal = numParticles;
        if (numParticles < 1) { throw std::runtime_error("no particles in input file found\n"); }

        auto [first, last] = partitionRange(numParticles, rank, numRanks);
        size_t count       = last - first;

        H5PartReadStepAttrib(h5_file, "time", &d.ttot);
        H5PartReadStepAttrib(h5_file, "minDt", &d.minDt);
        H5PartReadStepAttrib(h5_file, "minDt_m1", &d.minDt_m1);
        H5PartReadStepAttrib(h5_file, "step", &d.iteration);
        d.iteration++;
        H5PartReadStepAttrib(h5_file, "gravConstant", &d.g);
        H5PartReadStepAttrib(h5_file, "gamma", &d.gamma);

        double extents[6];
        H5PartReadStepAttrib(h5_file, "box", extents);
        h5part_int32_t b[3];
        H5PartReadStepAttrib(h5_file, "boundaryType", b);
        Boundary boundary[3] = {static_cast<Boundary>(b[0]), static_cast<Boundary>(b[1]), static_cast<Boundary>(b[2])};

        cstone::Box<T> box(extents[0], extents[1], extents[2], extents[3], extents[4], extents[5], boundary[0],
                           boundary[1], boundary[2]);

        d.resize(count);

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

        initField(h5_file, rank, d.du_m1, "du_m1", 0.0);
        initField(h5_file, rank, d.alpha, "alpha", d.alphamin);

        initXm1(h5_file, rank, d);

        std::fill(d.mue.begin(), d.mue.end(), 2.0);
        std::fill(d.mui.begin(), d.mui.end(), 10.0);

        H5PartCloseFile(h5_file);
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
                if (rank == 0) std::cout << name << " not provided, initializing to " << defaultValue << std::endl;
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
                std::cout << "no previous time-step coordinates provided, initializing from current coordinates and "
                             "velocities\n";

#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < d.x.size(); ++i)
            {
                d.x_m1[i] = d.x[i] - d.vx[i] * d.minDt;
                d.y_m1[i] = d.y[i] - d.vy[i] * d.minDt;
                d.z_m1[i] = d.z[i] - d.vz[i] * d.minDt;
            }
        }
    }
};

} // namespace sphexa
