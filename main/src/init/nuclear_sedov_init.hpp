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
 * @brief Nuclear sedov data initializing
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#include "sedov_init.hpp"

#include "nnet/sphexa/initializers.hpp"

#include "nnet/parameterization/net87/net87.hpp"
#include "nnet/parameterization/net14/net14.hpp"

namespace sphexa
{

template<class Dataset>
class NuclearSedovGrid : public ISimInitializer<Dataset>
{
    std::map<std::string, double> constants_;

public:
    NuclearSedovGrid() { constants_ = sedovConstants(); }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cubeSide,
                                                 Dataset& simData) const override
    {
        auto& d              = simData.hydro;
        auto& n              = simData.nuclearData;
        using KeyType        = typename Dataset::KeyType;
        using T              = typename Dataset::RealType;
        d.numParticlesGlobal = cubeSide * cubeSide * cubeSide;

        /* !!!!!!!!!!!!!!!!!!!! */
        /* hydro initialization */
        /* !!!!!!!!!!!!!!!!!!!! */

        auto [first, last] = partitionRange(d.numParticlesGlobal, rank, numRanks);
        d.resize(last - first);

        T              r = constants_.at("r1");
        cstone::Box<T> globalBox(-r, r, cstone::BoundaryType::periodic);
        regularGrid(r, cubeSide, first, last, d.x, d.y, d.z);
        syncCoords<KeyType>(rank, numRanks, d.numParticlesGlobal, d.x, d.y, d.z, globalBox);
        d.resize(d.x.size());
        initSedovFields(d, constants_);

        /* !!!!!!!!!!!!!!!!!!!!!! */
        /* nuclear initialization */
        /* !!!!!!!!!!!!!!!!!!!!!! */

        util::array<double, 87> Y0_87, X_87;
        for (int i = 0; i < 86; ++i)
        {
            X_87[i] = 0;
        }

        if (n.numSpecies == 14)
        {
            X_87[1] = 0.5;
            X_87[2] = 0.5;

            for (int i = 0; i < 14; ++i) 
            {
                Y0_87[i] = X_87[i]/nnet::net14::constants::A[i];
            }
        }
        else if (n.numSpecies == 86 || n.numSpecies == 87)
        {
            X_87[nnet::net86::constants::net14_species_order[1]] = 0.5;
            X_87[nnet::net86::constants::net14_species_order[2]] = 0.5;

            for (int i = 0; i < 86; ++i)
            {
                Y0_87[i] = X_87[i]/nnet::net86::constants::A[i];
            }
        }
        else
        {
            throw std::runtime_error("not able to initialize " + std::to_string(n.numSpecies) + " nuclear species !");
        }

        // sphexa::sphnnet::initNuclearDataFromConst(first, last, simData, Y0_87);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};


template<class Dataset>
class NuclearSedovGlass : public ISimInitializer<Dataset>
{
    std::string                   glassBlock;
    std::map<std::string, double> constants_;

public:
    NuclearSedovGlass(std::string initBlock)
        : glassBlock(initBlock)
    {
        constants_ = sedovConstants();
    }

    /*! @brief initialize particle data with a constant density cube
     *
     * @param[in]    rank             MPI rank ID
     * @param[in]    numRanks         number of MPI ranks
     * @param[in]    cbrtNumPart      the cubic root of the global number of particles to generate
     * @param[inout] d                particle dataset
     * @return                        the global coordinate bounding box
     */
    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart,
                                                 Dataset& simData) const override
    {
        auto& d       = simData.hydro;
        auto& n       = simData.nuclearData;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        /* !!!!!!!!!!!!!!!!!!!! */
        /* hydro initialization */
        /* !!!!!!!!!!!!!!!!!!!! */

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        size_t multiplicity  = std::rint(cbrtNumPart / std::cbrt(blockSize));
        d.numParticlesGlobal = multiplicity * multiplicity * multiplicity * blockSize;

        T              r = constants_.at("r1");
        cstone::Box<T> globalBox(-r, r, cstone::BoundaryType::periodic);

        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);
        assembleCube<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        d.resize(d.x.size());
        initSedovFields(d, constants_);

        /* !!!!!!!!!!!!!!!!!!!!!! */
        /* nuclear initialization */
        /* !!!!!!!!!!!!!!!!!!!!!! */

        util::array<double, 87> Y0_87, X_87;
        for (int i = 0; i < 86; ++i)
        {
            X_87[i] = 0;
        }

        if (n.numSpecies == 14)
        {
            X_87[1] = 0.5;
            X_87[2] = 0.5;

            for (int i = 0; i < 14; ++i) 
            {
                Y0_87[i] = X_87[i]/nnet::net14::constants::A[i];
            }
        }
        else if (n.numSpecies == 86 || n.numSpecies == 87)
        {
            X_87[nnet::net86::constants::net14_species_order[1]] = 0.5;
            X_87[nnet::net86::constants::net14_species_order[2]] = 0.5;

            for (int i = 0; i < 86; ++i)
            {
                Y0_87[i] = X_87[i]/nnet::net86::constants::A[i];
            }
        }
        else
        {
            throw std::runtime_error("not able to initialize " + std::to_string(n.numSpecies) + " nuclear species !");
        }

        // sphexa::sphnnet::initNuclearDataFromConst(first, last, simData, Y0_87);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa