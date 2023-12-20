/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief File I/O interface implementation with ADIOS2
 *
 * @author Yiqing Zhu <yiqing.zhu@unibas.ch>
 */

#include <mpi.h>

#include <filesystem>
#include <string>
#include <variant>
#include <vector>
#include <stdexcept>

#include "ifile_io_impl.h"
#ifdef SPH_EXA_HAVE_ADIOS
#include "adios_wrapper.hpp"
#endif


namespace sphexa
{
#ifdef SPH_EXA_HAVE_ADIOS
class ADIOSReader final : public IFileReader
{
public:
    using Base      = IFileReader;
    using FieldType = typename Base::FieldType;

    explicit ADIOSReader(MPI_Comm comm)
        : comm_(comm)
        , h5File_{nullptr}
    {
        MPI_Comm_rank(comm, &rank_);
    }

    ~ADIOSReader() override { closeStep(); }

    [[nodiscard]] int     rank() const override { return rank_; }
    [[nodiscard]] int64_t numParticles() const override
    {
        return 0;
    }

    /*! @brief open a file at a given step
     *
     * @param path  filesystem path
     * @param step  snapshot index to load from
     * @param mode  collective mode causes MPI ranks to distribute particles amongst themselves,
     *              independent mode causes all MPI ranks to load all particles
     */
    void setStep(std::string path, int step, FileMode mode) override
    {
        closeStep();

        pathStep_ = path;

        if (ADIOSGetNumSteps(as_) == 0) { return; }

        // set step to last step in file if negative
        if (step < 0) { step = ADIOSGetNumSteps(as_) - 1; }

        globalCount_ = ADIOSGetNumParticles(as_);
        if (globalCount_ < 1) { return; }

        int rank, numRanks;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &numRanks);

        if (mode == FileMode::collective)
        {
            std::tie(firstIndex_, lastIndex_) = partitionRange(globalCount_, rank, numRanks);
            localCount_                       = lastIndex_ - firstIndex_;
        }
        else { std::tie(firstIndex_, lastIndex_, localCount_) = std::make_tuple(0, globalCount_, globalCount_); }
    }

    std::vector<std::string> fileAttributes() override
    {
        return std::vector<std::string>();
    }

    std::vector<std::string> stepAttributes() override
    {
        return std::vector<std::string>();
    }

    int64_t fileAttributeSize(const std::string& key) override
    {
        return 1;
    }

    int64_t stepAttributeSize(const std::string& key) override
    {
        return 1;
    }

    void fileAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit(
            [this, size, &key](auto arg)
            {
                fileutils::readADIOSFileAttribute<FieldType>(as_, key)
            },
            val);
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit(
            [this, size, &key](auto arg)
            {
                fileutils::readADIOSStepAttribute<FieldType>(as_, key);
            },
            val);
    }

    void readField(const std::string& key, FieldType field) override
    {
        std::visit([this, &key](auto arg) {
            try {
                return fileutils::readADIOSField<FieldType>(as_, key, arg);
            }
            catch (std::invalid_argument &e)
            {
                std::cout << "Invalid argument exception, STOPPING PROGRAM from rank " << rank_ << "\n";
                std::cout << e.what() << "\n";
                throw std::runtime_error("Could not read field: " + key);
            }
            catch (std::ios_base::failure &e)
            {
                std::cout << "IO System base failure exception, STOPPING PROGRAM "
                            "from rank "
                        << rank_ << "\n";
                std::cout << e.what() << "\n";
                throw std::runtime_error("Could not read field: " + key);
            }
            catch (std::exception &e)
            {
                std::cout << "Exception, STOPPING PROGRAM from rank " << rank_ << "\n";
                std::cout << e.what() << "\n";
                throw std::runtime_error("Could not read field: " + key);
            }
        }, field);

    }

    uint64_t localNumParticles() override { return localCount_; }

    uint64_t globalNumParticles() override { return globalCount_; }

    void closeStep() override
    {

    }

private:
    int      rank_{0};
    MPI_Comm comm_;

    uint64_t    firstIndex_, lastIndex_;
    uint64_t    localCount_;
    uint64_t    globalCount_;
    std::string pathStep_;

    fileutils::ADIOS2Settings as_;
};

std::unique_ptr<IFileReader> makeADIOSReader(MPI_Comm comm) { return std::make_unique<ADIOSReader>(comm); }


}

#else

std::unique_ptr<IFileWriter> makeADIOSWriter(MPI_Comm, const std::string&,
                                               const std::string&) { return {}; }
std::unique_ptr<IFileReader> makeADIOSReader(MPI_Comm comm) { return {}; }

#endif

} // namespace sphexa