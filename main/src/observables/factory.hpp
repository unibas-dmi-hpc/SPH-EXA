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
 * @brief Select/calculate data to be printed to constants.txt each iteration
 *
 * @author Lukas Schmidt
 */

#pragma once


#include "cstone/sfc/box.hpp"
#include "iobservables.hpp"
#include "time_energy_growth.hpp"
#include "time_energies.hpp"


namespace sphexa
{

template<class Dataset>
std::unique_ptr<IObservables<Dataset>> observablesFactory(std::string testCase, std::ofstream& constantsFile)
{
    //observables to check
    const char* KelvinHelmholtz = "KelvinHelmholtzGrowthRate";
    int khgrAttr[1] = {0};

    if(std::filesystem::exists(testCase))
    {
        H5PartFile* h5_file = nullptr;
        h5_file = H5PartOpenFile(testCase.c_str(), H5PART_READ);
        size_t attrN = H5PartGetNumFileAttribs(h5_file);
        char* attrName = new char; 
        long int length = 40; //arbitrary high value to read enough
        long int* attr_type;
        long int* attr_nelem;

        

        for(size_t i = 0; i < attrN; i++)
        {
            H5PartGetFileAttribInfo(h5_file, i, attrName, length, attr_type, attr_nelem);
            if(strcmp(attrName, KelvinHelmholtz) == 0)
            {
                H5PartReadFileAttrib(h5_file, attrName, khgrAttr);
            }
            
        
        }
        H5PartCloseFile(h5_file);

    }

    if (khgrAttr[0])
    {
        return std::make_unique<TimeEnergyGrowth<Dataset>>(constantsFile);
    }
    else
    {
       return std::make_unique<TimeAndEnergy<Dataset>>(constantsFile);
    }
}

} // namespace sphexa
