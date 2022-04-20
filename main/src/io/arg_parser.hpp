#pragma once

#include <algorithm>
#include <string>
#include <sstream>
#include <vector>

namespace sphexa
{

class ArgParser
{
public:
    ArgParser(int argc, char** argv)
        : begin(argv)
        , end(argv + argc)
    {
    }

    std::string getString(const std::string& option, const std::string def = "") const
    {
        char** itr = std::find(begin, end, option);
        if (itr != end && ++itr != end) return std::string(*itr);
        return def;
    }

    int getInt(const std::string& option, const int def = 0) const
    {
        char** itr = std::find(begin, end, option);
        if (itr != end && ++itr != end) return (int)std::stof(*itr);
        return def;
    }

    double getDouble(const std::string& option, const double def = 0.) const
    {
        char** itr = find(begin, end, option);
        if (itr != end && ++itr != end) return std::stod(*itr);
        return def;
    }

    //! @brief parse a comma-separated list
    std::vector<std::string> getCommaList(const std::string& option) const
    {
        std::string listWithCommas = getString(option);

        std::replace(listWithCommas.begin(), listWithCommas.end(), ',', ' ');

        std::vector<std::string> list;
        std::stringstream        ss(listWithCommas);
        std::string              field;
        while (ss >> field)
        {
            list.push_back(field);
        }

        return list;
    }

    bool exists(const std::string& option) const { return std::find(begin, end, option) != end; }

private:
    char **begin, **end;
};

//! @brief returns true if all characters of @p str together represent a valid integral number
bool strIsIntegral(const std::string& str)
{
    char* ptr;
    std::strtol(str.c_str(), &ptr, 10);
    return (*ptr) == '\0' && !str.empty();
}

/*! @brief Evaluate if current step and simulation time should be output (to file)
 *
 * @param step          current simulation step
 * @param t1            simulation time at beginning of current step
 * @param t2            simulation time at end of current step
 * @param extraOutputs  list of strings of integral and/or floating point numbers
 * @return              true if @p step matches any integral numbers in @p extraOutput or
 *                      if any floating point number therein falls into the interval @p [t1, t2)
 */
bool isExtraOutputStep(size_t step, double t1, double t2, const std::vector<std::string>& extraOutputs)
{
    auto matchStepOrTime = [step, t1, t2](const std::string& token)
    {
        double time       = std::stod(token);
        bool   isIntegral = strIsIntegral(token);
        return (isIntegral && std::stoul(token) == step) || (!isIntegral && t1 <= time && time < t2);
    };

    return std::any_of(extraOutputs.begin(), extraOutputs.end(), matchStepOrTime);
}

bool isPeriodicOutputStep(size_t step, int writeFrequency)
{
    return writeFrequency == 0 || (writeFrequency > 0 && step % writeFrequency == 0);
}

} // namespace sphexa
