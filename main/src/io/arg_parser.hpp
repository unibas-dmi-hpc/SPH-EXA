#pragma once

#include <algorithm>
#include <string>
#include <sstream>
#include <vector>

namespace sphexa
{

//! @brief returns true if all characters of @p str together represent a valid integral number
bool strIsIntegral(const std::string& str)
{
    char* ptr;
    std::strtol(str.c_str(), &ptr, 10);
    return (*ptr) == '\0' && !str.empty();
}

class ArgParser
{
public:
    ArgParser(int argc, char** argv)
        : begin(argv)
        , end(argv + argc)
    {
    }

    //! @brief look for @p option in the supplied cmd-line arguments and convert to T if found
    template<class T = std::string>
    T get(const std::string& option, T def = T{}) const
    {
        char** itr = std::find(begin, end, option);
        if (itr != end && ++itr != end)
        {
            if constexpr (std::is_arithmetic_v<T>)
            {
                return strIsIntegral(*itr) ? T(std::stoi(*itr)) : T(std::stod(*itr));
            }
            else { return std::string(*itr); }
        }
        return def;
    }

    //! @brief parse a comma-separated list
    std::vector<std::string> getCommaList(const std::string& option) const
    {
        std::string listWithCommas = get(option);

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
