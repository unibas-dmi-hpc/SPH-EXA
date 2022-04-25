#pragma once

#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <cassert>

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
            else
            {
                return std::string(*itr);
            }
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

/*! @brief Evaluate if the current step should be output (to file) per time frequency
 *
 * @param t             simulation time at beginning of current step
 * @param dt            time step of the step
 * @param frequency     frequency time to output the simulation (>0.)
 * @return              true if the step have a time that fit with the usual time frequency output in [t, t+dt]
 */
bool isPeriodicOutputTime(double t, double dt, double frequency)
{
    assert(frequency > 0.);
    double currentTime = std::fmod(t, frequency);
    double futureTime  = currentTime + dt;
    return (currentTime < dt) || ((futureTime > frequency) && (frequency - currentTime < futureTime - frequency));
}

/*! @brief Evaluate if the current step should be output (to file) per iteration frequency
 *
 * @param step          number of simulation step
 * @param frequency     iteration frequency to output the simulation (>0)
 * @return              true if the step fit with the usual iteration frequency output
 */
bool isPeriodicOutputStep(size_t step, int frequency)
{
    assert(frequency > 0);
    return frequency != 0 && (step % frequency == 0);
}

} // namespace sphexa
