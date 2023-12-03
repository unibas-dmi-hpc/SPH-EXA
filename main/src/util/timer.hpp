#pragma once

#include <chrono>
#include <iostream>

#if defined(USE_PROFILING_NVTX) || defined(USE_PROFILING_SCOREP)

#ifdef USE_PROFILING_NVTX
#include <nvToolsExt.h>
#define MARK_BEGIN(xx) nvtxRangePush(xx);
#define MARK_END nvtxRangePop();
#endif

#ifdef USE_PROFILING_SCOREP
#include "scorep/SCOREP_User.h"
#define MARK_BEGIN(xx)                                                                                                 \
    {                                                                                                                  \
        SCOREP_USER_REGION(xx, SCOREP_USER_REGION_TYPE_COMMON)
#define MARK_END }
#endif

#else
#define MARK_BEGIN(xx)
#define MARK_END
#endif

namespace sphexa
{

class Timer
{
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<float>       Time;

public:
    Timer(std::ostream& out)
        : out(out)
    {
    }

    void start()
    {
        numStartCalled++;
        tstart = tlast = Clock::now();
    }

    void step(const std::string& name)
    {
        auto now = Clock::now();
        stepTimes.push_back(stepDuration(now));
        if (!name.empty()) { out << "# " << name << ": " << stepTimes.back() << "s" << std::endl; }
        tlast = now;
    }

    //! @brief time elapsed between tstart and last call of step()
    [[nodiscard]] float sumOfSteps() const { return std::chrono::duration_cast<Time>(tlast - tstart).count(); }

    //! @brief time elapsed between tstart and now
    [[nodiscard]] float elapsed() const { return std::chrono::duration_cast<Time>(Clock::now() - tstart).count(); }

    template<class Archive>
    void writeTimings(Archive* ar, const std::string& outFile)
    {
        ar->addStep(0, stepTimes.size(), outFile + ar->suffix());
        int numRanks = ar->numRanks();
        ar->stepAttribute("numRanks", &numRanks, 1);
        ar->stepAttribute("numIterations", &numStartCalled, 1);
        ar->writeField("timings", stepTimes.data(), stepTimes.size());
        ar->closeStep();

        numStartCalled = 0;
        stepTimes.clear();
    }

private:
    float stepDuration(auto now) { return std::chrono::duration_cast<Time>(now - tlast).count(); }

    std::ostream&                  out;
    std::chrono::time_point<Clock> tstart, tlast;
    std::vector<float>             stepTimes;
    int                            numStartCalled{0};
};

} // namespace sphexa
