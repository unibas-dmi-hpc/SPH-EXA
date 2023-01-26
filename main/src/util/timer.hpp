#pragma once

#include <chrono>
#include <iostream>
#include <functional>
#include "profiler.hpp"

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
public:
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::time_point<Clock>     TimePoint;
    typedef std::chrono::duration<float>       Time;

    Timer(std::ostream& out)
        : out(out)
    {
    }

    float duration() { return std::chrono::duration_cast<Time>(tstop - tstart).count(); }

    float getSimDuration() { return std::chrono::duration_cast<Time>(Clock::now() - tstart).count(); }

    void start() { tstart = tstop = tlast = Clock::now(); }

    void stop() { tstop = Clock::now(); }

    void step(const std::string& name)
    {
        stop();
        out << "# " << name << ": " << std::chrono::duration_cast<Time>(tstop - tlast).count() << "s" << std::endl;
        tlast = tstop;
    }

private:
    std::ostream& out;
    TimePoint     tstart, tstop, tlast;
};

class MasterProcessTimer : public Timer
{
public:
    MasterProcessTimer(std::ostream& out, int rank)
        : Timer(out)
        , rank(rank)
    {
    }

    void step(const std::string& name)
    {
        if (rank == 0) Timer::step(name);
    }

private:
    int rank;
};

class ProfilingTimer : public Timer
{
public:
    ProfilingTimer(std::ostream& out, int rank)
        : Timer(out)
        , rank(rank)
        , profiler(rank)
    {
    }

    void step(const std::string& name)
    {
        std::string name_per_rank = "RANK" + std::to_string(rank) + " " + name;
        Timer::step(name_per_rank);
    }

    void profilingStop(size_t iteration)
    {
        profiler.gatherTimings(duration(), iteration);
        stop();
    }

    void stop()
    {
        if (rank == 0) Timer::stop();
    }

    void printProfilingInfo()
    {
        profiler.printProfilingInfo();
    }

private:
    int rank;
    Profiler profiler;
};

} // namespace sphexa
