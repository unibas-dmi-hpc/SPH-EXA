#pragma once

#include <exception>

namespace sphexa
{

struct FileNotOpenedException : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

struct MPIFileNotOpenedException : public std::runtime_error
{
    MPIFileNotOpenedException(const std::string& what, const int mpierr)
        : std::runtime_error(what)
        , mpierr(mpierr)
    {
    }

    const int mpierr;
};
} // namespace sphexa
