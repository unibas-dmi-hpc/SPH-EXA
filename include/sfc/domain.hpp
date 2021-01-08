
#include <vector>

#include "sfc/halodiscovery.hpp"

namespace sphexa
{


template<class I, class T>
class Domain
{
public:
    explicit Domain(int rank) : myRank_(rank), particleStart_(0), particleEnd_(0) {}

    void sync(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, std::vector<T>& h)
    {

    }

private:

    int myRank_;
    int particleStart_;
    int particleEnd_;

    Box<T> box_;

    std::vector<I> tree_;
};

} // namespace sphexa
