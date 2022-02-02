// propagator.hpp

class Propagator
{
public:
    // advance simulation by one step
    hydroStep(Domain<...>& domain, ParticlesData<...>& d)
    {
        domain.sync(d.x, d.y, d.z, d.h, d.m, d.conservedQuantities);
        d.resize(domain.nParticlesWithHalos());
        ...
        computePositions(d, domain.box());
    }

private:
     MasterProcessTimer timer;
     const size_t nTasks, ngmax, ng0;
     TaskList taskList;
};
