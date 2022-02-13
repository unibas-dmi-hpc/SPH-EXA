#pragma once

struct ParticleIO
{

    size_t idx; // initial index order
    double r;   // radius
    double vel; // radius

    double x;   // x position
    double y;   // y position
    double z;   // z position
    double vx;  // vx velocity
    double vy;  // vy velocity
    double vz;  // vz velocity
    double h;   // smoothing length
    double rho; // density
    double u;   // internal energy
    double p;   // pressure
    double cs;  // sound speed
    double Px;  // x momentum
    double Py;  // y momentum
    double Pz;  // z momentum

    // To sort the particles by radius
    struct cmp
    {
        bool operator()(const ParticleIO& p1, const ParticleIO& p2) { return p1.r < p2.r; }
    };
};
